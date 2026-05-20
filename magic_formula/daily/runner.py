"""
magic_formula.daily.runner
==========================
매일 실행되는 데일리 시그널 파이프라인.

- 코어 종목에 대해 황금률 점수 계산
- 진입신호(R1 임계 돌파) 종목 추출
- JSON + Markdown 리포트 저장

설정 출처
---------
``configs/active_strategy.yaml`` 의 ``magic_formula.config.load_strategy()`` 결과.
가중치 / 진입규칙 / 임계값 / area4_mode / universe 모두 yaml 에서.

출력
----
- ``output/signals/daily_signal_YYYYMMDD.json``
- ``output/signals/daily_signal_YYYYMMDD.md``
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from magic_formula._vault import (
    SECTOR_ORDER,
    TICKER_SECTORS as SECTOR_MAP,
    get_ticker_name,
    get_universe,
)

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------

_PKG_DIR    = Path(__file__).resolve().parent.parent   # magic_formula/
_PROJ_ROOT  = _PKG_DIR.parent                          # Magic Formula/
OUTPUT_DIR  = _PROJ_ROOT / "output" / "signals"


# ---------------------------------------------------------------------------
# 점수 계산
# ---------------------------------------------------------------------------

def fetch_and_score(
    ticker: str,
    end_date: str,
    weights: dict[str, float],
    area4_mode: str,
) -> pd.DataFrame | None:
    """
    종목 OHLCV 로드 → 점수 계산.
    실패 시 None 반환 (개별 종목 오류가 전체를 중단시키지 않도록).
    """
    from stolab_data.data_service     import get_ohlcv      # type: ignore
    from magic_formula.scoring.scorer import compute_scores

    try:
        start = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=500)).strftime("%Y%m%d")
        df = get_ohlcv(ticker, start_date=start, end_date=end_date)
        if df is None or len(df) < 80:
            return None
        df = df.set_index("Date").sort_index()
        return compute_scores(df, weights, area4_mode=area4_mode)
    except Exception as e:
        warnings.warn(f"[daily.runner] {ticker} 처리 실패: {e}")
        return None


# ---------------------------------------------------------------------------
# 신호 감지
# ---------------------------------------------------------------------------

def check_r1_signal(scored: pd.DataFrame, threshold: float) -> bool:
    """
    최신 날짜 기준 R1 신호 여부 (이전 ≤ threshold < 오늘).

    반환 타입은 Python ``bool``. pandas/numpy 스칼라가 ``numpy.bool_`` 으로
    leak 되지 않도록 명시 변환한다 (JSON 직렬화 / `is True` 비교 호환).
    """
    score = scored["composite_score"]
    if len(score) < 2:
        return False
    return bool((score.iloc[-2] <= threshold) and (score.iloc[-1] > threshold))


# ---------------------------------------------------------------------------
# 전일 OHLCV 정보 추출
# ---------------------------------------------------------------------------

def get_prev_day_info(scored: pd.DataFrame) -> dict:
    """전일(최신일) OHLCV + 등락률 + 거래량 변동률."""
    if len(scored) < 2:
        return {}
    today = scored.iloc[-1]
    prev  = scored.iloc[-2]

    close       = float(today["Close"])
    open_       = float(today["Open"])
    high        = float(today["High"])
    low         = float(today["Low"])
    volume      = int(today["Volume"])
    prev_volume = int(prev["Volume"])
    change_pct  = round((close - prev["Close"]) / prev["Close"] * 100, 2) if prev["Close"] else 0.0
    vol_chg_pct = round((volume - prev_volume) / prev_volume * 100, 2) if prev_volume else 0.0

    return {
        "open":        open_,
        "high":        high,
        "low":         low,
        "close":       close,
        "volume":      volume,
        "change_pct":  change_pct,
        "vol_chg_pct": vol_chg_pct,
    }


# ---------------------------------------------------------------------------
# 메인 파이프라인
# ---------------------------------------------------------------------------

def run(
    target_date: str | None = None,
    config_path: str | None = None,
) -> dict:
    """
    데일리 파이프라인 실행.

    Parameters
    ----------
    target_date : vault 조회 종료 일자 ('YYYYMMDD'). None 이면 오늘.
                  이 값까지의 데이터를 vault 에서 받아온다.
    config_path : active_strategy.yaml 경로. None 이면 기본 위치.

    Returns
    -------
    결과 dict (JSON 저장용). 'date' 는 **실제 데이터의 마지막 거래일** 이며,
    'executed_at' 은 스크립트 실행 시점이다. 파일명도 데이터 기준일 사용.

    파일명 규칙
    -----------
    실행 시점이 아닌 **데이터 마지막 거래일** (모든 종목 중 최빈값) 을 사용한다.
    즉 5/20 오후에 실행해도 vault 의 가장 최근 종가가 5/19 라면
    daily_signal_20260519.json/md 로 저장된다.
    """
    from collections import Counter

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from magic_formula.config import load_strategy

    cfg = load_strategy(config_path)
    print(f"[daily.runner] 전략 설정: {cfg.summary()}")

    executed_at = datetime.today().strftime("%Y%m%d")
    if target_date is None:
        target_date = executed_at

    # universe — yaml 의 universe 식별자 → ticker 목록
    tickers = get_universe(cfg.universe)
    if not tickers:   # vault 미설치 안전망
        tickers = list(SECTOR_MAP.keys())
    print(f"[daily.runner] 실행={executed_at}  조회종료={target_date}  universe={cfg.universe} ({len(tickers)}개)")

    all_results: list[dict] = []
    signals:     list[dict] = []
    last_dates:  list = []           # 종목별 마지막 거래일 수집 (data_date 산정용)

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i:02d}/{len(tickers)}] {ticker} ...", end=" ", flush=True)
        scored = fetch_and_score(ticker, target_date, cfg.weights, cfg.area4_mode)
        if scored is None:
            print("SKIP (데이터 부족)")
            continue

        latest = scored.iloc[-1]
        prev   = scored.iloc[-2] if len(scored) >= 2 else scored.iloc[-1]

        # 마지막 거래일 — 종목별 최댓값 보존, 나중에 Counter 로 최빈값 결정
        last_dates.append(scored.index[-1])

        name   = get_ticker_name(ticker, latest.get("Name"))
        sector = SECTOR_MAP.get(ticker, "기타")

        composite = round(float(latest["composite_score"]), 2)
        prev_comp = round(float(prev["composite_score"]),   2)
        area1     = round(float(latest["area1_trend"]),     2)
        area2     = round(float(latest["area2_momentum"]),  2)
        area3     = round(float(latest["area3_volume"]),    2)
        area4     = round(float(latest["area4_volatility"]),2)
        area5     = round(float(latest["area5_wyckoff"]),   2)

        is_signal = check_r1_signal(scored, cfg.threshold)

        row = {
            "ticker":          ticker,
            "name":            name,
            "sector":          sector,
            "composite_score": composite,
            "prev_score":      prev_comp,
            "area1_trend":     area1,
            "area2_momentum":  area2,
            "area3_volume":    area3,
            "area4_volatility": area4,
            "area5_wyckoff":   area5,
            "is_signal":       is_signal,
            **get_prev_day_info(scored),
        }
        all_results.append(row)

        if is_signal:
            signals.append(row)
            print(f"✅ 신호! score={composite:.2f} (prev={prev_comp:.2f})")
        else:
            print(f"score={composite:.2f}")

    # 섹터별 정렬
    def sector_sort_key(r):
        try:
            return SECTOR_ORDER.index(r["sector"])
        except ValueError:
            return 999

    signals.sort(key=lambda r: (sector_sort_key(r), r["sector"], r["ticker"]))
    all_results.sort(key=lambda r: (sector_sort_key(r), r["sector"], r["ticker"]))

    # 데이터 기준일 산정 — 모든 종목의 마지막 거래일 중 최빈값
    # (휴장/거래정지 종목이 있어도 다수 종목이 동의하는 날짜로 안정)
    if last_dates:
        data_date_dt = Counter(last_dates).most_common(1)[0][0]
        data_date    = data_date_dt.strftime("%Y%m%d")
    else:
        # 모든 종목 SKIP — 데이터 기준일을 잡을 수 없음. 실행일로 폴백.
        data_date = executed_at
        print(f"[daily.runner] ⚠ 처리된 종목 0개 — 파일명을 실행일({executed_at})로 폴백")

    result = {
        "date":           data_date,          # ★ 데이터 마지막 거래일
        "executed_at":    executed_at,        # ★ 실행 시점
        "strategy_id":    cfg.strategy_id,
        "rule":           cfg.rule,
        "threshold":      cfg.threshold,
        "weights":        dict(cfg.weights),
        "area4_mode":     cfg.area4_mode,
        "total_tickers":  len(all_results),
        "signal_count":   len(signals),
        "signals":        signals,
        "all_scores":     all_results,
    }

    json_path = OUTPUT_DIR / f"daily_signal_{data_date}.json"
    md_path   = OUTPUT_DIR / f"daily_signal_{data_date}.md"

    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"\n[daily.runner] JSON 저장: {json_path}")

    _write_md(result, md_path)
    print(f"[daily.runner] MD 저장:   {md_path}")

    return result


# ---------------------------------------------------------------------------
# 직렬화 헬퍼
# ---------------------------------------------------------------------------

def _json_default(obj):
    """numpy bool_ / int_ / float_ → Python 기본 타입 변환."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_md(result: dict, path: Path) -> None:
    """결과 dict → Markdown 리포트."""
    date_str     = result["date"]
    signal_count = result["signal_count"]
    signals      = result["signals"]
    all_scores   = result["all_scores"]

    d = datetime.strptime(date_str, "%Y%m%d")
    date_ko = f"{d.year}.{d.month:02d}.{d.day:02d}"

    lines = [
        f"# 황금률 진입신호 리포트 — {date_ko}",
        "",
        f"- **진입 신호**: {signal_count}종목",
        f"- **규칙**: {result['rule']} (threshold={result['threshold']:.1f})",
        f"- **가중치**: T={result['weights']['trend']*100:.0f}% "
        f"M={result['weights']['momentum']*100:.0f}% "
        f"V={result['weights']['volume']*100:.0f}% "
        f"Vo={result['weights']['volatility']*100:.0f}% "
        f"W={result['weights']['wyckoff']*100:.0f}%",
        f"- **대상 종목**: {result['total_tickers']}개",
        "",
    ]

    if signals:
        lines += ["## 📈 진입 신호 종목", ""]
        current_sector = None
        for s in signals:
            if s["sector"] != current_sector:
                current_sector = s["sector"]
                lines += [f"### [{current_sector}]", ""]
                lines += ["| 종목 | 종합점수 | 추세 | 모멘텀 | 거래량 | 변동성 | 심리 | 종가 | 등락 | 거래량변동 |", ""]
                lines += ["|------|---------|------|--------|--------|--------|------|------|------|-----------|", ""]
            chg = s.get("change_pct", 0)
            vol = s.get("vol_chg_pct", 0)
            lines.append(
                f"| {s['name']}({s['ticker']}) "
                f"| **{s['composite_score']:.2f}** "
                f"| {s['area1_trend']:.1f} "
                f"| {s['area2_momentum']:.1f} "
                f"| {s['area3_volume']:.1f} "
                f"| {s['area4_volatility']:.1f} "
                f"| {s['area5_wyckoff']:.1f} "
                f"| {s.get('close', '-'):,.0f} "
                f"| {'+' if chg>=0 else ''}{chg:.1f}% "
                f"| {'+' if vol>=0 else ''}{vol:.0f}% |"
            )
        lines += [""]
    else:
        lines += ["## 📭 오늘 진입 신호 없음", ""]

    lines += ["## 전체 종목 현황", ""]
    lines += ["| 섹터 | 종목 | 종합점수 | 추세 | 모멘텀 | 거래량 | 변동성 | 심리 |", ""]
    lines += ["|------|------|---------|------|--------|--------|--------|------|", ""]
    for r in all_scores:
        signal_marker = " 🔔" if r["is_signal"] else ""
        lines.append(
            f"| {r['sector']} "
            f"| {r['name']}({r['ticker']}){signal_marker} "
            f"| {r['composite_score']:.2f} "
            f"| {r['area1_trend']:.1f} "
            f"| {r['area2_momentum']:.1f} "
            f"| {r['area3_volume']:.1f} "
            f"| {r['area4_volatility']:.1f} "
            f"| {r['area5_wyckoff']:.1f} |"
        )
    lines += [""]

    path.write_text("\n".join(lines), encoding="utf-8")
