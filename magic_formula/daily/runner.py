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

    # ── system_version 분기 (config.py 검증 전에 raw 로 확인) ──
    # v2 yaml 은 구조가 달라 ActiveStrategy.from_dict 가 거부하므로 먼저 raw 읽기.
    if _is_v2_config(config_path):
        print("[daily.runner] system_version=v2_combined → 결합 시스템 경로")
        return run_combined(target_date=target_date, config_path=config_path)

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


# ===========================================================================
# v2 — 5영역 결합 시스템 (M4 분석 확정, docs/area_specs/combined.md)
# ===========================================================================
# 데이터: LLV 가 채운 full-column OHLCV + Wyckoff_Label/Signal/Signal_Strength
#        를 그대로 받아 쓴다. Magic Formula 는 hillstorm 을 import 하지 않는다.
# 점수:  area_scores.compute_combined_score (영역별 레짐 + Markdown 게이트)
# 신호:  종합점수 prev <= threshold < today 돌파
# 산출:  daily_signal_YYYYMMDD.{json,md} — 기존 파일명 유지, 컬럼 확장

_DEFAULT_CONFIG = _PROJ_ROOT / "configs" / "active_strategy.yaml"


def _is_v2_config(config_path: str | None) -> bool:
    """yaml 을 raw 로 읽어 system_version == 'v2_combined' 여부 판정."""
    import yaml
    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    if not path.exists():
        return False
    try:
        d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return str(d.get("system_version", "")).strip() == "v2_combined"
    except Exception:
        return False


def _load_v2_config(config_path: str | None) -> dict:
    """v2 yaml 파싱 (검증 없이 raw dict)."""
    import yaml
    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _v2_load_ohlcv_full(ticker: str, end_date: str):
    """v2 — collector.fetch_ohlcv 로 vault full-column OHLCV 로드.

    collector 가 vault 의 모든 컬럼(지표/Wyckoff) 을 그대로 통과시킴 (2026-05-31 변경).
    Magic Formula 는 LLV 가 채운 컬럼을 받아 쓰기만 — hillstorm 직접 호출 없음.
    """
    from magic_formula.data.collector import fetch_ohlcv
    start = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=500)).strftime("%Y%m%d")
    df = fetch_ohlcv(ticker, start, end_date)
    if df is None or df.empty or len(df) < 80:
        return None
    return df


def _v2_phase_series(df: pd.DataFrame) -> pd.Series:
    """LLV 가 채운 Wyckoff_Label 컬럼을 그대로 게이트 입력으로 반환.

    없으면 빈 시리즈 → 게이트 NaN 매칭 (Markdown 아닌 것으로 처리되어 게이트 무력).
    Magic Formula 는 hillstorm 을 import 하지 않는다 — LLV 책임.
    """
    if "Wyckoff_Label" in df.columns:
        return df["Wyckoff_Label"]
    warnings.warn(
        "[daily.runner.v2] Wyckoff_Label 컬럼 없음 — LLV 백필이 안 됐을 가능성. "
        "게이트가 무력화됩니다.",
        stacklevel=2,
    )
    return pd.Series(index=df.index, dtype=object)


def run_combined(
    target_date: str | None = None,
    config_path: str | None = None,
) -> dict:
    """
    v2 결합 시스템 데일리 파이프라인.

    1. 전 종목 full-column OHLCV 로드 (vault → collector — Wyckoff 포함)
    2. Wyckoff_Label 게이트 시리즈 (LLV 가 채운 값)
    3. make_regimes → breadth + quickregime (전 종목 횡단면)
    4. compute_combined_score (robust 가중 + Markdown 게이트)
    5. 신호: 종합점수 prev <= threshold < today
    6. JSON + MD 저장 (기존 파일명 유지, Wyckoff_Signal/Strength 컬럼 추가)
    """
    from collections import Counter

    from magic_formula.analysis import area_scores as A
    from magic_formula.analysis.area_scores import COMBINED_WEIGHTS, COMBINED_THRESHOLD

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = _load_v2_config(config_path)
    sc = raw.get("scoring", {})
    weights = sc.get("weights", COMBINED_WEIGHTS)
    threshold = float(sc.get("threshold", COMBINED_THRESHOLD))
    gate_cfg = sc.get("gate", {})
    gate = bool(gate_cfg.get("enabled", True))
    strategy_id = raw.get("strategy_id", "COMBINED-v2")
    universe = (raw.get("scoring", {}).get("universe")
                or raw.get("universe", "core_excl_split"))

    executed_at = datetime.today().strftime("%Y%m%d")
    if target_date is None:
        target_date = executed_at

    tickers = get_universe(universe)
    if not tickers:
        tickers = list(SECTOR_MAP.keys())
    print(f"[daily.runner.v2] 실행={executed_at} 조회종료={target_date} "
          f"universe={universe} ({len(tickers)}개) thr={threshold} gate={gate}")

    # ── 1. 전 종목 로드 (레짐은 횡단면이라 전체 필요) ──
    stock_data: dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = _v2_load_ohlcv_full(t, target_date)
        if df is not None:
            stock_data[t] = df
    print(f"[daily.runner.v2] 로드 완료: {len(stock_data)}종목")

    if not stock_data:
        print("[daily.runner.v2] ⚠ 처리된 종목 0개 — 종료")
        return {}

    # ── 2. Wyckoff 국면 (LLV 가 채운 컬럼) ──
    phases = {t: _v2_phase_series(df) for t, df in stock_data.items()}

    # ── 3. 레짐 ──
    regime_b, regime_q = A.make_regimes(stock_data)

    # ── 4~5. 종합점수 + 신호 ──
    all_results: list[dict] = []
    signals: list[dict] = []
    last_dates: list = []

    for t, df in stock_data.items():
        comp = A.compute_combined_score(df, regime_b, regime_q, phases[t], weights, gate)
        if comp.dropna().empty or len(comp) < 2:
            continue
        last_dates.append(df.index[-1])

        st = A.score_trend(df, regime_b)
        sm = A.score_momentum(df)
        sv = A.score_volume(df, regime_q)
        sp = A.score_volatility(df, regime_q)

        cur = comp.iloc[-1]
        prv = comp.iloc[-2]
        # 게이트 제외(NaN)면 신호/점수 없음
        cur_v = float(cur) if pd.notna(cur) else None
        prv_v = float(prv) if pd.notna(prv) else None
        is_signal = bool(prv_v is not None and cur_v is not None
                         and prv_v <= threshold < cur_v)

        phase_now = phases[t].iloc[-1] if len(phases[t]) else ""

        # Wyckoff 전환 신호 + 강도 (LLV 가 채운 컬럼 — 평소 None)
        wy_sig = (df["Wyckoff_Signal"].iloc[-1]
                  if "Wyckoff_Signal" in df.columns else None)
        wy_str = (df["Wyckoff_Signal_Strength"].iloc[-1]
                  if "Wyckoff_Signal_Strength" in df.columns else None)
        if pd.isna(wy_sig):
            wy_sig = None
        if pd.isna(wy_str):
            wy_str = None
        else:
            try:
                wy_str = int(wy_str)
            except (ValueError, TypeError):
                wy_str = None

        name = get_ticker_name(t, df.iloc[-1].get("Name") if "Name" in df.columns else None)
        sector = SECTOR_MAP.get(t, "기타")

        row = {
            "ticker": t, "name": name, "sector": sector,
            "composite_score": round(cur_v, 2) if cur_v is not None else None,
            "prev_score": round(prv_v, 2) if prv_v is not None else None,
            "area1_trend": round(float(st.iloc[-1]), 2),
            "area2_momentum": round(float(sm.iloc[-1]), 2),
            "area3_volume": round(float(sv.iloc[-1]), 2),
            "area4_volatility": round(float(sp.iloc[-1]), 2),
            "area5_wyckoff": 0.0,                 # v2 는 Wyckoff 점수 미사용 (게이트로)
            "wyckoff_phase": str(phase_now) if phase_now and pd.notna(phase_now) else "",
            "wyckoff_signal": str(wy_sig) if wy_sig else "",
            "wyckoff_signal_strength": wy_str,
            "gated_out": bool(cur_v is None),     # Markdown 등으로 제외됨
            "is_signal": is_signal,
            **_v2_prev_day_info(df),
        }
        all_results.append(row)
        if is_signal:
            signals.append(row)

    # 정렬
    def sk(r):
        try:
            return SECTOR_ORDER.index(r["sector"])
        except ValueError:
            return 999
    signals.sort(key=lambda r: (sk(r), r["sector"], r["ticker"]))
    all_results.sort(key=lambda r: (sk(r), r["sector"], r["ticker"]))

    data_date = (Counter(last_dates).most_common(1)[0][0].strftime("%Y%m%d")
                 if last_dates else executed_at)

    result = {
        "date": data_date, "executed_at": executed_at,
        "strategy_id": strategy_id, "rule": "threshold_breakout",
        "threshold": threshold, "weights": dict(weights),
        "system_version": "v2_combined", "gate": gate,
        "total_tickers": len(all_results), "signal_count": len(signals),
        "signals": signals, "all_scores": all_results,
    }

    json_path = OUTPUT_DIR / f"daily_signal_{data_date}.json"
    md_path = OUTPUT_DIR / f"daily_signal_{data_date}.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8")
    print(f"[daily.runner.v2] JSON 저장: {json_path}")
    _write_md_v2(result, md_path)
    print(f"[daily.runner.v2] MD 저장:   {md_path}")
    return result


def _v2_prev_day_info(df: pd.DataFrame) -> dict:
    """v2 — 전일 OHLCV 정보 (df 는 raw OHLCV 인덱스)."""
    if len(df) < 2:
        return {}
    today = df.iloc[-1]; prev = df.iloc[-2]
    close = float(today["Close"]); pv = int(prev["Volume"]) if "Volume" in df else 0
    vol = int(today["Volume"]) if "Volume" in df else 0
    return {
        "open": float(today["Open"]), "high": float(today["High"]),
        "low": float(today["Low"]), "close": close, "volume": vol,
        "change_pct": round((close - prev["Close"]) / prev["Close"] * 100, 2) if prev["Close"] else 0.0,
        "vol_chg_pct": round((vol - pv) / pv * 100, 2) if pv else 0.0,
    }


def _phase_short(label: str) -> str:
    """국면 라벨 → 4글자 약어 (md 표 가독성)."""
    mapping = {
        "Accumulation": "Accum",
        "Markup": "Markup",
        "Distribution": "Dist",
        "Markdown": "Mark↓",
    }
    return mapping.get(label, label[:6] if label else "")


def _signal_short(sig: str, strength) -> str:
    """전환 신호 + 강도 → 표 셀 텍스트.

    예: ACC_COMPLETE strength=2 → 'ACC↑(2)'
        DIST_CONFIRM strength=3 → 'DIST↓(3)'
        평소 None → ''
    """
    if not sig:
        return ""
    label_map = {
        "ACC_COMPLETE":    "ACC↑",
        "DIST_CONFIRM":    "DIST↓",
        "MARKUP_WATCH":    "MU↘",
        "MARKDOWN_WATCH":  "MD↗",
        "PANIC_ZONE":      "PANIC",
    }
    short = label_map.get(sig, sig[:6])
    if strength is not None:
        return f"{short}({strength})"
    return short


def _write_md_v2(result: dict, path: Path) -> None:
    """v2 결과 → Markdown.

    표 컬럼 순서 (Kane 지시 2026-05-31):
      종목 | 국면 | 전환신호 | 종합점수 | 추세 | 모멘텀 | 거래량 | 변동성 | 종가 | 등락 | 거래량변동
    국면은 게이트 역할이므로 점수 앞에 배치. 전환신호는 5종(ACC/DIST/MU/MD/PANIC) 모두 기록.
    """
    d = datetime.strptime(result["date"], "%Y%m%d")
    date_ko = f"{d.year}.{d.month:02d}.{d.day:02d}"
    w = result["weights"]
    lines = [
        f"# 황금률 진입신호 리포트 (v2 결합) — {date_ko}", "",
        f"- **진입 신호**: {result['signal_count']}종목",
        f"- **전략**: {result['strategy_id']} (threshold={result['threshold']:.1f}, "
        f"게이트={'ON' if result['gate'] else 'OFF'})",
        f"- **가중치**: T={w.get('trend',0)*100:.0f}% M={w.get('momentum',0)*100:.0f}% "
        f"Vu={w.get('volume',0)*100:.0f}% Va={w.get('volatility',0)*100:.0f}%",
        f"- **대상 종목**: {result['total_tickers']}개", "",
    ]

    # 진입 신호 표 (섹터별 그룹)
    if result["signals"]:
        lines += ["## 📈 진입 신호 종목", ""]
        # 섹터별 그룹화
        from itertools import groupby
        for sector, rows in groupby(result["signals"], key=lambda r: r["sector"]):
            rows = list(rows)
            lines += [
                f"### [{sector}]", "",
                "| 종목 | 국면 | 전환신호 | 종합점수 | 추세 | 모멘텀 | 거래량 | 변동성 | 종가 | 등락 | 거래량변동 |",
                "|------|------|---------|---------|------|--------|--------|--------|------|------|-----------|",
            ]
            for s in rows:
                chg = s.get("change_pct", 0)
                vol_chg = s.get("vol_chg_pct", 0)
                lines.append(
                    f"| {s['name']}({s['ticker']}) "
                    f"| {_phase_short(s.get('wyckoff_phase',''))} "
                    f"| {_signal_short(s.get('wyckoff_signal',''), s.get('wyckoff_signal_strength'))} "
                    f"| **{s['composite_score']:.2f}** "
                    f"| {s['area1_trend']:.1f} | {s['area2_momentum']:.1f} "
                    f"| {s['area3_volume']:.1f} | {s['area4_volatility']:.1f} "
                    f"| {s.get('close','-'):,.0f} "
                    f"| {'+' if chg>=0 else ''}{chg:.1f}% "
                    f"| {'+' if vol_chg>=0 else ''}{vol_chg:.0f}% |"
                )
            lines += [""]
    else:
        lines += ["## 📭 오늘 진입 신호 없음", ""]

    # 전체 종목 현황
    lines += ["## 전체 종목 현황", "",
              "| 섹터 | 종목 | 국면 | 전환신호 | 종합점수 | 추세 | 모멘텀 | 거래량 | 변동성 |",
              "|------|------|------|---------|---------|------|--------|--------|--------|"]
    for r in result["all_scores"]:
        mark = " 🔔" if r["is_signal"] else (" ⛔" if r.get("gated_out") else "")
        cs = f"{r['composite_score']:.2f}" if r["composite_score"] is not None else "—(게이트)"
        lines.append(
            f"| {r['sector']} | {r['name']}({r['ticker']}){mark} "
            f"| {_phase_short(r.get('wyckoff_phase',''))} "
            f"| {_signal_short(r.get('wyckoff_signal',''), r.get('wyckoff_signal_strength'))} "
            f"| {cs} "
            f"| {r['area1_trend']:.1f} | {r['area2_momentum']:.1f} "
            f"| {r['area3_volume']:.1f} | {r['area4_volatility']:.1f} |"
        )
    lines += [""]
    path.write_text("\n".join(lines), encoding="utf-8")
