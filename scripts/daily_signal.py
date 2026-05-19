#!/usr/bin/env python3
"""
scripts/daily_signal.py
-----------------------
매일 실행: 59개 코어 종목에 대해 황금률 점수를 계산하고
R1 진입신호(composite_score가 +5.0 상향 돌파)가 발생한 종목을
섹터별로 정리해 JSON + Markdown 리포트로 저장한다.

확정 설정
---------
  가중치(CompR15): T=30%, M=15%, V=10%, Vo=25%, W=20%
  진입 규칙(R1):   composite_score가 +5.0 상향 돌파한 첫날
  Area4 모드:      trend (추세추종)
  threshold:       +5.0

출력
----
  Magic Formula/output/signals/daily_signal_YYYYMMDD.json
  Magic Formula/output/signals/daily_signal_YYYYMMDD.md

실행
----
  python scripts/daily_signal.py            # 오늘 날짜
  python scripts/daily_signal.py 20260519   # 특정 날짜(테스트용)
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------

MAGIC_FORMULA_ROOT = Path(__file__).parent.parent
SRC_PATH           = MAGIC_FORMULA_ROOT / "src"
LONGLIVEVAULT_PATH = Path("/Users/kaneyoun/DriveForALL/StoLab/longlivevault")
OUTPUT_DIR         = MAGIC_FORMULA_ROOT / "output" / "signals"

for p in [str(SRC_PATH), str(LONGLIVEVAULT_PATH)]:
    if p not in sys.path:
        sys.path.insert(0, p)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 확정 설정
# ---------------------------------------------------------------------------

COMPRE15_WEIGHTS: dict[str, float] = {
    "trend":      0.30,
    "momentum":   0.15,
    "volume":     0.10,
    "volatility": 0.25,
    "wyckoff":    0.20,
}

AREA4_MODE  = "trend"
THRESHOLD   = 5.0      # R1 상향 돌파 임계값 (백테스트 설정과 동일)
RULE        = "R1"

# ---------------------------------------------------------------------------
# 섹터 매핑
# ---------------------------------------------------------------------------

SECTOR_MAP: dict[str, str] = {
    # 반도체
    "000660": "반도체", "005930": "반도체", "042700": "반도체",
    "058470": "반도체", "240810": "반도체", "039030": "반도체",
    "000990": "반도체", "403870": "반도체", "357780": "반도체",
    "005290": "반도체",
    # 반도체핵심장비
    "000150": "반도체핵심장비", "007660": "반도체핵심장비", "095340": "반도체핵심장비",
    # 로봇
    "058610": "로봇", "277810": "로봇", "108490": "로봇",
    "454910": "로봇", "348340": "로봇", "056080": "로봇",
    # 에너지.전송
    "010120": "에너지.전송", "298040": "에너지.전송",
    "267260": "에너지.전송", "001440": "에너지.전송",
    # 에너지.생산
    "000720": "에너지.생산", "047040": "에너지.생산", "034020": "에너지.생산",
    "052690": "에너지.생산", "032820": "에너지.생산",
    # 에너지.보관
    "373220": "에너지.보관", "006400": "에너지.보관", "005490": "에너지.보관",
    "247540": "에너지.보관", "051910": "에너지.보관", "086520": "에너지.보관",
    # 바이오
    "068270": "바이오", "196170": "바이오", "207940": "바이오", "0126Z0": "바이오",
    # 조선
    "329180": "조선", "042660": "조선", "010140": "조선",
    # 방산
    "012450": "방산", "047810": "방산", "064350": "방산",
    "272210": "방산", "079550": "방산",
    # 은행
    "055550": "은행", "086790": "은행", "105560": "은행", "316140": "은행",
    # 증권
    "006800": "증권", "071050": "증권", "016360": "증권", "039490": "증권",
    # 인터넷통신
    "017670": "인터넷통신", "030200": "인터넷통신", "032640": "인터넷통신",
    "035420": "인터넷통신", "035720": "인터넷통신",
}

SECTOR_ORDER = [
    "반도체", "반도체핵심장비", "로봇",
    "에너지.전송", "에너지.생산", "에너지.보관",
    "바이오", "조선", "방산",
    "은행", "증권", "인터넷통신",
]

# ---------------------------------------------------------------------------
# 데이터 로드 & 점수 계산
# ---------------------------------------------------------------------------

def load_modules():
    """런타임 import (경로 설정 후 호출)."""
    global get_ohlcv, compute_scores, entry_signals
    from stolab_data.data_service import get_ohlcv        # type: ignore
    from scoring.scorer            import compute_scores   # type: ignore
    from signals.rules             import entry_signals    # type: ignore


def fetch_and_score(ticker: str, end_date: str) -> pd.DataFrame | None:
    """
    종목 OHLCV 로드 → 점수 계산.
    실패 시 None 반환 (개별 종목 오류가 전체를 중단시키지 않도록).
    """
    try:
        start = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=500)).strftime("%Y%m%d")
        df = get_ohlcv(ticker, start_date=start, end_date=end_date)
        if df is None or len(df) < 80:
            return None
        # Date 컬럼 → index
        df = df.set_index("Date").sort_index()
        scored = compute_scores(df, COMPRE15_WEIGHTS, area4_mode=AREA4_MODE)
        return scored
    except Exception as e:
        warnings.warn(f"[daily_signal] {ticker} 처리 실패: {e}")
        return None


# ---------------------------------------------------------------------------
# R1 신호 감지
# ---------------------------------------------------------------------------

def check_r1_signal(scored: pd.DataFrame, threshold: float = THRESHOLD) -> bool:
    """최신 날짜 기준 R1 신호 여부 반환."""
    score = scored["composite_score"]
    if len(score) < 2:
        return False
    today_score = score.iloc[-1]
    prev_score  = score.iloc[-2]
    return (prev_score <= threshold) and (today_score > threshold)


# ---------------------------------------------------------------------------
# 전일 OHLCV 정보 추출
# ---------------------------------------------------------------------------

def get_prev_day_info(scored: pd.DataFrame) -> dict:
    """전일(최신일) OHLCV + 거래량 변동률 반환."""
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
        "change_pct":  change_pct,    # 전일 대비 등락률 (%)
        "vol_chg_pct": vol_chg_pct,   # 전일 대비 거래량 변동률 (%)
    }


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def run(target_date: str | None = None) -> dict:
    """
    전체 파이프라인 실행.
    Returns: 결과 dict (JSON 저장용)
    """
    load_modules()

    if target_date is None:
        target_date = datetime.today().strftime("%Y%m%d")

    tickers = list(SECTOR_MAP.keys())
    print(f"[daily_signal] 실행 날짜: {target_date} | 대상 종목: {len(tickers)}개")

    # 전체 종목 점수 계산
    all_results: list[dict] = []
    signals: list[dict] = []

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i:02d}/{len(tickers)}] {ticker} ...", end=" ", flush=True)
        scored = fetch_and_score(ticker, target_date)
        if scored is None:
            print("SKIP (데이터 부족)")
            continue

        latest = scored.iloc[-1]
        prev   = scored.iloc[-2] if len(scored) >= 2 else scored.iloc[-1]

        name   = str(latest.get("Name", ticker))
        sector = SECTOR_MAP.get(ticker, "기타")

        composite    = round(float(latest["composite_score"]), 2)
        prev_comp    = round(float(prev["composite_score"]), 2)
        area1        = round(float(latest["area1_trend"]), 2)
        area2        = round(float(latest["area2_momentum"]), 2)
        area3        = round(float(latest["area3_volume"]), 2)
        area4        = round(float(latest["area4_volatility"]), 2)
        area5        = round(float(latest["area5_wyckoff"]), 2)

        is_signal = check_r1_signal(scored, THRESHOLD)

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

    result = {
        "date":           target_date,
        "rule":           RULE,
        "threshold":      THRESHOLD,
        "weights":        COMPRE15_WEIGHTS,
        "area4_mode":     AREA4_MODE,
        "total_tickers":  len(all_results),
        "signal_count":   len(signals),
        "signals":        signals,
        "all_scores":     all_results,
    }

    # 저장
    date_str = target_date
    json_path = OUTPUT_DIR / f"daily_signal_{date_str}.json"
    md_path   = OUTPUT_DIR / f"daily_signal_{date_str}.md"

    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[daily_signal] JSON 저장: {json_path}")

    _write_md(result, md_path)
    print(f"[daily_signal] MD 저장:   {md_path}")

    return result


def _write_md(result: dict, path: Path) -> None:
    """결과 dict → Markdown 리포트."""
    date_str      = result["date"]
    signal_count  = result["signal_count"]
    signals       = result["signals"]
    all_scores    = result["all_scores"]

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


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    result = run(target)
    print(f"\n✅ 완료: 신호 {result['signal_count']}종목 / 전체 {result['total_tickers']}종목")
