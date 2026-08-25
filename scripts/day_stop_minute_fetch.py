#!/usr/bin/env python3
"""
day_stop_minute_fetch.py — 데이 포트 청산규칙 검증용 KIS 분봉 수집기

목적: 데이 포트 손절폭 변동배율(YZ_20 연동) 검증에 필요한 **장중 경로**를 확보한다.
      LLV 에는 분봉 저장이 없으므로 KIS FHKST03010230 으로 직접 긁어 캐시한다.

⚠ 조회 전용 — LLV parquet 정본을 건드리지 않는다. 캐시는 이 스크립트 전용 경로.
⚠ KIS 분봉 보관 한도 **최대 1년** — 그 이전 일자는 빈 결과.

캐시: MagicFormula/output/day_stop_study/minute_{ticker}_{YYYYMMDD}.json
      (일자별 파일 — 재실행 시 기존 파일은 건너뛴다. 멱등)

사용:
    python3 scripts/day_stop_minute_fetch.py --tickers 319660,068270 --months 6
    python3 scripts/day_stop_minute_fetch.py --tickers 319660 --start 20260201 --end 20260814
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

# LLV 는 StoLab/ 아래 형제 저장소 — 머신(미니/에어) 무관 상대 경로.
# 빈 값은 미설정 취급(`or`), "~/..." 표기는 expanduser 로 편다.
STOLAB_ROOT = Path(__file__).resolve().parents[2]   # StoLab/
LLV = Path(os.getenv("LLV_PATH") or str(STOLAB_ROOT / "LongLiveVault")).expanduser()
sys.path.insert(0, str(LLV))

from stolab_data.kis_fetcher import fetch_minute_bars_on_date  # noqa: E402
from stolab_data.trading_calendar import is_trading_day        # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "output" / "day_stop_study"

# 정규장 09:00~15:30. KIS 는 기준시각부터 과거로 ~30봉을 준다 → 30분 간격으로 전진 스캔.
CHUNK_ENDS = ["0930", "1000", "1030", "1100", "1130", "1200",
              "1230", "1300", "1330", "1400", "1430", "1500", "1530"]


def fetch_day(ticker: str, d: date, sleep: float = 0.12) -> dict | None:
    """하루치 분봉 전체를 {HHMMSS: {...}} 로. 하나라도 실패하면 None (부분 저장 금지)."""
    ymd = d.strftime("%Y%m%d")
    bars: dict[str, dict] = {}
    for end in CHUNK_ENDS:
        chunk = fetch_minute_bars_on_date(ticker, ymd, end)
        if chunk is None:
            return None                      # API 실패 — 캐시하지 않는다 (재시도 여지)
        for b in chunk:
            bars[b["Time"]] = {"p": b["Price"], "h": b["High"], "l": b["Low"], "o": b["Open"]}
        time.sleep(sleep)
    return bars or None                      # 휴장/미상장 → 빈 dict → None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", required=True, help="쉼표 구분 종목코드")
    ap.add_argument("--months", type=int, default=6)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--sleep", type=float, default=0.12)
    a = ap.parse_args()

    end = datetime.strptime(a.end, "%Y%m%d").date() if a.end else date.today()
    start = (datetime.strptime(a.start, "%Y%m%d").date() if a.start
             else end - timedelta(days=int(a.months * 30.5)))

    days = []
    cur = start
    while cur <= end:
        if is_trading_day(cur):
            days.append(cur)
        cur += timedelta(days=1)

    OUT.mkdir(parents=True, exist_ok=True)
    tickers = [t.strip() for t in a.tickers.split(",") if t.strip()]
    print(f"대상 {tickers} · 거래일 {len(days)}일 ({start}~{end}) · 예상 콜 "
          f"{len(tickers) * len(days) * len(CHUNK_ENDS):,}회", flush=True)

    t0 = time.time()
    done = skip = fail = 0
    for tk in tickers:
        for i, d in enumerate(days, 1):
            f = OUT / f"minute_{tk}_{d:%Y%m%d}.json"
            if f.exists():
                skip += 1
                continue
            bars = fetch_day(tk, d, a.sleep)
            if bars is None:
                fail += 1
                print(f"  ✗ {tk} {d} 실패/무자료", flush=True)
                continue
            f.write_text(json.dumps(bars), encoding="utf-8")
            done += 1
            if done % 10 == 0:
                el = time.time() - t0
                print(f"  {tk} {i}/{len(days)}  수집 {done} 건너뜀 {skip} 실패 {fail}  "
                      f"{el/60:.1f}분 경과 ({el/max(done,1):.1f}초/일)", flush=True)

    print(f"\n완료 — 수집 {done} · 기존 {skip} · 실패 {fail} · {(time.time()-t0)/60:.1f}분")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
