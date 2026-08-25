#!/usr/bin/env python3
"""
day_stop_sweep.py — 손절폭 반응곡선 (데이 포트 변동배율 직접 검증)

핵심 질문: **종목별 최적 손절폭이 그 종목의 변동성에 비례하는가?**
  비례한다면 → 손절폭 = base × 배율(YZ_20 연동) 이라는 선형 스케일이 옳다.
  종목마다 최적점이 같다면 → 배율은 불필요하다.

방법: day_stop_backtest.simulate 를 그대로 쓰되 손절폭을 0.5~4.0% 로 쓸어
      종목별 반응곡선을 그린다. 최적점 비 vs 배율 비를 비교한다.

⚠ 이 스윕은 종목 2개의 결과다 — 유니버스 전체의 결론이 아니다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from day_stop_backtest import (MAX_HOLD_TD, load_minutes, load_scale,  # noqa: E402
                               simulate)

GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]


def sweep(ticker: str, sdf: pd.DataFrame, hold1: bool = False) -> pd.DataFrame:
    """hold1=True 면 익일 15:17 강제 마감 — 손절폭이 넓어질수록 보유가 길어져
    종목 추세를 먹는 교란을 제거하고 **장중 청산 타이밍 효과만** 분리한다."""
    minutes = load_minutes(ticker)
    idx = set(minutes)
    s = sdf[sdf["Ticker"] == ticker].reset_index(drop=True)
    eps = []
    for i in range(len(s) - 1):
        ek = s["Date"].iloc[i].strftime("%Y%m%d")
        if ek not in idx:
            continue
        fwd = [d.strftime("%Y%m%d") for d in s["Date"].iloc[i + 1: i + 1 + MAX_HOLD_TD]]
        if not fwd or fwd[0] not in idx:
            continue
        eb = minutes[ek]
        px = float(eb[max(eb)]["p"])
        adj = float(s["Close"].iloc[i])
        if adj > 0 and abs(px / adj - 1) > 0.20:
            continue
        eps.append((ek, px, fwd[:1] if hold1 else fwd, float(s["scale"].iloc[i])))

    rows = []
    for w in GRID:
        rs = [simulate(px, fwd, minutes, w / 100.0) for _, px, fwd, _ in eps]
        rs = [r for r in rs if r]
        v = pd.Series([r["ret_net_pct"] for r in rs])
        rows.append(dict(손절폭=w, n=len(v), 평균=v.mean(), 중앙=v.median(),
                         승률=(v > 0).mean() * 100, 표준편차=v.std(ddof=1),
                         최악=v.min(), 누적=(1 + v / 100).prod() ** (1 / max(len(v), 1)) * 100 - 100,
                         보유일=np.mean([r["hold_td"] for r in rs])))
    out = pd.DataFrame(rows)
    out.insert(0, "ticker", ticker)
    out.attrs["scale_med"] = float(s["scale"].median())
    out.attrs["n_eps"] = len(eps)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", required=True)
    ap.add_argument("--hold1", action="store_true", help="익일 15:17 강제 마감 (추세 교란 제거)")
    a = ap.parse_args()
    tickers = [t.strip() for t in a.tickers.split(",") if t.strip()]
    sdf = load_scale(tickers)
    if a.hold1:
        print("※ 보유 1거래일 고정 (익일 15:17 강제 마감) — 청산 타이밍 효과만 분리")

    best = {}
    for tk in tickers:
        r = sweep(tk, sdf, hold1=a.hold1)
        sm = r.attrs["scale_med"]
        print(f"\n{'='*76}\n■ {tk}  배율 중앙 {sm:.2f}  에피소드 {r.attrs['n_eps']}")
        print(r.drop(columns=['ticker']).round(3).to_string(index=False))
        b = r.loc[r["평균"].idxmax()]
        best[tk] = (float(b["손절폭"]), sm)
        print(f"  → 평균수익 최적 손절폭 {b['손절폭']}%  (배율 중앙 {sm:.2f})")

    if len(best) == 2:
        (t1, (w1, s1)), (t2, (w2, s2)) = best.items()
        print(f"\n{'='*76}\n■ 비례성 검정")
        print(f"  최적 손절폭 비  {t1}/{t2} = {w1}/{w2} = {w1/w2:.2f}")
        print(f"  배율 비        {t1}/{t2} = {s1:.2f}/{s2:.2f} = {s1/s2:.2f}")
        print("  두 비가 비슷하면 → 손절폭이 변동성에 비례한다 = 선형 배율 타당")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
