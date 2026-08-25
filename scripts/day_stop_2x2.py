#!/usr/bin/env python3
"""
day_stop_2x2.py — 변동성 × 추세 2×2 로 교락을 깬 손절폭 분석

문제: 단일 종목의 "최적 손절폭"은 그 종목의 **추세**가 결정해 버린다.
      (상승주는 손절이 뭘 해도 손해 → 넓을수록 유리 / 하락주는 반대)
      2종목만으로는 변동성 효과와 추세 효과가 완전히 교락된다.

해법: 같은 **추세군 안에서** 고변동 vs 저변동을 비교한다.
      두 추세군 **모두에서** 고변동 종목이 더 넓은 손절폭을 원하면,
      그건 추세가 아니라 변동성이 만든 차이다.

셀 (2025-08~2026-08):
    고변동×상승 036930 주성엔지니어링 | 저변동×상승 078930 GS
    고변동×하락 087010 펩트론         | 저변동×하락 003490 대한항공

⚠ 셀당 1종목 — 방향성 근거일 뿐 계수의 정밀 추정치가 아니다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from day_stop_backtest import MAX_HOLD_TD, load_minutes, load_scale, simulate  # noqa: E402

CELLS = {
    ("상승", "고변동"): ("036930", "주성엔지니어링"),
    ("상승", "저변동"): ("078930", "GS"),
    ("하락", "고변동"): ("087010", "펩트론"),
    ("하락", "저변동"): ("003490", "대한항공"),
}
GRID = [0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0]
NOSTOP = 99.0          # 사실상 무손절 (익일 15:17 강제 마감)


def episodes(ticker: str, sdf: pd.DataFrame):
    m = load_minutes(ticker)
    idx = set(m)
    s = sdf[sdf["Ticker"] == ticker].reset_index(drop=True)
    out = []
    for i in range(len(s) - 1):
        ek = s["Date"].iloc[i].strftime("%Y%m%d")
        if ek not in idx:
            continue
        fwd = [d.strftime("%Y%m%d") for d in s["Date"].iloc[i + 1: i + 2]]   # 1거래일 고정
        if not fwd or fwd[0] not in idx:
            continue
        eb = m[ek]
        px = float(eb[max(eb)]["p"])
        adj = float(s["Close"].iloc[i])
        if adj > 0 and abs(px / adj - 1) > 0.20:
            continue
        out.append((px, fwd))
    return m, out, s


def main() -> int:
    tks = [v[0] for v in CELLS.values()]
    sdf = load_scale(tks)
    curves, meta = {}, {}

    for (trend, vol), (tk, nm) in CELLS.items():
        m, eps, s = episodes(tk, sdf)
        if not eps:
            print(f"[{tk} {nm}] 에피소드 없음 — 분봉 캐시 확인")
            continue
        ns = pd.Series([r["ret_net_pct"] for r in
                        (simulate(px, fwd, m, NOSTOP / 100) for px, fwd in eps) if r])
        row = {}
        for w in GRID:
            v = pd.Series([r["ret_net_pct"] for r in
                           (simulate(px, fwd, m, w / 100) for px, fwd in eps) if r])
            row[w] = {"평균": v.mean(), "대비무손절": v.mean() - ns.mean(),
                      "승률": (v > 0).mean() * 100, "최악": v.min()}
        curves[(trend, vol)] = row
        meta[(trend, vol)] = {"ticker": tk, "name": nm, "n": len(eps),
                              "배율": float(s["scale"].median()),
                              "YZ20%": float(s["YZ_20"].median() * 100),
                              "무손절": ns.mean(),
                              "기간수익%": (eps[-1][0] / eps[0][0] - 1) * 100}

    print("■ 셀 요약")
    mt = pd.DataFrame(meta).T.round(3)
    print(mt.to_string())

    print("\n■ 손절폭별 '무손절 대비' 성과 (%p) — 추세 드리프트를 뺀 순수 손절효과")
    tbl = pd.DataFrame({f"{t}·{v}": {w: round(c[w]["대비무손절"], 3) for w in GRID}
                        for (t, v), c in curves.items()})
    tbl.index.name = "손절폭%"
    print(tbl.to_string())

    print("\n■ 셀별 최적 손절폭 (무손절 대비 최대)")
    best = {}
    for k, c in curves.items():
        w = max(GRID, key=lambda x: c[x]["대비무손절"])
        best[k] = w
        print(f"  {k[0]}·{k[1]:4s} {meta[k]['name']:8s} 배율 {meta[k]['배율']:.2f} → 최적 {w}%")

    print("\n■ 교락 차단 검정 — 같은 추세군 안에서 고변동/저변동 최적폭 비")
    ok = []
    for t in ("상승", "하락"):
        if (t, "고변동") in best and (t, "저변동") in best:
            r = best[(t, "고변동")] / best[(t, "저변동")]
            sr = meta[(t, "고변동")]["배율"] / meta[(t, "저변동")]["배율"]
            ok.append(r > 1)
            print(f"  {t}군: 최적폭 비 {best[(t,'고변동')]}/{best[(t,'저변동')]} = {r:.2f}"
                  f"   배율 비 {sr:.2f}   {'✓ 고변동이 더 넓음' if r > 1 else '✗ 역전'}")
    if len(ok) == 2:
        print("\n  → 두 추세군 모두 ✓ 이면 변동성 효과가 추세와 독립으로 확인된 것" if all(ok)
              else "\n  → 한쪽이라도 ✗ 이면 변동성만으로는 설명 안 됨 (배율 도입 근거 약함)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
