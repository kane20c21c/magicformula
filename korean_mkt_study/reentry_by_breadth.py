# -*- coding: utf-8 -*-
"""reentry_by_breadth.py — 재진입 백테스트(panel) 결과를 시장 브레드스 구간별로 나눈다 (케인 2026-11-05).

입력  out/reentry/panel/events.csv  (reentry_study.py panel)
      out/breadth/breadth_market.csv (breadth_study.py — 동전주 제외 시장 전체 EMA10)
구간  이벤트일(손절 판정일) 종가 시점의 EMA10. 그날 장 마감 후 알 수 있는 값이라 미래 참조 없음.
비교  60거래일 끝 기준: 버팀 vs 손절만 vs 손절→기본 재매수. '차이' = 같은 이벤트에서 (손절→재매수 − 버팀).
사용: python3 reentry_by_breadth.py [dist]
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ev = pd.read_csv(HERE / "out/reentry/panel/events.csv", parse_dates=["event_date", "base_day", "A_day", "B_day"])
br = pd.read_csv(HERE / "out/breadth/breadth_market.csv", index_col=0, parse_dates=True)
ev["ema10"] = br.ema10.reindex(ev.event_date).to_numpy()
ev["ratio"] = br.ratio.reindex(ev.event_date).to_numpy()
assert ev.ema10.notna().all(), "브레드스 없는 날짜"

if len(sys.argv) > 1 and sys.argv[1] == "dist":
    q = ev.ema10.quantile([0, .1, .2, .25, .33, .5, .67, .75, .8, .9, 1]).round(3)
    print("이벤트일 EMA10 분위:", q.to_dict())
    allq = br.loc["2023-01-02":, "ema10"].quantile([.1, .25, .5, .75, .9]).round(3)
    print("같은 기간 전체 거래일 EMA10 분위:", allq.to_dict())
    print("이벤트일 EMA10<0.40 비율", round((ev.ema10 < .40).mean(), 3), "/ 전체 거래일", round((br.loc['2023':, 'ema10'] < .40).mean(), 3))
    sys.exit()

# 이벤트가 약한 브레드스에 몰려 있어(0.55 이상 희소) 아래쪽을 촘촘히 — 11-05 분포 확인 후 결정
BINS = [0, 0.35, 0.40, 0.45, 0.50, 1.0]
LABELS = ["<0.35", "0.35~0.40", "0.40~0.45", "0.45~0.50", "≥0.50"]
ev["bin"] = pd.cut(ev.ema10, BINS, labels=LABELS, right=False)
e = ev[ev.c_d60.notna()].copy()
e["hold"] = e.c_d60 / e.event_close - 1
e["stop_only"] = e.stop_fill / e.event_close - 1
e["reb"] = np.where(e.base_px.notna(), e.stop_fill / e.base_px * e.c_d60 / e.event_close, e.stop_fill / e.event_close) - 1
e["diff"] = e.reb - e.hold
e["stop_diff"] = e.stop_only - e.hold

def tbl(g):
    return pd.Series({"n": len(g), "버팀_중앙": g.hold.median(), "버팀_플러스": (g.hold > 0).mean(),
                      "손절만_중앙": g.stop_only.median(), "손절→재매수_중앙": g.reb.median(),
                      "차이_중앙(재매수−버팀)": g["diff"].median(), "차이_평균": g["diff"].mean(),
                      "재매수가_나은_비율": (g["diff"] > 0).mean(), "손절만이_나은_비율": (g.stop_diff > 0).mean(),
                      "버팀_하위10%": g.hold.quantile(.1), "재매수_하위10%": g.reb.quantile(.1)})

pd.set_option("display.width", 240); pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
print(f"== 브레드스 구간별 경로 비교 (60거래일 끝, n={len(e)}) ==")
print(e.groupby("bin", observed=True).apply(tbl).to_string())
print("\n== 구간 × 연도 — 차이 중앙 (건수) ==")
pv = e.pivot_table(index="bin", columns=e.event_date.dt.year, values="diff", aggfunc="median", observed=True)
pn = e.pivot_table(index="bin", columns=e.event_date.dt.year, values="diff", aggfunc="size", observed=True)
print(pv.round(3).astype(str).add(" (").add(pn.astype("Int64").astype(str)).add(")").to_string())

# 20일 단기 지표 (기본 재매수 신호의 질)
s = ev[ev.base_day.notna()].copy()
s["base_ema"] = br.ema10.reindex(s.base_day).to_numpy()
s["bbin"] = pd.cut(s.base_ema, BINS, labels=LABELS, right=False)
print("\n== 기본 재매수 신호일의 EMA10 구간별 20일 성과 ==")
print(s.groupby("bbin", observed=True).agg(n=("base_r20", "count"), r20_중앙=("base_r20", "median"),
      승률20=("base_r20", lambda v: (v.dropna() > 0).mean()), mae20_중앙=("base_mae20", "median")).to_string())
e.to_csv(HERE / "out/reentry/panel/events_breadth.csv", index=False)
