# -*- coding: utf-8 -*-
"""breadth_panel.py — 유니버스·메가갤럭시 브레드스를 2015년부터 (케인 2026-11-06 08:47 "브레드스를 유니버스로 돌려보자").

유니버스    LLV 패널 217종목(ETF 제외), 가격 data_1104 (2015~) — 종목 명단이 2026년 기준이라 과거엔 '살아남아 커진 종목' 쏠림 있음
메가갤럭시  월별 편입(시총 4조↑·외인 30%↑) — 하루 30~40종목이라 비율이 거칠다 (참고용)
상승/하락  전일 종가 대비 >0 / <0, 보합은 분모에서 제외, |수익률|>35% 는 자료 오류로 제외. 전일 편입 종목만
저장       out/breadth/breadth_{universe,mega}_2015.csv (date, adv, dec, n, ratio, ema10, ew_ret)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import backtest as bt, stop_research as sr

def breadth(close, elig):
    r = close.pct_change(); r = r.where(r.abs() <= 0.35)
    mem = elig.shift(1).fillna(False).astype(bool) & r.notna(); rr = r.where(mem)
    adv = (rr > 0).sum(axis=1); dec = (rr < 0).sum(axis=1)
    d = pd.DataFrame({"adv": adv, "dec": dec, "n": mem.sum(axis=1), "ew_ret": rr.mean(axis=1)})
    d["ratio"] = d.adv / (d.adv + d.dec).replace(0, np.nan); d["ema10"] = d.ratio.ewm(span=10, adjust=False).mean()
    return d.loc["2015-01-02":]

def main():
    out = HERE / "out" / "breadth"; out.mkdir(parents=True, exist_ok=True)
    pdy = sr.load_day_panel(sr.END); d = breadth(pdy.close, pdy.close.notna()); d.to_csv(out / "breadth_universe_2015.csv")
    ps = bt.load_panel(bt.UniverseParams(), bt.VolScaleParams(), end=sr.END); m = breadth(ps.close, ps.elig.fillna(False)); m.to_csv(out / "breadth_mega_2015.csv")
    for nm, x in (("유니버스", d), ("메가갤럭시", m)):
        print(f"{nm}: {x.index[0].date()}~{x.index[-1].date()} · 하루 종목 수 {int(x.n.min())}~{int(x.n.max())} (중앙 {int(x.n.median())}) · ratio 연도별 평균",
              x.ratio.groupby(x.index.year).mean().round(3).to_dict())

if __name__ == "__main__":
    main()
