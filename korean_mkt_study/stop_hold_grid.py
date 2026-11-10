# -*- coding: utf-8 -*-
"""stop_hold_grid.py — 버팀 vs 고정−30% × 시간청산 T20/40/60/90/120, KOSPI200 고점-저점 구간별 (케인 2026-11-05).
운영 = stop_port.run_port (슬롯 비례, 후보 무작위 n회 중앙값). 사용: STOP_UNIV=swing|day python3 stop_hold_grid.py [회수]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import stop_research as sr, stop_port as sp

# KOSPI200 20% 반전 전환점 (stop_segments.py 2026-11-05 실행 결과)
PTS = [("2015-01-02", "L"), ("2017-11-03", "H"), ("2019-08-07", "L"), ("2020-01-22", "H"), ("2020-03-19", "L"), ("2021-06-25", "H"),
       ("2022-09-30", "L"), ("2024-07-11", "H"), ("2025-04-09", "L"), ("2026-02-26", "H"), ("2026-03-31", "L"), ("2026-06-22", "H"),
       ("2026-07-30", "L"), ("2026-09-23", "H"), ("2026-10-26", "L"), ("2026-11-04", "-")]
k = pd.read_parquet(HERE / "data_1104" / "kospi200_daily.parquet").Close.astype(float); k.index = pd.to_datetime(k.index)
SEGS = []
for (a, ta), (b, tb) in zip(PTS[:-1], PTS[1:]):
    ka, kb = k.loc[:a].iloc[-1], k.loc[:b].iloc[-1]
    SEGS.append((a, b, f"{a[2:7]}~{b[2:7]} {'상승' if ta=='L' else '하락'} {(kb/ka-1)*100:+.0f}%"))

def main(n_iter=30, seed=20261105):
    p, s, _, _, atr = sr.load()
    Ts = (20, 40, 60, 90, 120)
    runs = [(f"버팀 T{T}", None, T) for T in Ts] + [(f"고정−30% T{T}", sr.mk_fixed(.30), T) for T in Ts]
    res = {l: [] for l, *_ in runs}; seg = {l: [] for l, *_ in runs}
    for it in range(n_iter):
        for lbl, f, T in runs:
            rng = np.random.default_rng(seed + it)
            eq, tr = sp.run_port(p, s, atr, f, T, slot_mode="prop", rng=rng)
            res[lbl].append(sp.metrics(eq, tr)); seg[lbl].append(sp.period_returns(eq, SEGS))
    pd.set_option("display.width", 260)
    rows = []
    for lbl in res:
        d = pd.DataFrame(res[lbl])
        rows.append({"변형": lbl, "CAGR중앙%": d.cagr.median() * 100, "10~90%": f"{d.cagr.quantile(.1)*100:.1f}~{d.cagr.quantile(.9)*100:.1f}",
                     "Sharpe": d.sharpe.median(), "MDD중앙%": d.mdd.median() * 100, "MDD최악%": d.mdd.min() * 100, "승률%": d.win.median() * 100, "청산수": d.n_trade.median()})
    t = pd.DataFrame(rows).set_index("변형"); print(f"[유니버스 {sr.UNIV}] {n_iter}회\n", t.round(2).to_string())
    base = {T: np.array([m["cagr"] for m in res[f"버팀 T{T}"]]) for T in Ts}
    print("\n== 같은 T 에서 고정−30% − 버팀 CAGR 차이 (%p): 중앙 / 10~90% / 이긴 비율 ==")
    for T in Ts:
        dlt = (np.array([m["cagr"] for m in res[f"고정−30% T{T}"]]) - base[T]) * 100
        print(f"  T{T:<4} {np.median(dlt):+5.2f}  {np.quantile(dlt,.1):+5.2f}~{np.quantile(dlt,.9):+5.2f}  {np.mean(dlt>0)*100:4.0f}%")
    pr = pd.DataFrame({l: pd.DataFrame(seg[l]).median() for l in seg}) * 100
    print("\n== KOSPI200 구간별 수익률 중앙값 (%) — 행: 구간, 열: 변형 ==")
    print(pr.round(1).to_string())
    out = HERE / "out" / "stop_research" / ("port_" + sr.UNIV); t.to_csv(out / "hold_grid.csv"); pr.to_csv(out / "hold_grid_segments.csv")

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
