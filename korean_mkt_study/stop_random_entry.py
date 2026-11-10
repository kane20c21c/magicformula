# -*- coding: utf-8 -*-
"""stop_random_entry.py — 메가갤럭시에서 진입신호 없이 무작위 진입(매일 1종목 추첨 × 매수확률 20%) 버팀+T60 vs 스윙 신호 버팀+T60 (케인 2026-11-05).
사용: STOP_UNIV=swing python3 stop_random_entry.py [회수]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import stop_research as sr, stop_port as sp
from stop_hold_grid import SEGS

def random_signals(p, rng, prob=0.20):
    el = p.elig.loc[sr.START:sr.END].to_numpy(); out = np.zeros_like(el)
    for i in range(len(el)):
        js = np.flatnonzero(el[i])
        if js.size and rng.random() < prob:
            out[i, rng.integers(js.size)] = True
    return pd.DataFrame(out, index=p.elig.loc[sr.START:sr.END].index, columns=p.elig.columns)

def main(n=50, seed=20261105):
    p, s, _, _, atr = sr.load()
    res = {"스윙 신호 버팀 T60": [], "무작위 진입 버팀 T60": []}; seg = {k: [] for k in res}
    for it in range(n):
        rng = np.random.default_rng(seed + it)
        eq, tr = sp.run_port(p, s, atr, None, 60, slot_mode="prop", rng=rng); res["스윙 신호 버팀 T60"].append(sp.metrics(eq, tr)); seg["스윙 신호 버팀 T60"].append(sp.period_returns(eq, SEGS))
        rs = random_signals(p, np.random.default_rng(seed * 7 + it))
        eq, tr = sp.run_port(p, rs, atr, None, 60, slot_mode="prop", rng=rng); res["무작위 진입 버팀 T60"].append(sp.metrics(eq, tr)); seg["무작위 진입 버팀 T60"].append(sp.period_returns(eq, SEGS))
    pd.set_option("display.width", 220)
    rows = []
    for k, v in res.items():
        d = pd.DataFrame(v)
        rows.append({"변형": k, "CAGR중앙%": d.cagr.median() * 100, "10~90%": f"{d.cagr.quantile(.1)*100:.1f}~{d.cagr.quantile(.9)*100:.1f}", "Sharpe": d.sharpe.median(),
                     "MDD중앙%": d.mdd.median() * 100, "MDD최악%": d.mdd.min() * 100, "투자비중%": d.invested.median() * 100, "청산수": d.n_trade.median(), "승률%": d.win.median() * 100, "거래당평균%": d.avg_ret.median() * 100 if "avg_ret" in d else np.nan})
    print(pd.DataFrame(rows).set_index("변형").round(2).to_string())
    a = np.array([m["cagr"] for m in res["스윙 신호 버팀 T60"]]); b = np.array([m["cagr"] for m in res["무작위 진입 버팀 T60"]])
    d = (a - b) * 100
    print(f"\n스윙 − 무작위 CAGR 차이: 중앙 {np.median(d):+.2f}%p, 10~90% {np.quantile(d,.1):+.2f}~{np.quantile(d,.9):+.2f}, 스윙이 이긴 비율 {np.mean(d>0)*100:.0f}%")
    pr = pd.DataFrame({k: pd.DataFrame(v).median() for k, v in seg.items()}) * 100
    pr["차이"] = pr.iloc[:, 0] - pr.iloc[:, 1]
    print("\n== KOSPI200 구간별 수익률 중앙값 (%) ==\n", pr.round(1).to_string())

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
