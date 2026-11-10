# -*- coding: utf-8 -*-
"""entry_priority_split.py — 1단계 ② 우선순위의 기간 분할 검증 (케인 2026-11-05).
기간마다 1억에서 새로 시작: 2015–2019 / 2020–2022 / 2023–2026-05(보류 구간 전). 무작위 MC 대비 백분위.
사용  STOP_UNIV=swing|day python3 entry_priority_split.py [mc회수=30]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import stop_research as sr, stop_port as sp
from entry_priority import prios, OUT

PER = [("2015-01-01", "2019-12-31", "15–19"), ("2020-01-01", "2022-12-31", "20–22"), ("2023-01-01", "2026-05-29", "23–26.5")]

def main(n_mc=30, seed=20261105):
    p, s, _, _, atr = sr.load(); P = prios(p); rows = []
    for T in (40, 60):
        for mode in ("fixed", "prop"):
            for a, b, lbl in PER:
                kw = dict(T=T, start=a, end=b, slot_mode=mode)
                rnd = np.array([sp.metrics(*sp.run_port(p, s, atr, None, rng=np.random.default_rng(seed + i), **kw))["cagr"] for i in range(n_mc)]) * 100
                for nm, pr in P.items():
                    c = sp.metrics(*sp.run_port(p, s, atr, None, prio=pr, **kw))["cagr"] * 100
                    rows.append(dict(T=f"T{T}", 슬롯="1,000만×10" if mode == "fixed" else "자산÷10", 기간=lbl, 우선순위=nm, CAGR=c, 백분위=(rnd < c).mean() * 100))
                rows.append(dict(T=f"T{T}", 슬롯="1,000만×10" if mode == "fixed" else "자산÷10", 기간=lbl, 우선순위="무작위 중앙", CAGR=np.median(rnd), 백분위=50.0))
            print(f"  T{T} {mode} done", flush=True)
    d = pd.DataFrame(rows)
    t = d.pivot_table(index=["T", "슬롯", "우선순위"], columns="기간", values=["CAGR", "백분위"], sort=False)
    pd.set_option("display.width", 260); print(t.round(1).to_string())
    OUT.mkdir(parents=True, exist_ok=True); d.to_csv(OUT / "p2_priority_split.csv", index=False)

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
