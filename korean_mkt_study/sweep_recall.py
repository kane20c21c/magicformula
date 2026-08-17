"""sweep_recall.py — 축 R: 재현율 개선 스윕 (Kane 지시 2026-08-17).

진단(diag_recall.py): 눌림 후 반등 기회 2,605건 중 20.7%만 포착.
  놓친 이유 — MA120 아래 60.3% / onset 제한 12.6% / 슬롯·현금 6.4%

R1 추세 필터 완화  : 제거 / MA60 / MA120 허용오차 / MA120 위 여력 요구(대조군)
R2 onset 완화      : 같은 눌림 구간 내 N거래일 간격 재신호
R3 결합            : R1 × R2 × 후보우선순위(여력 큰 순)

청산 v1.2.0 고정 · 유니버스 4조/25% · 손절 청산가격×99.5% · 왕복 0.23%.
실행: python3 sweep_recall.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from backtest import (EntryParams, ExitParams, PortfolioParams, UniverseParams,
                      VolScaleParams, load_panel, metrics, run_backtest)
from sweep_entry import evaluate, show, OUT

NO_TREND = -1e9        # 추세 필터 사실상 해제

if __name__ == "__main__":
    print("패널 로드 중...")
    panel = load_panel(UniverseParams(), VolScaleParams())
    print(f"  종목 {len(panel.tickers)} · 거래일 {len(panel.dates)}\n")

    base = evaluate(panel, EntryParams(), "기준선 v1.0.0")

    print("── R1: 추세 필터 완화 (onset 유지) ──")
    r1 = [base]
    r1.append(evaluate(panel, EntryParams(trend_margin=NO_TREND), "R1a 추세필터 제거"))
    r1.append(evaluate(panel, EntryParams(trend_ma=60, trend_min_periods=40), "R1b MA60"))
    r1.append(evaluate(panel, EntryParams(trend_ma=200, trend_min_periods=140), "R1c MA200"))
    for tol in (0.05, 0.10, 0.20):
        r1.append(evaluate(panel, EntryParams(trend_margin=-tol), f"R1d MA120 −{tol:.0%} 허용"))
    show(r1)

    print("\n── R2: onset 완화 (추세 필터 유지) ──")
    r2 = [base]
    for g in (3, 5, 10, 20):
        r2.append(evaluate(panel, EntryParams(resignal_gap=g), f"R2 재신호 {g}일 간격"))
    show(r2)

    print("\n── R3: 결합 ──")
    r3 = [base]
    for tol, gap in [(0.05, 0), (0.10, 0), (0.05, 5), (0.10, 5), (0.10, 10),
                     (0.20, 5)]:
        lab = f"R3 MA120−{tol:.0%}"+(f"+재신호{gap}일" if gap else "")
        r3.append(evaluate(panel, EntryParams(trend_margin=-tol, resignal_gap=gap), lab))
    # 추세필터 완전 제거 + 재신호
    for gap in (0, 5, 10):
        lab = "R3 추세제거" + (f"+재신호{gap}일" if gap else "")
        r3.append(evaluate(panel, EntryParams(trend_margin=NO_TREND, resignal_gap=gap), lab))
    show(r3)

    print("\n── R4: 결합 + 후보우선순위 '여력 큰 순' (축 C) ──")
    r4 = [base]
    for tol, gap in [(0.05, 5), (0.10, 5), (0.10, 10)]:
        for pr in ("depth", "trend"):
            r4.append(evaluate(panel, EntryParams(trend_margin=-tol, resignal_gap=gap,
                                                  priority=pr),
                               f"R4 MA120−{tol:.0%}+재신호{gap}일+{pr}"))
    show(r4)

    allrows = r1 + r2[1:] + r3[1:] + r4[1:]
    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                  for r in allrows]).to_csv(OUT / "entry_sweep_R.csv", index=False)
    print(f"\n저장: {OUT/'entry_sweep_R.csv'}")
