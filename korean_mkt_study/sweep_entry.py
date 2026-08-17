"""sweep_entry.py — 진입 신호 개선 스윕 (축 A: 하락 멈춤 확인 / 축 B: 변동성 정규화 눌림).

청산은 v1.2.0 고정. 유니버스 4조/25% · 손절 체결 청산가격×99.5% (Kane 확정 2026-08-17).
결과는 out/entry_sweep_*.csv 로 저장.

실행: python3 sweep_entry.py A     (축 A만)
      python3 sweep_entry.py B     (축 B만)
      python3 sweep_entry.py AB    (결합)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from backtest import (EntryParams, ExitParams, PortfolioParams, UniverseParams,
                      VolScaleParams, load_panel, metrics, run_backtest)

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

REGIMES = {
    "2015~2019 평시": ("2015-01-01", "2019-12-31"),
    "2020~2021 코로나": ("2020-01-01", "2021-12-31"),
    "2022~2023 하락": ("2022-01-01", "2023-12-31"),
    "2024~2026 멜트업": ("2024-01-01", "2026-06-30"),
}


def evaluate(panel, ep, label, slip=0.005, regimes=True) -> dict:
    pp = PortfolioParams(stop_slippage=slip)
    res = run_backtest(panel, ep, ExitParams(), pp)
    m = metrics(res)
    m["label"] = label
    m["slip"] = slip
    if regimes:
        for name, (s, e) in REGIMES.items():
            r = run_backtest(panel, ep, ExitParams(), pp, start=s, end=e)
            mm = metrics(r)
            m[f"{name}_수익"] = mm["profit"]
            m[f"{name}_MDD"] = mm["mdd"]
    m["_res"] = res
    return m


def show(rows, cols=None):
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    base = ["label", "final", "cagr", "sharpe", "mdd", "invested", "n_buy", "n_sell",
            "win_rate", "avg_hold", "avg_ret"]
    df = df[[c for c in base if c in df] + [c for c in df.columns if c not in base]]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        d = df.copy()
        d["final"] = (d["final"] / 1e8).round(2)
        for c in ("cagr", "mdd", "win_rate", "avg_ret", "invested"):
            if c in d:
                d[c] = (d[c] * 100).round(1)
        d["sharpe"] = d["sharpe"].round(2)
        d["avg_hold"] = d["avg_hold"].round(1)
        for c in d.columns:
            if c.endswith("_수익"):
                d[c] = (d[c] / 1e4).round(0)
            if c.endswith("_MDD"):
                d[c] = (d[c] * 100).round(1)
        print(d.to_string(index=False))
    return df


if __name__ == "__main__":
    which = (sys.argv[1] if len(sys.argv) > 1 else "AB").upper()
    print("패널 로드 중...")
    panel = load_panel(UniverseParams(), VolScaleParams())
    print(f"  종목 {len(panel.tickers)} · 거래일 {len(panel.dates)}\n")

    rows = [evaluate(panel, EntryParams(), "기준선 v1.0.0")]

    if "A" in which:
        print("── 축 A: 하락 멈춤 확인 ──")
        for c in ("up1", "up2", "mid", "up1_mid", "ma5"):
            for w in (3, 5, 10, 20):
                rows.append(evaluate(panel, EntryParams(confirm=c, confirm_max_wait=w),
                                     f"A:{c}/wait{w}"))
        show(rows)
        pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                      for r in rows]).to_csv(OUT / "entry_sweep_A.csv", index=False)

    if "B" in which:
        print("\n── 축 B: 변동성 정규화 눌림 ──")
        rowsB = [rows[0]]
        for k in (0.08, 0.10, 0.12):
            for clip in ((0.05, 0.20), (0.06, 0.18), (0.07, 0.15), (0.05, 0.30)):
                rowsB.append(evaluate(panel,
                                      EntryParams(pullback_pct=k, vol_depth=True,
                                                  vol_depth_clip=clip),
                                      f"B:k{k:.2f}/clip{clip[0]:.2f}-{clip[1]:.2f}"))
        # 참고: 고정 임계 스윕 (정규화 효과와 분리하기 위해)
        for k in (0.06, 0.08, 0.10, 0.12, 0.15):
            rowsB.append(evaluate(panel, EntryParams(pullback_pct=k), f"고정 {k:.0%}"))
        show(rowsB)
        pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                      for r in rowsB]).to_csv(OUT / "entry_sweep_B.csv", index=False)

    if "C" in which:
        print("\n── 축 C: 추세 여력 필터 + 후보 우선순위 ──")
        rowsC = [rows[0]]
        for m in (0.03, 0.05, 0.10, 0.15, 0.20):
            rowsC.append(evaluate(panel, EntryParams(trend_margin=m), f"C:여력 ≥{m:.0%}"))
        for pr in ("depth_asc", "trend", "vol"):
            rowsC.append(evaluate(panel, EntryParams(priority=pr), f"C:우선순위 {pr}"))
        for m in (0.05, 0.10, 0.15):
            for pr in ("depth", "trend"):
                rowsC.append(evaluate(panel, EntryParams(trend_margin=m, priority=pr),
                                      f"C:여력≥{m:.0%}+{pr}"))
        show(rowsC)
        pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                      for r in rowsC]).to_csv(OUT / "entry_sweep_C.csv", index=False)
