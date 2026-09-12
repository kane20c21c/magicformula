"""r2_yz_s1.py — S1: 세 지표의 관계. MA200/HW40 onset 이벤트를 R²×YZ 셀로 갈라 H일 후 성과를 본다.

입력: out/r2yz/events_200_40.parquet, control_200.parquet (r2_yz_events.py)
출력: 표준출력 표 + out/r2yz/s1_tables.xlsx
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "r2yz"
MA, HW = (int(sys.argv[1]) if len(sys.argv) > 1 else 200), (int(sys.argv[2]) if len(sys.argv) > 2 else 40)
R2W = sys.argv[3] if len(sys.argv) > 3 else "R2_60"
SPLIT = pd.Timestamp("2022-01-01")
HS = (10, 20, 40)
pd.set_option("display.width", 200, "display.max_columns", 40, "display.float_format", "{:.3f}".format)


def tier(cap):
    return pd.cut(cap, [0, 3e11, 1e12, 4e12, np.inf], labels=["<3천억", "3천억~1조", "1조~4조", "4조↑"])


def metrics(g: pd.DataFrame) -> pd.Series:
    out = {"n": len(g)}
    for H in HS:
        r = g[f"RET_{H}"].dropna()
        out[f"win{H}"] = (r > 0).mean() if len(r) else np.nan
        out[f"med{H}"] = r.median() if len(r) else np.nan
        out[f"mean{H}"] = r.mean() if len(r) else np.nan
        out[f"mae{H}"] = g[f"MAE_{H}"].median()
    return pd.Series(out)


def cell_table(df: pd.DataFrame, a: str, b: str, qa=3, qb=3, labels_a=None, labels_b=None) -> pd.DataFrame:
    d = df.dropna(subset=[a, b]).copy()
    d["A"] = pd.qcut(d[a], qa, labels=labels_a or [f"{a}:L", f"{a}:M", f"{a}:H"][:qa])
    d["B"] = pd.qcut(d[b], qb, labels=labels_b or [f"{b}:L", f"{b}:M", f"{b}:H"][:qb])
    t = d.groupby(["A", "B"], observed=True).apply(metrics)
    return t


def main():
    ev = pd.read_parquet(OUT / f"events_{MA}_{HW}.parquet")
    ct = pd.read_parquet(OUT / f"control_{MA}.parquet")
    for df in (ev, ct):
        for H in HS:
            df.loc[~np.isfinite(df[f"RET_{H}"]), f"RET_{H}"] = np.nan
    ev["TIER"] = tier(ev.MKTCAP)
    ev["PERIOD"] = np.where(ev.date < SPLIT, "A:2014-21", "B:2022-26")
    ev["SLOPE_SIGN"] = np.where(ev[R2W.replace("R2", "SLOPE")] > 0, "slope+", "slope-")
    sheets = {}

    print(f"=== MA{MA}/HW{HW} onset events: {len(ev):,} | R² 창 {R2W} ===")
    base = pd.DataFrame({"events(all)": metrics(ev),
                         "ctrl_all": metrics(ct[ct.KIND == "ctrl_all"]),
                         "ctrl_up(추세중·비눌림)": metrics(ct[ct.KIND == "ctrl_up"]),
                         "events(4조/30% 편입)": metrics(ev[ev.ELIG == 1])}).T
    print("\n[0] 기준선 — 이벤트 vs 대조군"); print(base); sheets["0_baseline"] = base

    t = ev.groupby("TIER", observed=True).apply(metrics)
    print("\n[1] 시총 티어별"); print(t); sheets["1_tier"] = t

    t = ev.groupby("SLOPE_SIGN").apply(metrics)
    print("\n[2] 기울기 부호별"); print(t); sheets["2_slope"] = t

    q = ev.dropna(subset=[R2W]).copy(); q["R2q"] = pd.qcut(q[R2W], 5, labels=[f"Q{i}" for i in range(1, 6)])
    t = q.groupby("R2q", observed=True).apply(metrics); t["R2_lo"] = q.groupby("R2q", observed=True)[R2W].min()
    print(f"\n[3] {R2W} 5분위 단독"); print(t); sheets["3_r2_quintile"] = t

    for col, name in (("YZR", "YZ20/YZ60 비율"), ("YZP", "YZ20 자기1년 백분위"), ("YZ_20", "YZ_20 절대수준")):
        q = ev.dropna(subset=[col]).copy(); q["q"] = pd.qcut(q[col], 5, labels=[f"Q{i}" for i in range(1, 6)])
        t = q.groupby("q", observed=True).apply(metrics); t[f"{col}_lo"] = q.groupby("q", observed=True)[col].min()
        print(f"\n[4] {name} 5분위 단독"); print(t); sheets[f"4_{col}_quintile"] = t

    for col, name in (("YZR", "YZR"), ("YZP", "YZP")):
        t = cell_table(ev, R2W, col)
        print(f"\n[5] 9셀 {R2W} 3분위 × {name} 3분위 (전 기간)"); print(t); sheets[f"5_cell_{col}"] = t
        for p, g in ev.groupby("PERIOD"):
            tp = cell_table(g, R2W, col)
            print(f"\n[5-{p}] 9셀 {R2W} × {name} — {p}"); print(tp[["n", "win20", "med20", "win40", "med40"]])
            sheets[f"5_cell_{col}_{p[:1]}"] = tp

    # slope+ 만 (R² 는 방향 중립이므로 상승 일관성으로 한정한 버전)
    t = cell_table(ev[ev.SLOPE_SIGN == "slope+"], R2W, "YZR")
    print(f"\n[6] 9셀 (기울기+ 만) {R2W} × YZR"); print(t); sheets["6_cell_slopeplus"] = t

    # 상관 (스피어만) — 세 지표끼리, 그리고 수익과
    cols = [R2W, "YZ_20", "YZR", "YZP", "DEPTH", "MARGIN", "RET_10", "RET_20", "RET_40"]
    c = ev[cols].corr(method="spearman")
    print("\n[7] 스피어만 상관"); print(c); sheets["7_spearman"] = c

    with pd.ExcelWriter(OUT / f"s1_tables_{MA}_{HW}_{R2W}.xlsx") as xw:
        for k, v in sheets.items():
            v.to_excel(xw, sheet_name=k[:31])
    print("\nsaved", OUT / f"s1_tables_{MA}_{HW}_{R2W}.xlsx")


if __name__ == "__main__":
    main()
