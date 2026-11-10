# -*- coding: utf-8 -*-
"""stop_segments.py — 손절 vs 버팀 신호 단위 결과를 KOSPI200 고점-저점 구간별로 나눈다 (케인 2026-11-05).
구간 = KOSPI200 종가 zigzag(반전 20%) 전환점. 배정은 신호일 기준. 마지막 미확정 구간은 '진행중' 표기.
사용: python3 stop_segments.py [H=60] [data_1104]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
H = int(sys.argv[1]) if len(sys.argv) > 1 else 60
DD = sys.argv[2] if len(sys.argv) > 2 else "data_1104"
df = pd.read_csv(HERE / "out/stop_research" / DD / "signal_paths.csv", parse_dates=["date"])
k = pd.read_parquet(HERE / DD.replace("_day", "") / "kospi200_daily.parquet").Close.astype(float).sort_index()
k.index = pd.to_datetime(k.index); k = k.loc["2014-06-01":]

def zigzag(s, th):
    idx, v = s.index, s.to_numpy(); pts = []; mode = None; ext = 0
    for i in range(1, len(v)):
        if mode is None:
            if v[i] >= v[ext] * (1 + th): mode = "up"; pts.append((idx[0], "L")); ext = i
            elif v[i] <= v[ext] * (1 - th): mode = "down"; pts.append((idx[0], "H")); ext = i
            continue
        if mode == "up":
            if v[i] > v[ext]: ext = i
            elif v[i] <= v[ext] * (1 - th): pts.append((idx[ext], "H")); mode = "down"; ext = i
        else:
            if v[i] < v[ext]: ext = i
            elif v[i] >= v[ext] * (1 + th): pts.append((idx[ext], "L")); mode = "up"; ext = i
    pts.append((idx[ext], "H" if mode == "up" else "L", "(진행중)"))
    return pts

pts = zigzag(k, 0.20)
pts = [p for p in pts if p[0] >= pd.Timestamp("2015-01-01")] if pts[0][0] < pd.Timestamp("2015-01-01") else pts
if pts[0][0] > pd.Timestamp("2015-01-02"): pts = [(pd.Timestamp("2015-01-02"), "L")] + pts
labels, bounds = [], []
for a, b in zip(pts[:-1], pts[1:]):
    kind = "상승" if a[1] == "L" else "하락"; ka, kb = k.loc[:a[0]].iloc[-1], k.loc[:b[0]].iloc[-1]
    tag = " 진행중" if len(b) > 2 else ""
    labels.append(f"{a[0].strftime('%y-%m')}~{b[0].strftime('%y-%m')} {kind} {(kb/ka-1)*100:+.0f}%{tag}"); bounds.append(a[0])
bounds.append(k.index[-1] + pd.Timedelta(days=1))
print("전환점:", ", ".join(f"{d.date()} {t}" for d, t, *_ in pts))
df["seg"] = pd.cut(df.date, bounds, labels=labels, right=False)
cols = {"버팀": f"hold{H}", "현행③": f"현행③|{H}", "고정−5%": f"고정−5%|{H}", "고정−10%": f"고정−10%|{H}", "고정−20%": f"고정−20%|{H}", "ATR×3": f"ATR×3|{H}"}
d = df.dropna(subset=[f"hold{H}"]); g = d.groupby("seg", observed=True)
pd.set_option("display.width", 250)
def show(fn, title):
    t = pd.DataFrame({n: g[c].agg(fn) * 100 for n, c in cols.items()}); t.insert(0, "n", g.size())
    print(f"\n== {title} (보유 {H}거래일, %) =="); print(t.round(1).to_string())
show("mean", "평균"); show("median", "중앙")
for q, t in [(.05, "하위 5%"), (.10, "하위 10%"), (.20, "하위 20%"), (.80, "상위 20%"), (.90, "상위 10%"), (.95, "상위 5%")]:
    show(lambda v, q=q: v.quantile(q), t)
t = pd.DataFrame({n: g[c].apply(lambda v: (v > 0).mean() * 100) for n, c in cols.items()}); t.insert(0, "n", g.size())
print("\n== 플러스 비율 =="); print(t.round(0).to_string())
t = pd.DataFrame({n: g[f"{n}|{H}|stopped"].mean() * 100 for n in cols if n != "버팀"}); print("\n== 손절된 비율 =="); print(t.round(0).to_string())
