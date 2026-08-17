"""diag_entry.py — 진입 신호 진단.

'규칙을 바꿔봤더니 나빴다' 로 끝내지 않기 위해, 기준선 신호가 **무엇을 잡고 무엇을
놓치는지**를 조건별로 분해한다. 포트폴리오 시뮬레이션이 아니라 **신호 단위 전방수익**
분석이라 슬롯·현금 제약의 교란이 없다.

측정: 신호일 t → t+1 시가 매수 가정 → t+N 종가까지 수익률 (N=5/10/20)
분해축: 레짐 / 눌림깊이 / 변동배율 / 지수 추세 / 종목 추세 여력 / 외국인 수급

실행: python3 diag_entry.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from backtest import (EntryParams, UniverseParams, VolScaleParams, compute_signals,
                      load_panel, DATA)

HORIZONS = (5, 10, 20)


def build_events(panel, ep: EntryParams) -> pd.DataFrame:
    sig = compute_signals(panel, ep)
    signal, depth = sig["signal"], sig["depth"]

    op = panel.open.to_numpy(); cl = panel.close.to_numpy()
    dp = depth.to_numpy(); sc = panel.vol_scale.to_numpy(); yz = panel.yz20.to_numpy()
    ma120 = panel.close.rolling(120, min_periods=80).mean().to_numpy()
    ma20 = panel.close.rolling(20, min_periods=20).mean().to_numpy()
    sg = signal.to_numpy()
    dates = panel.dates
    tick = list(panel.tickers)

    rows = []
    n_d = len(dates)
    di, tj = np.nonzero(sg)
    for i, j in zip(di, tj):
        if i + 1 >= n_d or not np.isfinite(op[i + 1, j]) or op[i + 1, j] <= 0:
            continue
        entry = op[i + 1, j]
        r = {"date": dates[i], "ticker": tick[j], "depth": dp[i, j],
             "scale": sc[i, j], "yz20": yz[i, j],
             "above_ma120": cl[i, j] / ma120[i, j] - 1 if np.isfinite(ma120[i, j]) else np.nan,
             "above_ma20": cl[i, j] / ma20[i, j] - 1 if np.isfinite(ma20[i, j]) else np.nan,
             "entry": entry}
        for h in HORIZONS:
            k = min(i + h, n_d - 1)
            r[f"fwd{h}"] = cl[k, j] / entry - 1 if np.isfinite(cl[k, j]) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def bucket_report(ev: pd.DataFrame, col: str, q: int = 5, label: str = "") -> pd.DataFrame:
    x = ev[col].replace([np.inf, -np.inf], np.nan)
    try:
        b = pd.qcut(x, q, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    tmp = ev.copy()
    for h in HORIZONS:
        tmp[f"_w{h}"] = tmp[f"fwd{h}"] > 0
    g = tmp.groupby(b, observed=True)
    out = pd.DataFrame({
        "n": g.size(),
        **{f"fwd{h}_평균": (g[f"fwd{h}"].mean() * 100).round(2) for h in HORIZONS},
        **{f"fwd{h}_승률": (g[f"_w{h}"].mean() * 100).round(1) for h in HORIZONS},
    })
    out.index.name = label or col
    return out


if __name__ == "__main__":
    panel = load_panel(UniverseParams(), VolScaleParams())
    print(f"패널 종목 {len(panel.tickers)} · 거래일 {len(panel.dates)}\n")

    ev = build_events(panel, EntryParams())
    ev = ev[(ev.date >= "2015-01-01") & (ev.date <= "2026-06-30")].copy()
    print(f"기준선 신호 이벤트 {len(ev):,}건 "
          f"({ev.date.min().date()}~{ev.date.max().date()})\n")

    print("── 전체 ──")
    for h in HORIZONS:
        s = ev[f"fwd{h}"].dropna()
        print(f"  fwd{h:>2}일  평균 {s.mean()*100:6.2f}%  중앙 {s.median()*100:6.2f}%  "
              f"승률 {s.gt(0).mean()*100:5.1f}%  n={len(s):,}")

    print("\n── 연도별 ──")
    ev["_win10"] = ev["fwd10"] > 0
    g = ev.groupby(ev.date.dt.year)
    print(pd.DataFrame({
        "n": g.size(),
        "fwd10_평균": (g["fwd10"].mean() * 100).round(2),
        "fwd10_승률": (g["_win10"].mean() * 100).round(1),
        "fwd20_평균": (g["fwd20"].mean() * 100).round(2),
    }).to_string())

    for col, lab in (("depth", "눌림깊이"), ("scale", "변동배율"),
                     ("yz20", "YZ_20"), ("above_ma120", "MA120 대비 여력"),
                     ("above_ma20", "MA20 대비")):
        r = bucket_report(ev, col, 5, lab)
        if len(r):
            print(f"\n── {lab} 5분위 ──")
            print(r.to_string())

    out = Path(__file__).resolve().parent / "out"
    out.mkdir(exist_ok=True)
    ev.to_csv(out / "diag_entry_events.csv", index=False)
    print(f"\n저장: {out/'diag_entry_events.csv'}")
