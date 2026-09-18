# -*- coding: utf-8 -*-
"""mid_market_plot.py — 시장 전기간 그래프: KOSPI200 vs 우리 유니버스(시총가중) 일간 + MA60/120/200 + MA200 4국면 띠
실행: python3 mid_market_plot.py  → out/mid/market_ma.png (+ ~/DriveForALL/StoLab/RESULTS/ 복사)
KOSPI200 일별: data_ext/kospi200_daily.parquet (FinanceDataReader KS200)
"""
import shutil
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mid_gate as g

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
RED, BLUE = "#ef5350", "#1976D2"
REG = {"위+상승": "#ef5350", "위+하락": "#f5b7b1", "아래+상승": "#9ec5ea", "아래+하락": "#1976D2"}


def quad(price, ma, slope):
    up, rise = price > ma, slope > 0
    q = np.where(up & rise, "위+상승", np.where(up & ~rise, "위+하락", np.where(~up & rise, "아래+상승", "아래+하락")))
    return pd.Series(np.where(ma.isna() | slope.isna(), None, q), index=price.index)


def universe_index():
    W, mktcap, foreign, managed, static, sector, fw = g.load(False)
    C = W["close"]
    base = g.build_masks(W, mktcap, foreign, managed, static)["all"]
    r = g.pc(C)
    wgt = mktcap.where(base & r.notna())
    mkt_ret = (r * wgt).sum(1) / wgt.sum(1)
    return (1 + mkt_ret.fillna(0)).cumprod() * 100


def panel(ax, axr, s, title):
    mas = {p: s.rolling(p, min_periods=p).mean() for p in [60, 120, 200]}
    ax.plot(s.index, s, color="#444", lw=0.7, label="일간")
    for p, c, lw in [(60, "#f4a261", 0.9), (120, "#2a9d8f", 0.9), (200, "#e63946", 1.4)]:
        ax.plot(mas[p].index, mas[p], color=c, lw=lw, label=f"MA{p}")
    ax.set_yscale("log"); ax.set_title(title, loc="left", fontsize=12)
    ax.legend(loc="upper left", fontsize=8, ncol=4, frameon=False)
    ax.grid(alpha=0.25, which="both")
    reg = quad(s, mas[200], mas[200] / mas[200].shift(20) - 1)
    for st, col in REG.items():
        m = (reg == st).to_numpy()
        axr.fill_between(s.index, 0, 1, where=m, color=col, step="mid", linewidth=0)
    axr.set_ylim(0, 1); axr.set_yticks([]); axr.set_ylabel("MA200\n4국면", fontsize=8, rotation=0, labelpad=28, va="center")
    for sp in axr.spines.values():
        sp.set_visible(False)


def main():
    k = pd.read_parquet(g.DATA / "kospi200_daily.parquet")["Close"].astype(float)
    k.index = pd.to_datetime(k.index)
    k = k[(k.index >= "2013-06-01") & (k.index <= "2026-09-11")]
    u = universe_index()
    lo, hi = "2014-01-01", "2026-06-30"
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(5, 1, height_ratios=[5, 0.35, 0.9, 5, 0.35], hspace=0.05)
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[3]), fig.add_subplot(gs[4])]
    for ax in axes[1:]:
        ax.sharex(axes[0])
    for ax in axes[:3]:
        ax.tick_params(labelbottom=False)
    panel(axes[0], axes[1], k[k.index >= "2014-01-01"], "KOSPI200 (일간, FinanceDataReader) — 2014-01 ~ 2026-09")
    panel(axes[2], axes[3], u[(u.index >= lo) & (u.index <= hi)], "우리 유니버스 시총가중 지수 (all 계층, 2014-01=100) — 2014-01 ~ 2026-06")
    axes[3].xaxis.set_major_locator(mdates.YearLocator()); axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in REG.values()]
    fig.legend(handles, list(REG.keys()), loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.005))
    out = g.OUT / "market_ma.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    dst = Path.home() / "DriveForALL/StoLab/RESULTS/mid_market_ma.png"
    dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(out, dst)
    print("saved", out, "->", dst)


if __name__ == "__main__":
    main()
