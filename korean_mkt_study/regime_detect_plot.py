# -*- coding: utf-8 -*-
"""regime_detect_plot.py — 2-c 설명용: 판정기 상태 띠 그림 + 기준선(항상 상승) 일치율 (케인 2026-11-06)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from regime_detect import truth, OUT, TRAIN_END, EVAL0

SHOW = ["K200 200일선 ±2% 밴드", "K200 120일선", "K200 실시간 zigzag 10%", "K200 60일선",
        "괴리형: K200·중형TMI 둘 다 200일선 위", "[조합 C2] 괴리형 퇴출 → 퇴출 뒤 저점 +10% 반등 복귀",
        "브레드스 EMA20 − 250일 중앙 ±0.05", "[유니버스 브레드스] EMA20 − 250일 중앙 ±0.05", "[메가갤럭시 브레드스] EMA20 − 250일 중앙 ±0.05",
        "[유니버스 브레드스] ±0.05 신호 뒤 20일만, 그 밖엔 K200 120일선", "[메가갤럭시 브레드스] ±0.05 신호 뒤 20일만, 그 밖엔 K200 120일선"]
SHOW = [c for c in SHOW if c not in ("K200 60일선", "괴리형: K200·중형TMI 둘 다 200일선 위")]

def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    plt.rcParams["font.family"] = "NanumSquare"; plt.rcParams["axes.unicode_minus"] = False
    lv = pd.read_parquet(OUT / "layer_levels.parquet"); S = pd.read_parquet(OUT / "detect_states.parquet")
    k = lv["코스피 200"]; tr, turns = truth(k); w = (k.index >= EVAL0) & (k.index <= TRAIN_END)
    print(f"평가 창 정답 상승일 비중 {tr[w].mean()*100:.1f}% → '항상 상승' 판정기의 일치율 = {tr[w].mean()*100:.1f}%")
    for c in SHOW:
        s = S[c].reindex(k.index).ffill().astype(bool)[w]; t = tr[w]
        print(f"  {c:<34} 하락 국면 중 '하락' 판정일 {(~s[~t]).mean()*100:5.1f}% · 상승 국면 중 '하락' 판정(헛 현금)일 {(~s[t]).mean()*100:5.1f}%")
    kk = k.loc[EVAL0:]; rows = [("정답 (zigzag 20%, 사후)", tr.loc[EVAL0:])] + [(c, S[c].reindex(kk.index).ffill().astype(bool)) for c in SHOW]
    fig, axs = plt.subplots(len(rows) + 1, 1, figsize=(16, 2.2 + 0.45 * len(rows)), sharex=True,
                            gridspec_kw={"height_ratios": [4] + [0.6] * len(rows)})
    axs[0].plot(kk.index, kk, color="black", lw=1); axs[0].set_yscale("log"); axs[0].set_title("코스피 200 (로그) · 띠: 빨강 = 상승 판정, 파랑 = 하락 판정 · 회색 = 보류 구간")
    for d, kd in turns:
        if d >= EVAL0: axs[0].axvline(d, color="#ef5350" if kd == "고점" else "#1976D2", lw=0.8, ls="--")
    cm = ListedColormap(["#1976D2", "#ef5350"])
    for ax, (nm, s) in zip(axs[1:], rows):
        ax.imshow(s.astype(int).to_numpy()[None, :], aspect="auto", cmap=cm, vmin=0, vmax=1,
                  extent=[matplotlib.dates.date2num(s.index[0]), matplotlib.dates.date2num(s.index[-1]), 0, 1])
        ax.set_yticks([]); ax.set_ylabel(nm.replace("−", "-"), rotation=0, ha="right", va="center", fontsize=8)
    for ax in axs: ax.axvspan(pd.Timestamp("2026-06-01"), kk.index[-1], color="grey", alpha=0.35)
    axs[-1].xaxis_date(); fig.tight_layout(); fig.savefig(OUT / "detect_timeline.png", dpi=130); plt.close(fig)
    print("그림", OUT / "detect_timeline.png")

if __name__ == "__main__":
    main()
