# -*- coding: utf-8 -*-
"""regime_layers.py — 작업계획 2-a 정답지 · 2-b 층위 지도 (케인 2026-11-05).

정답지  KOSPI200 종가 zigzag 20% 전환점 (stop_regime.regime 과 동일). 학습 = 2026-05-29 까지의 전환점.
        보류 구간(2026-06-01~) 전환점은 표에만 적고 분석에 쓰지 않는다.
층      data_idx/krx_indices.parquet 24개 + 직접 만든 4개 (메가갤럭시·유니버스 × 시총가중·동일가중)
        직접 만든 지수: 전일 편입 종목의 당일 수익률. 시총 = 전일 종가 × 직전 월말 상장주식수(meta). |수익률|>35% 는 자료 오류로 제외
선행    ① 전환점마다 각 층의 자기 극값(±30거래일 안 최고/최저 종가)이 KOSPI200 전환일보다 며칠 앞/뒤인가 (음수 = 먼저)
        ② 일별 수익률 앞뒤 상관: corr(층 t, KOSPI200 t+k) − corr(KOSPI200 t, 층 t+k), k=1..3
사용    python3 regime_layers.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import backtest as bt, stop_research as sr
from stop_regime import zigzag

TRAIN_END = pd.Timestamp("2026-05-29"); W = 30
OUT = HERE / "out" / "regime"; OUT.mkdir(parents=True, exist_ok=True)
ORDER = ["KTOP 30", "코스피 50", "KRX 100", "코스피 100", "코스피 200", "코스피200 동일가중(ETF 252650)", "KRX 300",
         "코스피 200 초대형제외 지수", "코스피 200 중소형주", "코스피 대형주", "코스피 중형주", "코스피 소형주",
         "KRX 중대형 TMI", "KRX 중형 TMI", "KRX 소형 TMI", "KRX 초소형 TMI", "코스피", "코스피200제외 코스피지수", "KRX TMI",
         "코스닥 150", "코스닥 대형주", "코스닥 중형주", "코스닥 소형주", "코스닥",
         "메가갤럭시 시총가중", "메가갤럭시 동일가중", "유니버스 시총가중", "유니버스 동일가중"]

def composite(close, elig, shares):
    r = close.pct_change(); r = r.where(r.abs() <= 0.35)
    mem = elig.shift(1).fillna(False).astype(bool) & r.notna()
    sh = shares.reindex(index=close.index, method="ffill").reindex(columns=close.columns)
    mc = (close.shift(1) * sh).where(mem)
    ew = r.where(mem).mean(axis=1); cw = (r.where(mem) * mc).sum(axis=1) / mc.sum(axis=1)
    return (1 + ew.fillna(0)).cumprod(), (1 + cw.fillna(0)).cumprod()

def load_levels():
    d = pd.read_parquet(HERE / "data_idx" / "krx_indices.parquet")
    lv = d.pivot_table(index="date", columns="name", values="close").sort_index(); lv.index = pd.to_datetime(lv.index)
    meta = pd.read_parquet(bt.DATA / "meta.parquet", columns=["shares"]).reset_index()
    meta["date"] = pd.to_datetime(meta.date) + pd.offsets.MonthBegin(1)   # 월말 주식수는 다음 달부터 사용
    shares = meta.pivot_table(index="date", columns="ticker", values="shares")
    ps = bt.load_panel(bt.UniverseParams(), bt.VolScaleParams(), end=sr.END)
    pdy = sr.load_day_panel(sr.END)
    for nm, P in (("메가갤럭시", ps), ("유니버스", pdy)):
        ew, cw = composite(P.close, P.elig.fillna(False), shares)
        lv[f"{nm} 동일가중"] = ew.reindex(lv.index); lv[f"{nm} 시총가중"] = cw.reindex(lv.index)
    return lv.loc["2015-01-02":, [c for c in ORDER if c in lv.columns]]

def turns(k200):
    pts = zigzag(k200.loc["2014-06-01":], 0.20)
    pts = [p for p in pts[1:-1] if p[0] >= pd.Timestamp("2015-01-02")]   # 첫 점(시작 인공점)·마지막(미확정) 제외
    rows = []; prev = None
    for d, t in pts:
        lvl = k200.loc[d]; rows.append(dict(date=d, kind="고점" if t == "H" else "저점", K200=lvl,
                                            직전전환대비=(lvl / prev - 1) * 100 if prev else np.nan, 학습=d <= TRAIN_END)); prev = lvl
    return pd.DataFrame(rows)

def offsets(lv, T):
    idx = lv.index; rows = []
    for _, t in T.iterrows():
        i0 = idx.get_loc(t.date); a, b = max(0, i0 - W), min(len(idx) - 1, i0 + W)
        r = {"전환": f"{t.date.date()} {t.kind}", "kind": t.kind, "학습": t.학습}
        for c in lv.columns:
            w = lv[c].iloc[a:b + 1]
            if w.notna().sum() < (b - a) * 0.8: r[c] = np.nan; continue
            j = w.idxmax() if t.kind == "고점" else w.idxmin()
            r[c] = idx.get_loc(j) - i0
        rows.append(r)
    return pd.DataFrame(rows).set_index("전환")

def leadlag(lv, end=TRAIN_END):
    r = lv.loc[:end].pct_change(); k = r["코스피 200"]; rows = {}
    for c in lv.columns:
        x = r[c]; d = {}
        for L in (1, 2, 3):
            d[f"층→K200 lag{L}"] = x.corr(k.shift(-L)); d[f"K200→층 lag{L}"] = k.corr(x.shift(-L))
        d["동시 상관"] = x.corr(k)
        d["선행 우위 (lag1)"] = d["층→K200 lag1"] - d["K200→층 lag1"]
        rows[c] = d
    return pd.DataFrame(rows).T

def plot(lv, off, T):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "NanumSquare"; plt.rcParams["axes.unicode_minus"] = False
    o = off[off.학습].drop(columns=["kind", "학습"]).astype(float)
    fig, ax = plt.subplots(figsize=(14, 0.42 * len(o.columns) + 2))
    im = ax.imshow(o.T.values, cmap="RdBu_r", vmin=-20, vmax=20, aspect="auto")   # 케인 색 규칙: 음수(먼저) 파랑 · 양수(늦게) 빨강
    ax.set_xticks(range(len(o.index))); ax.set_xticklabels(o.index, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(o.columns))); ax.set_yticklabels(o.columns, fontsize=9)
    for i in range(o.shape[0]):
        for j in range(o.shape[1]):
            v = o.iloc[i, j]
            if np.isfinite(v): ax.text(i, j, f"{int(v):+d}", ha="center", va="center", fontsize=7)
    ax.set_title("KOSPI200 전환일 대비 각 층의 자기 극값 날짜 (거래일, 음수·파랑=먼저 · 양수·빨강=늦게) · 학습 구간 전환점")
    fig.colorbar(im, ax=ax, shrink=0.6); fig.tight_layout(); fig.savefig(OUT / "layer_offsets_heatmap.png", dpi=130); plt.close(fig)
    # 전 기간 규모별 층 (로그)
    cols = ["KTOP 30", "KRX 100", "코스피 200", "KRX 중대형 TMI", "KRX 중형 TMI", "KRX 소형 TMI", "KRX 초소형 TMI", "코스닥", "메가갤럭시 시총가중", "유니버스 동일가중"]
    fig, ax = plt.subplots(figsize=(14, 6)); z = lv[cols].div(lv[cols].iloc[0])
    for c in cols: ax.plot(z.index, z[c], lw=1.1, label=c)
    for _, t in T.iterrows(): ax.axvline(t.date, color="#ef5350" if t.kind == "고점" else "#1976D2", lw=0.7, ls="--", alpha=0.6)
    ax.axvspan(pd.Timestamp("2026-06-01"), lv.index[-1], color="grey", alpha=0.12, label="보류 구간")
    ax.set_yscale("log"); ax.legend(ncol=4, fontsize=8); ax.set_title("층별 지수 (2015-01-02=1, 로그) · 점선 = KOSPI200 zigzag 20% 전환점 (빨강 고점, 파랑 저점)")
    fig.tight_layout(); fig.savefig(OUT / "layer_levels.png", dpi=130); plt.close(fig)

def main():
    lv = load_levels(); lv.to_parquet(OUT / "layer_levels.parquet")
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
    T = turns(lv["코스피 200"]); T.to_csv(OUT / "turns_k200_zz20.csv", index=False)
    print("== 2-a 정답지: KOSPI200 zigzag 20% 전환점 ==\n", T.round(1).to_string(index=False))
    off = offsets(lv, T); off.to_csv(OUT / "layer_offsets.csv")
    o = off[off.학습]
    summ = pd.DataFrame({"고점 중앙": o[o.kind == "고점"].drop(columns=["kind", "학습"]).median(),
                         "고점 먼저 비율%": (o[o.kind == "고점"].drop(columns=["kind", "학습"]) < 0).mean() * 100,
                         "저점 중앙": o[o.kind == "저점"].drop(columns=["kind", "학습"]).median(),
                         "저점 먼저 비율%": (o[o.kind == "저점"].drop(columns=["kind", "학습"]) < 0).mean() * 100})
    print("\n== 2-b ① 전환점별 자기 극값 날짜 (거래일, 음수=KOSPI200보다 먼저) ==\n", off.drop(columns=["kind"]).T.to_string())
    print("\n== 학습 구간 요약 (고점·저점 따로) ==\n", summ.round(1).to_string()); summ.to_csv(OUT / "layer_offsets_summary.csv")
    ll = leadlag(lv); ll.to_csv(OUT / "layer_leadlag.csv")
    print("\n== 2-b ② 일별 수익률 앞뒤 상관 (2015-01 ~ 2026-05) ==\n", ll.round(3).to_string())
    plot(lv, off, T); print("그림:", OUT / "layer_offsets_heatmap.png", OUT / "layer_levels.png")

if __name__ == "__main__":
    main()
