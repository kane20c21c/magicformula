# -*- coding: utf-8 -*-
"""mid_sim2.py — 급락 모듈(R2) 다듬기: 슬롯 수 × 변동성 상한 × 선택 × 진입 창 × 손절 슬리피지 격자

1차 시뮬(mid_sim.py) 결론: 후보 안에 순위 정보가 없고(무작위 ≥ 어떤 정렬), 변동성 꼬리만 피하면 된다.
→ 슬롯을 늘려 셀 평균에 수렴시키고, 변동성 상한을 걸고, 손절 슬리피지를 넣어 실전 쪽으로 당겨 본다.

격자:
  slots       10 / 20 / 30
  volcap      없음 / 그날 후보 중 vol20 상위 25% 제외
  select      vol20최저 / 무작위(시드 5 평균 — 지표는 시드 평균, NAV 는 중앙값 시드)
  window      당일~20 (기본) / 당일~5 / 6~20
  stop_slip   0 / 1% (하단 배리어 청산에만)
  규칙 기본형: R2 = crash_days∈window × 시장200∈{아래+상승, 아래+하락} × dd20 ≤ −12

산출: out/mid/sim2_grid.csv · sim2_summary.md · sim2_equity.png (상위 4 + 기준선)
실행: python3 mid_sim2.py
"""
import time, itertools
import numpy as np, pandas as pd
import mid_gate as g
import mid_sim as ms

OUT = g.OUT
WINDOWS = {"당일~20": ["당일", "1~5", "6~20"], "당일~5": ["당일", "1~5"], "6~20": ["6~20"]}
SEEDS = [0, 1, 2, 3, 4]


def main():
    t0 = time.time()
    df = pd.read_parquet(OUT / "states2.parquet")
    W, mktcap, foreign, managed, static, sector, fw = g.load(False)
    C = W["close"]; dates = C.index
    r_path, h, lab, en_arr, close = ms.labels_with_hold(W)
    vol20n = g.pc(C).rolling(20, min_periods=10).std().to_numpy()
    di = pd.Series(np.arange(len(dates)), index=dates); ti = pd.Series(np.arange(len(C.columns)), index=C.columns)
    df["di"] = di.reindex(df.date).to_numpy(); df["ti"] = ti.reindex(df.ticker).to_numpy()
    df = df.dropna(subset=["di", "ti"]); df["di"] = df.di.astype(int); df["ti"] = df.ti.astype(int)
    df["vol20n"] = vol20n[df.di, df.ti]; df["h"] = h[df.di, df.ti]
    df = df.dropna(subset=["h"])
    base = df[df.mkt_200.isin(["아래+상승", "아래+하락"]) & df.dd20.isin(["<-20", "-20~-12"]) & (df.date >= g.EVAL_START)]
    base = base.rename(columns={"rpath10": "r_path", "lab10": "lab"})
    g.log(f"R2 모집단 {len(base):,}셀 ({time.time()-t0:.0f}s)")
    ev_dates = dates[dates >= g.EVAL_START]; d0 = int(di[ev_dates[0]])

    rows, navs = [], {}
    grid = list(itertools.product(WINDOWS, [10, 20, 30], ["없음", "vol상위25%제외"], ["vol20최저", "무작위"], [0.0, 0.01]))
    for win, slots, volcap, sel, slip in grid:
        s = base[base.crash_days.isin(WINDOWS[win])].copy()
        if volcap != "없음":
            q = s.groupby("date").vol20n.transform(lambda x: x.quantile(0.75))
            s = s[s.vol20n < q]
        s["rule"] = "R2"; s["prio"] = 0
        name = f"{win}|s{slots}|{volcap}|{sel}|slip{int(slip*100)}"
        mets, nav_list = [], []
        seeds = SEEDS if sel == "무작위" else [None]
        for sd in seeds:
            if sd is None:
                s["key"] = s.vol20n
            else:
                s["key"] = np.random.default_rng(sd).random(len(s))
            nav, tr = ms.simulate(s, dates, close, ti, en_arr, slots=slots, stop_slip=slip)
            nav = nav.iloc[d0:]; nav = nav / nav.iloc[0]
            m = ms.metrics(nav, tr, s.date.nunique(), len(nav), (tr.h.sum() / slots / len(nav)) if len(tr) else 0.0)
            for half, selh in [("A", nav.index.year < 2021), ("B", nav.index.year >= 2021)]:
                nv = nav[selh]; nv = nv / nv.iloc[0]; yrs = max((nv.index[-1] - nv.index[0]).days / 365.25, 0.5)
                m[f"cagr_{half}"] = nv.iloc[-1] ** (1 / yrs) - 1; m[f"mdd_{half}"] = (nv / nv.cummax() - 1).min()
            mets.append(m); nav_list.append(nav)
        M = pd.DataFrame(mets).mean(numeric_only=True).to_dict()
        M.update(dict(window=win, slots=slots, volcap=volcap, select=sel, stop_slip=slip, name=name, n_cand=len(s), cand_per_day=len(s) / max(s.date.nunique(), 1),
                      cell_avg=float(s.r_path.mean())))
        rows.append(M)
        med = int(np.argsort([nv.iloc[-1] for nv in nav_list])[len(nav_list) // 2])
        navs[name] = nav_list[med]
        g.log(f"{name}: CAGR {M['cagr']:+.1%} MDD {M['mdd']:.1%} Sharpe {M['sharpe']:.2f} 승률 {M['win']:.2f} 노출 {M['slot_use']:.1%} ({time.time()-t0:.0f}s)")
    grid_df = pd.DataFrame(rows)
    cols = ["window", "slots", "volcap", "select", "stop_slip", "cand_per_day", "cell_avg", "total", "cagr", "mdd", "sharpe", "trades", "win", "avg_r", "slot_use", "cagr_A", "mdd_A", "cagr_B", "mdd_B"]
    grid_df = grid_df[cols + ["name"]].sort_values("sharpe", ascending=False)
    grid_df.to_csv(OUT / "sim2_grid.csv", index=False)

    # 벤치마크 & 그래프
    bm = g.build_masks(W, mktcap, foreign, managed, static)["all"]
    rr = g.pc(C); wgt = mktcap.where(bm & rr.notna())
    bench = (1 + ((rr * wgt).sum(1) / wgt.sum(1)).fillna(0)).cumprod(); bench = bench.loc[ev_dates]; bench = bench / bench.iloc[0]
    top = grid_df.head(4).name.tolist()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "AppleGothic"; plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(bench.index, bench, color="#444", lw=1.6, label="유니버스 지수")
    for nm in top:
        ax.plot(navs[nm].index, navs[nm], lw=1.1, label=nm)
    ax.set_yscale("log"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, frameon=False)
    ax.set_title("급락 모듈(R2) 격자 — Sharpe 상위 4 (2015-01=1)", loc="left")
    fig.savefig(OUT / "sim2_equity.png", dpi=130, bbox_inches="tight")

    fmt = {"cand_per_day": "{:.0f}", "cell_avg": "{:+.2%}", "total": "{:+.0%}", "cagr": "{:+.1%}", "mdd": "{:.1%}", "sharpe": "{:.2f}", "trades": "{:,.0f}",
           "win": "{:.2f}", "avg_r": "{:+.2%}", "slot_use": "{:.1%}", "cagr_A": "{:+.1%}", "mdd_A": "{:.1%}", "cagr_B": "{:+.1%}", "mdd_B": "{:.1%}", "stop_slip": "{:.0%}"}
    def tbl(d):
        L = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in d.iterrows():
            L.append("| " + " | ".join((fmt.get(c, "{}").format(r[c]) if pd.notna(r[c]) else "—") for c in cols) + " |")
        return "\n".join(L)
    o = ["# 급락 모듈(R2) 격자 시뮬", f"{ev_dates[0].date()}~{ev_dates[-1].date()} · 비용 왕복 0.30% · 무작위는 시드 5 평균 · 유니버스 지수 CAGR {bench.iloc[-1] ** (1/((bench.index[-1]-bench.index[0]).days/365.25)) - 1:+.1%} MDD {(bench/bench.cummax()-1).min():.1%}", ""]
    o.append("## Sharpe 상위 15"); o.append(tbl(grid_df.head(15)))
    o.append("\n## 실전 가정(slip 1% · vol상위25%제외) 만 — Sharpe 순"); o.append(tbl(grid_df[(grid_df.stop_slip == 0.01) & (grid_df.volcap != "없음")]))
    o.append("\n## 축별 평균 효과 (Sharpe / CAGR / MDD)")
    for ax_ in ["window", "slots", "volcap", "select", "stop_slip"]:
        t = grid_df.groupby(ax_)[["sharpe", "cagr", "mdd"]].mean()
        o.append(f"### {ax_}"); o.append("```\n" + t.round(3).to_string() + "\n```")
    (OUT / "sim2_summary.md").write_text("\n".join(o), encoding="utf-8")
    g.log(f"완료 ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
