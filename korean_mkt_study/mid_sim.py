# -*- coding: utf-8 -*-
"""mid_sim.py — 중기 모델 규칙 시뮬 (10슬롯 · 동일비중 · 트리플 배리어 청산 N=10 k=1.0)

셀 평균은 급락일에 수백 종목이 한꺼번에 조건을 만족하는 문제를 못 본다 → 실제 포트로 재본다.
  진입: 신호일 t 종가 기준 규칙 충족 → t+1 시가 매수 (슬롯 빈 만큼, gap5 최저순 = 가장 깊게 빠진 순)
  청산: 상단/하단 배리어 터치일(배리어 가격 체결 가정) 또는 N일 만기 종가. 슬롯은 청산일에 풀려 같은 날 신호부터 재사용.
  비용: 왕복 0.30% (화이트포트 가정과 동일). 자금: 진입 시 NAV 의 1/10 (복리), 일별 MTM 은 종가.

규칙 (2회차 사다리 결과):
  R1 회복초입 스프링  시장200 아래+상승 × gap5<−5 × 종목200 아래+상승 × dd20<−20 × 전환 1~5일째 제외
  R2 회복국면 급락    crash_days∈{당일,1~5,6~20} × 시장200∈{아래+상승,아래+하락} × dd20 ≤ −12
  R3 강세장 눌림      시장200 위+상승 × 시장60 아래+상승 × 종목60 아래 × gap5<−5   (저강도 채움용)
  조합: R1+R2, R1+R2+R3 (우선순위 R1 > R2 > R3, 같은 규칙 안에서는 gap5 최저순)

판정: 연환산·MDD·거래수·승률·무신호일 비율·슬롯 점유율·연도별·전반/후반 · 대조 = 유니버스 지수 보유
실행: python3 mid_sim.py  (states2.parquet 필요)
산출: out/mid/sim_summary.csv · sim_trades.csv · sim_nav.csv · sim_equity.png · sim_summary.md
"""
import time
from pathlib import Path
import numpy as np, pandas as pd
import mid_gate as g

OUT = g.OUT
N, K = 10, 1.0
SLOTS, COST = 10, 0.0030
RULES = {
    "R1": lambda d: (d.mkt_200 == "아래+상승") & (d.gap5 == "<-5") & (d.stk_200 == "아래+상승") & (d.dd20 == "<-20") & (d.mkt_turn_days != "1~5"),
    "R2": lambda d: d.crash_days.isin(["당일", "1~5", "6~20"]) & d.mkt_200.isin(["아래+상승", "아래+하락"]) & d.dd20.isin(["<-20", "-20~-12"]),
    "R3": lambda d: (d.mkt_200 == "위+상승") & (d.mkt_60 == "아래+상승") & d.stk_60.str.startswith("아래") & (d.gap5 == "<-5"),
}
COMBOS = {"R1": ["R1"], "R2": ["R2"], "R3": ["R3"], "R1+R2": ["R1", "R2"], "R1+R2+R3": ["R1", "R2", "R3"]}


def labels_with_hold(W):
    """build_labels 와 같은 로직 + 보유일 h(1..N) 반환. r_path 는 배리어 체결 가정(비용 전)."""
    C, H, L, O = W["close"], W["high"], W["low"], W["open"]
    sig = g.pc(C).rolling(20, min_periods=10).std()
    en = O.shift(-1).to_numpy()
    width = (K * sig * np.sqrt(N)).to_numpy()
    upv, dnv = en * (1 + width), en * (1 - width)
    Hn, Ln = H.to_numpy(), L.to_numpy()
    big = N + 1
    t_up = np.full(C.shape, big, np.int16); t_dn = t_up.copy()
    for d in range(1, N + 1):
        hs = np.full_like(Hn, np.nan); hs[:-d] = Hn[d:]
        ls = np.full_like(Ln, np.nan); ls[:-d] = Ln[d:]
        t_up[(hs >= upv) & (t_up == big)] = d
        t_dn[(ls <= dnv) & (t_dn == big)] = d
    r_raw = C.shift(-N).to_numpy() / en - 1
    valid = ~np.isnan(r_raw) & ~np.isnan(width) & (width > 0) & ~np.isnan(en)
    lab = np.where(t_dn <= t_up, -1.0, 1.0); lab = np.where((t_up == big) & (t_dn == big), 0.0, lab)
    r_path = np.where(lab == 1, width, np.where(lab == -1, -width, r_raw))
    h = np.where(lab == 1, t_up, np.where(lab == -1, t_dn, N)).astype(float)
    return (np.where(valid, r_path, np.nan), np.where(valid, h, np.nan), np.where(valid, lab, np.nan), en, C.to_numpy())


def simulate(sig, dates, close, tick_idx, en_arr, nav0=1.0, slots=SLOTS, stop_slip=0.0):
    """slots: 슬롯 수, stop_slip: 하단 배리어 청산(lab=-1) 에 추가되는 슬리피지(수익률 차감)."""
    """sig: DataFrame[date, ticker, di(날짜 idx), ti(종목 idx), prio, gap5n, r_path, h] — 신호일 기준.
    반환: nav Series(일별), trades DataFrame."""
    by_day = {k: v.sort_values(["prio", "key"]) for k, v in sig.groupby("di")}
    cash, pos, nav_hist, trades = nav0, [], [], []
    T = len(dates)
    for t in range(T):
        # 1) 오늘 청산 (exit_di == t): 배리어 가격 or 만기 종가 — r_path 반영
        keep = []
        for p in pos:
            if p["exit_di"] == t:
                rp = p["r_path"] - (stop_slip if p.get("lab") == -1 else 0.0)
                val = p["size"] * (1 + rp) * (1 - COST / 2)
                cash += val
                trades.append(dict(rule=p["rule"], ticker=p["ticker"], signal=dates[p["sig_di"]], entry=dates[p["entry_di"]], exit=dates[t],
                                   h=p["h"], r_gross=rp, r_net=(1 + rp) * (1 - COST) - 1, size=p["size"]))
            else:
                keep.append(p)
        pos = keep
        # 2) 오늘 신호 → 내일 시가 진입 (슬롯 빈 만큼). 내일이 없으면 스킵.
        if t in by_day and t + 1 < T:
            held = {p["ticker"] for p in pos}
            nav_now = cash + sum(p["size"] * close[t, p["ti"]] / en_arr[p["sig_di"], p["ti"]] for p in pos if not np.isnan(close[t, p["ti"]]))
            unit = nav_now / slots
            for _, s in by_day[t].iterrows():
                if len(pos) >= slots or cash < unit * 0.999:
                    break
                if s.ticker in held:
                    continue
                size = min(unit, cash) * (1 - COST / 2)
                cash -= min(unit, cash)
                pos.append(dict(rule=s.rule, ticker=s.ticker, ti=int(s.ti), sig_di=t, entry_di=t + 1, exit_di=t + int(s.h),
                                r_path=float(s.r_path), h=int(s.h), size=size, lab=int(s.lab) if "lab" in s else None))
                held.add(s.ticker)
        # 3) MTM (종가 기준; 진입 전날은 현금)
        mtm = cash
        for p in pos:
            if t >= p["entry_di"]:
                c = close[t, p["ti"]]; e = en_arr[p["sig_di"], p["ti"]]
                mtm += p["size"] * (c / e if (not np.isnan(c) and e > 0) else 1.0)
            else:
                mtm += p["size"]
        nav_hist.append(mtm)
    return pd.Series(nav_hist, index=dates), pd.DataFrame(trades)


def metrics(nav, trades, nsig_days, total_days, slot_use):
    r = nav.pct_change().dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = nav.iloc[-1] ** (1 / yrs) - 1
    dd = (nav / nav.cummax() - 1).min()
    vol = r.std() * np.sqrt(252)
    m = dict(total=nav.iloc[-1] - 1, cagr=cagr, mdd=dd, sharpe=(r.mean() * 252) / vol if vol > 0 else np.nan,
             trades=len(trades), win=(trades.r_net > 0).mean() if len(trades) else np.nan,
             avg_r=trades.r_net.mean() if len(trades) else np.nan, avg_h=trades.h.mean() if len(trades) else np.nan,
             signal_days=nsig_days / total_days, slot_use=slot_use)
    return m


def main():
    t0 = time.time()
    df = pd.read_parquet(OUT / "states2.parquet")
    W, mktcap, foreign, managed, static, sector, fw = g.load(False)
    C = W["close"]; dates = C.index
    r_path, h, lab, en_arr, close = labels_with_hold(W)
    gap5n = (C / C.rolling(5, min_periods=5).mean() - 1).to_numpy()
    vol20n = g.pc(C).rolling(20, min_periods=10).std().to_numpy()
    amt20n = (C * W["volume"]).rolling(20, min_periods=10).mean().to_numpy()
    di = pd.Series(np.arange(len(dates)), index=dates); ti = pd.Series(np.arange(len(C.columns)), index=C.columns)
    df["di"] = di.reindex(df.date).to_numpy(); df["ti"] = ti.reindex(df.ticker).to_numpy()
    df = df.dropna(subset=["di", "ti"]); df["di"] = df.di.astype(int); df["ti"] = df.ti.astype(int)
    df["gap5n"] = gap5n[df.di, df.ti]; df["vol20n"] = vol20n[df.di, df.ti]; df["amt20n"] = amt20n[df.di, df.ti]
    df["h"] = h[df.di, df.ti]; df["r_path_chk"] = r_path[df.di, df.ti]
    df = df.dropna(subset=["h"])
    g.log(f"준비 {df.shape} ({time.time()-t0:.0f}s)")

    # 벤치마크: 유니버스 시총가중 지수
    base = g.build_masks(W, mktcap, foreign, managed, static)["all"]
    rr = g.pc(C); wgt = mktcap.where(base & rr.notna())
    bench = (1 + ((rr * wgt).sum(1) / wgt.sum(1)).fillna(0)).cumprod()
    bench = bench / bench.loc[dates >= g.EVAL_START].iloc[0]

    sigs = []
    for rid, f in RULES.items():
        m = f(df)
        s = df.loc[m, ["date", "ticker", "di", "ti", "gap5n", "vol20n", "amt20n", "rpath10", "h"]].rename(columns={"rpath10": "r_path"})
        s["rule"] = rid; s["prio"] = list(RULES).index(rid)
        sigs.append(s)
        g.log(f"{rid}: 신호 {len(s):,}셀 · {s.date.nunique()}일 · 평균 {len(s)/max(s.date.nunique(),1):.0f}종목/일")
    sig_all = pd.concat(sigs, ignore_index=True)
    sig_all = sig_all[sig_all.date >= g.EVAL_START]
    ev_dates = dates[dates >= g.EVAL_START]
    d0 = int(di[ev_dates[0]])

    SELECT = {"gap5최저": ("gap5n", 1), "vol20최저": ("vol20n", 1), "거래대금최대": ("amt20n", -1), "무작위": ("rand", 1)}
    rows, navs, trades_all = [], {}, []
    for selname, (kcol, sgn) in SELECT.items():
      for name, rules in COMBOS.items():
        s = sig_all[sig_all.rule.isin(rules)].copy()
        if kcol == "rand":
            rng = np.random.default_rng(0); s["key"] = rng.random(len(s))
        else:
            s["key"] = s[kcol] * sgn
        name = f"{name}·{selname}"
        nav, tr = simulate(s, dates, close, ti, en_arr)
        nav = nav.iloc[d0:]; nav = nav / nav.iloc[0]
        navs[name] = nav
        tr["combo"] = name; trades_all.append(tr)
        slot_use = (tr.h.sum() / SLOTS / len(nav)) if len(tr) else 0.0
        m = metrics(nav, tr, s.date.nunique(), len(nav), slot_use); m["combo"] = name
        for half, sel in [("A", nav.index.year < 2021), ("B", nav.index.year >= 2021)]:
            nv = nav[sel]; nv = nv / nv.iloc[0]; yrs = max((nv.index[-1] - nv.index[0]).days / 365.25, 0.5)
            m[f"cagr_{half}"] = nv.iloc[-1] ** (1 / yrs) - 1; m[f"mdd_{half}"] = (nv / nv.cummax() - 1).min()
        rows.append(m)
        g.log(f"{name}: 총 {m['total']:+.1%} CAGR {m['cagr']:+.1%} MDD {m['mdd']:.1%} 거래 {m['trades']} 승률 {m['win']:.2f} ({time.time()-t0:.0f}s)")
    bench = bench.loc[navs["R1·gap5최저"].index]
    bm = dict(combo="유니버스 지수", total=bench.iloc[-1] - 1, cagr=bench.iloc[-1] ** (1 / ((bench.index[-1] - bench.index[0]).days / 365.25)) - 1,
              mdd=(bench / bench.cummax() - 1).min())
    rows.append(bm)
    summ = pd.DataFrame(rows); summ.to_csv(OUT / "sim_summary.csv", index=False)
    trades = pd.concat(trades_all, ignore_index=True); trades.to_csv(OUT / "sim_trades.csv", index=False)
    navdf = pd.DataFrame(navs); navdf["bench"] = bench; navdf.to_csv(OUT / "sim_nav.csv")

    yearly = navdf.resample("YE").last().pct_change(fill_method=None)
    yearly.iloc[0] = navdf.resample("YE").last().iloc[0] - 1
    yearly.index = yearly.index.year

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "AppleGothic"; plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(14, 6))
    for c in navdf.columns:
        if c == "bench" or "vol20최저" in c or "거래대금최대" in c:
            ax.plot(navdf.index, navdf[c], lw=1.6 if c == "bench" else 1.0, label=c, color="#444" if c == "bench" else None)
    ax.set_yscale("log"); ax.grid(alpha=0.3, which="both"); ax.legend(ncol=6, fontsize=9, frameon=False)
    ax.set_title("중기 규칙 시뮬 — 10슬롯 · 배리어 청산 N=10 k=1 · 왕복 0.30% (2015-01=1)", loc="left")
    fig.savefig(OUT / "sim_equity.png", dpi=130, bbox_inches="tight")

    o = ["# 중기 규칙 시뮬 결과", f"{navdf.index[0].date()}~{navdf.index[-1].date()} · 10슬롯 동일비중 · 진입 t+1 시가 · 청산 배리어/만기 · 왕복 0.30%", ""]
    o.append("## 요약")
    cols = ["combo", "total", "cagr", "mdd", "sharpe", "trades", "win", "avg_r", "avg_h", "signal_days", "slot_use", "cagr_A", "mdd_A", "cagr_B", "mdd_B"]
    fmt = {"total": "{:+.1%}", "cagr": "{:+.1%}", "mdd": "{:.1%}", "sharpe": "{:.2f}", "trades": "{:,.0f}", "win": "{:.2f}", "avg_r": "{:+.2%}", "avg_h": "{:.1f}",
           "signal_days": "{:.1%}", "slot_use": "{:.1%}", "cagr_A": "{:+.1%}", "mdd_A": "{:.1%}", "cagr_B": "{:+.1%}", "mdd_B": "{:.1%}"}
    o.append("| " + " | ".join(cols) + " |"); o.append("|" + "---|" * len(cols))
    for _, r in summ.iterrows():
        o.append("| " + " | ".join((fmt.get(c, "{}").format(r[c]) if c in r and pd.notna(r[c]) else "—") for c in cols) + " |")
    o.append("\n## 연도별 수익률")
    o.append("| year | " + " | ".join(yearly.columns) + " |"); o.append("|" + "---|" * (len(yearly.columns) + 1))
    for y, r in yearly.iterrows():
        o.append(f"| {y} | " + " | ".join(f"{v:+.1%}" if pd.notna(v) else "—" for v in r) + " |")
    o.append("\n## 규칙×선택기준별 거래 통계")
    tstat = trades.groupby("combo").agg(n=("r_net", "size"), win=("r_net", lambda x: (x > 0).mean()), avg=("r_net", "mean"),
                                                                                med=("r_net", "median"), avg_h=("h", "mean"), worst=("r_net", "min"), best=("r_net", "max"))
    o.append("```\n" + tstat.round(4).to_string() + "\n```")
    (OUT / "sim_summary.md").write_text("\n".join(o), encoding="utf-8")
    g.log(f"완료 → {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
