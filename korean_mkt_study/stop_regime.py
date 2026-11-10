# -*- coding: utf-8 -*-
"""stop_regime.py — 손절 vs 버팀 5단계: 국면(사후 zigzag)·재난선·상승국면 짧은 손절 (케인 2026-11-05).

국면   KOSPI200 종가 zigzag 반전 20% (stop_segments.py 와 동일) — 사후 구분(오라클). 실시간 판정 아님.
질문   ① 국면을 알고 진입을 거르면(상승 국면 신호만) 얼마나 나아지나 = 국면 진입 필터의 상한
       ② 재난선 −25/−30/−35/−40% (+T60)
       ③ 상승 국면에서만 −10% 손절(빈 슬롯은 새 신호로 교체), 하락 국면은 버팀 (+T60)
운영   stop_port.run_port 그대로 (1억·10슬롯×1,000만·종목당 2슬롯·눌림 깊은 순·t+1 시가 · 손절 익일 시가×99.5%)
사용   STOP_UNIV=swing python3 stop_regime.py [tails|single|mc [회수]]
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stop_research as sr
import stop_port as sp

OUT = HERE / "out" / "stop_research" / ("port_" + sr.UNIV)

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
    pts.append((idx[ext], "H" if mode == "up" else "L"))
    return pts

def regime(dates):
    """날짜별 (up bool, 구간 라벨). stop_segments.py 와 같은 구간 경계 — 구간 시작일(전환점) 포함, 다음 전환점 미포함."""
    k = pd.read_parquet(HERE / "data_1104" / "kospi200_daily.parquet").Close.astype(float).sort_index()
    k.index = pd.to_datetime(k.index); k = k.loc["2014-06-01":]
    pts = zigzag(k, 0.20)
    pts = [p for p in pts if p[0] >= pd.Timestamp("2015-01-01")] if pts[0][0] < pd.Timestamp("2015-01-01") else pts
    if pts[0][0] > pd.Timestamp("2015-01-02"): pts = [(pd.Timestamp("2015-01-02"), "L")] + pts
    up = pd.Series(False, index=pd.DatetimeIndex(dates)); lab = pd.Series("", index=up.index, dtype=object)
    ends = [p[0] for p in pts[1:]] + [up.index[-1] + pd.Timedelta(days=1)]
    for (a, t), b in zip(pts, ends):
        m = (up.index >= a) & (up.index < b)
        if not m.any(): continue
        ka, kb = k.loc[:a].iloc[-1], k.loc[:min(b, k.index[-1])].iloc[-1]
        up[m] = (t == "L"); lab[m] = f"{a.strftime('%y-%m')}~{min(b, k.index[-1]).strftime('%y-%m')} {'상승' if t == 'L' else '하락'} {(kb/ka-1)*100:+.0f}%"
    return up, lab

def enrich(tr, p, up, lab):
    """SELL 행에 진입일·신호일 국면·보유 중 최저/최고(첫 매수 시가 대비 종가) 부착."""
    t = tr[tr.side == "SELL"].copy().reset_index(drop=True)
    dates = list(p.dates); tk = list(p.tickers); cl = p.close.to_numpy(); op = p.open.to_numpy()
    rows = []
    for _, r in t.iterrows():
        i = dates.index(pd.Timestamp(r.date)); f = i - int(r.hold); j = tk.index(r.ticker)
        path = cl[f:i, j] / op[f, j] - 1; sd = dates[f - 1]
        rows.append(dict(entry=dates[f], sig_date=sd, up=bool(up.get(sd, False)), seg=lab.get(sd, ""),
                         mae=np.nanmin(path) * 100, mfe=np.nanmax(path) * 100, fi=f, j=j))
    t = pd.concat([t, pd.DataFrame(rows)], axis=1); t["r"] = t.ret * 100
    return t

def tail_stats(g):
    r = g.r; w = r[r > 0]; l = r[r <= 0]
    return pd.Series({"건수": len(r), "승률%": (r > 0).mean() * 100, "평균%": r.mean(), "중앙%": r.median(),
                      "하위5%": r.quantile(.05), "하위10%": r.quantile(.10), "상위10%": r.quantile(.90), "상위5%": r.quantile(.95),
                      "최악%": r.min(), "최고%": r.max(), "이긴평균%": w.mean(), "진평균%": l.mean(),
                      "손익(백만)": g.pnl.sum() / 1e6, "총이익/총손실": g.pnl[g.pnl > 0].sum() / -g.pnl[g.pnl <= 0].sum(),
                      "보유중≤−20%비율": (g.mae <= -20).mean() * 100, "보유중≤−30%비율": (g.mae <= -30).mean() * 100,
                      "≤−20%후플러스": (g[g.mae <= -20].r > 0).mean() * 100 if (g.mae <= -20).any() else np.nan,
                      "보유중최저 중앙%": g.mae.median()})

def main_tails():
    p, s, _, _, atr = sr.load(); up, lab = regime(p.dates)
    eq, tr = sp.run_port(p, s, atr, None, 60); t = enrich(tr, p, up, lab)
    pd.set_option("display.width", 250)
    a = t.groupby(t.up.map({True: "상승 국면 진입", False: "하락 국면 진입"})).apply(tail_stats)
    a.loc["전체"] = tail_stats(t)
    print("== 버팀+T60 거래 284건 — 신호일 국면별 꼬리 ==\n", a.round(1).T.to_string())
    b = t.groupby("seg", sort=False).apply(tail_stats)[["건수", "승률%", "평균%", "중앙%", "하위10%", "상위10%", "최악%", "최고%", "손익(백만)", "보유중≤−20%비율"]]
    print("\n== 구간별 ==\n", b.round(1).to_string())
    for k in (30,):
        lo = t.nsmallest(k, "pnl"); hi = t.nlargest(k, "pnl")
        print(f"\n하위 {k}건 중 하락 국면 진입 {int((~lo.up).sum())}건 · 상위 {k}건 중 하락 국면 진입 {int((~hi.up).sum())}건 · 전체 하락 국면 진입 {int((~t.up).sum())}/{len(t)}")
    days_up = up.loc["2015-01-01":sr.END].mean() * 100
    print(f"상승 국면 거래일 비중 {days_up:.1f}%")
    OUT.mkdir(parents=True, exist_ok=True); a.to_csv(OUT / "regime_tails.csv"); b.to_csv(OUT / "regime_tails_segments.csv")
    t.drop(columns=["fi", "j"]).to_csv(OUT / "regime_trades_holdT60.csv", index=False)

def variants(s, up):
    sb = s.fillna(False).astype(bool); u = up.reindex(sb.index).fillna(False).astype(bool).to_numpy()[:, None]
    s_up = sb & u; s_dn = sb & ~u
    assert int(s_up.to_numpy().sum()) + int(s_dn.to_numpy().sum()) == int(sb.to_numpy().sum())
    V = [("버팀 +T60 (기준)", dict(sig=s, rule=None)),
         ("[국면] 상승 국면 신호만 진입", dict(sig=s_up, rule=None)),
         ("[국면] 하락 국면 신호만 진입", dict(sig=s_dn, rule=None))]
    for x in (.25, .30, .35, .40):
        V.append((f"[재난선] −{int(x*100)}%", dict(sig=s, rule=sr.mk_fixed(x))))
    V += [("[상승 −10%] 상승 국면 −10% · 하락 버팀", dict(sig=s, rule=None, rule_up=sr.mk_fixed(.10))),
          ("(참고) 전 국면 −10%", dict(sig=s, rule=sr.mk_fixed(.10)))]
    return V

def regime_ret(eq, up):
    r = eq.equity.pct_change().fillna(0); u = up.reindex(r.index).fillna(False).astype(bool)
    return (1 + r[u]).prod() - 1, (1 + r[~u]).prod() - 1

def run_v(p, atr, up, kw, **x):
    return sp.run_port(p, kw["sig"], atr, kw["rule"], 60, up=up if "rule_up" in kw else None, rule_up=kw.get("rule_up"), **x)

def main_single():
    p, s, _, _, atr = sr.load(); up, lab = regime(p.dates)
    rows = []; keep = {}
    for lbl, kw in variants(s, up):
        eq, tr = run_v(p, atr, up, kw); m = sp.metrics(eq, tr); ru, rd = regime_ret(eq, up)
        rows.append({"변형": lbl, "최종(억)": m["final"], "CAGR%": m["cagr"] * 100, "Sharpe": m["sharpe"], "MDD%": m["mdd"] * 100,
                     "투자비중%": m["invested"] * 100, "청산수": m["n_trade"], "승률%": m["win"] * 100, "거래평균%": m["avg_ret"] * 100,
                     "손절비중%": m["stop_share"] * 100, "상승국면 누적%": ru * 100, "하락국면 누적%": rd * 100})
        keep[lbl] = tr
    t = pd.DataFrame(rows).set_index("변형"); pd.set_option("display.width", 250)
    print("== 단일 경로 (라이브: 1,000만 고정 슬롯) ==\n", t.round(2).to_string())
    OUT.mkdir(parents=True, exist_ok=True); t.to_csv(OUT / "regime_single.csv")
    # 상승 −10% 로 잘린 거래: 버텼다면(첫 매수 +61거래일 시가) 어땠나
    tr = keep["[상승 −10%] 상승 국면 −10% · 하락 버팀"]; e = enrich(tr, p, up, lab); st = e[e.reason == "stop"].copy()
    op = p.open.to_numpy(); n = len(p.dates)
    st["버텼다면%"] = [((op[min(f + 61, n - 1), j] / op[f, j]) * (1 - sp.SELL_FEE - sp.TAX) / (1 + sp.BUY_FEE) - 1) * 100 for f, j in zip(st.fi, st.j)]
    print(f"\n상승 국면 −10% 손절 {len(st)}건: 실제 평균 {st.r.mean():.1f}% · 버텼다면 평균 {st['버텼다면%'].mean():.1f}% (중앙 {st['버텼다면%'].median():.1f}%) · "
          f"버텼다면 플러스 {(st['버텼다면%'] > 0).mean()*100:.0f}% · 버텼다면 +20%↑ {(st['버텼다면%'] >= 20).mean()*100:.0f}%")
    st.drop(columns=["fi", "j"]).to_csv(OUT / "regime_up10_stopped.csv", index=False)

def main_mc(n_iter=50, seed=20261105):
    p, s, _, _, atr = sr.load(); up, lab = regime(p.dates); V = variants(s, up)
    res = {lbl: [] for lbl, _ in V}
    for it in range(n_iter):
        for lbl, kw in V:
            rng = np.random.default_rng(seed + it)
            eq, tr = run_v(p, atr, up, kw, slot_mode="prop", rng=rng)
            m = sp.metrics(eq, tr); m["ru"], m["rd"] = regime_ret(eq, up); res[lbl].append(m)
        print(f"  {it+1}/{n_iter}", flush=True)
    base = np.array([m["cagr"] for m in res[V[0][0]]]); rows = []
    for lbl in res:
        d = pd.DataFrame(res[lbl]); dl = (d.cagr.to_numpy() - base) * 100
        rows.append({"변형": lbl, "CAGR중앙%": d.cagr.median() * 100, "CAGR10%": d.cagr.quantile(.1) * 100, "CAGR90%": d.cagr.quantile(.9) * 100,
                     "Sharpe중앙": d.sharpe.median(), "MDD중앙%": d.mdd.median() * 100, "MDD최악%": d.mdd.min() * 100, "승률%": d.win.median() * 100,
                     "투자비중%": d.invested.median() * 100, "상승국면 누적 중앙%": d.ru.median() * 100, "하락국면 누적 중앙%": d.rd.median() * 100,
                     "기준대비 CAGR차 중앙%p": np.median(dl), "차10%": np.quantile(dl, .1), "차90%": np.quantile(dl, .9), "기준 이긴 비율%": (dl > 0).mean() * 100})
    t = pd.DataFrame(rows).set_index("변형"); pd.set_option("display.width", 300)
    print(f"\n== 몬테카를로 {n_iter}회 (자산÷10 비례 슬롯 · 후보 순서 무작위 · 같은 seed 짝 비교) ==\n", t.round(2).to_string())
    OUT.mkdir(parents=True, exist_ok=True); t.to_csv(OUT / "regime_mc.csv")

if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "single"
    if a == "tails": main_tails()
    elif a == "mc": main_mc(int(sys.argv[2]) if len(sys.argv) > 2 else 50)
    else: main_single()
