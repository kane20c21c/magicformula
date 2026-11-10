# -*- coding: utf-8 -*-
"""stop_hypotheses.py — '왜 버팀이 손절보다 나았나' 가설 검증 (케인 2026-11-05).
  ma200     ④-1 논리 기반 청산: 종가 < MA200(또는 −3% 완충) 이면 청산 vs 가격 손절 vs 버팀  (STOP_UNIV=swing|day)
  reversal  ①   단기 반전: 직전 20일 수익률 구간별 이후 20일 초과수익 (유니버스 동일가중 대비)
  quality   ②   유니버스 품질: 무작위 진입 버팀 T60 — 전 종목(동전주 제외) vs 유니버스 vs 메가갤럭시 (+④-2 무작위+손절)
사용: python3 stop_hypotheses.py ma200|reversal|quality [회수]
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import stop_research as sr, stop_port as sp, backtest as bt
from stop_hold_grid import SEGS
from stop_random_entry import random_signals

def summarize(res, seg=None):
    rows = []
    for k, v in res.items():
        d = pd.DataFrame(v)
        rows.append({"변형": k, "CAGR중앙%": d.cagr.median() * 100, "10~90%": f"{d.cagr.quantile(.1)*100:.1f}~{d.cagr.quantile(.9)*100:.1f}", "Sharpe": d.sharpe.median(),
                     "MDD중앙%": d.mdd.median() * 100, "MDD최악%": d.mdd.min() * 100, "투자비중%": d.invested.median() * 100, "청산수": d.n_trade.median(), "승률%": d.win.median() * 100, "평균보유": d.avg_hold.median()})
    pd.set_option("display.width", 240); print(pd.DataFrame(rows).set_index("변형").round(2).to_string())
    if seg:
        pr = pd.DataFrame({k: pd.DataFrame(v).median() for k, v in seg.items()}) * 100
        print("\n== KOSPI200 구간별 수익률 중앙값 (%) ==\n", pr.round(1).to_string())

def paired(res, base):
    b = np.array([m["cagr"] for m in res[base]])
    print(f"\n== 같은 경로에서 '{base}' 대비 CAGR 차이 (%p): 중앙 / 10~90% / 이긴 비율 ==")
    for k in res:
        if k == base: continue
        d = (np.array([m["cagr"] for m in res[k]]) - b) * 100
        print(f"  {k:<30} {np.median(d):+5.2f}  {np.quantile(d,.1):+5.2f}~{np.quantile(d,.9):+5.2f}  {np.mean(d>0)*100:4.0f}%")

def run_set(p, s, atr, runs, n, seed=20261105, sig_fn=None):
    res = {l: [] for l, *_ in runs}; seg = {l: [] for l, *_ in runs}
    for it in range(n):
        sig = s if sig_fn is None else sig_fn(it)
        for lbl, f, T, xm in runs:
            eq, tr = sp.run_port(p, sig, atr, f, T, slot_mode="prop", rng=np.random.default_rng(seed + it), exit_mask=xm)
            res[lbl].append(sp.metrics(eq, tr)); seg[lbl].append(sp.period_returns(eq, SEGS))
    return res, seg

def ma200(n):
    p, s, _, _, atr = sr.load()
    ma = p.close.rolling(200, min_periods=140).mean()
    below = (p.close < ma); below3 = (p.close < ma * 0.97)
    runs = [("버팀 +T60", None, 60, None), ("현행③ (라이브)", sr.rule_current, None, None), ("고정−10% +T60", sr.mk_fixed(.10), 60, None),
            ("MA200 이탈 +T60", None, 60, below), ("MA200 이탈 (시간청산 없음)", None, None, below), ("MA200 −3% 이탈 +T60", None, 60, below3),
            ("MA200 이탈 + 재난선−30% +T60", sr.mk_fixed(.30), 60, below), ("MA200 이탈 +T120", None, 120, below)]
    print(f"[유니버스 {sr.UNIV}] 종목 {len(p.tickers)} 신호 {int(s.to_numpy().sum())}  {n}회")
    res, seg = run_set(p, s, atr, runs, n); summarize(res, seg); paired(res, "버팀 +T60")

def reversal():
    out = {}
    for univ in ("swing", "day"):
        os.environ["STOP_UNIV"] = univ; sr.UNIV = univ
        p, s, _, _, atr = sr.load()
        cl = p.close.loc[sr.START:sr.END]; el = p.elig.loc[sr.START:sr.END]
        r20p = (cl / cl.shift(20) - 1).where(el); r20f = (cl.shift(-20) / cl - 1).where(el); r5p = (cl / cl.shift(5) - 1).where(el)
        ew = r20f.mean(axis=1); x = r20f.sub(ew, axis=0)
        for lbl, prior in (("직전 20일", r20p), ("직전 5일", r5p)):
            df = pd.DataFrame({"prior": prior.stack(), "x": x.stack(), "f": r20f.stack()}).dropna()
            bins = [-1, -.20, -.15, -.10, -.05, 0, .05, .10, .15, .20, 9]
            df["b"] = pd.cut(df.prior, bins)
            g = df.groupby("b", observed=True)
            t = pd.DataFrame({"표본": g.size(), "이후20일_평균%": g.f.mean() * 100, "초과_평균%": g.x.mean() * 100, "초과_중앙%": g.x.median() * 100, "초과플러스%": g.x.apply(lambda v: (v > 0).mean() * 100)})
            out[(univ, lbl)] = t
    pd.set_option("display.width", 200)
    for (univ, lbl), t in out.items():
        print(f"\n== [{ {'swing':'메가갤럭시','day':'유니버스'}[univ] }] {lbl} 수익률 구간 → 이후 20일 (유니버스 동일가중 대비 초과) ==")
        print(t.round(2).to_string())
    # 상관
    print()

def load_all_panel(end):
    """전 종목(동전주 제외: 전년 일평균 거래대금 ≥10억 & 전년말 시총 ≥1,000억, breadth_study 와 동일) — data_1104 prices + data_2015 meta"""
    px = pd.read_parquet(bt.DATA / "prices.parquet", columns=["close", "volume"]).reset_index(); px["ticker"] = px.ticker.astype(str)
    import breadth_study as bs
    el = bs.eligibility(px[px.date >= "2014-01-01"]); keys = el.loc[el.eligible, ["ticker", "year"]]
    full = pd.read_parquet(bt.DATA / "prices.parquet")
    full = full[(full.index.get_level_values("date") >= pd.Timestamp("2013-01-01")) & (full.index.get_level_values("date") <= pd.Timestamp(end))]
    ever = set(keys.ticker); full = full[full.index.get_level_values("ticker").isin(ever)]
    wide = {c: full[c].astype("float64").unstack("ticker").sort_index() for c in ("open", "high", "low", "close", "volume")}
    for c in ("open", "high", "low", "volume"): wide[c] = wide[c].where(wide[c] > 0)
    dates, tickers = wide["close"].index, wide["close"].columns
    yr = pd.DataFrame(False, index=dates, columns=tickers)
    kk = set(map(tuple, keys.to_numpy()))
    for t in tickers:
        ys = {y for (tk, y) in kk if tk == t}
        if ys: yr.loc[dates.year.isin(ys), t] = True
    elig = yr & wide["close"].notna()
    return bt.Panel(wide["open"], wide["high"], wide["low"], wide["close"], wide["volume"], elig, wide["close"] * np.nan, pd.DataFrame(1.0, index=dates, columns=tickers), dates, tickers, {})

def quality(n):
    pd.set_option("display.width", 240)
    results = {}
    for name, univ in (("메가갤럭시", "swing"), ("유니버스", "day"), ("전 종목(동전주 제외)", "all")):
        if univ == "all": p = load_all_panel(sr.END); s = None
        else: os.environ["STOP_UNIV"] = univ; sr.UNIV = univ; p, s, _, _, _ = sr.load()
        pc = p.close.shift(1); tr_ = pd.concat([(p.high - p.low), (p.high - pc).abs(), (p.low - pc).abs()]).groupby(level=0).max(); atr = tr_.rolling(14, min_periods=14).mean()
        print(f"\n[{name}] 종목 {len(p.tickers)}  일평균 편입 {p.elig.loc[sr.START:sr.END].sum(axis=1).mean():.0f}")
        runs = [("무작위 진입 버팀 T60", None, 60, None), ("무작위 진입 현행③", sr.rule_current, None, None), ("무작위 진입 고정−10% +T60", sr.mk_fixed(.10), 60, None)]
        res, seg = run_set(p, None, atr, runs, n, sig_fn=lambda it, p=p: random_signals(p, np.random.default_rng(20261105 * 7 + it)))
        summarize(res, seg); paired(res, "무작위 진입 버팀 T60"); results[name] = res
        # 무조건 60일 보유 수익 (편입 종목·전 일자)
        cl = p.close.loc[sr.START:sr.END]; el = p.elig.loc[sr.START:sr.END]
        r = (cl.shift(-60) / cl - 1).where(el).stack().dropna()
        print(f"  편입 종목·전 일자 60일 보유: 평균 {r.mean()*100:+.2f}% 중앙 {r.median()*100:+.2f}% 하위5% {r.quantile(.05)*100:+.1f}% 플러스 {(r>0).mean()*100:.1f}% (n={len(r):,})")
    print("\n== 요약: 무작위 진입 버팀 T60 CAGR 중앙 ==")
    for k, v in results.items(): print(f"  {k:<14} {pd.DataFrame(v['무작위 진입 버팀 T60']).cagr.median()*100:.2f}%")

if __name__ == "__main__":
    mode = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    {"ma200": lambda: ma200(n), "reversal": reversal, "quality": lambda: quality(n)}[mode]()
