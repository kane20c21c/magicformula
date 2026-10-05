"""h1_followup.py — 가설 1 후속 (2026-10-05): ① 블록 부트스트랩 ② 전환일 스윕 ④ −100% 거래 원인
h1_stop_width_sweep.py 의 make_stop / 마스크를 그대로 쓴다. 출력: out/h1/"""
from __future__ import annotations
import sys, warnings
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from backtest import (EntryParams, PortfolioParams, UniverseParams, VolScaleParams, _yang_zhang,
                      compute_signals, load_panel, metrics)
from r2_yz_events import rolling_r2_slope
from r2_yz_s3_portsim import run_backtest2, period_profit, SPLIT
from h1_stop_width_sweep import make_stop, D6, SIZINGS, OUT

BLOCK, REPS, SEED = 20, 5000, 20261005


def sharpe(r):
    s = r.std(axis=-1)
    return r.mean(axis=-1) / s * np.sqrt(252)


def mdd_of(r):
    eq = np.cumprod(1 + r, axis=-1)
    return (eq / np.maximum.accumulate(eq, axis=-1) - 1).min(axis=-1)


def boot(r0, r1, rng):
    """원형 블록 부트스트랩 — 두 수익률 열을 같은 블록으로 뽑는다(짝 유지)."""
    n = len(r0); nb = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n, size=(REPS, nb))
    idx = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(REPS, -1)[:, :n] % n
    a, b = r0[idx], r1[idx]
    d_mean = (b.mean(1) - a.mean(1)) * 252
    d_sh = sharpe(b) - sharpe(a)
    d_mdd = mdd_of(b) - mdd_of(a)
    return dict(p_ret=(d_mean <= 0).mean(), ret_lo=np.quantile(d_mean, .05), ret_hi=np.quantile(d_mean, .95),
                p_sharpe=(d_sh <= 0).mean(), sh_lo=np.quantile(d_sh, .05), sh_hi=np.quantile(d_sh, .95),
                p_mdd=(d_mdd <= 0).mean())


def main():
    panel = load_panel(UniverseParams(), VolScaleParams()); ep = EntryParams()
    base_sig = compute_signals(panel, ep)["signal"]
    r2, _ = rolling_r2_slope(np.log(panel.close), 120)
    yzr = _yang_zhang(panel.open, panel.high, panel.low, panel.close, 10) / \
          _yang_zhang(panel.open, panel.high, panel.low, panel.close, 120)
    sigA = base_sig & (base_sig.index < SPLIT).reshape(-1, 1)
    r2_hi = r2.where(sigA).stack().dropna().quantile(2 / 3); yzr_lo = yzr.where(sigA).stack().dropna().quantile(1 / 3)
    nw = ~((r2 >= r2_hi) & (yzr < yzr_lo))

    def run(sz, **stop):
        return run_backtest2(panel, ep, pp=SIZINGS[sz], signal_mask=nw, cond_exit=D6, stop_fn=make_stop(**stop))

    rng = np.random.default_rng(SEED)
    # ② 전환일 스윕 (switch_day = 마지막 조기일. 전환일 = switch_day+1) + ① 부트스트랩
    rows = []
    for sz in SIZINGS:
        base = run(sz); r0 = base["equity"]["equity"].pct_change().dropna().to_numpy()
        y0 = base["equity"]["equity"].groupby(base["equity"].index.year).agg(lambda s: s.iloc[-1] / s.iloc[0] - 1)
        cands = [(f"전환일 D+{k+1}", dict(switch_day=k)) for k in (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 19)]
        cands += [("C1 단계 D+6~10 10%", dict(step=(10, 0.10))), ("A1 후기폭 7.5%", dict(late_up=0.075)),
                  ("B2 쿠션<5%", dict(cushion=0.05))]
        for name, st in cands:
            res = run(sz, **st); m = metrics(res); pa, pb = period_profit(res)
            r1 = res["equity"]["equity"].pct_change().dropna().to_numpy()
            yr = res["equity"]["equity"].groupby(res["equity"].index.year).agg(lambda s: s.iloc[-1] / s.iloc[0] - 1)
            b = boot(r0, r1, rng) if st != dict(switch_day=5) else {}
            rows.append(dict(sizing=sz, variant=name, final=m["final"] / 1e8, sharpe=m["sharpe"], mdd=m["mdd"],
                             invested=m["invested"], n_buy=m["n_buy"], avg_hold=m["avg_hold"], ret_A=pa, ret_B=pb,
                             yrs_beat=int((yr > y0 + 1e-9).sum()), **b))
            r = rows[-1]
            print(f"[{sz}] {name:<20} 최종 {r['final']:.2f}억 Sh {r['sharpe']:.2f} MDD {r['mdd']:.1%} 투자 {r['invested']:.0%} "
                  f"A {pa:+.0%}/B {pb:+.0%} 이긴해 {r['yrs_beat']}/12 | "
                  + (f"P(수익차≤0) {b['p_ret']:.3f} [{b['ret_lo']:+.2%},{b['ret_hi']:+.2%}]/년  P(Sharpe차≤0) {b['p_sharpe']:.3f} "
                     f"[{b['sh_lo']:+.2f},{b['sh_hi']:+.2f}]  P(MDD 악화) {b['p_mdd']:.3f}" if b else "기준"), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "h1_switch_sweep_boot.csv", index=False, encoding="utf-8-sig")

    # ④ −100% 거래
    tr = run("20x500")["trades"]; s = tr[(tr.side == "SELL") & (tr.ret < -0.5)]
    print("\n[−50% 이하 거래]"); print(s.to_string(index=False))
    for _, x in s.iterrows():
        j = list(panel.tickers).index(x.ticker); i = list(panel.dates).index(x.date)
        w = slice(max(0, i - 4), i + 3)
        print(x.ticker, pd.DataFrame(dict(open=panel.open.iloc[w, j], high=panel.high.iloc[w, j],
                                          low=panel.low.iloc[w, j], close=panel.close.iloc[w, j])).to_string())


if __name__ == "__main__":
    main()
