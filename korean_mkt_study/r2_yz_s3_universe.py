"""r2_yz_s3_universe.py — E1×X2 조합의 유니버스 민감도 (Kane 2026-09-12): 4조/30% → 2조/20% 등."""
import sys, warnings; warnings.simplefilter("ignore")
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt
from backtest import EntryParams, UniverseParams, VolScaleParams, _yang_zhang, compute_signals, load_panel, metrics
from r2_yz_events import rolling_r2_slope
from r2_yz_s3_portsim import run_backtest2, period_profit, SPLIT, OUT

START = sys.argv[1] if len(sys.argv) > 1 else "2015-01-01"
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--start", default="2015-01-01"); ap.add_argument("--end", default="2026-06-30"); ap.add_argument("--data", default="data")
args = ap.parse_args()
bt.DATA = Path(__file__).resolve().parent / args.data          # data_ext 로 전환 가능 (build_data_ext.py)
START, END = args.start, args.end
rows = []
for cap, fo in [(4e12, 30), (4e12, 20), (2e12, 30), (2e12, 20)]:
    label = f"{cap/1e12:.0f}조/{fo}%"
    panel = load_panel(UniverseParams(mktcap_min_krw=cap, foreign_min_pct=fo), VolScaleParams(), end=END)
    n_univ = panel.elig.sum(axis=1)
    sig = compute_signals(panel, EntryParams())["signal"]
    r2, _ = rolling_r2_slope(np.log(panel.close), 120)
    yzr = _yang_zhang(panel.open, panel.high, panel.low, panel.close, 10) / _yang_zhang(panel.open, panel.high, panel.low, panel.close, 120)
    sigA = sig & (sig.index < SPLIT).reshape(-1, 1)
    r2_hi = r2.where(sigA).stack().dropna().quantile(2 / 3); yzr_lo = yzr.where(sigA).stack().dropna().quantile(1 / 3)
    not_worst = ~((r2 >= r2_hi) & (yzr < yzr_lo))
    print(f"\n[{label}] 월평균 편입 {n_univ[n_univ.index>=START].mean():.0f}종목 (2026-06: {int(n_univ.iloc[-1])}) | 임계 R2≥{r2_hi:.3f} YZR<{yzr_lo:.3f}", flush=True)
    for name, kw in [("base", {}), ("E1×X2", dict(signal_mask=not_worst, cond_exit=(7, 0.0, "once")))]:
        res = run_backtest2(panel, EntryParams(), start=START, **kw); m = metrics(res); pa, pb = period_profit(res, split=pd.Timestamp("2026-01-01"))
        print(bt.fmt(m, f"{label} {name}") + f"  투자비중 {m['invested']:.0%}  A {pa:+.0%} / B {pb:+.0%}", flush=True)
        rows.append(dict(univ=label, n_univ=n_univ[n_univ.index>='2015'].mean(), variant=name, final=m["final"]/1e8, cagr=m["cagr"], sharpe=m["sharpe"], mdd=m["mdd"],
                         invested=m["invested"], n_buy=m["n_buy"], win=m["win_rate"], avg_hold=m["avg_hold"], ret_A=pa, ret_B=pb))
pd.DataFrame(rows).to_excel(OUT / f"s3_universe_{START[:4]}.xlsx", index=False)
