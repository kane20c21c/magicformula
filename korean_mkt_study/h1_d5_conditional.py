"""h1_d5_conditional.py — 조건부 전환 2단계 (2026-10-05, Kane): 1단계에서 남은 두 후보를 포트 엔진에 넣는다.

후보  P1 거래량: 진입 후 5일 평균 거래량 ÷ 직전 20일 평균이 크면 D+6 에 좁히고(현행), 아니면 D+9 로 늦춤
      P2 급반등: 신호일 → D+5 RSI(14) 상승폭이 크면 D+6 에 좁히고, 아니면 D+9 로 늦춤
임계  기간 A(~2021) 진입 건의 분위에서만 산출해 전 기간 고정 (룩어헤드 회피). out/h1/d5_features.parquet 사용.
대조  S9 = 조건 없이 전부 D+9 (조건이 일괄 지연보다 나은지), R = 조건을 뒤집은 것(부호 확인).
판정  D+ 는 최근 매수일 기준(엔진과 동일). 지표는 D+5 종가까지만 사용. 0값은 결측 처리.
출력  out/h1/d5_conditional.csv
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "HillStorm"))
import backtest as bt
from backtest import EntryParams, _yang_zhang, compute_signals, metrics
from r2_yz_events import rolling_r2_slope
from r2_yz_s3_portsim import run_backtest2, period_profit, SPLIT
from h1_stop_width_sweep import D6, SIZINGS, OUT, clip
from h1_followup import boot
from h1_d5_features import load_fixed, rsi14


def cond_stop(flag: np.ndarray, col: dict, early_sw=5, late_sw=8, invert=False):
    """flag[i, j] = D+5 종가(i)에 '일찍 좁힐' 조건이 참인가. 참이면 전환 D+(early_sw+1), 아니면 D+(late_sw+1)."""
    def _fn(pz, i, xp):
        avg = pz.cost / max(pz.shares, 1); d = i - pz.entry_idx; s = pz.scale if xp.vol_linked else 1.0
        above = pz.peak > avg; sw = early_sw
        if d > early_sw:
            f = bool(flag[pz.entry_idx + 5, col[pz.ticker]])
            sw = early_sw if (f != invert) else late_sw
        if d <= sw:
            return (pz.peak if above else avg) * (1 - clip(0.20 * s, 0.10, 0.40))
        if above:
            return pz.peak * (1 - clip(0.05 * s, 0.03, 0.15))
        return avg * (1 - clip(0.10 * s, 0.05, 0.25))
    return _fn


def main():
    p = load_fixed(); ep = EntryParams(); O, H, L, C, V = p.open, p.high, p.low, p.close, p.volume
    base_sig = compute_signals(p, ep)["signal"]
    r2, _ = rolling_r2_slope(np.log(C), 120)
    yzr = _yang_zhang(O, H, L, C, 10) / _yang_zhang(O, H, L, C, 120)
    sigA = base_sig & (base_sig.index < SPLIT).reshape(-1, 1)
    nw = ~((r2 >= r2.where(sigA).stack().dropna().quantile(2 / 3)) & (yzr < yzr.where(sigA).stack().dropna().quantile(1 / 3)))
    col = {t: j for j, t in enumerate(p.tickers)}
    # D+5 종가(t) 기준 패널: 진입 후 5일(t-4..t) 평균 ÷ 신호일까지 20일(t-25..t-6) 평균 / RSI(t) − RSI(t-6)
    vr = np.log(V.rolling(5, min_periods=3).mean() / V.rolling(20, min_periods=10).mean().shift(6)).to_numpy()
    rsi = C.apply(rsi14); rc = (rsi - rsi.shift(6)).to_numpy()
    feat = pd.read_parquet(OUT / "d5_features.parquet"); A = feat[feat.date < SPLIT]
    v_hi, v_med = A.vol_ratio.quantile(2 / 3), A.vol_ratio.median(); r_hi = A.rsi_chg.quantile(2 / 3)
    print(f"임계(기간 A {len(A)}건): 거래량비 상위⅓ ≥ {np.exp(v_hi):.2f}배 · 중앙값 {np.exp(v_med):.2f}배 | RSI 상승폭 상위⅓ ≥ {r_hi:+.1f}")
    F = {"v_hi": np.nan_to_num(vr, nan=-9) >= v_hi, "v_med": np.nan_to_num(vr, nan=-9) >= v_med,
         "r_hi": np.nan_to_num(rc, nan=-99) >= r_hi}
    F["either"] = F["v_hi"] | F["r_hi"]; none = np.zeros_like(F["v_hi"]); allf = ~none
    VARS = [("V0 현행 (전부 D+6)", cond_stop(allf, col)),
            ("S9 전부 D+9", cond_stop(none, col)),
            ("P1 거래량 상위⅓만 D+6, 나머지 D+9", cond_stop(F["v_hi"], col)),
            ("P1m 거래량 중앙값 이상 D+6, 나머지 D+9", cond_stop(F["v_med"], col)),
            ("P1L 거래량 상위⅓만 D+6, 나머지 D+11", cond_stop(F["v_hi"], col, late_sw=10)),
            ("P2 급반등 상위⅓만 D+6, 나머지 D+9", cond_stop(F["r_hi"], col)),
            ("P3 둘 중 하나면 D+6, 나머지 D+9", cond_stop(F["either"], col)),
            ("R1 (뒤집기) 거래량 상위⅓만 D+9", cond_stop(F["v_hi"], col, invert=True)),
            ("R2 (뒤집기) 급반등 상위⅓만 D+9", cond_stop(F["r_hi"], col, invert=True))]
    rng = np.random.default_rng(20261005); rows = []
    for sz in SIZINGS:
        rets = {}; y0 = None
        for name, fn in VARS:
            res = run_backtest2(p, ep, pp=SIZINGS[sz], signal_mask=nw, cond_exit=D6, stop_fn=fn)
            m = metrics(res); pa, pb = period_profit(res); eq = res["equity"]["equity"]
            rets[name] = eq.pct_change().dropna().to_numpy()
            yr = eq.groupby(eq.index.year).agg(lambda s: s.iloc[-1] / s.iloc[0] - 1)
            y0 = yr if y0 is None else y0
            b0 = boot(rets[VARS[0][0]], rets[name], rng) if name != VARS[0][0] else {}
            b9 = boot(rets[VARS[1][0]], rets[name], rng) if name not in (VARS[0][0], VARS[1][0]) else {}
            rows.append(dict(sizing=sz, variant=name, final=m["final"] / 1e8, sharpe=m["sharpe"], mdd=m["mdd"], invested=m["invested"],
                             n_buy=m["n_buy"], win=m["win_rate"], avg_hold=m["avg_hold"], ret_A=pa, ret_B=pb, yrs_beat_V0=int((yr > y0 + 1e-9).sum()),
                             p_ret_vs_V0=b0.get("p_ret"), p_sharpe_vs_V0=b0.get("p_sharpe"), p_mdd_vs_V0=b0.get("p_mdd"),
                             p_ret_vs_S9=b9.get("p_ret"), p_sharpe_vs_S9=b9.get("p_sharpe")))
            r = rows[-1]; f = lambda x: "  —  " if x is None else f"{x:.3f}"
            print(f"[{sz}] {name:<34} 최종 {r['final']:.2f}억 Sh {r['sharpe']:.2f} MDD {r['mdd']:.1%} 투자 {r['invested']:.0%} 매수 {r['n_buy']} "
                  f"A {pa:+.0%}/B {pb:+.0%} 이긴해 {r['yrs_beat_V0']}/12 | 대V0 P수익 {f(r['p_ret_vs_V0'])} PSh {f(r['p_sharpe_vs_V0'])} PMDD악화 {f(r['p_mdd_vs_V0'])}"
                  f" | 대S9 P수익 {f(r['p_ret_vs_S9'])} PSh {f(r['p_sharpe_vs_S9'])}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "d5_conditional.csv", index=False, encoding="utf-8-sig")
    print("조건 참 비율(전체 셀): 거래량 상위⅓ %.1f%% · RSI 상위⅓ %.1f%%" % (100 * F["v_hi"].mean(), 100 * F["r_hi"].mean()))


if __name__ == "__main__":
    main()
