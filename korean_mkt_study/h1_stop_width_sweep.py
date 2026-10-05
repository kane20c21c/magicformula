"""h1_stop_width_sweep.py — 가설 1 검증: 스윙 손절 폭을 넓히면(손절 비율↓·거래↓) 어떻게 되나 (2026-10-05)

배경: 가상계좌 손실 중 손절 33건이 전부 '피크 기준' 분기였고 19건이 D+6 당일(−20%×배율 → −5%×배율 전환)에 나왔다.
      → 검증 대상은 '평단 분기'가 아니라 D+6 전환 폭이다.
방법: r2_yz_s3_portsim.run_backtest2 (라이브 조건) 에 stop_fn 만 바꿔 끼운다. 정본 엔진은 건드리지 않는다.
      현행 = v1.2.3.4 (E1 필터 + 청산③ + D+6 종가검사·최근매수일). 변형은 실행 전에 고정 (아래 VARIANTS).
출력: out/h1/h1_sweep_{sizing}.csv
"""
from __future__ import annotations
import sys, warnings
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from backtest import (EntryParams, ExitParams, PortfolioParams, UniverseParams, VolScaleParams,   # noqa: E402
                      _yang_zhang, compute_signals, load_panel, metrics)
from r2_yz_events import rolling_r2_slope                                                        # noqa: E402
from r2_yz_s3_portsim import run_backtest2, period_profit, SPLIT                                 # noqa: E402

OUT = HERE / "out" / "h1"; OUT.mkdir(parents=True, exist_ok=True)
D6 = (6, 0.0, "once", "last")


def clip(x, lo, hi):
    return min(max(x, lo), hi)


def make_stop(late_up=0.05, cushion=0.0, step=None, switch_day=5, early=0.20, late_flat=0.10):
    """현행 4분기의 일반화.
    late_up  : D+switch+1~ 피크 분기 기본폭 (클립은 현행 0.03~0.15 를 같은 비율로 늘림)
    cushion  : 후기에 피크 < 평단×(1+cushion) 이면 '이익 구간 아님'으로 보고 평단 분기(평단 −late_flat×배율) 적용
    step     : (끝일, 중간폭) — D+switch+1~끝일 은 중간폭, 그 뒤 late_up (단계적 축소)
    """
    k = late_up / 0.05
    def _fn(pz, i, xp):
        avg = pz.cost / max(pz.shares, 1)
        d = i - pz.entry_idx
        s = pz.scale if xp.vol_linked else 1.0
        above = pz.peak > avg
        if d <= switch_day:
            return (pz.peak if above else avg) * (1 - clip(early * s, early / 2, early * 2))
        if (not above) or pz.peak < avg * (1 + cushion):
            return avg * (1 - clip(late_flat * s, 0.05, 0.25))
        if step is not None and d <= step[0]:
            kk = step[1] / 0.05
            return pz.peak * (1 - clip(step[1] * s, 0.03 * kk, 0.15 * kk))
        return pz.peak * (1 - clip(late_up * s, 0.03 * k, min(0.15 * k, 0.40)))
    return _fn


VARIANTS = [
    ("V0 현행 v1.2.3.4",                 dict(stop=dict(),                       d6=True)),
    ("A1 후기 피크폭 5→7.5%",            dict(stop=dict(late_up=0.075),          d6=True)),
    ("A2 후기 피크폭 5→10%",             dict(stop=dict(late_up=0.10),           d6=True)),
    ("A3 후기 피크폭 5→15%",             dict(stop=dict(late_up=0.15),           d6=True)),
    ("A4 후기 피크폭 5→20% (전환 없음)", dict(stop=dict(late_up=0.20),           d6=True)),
    ("B1 쿠션<3% 는 평단 분기",          dict(stop=dict(cushion=0.03),           d6=True)),
    ("B2 쿠션<5% 는 평단 분기",          dict(stop=dict(cushion=0.05),           d6=True)),
    ("B3 쿠션<8% 는 평단 분기",          dict(stop=dict(cushion=0.08),           d6=True)),
    ("C1 단계 축소 D+6~10 10% → 5%",     dict(stop=dict(step=(10, 0.10)),        d6=True)),
    ("C2 전환일 D+6 → D+9",              dict(stop=dict(switch_day=8),           d6=True)),
    ("C3 전환일 D+6 → D+11",             dict(stop=dict(switch_day=10),          d6=True)),
    ("D0 현행에서 D+6 종가검사 끔",      dict(stop=dict(),                       d6=False)),
    ("D2 A2 + D+6 종가검사 끔",          dict(stop=dict(late_up=0.10),           d6=False)),
    ("D3 B2 + D+6 종가검사 끔",          dict(stop=dict(cushion=0.05),           d6=False)),
    ("F1 조기폭 20→10% (조기 청산 대조)", dict(stop=dict(early=0.10),            d6=True)),
    ("F2 조기폭 20→15%",                 dict(stop=dict(early=0.15),             d6=True)),
]
SIZINGS = {"20x500": PortfolioParams(), "10x1000": replace(PortfolioParams(), max_positions=10, slot_krw=10_000_000.0)}


def yearly(eq):
    y = eq.groupby(eq.index.year).agg(["first", "last"])
    return y["last"] / y["first"] - 1


def main():
    which = sys.argv[1:] or list(SIZINGS)
    panel = load_panel(UniverseParams(), VolScaleParams()); ep = EntryParams()
    base_sig = compute_signals(panel, ep)["signal"]
    r2, _ = rolling_r2_slope(np.log(panel.close), 120)
    yzr = _yang_zhang(panel.open, panel.high, panel.low, panel.close, 10) / \
          _yang_zhang(panel.open, panel.high, panel.low, panel.close, 120)
    sigA = base_sig & (base_sig.index < SPLIT).reshape(-1, 1)
    r2_hi = r2.where(sigA).stack().dropna().quantile(2 / 3); yzr_lo = yzr.where(sigA).stack().dropna().quantile(1 / 3)
    not_worst = ~((r2 >= r2_hi) & (yzr < yzr_lo))
    for sz in which:
        rows, y0 = [], None
        for name, v in VARIANTS:
            res = run_backtest2(panel, ep, pp=SIZINGS[sz], signal_mask=not_worst,
                                cond_exit=D6 if v["d6"] else None, stop_fn=make_stop(**v["stop"]))
            m = metrics(res); pa, pb = period_profit(res)
            tr = res["trades"]; s = tr[tr.side == "SELL"]; st = s[s.why == "stop"]
            yr = yearly(res["equity"]["equity"])
            if y0 is None:
                y0 = yr
            big = s[s.ret >= 0.15]
            rows.append(dict(variant=name, final=m["final"] / 1e8, cagr=m["cagr"], sharpe=m["sharpe"], mdd=m["mdd"],
                             invested=m["invested"], n_buy=m["n_buy"], n_sell=len(s), win=m["win_rate"],
                             avg_hold=m["avg_hold"], avg_ret=m["avg_ret"], ret_A=pa, ret_B=pb,
                             stop_share=len(st) / max(len(s), 1), loss_stop_share=(st.ret < 0).sum() / max(len(s), 1),
                             loss_stop_avg=st[st.ret < 0].ret.mean(), cond_share=(s.why == "cond").mean(),
                             big_win_n=len(big), big_win_pnl=big.pnl.sum() / 1e8,
                             worst_trade=s.ret.min(), yrs_beat_V0=int((yr > y0 + 1e-9).sum()), yrs=len(yr)))
            r = rows[-1]
            print(f"[{sz}] {name:<32} 최종 {r['final']:.2f}억 Sharpe {r['sharpe']:.2f} MDD {r['mdd']:.1%} 투자 {r['invested']:.0%} "
                  f"매수 {r['n_buy']} 승률 {r['win']:.1%} 보유 {r['avg_hold']:.1f}일 A {pa:+.0%}/B {pb:+.0%} "
                  f"손실손절 {r['loss_stop_share']:.0%} V0이긴해 {r['yrs_beat_V0']}/{r['yrs']}", flush=True)
        pd.DataFrame(rows).to_csv(OUT / f"h1_sweep_{sz}.csv", index=False, encoding="utf-8-sig")
    print("saved", OUT)


if __name__ == "__main__":
    main()
