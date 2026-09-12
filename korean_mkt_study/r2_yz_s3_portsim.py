"""r2_yz_s3_portsim.py — S3: R²·YZR 필터와 단기 청산 규칙을 스윙 포트 엔진(라이브 조건)으로 시뮬레이션.

backtest.py 정본 엔진은 건드리지 않는다. 시간청산·목표가 청산이 필요해 run_backtest 를
이 파일 안에 복제(run_backtest2)하고 두 파라미터를 얹었다. 기본값이면 원본과 동일 경로.

변형 (2026-09-12 Kane "gogo"):
  base            v1.2.2.3 그대로 (4조/30% · W1 · 청산③)
  xYZR_L          YZR10/120 하위 ⅓(조용한 눌림) 제외
  xR2H_YZRL       R²120 상위 ⅓ AND YZR 하위 ⅓ 제외 (S2 최악 셀)
  onlyYZR_H       YZR 상위 ⅓(시끄러운 눌림)만 매수
  base+T5         base + 5거래일 시간청산(종가)
  base+T5+G3      base + 5일 시간청산 + 목표가 +3% 청산
  onlyYZR_H+T5+G3 시끄러운 눌림만 + 5일 + 목표 +3%
분위 임계는 **기간 A(2014-21) 운영 유니버스 신호에서만** 산출해 전 기간에 고정 적용 (룩어헤드 회피).
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
import backtest as bt                                        # noqa: E402
from backtest import (EntryParams, ExitParams, PortfolioParams, UniverseParams, VolScaleParams,  # noqa: E402
                      Panel, Position, _stop_line, _yang_zhang, compute_signals, load_panel, metrics)
from r2_yz_events import rolling_r2_slope                    # noqa: E402

OUT = HERE / "out" / "r2yz"
SPLIT = pd.Timestamp("2022-01-01")


def run_backtest2(p: Panel, ep: EntryParams, xp: ExitParams = ExitParams(),
                  pp: PortfolioParams = PortfolioParams(),
                  start: str = "2015-01-01", end: str = "2026-06-30",
                  signal_mask: pd.DataFrame | None = None,
                  max_hold: int | None = None, target_pct: float | None = None,
                  cond_exit: tuple | None = None, stop_fn=None) -> dict:
    """cond_exit=(k, thr, mode): 최초 진입 D+k 에 종가 수익률 ≤ thr 이면 종가 청산.
    mode='once' 는 D+k 하루만 검사, 'after' 는 D+k 이후 매일 검사(승자도 되돌아오면 청산)."""
    sig = compute_signals(p, ep)
    signal, depth, rank = sig["signal"], sig["depth"], sig["rank"]
    if signal_mask is not None:
        signal = signal & signal_mask.reindex_like(signal).fillna(False)
    ps = signal.to_numpy()
    mask = (p.dates >= pd.Timestamp(start)) & (p.dates <= pd.Timestamp(end))
    idxs = np.flatnonzero(mask); dates = p.dates
    op = p.open.to_numpy(); hi = p.high.to_numpy(); lo = p.low.to_numpy(); cl = p.close.to_numpy()
    sg = signal.to_numpy(); sc = p.vol_scale.to_numpy(); rk = rank.to_numpy()
    tick = list(p.tickers); col = {t: j for j, t in enumerate(tick)}
    cash = pp.capital_krw; pending_cash = 0.0
    pos: dict[str, Position] = {}; last_sell_i: dict[str, int] = {}
    trades: list[dict] = []; equity: list[tuple] = []

    def sell(tk, i, fill, why):
        nonlocal pending_cash
        pz = pos[tk]
        net = fill * pz.shares * (1.0 - pp.sell_fee - pp.sell_tax)
        pending_cash += net
        trades.append(dict(date=dates[i], ticker=tk, side="SELL", shares=pz.shares, price=fill, amount=net,
                           pnl=net - pz.cost, hold_days=i - pz.first_entry_idx, ret=(net - pz.cost) / pz.cost, why=why))
        last_sell_i[tk] = i
        del pos[tk]

    for i in idxs:
        cash += pending_cash; pending_cash = 0.0
        # 1) 손절
        for tk in list(pos):
            j = col[tk]; pz = pos[tk]; line = (stop_fn or _stop_line)(pz, i, xp)
            if np.isnan(lo[i, j]):
                continue
            if lo[i, j] <= line:
                fill = min(op[i, j], line) if not np.isnan(op[i, j]) else line
                sell(tk, i, fill * (1.0 - pp.stop_slippage), "stop")
        # 1b) 목표가 / 시간 청산
        for tk in list(pos):
            j = col[tk]; pz = pos[tk]
            if np.isnan(cl[i, j]):
                continue
            avg = pz.cost / max(pz.shares, 1)
            if target_pct is not None and not np.isnan(hi[i, j]) and hi[i, j] >= avg * (1 + target_pct):
                fill = max(op[i, j], avg * (1 + target_pct)) if not np.isnan(op[i, j]) else avg * (1 + target_pct)
                sell(tk, i, fill, "target"); continue
            if max_hold is not None and (i - pz.first_entry_idx) >= max_hold:
                sell(tk, i, cl[i, j], "time"); continue
            if cond_exit is not None:
                k, thr, mode = cond_exit[:3]; basis = cond_exit[3] if len(cond_exit) > 3 else "first"
                d = i - (pz.first_entry_idx if basis == "first" else pz.entry_idx)   # 'last' = 최근매수일(불타기 리셋)
                if (d == k if mode == "once" else d >= k) and cl[i, j] / avg - 1 <= thr:
                    sell(tk, i, cl[i, j], "cond")
        # 2) 매수
        if i > 0:
            fired = np.flatnonzero(sg[i - 1])
            if fired.size:
                cands = []
                for j in fired:
                    tk = tick[j]
                    if np.isnan(op[i, j]) or op[i, j] <= 0:
                        continue
                    held = pos.get(tk)
                    if held is not None:
                        if held.slots >= pp.max_slots_per_stock or not ps[i - 1, j]:
                            continue
                    else:
                        ls = last_sell_i.get(tk)
                        if ls is not None and (i - ls) < pp.cooldown_trading_days:
                            continue
                    r = rk[i - 1, j]
                    cands.append((r if np.isfinite(r) else -1e18, j, tk))
                cands.sort(reverse=True)
                used = sum(q.slots for q in pos.values())
                for d_, j, tk in cands:
                    if used >= pp.max_positions:
                        break
                    price = op[i, j]; shares = int(pp.slot_krw // price)
                    if shares <= 0:
                        continue
                    cost = price * shares * (1.0 + pp.buy_fee)
                    if cost > cash:
                        shares = int(cash / (price * (1.0 + pp.buy_fee)))
                        if shares <= 0:
                            continue
                        cost = price * shares * (1.0 + pp.buy_fee)
                    cash -= cost; used += 1
                    s = sc[i - 1, j]; s = 1.0 if not np.isfinite(s) else float(s)
                    if tk in pos:
                        q = pos[tk]; q.slots += 1; q.shares += shares; q.cost += cost
                        q.entry_idx = i; q.peak = max(q.peak, price); q.scale = s
                    else:
                        pos[tk] = Position(tk, 1, shares, cost, i, i, price, s)
                    trades.append(dict(date=dates[i], ticker=tk, side="BUY", shares=shares, price=price,
                                       amount=cost, pnl=np.nan, hold_days=np.nan, ret=np.nan, why=""))
        # 3) peak·평가
        mv = 0.0
        for tk, q in pos.items():
            j = col[tk]
            if not np.isnan(hi[i, j]):
                q.peak = max(q.peak, hi[i, j])
            c = cl[i, j]
            mv += (c if not np.isnan(c) else q.cost / max(q.shares, 1)) * q.shares
        equity.append((dates[i], cash + pending_cash + mv, cash, mv, len(pos)))
    eq = pd.DataFrame(equity, columns=["date", "equity", "cash", "mv", "n_pos"]).set_index("date")
    return {"equity": eq, "trades": pd.DataFrame(trades), "params": ep}


def make_kane_stop(switch_day: int = 7, buffer: float = 0.0, basis: str = "last"):
    """Kane 2026-09-12 단순화 청산 (D+ = 최근 매수일 기본, 불타기 시 리셋 — 현행 엔진과 동일):
    D+0~(switch_day-1): 기준(피크>평단이면 피크, 아니면 평단) × (1 − clip(0.20×배율, 10~40%))
    D+switch_day~     : max(평단×(1−buffer), 피크 × (1 − clip(0.05×배율, 3~15%)))
    buffer = 평단선 완충 (0 이면 평단 그대로)."""
    def _fn(pz, i, xp):
        avg = pz.cost / max(pz.shares, 1)
        d = i - (pz.entry_idx if basis == "last" else pz.first_entry_idx)
        s_ = pz.scale if xp.vol_linked else 1.0
        if d < switch_day:
            pct = min(max(0.20 * s_, 0.10), 0.40)
            ref = pz.peak if pz.peak > avg else avg
            return ref * (1.0 - pct)
        pct = min(max(0.05 * s_, 0.03), 0.15)
        return max(avg * (1.0 - buffer), pz.peak * (1.0 - pct))
    return _fn


def period_profit(res, split=SPLIT):
    eq = res["equity"]["equity"]
    a = eq[eq.index < split]; b = eq[eq.index >= split]
    return (a.iloc[-1] / a.iloc[0] - 1, b.iloc[-1] / b.iloc[0] - 1)


def main():
    print("패널 로드...", flush=True)
    panel = load_panel(UniverseParams(), VolScaleParams())
    ep = EntryParams()
    base_sig = compute_signals(panel, ep)["signal"]

    # 특징 (신호일 t 종가까지)
    ly = np.log(panel.close)
    r2, _ = rolling_r2_slope(ly, 120)
    yzr = _yang_zhang(panel.open, panel.high, panel.low, panel.close, 10) / \
          _yang_zhang(panel.open, panel.high, panel.low, panel.close, 120)
    # 임계: 기간 A 신호일 값의 3분위 (운영 유니버스)
    sigA = base_sig & (base_sig.index < SPLIT).reshape(-1, 1)
    r2_vals = r2.where(sigA).stack().dropna(); yzr_vals = yzr.where(sigA).stack().dropna()
    r2_hi = r2_vals.quantile(2 / 3); yzr_lo, yzr_hi = yzr_vals.quantile(1 / 3), yzr_vals.quantile(2 / 3)
    print(f"임계(기간A 신호 {len(yzr_vals)}건): R2_120 상위⅓ ≥ {r2_hi:.3f} | YZR10/120 하위⅓ < {yzr_lo:.3f}, 상위⅓ ≥ {yzr_hi:.3f}")
    not_yzr_L = ~(yzr < yzr_lo)
    not_worst = ~((r2 >= r2_hi) & (yzr < yzr_lo))
    only_yzr_H = yzr >= yzr_hi

    variants = [("base v1.2.2.3", dict())]
    for k in (4, 5, 6, 7, 8, 9, 10):
        variants.append((f"E1 + D+{k} 종가검사 (최근매수일)", dict(signal_mask=not_worst, cond_exit=(k, 0.0, "once", "last"))))
    for k in (6, 7, 8):
        variants.append((f"E1 + D+{k} 종가검사 (최초진입일)", dict(signal_mask=not_worst, cond_exit=(k, 0.0, "once", "first"))))
    rows = []
    for name, kw in variants:
        res = run_backtest2(panel, ep, **kw)
        m = metrics(res); pa, pb = period_profit(res)
        tr = res["trades"]; sells = tr[tr.side == "SELL"]
        why = sells.why.value_counts(normalize=True).to_dict() if len(sells) else {}
        rows.append(dict(variant=name, final=m["final"] / 1e8, cagr=m["cagr"], sharpe=m["sharpe"], mdd=m["mdd"],
                         invested=m["invested"], n_buy=m["n_buy"], win=m["win_rate"], avg_hold=m["avg_hold"],
                         avg_ret=m["avg_ret"], ret_A=pa, ret_B=pb,
                         stop=why.get("stop", 0), target=why.get("target", 0), time=why.get("time", 0), cond=why.get("cond", 0)))
        print(bt.fmt(m, name) + f"  투자비중 {m['invested']:.0%}  A {pa:+.0%} / B {pb:+.0%}", flush=True)
        res["trades"].to_csv(OUT / f"s3_trades_{name.replace('+','_')}.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_excel(OUT / "s3_portsim_v2.xlsx", index=False)
    print("\nsaved", OUT / "s3_portsim_kday.xlsx")


if __name__ == "__main__":
    main()
