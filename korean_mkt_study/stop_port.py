# -*- coding: utf-8 -*-
"""stop_port.py — 손절 vs 버팀 4단계: 포트 단위 비교 (케인 2026-11-05).

대상   메가갤럭시(4조·외인30%, 월별) / 유니버스(LLV 멤버 217, 상시) — STOP_UNIV=swing|day
진입   스윙포트 진입신호 (W1 onset + R²·YZR 제외) — stop_research.load() 와 동일
운영   스윙포트 라이브 방식 — 1억 · 10슬롯×1,000만 · 종목당 2슬롯 · 쿨다운 1일 · 눌림 깊은 순 · t+1 시가 매수
       손절 = 종가 ≤ 선 → 익일 시가×99.5% (v1.2.3.8) · 비용 0.015/0.015+0.2% · 매도대금 T+1
       시간 청산 T거래일 = 보유 D+T 종가 확인 → 익일 시가 (슬리피지 없음). 버팀 = 시간 청산만.
사용   STOP_UNIV=swing|day python3 stop_port.py [mc [회수]]   (mc = 슬롯 비례 + 후보 무작위 반복 + WRST)
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stop_research as sr
import backtest as bt

CAP, SLOT, NSLOT, MAXPER, COOL = 1e8, 1e7, 10, 2, 1
BUY_FEE, SELL_FEE, TAX, SLIP = 0.00015, 0.00015, 0.002, 0.005

ETF_FEE = 0.00015   # 지수 ETF 대용 매매 수수료 (거래세 없음) — 2026-11-09 regime_exposure.py

def run_port(p, s, atr, rule, T=None, start="2015-01-01", end=None, slot_mode="fixed", rng=None, stop_prob=1.0, exit_mask=None, up=None, rule_up=None, prio=None, nslot=None, log=None, skip=None, hold_exit=None,
             gate_in=None, cap=None, liq=None, etf=None):
    """rule(k, entry_avg, peak_close, atr0) -> 손절선 (None 이면 손절 없음). T = 시간 청산 거래일 (None 이면 없음).
    slot_mode: fixed = 1,000만 고정(라이브) / prop = 전일 자산÷10 (투자비중 왜곡 제거용)
    rng: numpy Generator — 주면 같은 날 후보 순서를 무작위로 섞는다(경로 의존성 측정). stop_prob<1 = WRST(손절선 닿은 날 그 확률로만 청산, 아니면 다음 날 다시 판정)
    up: 날짜별 bool Series (True=상승국면) · rule_up: 판정일(i-1)이 상승국면이면 rule 대신 쓸 손절 규칙 (2026-11-05 stop_regime.py)
    skip: 날짜별 bool Series — 판정일(i-1)이 True 면 그날 손절 판정을 건너뛴다 (손절 유예, 2026-11-07 swing_shock_gates.py)
    hold_exit: f(j, first_i, i_prev) -> bool — 보유 중 사건 청산 (종가 판정 → 익일 시가×99.5%, reason "hold_exit"). 기본 None = 기존 동작 (2026-11-08 swing_surprise_test.py)
    ── 2026-11-09 regime_exposure.py (시장 층 노출 통제) — 전부 판정일(i-1) 값으로 오늘(i) 시가에 집행, 기본 None = 기존 동작
    gate_in: 날짜별 bool — False 면 그날 신규 매수 안 함
    cap: 날짜별 int — 그날 슬롯 상한 (기본 nslot). 슬롯 크기는 그대로 (fixed 1,000만 / prop 자산÷nslot)
    liq: 날짜별 bool — True 면 보유 전량 시가×99.5% 매도 (reason "liq")
    etf: dict(open=Series, close=Series, on=bool Series) — on 이 켜지는 판정일 다음 날 시가에 남는 현금 전부로 지수 매수,
         on 이 꺼진 판정일 다음 날 시가에 전량 매도(대금 T+1). 종목 매수 현금이 모자라면 그만큼 지수를 팔아 바로 쓴다 [가정].
         수수료 ETF_FEE, 거래세·슬리피지 없음. eq 에 'etf' 열(지수 평가액)이 붙는다"""
    gv = gate_in.reindex(p.dates).fillna(True).to_numpy(bool) if gate_in is not None else None
    cv = cap.reindex(p.dates).to_numpy(float) if cap is not None else None
    lv = liq.reindex(p.dates).fillna(False).to_numpy(bool) if liq is not None else None
    if etf is not None:
        eo = etf["open"].reindex(p.dates).to_numpy(float); ecl = etf["close"].reindex(p.dates).to_numpy(float)
        eon = etf["on"].reindex(p.dates).fillna(False).to_numpy(bool)
    units = 0.0
    skv = skip.reindex(p.dates).fillna(False).to_numpy() if skip is not None else None
    upv = up.reindex(p.dates).fillna(False).to_numpy() if up is not None else None
    dates = p.dates; mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end or sr.END))
    idxs = np.flatnonzero(mask)
    op = p.open.to_numpy(); cl = p.close.to_numpy(); at = atr.reindex(index=p.dates, columns=p.tickers).to_numpy()
    sig = s.reindex(index=p.dates, columns=p.tickers).fillna(False).to_numpy()
    xm = exit_mask.reindex(index=p.dates, columns=p.tickers).fillna(False).to_numpy() if exit_mask is not None else None   # 논리 붕괴 청산 (종가 판정 → 익일 시가×99.5%)
    # 눌림 깊이 (우선순위) — 40일 고점 대비
    depth = ((p.close.rolling(40, min_periods=20).max() - p.close) / p.close.rolling(40, min_periods=20).max()).to_numpy()
    if prio is not None:   # 우선순위 점수(클수록 먼저) — 기본은 눌림 깊이 (2026-11-05 entry_priority.py)
        depth = prio.reindex(index=p.dates, columns=p.tickers).to_numpy(float)
    NS = nslot or NSLOT
    tick = list(p.tickers); t2j = {t: j for j, t in enumerate(tick)}; cash = CAP; pending = 0.0
    pos = {}   # tk -> dict(slots, shares, cost, entry_i(최근), first_i, peak, atr0)
    last_sell = {}; trades = []; eq = []; equity_prev = CAP
    for i in idxs:
        cash += pending; pending = 0.0
        slot = SLOT if slot_mode == "fixed" else equity_prev / NS
        # 1) 청산 판정: 전일(i-1) 종가 기준 → 오늘 시가 체결
        for tk in list(pos):
            j = t2j[tk]; q = pos[tk]
            if i - 1 < q["entry_i"]: continue
            c_prev = cl[i - 1, j]
            if not np.isfinite(c_prev) or not np.isfinite(op[i, j]) or op[i, j] <= 0: continue
            k = i - 1 - q["entry_i"]; avg = q["cost"] / q["shares"]
            reason = "liq" if (lv is not None and lv[i - 1]) else None
            rr = rule_up if (upv is not None and upv[i - 1]) else rule
            if reason is None and rr is not None and not (skv is not None and skv[i - 1]):
                line = rr(k, avg, q["peak"], q["atr0"])
                if c_prev <= line and (stop_prob >= 1.0 or rng.random() < stop_prob): reason = "stop"
            if reason is None and xm is not None and xm[i - 1, j]: reason = "stop"
            if reason is None and hold_exit is not None and hold_exit(j, q["first_i"], i - 1): reason = "hold_exit"
            if reason is None and T is not None and (i - 1 - q["first_i"]) >= T: reason = "time"
            if reason:
                fill = op[i, j] * ((1 - SLIP) if reason in ("stop", "hold_exit", "liq") else 1.0)
                net = fill * q["shares"] * (1 - SELL_FEE - TAX); pending += net
                trades.append(dict(date=dates[i], ticker=tk, side="SELL", reason=reason, ret=net / q["cost"] - 1, hold=i - q["first_i"], pnl=net - q["cost"]))
                last_sell[tk] = i; del pos[tk]
        # 1b) 지수(ETF 대용) 청산: on 이 꺼진 판정일 다음 날 시가 전량 — 대금 T+1
        if etf is not None and units > 0 and i > 0 and not eon[i - 1] and np.isfinite(eo[i]):
            net = units * eo[i] * (1 - ETF_FEE); pending += net
            trades.append(dict(date=dates[i], ticker="ETF", side="SELL", reason="etf_off", ret=np.nan, hold=np.nan, pnl=np.nan)); units = 0.0
        # 2) 매수: 전일 신호 → 오늘 시가
        if i > 0 and (gv is None or gv[i - 1]):
            fired = np.flatnonzero(sig[i - 1]); cands = []
            for j in fired:
                tk = tick[j]
                if not np.isfinite(op[i, j]) or op[i, j] <= 0:
                    if log is not None: log.append((dates[i - 1], tk, "가격없음"))
                    continue
                if tk in pos and pos[tk]["slots"] >= MAXPER:
                    if log is not None: log.append((dates[i - 1], tk, "종목당2슬롯"))
                    continue
                if tk not in pos and tk in last_sell and i - last_sell[tk] < COOL:
                    if log is not None: log.append((dates[i - 1], tk, "쿨다운"))
                    continue
                d = depth[i - 1, j]; cands.append((d if np.isfinite(d) else -1e9, j, tk))
            if rng is not None: rng.shuffle(cands)
            else: cands.sort(reverse=True)
            used = sum(q["slots"] for q in pos.values())
            NSd = int(cv[i - 1]) if (cv is not None and np.isfinite(cv[i - 1])) else NS
            for _, j, tk in cands:
                if used >= NSd:
                    if log is not None: log.append((dates[i - 1], tk, "슬롯꽉참"))
                    continue
                price = op[i, j]
                need = slot * (1 + BUY_FEE)
                if etf is not None and units > 0 and cash < need and np.isfinite(eo[i]):   # 지수를 팔아 종목 매수 자금을 바로 댄다 [가정]
                    sell_u = min(units, (need - cash) / (eo[i] * (1 - ETF_FEE)))
                    cash += sell_u * eo[i] * (1 - ETF_FEE); units -= sell_u
                    if units < 1e-9: units = 0.0
                shares = int(min(slot, cash / (1 + BUY_FEE)) // price)
                if shares <= 0:
                    if log is not None: log.append((dates[i - 1], tk, "현금부족"))
                    continue
                if log is not None: log.append((dates[i - 1], tk, "체결(추가)" if tk in pos else "체결"))
                cost = price * shares * (1 + BUY_FEE); cash -= cost; used += 1
                a0 = at[i - 1, j]
                if tk in pos:
                    q = pos[tk]; q["slots"] += 1; q["shares"] += shares; q["cost"] += cost; q["entry_i"] = i; q["peak"] = max(q["peak"], price); q["atr0"] = a0
                else:
                    pos[tk] = dict(slots=1, shares=shares, cost=cost, entry_i=i, first_i=i, peak=price, atr0=a0)
                trades.append(dict(date=dates[i], ticker=tk, side="BUY", reason="", ret=np.nan, hold=np.nan, pnl=np.nan))
        # 2b) 지수(ETF 대용) 매수: on 이 켜진 첫 판정일 다음 날 시가에, 종목 매수 뒤 남은 현금 전부
        if etf is not None and i > 0 and eon[i - 1] and not (i > 1 and eon[i - 2]) and units == 0 and cash > 0 and np.isfinite(eo[i]):
            units = cash * (1 - ETF_FEE) / eo[i]; cash = 0.0
            trades.append(dict(date=dates[i], ticker="ETF", side="BUY", reason="etf_on", ret=np.nan, hold=np.nan, pnl=np.nan))
        # 3) peak(종가) 갱신 + 평가
        mv = 0.0
        for tk, q in pos.items():
            j = t2j[tk]; c = cl[i, j]
            if np.isfinite(c): q["peak"] = max(q["peak"], c); mv += c * q["shares"]
            else: mv += q["cost"]
        ev = units * ecl[i] if (etf is not None and units > 0) else 0.0
        equity_prev = cash + pending + mv + ev
        eq.append((dates[i], equity_prev, mv, len(pos), ev))
    eq = pd.DataFrame(eq, columns=["date", "equity", "mv", "n", "etf"]).set_index("date")
    if etf is None: eq = eq.drop(columns="etf")
    return eq, pd.DataFrame(trades)

def metrics(eq, tr):
    e = eq.equity; r = e.pct_change().dropna(); yrs = (e.index[-1] - e.index[0]).days / 365.25
    dd = e / e.cummax() - 1; sells = tr[tr.side == "SELL"] if len(tr) else pd.DataFrame()
    return dict(final=e.iloc[-1] / 1e8, cagr=(e.iloc[-1] / CAP) ** (1 / yrs) - 1, sharpe=r.mean() / r.std() * np.sqrt(252),
                mdd=dd.min(), invested=(eq.mv / eq.equity).mean(), n_trade=len(sells), win=(sells.ret > 0).mean() if len(sells) else np.nan,
                avg_hold=sells.hold.mean() if len(sells) else np.nan, avg_ret=sells.ret.mean() if len(sells) else np.nan, stop_share=(sells.reason == "stop").mean() if len(sells) else np.nan)

def period_returns(eq, bounds):
    out = {}
    for (a, b, lbl) in bounds:
        s = eq.equity.loc[a:b]
        if len(s) > 1: out[lbl] = s.iloc[-1] / s.iloc[0] - 1
    return out

RULES = {"버팀(시간청산만)": None, "현행③": sr.rule_current,
         "고정−5%": sr.mk_fixed(.05), "고정−10%": sr.mk_fixed(.10), "고정−15%": sr.mk_fixed(.15), "고정−20%": sr.mk_fixed(.20), "고정−30%": sr.mk_fixed(.30),
         "ATR×3": sr.mk_atr(3), "추적−20%": sr.mk_trail(.20)}
PERIODS = [("2015-01-01", "2017-12-31", "15-17"), ("2018-01-01", "2019-12-31", "18-19"), ("2020-01-01", "2021-12-31", "20-21"),
           ("2022-01-01", "2023-12-31", "22-23"), ("2024-01-01", "2026-06-30", "24-26.6"), ("2026-07-01", "2026-11-04", "26.7-11")]

def main():
    p, s, _, _, atr = sr.load()
    print(f"[유니버스 {sr.UNIV}] 종목 {len(p.tickers)}  신호 {int(s.to_numpy().sum())}  기간 2015-01-01~{sr.END}")
    rows = []; pr = {}; eqs = {}
    runs = [("현행③ (라이브, 시간청산 없음)", sr.rule_current, None)] + [(f"{n} +T60", f, 60) for n, f in RULES.items()] + [("버팀 +T120", None, 120), ("버팀 +T20", None, 20)]
    for lbl, f, T in runs:
        eq, tr = run_port(p, s, atr, f, T)
        m = metrics(eq, tr); m["변형"] = lbl; rows.append(m); pr[lbl] = period_returns(eq, PERIODS); eqs[lbl] = eq.equity
    t = pd.DataFrame(rows).set_index("변형")
    pd.set_option("display.width", 250)
    show = pd.DataFrame({"최종(억)": t.final.round(2), "CAGR%": (t.cagr * 100).round(2), "Sharpe": t.sharpe.round(2), "MDD%": (t.mdd * 100).round(1),
                         "투자비중%": (t.invested * 100).round(0), "청산수": t.n_trade, "승률%": (t.win * 100).round(1), "평균보유일": t.avg_hold.round(1), "손절비중%": (t.stop_share * 100).round(0)})
    print("\n== 포트 성과 (1억, 10슬롯×1,000만, 라이브 조건) ==\n", show.to_string())
    print("\n== 구간별 수익률 (%) ==\n", (pd.DataFrame(pr).T * 100).round(1).to_string())
    out = HERE / "out" / "stop_research" / ("port_" + sr.UNIV); out.mkdir(parents=True, exist_ok=True)
    show.to_csv(out / "summary.csv"); pd.DataFrame(eqs).to_csv(out / "equity.csv")

def main_mc(n_iter=50, seed=20261105):
    """슬롯 비례 + 후보 순서 무작위 n_iter 회 — 변형별 분포. WRST = 고정−15% 를 50% 확률로만 집행."""
    p, s, _, _, atr = sr.load()
    print(f"[유니버스 {sr.UNIV}] 종목 {len(p.tickers)}  신호 {int(s.to_numpy().sum())}  슬롯 비례(자산÷10) · 후보 무작위 {n_iter}회")
    runs = [("현행③ (시간청산 없음)", sr.rule_current, None, 1.0), ("버팀 +T60", None, 60, 1.0), ("버팀 +T120", None, 120, 1.0),
            ("현행③ +T60", sr.rule_current, 60, 1.0), ("고정−5% +T60", sr.mk_fixed(.05), 60, 1.0), ("고정−10% +T60", sr.mk_fixed(.10), 60, 1.0),
            ("고정−15% +T60", sr.mk_fixed(.15), 60, 1.0), ("WRST 고정−15%×50% +T60", sr.mk_fixed(.15), 60, 0.5),
            ("고정−20% +T60", sr.mk_fixed(.20), 60, 1.0), ("고정−30% +T60", sr.mk_fixed(.30), 60, 1.0), ("추적−20% +T60", sr.mk_trail(.20), 60, 1.0)]
    res = {lbl: [] for lbl, *_ in runs}; per = {lbl: [] for lbl, *_ in runs}
    for it in range(n_iter):
        for lbl, f, T, sp in runs:
            rng = np.random.default_rng(seed + it)
            eq, tr = run_port(p, s, atr, f, T, slot_mode="prop", rng=rng, stop_prob=sp)
            m = metrics(eq, tr); res[lbl].append(m); per[lbl].append(period_returns(eq, PERIODS))
    rows = []
    for lbl in res:
        d = pd.DataFrame(res[lbl])
        q = lambda c, x: d[c].quantile(x)
        rows.append({"변형": lbl, "CAGR중앙%": q("cagr", .5) * 100, "CAGR 10~90%": f"{q('cagr',.1)*100:.1f}~{q('cagr',.9)*100:.1f}",
                     "Sharpe중앙": q("sharpe", .5), "MDD중앙%": q("mdd", .5) * 100, "MDD 최악%": d.mdd.min() * 100,
                     "투자비중%": d.invested.median() * 100, "청산수": d.n_trade.median(), "승률%": d.win.median() * 100, "평균보유": d.avg_hold.median()})
    t = pd.DataFrame(rows).set_index("변형")
    pd.set_option("display.width", 250)
    print(f"\n== {n_iter}회 분포 ==\n", t.round(2).to_string())
    pr = pd.DataFrame({lbl: pd.DataFrame(per[lbl]).median() for lbl in per}).T * 100
    print("\n== 구간별 수익률 중앙값 (%) ==\n", pr.round(1).to_string())
    # 짝지은 비교: 같은 seed 에서 (변형 − 버팀T60) CAGR 차이의 분포
    base = np.array([m["cagr"] for m in res["버팀 +T60"]])
    print("\n== 같은 경로(seed)에서 버팀 +T60 대비 CAGR 차이 (%p): 중앙 / 10~90% / 버팀을 이긴 비율 ==")
    for lbl in res:
        if lbl == "버팀 +T60": continue
        dlt = (np.array([m["cagr"] for m in res[lbl]]) - base) * 100
        print(f"  {lbl:<24} {np.median(dlt):+5.2f}  {np.quantile(dlt,.1):+5.2f}~{np.quantile(dlt,.9):+5.2f}  {np.mean(dlt>0)*100:4.0f}%")
    out = HERE / "out" / "stop_research" / ("port_" + sr.UNIV); out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "mc_summary.csv"); pr.to_csv(out / "mc_periods.csv")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "mc":
        main_mc(int(sys.argv[2]) if len(sys.argv) > 2 else 50)
    else:
        main()
