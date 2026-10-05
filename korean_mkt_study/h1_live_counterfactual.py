"""h1_live_counterfactual.py — 가설 1 후속 ③ (2026-10-05): 스윙 가상계좌 실제 진입(2026-07-06~10-02)에
현행 규칙(v1.2.3.4)과 전환일 변형을 일봉으로 다시 적용해 본다. 읽기 전용 — 운영 파일은 건드리지 않는다.

근사: 일봉(저가로 손절 감지, 체결 = min(시가, 손절선)×0.995 — backtest.py 와 같은 가정), 매수는 실제 체결가·수량,
      변동배율 = 신호일(매수 전 거래일) YZ_20 ÷ 0.031911 (LLV vol_scale.json 2026-08-24 분모 고정),
      슬롯·현금 제약 없음(변형이 더 오래 들어도 다른 매수를 막지 않는다고 가정). 비용: 매도 0.215%, 매수 0.015%.
"""
from __future__ import annotations
import csv, re, sys
from pathlib import Path
import numpy as np, pandas as pd
STOLAB = Path(__file__).resolve().parents[2]
SP = STOLAB / "StockPortfolio"; LLV = STOLAB / "longlivevault"
DENOM = 0.031911
OUT = Path(__file__).resolve().parent / "out" / "h1"; OUT.mkdir(parents=True, exist_ok=True)

px = pd.concat([pd.read_parquet(LLV / "data" / "ohlcv" / f, columns=["Date", "Ticker", "Open", "High", "Low", "Close", "YZ_20"])
                for f in ("core.parquet", "extend.parquet")]).drop_duplicates(["Date", "Ticker"])
px["Ticker"] = px["Ticker"].astype(str).str.zfill(6); px = px[px.Date >= "2026-06-01"].sort_values("Date")
DAYS = sorted(px.Date.unique()); DI = {d: i for i, d in enumerate(DAYS)}
BY = {t: g.set_index("Date") for t, g in px.groupby("Ticker")}
trades = list(csv.DictReader(open(SP / "data" / "paper" / "trades.csv", encoding="utf-8")))


def clip(x, lo, hi): return min(max(x, lo), hi)


def stop_line(peak, avg, age, s, switch):
    above = peak > avg
    if age <= switch:
        return (peak if above else avg) * (1 - clip(0.20 * s, 0.10, 0.40))
    if above:
        return peak * (1 - clip(0.05 * s, 0.03, 0.15))
    return avg * (1 - clip(0.10 * s, 0.05, 0.25))


def simulate(tk, buys, switch, check_day=6):
    """buys = [(date, price, qty)]. 반환: (실현+평가 손익, [매도 dict])"""
    g = BY.get(tk)
    if g is None: return None
    bmap = {}
    for d, p, q in buys: bmap.setdefault(pd.Timestamp(d), []).append((p, q))
    pos = None; cash = 0.0; sells = []; last_sell = None
    for d in [x for x in DAYS if x >= pd.Timestamp(buys[0][0])]:
        if d not in g.index: continue
        o, h, l, c = (float(g.at[d, k]) for k in ("Open", "High", "Low", "Close"))
        bought = False
        for p, q in bmap.get(d, []):
            if pos is None and last_sell is not None and DI[d] - DI[last_sell] < 1: continue   # 쿨다운 1일
            prev = DAYS[DI[d] - 1]; yz = float(g.at[prev, "YZ_20"]) if prev in g.index else np.nan
            s = yz / DENOM if np.isfinite(yz) else 1.0
            cash -= p * q * 1.00015
            if pos is None: pos = dict(qty=q, cost=p * q, peak=p, entry=d, first=d, s=s)
            else: pos.update(qty=pos["qty"] + q, cost=pos["cost"] + p * q, peak=max(pos["peak"], p), entry=d, s=s)
            bought = True
        if pos is None: continue
        avg = pos["cost"] / pos["qty"]; age = DI[d] - DI[pos["entry"]]
        sold = None
        if not bought:
            line = stop_line(pos["peak"], avg, age, pos["s"], switch)
            if l <= line: sold = (min(o, line) * 0.995, "stop")
            elif check_day and age == check_day and c <= avg: sold = (c, "d6_check")
        if sold:
            cash += sold[0] * pos["qty"] * (1 - 0.00215)
            sells.append(dict(ticker=tk, date=d, why=sold[1], age=age, ret=sold[0] / avg - 1, hold=DI[d] - DI[pos["first"]],
                              pnl=(sold[0] * (1 - 0.00215) - avg * 1.00015) * pos["qty"]))
            pos = None; last_sell = d; continue
        pos["peak"] = max(pos["peak"], h)
    mark = 0.0
    if pos is not None:
        lastc = float(g["Close"].iloc[-1]); mark = lastc * pos["qty"]
        sells.append(dict(ticker=tk, date=None, why="보유", age=None, ret=lastc / (pos["cost"] / pos["qty"]) - 1, hold=None,
                          pnl=(lastc - pos["cost"] / pos["qty"] * 1.00015) * pos["qty"]))
    return cash + mark, sells


def main():
    names = {}; buys = {}; act_cash = {}; act_qty = {}; act_sells = []
    for t in trades:
        tk = t["ticker"].zfill(6); names[tk] = t["name"]; q = float(t["qty"]); p = float(t["price"])
        if t["side"] == "BUY":
            buys.setdefault(tk, []).append((t["date"], p, q)); act_cash[tk] = act_cash.get(tk, 0) + float(t["net"]); act_qty[tk] = act_qty.get(tk, 0) + q
        else:
            act_cash[tk] = act_cash.get(tk, 0) + float(t["net"]); act_qty[tk] -= q
            n = dict(re.findall(r"(\w+)=([\d.]+)", t["note"])); act_sells.append(dict(ticker=tk, date=t["date"], ret=p / float(n["avg"]) - 1))
    actual = {tk: act_cash[tk] + act_qty[tk] * float(BY[tk]["Close"].iloc[-1]) for tk in buys if tk in BY}
    miss = [names[t] for t in buys if t not in BY]
    print("LLV 시세 없는 종목(제외):", miss)
    rules = {"현행 재현(D+6)": 5, "전환 D+7": 6, "전환 D+9": 8, "전환 D+11": 10, "전환 D+15": 14}
    res = {}; rows = []
    for nm, sw in rules.items():
        tot = 0; S = []
        for tk in buys:
            r = simulate(tk, buys[tk], sw)
            if r is None: continue
            tot += r[0]; S += r[1]; res[(nm, tk)] = r[0]
        cl = [x for x in S if x["why"] != "보유"]
        rows.append(dict(rule=nm, pnl=tot, n_sell=len(cl), n_stop=sum(x["why"] == "stop" for x in cl), n_d6=sum(x["why"] == "d6_check" for x in cl),
                         loss_stop=sum(x["why"] == "stop" and x["ret"] < 0 for x in cl), win=np.mean([x["ret"] > 0 for x in cl]),
                         avg_ret=np.mean([x["ret"] for x in cl]), avg_hold=np.mean([x["hold"] for x in cl]), open=len(S) - len(cl)))
        pd.DataFrame(S).assign(name=lambda d: d.ticker.map(names)).to_csv(OUT / f"live_cf_{nm}.csv", index=False, encoding="utf-8-sig")
    a_tot = sum(actual.values())
    print(f"\n실제 가상계좌: 손익 {a_tot:+,.0f}원 · 매도 {len(act_sells)}건 · 승률 {np.mean([s['ret']>0 for s in act_sells]):.1%} · 평균 {np.mean([s['ret'] for s in act_sells]):+.1%}")
    for r in rows:
        print(f"{r['rule']:<14} 손익 {r['pnl']:+,.0f}원 · 매도 {r['n_sell']}(손절 {r['n_stop']}·D+6검사 {r['n_d6']}) · 손실손절 {r['loss_stop']} · "
              f"승률 {r['win']:.1%} · 평균 {r['avg_ret']:+.1%} · 보유 {r['avg_hold']:.1f}일 · 미청산 {r['open']}")
    # 종목별: 현행 재현 vs 전환 D+9
    d = pd.DataFrame([dict(name=names[tk], actual=actual[tk], v0=res[("현행 재현(D+6)", tk)], d7=res[("전환 D+7", tk)],
                           d9=res[("전환 D+9", tk)], d11=res[("전환 D+11", tk)]) for tk in actual])
    d["d9_v0"] = d.d9 - d.v0
    print("\n[전환 D+9 − 현행 재현] 종목별 차이 상하위"); print(d.sort_values("d9_v0")[["name", "actual", "v0", "d9", "d9_v0"]].round(0).to_string(index=False))
    print("개선 종목", (d.d9_v0 > 1).sum(), "동일", (d.d9_v0.abs() <= 1).sum(), "악화", (d.d9_v0 < -1).sum())
    d.to_csv(OUT / "live_cf_by_ticker.csv", index=False, encoding="utf-8-sig")
    # 월별(매도일 기준) 손익
    for nm in ("현행 재현(D+6)", "전환 D+9"):
        s = pd.read_csv(OUT / f"live_cf_{nm}.csv"); s["m"] = s.date.fillna("보유").astype(str).str[:7]
        print(nm, s.groupby("m").pnl.sum().round(0).to_dict())


if __name__ == "__main__":
    main()
