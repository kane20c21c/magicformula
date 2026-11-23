"""선행 3종목 → TIGER200 / KODEX 인버스 슬롯 이동 시뮬레이션, 2026Q2 (2026-11-23 케인 설계)

선행 종목 = RFHIC(218410)·LG생활건강(051900)·피에스케이홀딩스(031980) — 2026Q1 까지 균형적중률로 선정.
신호(t 종가): 종목별 10거래일 수익률 부호.
  3개 모두 + → 다음 날 시가에 롱(TIGER200) 2슬롯 매수 · 숏(KODEX 인버스) 2슬롯 매도
  정확히 2개 + → 롱 1 매수 · 숏 1 매도   /  정확히 2개 − → 숏 1 매수 · 롱 1 매도
  3개 모두 − → 숏 2 매수 · 롱 2 매도
시작: 2026-03-31 종가로 롱 10슬롯 + 숏 10슬롯 (슬롯 = 200만원, 총 4,000만원, 시작 매수 수수료 없음 — 케인 '이론상 0%' 기준)
슬롯 이동 = 매도할 쪽에서 200만원어치(정수 주, 남은 수량이 적으면 전량) 시가 매도 → 그 대금으로 반대쪽 정수 주 시가 매수.
수수료 = 거래당 0.13% (매도·매수 각각). 마지막 신호 = 6/29 종가(6/30 시가 체결). 6/30 종가에 전량 매도(수수료 0.13%).
비교: ① 매매 없음 ② 케인 규칙 ③ 3개 일치일만 매매(2개 일치는 쉼)
자료: 종목·KOSPI200 = data_2015, ETF = LLV panel(102110)·tickers/114800 (새 수집 없음).
"""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "out", "k200_quarterly")
LLV = os.path.expanduser("~/mnt/LongLiveVault/data/ohlcv")
LEAD = {"218410": "RFHIC", "051900": "LG생활건강", "031980": "피에스케이홀딩스"}
SLOT, FEE, START, END = 2_000_000, 0.0013, "2026-03-31", "2026-06-30"

px = pd.read_parquet(f"{ROOT}/data_2015/prices.parquet", columns=["close"])["close"].unstack("ticker")[list(LEAD)]
r10 = np.log(px / px.shift(10)).rename(columns=LEAD)
L = pd.read_parquet(f"{LLV}/panel.parquet", columns=["Date", "Ticker", "Open", "Close"]); L = L[L.Ticker == "102110"].set_index("Date")[["Open", "Close"]]
S = pd.read_parquet(f"{LLV}/tickers/114800.parquet").set_index("Date")[["Open", "Close"]]
k = pd.read_parquet(f"{ROOT}/data_2015/kospi200_daily.parquet")["Close"]; k.index = pd.to_datetime(k.index)
days = L.loc[START:END].index
assert len(days) == len(S.loc[START:END]) and (S.loc[START:END].index == days).all()

def signal(d):
    s = np.sign(r10.loc[d]); up, dn = int((s > 0).sum()), int((s < 0).sum())
    return {3: 2, 2: 1}.get(up, 0) if up >= 2 else -{3: 2, 2: 1}.get(dn, 0) if dn >= 2 else 0, up, dn

def run(rule):
    p0L, p0S = L.Close[START], S.Close[START]
    sh = {"L": 10 * SLOT // p0L, "S": 10 * SLOT // p0S}; cash = 20 * SLOT - sh["L"] * p0L - sh["S"] * p0S
    fees, log = 0.0, []
    for i in range(1, len(days)):
        d, prev = days[i], days[i - 1]
        mv, up, dn = signal(prev)
        if rule == "none": mv = 0
        if rule == "three" and abs(mv) == 1: mv = 0
        o = {"L": L.Open[d], "S": S.Open[d]}
        sell, buy = ("S", "L") if mv > 0 else ("L", "S")
        for _ in range(abs(mv)):
            q = min(sh[sell], int(SLOT // o[sell]))
            if q <= 0: break
            amt = q * o[sell]; f = amt * FEE; sh[sell] -= q; cash += amt - f; fees += f
            qb = int(cash // (o[buy] * (1 + FEE))); amtb = qb * o[buy]; fb = amtb * FEE
            sh[buy] += qb; cash -= amtb + fb; fees += fb
        val = sh["L"] * L.Close[d] + sh["S"] * S.Close[d] + cash
        log.append(dict(date=d.date(), signal_up=up, signal_dn=dn, move=mv, long_sh=sh["L"], short_sh=sh["S"],
                        long_val=sh["L"] * L.Close[d], short_val=sh["S"] * S.Close[d], cash=cash, total=val,
                        k200_ret=k.get(d) / k.get(prev) - 1 if d in k.index and prev in k.index else np.nan))
    fin = sh["L"] * L.Close[END] + sh["S"] * S.Close[END]; fsell = fin * FEE; fees += fsell
    final = fin - fsell + cash
    return final, fees, pd.DataFrame(log)

res = {}
for rule, nm in [("none", "① 매매 없음"), ("kane", "② 케인 규칙"), ("three", "③ 3개 일치일만")]:
    final, fees, lg = run(rule); res[nm] = (final, fees, lg)
    print(f"{nm}: 최종 {final:,.0f}원  수익 {final - 20 * SLOT:+,.0f}원 ({final / (20 * SLOT) - 1:+.2%})  수수료 {fees:,.0f}원  "
          f"슬롯이동일 {(lg.move != 0).sum()}  최대 롱 {lg.long_val.max() / SLOT:.1f}슬롯  최소 롱 {lg.long_val.min() / SLOT:.1f}슬롯")
lg = res["② 케인 규칙"][2]; lg.to_csv(f"{OUT}/sim_2026q2_kane_daily.csv", index=False, encoding="utf-8-sig")
res["③ 3개 일치일만"][2].to_csv(f"{OUT}/sim_2026q2_three_daily.csv", index=False, encoding="utf-8-sig")
# 신호 적중 (Q2): 신호일 t → 다음 날 KOSPI200
ev = []
for i in range(1, len(days)):
    mv, up, dn = signal(days[i - 1]); ret = k[days[i]] / k[days[i - 1]] - 1
    ev.append((mv, ret))
ev = pd.DataFrame(ev, columns=["mv", "ret"]); ev = ev[ev.ret != 0]
for nm, m in [("3개 일치", ev.mv.abs() == 2), ("정확히 2개", ev.mv.abs() == 1), ("전체", ev.mv != 0)]:
    e = ev[m]; h = np.sign(e.mv) == np.sign(e.ret); upd = e.ret > 0
    bal = 0.5 * (h[upd].mean() + h[~upd].mean())
    print(f"{nm}: {len(e)}일 적중 {h.mean():.1%} 균형 {bal:.1%}  (상승예측 {int((e.mv>0).sum())}일 / 하락예측 {int((e.mv<0).sum())}일)")
print("Q2 지수 상승일 비율", f"{(ev.ret > 0).mean():.1%}", " KOSPI200 Q2", f"{k[END] / k[START] - 1:+.2%}",
      " TIGER200", f"{L.Close[END] / L.Close[START] - 1:+.2%}", " 인버스", f"{S.Close[END] / S.Close[START] - 1:+.2%}")
# 선행 3종목 개별 Q2 균형적중률
for c in LEAD.values():
    e = pd.DataFrame({"s": np.sign(r10[c].reindex(days[:-1]).values), "y": np.sign([k[days[i]] / k[days[i - 1]] - 1 for i in range(1, len(days))])})
    e = e[(e.s != 0) & (e.y != 0)]; h = e.s == e.y; print(f"  {c}: 적중 {h.mean():.1%} 균형 {0.5 * (h[e.y > 0].mean() + h[e.y < 0].mean()):.1%}")

# ── 운 비교: 같은 슬롯 이동 횟수·크기를 날짜만 무작위로 섞어 1,000회 (신호가 '언제' 나왔는지만 지운다)
def run_moves(moves):
    p0L, p0S = L.Close[START], S.Close[START]
    sh = {"L": 10 * SLOT // p0L, "S": 10 * SLOT // p0S}; cash = 20 * SLOT - sh["L"] * p0L - sh["S"] * p0S
    for i in range(1, len(days)):
        d = days[i]; mv = moves[i - 1]; o = {"L": L.Open[d], "S": S.Open[d]}
        sell, buy = ("S", "L") if mv > 0 else ("L", "S")
        for _ in range(abs(mv)):
            q = min(sh[sell], int(SLOT // o[sell]))
            if q <= 0: break
            amt = q * o[sell]; sh[sell] -= q; cash += amt * (1 - FEE)
            qb = int(cash // (o[buy] * (1 + FEE))); sh[buy] += qb; cash -= qb * o[buy] * (1 + FEE)
    return (sh["L"] * L.Close[END] + sh["S"] * S.Close[END]) * (1 - FEE) + cash
rng = np.random.default_rng(20261123)
for nm in ["② 케인 규칙", "③ 3개 일치일만"]:
    mv = res[nm][2].move.values; real = res[nm][0]
    null = np.array([run_moves(rng.permutation(mv)) for _ in range(1000)])
    print(f"{nm}: 실제 {real / (20 * SLOT) - 1:+.1%} | 날짜 섞기 1,000회 중앙값 {np.median(null) / (20 * SLOT) - 1:+.1%}, "
          f"10~90% [{np.quantile(null, .1) / (20 * SLOT) - 1:+.1%}, {np.quantile(null, .9) / (20 * SLOT) - 1:+.1%}], 실제 이상 비율 {(null >= real).mean():.1%}")
print("참고 ④ 시작부터 롱 20슬롯(TIGER200만):", f"{(20 * SLOT // L.Close[START]) * L.Close[END] * (1 - FEE) / (20 * SLOT) - 1:+.1%}")
lg = res["② 케인 규칙"][2]; lg["month"] = pd.to_datetime(lg.date).dt.month
print(lg.groupby("month").agg(롱슬롯_말=("long_val", lambda x: x.iloc[-1] / SLOT), 총자산_말=("total", "last"), 이동합=("move", "sum")).round(1))
