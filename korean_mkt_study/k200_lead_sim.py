"""선행 3종목 → TIGER200 / KODEX 인버스 슬롯 이동 시뮬레이션 (일반판, 2026-11-23 케인)
사용: python3 k200_lead_sim.py START END 티커,티커,티커 [라벨]
  예) python3 k200_lead_sim.py 2026-06-30 2026-09-30 218410,214370,056080 가
규칙은 k200_lead_sim_2026q2.py 와 같다:
  t 종가 신호 = 선행 3종목 10거래일 수익률 부호 → t+1 시가 체결
  3개 + : 롱 2 매수·숏 2 매도 / 정확히 2개 + : 롱 1·숏 1 / 정확히 2개 − : 숏 1·롱 1 / 3개 − : 숏 2·롱 2
  시작 = START 종가 롱 10 + 숏 10 슬롯(슬롯 200만, 시작 수수료 없음) · END 종가 전량 매도 · 수수료 거래당 0.13%
  비교: ① 매매 없음 ② 케인 규칙(2·3개 일치) ③ 3개 일치일만 · 운 비교 = 슬롯 이동 날짜 섞기 1,000회
자료: 종목·KOSPI200 = data_2015, ETF = LLV panel(102110)·tickers/114800 (새 수집 없음)
"""
import os, sys, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__)); LLV = os.path.expanduser("~/mnt/LongLiveVault/data/ohlcv")
START, END, TK = sys.argv[1], sys.argv[2], sys.argv[3].split(","); TAG = sys.argv[4] if len(sys.argv) > 4 else ""
OUT = os.path.join(ROOT, "out", "k200_lead_sim"); os.makedirs(OUT, exist_ok=True)
SLOT, FEE = 2_000_000, 0.0013
nm = pd.read_parquet(f"{ROOT}/data_2015/meta.parquet", columns=["name"]).reset_index().sort_values("date").groupby("ticker")["name"].last()
SRC = os.environ.get("SRC", "data_2015")   # SRC=llv → 종목·KOSPI200 도 LLV(2026-11-20 까지)에서 읽는다
if SRC == "llv":
    _p = pd.read_parquet(f"{LLV}/panel.parquet", columns=["Date", "Ticker", "Close"])
    px = _p[_p.Ticker.isin(TK)].pivot(index="Date", columns="Ticker", values="Close")[TK]
else:
    px = pd.read_parquet(f"{ROOT}/data_2015/prices.parquet", columns=["close"])["close"].unstack("ticker")[TK]
r10 = np.log(px / px.shift(10))
L = pd.read_parquet(f"{LLV}/panel.parquet", columns=["Date", "Ticker", "Open", "Close"]); L = L[L.Ticker == "102110"].set_index("Date")[["Open", "Close"]]
S = pd.read_parquet(f"{LLV}/tickers/114800.parquet").set_index("Date")[["Open", "Close"]]
if SRC == "llv":
    k = pd.read_parquet(f"{LLV}/tickers/KOSPI200.parquet").set_index("Date")["Close"]
else:
    k = pd.read_parquet(f"{ROOT}/data_2015/kospi200_daily.parquet")["Close"]
k.index = pd.to_datetime(k.index)
days = L.loc[START:END].index
assert (S.loc[START:END].index == days).all() and all(d in k.index for d in days) and all(d in r10.index for d in days[:-1])

def signal(d):
    s = np.sign(r10.loc[d]); up, dn = int((s > 0).sum()), int((s < 0).sum())
    if up == 3: return 2, up, dn
    if dn == 3: return -2, up, dn
    if up == 2: return 1, up, dn
    if dn == 2: return -1, up, dn
    return 0, up, dn

def run(moves):
    p0L, p0S = L.Close[START], S.Close[START]
    sh = {"L": 10 * SLOT // p0L, "S": 10 * SLOT // p0S}; cash = 20 * SLOT - sh["L"] * p0L - sh["S"] * p0S; fees = 0.0; log = []
    for i in range(1, len(days)):
        d = days[i]; mv = moves[i - 1]; o = {"L": L.Open[d], "S": S.Open[d]}
        sell, buy = ("S", "L") if mv > 0 else ("L", "S")
        for _ in range(abs(mv)):
            q = min(sh[sell], int(SLOT // o[sell]))
            if q <= 0: break
            amt = q * o[sell]; sh[sell] -= q; cash += amt * (1 - FEE); fees += amt * FEE
            qb = int(cash // (o[buy] * (1 + FEE))); sh[buy] += qb; cash -= qb * o[buy] * (1 + FEE); fees += qb * o[buy] * FEE
        log.append(dict(date=d.date(), move=mv, k200=k[d], long_val=sh["L"] * L.Close[d], short_val=sh["S"] * S.Close[d], cash=cash,
                        total=sh["L"] * L.Close[d] + sh["S"] * S.Close[d] + cash))
    fin = sh["L"] * L.Close[END] + sh["S"] * S.Close[END]; fees += fin * FEE
    return fin * (1 - FEE) + cash, fees, pd.DataFrame(log)

sig = [signal(days[i - 1]) for i in range(1, len(days))]
mv_all = np.array([s[0] for s in sig])
rules = {"① 매매 없음": np.zeros_like(mv_all), "② 케인 규칙(2·3개)": mv_all, "③ 3개 일치일만": np.where(np.abs(mv_all) == 2, mv_all, 0)}
print(f"[{TAG}] {START}~{END}  선행: " + ", ".join(f"{nm.get(t, t)}({t})" for t in TK))
print(f"  KOSPI200 {k[END] / k[START] - 1:+.1%} · TIGER200 {L.Close[END] / L.Close[START] - 1:+.1%} · 인버스 {S.Close[END] / S.Close[START] - 1:+.1%}")
rng = np.random.default_rng(20261123); rows = []
for rn, mv in rules.items():
    fin, fees, lg = run(mv); ret = fin / (20 * SLOT) - 1
    line = dict(라벨=TAG, 규칙=rn, 최종=round(fin), 수익률=ret, 수수료=round(fees), 이동일=int((mv != 0).sum()),
                최대롱슬롯=lg.long_val.max() / SLOT, 최고총자산=lg.total.max())
    if (mv != 0).any():
        null = np.array([run(rng.permutation(mv))[0] for _ in range(1000)]) / (20 * SLOT) - 1
        line.update(섞기중앙=np.median(null), 섞기10=np.quantile(null, .1), 섞기90=np.quantile(null, .9), 실제이상비율=(null >= ret).mean())
        lg.to_csv(f"{OUT}/{START[:7]}_{END[:7]}_{TAG}_{rn[0]}{'_llv' if SRC == 'llv' else ''}_daily.csv", index=False, encoding="utf-8-sig")
    rows.append(line)
    extra = f" | 섞기 중앙 {line['섞기중앙']:+.1%} [10~90% {line['섞기10']:+.1%}~{line['섞기90']:+.1%}], 실제 이상 {line['실제이상비율']:.1%}" if "섞기중앙" in line else ""
    print(f"  {rn}: {fin:,.0f}원 ({ret:+.1%}) 수수료 {fees:,.0f}원 이동일 {line['이동일']}{extra}")
ev = pd.DataFrame({"mv": mv_all, "ret": [k[days[i]] / k[days[i - 1]] - 1 for i in range(1, len(days))]}); ev = ev[ev.ret != 0]
for n, m in [("3개 일치", ev.mv.abs() == 2), ("정확히 2개", ev.mv.abs() == 1), ("전체", ev.mv != 0)]:
    e = ev[m]; h = np.sign(e.mv) == np.sign(e.ret); upd = e.ret > 0
    print(f"  신호 {n}: {len(e)}일 적중 {h.mean():.1%} 균형 {0.5 * (h[upd].mean() + h[~upd].mean()):.1%} (상승예측 {int((e.mv > 0).sum())} / 하락예측 {int((e.mv < 0).sum())})")
print(f"  지수 상승일 비율 {(ev.ret > 0).mean():.1%}")
for t in TK:
    e = pd.DataFrame({"s": np.sign(r10[t].reindex(days[:-1]).values), "y": np.sign(ev.ret.reindex(range(len(days) - 1)).values)}).dropna()
    e = e[(e.s != 0) & (e.y != 0)]; h = e.s == e.y
    print(f"    {nm.get(t, t)}: 적중 {h.mean():.1%} 균형 {0.5 * (h[e.y > 0].mean() + h[e.y < 0].mean()):.1%}")
R = pd.DataFrame(rows); f = f"{OUT}/summary.csv"
R.to_csv(f, mode="a", header=not os.path.exists(f), index=False, encoding="utf-8-sig")
