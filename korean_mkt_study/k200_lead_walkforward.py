"""선행 3종목 전진 검증 (2026-11-23 케인 설계)

선정 (매월 초): 그 달 첫 거래일 이전 결과만으로 직전 240거래일(≈12개월)을 40거래일 6구간으로 나눔.
  관측 = (신호일 t, 결과일 t+1) — 종목 10거래일 수익률 부호 vs KOSPI200 다음 날 부호, 결과일 기준으로 배정.
  구간별 적중률(0 제외)·균형적중률(지수 상승일/하락일 적중률 평균). 6구간 모두 30관측 이상인 종목만.
  적중률 기울기(구간 번호 0~5 OLS) > 0 중 최근 4구간 균형적중률 평균 상위 3 = 그 달 선행 3종목.
  그 달에 체결되는 매매(신호일 t 의 다음 날이 그 달)는 그 달 선행 3종목의 t 종가 신호로 한다.
운용 (분기 리셋): 직전 분기 말 종가에 롱 10 + 숏 10 슬롯(슬롯 200만, 시작 수수료 없음) → 분기 말 종가 전량 매도.
  ② 2개 이상 일치: 3개 같은 방향 ±2슬롯, 정확히 2개 ±1슬롯 (기본)   ③ 3개 일치일만 ±2
  기준: KOSPI200 자기 10일 수익률 부호 ±1 / ±2 · 매매 없음.  수수료 거래당 0.13%. 수량은 소수(분할) 허용.
가격: 롱 = KOSPI200 지수 시가·종가, 숏 = 지수 일간 −1배 합성(시가 = 전일 종가×(1 − 지수 시가갭)).
  검증용으로 2024Q1~ 는 실제 TIGER200(102110)·KODEX 인버스(114800) 로도 돌린다.
자료: 종목 = data_2015(~2026-10-29) + LLV panel(이후, 10-29 비율로 이음) · 지수 = LLV KOSPI200 · 새 수집 없음.
⚠ 유니버스 = LLV universe_2026-11-16 (262, 현재 명단 → 생존편향).
"""
import json, os, numpy as np, pandas as pd
from scipy import stats
ROOT = os.path.dirname(os.path.abspath(__file__)); LLV = os.path.expanduser("~/mnt/LongLiveVault/data/ohlcv")
SIG_N = int(os.environ.get("SIG_N", 10))   # 매매 신호 수익률 기간 (종목·지수 공통, 2026-11-23 케인: 5 도 시험)
SEL_N = int(os.environ.get("SEL_N", 10))   # 선행종목 선정에 쓰는 종목 수익률 기간
OUT = os.path.join(ROOT, "out", "k200_walkforward" + ("" if (SIG_N, SEL_N) == (10, 10) else f"_sig{SIG_N}_sel{SEL_N}")); os.makedirs(OUT, exist_ok=True)
SLOT, FEE, NB, BL, MINB = 2_000_000, 0.0013, 6, 40, 30
T0, T1 = "2020-01-01", "2026-11-20"
u = json.load(open(os.path.expanduser("~/mnt/LongLiveVault/groups/universe_2026-11-16.json")))["members"]
a = pd.read_parquet(f"{ROOT}/data_2015/prices.parquet", columns=["close"])["close"].unstack("ticker"); a = a[[t for t in u if t in a.columns]]
p = pd.read_parquet(f"{LLV}/panel.parquet", columns=["Date", "Ticker", "Close"]); b = p[p.Ticker.isin(a.columns)].pivot(index="Date", columns="Ticker", values="Close")[a.columns]
seam = a.index.max(); b = b.loc[b.index > seam] * (a.loc[seam] / b.reindex([seam]).iloc[0])
px = pd.concat([a, b]).sort_index()
K = pd.read_parquet(f"{LLV}/tickers/KOSPI200.parquet").set_index("Date")[["Open", "Close"]]; K.index = pd.to_datetime(K.index)
k15 = pd.read_parquet(f"{ROOT}/data_2015/kospi200_daily.parquet")["Close"]; k15.index = pd.to_datetime(k15.index)
assert (K.Close.reindex(k15.index) - k15).abs().max() < 1e-6, "KOSPI200 두 원천 불일치"
idx = px.index.intersection(K.index); px, K = px.loc[idx], K.loc[idx]
SG_SEL = np.sign(np.log(px / px.shift(SEL_N))).values              # (T,N) 선정용
SG = np.sign(np.log(px / px.shift(SIG_N))).values                  # (T,N) 매매 신호
k10 = np.sign(np.log(K.Close / K.Close.shift(SIG_N))).values   # 이름은 그대로, 기간은 SIG_N
Yn = np.r_[np.sign(np.diff(np.log(K.Close.values))), np.nan]        # t 의 다음 날 방향
nextd = pd.DatetimeIndex(list(idx[1:]) + [pd.NaT]); tick = list(px.columns)
nm = pd.read_parquet(f"{ROOT}/data_2015/meta.parquet", columns=["name"]).reset_index().sort_values("date").groupby("ticker")["name"].last()

def pick(month_start):
    m = np.where(np.asarray(~nextd.isna() & (nextd < month_start)))[0][-NB * BL:]
    if len(m) < NB * BL: return None, None
    ss, yy = SG_SEL[m], Yn[m]; HR = np.full((NB, len(tick)), np.nan); BA = HR.copy()
    for i in range(NB):
        s_, y_ = ss[i * BL:(i + 1) * BL], yy[i * BL:(i + 1) * BL]
        ok = (s_ != 0) & ~np.isnan(s_) & (y_ != 0)[:, None]; h = (s_ == y_[:, None]) & ok; n = ok.sum(0)
        up, dn = (y_ > 0)[:, None] & ok, (y_ < 0)[:, None] & ok
        with np.errstate(invalid="ignore", divide="ignore"):
            hr = h.sum(0) / n; ba = 0.5 * ((h & up).sum(0) / up.sum(0) + (h & dn).sum(0) / dn.sum(0))
        hr[n < MINB] = np.nan; ba[n < MINB] = np.nan; HR[i], BA[i] = hr, ba
    valid = ~np.isnan(HR).any(0); x = np.arange(NB) - (NB - 1) / 2
    slope = np.where(valid, (x[:, None] * (HR - np.nanmean(HR, 0))).sum(0) / (x ** 2).sum(), np.nan)
    rec = np.nanmean(BA[-4:], 0); cand = np.where(valid & (slope > 0))[0]
    top = cand[np.argsort(-rec[cand])][:3]
    return [tick[j] for j in top], dict(slope=slope[top], rec=rec[top], n_cand=len(cand), n_valid=int(valid.sum()))

months = pd.date_range(T0, T1, freq="MS"); picks = {}; prow = []
for ms in months:
    first = idx[idx >= ms]
    if len(first) == 0 or first[0] > pd.Timestamp(T1): continue
    tk, info = pick(first[0]); picks[ms.strftime("%Y-%m")] = tk
    prow.append(dict(월=ms.strftime("%Y-%m"), **{f"종목{i+1}": f"{nm.get(t, t)}({t})" for i, t in enumerate(tk)},
                     **{f"최근4구간균형{i+1}": info["rec"][i] for i in range(3)}, 후보수=info["n_cand"], 유효종목=info["n_valid"]))
pd.DataFrame(prow).to_csv(f"{OUT}/monthly_picks.csv", index=False, encoding="utf-8-sig")

# 선행 신호(그 달 선행 3종목) — 매매일 d 기준: 신호일 = d 전날, 종목 = d 가 속한 달의 선정
pos = {t: j for j, t in enumerate(tick)}
def lead_move(i_d):   # i_d = 매매일 인덱스
    tk = picks.get(idx[i_d].strftime("%Y-%m")); s = SG[i_d - 1, [pos[t] for t in tk]]
    up, dn = int((s > 0).sum()), int((s < 0).sum())
    return (2 if up == 3 else -2 if dn == 3 else 1 if up == 2 else -1 if dn == 2 else 0), up, dn

# 가격 (합성)
KO, KC = K.Open.values, K.Close.values
r = np.r_[0, KC[1:] / KC[:-1] - 1]; SC = np.cumprod(1 - r); SO = np.r_[np.nan, SC[:-1] * (1 - (KO[1:] / KC[:-1] - 1))]
SYN = dict(LO=KO, LC=KC, SO=SO, SC=SC)
Lr = pd.read_parquet(f"{LLV}/panel.parquet", columns=["Date", "Ticker", "Open", "Close"]); Lr = Lr[Lr.Ticker == "102110"].set_index("Date").reindex(idx)
Sr = pd.read_parquet(f"{LLV}/tickers/114800.parquet").set_index("Date")[["Open", "Close"]].reindex(idx)
REAL = dict(LO=Lr.Open.values, LC=Lr.Close.values, SO=Sr.Open.values, SC=Sr.Close.values)

def sim_cash(i0, i1, moves, P):   # 롱/현금 틀 (2026-11-23 케인): 숏 자리에 현금(이자 0%, 현금 쪽 수수료 없음)
    uL, cash = 10 * SLOT / P["LC"][i0], 10 * SLOT
    for i in range(i0 + 1, i1 + 1):
        mv = moves[i]; o = P["LO"][i]
        for _ in range(abs(mv)):
            if mv > 0:
                amt = min(cash, SLOT)
                if amt <= 1e-6: break
                uL += amt / (o * (1 + FEE)); cash -= amt
            else:
                q = min(uL, SLOT / o)
                if q <= 1e-12: break
                uL -= q; cash += q * o * (1 - FEE)
    return (uL * P["LC"][i1] * (1 - FEE) + cash) / (20 * SLOT) - 1

def sim(i0, i1, moves, P):   # i0 = 시작(직전 분기 말) 인덱스, i1 = 분기 말 인덱스, moves[i] for i in (i0, i1]
    uL, uS = 10 * SLOT / P["LC"][i0], 10 * SLOT / P["SC"][i0]; fee = 0.0; tv = []
    for i in range(i0 + 1, i1 + 1):
        mv = moves[i]
        for _ in range(abs(mv)):
            if mv > 0: so, bo, us = P["SO"][i], P["LO"][i], uS
            else: so, bo, us = P["LO"][i], P["SO"][i], uL
            q = min(us, SLOT / so)
            if q <= 1e-12: break
            pr = q * so * (1 - FEE); fee += q * so * FEE; qb = pr / (bo * (1 + FEE)); fee += qb * bo * FEE
            if mv > 0: uS -= q; uL += qb
            else: uL -= q; uS += qb
        tv.append((idx[i], uL * P["LC"][i] + uS * P["SC"][i], uL * P["LC"][i] / SLOT, uS * P["SC"][i] / SLOT))
    fin = (uL * P["LC"][i1] + uS * P["SC"][i1]) * (1 - FEE); fee += (uL * P["LC"][i1] + uS * P["SC"][i1]) * FEE
    return fin / (20 * SLOT) - 1, fee, tv

N = len(idx); mv2 = np.zeros(N, int); ups = np.zeros(N, int)
for i in range(1, N):
    if idx[i] >= pd.Timestamp(T0): mv2[i], ups[i], _ = lead_move(i)
mv3 = np.where(np.abs(mv2) == 2, mv2, 0)
kb = np.r_[0, k10[:-1]]; kb = np.nan_to_num(kb).astype(int)
RULES = {"선행②": mv2, "선행③": mv3, "자기추세±1": kb, "자기추세±2": kb * 2, "매매없음": np.zeros(N, int)}
qs = pd.period_range("2020Q1", "2026Q4", freq="Q"); rows = []; daily = []
for q in qs:
    ins = np.where((idx >= q.start_time) & (idx <= min(q.end_time, pd.Timestamp(T1))))[0]
    if len(ins) == 0: continue
    i0, i1 = ins[0] - 1, ins[-1]; row = dict(분기=str(q), 거래일=len(ins), KOSPI200=KC[i1] / KC[i0] - 1, 끝=idx[i1].date())
    for rn, mv in RULES.items():
        ret, fee, tv = sim(i0, i1, mv, SYN); row[rn] = ret; row[rn + "_수수료"] = fee
        if rn in ("선행②", "자기추세±1"): daily += [dict(분기=str(q), 규칙=rn, date=d.date(), total=v, long_slots=l, short_slots=s) for d, v, l, s in tv]
    for rn in ("선행②", "선행③", "자기추세±1", "자기추세±2"):
        row[rn + "_현금"] = sim_cash(i0, i1, RULES[rn], SYN)
    if not np.isnan(REAL["SO"][i0 + 1:i1 + 1]).any() and not np.isnan(REAL["LC"][i0]) and not np.isnan(REAL["SC"][i0]):
        for rn in ("선행②", "선행③", "자기추세±1", "자기추세±2", "매매없음"):
            row[rn + "_실제ETF"] = sim(i0, i1, RULES[rn], REAL)[0]
    m = ins; e = pd.DataFrame({"mv": mv2[m], "y": Yn[m - 1]}); e = e[(e.mv != 0) & (e.y != 0)]
    h = np.sign(e.mv) == np.sign(e.y); row["선행②_균형적중"] = 0.5 * (h[e.y > 0].mean() + h[e.y < 0].mean()) if len(e) else np.nan
    rows.append(row)
Q = pd.DataFrame(rows)
# 기준 추가 (2026-11-23 케인): 롱 10슬롯 + 현금 10슬롯, 분기 내 매매 없음 · 분기 말 롱 매도 수수료만 (현금 이자 0%)
Q["롱10+현금10"] = 0.5 * ((1 + Q.KOSPI200) * (1 - FEE) - 1)
Q.to_csv(f"{OUT}/quarterly_results.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(daily).to_csv(f"{OUT}/daily_slots.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
show = ["분기", "KOSPI200", "선행②", "선행③", "자기추세±1", "자기추세±2", "매매없음", "선행②_균형적중"]
print((Q[show].set_index("분기") * 100).round(1).to_string())
full = Q[Q.분기 != "2026Q4"]
for a_, b_ in [("선행②", "자기추세±1"), ("선행③", "자기추세±2"), ("선행②", "매매없음"), ("선행②", "선행③")]:
    d = full[a_] - full[b_]; t = stats.ttest_1samp(d, 0); w = (d > 0).sum()
    print(f"{a_} − {b_}: 평균 {d.mean() * 100:+.2f}%p, 중앙 {d.median() * 100:+.2f}%p, 이긴 분기 {w}/{len(d)}, t p={t.pvalue:.3f}, 부호검정 p={stats.binomtest(int(w), len(d)).pvalue:.3f}")
cum = (1 + Q.set_index("분기")[["선행②", "선행③", "자기추세±1", "자기추세±2", "매매없음"]]).cumprod()
print("누적(분기 복리, ~2026-11-20):", (cum.iloc[-1] - 1).round(3).to_dict())
re = Q.dropna(subset=["선행②_실제ETF"])
print("실제ETF 대조 분기:", list(re.분기)); print((re[["분기", "선행②", "선행②_실제ETF", "자기추세±1", "자기추세±1_실제ETF", "매매없음", "매매없음_실제ETF"]].set_index("분기") * 100).round(1).to_string())
