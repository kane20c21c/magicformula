"""분기별 '종목 10일 수익률 방향 → 다음 날 KOSPI200 방향' 적중률, 2020Q1~END 분기 (2026-11-23 케인)
목적: END(기본 2026-03-31, 인자로 지정) 까지로 후보를 고르고 다음 분기에 예측해 본다 → 선정에 END 이후 결과를 쓰지 않는다.
  t 의 분기 = t 날짜 기준, 결과 y_t = ln(K_{t+1}/K_t). t+1 > END 인 행은 제외 (다음 분기 눈가림).
척도: 적중률(부호 일치, 0 제외) · 균형적중률(지수 상승일·하락일 적중률 평균)
추세: 분기 번호(0~24)에 대한 적중률 OLS 기울기
선정: 25분기 중 20분기 이상 관측 & 적중률 기울기 > 0 → 최근 4분기(END 기준 직전 4분기) 평균 적중률 순
우연 수준: 결과 계열을 무작위 날짜만큼 원형 이동 200회 (같은 선정 절차의 1등·10등 값)
"""
import json, os, sys, numpy as np, pandas as pd
from scipy import stats
ROOT = os.path.dirname(os.path.abspath(__file__)); END = sys.argv[1] if len(sys.argv) > 1 else "2026-03-31"   # 선정 마지막 날 (다음 날 결과가 이 날 이후면 제외)
OUT = os.path.join(ROOT, "out", "k200_quarterly" if END == "2026-03-31" else f"k200_quarterly_{END[:7]}"); os.makedirs(OUT, exist_ok=True)
u = json.load(open(os.path.expanduser("~/mnt/LongLiveVault/groups/universe_2026-11-16.json")))["members"]
px = pd.read_parquet(f"{ROOT}/data_2015/prices.parquet", columns=["close"])["close"].unstack("ticker"); px = px[[t for t in u if t in px.columns]]
k = pd.read_parquet(f"{ROOT}/data_2015/kospi200_daily.parquet")["Close"]; k.index = pd.to_datetime(k.index)
idx = px.index.intersection(k.index); px, k = px.loc[idx], k.loc[idx]
s = np.log(px / px.shift(10)); y = np.log(k.shift(-1) / k); nxt = pd.Series(k.index, index=k.index).shift(-1)
keep = (s.index >= "2020-01-01") & (nxt <= END).values
s, y = s[keep], y[keep]
q = s.index.to_period("Q"); qs = sorted(q.unique()); qi = np.array([qs.index(x) for x in q])
S = np.sign(s.values); Y = np.sign(y.values); NQ = len(qs); MINQ = 40
rec = np.arange(NQ) >= NQ - 4
def qstats(Y):
    HR = np.full((NQ, S.shape[1]), np.nan); BAL = HR.copy()
    for i in range(NQ):
        m = qi == i; ss, yy = S[m], Y[m]
        ok = (ss != 0) & ~np.isnan(ss) & (yy != 0)[:, None]
        h = (ss == yy[:, None]) & ok; n = ok.sum(0); up = (yy > 0)[:, None] & ok; dn = (yy < 0)[:, None] & ok
        with np.errstate(invalid="ignore", divide="ignore"):
            hr = h.sum(0) / n; bal = 0.5 * ((h & up).sum(0) / up.sum(0) + (h & dn).sum(0) / dn.sum(0))
        hr[n < MINQ] = np.nan; bal[n < MINQ] = np.nan; HR[i], BAL[i] = hr, bal
    return HR, BAL
def select(HR):
    x = np.arange(NQ, dtype=float); out = []
    for j in range(HR.shape[1]):
        v = HR[:, j]; m = ~np.isnan(v)
        if m.sum() < 20: out.append((np.nan, np.nan, np.nan, np.nan)); continue
        r = stats.linregress(x[m], v[m]); out.append((r.slope, r.pvalue, np.nanmean(v[rec]), np.nanmean(v[~rec])))
    return np.array(out)
HR, BAL = qstats(Y); sel = select(HR)
tick = list(px.columns)
nm = pd.read_parquet(f"{ROOT}/data_2015/meta.parquet", columns=["name"]).reset_index().sort_values("date").groupby("ticker")["name"].last()
D = pd.DataFrame(dict(ticker=tick, name=[nm.get(t, "") for t in tick], quarters=(~np.isnan(HR)).sum(0),
                      hit_all=np.nanmean(HR, 0), hit_prev=sel[:, 3], hit_recent4=sel[:, 2], bal_recent4=np.nanmean(BAL[rec], 0),
                      slope_pp_per_q=sel[:, 0] * 100, slope_p=sel[:, 1],
                      up_quarters_last8=[int(np.nansum(np.diff(HR[-9:, j]) > 0)) for j in range(len(tick))]))
up_share = np.array([(Y[qi == i] > 0).mean() for i in range(NQ)])
QH = pd.DataFrame(HR, index=[str(x) for x in qs], columns=tick).T; QH["ticker"] = QH.index
QB = pd.DataFrame(BAL, index=[str(x) for x in qs], columns=tick).T
D.to_csv(f"{OUT}/summary.csv", index=False, encoding="utf-8-sig"); QH.to_csv(f"{OUT}/quarterly_hit.csv", encoding="utf-8-sig")
QB.to_csv(f"{OUT}/quarterly_bal.csv", encoding="utf-8-sig")
pd.Series(up_share, index=[str(x) for x in qs]).to_csv(f"{OUT}/k200_up_share.csv", encoding="utf-8-sig")
# 우연 수준
rng = np.random.default_rng(20261123); nul = []
for it in range(200):
    Y2 = np.roll(Y, rng.integers(60, len(Y) - 60)); H2, _ = qstats(Y2); s2 = select(H2)
    ok = s2[:, 0] > 0; r4 = np.sort(s2[ok, 2][~np.isnan(s2[ok, 2])])[::-1]
    nul.append([r4[0], r4[9], np.nanmax(s2[:, 0]) * 100])
np.save(f"{OUT}/null.npy", np.array(nul)); print("done", len(qs), qs[0], qs[-1])
