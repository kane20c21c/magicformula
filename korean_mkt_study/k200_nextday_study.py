"""종목 10일 수익률 → KOSPI200 다음 날 수익률 상관 연구 (2026-11-23 케인 정정 요청)

신호 s_t = ln(C_t / C_{t-10})   종목
결과 y_t = ln(K_{t+1} / K_t)    KOSPI200 다음 거래일
척도: 연도별 Spearman 상관(주) · 방향 적중률 · 균형 적중률(상승일/하락일 적중률 평균)
기준선: KOSPI200 자기 10일 수익률 → 다음 날
우연 수준: y 계열을 무작위 날짜만큼 원형 이동, 200회
선정: 관측 5년 이상 & 상관 기울기 > 0 중 최근 3년(2024~2026.10) 상관 상위
유니버스 = LLV universe_2026-11-16 (262) — 생존편향. 자료 = data_2015 (새 수집 없음). 보류 구간 없음.
"""
import json, os
import numpy as np, pandas as pd
from scipy import stats

H_IN, H_OUT = 10, 1
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out", "k200_nextday"); os.makedirs(OUT, exist_ok=True)
UNI = os.path.expanduser("~/mnt/LongLiveVault/groups/universe_2026-11-16.json")
N_SHUF, MIN_N = 200, 120

def rk(a):
    r = np.empty(len(a)); r[np.argsort(a, kind="mergesort")] = np.arange(len(a)); return r

def spear(x, y):
    a, b = rk(x), rk(y); a -= a.mean(); b -= b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum()); return (a * b).sum() / d if d > 0 else np.nan

def load():
    u = json.load(open(UNI))["members"]
    px = pd.read_parquet(os.path.join(ROOT, "data_2015/prices.parquet"), columns=["close"])["close"].unstack("ticker")
    px = px[[t for t in u if t in px.columns]]
    k = pd.read_parquet(os.path.join(ROOT, "data_2015/kospi200_daily.parquet"))["Close"]; k.index = pd.to_datetime(k.index)
    idx = px.index.intersection(k.index); return px.loc[idx], k.loc[idx]

def year_stats(s, y, years, yrs, full=True):
    N = s.shape[1]; IC = np.full((len(yrs), N), np.nan); HR = IC.copy(); BAL = IC.copy(); NN = np.zeros((len(yrs), N), int)
    for i, yr in enumerate(yrs):
        m = years == yr; ss, yy = s[m], y[m]
        for j in range(N):
            o = ~np.isnan(ss[:, j]) & ~np.isnan(yy); n = o.sum(); NN[i, j] = n
            if n < MIN_N: continue
            x, z = ss[o, j], yy[o]; IC[i, j] = spear(x, z)
            if full:
                nz = (x != 0) & (z != 0); h = np.sign(x[nz]) == np.sign(z[nz]); up = z[nz] > 0
                HR[i, j] = h.mean(); BAL[i, j] = 0.5 * (h[up].mean() + h[~up].mean())
    return IC, HR, BAL, NN

def slope(v, x):
    m = ~np.isnan(v)
    if m.sum() < 5: return np.nan, np.nan
    r = stats.linregress(x[m], v[m]); return r.slope, r.pvalue

def main():
    px, k = load(); lp, lk = np.log(px.values), np.log(k.values)
    s = np.full_like(lp, np.nan); s[H_IN:] = lp[H_IN:] - lp[:-H_IN]
    sk = np.full_like(lk, np.nan); sk[H_IN:] = lk[H_IN:] - lk[:-H_IN]
    y = np.full_like(lk, np.nan); y[:-H_OUT] = lk[H_OUT:] - lk[:-H_OUT]
    years = px.index.year.values; yrs = np.unique(years); yf = yrs.astype(float)
    IC, HR, BAL, NN = year_stats(s, y, years, yrs)
    bIC, bHR, bBAL, _ = year_stats(sk[:, None], y, years, yrs)
    medIC = np.nanmedian(IC, axis=1)
    tick = list(px.columns); rows = []
    for i, yr in enumerate(yrs):
        for j, t in enumerate(tick):
            rows.append(dict(ticker=t, year=yr, n=NN[i, j], ic=IC[i, j], hit=HR[i, j], bal=BAL[i, j],
                             k200_mom_ic=bIC[i, 0], k200_mom_hit=bHR[i, 0], k200_mom_bal=bBAL[i, 0], univ_med_ic=medIC[i]))
    pd.DataFrame(rows).to_csv(f"{OUT}/yearly_long.csv", index=False, encoding="utf-8-sig")
    rec = yrs >= 2024; summ = []
    for j, t in enumerate(tick):
        sl, p = slope(IC[:, j], yf); exs, exp_ = slope(IC[:, j] - medIC, yf)
        m26 = yrs < 2026; s26, p26 = slope(IC[m26, j], yf[m26])
        summ.append(dict(ticker=t, years=int((~np.isnan(IC[:, j])).sum()),
                         ic_all=np.nanmean(IC[:, j]), ic_recent3=np.nanmean(IC[rec, j]),
                         hit_recent3=np.nanmean(HR[rec, j]), bal_recent3=np.nanmean(BAL[rec, j]),
                         pos_years=int(np.nansum(IC[:, j] > 0)),
                         ic_slope=sl, ic_slope_p=p, ic_ex_slope=exs, ic_ex_slope_p=exp_, ic_slope_no26=s26, ic_slope_no26_p=p26))
    S = pd.DataFrame(summ); S.to_csv(f"{OUT}/summary.csv", index=False, encoding="utf-8-sig")
    # 우연 수준
    rng = np.random.default_rng(20261123); v = ~np.isnan(y); yv = y[v]; T = v.sum()
    nul = []
    el = S.years.values >= 5
    for it in range(N_SHUF):
        y2 = y.copy(); y2[v] = np.roll(yv, rng.integers(250, T - 250))
        I2 = year_stats(s, y2, years, yrs, full=False)[0]
        r3 = np.nanmean(I2[rec], axis=0); sl2 = np.array([slope(I2[:, j], yf)[0] for j in range(I2.shape[1])])
        r3e, sle = r3[el], sl2[el]
        ok = sle > 0
        nul.append([np.nanmax(r3e), np.nanmax(sle), np.nanmax(r3e[ok]) if ok.any() else np.nan, np.nanmean(r3e)])
        if it % 50 == 0: print("shuffle", it, flush=True)
    np.save(f"{OUT}/null.npy", np.array(nul)); print("done")

if __name__ == "__main__":
    main()
