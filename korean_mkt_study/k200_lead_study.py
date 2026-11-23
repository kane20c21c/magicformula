"""KOSPI200 선행 종목 연구 (2026-11-23 케인 요청)

질문: 유니버스(LLV groups/universe_2026-11-16.json, 262종목) 안에서
      종목의 지난 10거래일 수익률로 KOSPI200 의 다음 10거래일 방향을 가장 잘 맞추는 종목,
      특히 연도별로 적중률이 높아지고 있는 종목은?

정의 (케인 승인 2026-11-23)
  신호 s_t = ln(C_t / C_{t-10})          종목 (t 종가까지)
  결과 y_t = ln(K_{t+10} / K_t)          KOSPI200 (t 종가 이후 10거래일)
  적중    = sign(s_t) == sign(y_t)       (0 은 제외)
  균형 적중률 = (지수 상승일 적중률 + 지수 하락일 적중률) / 2  — 상승장 쏠림 보정
  상관    = Spearman(s_t, y_t)
  기준선  = KOSPI200 자기 자신의 지난 10일 수익률(지수 모멘텀)
  우연 수준 = KOSPI200 결과 계열을 무작위 날짜만큼 원형 이동(블록 구조 유지)해 같은 통계를 재계산
데이터: data_2015/prices.parquet · kospi200_daily.parquet (새 수집 없음). 보류 구간 없음 (케인).
⚠ 유니버스 = 2026-11 현재 명단 → 과거 연도는 생존편향. 10일 구간이 겹쳐 연도당 독립 관측은 약 25개.
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy import stats

H = 10
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out", "k200_lead"); os.makedirs(OUT, exist_ok=True)
UNI = os.path.expanduser("~/mnt/LongLiveVault/groups/universe_2026-11-16.json")
if len(sys.argv) > 1: UNI = sys.argv[1]
N_SHUF = 200
MIN_DAYS_YEAR = 120   # 한 해에 이 이상 관측이 있어야 그 해를 씀

def load():
    u = json.load(open(UNI))["members"]
    p = pd.read_parquet(os.path.join(ROOT, "data_2015/prices.parquet"), columns=["close"])
    px = p["close"].unstack("ticker")
    px = px[[t for t in u if t in px.columns]]
    k = pd.read_parquet(os.path.join(ROOT, "data_2015/kospi200_daily.parquet"))["Close"]
    k.index = pd.to_datetime(k.index)
    idx = px.index.intersection(k.index)
    return px.loc[idx], k.loc[idx]

def year_stats(s, y, years, ic=True):
    """s: (T,N) 신호, y: (T,) 결과. 연도별 [적중률, 균형적중률, 상관, n]."""
    res = {}
    for yr in np.unique(years):
        m = years == yr
        ss, yy = s[m], y[m]
        ok = ~np.isnan(ss) & ~np.isnan(yy)[:, None] & (ss != 0) & (yy != 0)[:, None]
        n = ok.sum(0)
        hit = ((np.sign(ss) == np.sign(yy)[:, None]) & ok)
        up = (yy > 0)[:, None] & ok; dn = (yy < 0)[:, None] & ok
        with np.errstate(invalid="ignore", divide="ignore"):
            hr = hit.sum(0) / n
            bal = 0.5 * ((hit & up).sum(0) / up.sum(0) + (hit & dn).sum(0) / dn.sum(0))
        # Spearman: 순위 상관
        rs = np.full(s.shape[1], np.nan)
        for j in (range(s.shape[1]) if ic else []):
            o = ok[:, j]
            if o.sum() >= MIN_DAYS_YEAR:
                rs[j] = stats.spearmanr(ss[o, j], yy[o]).correlation
        hr[n < MIN_DAYS_YEAR] = np.nan; bal[n < MIN_DAYS_YEAR] = np.nan
        res[yr] = (hr, bal, rs, n)
    return res

def slope(vals, yrs):
    m = ~np.isnan(vals)
    if m.sum() < 5: return np.nan, np.nan
    r = stats.linregress(yrs[m], vals[m])
    return r.slope, r.pvalue

def main():
    px, k = load()
    lp, lk = np.log(px.values), np.log(k.values)
    s = np.full_like(lp, np.nan); s[H:] = lp[H:] - lp[:-H]
    y = np.full_like(lk, np.nan); y[:-H] = lk[H:] - lk[:-H]
    sk = np.full_like(lk, np.nan); sk[H:] = lk[H:] - lk[:-H]
    years = px.index.year.values
    yrs = np.unique(years)

    R = year_stats(s, y, years)
    B = year_stats(sk[:, None], y, years)   # 지수 모멘텀 기준선
    tick = list(px.columns)
    rows = []
    for yr in yrs:
        hr, bal, rs, n = R[yr]
        bh, bb, br, bn = B[yr]
        yy = y[years == yr]; base_up = np.nanmean(yy[~np.isnan(yy)] > 0)
        for j, t in enumerate(tick):
            rows.append(dict(ticker=t, year=yr, n=int(n[j]), hit=hr[j], bal=bal[j], ic=rs[j],
                             k200_mom_hit=bh[0], k200_mom_bal=bb[0], k200_mom_ic=br[0], k200_up_share=base_up))
    long = pd.DataFrame(rows)
    long.to_csv(os.path.join(OUT, "yearly_long.csv"), index=False, encoding="utf-8-sig")

    # 종목 요약: 기울기·최근 3년(2024~2026)
    summ = []
    for t, g in long.groupby("ticker"):
        g = g.set_index("year")
        sb, pb = slope(g["bal"].values, g.index.values.astype(float))
        sh, ph = slope(g["hit"].values, g.index.values.astype(float))
        si, pi = slope(g["ic"].values, g.index.values.astype(float))
        rec = g.loc[g.index >= 2024]
        summ.append(dict(ticker=t, years=int(g["bal"].notna().sum()),
                         bal_all=np.nanmean(g["bal"]), hit_all=np.nanmean(g["hit"]), ic_all=np.nanmean(g["ic"]),
                         bal_recent3=rec["bal"].mean(), hit_recent3=rec["hit"].mean(), ic_recent3=rec["ic"].mean(),
                         bal_slope=sb, bal_slope_p=pb, hit_slope=sh, hit_slope_p=ph, ic_slope=si, ic_slope_p=pi))
    S = pd.DataFrame(summ)

    # 우연 수준: 결과 계열 원형 이동
    rng = np.random.default_rng(20261123)
    valid = ~np.isnan(y); T = valid.sum()
    null_rec, null_slope, null_both = [], [], []
    yv = y[valid]
    for i in range(N_SHUF):
        sh_ = rng.integers(250, T - 250)
        y2 = y.copy(); y2[valid] = np.roll(yv, sh_)
        R2 = year_stats(s, y2, years, ic=False)
        bal_mat = np.vstack([R2[yr][1] for yr in yrs])   # (Y,N)
        rec = np.nanmean(bal_mat[yrs >= 2024], axis=0)
        sl = np.array([slope(bal_mat[:, j], yrs.astype(float))[0] for j in range(bal_mat.shape[1])])
        null_rec.append(np.nanmax(rec)); null_slope.append(np.nanmax(sl))
        null_both.append(np.nansum((rec >= 0.60) & (sl > 0.01)))
        if i % 50 == 0: print("shuffle", i, flush=True)
    np.save(os.path.join(OUT, "null.npy"), np.array([null_rec, null_slope, null_both]))
    S.to_csv(os.path.join(OUT, "summary.csv"), index=False, encoding="utf-8-sig")
    print("done", S.shape)

if __name__ == "__main__":
    main()
