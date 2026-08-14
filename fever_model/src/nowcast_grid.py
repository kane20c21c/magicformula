#!/usr/bin/env python3
"""
나우캐스트 격자 실험: MIN_WAVES {4,5,6} × W {40,60,80,100}  (2026-08-13)
- 목적: 파동 개수 하한 완화가 커버리지·안정성·판정 품질에 미치는 영향
- 방식: nowcast.py와 동일 파이프라인(선견 없음, 패널 k=3 배정).
  특징은 gate=4로 1회 계산 후 n_waves로 사후 마스킹 (MINW는 게이트일 뿐 특징값 불변)
- 기준안: W=100 & MIN_WAVES=6 (8/12 채택) — 공통 셀 판정 일치율로 비교
- 사전 기준(8/12와 동일): 커버리지>=90% & 요동비율 정점 26Q1(판정 지연 ~W/2 감안) → 만족 중 최단 창
- 출력: nowcast_grid_compare.csv(요약), nowcast_grid_hot.csv(요동비율 시계열), nowcast_grid_cells.csv(셀 단위)
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import skew as sk, spearmanr
from sklearn.preprocessing import RobustScaler

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
GATE = 4                      # 특징 계산 최소 파동 (amp_vol_corr 스피어만 하한)
MINW_LIST = [4, 5, 6]
WINDOWS = [40, 60, 80, 100]
STEP = 5

FCOLS = ["waves_per_yr", "amp_cv", "amp_skew", "up_dn_amp", "up_dn_days",
         "maxwave_pos", "late_amp_ratio", "days_mean_log", "days_cv",
         "amp_vol_corr", "yz20_med", "yz60_med", "yz_ratio"]

# ---- 패널 모델(자) 로드: 스케일러 + 중심점 ----
pf = pd.read_csv(os.path.join(BASE, "panel_states.csv"), dtype={"ticker": str})
scaler = RobustScaler().fit(pf[FCOLS].values)
Xp = scaler.transform(pf[FCOLS].values)
cent = np.vstack([Xp[pf["state"].values == s].mean(axis=0) for s in (1, 2, 3)])
SNAME = {0: "단기", 1: "장기", 2: "저변"}

# ---- 일별 데이터 ----
excl = set(open(os.path.join(ROOT, "clustering", "excluded_tickers.txt")).read().split())
frames = [pd.read_parquet(os.path.join(ROOT, f), columns=[
    "Date", "Ticker", "Name", "Weis_Dir", "Weis_Chg", "Weis_Days", "Weis_Vol",
    "YZ_20_Ann", "YZ_60_Ann"])
    for f in ["core 260809.parquet", "extend 260809.parquet"]]
df = (pd.concat(frames).drop_duplicates(["Date", "Ticker"])
      .sort_values(["Ticker", "Date"]))
df = df[~df["Ticker"].isin(excl)]
dates = np.array(sorted(df["Date"].unique()))
d2i = {d: i for i, d in enumerate(dates)}

tickers = sorted(df["Ticker"].unique())
N, T = len(tickers), len(dates)
arr = {c: np.full((N, T), np.nan) for c in
       ["dir", "chg", "wdays", "vol", "yz20", "yz60"]}
for j, (t, g) in enumerate(df.groupby("Ticker")):
    idx = g["Date"].map(d2i).values
    arr["dir"][j, idx] = g["Weis_Dir"].values
    arr["chg"][j, idx] = g["Weis_Chg"].values
    arr["wdays"][j, idx] = g["Weis_Days"].values
    arr["vol"][j, idx] = g["Weis_Vol"].values
    arr["yz20"][j, idx] = g["YZ_20_Ann"].values
    arr["yz60"][j, idx] = g["YZ_60_Ann"].values

def build_waves(j, lo, hi):
    dirv = arr["dir"][j, lo:hi + 1]
    ok = ~np.isnan(dirv)
    if ok.sum() < 10:
        return None
    ii = np.where(ok)[0]
    dv = dirv[ii]
    cut = np.where(np.diff(dv) != 0)[0]
    starts = np.r_[0, cut + 1]
    ends = np.r_[cut, len(dv) - 1]
    out = []
    for s, e in zip(starts, ends):
        seg = ii[s:e + 1] + lo
        out.append((dv[s],
                    np.nanmax(np.abs(arr["chg"][j, seg])),
                    np.nanmax(arr["wdays"][j, seg]),
                    np.nanmax(arr["vol"][j, seg]),
                    seg[-1]))
    return out

full_waves = {}
for j in range(N):
    w = build_waves(j, 0, T - 1)
    full_waves[j] = w if w else []

def vol_median_upto(j, ti):
    vols = [w[3] for w in full_waves[j] if w[4] < ti]
    return np.median(vols) if len(vols) >= 5 else np.nan

def features(j, ti, W):
    """gate=4로 계산. 반환 (특징, 파동수) 또는 None. MINW 마스킹은 사후."""
    lo = ti - W + 1
    if lo < 0:
        return None
    w = build_waves(j, lo, ti)
    if not w or len(w) < GATE:
        return None
    dirs = np.array([x[0] for x in w]); amps = np.array([x[1] for x in w])
    days = np.array([x[2] for x in w]); vols = np.array([x[3] for x in w])
    endi = np.array([x[4] for x in w], dtype=float)
    vm = vol_median_upto(j, ti)
    if not np.isfinite(vm) or vm <= 0:
        return None
    vn = vols / vm
    up, dn = dirs > 0, dirs < 0
    if not up.any() or not dn.any():
        return None
    span = max(endi[-1] - endi[0], 1)
    pos = (endi - endi[0]) / span
    late = pos > 0.75
    yz20 = np.nanmedian(arr["yz20"][j, lo:ti + 1])
    yz60 = np.nanmedian(arr["yz60"][j, lo:ti + 1])
    if not (np.isfinite(yz20) and np.isfinite(yz60)) or yz60 <= 0:
        return None
    rho = spearmanr(amps, vn).correlation if len(w) >= 4 else np.nan
    f = [len(w) / (W / 252),
         amps.std() / amps.mean(),
         sk(amps, bias=False),
         np.log(amps[up].mean() / amps[dn].mean()),
         np.log(days[up].mean() / days[dn].mean()),
         float(pos[np.argmax(amps)]),
         amps[late].mean() / amps.mean() if late.any() else np.nan,
         np.log(days.mean()),
         days.std() / days.mean(),
         rho,
         yz20, yz60, yz20 / yz60]
    if any(not np.isfinite(x) for x in f):
        return None
    return f, len(w)

GRID = list(range(100, T, STEP))
if GRID[-1] != T - 1:
    GRID.append(T - 1)
all_dates = [dates[gi] for gi in GRID]
n_dates = len(GRID)

# ---- 셀 단위 판정 (W별 배치 변환) ----
cell_frames = []
for W in WINDOWS:
    feats, meta = [], []
    for gi in GRID:
        for j in range(N):
            r = features(j, gi, W)
            if r is None:
                continue
            f, nw = r
            feats.append(f)
            meta.append((dates[gi], tickers[j], nw))
    Z = scaler.transform(np.array(feats))
    D = np.linalg.norm(Z[:, None, :] - cent[None, :, :], axis=2)
    o = np.argsort(D, axis=1)
    m = np.arange(len(D))
    cf = pd.DataFrame(meta, columns=["date", "ticker", "n_waves"])
    cf["W"] = W
    cf["state"] = [SNAME[i] for i in o[:, 0]]
    cf["conf"] = D[m, o[:, 1]] / D[m, o[:, 0]]
    cell_frames.append(cf)
    print(f"W={W}: gate-4 판정 셀 {len(cf)}")

cdf = pd.concat(cell_frames, ignore_index=True)
cdf.to_csv(os.path.join(BASE, "nowcast_grid_cells.csv"), index=False)

# ---- 기준안: W=100 & MINW=6 ----
base = (cdf[(cdf["W"] == 100) & (cdf["n_waves"] >= 6)]
        [["date", "ticker", "state"]].rename(columns={"state": "bstate"}))

rows, hot_series = [], {}
for W in WINDOWS:
    sub_all = cdf[cdf["W"] == W]
    for MINW in MINW_LIST:
        sub = sub_all[sub_all["n_waves"] >= MINW]
        cnt = sub.groupby("date").size().reindex(all_dates, fill_value=0)
        coverage = (cnt / N).mean()
        hot = (sub.groupby("date")["state"]
               .apply(lambda s: s.isin(["단기", "장기"]).mean())
               .reindex(all_dates))
        piv = sub.pivot(index="ticker", columns="date", values="state") \
                 .reindex(columns=all_dates)
        pv = piv.values
        both = (~pd.isna(pv[:, :-1])) & (~pd.isna(pv[:, 1:]))
        flip = ((pv[:, :-1] != pv[:, 1:]) & both).sum() / max(both.sum(), 1)
        mg = sub.merge(base, on=["date", "ticker"], how="left")
        common = mg.dropna(subset=["bstate"])
        agree = (common["state"] == common["bstate"]).mean()
        new = mg[mg["bstate"].isna()]
        nd = new["state"].value_counts(normalize=True)
        peak = hot.idxmax()
        rows.append(dict(
            W=W, MINW=MINW,
            coverage=round(coverage, 4), flip=round(flip, 4),
            peak=str(pd.Timestamp(peak).date()),
            hot_peak=round(hot.max(), 4),
            hot_last=round(hot.dropna().iloc[-1], 4),
            agree_base=round(agree, 4), n_common=len(common),
            new_per_date=round(len(new) / n_dates, 1),
            new_저변=round(nd.get("저변", 0.0), 3),
            new_장기=round(nd.get("장기", 0.0), 3),
            new_단기=round(nd.get("단기", 0.0), 3)))
        hot_series[f"W{W}_M{MINW}"] = hot.values

res = pd.DataFrame(rows)
res.to_csv(os.path.join(BASE, "nowcast_grid_compare.csv"), index=False)
hs = pd.DataFrame(hot_series, index=[str(pd.Timestamp(d).date()) for d in all_dates])
hs.to_csv(os.path.join(BASE, "nowcast_grid_hot.csv"))

print()
print(res.to_string(index=False))
print()
print("사전 기준: 커버리지>=90% & 정점 26Q1(지연 ~W/2 감안) → 만족 중 최단 창")
print("검산: W=100/M6은 8/12 원 결과(커버리지 94.7%, 뒤집힘 8.4%, 정점 2026-05-12)와 일치해야 함")
