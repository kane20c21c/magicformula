#!/usr/bin/env python3
"""
탄력 점수 V2 (2026-08-13, 케인 지시)
- 발견: V1의 S2=exp(-S1) 동어반복(p1≡p2, V4는 실질 추세우위 50% 가중)
- 수정: S2·S3를 fate_analysis 원안대로 "점화부(창 후반 1/3)" 기준으로 재구현
    S1 추세 우위     = log(상승/하락 평균진폭), 최근 8파동 (V1과 동일)
    S2n 점화부 눌림 진폭   = 후반 1/3 파동의 하락/상승 평균진폭비 (낮을수록 ↑, 역백분위)
    S3n 점화부 눌림 거래량 = 후반 1/3 파동의 하락/상승 평균 vol_norm 비 (낮을수록 ↑, 역백분위)
    S4 YZ 수준       = 창 내 YZ20 중앙값 (V1과 동일)
  후반 1/3에 상승·하락 쌍이 없으면 p2n/p3n = 50(중립) 처리, 비율 보고
- 비교: 신호 간 상관(신·구), 백테스트(V4n vs V4old vs 단일 신호), 온도 분해
- 선견 없음: V1과 동일 (vol_norm 중앙값은 판정일 이전 완결 파동만)
"""
import os
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
W = 100
NW_RECENT = 8
MIN_WAVES = 6

excl = set(open(os.path.join(ROOT, "clustering", "excluded_tickers.txt")).read().split())
frames = [pd.read_parquet(os.path.join(ROOT, f), columns=[
    "Date", "Ticker", "Name", "Weis_Dir", "Weis_Chg", "Weis_Vol",
    "YZ_20_Ann", "Close"])
    for f in ["core 260809.parquet", "extend 260809.parquet"]]
df = (pd.concat(frames).drop_duplicates(["Date", "Ticker"])
      .sort_values(["Ticker", "Date"]))
df = df[~df["Ticker"].isin(excl)]
dates = np.array(sorted(df["Date"].unique()))
d2i = {d: i for i, d in enumerate(dates)}
tickers = sorted(df["Ticker"].unique())
N, T = len(tickers), len(dates)

arr = {c: np.full((N, T), np.nan) for c in ["dir", "chg", "vol", "yz20", "close"]}
for j, (t, g) in enumerate(df.groupby("Ticker")):
    idx = g["Date"].map(d2i).values
    arr["dir"][j, idx] = g["Weis_Dir"].values
    arr["chg"][j, idx] = g["Weis_Chg"].values
    arr["vol"][j, idx] = g["Weis_Vol"].values
    arr["yz20"][j, idx] = g["YZ_20_Ann"].values
    arr["close"][j, idx] = g["Close"].values

def build_waves(j, lo, hi):
    dirv = arr["dir"][j, lo:hi + 1]
    ok = ~np.isnan(dirv)
    if ok.sum() < 10:
        return []
    ii = np.where(ok)[0]
    dv = dirv[ii]
    cut = np.where(np.diff(dv) != 0)[0]
    starts = np.r_[0, cut + 1]
    ends = np.r_[cut, len(dv) - 1]
    out = []
    for s, e in zip(starts, ends):
        seg = ii[s:e + 1] + lo
        out.append((dv[s], np.nanmax(np.abs(arr["chg"][j, seg])),
                    np.nanmax(arr["vol"][j, seg]), seg[-1]))
    return out

full = {j: build_waves(j, 0, T - 1) for j in range(N)}

def signals(j, ti):
    """반환: s1, s2n, s3n, s3old, yz (s2n/s3n은 점화부 기준, NaN 허용)"""
    lo = ti - W + 1
    w = build_waves(j, lo, ti)
    if len(w) < MIN_WAVES:
        return None
    r = w[-NW_RECENT:]
    up = [x for x in r if x[0] > 0]
    dn = [x for x in r if x[0] < 0]
    if not up or not dn:
        return None
    hist_vols = [x[2] for x in full[j] if x[3] < ti]
    if len(hist_vols) < 5:
        return None
    vm = np.median(hist_vols)
    if vm <= 0:
        return None
    ua = np.mean([x[1] for x in up]); da = np.mean([x[1] for x in dn])
    uv = np.mean([x[2] for x in up]) / vm; dv_ = np.mean([x[2] for x in dn]) / vm
    yz = np.nanmedian(arr["yz20"][j, lo:ti + 1])
    if not np.isfinite(yz):
        return None
    s1 = np.log(ua / da)
    s3old = dv_ / uv if uv > 0 else np.nan
    # 점화부: 창 후반 1/3 (거래일 위치 기준)
    cut = lo + (W * 2) // 3
    lt = [x for x in w if x[3] >= cut]
    lu = [x for x in lt if x[0] > 0]
    ld = [x for x in lt if x[0] < 0]
    if lu and ld:
        s2n = np.mean([x[1] for x in ld]) / np.mean([x[1] for x in lu])
        luv = np.mean([x[2] for x in lu]) / vm
        ldv = np.mean([x[2] for x in ld]) / vm
        s3n = ldv / luv if luv > 0 else np.nan
    else:
        s2n, s3n = np.nan, np.nan
    if not np.isfinite(s1):
        return None
    return s1, s2n, s3n, s3old, yz

def pct(s):
    return s.rank(pct=True) * 100

def score_at(ti):
    raw = {}
    for j in range(N):
        sg = signals(j, ti)
        if sg:
            raw[j] = sg
    if len(raw) < 30:
        return None
    d = pd.DataFrame(raw, index=["s1", "s2n", "s3n", "s3old", "yz"]).T
    d["p1"] = pct(d["s1"])
    d["p2n"] = (100 - pct(d["s2n"])).fillna(50.0)
    d["p3n"] = (100 - pct(d["s3n"])).fillna(50.0)
    d["p3o"] = 100 - pct(d["s3old"])
    d["p4"] = pct(d["yz"])
    d["neutral"] = d["s2n"].isna()
    d["V4n"] = (d["p1"] + d["p2n"] + d["p3n"] + d["p4"]) / 4
    d["V4o"] = (2 * d["p1"] + d["p3o"] + d["p4"]) / 4   # V1과 동일 (p2o=p1)
    return d

REB = list(range(100, T - 21, 20))
SIG = ["p1", "p2n", "p3n", "p4"]

corrs, neutral_share = [], []
rows = []
for ti in REB:
    sc = score_at(ti)
    if sc is None:
        continue
    corrs.append(sc[SIG + ["p3o"]].corr(method="spearman").values)
    neutral_share.append(sc["neutral"].mean())
    js = np.array(sc.index)
    c0 = arr["close"][js, ti]
    r20 = arr["close"][js, min(ti + 20, T - 1)] / c0 - 1
    r60 = (arr["close"][js, min(ti + 60, T - 1)] / c0 - 1
           if ti + 60 <= T - 1 else np.full(len(js), np.nan))
    uni20, uni60 = np.nanmean(r20), np.nanmean(r60)
    for ver in ["V4n", "V4o", "p1", "p2n", "p3n", "p4"]:
        top = sc[ver].nlargest(20).index
        m = np.isin(js, top)
        rows.append(dict(date=str(pd.Timestamp(dates[ti]).date()), ver=ver,
                         ex20=np.nanmean(r20[m]) - uni20,
                         ex60=(np.nanmean(r60[m]) - uni60 if np.isfinite(uni60) else np.nan),
                         uni20=uni20))

print(f"리밸런스 {len(corrs)}회, 점화부 중립(쌍 없음) 비율 평균 {np.mean(neutral_share)*100:.1f}%")
print("\n[1] 신호 간 스피어만 상관 (시점 평균) — p3o는 V1의 눌림거래량")
print(pd.DataFrame(np.nanmean(corrs, axis=0), index=SIG + ["p3o"],
                   columns=SIG + ["p3o"]).round(2).to_string())

bt = pd.DataFrame(rows)
bt.to_csv(os.path.join(BASE, "resilience_v2_backtest.csv"), index=False)

# 온도(따뜻/차가움): 나우캐스트 W100 시리즈 기준 (V1과 동일 프로토콜)
rc = pd.read_csv(os.path.join(BASE, "nowcast_rolling_W100.csv"),
                 dtype={"ticker": str}, parse_dates=["date"])
hot = rc.dropna(subset=["state"]).groupby("date")["state"].apply(
    lambda s: s.isin(["단기", "장기"]).mean())
med_hot = float(np.median(hot.values))
gd = sorted(hot.index)
def is_warm(dstr):
    gdate = min(gd, key=lambda x: abs(pd.Timestamp(x) - pd.Timestamp(dstr)))
    return hot[gdate] >= med_hot
bt["warm"] = bt["date"].map(is_warm)

print("\n[2] 백테스트 (상위 20 vs 판정 유니버스, 20거래일 리밸런스)")
print(f"{'버전':5s} {'20일초과':>9s} {'승률':>5s} {'60일초과':>9s} {'누적':>7s} {'MDD':>7s} | {'따뜻':>8s} {'차가움':>8s}")
for ver in ["V4n", "V4o", "p1", "p2n", "p3n", "p4"]:
    g = bt[bt["ver"] == ver].sort_values("date")
    eq = (1 + (g["ex20"] + g["uni20"])).cumprod()
    mdd = ((eq / eq.cummax()) - 1).min()
    w = g[g["warm"]]; c = g[~g["warm"]]
    print(f"{ver:5s} {g['ex20'].mean()*100:+8.2f}%p {(g['ex20']>0).mean()*100:4.0f}% "
          f"{g['ex60'].mean()*100:+8.2f}%p {eq.iloc[-1]:6.2f}x {mdd*100:6.1f}% | "
          f"{w['ex20'].mean()*100:+7.2f}%p {c['ex20'].mean()*100:+7.2f}%p")
g = bt[bt["ver"] == "V4o"].sort_values("date")
un = (1 + g["uni20"]).cumprod()
print(f"유니버스: 누적 {un.iloc[-1]:.2f}x, MDD {((un/un.cummax())-1).min()*100:.1f}%")

# 현재 시점 상위 20 비교 (신 vs 구)
cur = score_at(T - 1)
tn = set(cur["V4n"].nlargest(20).index)
to = set(cur["V4o"].nlargest(20).index)
print(f"\n[3] 현재(기준일 {pd.Timestamp(dates[-1]).date()}) 상위 20 교집합: {len(tn & to)}/20")
