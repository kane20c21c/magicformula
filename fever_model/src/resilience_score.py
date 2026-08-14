#!/usr/bin/env python3
"""
탄력 점수 (Resilience Score) — "지금 움직이고 있고 눌림을 버티는 종목"의 관측
- 창: trailing 100거래일, 최근 8개 파동 (판정 요건: 창 내 파동 6개 이상 + 최근 8파동에 상승/하락 각 1개 이상)
- 4신호 횡단면 백분위(0~100) 균등 평균:
    S1 추세 우위   = log(상승파 평균진폭 / 하락파 평균진폭)          높을수록 ↑
    S2 눌림 진폭   = 하락파 평균진폭 / 상승파 평균진폭               낮을수록 ↑ (역백분위)
    S3 눌림 거래량 = 하락파 평균 vol_norm / 상승파 평균 vol_norm     낮을수록 ↑ (역백분위)
    S4 YZ 수준     = 창 내 YZ20 중앙값                               높을수록 ↑
- 버전: V4(4신호) / V3(YZ 제외) × 과열 감점(-25) 유/무
- 과열 = 마지막 상승파 진폭이 판정일 이전 자기 이력 상승파의 95백분위 이상
- 선견 없음: vol_norm 중앙값·과열 임계 모두 판정일 이전 완결 파동만 사용
- 백테스트: 20거래일 리밸런스, 상위 20 vs 판정 가능 유니버스(동일가중), 20/60일 초과수익
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
W = 100
NW_RECENT = 8
MIN_WAVES = 6
PENALTY = 25.0

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
nm = df.groupby("Ticker")["Name"].last().to_dict()

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
    return out  # (dir, amp, vol, end_idx)

# 전 이력 파동 1회 복원 (완결 판정: end_idx < ti 이면 ti 시점에 완결)
full = {j: build_waves(j, 0, T - 1) for j in range(N)}

def signals(j, ti):
    """판정일 ti의 원신호 (선견 없음). 반환: S1~S4 원값, climax, 유효성"""
    w = build_waves(j, ti - W + 1, ti)
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
    uv = np.mean([x[2] for x in up]) / vm; dv = np.mean([x[2] for x in dn]) / vm
    yz = np.nanmedian(arr["yz20"][j, ti - W + 1:ti + 1])
    if not np.isfinite(yz):
        return None
    s1 = np.log(ua / da)
    s2 = da / ua
    s3 = dv / uv if uv > 0 else np.nan
    # 과열: 마지막 상승파 vs 이력(완결 상승파, ti 이전) 95백분위
    hist_up = [x[1] for x in full[j] if x[0] > 0 and x[3] < ti]
    last_up = next((x for x in reversed(r) if x[0] > 0), None)
    climax = int(len(hist_up) >= 10 and last_up is not None
                 and last_up[1] >= np.percentile(hist_up, 95))
    if not all(np.isfinite(x) for x in (s1, s2, s3)):
        return None
    return s1, s2, s3, yz, climax

def pct(v):
    s = pd.Series(v)
    return s.rank(pct=True) * 100

def score_at(ti):
    """판정일 ti의 전 종목 점수표"""
    raw = {}
    for j in range(N):
        sg = signals(j, ti)
        if sg:
            raw[j] = sg
    if len(raw) < 30:
        return None
    d = pd.DataFrame(raw, index=["s1", "s2", "s3", "yz", "climax"]).T
    p1 = pct(d["s1"]); p2 = 100 - pct(d["s2"]); p3 = 100 - pct(d["s3"]); p4 = pct(d["yz"])
    d["V4"] = (p1 + p2 + p3 + p4) / 4
    d["V3"] = (p1 + p2 + p3) / 3
    d["V4p"] = d["V4"] - PENALTY * d["climax"]
    d["V3p"] = d["V3"] - PENALTY * d["climax"]
    d["p1"], d["p2"], d["p3"], d["p4"] = p1, p2, p3, p4
    return d

# ================= 현재 시점 점수표 =================
cur = score_at(T - 1)
sec = pd.read_csv(os.path.join(BASE, "sector_map.csv"), dtype=str)
sec.columns = ["ticker", "sector"]
smap = dict(zip(sec.ticker, sec.sector))

# 나우캐스트 국면 + 재가열 플래그 (최근 60거래일 내 장기 이력, 현재는 장기 아님)
rc = pd.read_csv(os.path.join(BASE, "nowcast_rolling_W100.csv"),
                 dtype={"ticker": str}, parse_dates=["date"])
gd = sorted(rc["date"].unique())
piv = rc.pivot(index="ticker", columns="date", values="state")
last = gd[-1]
recent12 = gd[-13:-1]  # 마지막 제외 직전 12그리드 ≈ 60거래일
reheat, nowst = {}, {}
for t in piv.index:
    nowst[t] = piv.loc[t, last]
    reheat[t] = int(piv.loc[t, last] != "장기"
                    and (piv.loc[t, recent12] == "장기").any())

out = cur.copy()
out["ticker"] = [tickers[j] for j in out.index]
out["name"] = out["ticker"].map(nm)
out["sector"] = out["ticker"].map(smap)
out["nowcast"] = out["ticker"].map(nowst)
out["reheat"] = out["ticker"].map(reheat)
cols = ["ticker", "name", "sector", "nowcast", "V4", "V4p", "V3", "climax",
        "reheat", "p1", "p2", "p3", "p4"]
out = out[cols].sort_values("V4", ascending=False)
for c in ["V4", "V4p", "V3", "p1", "p2", "p3", "p4"]:
    out[c] = out[c].round(1)
out.to_csv(os.path.join(ROOT, "탄력점수_현재.csv"), index=False, encoding="utf-8-sig")
print(f"=== 현재 점수표 (기준일 {pd.Timestamp(dates[-1]).date()}, 판정 {len(out)}종목) ===")
print("상위 20 섹터:", out.head(20)["sector"].value_counts().head(6).to_dict())
print("상위 20 나우캐스트 국면:", out.head(20)["nowcast"].value_counts(dropna=False).to_dict())
print("상위 20 중 과열 플래그:", int(out.head(20)["climax"].sum()),
      "| 재가열:", int(out.head(20)["reheat"].sum()))
print("저장: 탄력점수_현재.csv")

# ================= 롤링 백테스트 =================
hot = rc.dropna(subset=["state"]).groupby("date")["state"].apply(
    lambda s: s.isin(["단기", "장기"]).mean())
med_hot = float(np.median(hot.values))

REB = list(range(100, T - 21, 20))
rows = []
for ti in REB:
    sc = score_at(ti)
    if sc is None:
        continue
    js = np.array(sc.index)
    c0 = arr["close"][js, ti]
    c20 = arr["close"][js, min(ti + 20, T - 1)]
    r20 = c20 / c0 - 1
    r60 = (arr["close"][js, min(ti + 60, T - 1)] / c0 - 1
           if ti + 60 <= T - 1 else np.full(len(js), np.nan))
    uni20, uni60 = np.nanmean(r20), np.nanmean(r60)
    gdate = min(gd, key=lambda x: abs(pd.Timestamp(x) - pd.Timestamp(dates[ti])))
    warm = hot[gdate] >= med_hot
    for ver in ["V4", "V4p", "V3", "V3p"]:
        top = sc[ver].nlargest(20).index
        m = np.isin(js, top)
        rows.append(dict(date=str(pd.Timestamp(dates[ti]).date()), ver=ver, warm=warm,
                         ex20=np.nanmean(r20[m]) - uni20,
                         ex60=(np.nanmean(r60[m]) - uni60 if np.isfinite(uni60) else np.nan),
                         uni20=uni20))
    # 과열 플래그 직접 검정용
    cm = sc["climax"].values.astype(bool)
    rows.append(dict(date=str(pd.Timestamp(dates[ti]).date()), ver="CLIMAX", warm=warm,
                     ex20=np.nanmean(r20[cm]) - uni20 if cm.sum() >= 3 else np.nan,
                     ex60=np.nan, uni20=uni20))
bt = pd.DataFrame(rows)
bt.to_csv(os.path.join(BASE, "resilience_backtest.csv"), index=False)

print(f"\n=== 백테스트 (리밸런스 {bt['date'].nunique()}회, 20거래일 간격) ===")
print(f"{'버전':6s} {'20일 초과':>12s} {'승률':>6s} {'60일 초과':>12s} | {'따뜻':>10s} {'차가움':>10s}")
for ver in ["V4", "V4p", "V3", "V3p"]:
    g = bt[bt["ver"] == ver]
    w = g[g["warm"]]; c = g[~g["warm"]]
    print(f"{ver:6s} {g['ex20'].mean()*100:+10.2f}%p {(g['ex20']>0).mean()*100:5.0f}% "
          f"{g['ex60'].mean()*100:+10.2f}%p | {w['ex20'].mean()*100:+8.2f}%p {c['ex20'].mean()*100:+8.2f}%p")
cl = bt[bt["ver"] == "CLIMAX"].dropna(subset=["ex20"])
print(f"\n과열 플래그 종목 20일 초과수익: 평균 {cl['ex20'].mean()*100:+.2f}%p, "
      f"양수 비율 {(cl['ex20']>0).mean()*100:.0f}% (관측 {len(cl)}회)")

# 누적 곡선 + MDD (V4, 20일 홀드 체인)
for ver in ["V4", "V3"]:
    g = bt[bt["ver"] == ver].sort_values("date")
    eq = (1 + (g["ex20"] + g["uni20"])).cumprod()
    un = (1 + g["uni20"]).cumprod()
    mdd = ((eq / eq.cummax()) - 1).min()
    mddu = ((un / un.cummax()) - 1).min()
    print(f"{ver}: 누적 {eq.iloc[-1]:.2f}x vs 유니버스 {un.iloc[-1]:.2f}x, "
          f"MDD {mdd*100:.1f}% vs {mddu*100:.1f}%")
