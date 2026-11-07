# -*- coding: utf-8 -*-
"""stock_gapshock.py — 개별 종목 하루 급락(쇼크)일: 갭 vs 장중, 영향 지속성 (케인 2026-11-06~07).

regime_gapshock.py(코스피200 지수판)의 종목 확장. 자료 = data_2015/prices (수정주가 OHLCV, 2015-01-02~2026-10-29).

유니버스 (케인 2026-11-06)  연도 Y 분석 대상 = Y−1 년말 시총 > 5천억 & Y−1 일별 종가 평균 > 1,000원, 우선주·스팩 제외.
                            2015 은 2014 자료가 없어 2015-01-31 시총 · 2015-01 평균 종가로 대신 (케인 (가)).
쇼크                        종가 기준 하루 ≤ 임계 (주 −7%, 확인 −5%·−10% — 케인 2026-11-06 동의).
제외 규칙 (케인 2026-11-07 승인)
  R1 전일·당일 거래량 0   R2 시가 ≤ 0   R3 수익률·갭이 그때 제한폭(2015-06-15 전 15% / 이후 30%) 초과
  R4 상폐(데이터 끝) 전 마지막 10거래일   R5 쇼크 뒤 20거래일 안에 R1~R4 날이 있으면 그 쇼크 제외
  R6 고저가가 시가·종가를 0.5% 이내로 못 감싸는 것은 수정주가 반올림으로 허용
분류 (케인 2026-11-07, 결과 보기 전 확정)  갭 몫 = ln(1+갭)/ln(1+하루)
  ① 갭상승→붕괴 <0 · ② 장중 주도 0~0.4 · ③ 섞임 0.4~0.6 · ④ 갭+추가하락 0.6~1 · ⑤ 갭→반등(장중 상승 마감) >1
가설 (케인 2026-11-07 06:56, 사전 등록)
  H1 ① 갭상승 뒤 붕괴 → 참여자 태도가 소극적으로 변해 회복이 느리다
  H2 ⑤ 갭하락 뒤 상승 마감 → 희망 신호로 읽혀 회복이 빠르다
추가 지표 (2단계 결과를 본 뒤 케인 결정, 사후 추가로 표시)  쇼크일 저가 10일 유지 = 이후 10거래일 종가 ≥ 쇼크일 저가
  ⚠ 저가 유지는 '쿠션'(쇼크일 종가가 저가 위에서 끝난 폭)에 기계적으로 좌우된다 — ⑤ 쿠션 중앙 3.9% vs ①② 0.7~0.8%,
    같은 쿠션 구간에선 ⑤≈② (2026-11-07 확인). 보완 지표 = 10일 최대 추가 하락 dd10 (쇼크일 종가 기준, 케인 2026-11-07).
  결과 지표 묶음 (케인 2026-11-07): 10일 안 회복 · 20일 안 회복 (이후 종가 ≥ 쇼크 전날 종가) · 저가 10일 유지 · 10일 최대 추가 하락
목표 (케인 2026-11-07)  범위(시장·섹터·종목) × 쇼크 크기 × 쇼크 형태 → 이후 주가 움직임
섹터 (케인 2026-11-07)  KRX 공식 업종 (fetch_sector_indices.py → data_idx/). 연도 Y 업종 = Y−1 연말 구성, 겹치면 가장 세분된 업종.
  소속부(2181~2184)·코스닥 글로벌(2189)·제조 집계(1027·2024) 제외 → 44 업종.
  미매핑(옛 코스닥 IT 지수 폐지로 2024 이전 ~9%)은 (나): 가장 가까운 뒤 스냅샷 업종으로 소급(없으면 앞 스냅샷) — map_src 로 표시.
  종목 쇼크의 섹터 맥락 = 같은 업종 다른 종목(prices 전체, 정상 행만, 자기 제외, 5종목 이상) 그날 수익률 동일가중 평균.
크기 구간 (케인 2026-11-07, 범위 간 같은 드묾으로 맞춤)
  종목 −5~−7 / −7~−10 / ≤−10%   섹터 −3~−4.5 / −4.5~−6 / ≤−6%   지수(코스피200) −2~−3 / −3~−5 / ≤−5%
맥락 (종목 쇼크)  시장 동반 = 그날 코스피200 ≤ −2% · 섹터 동반 = 시장 아님 & 업종 동료 ≤ −3% · 단독 = 둘 다 아님
보류 구간 2026-06-01~ 은 6단계 전까지 열지 않는다 (HOLD).
사용  python3 stock_gapshock.py events   → out/regime/stock_shock_events.parquet (임계 −5% 이하 전부, 분류 전 원자료)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; D = HERE / "data_2015"; OUT = HERE / "out" / "regime"
HOLD = pd.Timestamp("2026-06-01"); LIMIT_CHANGE = pd.Timestamp("2015-06-15"); TH_MIN = -0.05

def universe(p, meta):
    p = p.assign(year=p.date.dt.year); m = meta.reset_index()
    ye = m[m.date.dt.month == 12][["ticker", "date", "mktcap", "is_pref", "is_spac"]].copy(); ye["year"] = ye.date.dt.year + 1
    f15 = m[m.date == m.date.min()][["ticker", "mktcap", "is_pref", "is_spac"]].assign(year=2015)
    avg = p[p.close > 0].groupby(["ticker", "year"]).close.mean().rename("avgc").reset_index(); avg["year"] += 1
    a15 = p[(p.date < "2015-02-01") & (p.close > 0)].groupby("ticker").close.mean().rename("avgc").reset_index().assign(year=2015)
    u = pd.concat([ye.drop(columns="date").merge(avg, on=["ticker", "year"], how="left"), f15.merge(a15, on=["ticker", "year"], how="left")])
    u = u[(u.mktcap > 5e11) & (u.avgc > 1000) & ~u.is_pref.astype(bool) & ~u.is_spac.astype(bool)]
    return u[["ticker", "year"]].drop_duplicates()

def build_events():
    p = pd.read_parquet(D / "prices.parquet").reset_index().sort_values(["ticker", "date"]).reset_index(drop=True)
    k = pd.read_parquet(D / "kospi200_daily.parquet")[["Open", "High", "Low", "Close"]].astype(float); k.index = pd.to_datetime(k.index)
    g = p.groupby("ticker", sort=False)
    p["pc"] = g.close.shift(1); p["pv"] = g.volume.shift(1)
    p["r"] = p.close / p.pc - 1; p["gap"] = p.open / p.pc - 1; p["intra"] = p.close / p.open - 1
    end = p.date.max(); last = g.date.transform("max")
    lim = np.where(p.date < LIMIT_CHANGE, 0.1505, 0.3005)
    p["bad"] = ((p.pv == 0) | (p.volume == 0) | (p.open <= 0) | p.open.isna() | (p.r.abs() > lim) | (p.gap.abs() > lim)
                | ((last < end - pd.Timedelta(days=10)) & (g.cumcount(ascending=False) < 10)))
    gb = p.groupby("ticker", sort=False).bad
    p["bad_fwd"] = np.logical_or.reduce([gb.shift(-j).fillna(False).to_numpy(dtype=bool) for j in range(1, 21)])
    p["n_after"] = g.cumcount(ascending=False); p["pos"] = g.cumcount()
    # 일중 폭·갭 크기의 직전 20일 평균 (정지 등 bad 날 제외)
    p["rng"] = ((p.high - p.low) / p.pc).where(~p.bad); ag = p.gap.abs().where(~p.bad)
    roll = lambda s: s.groupby(p.ticker, sort=False).rolling(20, min_periods=15).mean().reset_index(level=0, drop=True).sort_index()
    p["rng20"] = roll(p.rng).groupby(p.ticker, sort=False).shift(1); p["agap20"] = roll(ag).groupby(p.ticker, sort=False).shift(1)
    p["vol20"] = roll(p.volume.where(~p.bad)).groupby(p.ticker, sort=False).shift(1)
    # 이후 경로는 후보 행만 위치 인덱싱으로 (같은 종목 보장 = n_after ≥ j)
    cand = np.flatnonzero(((p.r <= TH_MIN) & ~p.bad & ~p.bad_fwd & (p.n_after >= 20)).to_numpy())
    C = p.close.to_numpy(); H = p.high.to_numpy(); G = p.gap.to_numpy(); R = p.rng.to_numpy(); NA = p.n_after.to_numpy(); PC = p.pc.to_numpy()
    at = lambda arr, j: np.where(NA[cand] >= j, arr[np.minimum(cand + j, len(arr) - 1)], np.nan)
    ev = p.iloc[cand].copy()
    for kk in (1, 3, 5, 10, 20): ev[f"f{kk}"] = at(C, kk) / C[cand] - 1
    ev["next_gap"] = at(G, 1); V = p.volume.to_numpy()
    ev["vol_n"] = V[cand] / ev.vol20.to_numpy()                                   # 쇼크일 거래량 ÷ 직전 20일 평균
    ev["vq1_5"] = np.nanmean([at(V, j) for j in range(1, 6)], axis=0) / ev.vol20.to_numpy()   # 이후 1~5일 참여(거래량) 배수
    ev["vq6_20"] = np.nanmean([at(V, j) for j in range(6, 21)], axis=0) / ev.vol20.to_numpy()
    ev["v1_5"] = np.nanmean([at(R, j) for j in range(1, 6)], axis=0) / ev.rng20.to_numpy()
    ev["v6_20"] = np.nanmean([at(R, j) for j in range(6, 21)], axis=0) / ev.rng20.to_numpy()
    ev["fill5"] = np.nanmax([at(H, j) for j in range(1, 6)], axis=0) >= PC[cand]
    # 쇼크일 저가 10일 유지 (케인 2026-11-07 추가 — 2단계 결과를 본 뒤 넣은 지표): 이후 10거래일 종가(주)·장중 저가(보조)가 쇼크일 저가 이상
    LO = p.low.to_numpy()
    mn10 = np.nanmin([at(C, j) for j in range(1, 11)], axis=0)
    ev["hold10"] = mn10 >= LO[cand]
    ev["dd10"] = mn10 / C[cand] - 1          # 10일 최대 추가 하락: 쇼크일 종가 → 이후 10거래일 최저 종가 (케인 2026-11-07 — 저가 유지의 쿠션 편향 보완)
    ev["hold10_low"] = np.nanmin([at(LO, j) for j in range(1, 11)], axis=0) >= LO[cand]
    cl = np.array([at(C, j) for j in range(1, 61)]).T; okm = cl >= PC[cand][:, None]
    ev["recover"] = np.where(okm.any(1), okm.argmax(1) + 1, np.nan)
    # 지수 (코스피200) 같은 날·이후
    kc = k.Close; kr = kc / kc.shift(1) - 1
    kf = {kk: kc.shift(-kk) / kc - 1 for kk in (1, 3, 5, 10, 20)}
    ev["year"] = ev.date.dt.year
    ev = ev.merge(universe(p, pd.read_parquet(D / "meta.parquet")), on=["ticker", "year"])
    ev["k_r"] = ev.date.map(kr)
    for kk in (1, 3, 5, 10, 20): ev[f"x{kk}"] = ev[f"f{kk}"] - ev.date.map(kf[kk])
    ev["share"] = np.log1p(ev.gap) / np.log1p(ev.r)
    hl = (ev.high - ev.low).replace(0, np.nan)
    ev["clv"] = (ev.close - ev.low) / hl                      # 마감 위치 0=저가 마감, 1=고가 마감
    ev["h_pct"] = ev.high / ev.pc - 1; ev["l_pct"] = ev.low / ev.pc - 1
    ev["up_from_open"] = (ev.high / ev.open - 1)             # 시가 뒤 고가까지 오른 폭
    ev["rebound"] = ev.close / ev.low - 1                     # 저가에서 반등한 폭
    ev["rng_n"] = ((ev.high - ev.low) / ev.pc) / ev.rng20
    ev["gap_n"] = ev.gap.abs() / ev.agap20
    # 첫 쇼크: 같은 종목 앞 10거래일 안에 −임계 쇼크 없음 (임계별로 다시 계산하므로 여기선 날짜 순번만)
    # pos = 종목 안 거래일 순번 (첫 쇼크 판정용)
    ev["hold"] = ev.date >= HOLD
    cols = ["date", "ticker", "year", "hold", "pos", "pc", "open", "high", "low", "close", "volume", "r", "gap", "intra", "share", "clv",
            "h_pct", "l_pct", "up_from_open", "rebound", "rng_n", "gap_n", "rng20", "k_r", "next_gap", "vol_n", "vq1_5", "vq6_20", "v1_5", "v6_20", "fill5", "hold10", "hold10_low", "dd10", "recover"] + \
           [f"f{kk}" for kk in (1, 3, 5, 10, 20)] + [f"x{kk}" for kk in (1, 3, 5, 10, 20)]
    OUT.mkdir(parents=True, exist_ok=True); ev[cols].to_parquet(OUT / "stock_shock_events.parquet")
    print(f"events: {len(ev):,} (≤{TH_MIN:.0%}) · 학습 {int((~ev.hold).sum()):,} · 보류 {int(ev.hold.sum()):,} → {OUT/'stock_shock_events.parquet'}")

EXCL_CODES = {"2181", "2182", "2183", "2184", "2189", "1027", "2024"}

def sector_map():
    """(ticker, year) → 업종 코드·이름. 연도 Y = Y−1 연말 스냅샷. 미매핑은 가까운 뒤(없으면 앞) 스냅샷으로 소급."""
    m = pd.read_parquet(HERE / "data_idx" / "krx_sector_members.parquet"); m = m[~m.code.isin(EXCL_CODES)]
    m = m.merge(m.groupby(["snap", "code"]).size().rename("n").reset_index(), on=["snap", "code"])
    leaf = m.sort_values("n").groupby(["snap", "ticker"]).first().reset_index()
    leaf["year"] = pd.to_datetime(leaf.snap).dt.year + 1
    leaf = leaf[["ticker", "year", "code", "name"]]
    years = range(2015, 2027); tick = leaf.ticker.unique()
    full = pd.MultiIndex.from_product([tick, years], names=["ticker", "year"]).to_frame(index=False).merge(leaf, how="left")
    full["map_src"] = np.where(full.code.notna(), "krx", None)
    full = full.sort_values(["ticker", "year"])
    for c in ("code", "name"):
        full[c] = full.groupby("ticker")[c].transform(lambda s: s.bfill().ffill())
    full["map_src"] = full.map_src.fillna("backfill")
    return full

def build_context():
    ev = pd.read_parquet(OUT / "stock_shock_events.parquet")
    sm = sector_map()
    p = pd.read_parquet(D / "prices.parquet", columns=["close", "volume"]).reset_index().sort_values(["ticker", "date"])
    g = p.groupby("ticker", sort=False); p["r"] = p.close / g.close.shift(1) - 1; pv = g.volume.shift(1)
    lim = np.where(p.date < LIMIT_CHANGE, 0.1505, 0.3005)
    p = p[(pv > 0) & (p.volume > 0) & (p.r.abs() <= lim)]
    p["year"] = p.date.dt.year; p = p.merge(sm[["ticker", "year", "code"]], on=["ticker", "year"])
    agg = p.groupby(["date", "code"]).r.agg(["sum", "count"]).reset_index()
    ev = ev.merge(sm[["ticker", "year", "code", "name", "map_src"]], on=["ticker", "year"], how="left")
    ev = ev.merge(agg, on=["date", "code"], how="left")
    own = ev.r.where(ev.r.abs() <= 0.3005)        # 자기 자신 빼기 (자기 행은 정상 행이라 agg 에 들어 있다)
    ev["peer_n"] = ev["count"] - 1
    ev["peer_r"] = np.where(ev.peer_n >= 5, (ev["sum"] - own) / ev.peer_n, np.nan)
    ev["ctx"] = np.select([ev.k_r <= -0.02, ev.peer_r <= -0.03], ["시장 동반", "섹터 동반"], "단독")
    ev.loc[ev.k_r.isna() | (ev.peer_r.isna() & (ev.k_r > -0.02)), "ctx"] = "맥락 미상"
    ev = ev.drop(columns=["sum", "count"]); ev.to_parquet(OUT / "stock_shock_events_ctx.parquet")
    print(f"context: {len(ev):,} · 업종 매핑 {ev.code.notna().mean():.1%} (소급 {(ev.map_src=='backfill').mean():.1%}) · 맥락 {ev.ctx.value_counts().to_dict()}")

def add_beta(win=60):
    """종목 쇼크에 직전 60거래일 베타(코스피200) + 베타 보정 초과수익(xb5·xb20) 추가 (케인 2026-11-07 승인, 확인용 열)."""
    ev = pd.read_parquet(OUT / "stock_shock_events_ctx.parquet")
    p = pd.read_parquet(D / "prices.parquet", columns=["close"]).reset_index().sort_values(["ticker", "date"]).reset_index(drop=True)
    p["r"] = p.groupby("ticker").close.pct_change()
    k = pd.read_parquet(D / "kospi200_daily.parquet").Close.astype(float); k.index = pd.to_datetime(k.index)
    p["kr"] = p.date.map(k.pct_change())
    p.loc[p.r.abs() > 0.3005, "r"] = np.nan
    key = pd.Series(np.arange(len(p)), index=pd.MultiIndex.from_frame(p[["ticker", "date"]]))
    pos = key.reindex(pd.MultiIndex.from_frame(ev[["ticker", "date"]])).to_numpy()
    R, KR = p.r.to_numpy(), p.kr.to_numpy(); TK = p.ticker.to_numpy(); beta = np.full(len(ev), np.nan)
    for j, i in enumerate(pos):
        if np.isnan(i) or i < win: continue
        i = int(i); sl = slice(i - win, i)
        if TK[i - win] != TK[i]: continue
        a, b = R[sl], KR[sl]; m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() >= 40: beta[j] = np.cov(a[m], b[m])[0, 1] / np.var(b[m], ddof=1)
    ev["beta60"] = beta
    for kk in (5, 20):
        kf = ev[f"f{kk}"] - ev[f"x{kk}"]; ev[f"xb{kk}"] = ev[f"f{kk}"] - ev.beta60 * kf
    ev.to_parquet(OUT / "stock_shock_events_ctx.parquet")
    print(f"beta: {np.isfinite(beta).mean():.1%} 채움 · 중앙 {np.nanmedian(beta):.2f} (p10 {np.nanpercentile(beta,10):.2f} / p90 {np.nanpercentile(beta,90):.2f})")

def series_events(d, th_hi, name=""):
    """OHLC 시계열 하나(지수·업종지수)의 쇼크 표. d: date 인덱스, Open/High/Low/Close(/Volume). th_hi = 작은 쇼크 상한(예 −0.02)."""
    d = d.sort_index(); c, o, h, l = (d[x].to_numpy(float) for x in ("Close", "Open", "High", "Low"))
    v = d["Volume"].to_numpy(float) if "Volume" in d else np.full(len(d), np.nan)
    pc = np.r_[np.nan, c[:-1]]; r = c / pc - 1; gap = o / pc - 1; rng = (h - l) / pc
    rng20 = pd.Series(rng).rolling(20, min_periods=15).mean().shift(1).to_numpy()
    v20 = pd.Series(v).rolling(20, min_periods=15).mean().shift(1).to_numpy(); n = len(d); rows = []
    for t in np.flatnonzero(r <= th_hi):
        if t + 20 >= n or o[t] <= 0 or not np.isfinite(r[t]): continue
        seg = c[t + 1:min(n, t + 61)]; kk = np.flatnonzero(seg >= pc[t])
        rows.append(dict(name=name, date=d.index[t], r=r[t], gap=gap[t], share=np.log1p(gap[t]) / np.log1p(r[t]),
            f5=c[t + 5] / c[t] - 1, f10=c[t + 10] / c[t] - 1, f20=c[t + 20] / c[t] - 1,
            recover=kk[0] + 1 if kk.size else np.nan, fill5=h[t + 1:t + 6].max() >= pc[t], hold10=c[t + 1:t + 11].min() >= l[t],
            dd10=c[t + 1:t + 11].min() / c[t] - 1,
            v1_5=np.nanmean(rng[t + 1:t + 6]) / rng20[t], vol_n=v[t] / v20[t] if np.isfinite(v20[t]) and v20[t] > 0 else np.nan, vq1_5=np.nanmean(v[t + 1:t + 6]) / v20[t] if np.isfinite(v20[t]) and v20[t] > 0 else np.nan))
    return pd.DataFrame(rows)

def build_scopes():
    """섹터(KRX 업종 44)·지수(코스피200 2000~) 쇼크 이벤트 → out/regime/sector_shock_events.parquet · index_shock_events.parquet"""
    ix = pd.read_parquet(HERE / "data_idx" / "krx_sector_indices.parquet"); ix["date"] = pd.to_datetime(ix.date)
    codes = set(pd.read_parquet(HERE / "data_idx" / "krx_sector_members.parquet").code) - EXCL_CODES
    k = pd.read_parquet(D / "kospi200_daily.parquet")[["Open", "High", "Low", "Close", "Volume"]].astype(float); k.index = pd.to_datetime(k.index)
    a = pd.read_parquet(HERE / "data_idx" / "kospi200_2000_2014.parquet")[["Open", "High", "Low", "Close"]].astype(float); a.index = pd.to_datetime(a.index)
    kk = pd.concat([a, k[k.index > a.index.max()]]).sort_index()
    kret = {j: kk.Close.shift(-j) / kk.Close - 1 for j in (5, 10, 20)}; kr = kk.Close.pct_change()
    sec = []
    for code, g in ix[ix.code.isin(codes)].groupby("code"):
        d = g.set_index("date").rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
        e = series_events(d, -0.03, g.name.iloc[0]); e["code"] = code; e["market"] = g.market.iloc[0]; sec.append(e)
    sec = pd.concat(sec, ignore_index=True); sec["k_r"] = sec.date.map(kr)
    for j in (5, 10, 20): sec[f"x{j}"] = sec[f"f{j}"] - sec.date.map(kret[j])
    sec["hold"] = sec.date >= HOLD; sec.to_parquet(OUT / "sector_shock_events.parquet")
    ide = series_events(kk, -0.02, "코스피200"); ide["hold"] = ide.date >= HOLD
    ide.to_parquet(OUT / "index_shock_events.parquet")
    print(f"scopes: 섹터 {len(sec):,} (학습 {int((~sec.hold).sum()):,}) · 지수 {len(ide):,} (학습 {int((~ide.hold).sum()):,})")

if __name__ == "__main__":
    if sys.argv[1:2] == ["beta"]: add_beta()
    if sys.argv[1:2] == ["scopes"]: build_scopes()
    if sys.argv[1:2] == ["events"]: build_events()
    if sys.argv[1:2] == ["context"]: build_context()
