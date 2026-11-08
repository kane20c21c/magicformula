# -*- coding: utf-8 -*-
"""stock_surprise.py — 개별 종목·업종·지수 하루 급등(서프라이즈)일: 갭 vs 장중, 이후 경로 (케인 2026-11-08).

stock_gapshock.py(급락)의 거울상. 유니버스·제외 규칙 R1~R6·업종 매핑은 stock_gapshock 의 함수를 그대로 쓴다 (import).
정의 (케인 2026-11-08 — 결과 보기 전 확정)
  급등   종가 기준 하루 r ≥ +5% (종목) · ≥ +3% (KRX 업종지수) · ≥ +2% (코스피200). 급락 임계를 그대로 뒤집음 — 분포를 본 뒤 재판단.
  크기   종목 작은 [+5, +7) · 중간 [+7, +10) · 큰 ≥ +10%   섹터 [+3, +4.5) / [+4.5, +6) / ≥ +6%   지수 [+2, +3) / [+3, +5) / ≥ +5%
  상한가 분리하지 않음 (표본 적어 일반화 어려움 — 케인)
  범위   시장 동반 = 코스피200 ≥ +2% · 섹터 동반 = 시장 아님 & 업종 동료 평균 ≥ +3% · 단독 = 그 밖 · 맥락 미상
  형태   갭 몫 φ = ln(1+갭)/ln(1+r)
         ① 갭하락→급등 φ<0 · ② 장중 주도 0~0.4 · ③ 섞임 0.4~0.6 · ④ 갭+추가상승 0.6~1 · ⑤ 갭상승→밀림(장중 하락 마감) φ>1
  거래량 v = 당일 ÷ 직전 20일 평균 (정상 행, 최소 15일) — 저 <1 · 중 1~2 · 고 ≥2 (분포 본 뒤 재상의)
  꼬리표 pre_shock = 직전 10거래일(t−10…t−1)에 같은 종목 r ≤ −5% · pre_mkt = 직전 10거래일에 코스피200 ≤ −2%
결과 지표 (케인 2026-11-08, 5일 추가)  pc = 급등 전날 종가, H = 급등일 고가, C = 급등일 종가
  g5/g10/g20     k일 안 종가 ≤ pc 인 날 있음 (급등분 전부 반납)
  half5/half10   k일 안 종가 ≤ pc + ½(C − pc) (상승분 절반 반납)
  b5/b10         k일 안 종가 > H (급등일 고가를 종가로 돌파 — 2차 상승)
  du5/du10 · dd5/dd10   이후 k일 최고·최저 종가 ÷ C − 1
  a5/a10/a20     C_{t+k} ÷ O_{t+1} − 1 − 0.23% (다음 날 시가 진입, 왕복 비용)   f/x = 종가 기준 수익·코스피200 초과
학습 2015-01-02 ~ 2026-04-30 · 보류 2026-05-01 ~ (HOLD, 케인 2026-11-08) — 5단계 전까지 열지 않는다.
사용  python3 stock_surprise.py events | context | beta | scopes | step1
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from stock_gapshock import universe, sector_map, LIMIT_CHANGE, D, OUT, HERE, EXCL_CODES
HOLD = pd.Timestamp("2026-05-01"); TH = 0.05; COST = 0.0023
EV = OUT / "stock_surprise_events.parquet"; EVC = OUT / "stock_surprise_events_ctx.parquet"

def _k200():
    k = pd.read_parquet(D / "kospi200_daily.parquet")[["Open", "High", "Low", "Close", "Volume"]].astype(float); k.index = pd.to_datetime(k.index)
    return k.sort_index()

def build_events():
    p = pd.read_parquet(D / "prices.parquet").reset_index().sort_values(["ticker", "date"]).reset_index(drop=True)
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
    p["vol20"] = (p.volume.where(~p.bad).groupby(p.ticker, sort=False).rolling(20, min_periods=15).mean()
                  .reset_index(level=0, drop=True).sort_index().groupby(p.ticker, sort=False).shift(1))
    # 직전 10거래일 같은 종목 급락(≤ −5%, 정상 행) 여부
    sh = ((p.r <= -0.05) & ~p.bad).astype(float)
    p["pre_shock"] = sh.groupby(p.ticker, sort=False).transform(lambda s: s.shift(1).rolling(10, min_periods=1).max()).fillna(0).astype(bool)
    cand = np.flatnonzero(((p.r >= TH) & ~p.bad & ~p.bad_fwd & (p.n_after >= 20)).to_numpy())
    C = p.close.to_numpy(); O = p.open.to_numpy(); G = p.gap.to_numpy(); NA = p.n_after.to_numpy(); V = p.volume.to_numpy()
    at = lambda arr, j: np.where(NA[cand] >= j, arr[np.minimum(cand + j, len(arr) - 1)], np.nan)
    ev = p.iloc[cand].copy(); c0 = C[cand]; pc = p.pc.to_numpy()[cand]; h0 = p.high.to_numpy()[cand]
    cl = np.array([at(C, j) for j in range(1, 21)]).T            # 이후 1~20일 종가
    for kk in (1, 3, 5, 10, 20): ev[f"f{kk}"] = cl[:, kk - 1] / c0 - 1
    o1 = at(O, 1)
    for kk in (5, 10, 20): ev[f"a{kk}"] = cl[:, kk - 1] / o1 - 1 - COST
    half = pc + 0.5 * (c0 - pc)
    for kk in (5, 10, 20): ev[f"g{kk}"] = (cl[:, :kk] <= pc[:, None]).any(1)
    for kk in (5, 10):
        ev[f"half{kk}"] = (cl[:, :kk] <= half[:, None]).any(1)
        ev[f"b{kk}"] = (cl[:, :kk] > h0[:, None]).any(1)
        ev[f"du{kk}"] = cl[:, :kk].max(1) / c0 - 1; ev[f"dd{kk}"] = cl[:, :kk].min(1) / c0 - 1
    ev["next_gap"] = at(G, 1); ev["vol_n"] = V[cand] / ev.vol20.to_numpy()
    k = _k200(); kc = k.Close; kr = kc.pct_change()
    ev["year"] = ev.date.dt.year
    ev = ev.merge(universe(p, pd.read_parquet(D / "meta.parquet")), on=["ticker", "year"])
    ev["k_r"] = ev.date.map(kr)
    kmin10 = kr.shift(1).rolling(10, min_periods=1).min(); ev["pre_mkt"] = ev.date.map(kmin10) <= -0.02
    for kk in (1, 3, 5, 10, 20): ev[f"x{kk}"] = ev[f"f{kk}"] - ev.date.map(kc.shift(-kk) / kc - 1)
    ev["share"] = np.log1p(ev.gap) / np.log1p(ev.r)
    hl = (ev.high - ev.low).replace(0, np.nan); ev["clv"] = (ev.close - ev.low) / hl
    ev["limit_up"] = ev.r >= np.where(ev.date < LIMIT_CHANGE, 0.1495, 0.2995) - 0.0005
    ev["hold"] = ev.date >= HOLD
    cols = ["date", "ticker", "year", "hold", "pos", "pc", "open", "high", "low", "close", "volume", "r", "gap", "intra", "share", "clv",
            "limit_up", "k_r", "next_gap", "vol_n", "pre_shock", "pre_mkt"] + [f"f{k}" for k in (1, 3, 5, 10, 20)] + [f"x{k}" for k in (1, 3, 5, 10, 20)] + \
           ["a5", "a10", "a20", "g5", "g10", "g20", "half5", "half10", "b5", "b10", "du5", "du10", "dd5", "dd10"]
    OUT.mkdir(parents=True, exist_ok=True); ev[cols].to_parquet(EV)
    print(f"events: {len(ev):,} (≥+{TH:.0%}) · 학습 {int((~ev.hold).sum()):,} · 보류 {int(ev.hold.sum()):,} → {EV}")

def build_context():
    ev = pd.read_parquet(EV); sm = sector_map()
    p = pd.read_parquet(D / "prices.parquet", columns=["close", "volume"]).reset_index().sort_values(["ticker", "date"])
    g = p.groupby("ticker", sort=False); p["r"] = p.close / g.close.shift(1) - 1; pv = g.volume.shift(1)
    lim = np.where(p.date < LIMIT_CHANGE, 0.1505, 0.3005)
    p = p[(pv > 0) & (p.volume > 0) & (p.r.abs() <= lim)]
    p["year"] = p.date.dt.year; p = p.merge(sm[["ticker", "year", "code"]], on=["ticker", "year"])
    agg = p.groupby(["date", "code"]).r.agg(["sum", "count"]).reset_index()
    ev = ev.merge(sm[["ticker", "year", "code", "name", "map_src"]], on=["ticker", "year"], how="left").merge(agg, on=["date", "code"], how="left")
    own = ev.r.where(ev.r.abs() <= 0.3005); ev["peer_n"] = ev["count"] - 1
    ev["peer_r"] = np.where(ev.peer_n >= 5, (ev["sum"] - own) / ev.peer_n, np.nan)
    ev["ctx"] = np.select([ev.k_r >= 0.02, ev.peer_r >= 0.03], ["시장 동반", "섹터 동반"], "단독")
    ev.loc[ev.k_r.isna() | (ev.peer_r.isna() & (ev.k_r < 0.02)), "ctx"] = "맥락 미상"
    ev = ev.drop(columns=["sum", "count"]); ev.to_parquet(EVC)
    print(f"context: {len(ev):,} · 업종 매핑 {ev.code.notna().mean():.1%} (소급 {(ev.map_src=='backfill').mean():.1%}) · 맥락 {ev.ctx.value_counts().to_dict()}")

def add_beta(win=60):
    ev = pd.read_parquet(EVC)
    p = pd.read_parquet(D / "prices.parquet", columns=["close"]).reset_index().sort_values(["ticker", "date"]).reset_index(drop=True)
    p["r"] = p.groupby("ticker").close.pct_change(); k = _k200().Close; p["kr"] = p.date.map(k.pct_change())
    p.loc[p.r.abs() > 0.3005, "r"] = np.nan
    key = pd.Series(np.arange(len(p)), index=pd.MultiIndex.from_frame(p[["ticker", "date"]]))
    pos = key.reindex(pd.MultiIndex.from_frame(ev[["ticker", "date"]])).to_numpy()
    R, KR, TK = p.r.to_numpy(), p.kr.to_numpy(), p.ticker.to_numpy(); beta = np.full(len(ev), np.nan)
    for j, i in enumerate(pos):
        if np.isnan(i) or i < win: continue
        i = int(i); sl = slice(i - win, i)
        if TK[i - win] != TK[i]: continue
        a, b = R[sl], KR[sl]; m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() >= 40: beta[j] = np.cov(a[m], b[m])[0, 1] / np.var(b[m], ddof=1)
    ev["beta60"] = beta
    for kk in (5, 20): ev[f"xb{kk}"] = ev[f"f{kk}"] - ev.beta60 * (ev[f"f{kk}"] - ev[f"x{kk}"])
    ev.to_parquet(EVC); print(f"beta: {np.isfinite(beta).mean():.1%} 채움 · 중앙 {np.nanmedian(beta):.2f}")

def series_up(d, th, name=""):
    """OHLC 시계열(지수·업종지수) 급등 표 — stock_gapshock.series_events 의 거울상 + 급등 결과 지표."""
    d = d.sort_index(); c, o, h, l = (d[x].to_numpy(float) for x in ("Close", "Open", "High", "Low"))
    v = d["Volume"].to_numpy(float) if "Volume" in d else np.full(len(d), np.nan)
    pc = np.r_[np.nan, c[:-1]]; r = c / pc - 1; gap = o / pc - 1
    v20 = pd.Series(v).rolling(20, min_periods=15).mean().shift(1).to_numpy(); n = len(d); rows = []
    for t in np.flatnonzero(r >= th):
        if t + 20 >= n or o[t] <= 0 or not np.isfinite(r[t]): continue
        f = c[t + 1:t + 21]; half = pc[t] + 0.5 * (c[t] - pc[t])
        rows.append(dict(name=name, date=d.index[t], r=r[t], gap=gap[t], share=np.log1p(gap[t]) / np.log1p(r[t]),
            f5=f[4] / c[t] - 1, f10=f[9] / c[t] - 1, f20=f[19] / c[t] - 1,
            g5=(f[:5] <= pc[t]).any(), g10=(f[:10] <= pc[t]).any(), g20=(f <= pc[t]).any(),
            half5=(f[:5] <= half).any(), half10=(f[:10] <= half).any(), b5=(f[:5] > h[t]).any(), b10=(f[:10] > h[t]).any(),
            du5=f[:5].max() / c[t] - 1, du10=f[:10].max() / c[t] - 1, dd5=f[:5].min() / c[t] - 1, dd10=f[:10].min() / c[t] - 1,
            vol_n=v[t] / v20[t] if np.isfinite(v20[t]) and v20[t] > 0 else np.nan))
    return pd.DataFrame(rows)

def build_scopes():
    ix = pd.read_parquet(HERE / "data_idx" / "krx_sector_indices.parquet"); ix["date"] = pd.to_datetime(ix.date)
    codes = set(pd.read_parquet(HERE / "data_idx" / "krx_sector_members.parquet").code) - EXCL_CODES
    k = _k200()
    a = pd.read_parquet(HERE / "data_idx" / "kospi200_2000_2014.parquet")[["Open", "High", "Low", "Close"]].astype(float); a.index = pd.to_datetime(a.index)
    kk = pd.concat([a, k[k.index > a.index.max()]]).sort_index(); kr = kk.Close.pct_change()
    kret = {j: kk.Close.shift(-j) / kk.Close - 1 for j in (5, 10, 20)}
    sec = []
    for code, g in ix[ix.code.isin(codes)].groupby("code"):
        d = g.set_index("date").rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
        e = series_up(d, 0.03, g.name.iloc[0]); e["code"] = code; e["market"] = g.market.iloc[0]; sec.append(e)
    sec = pd.concat(sec, ignore_index=True); sec["k_r"] = sec.date.map(kr)
    for j in (5, 10, 20): sec[f"x{j}"] = sec[f"f{j}"] - sec.date.map(kret[j])
    sec["ctx"] = np.where(sec.k_r >= 0.02, "시장 동반", "섹터 단독")
    sec["hold"] = sec.date >= HOLD; sec.to_parquet(OUT / "sector_surprise_events.parquet")
    ide = series_up(kk, 0.02, "코스피200"); ide["hold"] = ide.date >= HOLD; ide.to_parquet(OUT / "index_surprise_events.parquet")
    print(f"scopes: 섹터 {len(sec):,} (학습 {int((~sec.hold).sum()):,}) · 지수 {len(ide):,} (학습 {int((~ide.hold).sum()):,})")

if __name__ == "__main__":
    cmd = sys.argv[1:2]
    if cmd == ["events"]: build_events()
    if cmd == ["context"]: build_context()
    if cmd == ["beta"]: add_beta()
    if cmd == ["scopes"]: build_scopes()
