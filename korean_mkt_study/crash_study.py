# -*- coding: utf-8 -*-
"""crash_study.py — 급락 종목 연구 (케인 2026-10-30). 데이터 = data_2015/.

목적: 하락이 시작된 시점에서 '진짜 급락'을 며칠 안에 가려내 손절 판단에 쓸 수 있는 거래 특성을 찾는다 (예측이 아니라 초기 판별).

정의 (2026-10-30 케인 합의):
  유니버스  전년도 일평균 거래대금 ≥ 10억 AND 전년 12월말 시총 ≥ 1,000억 (하나라도 미달이면 그 해 제외 = 'OR 제외').
            전년 자료 없는 종목(신규상장)은 첫 해 제외. 상장폐지 종목의 마지막 10거래일 안 에피소드 제외(정리매매).
  에피소드  3거래일 누적 수익률 ≤ -10% 가 처음 성립한 날 s (판단 시점). 같은 종목 20거래일 쿨다운. 기준가 p0 = s-3 종가.
  라벨      s+1..s+7 최저 종가 / p0 - 1 ≤ -30%  → crash(진짜 급락)
            아니면 s+7 종가 / s 종가 - 1 : ≥ +5% 되돌림 / ≤ -5% 완만추가하락 / 그 사이 횡보
  태그      시장동반 = 소속 시장 지수(KOSPI→KOSPI, KOSDAQ→KOSDAQ150) 의 [s-3, s+7] 수익률 ≤ -10% (사후 태그)
            시장동반_s = 지수 3일 수익률(s 시점) ≤ -4% (판단 시점에 알 수 있는 버전)
            시총 구간(전년말) · 소속시장 · 10일 하락일 수 · 최악 1일 비중
사용: python3 crash_study.py episodes   → out/crash/episodes.parquet + 요약 출력
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data_2015"; OUT = HERE / "out" / "crash"; OUT.mkdir(parents=True, exist_ok=True)
LLV_IDX = HERE.parents[1] / "LongLiveVault" / "data" / "ohlcv" / "tickers"
END = pd.Timestamp("2026-10-29")
AMT_MIN, CAP_MIN = 1e9, 1e11
DROP3, CRASH, HORIZON, COOLDOWN = -0.10, -0.30, 7, 20
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def load_prices() -> pd.DataFrame:
    # ⚠ VM 메모리 3.9GB — 티커는 category, 수치는 float32, 중간 컬럼은 즉시 삭제
    px = pd.read_parquet(DATA / "prices.parquet", columns=["close", "volume"]).reset_index()
    px["ticker"] = px.ticker.astype("category"); px["close"] = px.close.astype("float32")
    px = px.sort_values(["ticker", "date"]).reset_index(drop=True)
    px["year"] = px.date.dt.year.astype("int16")
    return px

def eligibility(px: pd.DataFrame) -> pd.DataFrame:
    """(ticker, year) → 포함 여부 (전년도 기준) + 전년말 시총·시장"""
    amt = (px.close.astype("float64") * px.volume).groupby([px.ticker.astype(str), px.year]).mean().rename("amt_avg")
    amt.index.names = ["ticker", "year"]
    meta = pd.read_parquet(DATA / "meta.parquet", columns=["mktcap", "market"]).reset_index()
    ye = meta[meta.date.dt.month == 12].assign(year=lambda d: d.date.dt.year).set_index(["ticker", "year"])[["mktcap", "market"]]
    b = pd.concat([amt, ye], axis=1).reset_index()
    b["eligible"] = (b.amt_avg >= AMT_MIN) & (b.mktcap >= CAP_MIN)
    b["year"] = b.year + 1                                   # 전년도 기준 → 다음 해에 적용
    b["cap_bucket"] = pd.cut(b.mktcap, [0, 3e11, 1e12, 5e12, np.inf], labels=["1천억~3천억", "3천억~1조", "1조~5조", "5조+"])
    return b.rename(columns={"mktcap": "mcap_prev_ye", "market": "market_prev"})[["ticker", "year", "eligible", "mcap_prev_ye", "market_prev", "cap_bucket", "amt_avg"]]

def load_index() -> dict[str, pd.Series]:
    out = {}
    for name, f in (("KOSPI", "KOSPI.parquet"), ("KOSDAQ", "KOSDAQ150.parquet")):
        d = pd.read_parquet(LLV_IDX / f); d["Date"] = pd.to_datetime(d["Date"])
        out[name] = d.set_index("Date")["Close"].astype(float).sort_index()
    return out

def build_episodes() -> pd.DataFrame:
    px = load_prices(); el = eligibility(px); idx = load_index()
    g = px.groupby("ticker", sort=False, observed=True)
    px["r1"] = g.close.pct_change().astype("float32"); px["r3"] = g.close.pct_change(3).astype("float32"); px["p0"] = g.close.shift(3)
    down = (px.r1 < 0).astype("int8")
    # 창 [s-2, s+7] = 10행: s+7 행에서의 rolling(10) 을 -7 시프트
    px["down_days_10"] = down.groupby(px.ticker, observed=True).transform(lambda s: s.rolling(10).sum().shift(-HORIZON)).astype("float32")
    px["worst_day"] = g.r1.transform(lambda s: s.rolling(10).min().shift(-HORIZON)).astype("float32")
    px["min_fwd7"] = g.close.transform(lambda s: s[::-1].rolling(HORIZON).min()[::-1].shift(-1)).astype("float32")   # s+1..s+7 최저 종가
    px["c_fwd7"] = g.close.shift(-HORIZON)
    px["last_rank"] = g.cumcount(ascending=False).astype("int32"); px["last_date"] = g.date.transform("max")
    del down
    key = pd.MultiIndex.from_arrays([px.ticker.astype(str), px.year.astype(int)])
    eli = el.set_index(["ticker", "year"])
    px["eligible"] = eli["eligible"].reindex(key).fillna(False).to_numpy()
    cand = px[(px.r3 <= DROP3) & px.eligible & px.c_fwd7.notna()].copy(); del px
    cand = cand[~((cand.last_date < END) & (cand.last_rank < 10))]            # 정리매매 제외
    cand["ticker"] = cand.ticker.astype(str)
    cand = cand.merge(el.drop(columns="eligible"), on=["ticker", "year"], how="left")
    cand = cand.sort_values(["ticker", "date"]); keep = []; last = {}
    for t, d in zip(cand.ticker, cand.date):                                   # 쿨다운 (순차)
        if t not in last or (d - last[t]).days > 30: keep.append(True); last[t] = d
        else: keep.append(False)
    ep = cand[keep].copy(); del cand
    ep["drop_s"] = ep.close / ep.p0 - 1; ep["min_fwd"] = ep.min_fwd7 / ep.p0 - 1; ep["ret_fwd7"] = ep.c_fwd7 / ep.close - 1
    ep["label"] = np.select([ep.min_fwd <= CRASH, ep.ret_fwd7 >= 0.05, ep.ret_fwd7 <= -0.05], ["crash", "되돌림", "완만추가하락"], "횡보")
    # 지수 태그 (벡터): 거래일 축에 지수를 맞춘 뒤 위치 오프셋으로 조회
    dates = pd.DatetimeIndex(sorted(ep.date.unique().tolist()))                 # 에피소드 날짜만으로는 오프셋이 안 맞으므로 전체 거래일 축 사용
    dates = pd.read_parquet(DATA / "kospi200_daily.parquet").index.sort_values()
    pos = pd.Series(np.arange(len(dates)), index=dates)
    ep["pos"] = pos.reindex(ep.date).to_numpy()
    for name, ser in idx.items():
        arr = ser.reindex(dates).ffill().to_numpy()
        m = (ep.market_prev == ("KOSDAQ" if name == "KOSDAQ" else "KOSPI")) | ((name == "KOSPI") & ep.market_prev.isna())
        p = ep.loc[m, "pos"].to_numpy().astype(int)
        ok = (p - 3 >= 0) & (p + HORIZON < len(arr))
        win = np.full(len(p), np.nan); s3 = np.full(len(p), np.nan)
        win[ok] = arr[p[ok] + HORIZON] / arr[p[ok] - 3] - 1; s3[ok] = arr[p[ok]] / arr[p[ok] - 3] - 1
        ep.loc[m, "idx_ret_win"] = win; ep.loc[m, "idx_ret3_s"] = s3
    ep["mkt_co"] = ep.idx_ret_win <= -0.10; ep["mkt_co_s"] = ep.idx_ret3_s <= -0.04
    cols = ["ticker", "date", "year", "market_prev", "cap_bucket", "mcap_prev_ye", "amt_avg", "p0", "close", "drop_s", "min_fwd", "ret_fwd7", "label",
            "idx_ret_win", "idx_ret3_s", "mkt_co", "mkt_co_s", "down_days_10", "worst_day"]
    ep = ep[cols].reset_index(drop=True)
    ep.to_parquet(OUT / "episodes.parquet"); ep.to_csv(OUT / "episodes.csv", index=False)
    return ep

def summarize(ep: pd.DataFrame):
    pd.set_option("display.width", 200)
    log(f"에피소드 {len(ep):,}건 · 종목 {ep.ticker.nunique():,} · {ep.date.min().date()}~{ep.date.max().date()}")
    print("\n[라벨 분포 전체]"); print(ep.label.value_counts().to_frame("건수").assign(비율=lambda d: (d.건수/len(ep)*100).round(1)).to_string())
    print("\n[연도별 건수 · crash 비율 %]"); y = ep.groupby("year").agg(건수=("label", "size"), crash=("label", lambda s: round((s=="crash").mean()*100, 1)), 시장동반=("mkt_co", "sum")); print(y.T.to_string())
    for col, name in (("mkt_co", "시장동반(사후, 지수 -10%)"), ("mkt_co_s", "시장동반_s(판단시점, 지수3일 -4%)"), ("market_prev", "소속시장"), ("cap_bucket", "시총구간(전년말)")):
        t = pd.crosstab(ep[col], ep.label); t["건수"] = t.sum(axis=1)
        for l in ("crash", "되돌림", "완만추가하락", "횡보"):
            if l in t: t[l] = (t[l] / t.건수 * 100).round(1)
        print(f"\n[{name}] (행 % · 마지막 열 건수)"); print(t.to_string())
    print("\n[s 시점 낙폭 drop_s 분위]"); print(ep.groupby("label").drop_s.describe()[["25%", "50%", "75%"]].round(3).to_string())
    print("\n[crash 의 10일 하락일 수 분포]"); print(ep[ep.label=="crash"].down_days_10.value_counts().sort_index().to_dict())

if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "episodes"
    if step == "episodes":
        ep = build_episodes(); summarize(ep)
    elif step == "summary":
        summarize(pd.read_parquet(OUT / "episodes.parquet"))

# ═══════════════════════════ 3단계: 판단 시점 s 의 거래 특성 ═══════════════════════════
# 전부 s 당일 종가까지의 정보만 쓴다 (s+1 이후 금지). 창 표기: 3d = s-2..s, 20d = s-22..s-3 (급락 직전 평상시 기준).
import os, warnings as _w; _w.filterwarnings("ignore")
HILLSTORM = HERE.parents[1] / "HillStorm"
FEAT_DIR = OUT / "feat_cache"; FEAT_DIR.mkdir(exist_ok=True)

def market_breadth(px: pd.DataFrame) -> pd.DataFrame:
    """전종목 기준 일별 시장 폭: 하락 종목 비율 · 3일 -10% 종목 수 · 거래 종목 수"""
    g = px.groupby("ticker", sort=False, observed=True)
    r1 = g.close.pct_change(); r3 = g.close.pct_change(3)
    b = pd.DataFrame({"date": px.date, "dn": (r1 < 0), "crash3": (r3 <= DROP3), "n": r1.notna()})
    b = b.groupby("date").agg(dn=("dn", "sum"), crash3=("crash3", "sum"), n=("n", "sum"))
    b["breadth_dn"] = b.dn / b.n; b["n_crash3"] = b.crash3
    b["breadth_dn_3d"] = b.breadth_dn.rolling(3).mean()
    return b[["breadth_dn", "breadth_dn_3d", "n_crash3", "n"]]

def ticker_features(d: pd.DataFrame, flow: pd.DataFrame | None, weis) -> pd.DataFrame:
    """한 종목 전체 시계열에 대해 모든 날의 피처를 계산 (에피소드 날만 나중에 고른다)."""
    d = d.sort_values("date").reset_index(drop=True)
    c, o, h, l, v = d.close.astype(float), d.open.astype(float), d.high.astype(float), d.low.astype(float), d.volume.astype(float)
    amt = c * v
    f = pd.DataFrame({"date": d.date})
    r1 = c.pct_change()
    f["drop_s"] = c / c.shift(3) - 1
    # 경로·갭·종가 위치 (3d)
    gap = (o / c.shift(1) - 1)
    f["gap_share_3d"] = gap.rolling(3).sum() / f.drop_s.replace(0, np.nan)          # 시가 갭이 낙폭에서 차지하는 비중
    f["worst_day_3d"] = r1.rolling(3).min()
    rng = (h - l).replace(0, np.nan)
    f["clv_s"] = ((c - l) / rng)                                                       # 0 = 저가 마감, 1 = 고가 마감
    f["clv_3d"] = f.clv_s.rolling(3).mean()
    # 거래량·거래대금
    v20 = v.shift(3).rolling(20).mean(); a20 = amt.shift(3).rolling(20).mean()
    f["vol_mult_s"] = v / v20; f["vol_mult_3d"] = v.rolling(3).mean() / v20
    f["amt_mult_3d"] = amt.rolling(3).mean() / a20
    dnv = (v * (r1 < 0)).rolling(3).sum(); f["dn_vol_share_3d"] = dnv / v.rolling(3).sum().replace(0, np.nan)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    f["range_mult_3d"] = tr.rolling(3).mean() / tr.shift(3).rolling(20).mean()
    # 변동성 정규화·추세 위치 (s-3 기준 = 하락 시작 전 상태)
    sig60 = r1.shift(3).rolling(60).std()
    f["drop_sigma"] = f.drop_s / (sig60 * np.sqrt(3))
    f["sig60"] = sig60
    p0 = c.shift(3)
    f["ret20_pre"] = p0 / c.shift(23) - 1; f["ret60_pre"] = p0 / c.shift(63) - 1; f["ret120_pre"] = p0 / c.shift(123) - 1
    f["dd60_pre"] = p0 / c.shift(3).rolling(60).max() - 1; f["dd120_pre"] = p0 / c.shift(3).rolling(120).max() - 1   # 하락 시작 전 고점 대비 위치
    f["ma20_gap_s"] = c / c.rolling(20).mean() - 1; f["ma60_gap_s"] = c / c.rolling(60).mean() - 1
    f["ma20_slope_pre"] = c.shift(3).rolling(20).mean() / c.shift(13).rolling(20).mean() - 1
    f["high60_break_s"] = (c < c.shift(1).rolling(60).min()).astype(float)             # s 가 60일 신저가인가
    # 수급 (KRX 정규장, 원) — 3d 순매수 / 3d 거래대금, 20d 사전
    if flow is not None and len(flow):
        fl = flow.set_index("date").reindex(d.date).fillna(0.0).reset_index(drop=True)
        a3 = amt.rolling(3).sum().replace(0, np.nan); a20s = amt.shift(3).rolling(20).sum().replace(0, np.nan)
        for k, col in (("frgn", "외국인"), ("inst", "기관"), ("indiv", "개인")):
            f[f"{k}_net3_r"] = fl[col].rolling(3).sum() / a3
            f[f"{k}_net20_pre_r"] = fl[col].shift(3).rolling(20).sum() / a20s
        sell = (fl["외국인"] < 0).astype(int); f["frgn_sell_streak"] = sell.groupby((sell != sell.shift()).cumsum()).cumsum() * sell
        sell = (fl["기관"] < 0).astype(int); f["inst_sell_streak"] = sell.groupby((sell != sell.shift()).cumsum()).cumsum() * sell
    # Weis 파동 (HillStorm 정본 — ATR14×1.5 반전, 인과)
    if weis is not None and len(d) > 30:
        w = weis(d.rename(columns={"date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}))
        wd, wdays, wchg, wvol = w["Weis_Dir"].to_numpy(), w["Weis_Days"].to_numpy(), w["Weis_Chg"].to_numpy(), w["Weis_Vol"].to_numpy()
        f["weis_dir_s"] = wd; f["weis_days_s"] = wdays; f["weis_chg_s"] = wchg
        # 직전 완결 파동(반대 방향) 의 chg·vol — 파동 시작(Weis_Days==1) 지점에서 직전 바의 값을 넘겨받아 ffill
        start = (wdays == 1)
        prev_chg = pd.Series(np.where(start, np.roll(wchg, 1), np.nan)).ffill().to_numpy()
        prev_vol = pd.Series(np.where(start, np.roll(wvol, 1), np.nan)).ffill().to_numpy()
        prev_days = pd.Series(np.where(start, np.roll(wdays, 1), np.nan)).ffill().to_numpy()
        f["weis_prev_chg"] = prev_chg; f["weis_prev_days"] = prev_days
        f["weis_vol_ratio"] = wvol / np.where(prev_vol > 0, prev_vol, np.nan)          # 현재 파동 거래량 / 직전 파동 거래량
        # 상승 파동 되돌림 위치: 현재가 하락 파동이면 직전 상승 파동의 (시작가, 꼭대기) 사이 어디인가 (1 = 꼭대기, 0 = 상승 시작가, 음수 = 그 아래)
        up_chg = np.where(wd == -1, prev_chg, np.where(wd == 1, wchg, np.nan))       # 참조할 상승 파동 상승률
        f["weis_up_chg"] = up_chg
        peak = c.to_numpy() / np.where(wd == -1, 1 + wchg, 1.0)                        # 하락 파동이면 파동 시작가(=꼭대기)
        base = peak / (1 + up_chg)
        f["weis_retrace_pos"] = np.where(peak > base, (c.to_numpy() - base) / (peak - base), np.nan)
    return f

def build_features(chunk_sec: float = 160):
    import sys as _s; _s.path.insert(0, str(HILLSTORM))
    try:
        from weis_wave import compute_weis_wave as weis
    except Exception as e:
        log(f"HillStorm weis 불가: {e}"); weis = None
    ep = pd.read_parquet(OUT / "episodes.parquet"); tickers = sorted(ep.ticker.unique())
    done = {p.stem for p in FEAT_DIR.glob("*.parquet")}
    todo = [t for t in tickers if t not in done]
    log(f"피처: 종목 {len(tickers)} / 완료 {len(done)} / 남은 {len(todo)}")
    if todo:
        px = pd.read_parquet(DATA / "prices.parquet").reset_index()
        px = px[px.ticker.isin(todo)]
        fl = pd.read_parquet(DATA / "investor_flow.parquet").reset_index(); fl = fl[fl.ticker.isin(todo)]
        t0 = time.time(); n = 0
        for t, d in px.groupby("ticker", sort=False):
            if time.time() - t0 > chunk_sec: break
            f = ticker_features(d, fl[fl.ticker == t], weis); f["ticker"] = t
            ev = ep.loc[ep.ticker == t, ["date"]]
            f = f.merge(ev, on="date", how="inner")                    # 에피소드 날만 저장
            f.to_parquet(FEAT_DIR / f"{t}.parquet"); n += 1
        log(f"  이번 호출 {n}종목 ({int(time.time()-t0)}s) — 남은 {len(todo)-n}")
        if n < len(todo): return None
    feats = pd.concat([pd.read_parquet(p) for p in FEAT_DIR.glob("*.parquet")], ignore_index=True)
    full = ep.merge(feats.drop(columns=["drop_s"]), on=["ticker", "date"], how="left")
    # 시장 폭·지수 10일 (전종목·지수)
    px_all = pd.read_parquet(DATA / "prices.parquet", columns=["close"]).reset_index(); px_all["ticker"] = px_all.ticker.astype("category")
    px_all = px_all.sort_values(["ticker", "date"]).reset_index(drop=True)
    br = market_breadth(px_all); del px_all
    full = full.merge(br.reset_index(), on="date", how="left")
    idx = load_index(); dates = pd.read_parquet(DATA / "kospi200_daily.parquet").index.sort_values()
    for name, ser in idx.items():
        s = ser.reindex(dates).ffill(); r10 = (s / s.shift(10) - 1); r3 = (s / s.shift(3) - 1)
        m = (full.market_prev == ("KOSDAQ" if name == "KOSDAQ" else "KOSPI")) | ((name == "KOSPI") & full.market_prev.isna())
        full.loc[m, "idx_ret10_s"] = r10.reindex(full.loc[m, "date"]).to_numpy(); full.loc[m, "idx_ret3_s"] = r3.reindex(full.loc[m, "date"]).to_numpy()
    full["rel_drop_s"] = full.drop_s - full.idx_ret3_s
    full.to_parquet(OUT / "features.parquet")
    log(f"features.parquet: {full.shape}")
    return full

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "features":
    build_features()

# ═══════════════════════════ 3단계 분석: 변수별 판별력 ═══════════════════════════
FEATURES = ["drop_s", "rel_drop_s", "drop_sigma", "worst_day_3d", "gap_share_3d", "clv_s", "clv_3d",
            "vol_mult_s", "vol_mult_3d", "amt_mult_3d", "dn_vol_share_3d", "range_mult_3d", "sig60",
            "ret20_pre", "ret60_pre", "ret120_pre", "dd60_pre", "dd120_pre", "ma20_gap_s", "ma60_gap_s", "ma20_slope_pre", "high60_break_s",
            "frgn_net3_r", "inst_net3_r", "indiv_net3_r", "frgn_net20_pre_r", "inst_net20_pre_r", "indiv_net20_pre_r", "frgn_sell_streak", "inst_sell_streak",
            "weis_dir_s", "weis_days_s", "weis_chg_s", "weis_prev_chg", "weis_prev_days", "weis_vol_ratio", "weis_up_chg", "weis_retrace_pos",
            "breadth_dn", "breadth_dn_3d", "n_crash3", "idx_ret3_s", "idx_ret10_s", "mcap_prev_ye", "amt_avg"]

def auc(x: pd.Series, y: pd.Series) -> float:
    """rank 기반 AUC (crash=1). 0.5 = 무정보. >0.5 = 값이 클수록 crash."""
    m = x.notna() & y.notna(); x, y = x[m], y[m].astype(bool)
    if y.sum() < 20 or (~y).sum() < 20: return np.nan
    r = x.rank(); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def analyze():
    f = pd.read_parquet(OUT / "features.parquet"); y = (f.label == "crash")
    f["stratum"] = pd.cut(f.drop_s, [-1, -0.15, -0.12, -0.10001], labels=["≤-15%", "-15~-12%", "-12~-10%"])
    rows = []
    for c in FEATURES:
        if c not in f: continue
        a_all = auc(f[c], y)
        a_str = {s: auc(f.loc[f.stratum == s, c], y[f.stratum == s]) for s in ["-12~-10%", "-15~-12%", "≤-15%"]}
        yrs = [auc(f.loc[f.year == yy, c], y[f.year == yy]) for yy in range(2016, 2027)]
        yrs = [v for v in yrs if not np.isnan(v)]
        sign = np.sign(a_all - 0.5); stab = np.mean([np.sign(v - 0.5) == sign for v in yrs]) if yrs else np.nan
        q = pd.qcut(f[c].rank(method="first"), 5, labels=False, duplicates="drop") if f[c].notna().sum() > 100 else None
        top = y[q == q.max()].mean() * 100 if q is not None else np.nan; bot = y[q == 0].mean() * 100 if q is not None else np.nan
        rows.append((c, a_all, a_str["-12~-10%"], a_str["-15~-12%"], a_str["≤-15%"], stab, bot, top, f[c].notna().mean() * 100))
    r = pd.DataFrame(rows, columns=["변수", "AUC", "AUC(-12~-10)", "AUC(-15~-12)", "AUC(≤-15)", "연도부호일치", "하위20% crash%", "상위20% crash%", "유효%"])
    r["|AUC-.5|"] = (r.AUC - 0.5).abs()
    r["층별최소|AUC-.5|"] = (r[["AUC(-12~-10)", "AUC(-15~-12)", "AUC(≤-15)"]] - 0.5).abs().min(axis=1)
    r = r.sort_values("|AUC-.5|", ascending=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 100)
    print(r.round(3).to_string(index=False)); r.to_csv(OUT / "feature_ranking.csv", index=False)
    print("\n기준 crash 비율:", round(y.mean() * 100, 2), "% | 층별:", f.groupby("stratum", observed=True).apply(lambda g: round((g.label == "crash").mean() * 100, 1)).to_dict(), "| 층별 건수:", f.stratum.value_counts().to_dict())

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "analyze":
    analyze()

# ═══════════════════════════ 4단계: s 이후 판단 시점 찾기 (데드캣) ═══════════════════════════
# 판단일 k ∈ {1,2,3} = s+k 종가. 그날까지의 정보만으로 상태를 만들고, 그날 종가 기준 이후 7일의 수익·최대낙폭을 붙인다.
BOUNCE = 0.03; FWD = 7
def build_stage4() -> pd.DataFrame:
    ep = pd.read_parquet(OUT / "episodes.parquet")
    f3 = pd.read_parquet(OUT / "features.parquet", columns=["ticker", "date", "sig60", "vol_mult_3d", "clv_3d", "gap_share_3d"])
    ep = ep.merge(f3, on=["ticker", "date"], how="left")
    px = pd.read_parquet(DATA / "prices.parquet").reset_index(); px = px[px.ticker.isin(ep.ticker.unique())]
    fl = pd.read_parquet(DATA / "investor_flow.parquet").reset_index(); fl = fl[fl.ticker.isin(ep.ticker.unique())]
    rows = []
    fl_by = {t: g.set_index("date")[["외국인", "기관"]] for t, g in fl.groupby("ticker", sort=False)}      # ⚠ 종목마다 560만 행을 거르면 O(n²) — 미리 묶는다
    ep_by = {t: g for t, g in ep.groupby("ticker", sort=False)}
    t0 = time.time()
    for t, d in px.groupby("ticker", sort=False):
        d = d.sort_values("date").reset_index(drop=True)
        dates = d.date.to_numpy(); c = d.close.to_numpy(float); o = d.open.to_numpy(float); h = d.high.to_numpy(float); l = d.low.to_numpy(float); v = d.volume.to_numpy(float)
        flt = fl_by.get(t); flt = flt.reindex(d.date).fillna(0.0) if flt is not None else pd.DataFrame({"외국인": 0.0, "기관": 0.0}, index=d.date)
        fr, ins = flt["외국인"].to_numpy(float), flt["기관"].to_numpy(float)
        pos = pd.Series(np.arange(len(d)), index=d.date)
        r1 = np.concatenate([[np.nan], c[1:] / c[:-1] - 1])
        for e in ep_by.get(t, pd.DataFrame()).itertuples(index=False):
            p = int(pos.get(e.date, -1))
            if p < 3 or p + 3 + 20 >= len(d): continue
            vol_drop = v[p-2:p+1].mean()                                   # 하락 3일 평균 거래량
            low_s = l[p-2:p+1].min()
            for k in (1, 2, 3):
                q = p + k
                win = np.arange(p + 1, q + 1)                              # s+1..s+k
                rr = r1[win]
                up = rr >= BOUNCE
                bi = win[np.argmax(rr)] if up.any() else -1               # 가장 큰 반등일
                st = {"ticker": t, "date": e.date, "k": k, "label": e.label, "year": e.year, "drop_s": e.drop_s, "sig60": e.sig60,
                      "idx_ret3_s": e.idx_ret3_s, "mkt_co": e.mkt_co,
                      "ret_s_k": c[q] / c[p] - 1, "cum_p0_k": c[q] / e.p0 - 1,
                      "n_up": int((rr > 0).sum()), "max_bounce": float(np.nanmax(rr)), "bounce": bool(up.any()),
                      "consec_down": int(len(rr) - (np.where(rr >= 0)[0].max() + 1) if (rr >= 0).any() else len(rr)),
                      "newlow_k": bool(c[q] < c[p-2:q].min()), "below_s_low": bool(c[q] < low_s),
                      "vol_k_ratio": v[q] / vol_drop if vol_drop > 0 else np.nan,
                      "clv_k": (c[q] - l[q]) / (h[q] - l[q]) if h[q] > l[q] else np.nan,
                      "frgn_cum_r": fr[p+1:q+1].sum() / max((c[p+1:q+1] * v[p+1:q+1]).sum(), 1),
                      "inst_cum_r": ins[p+1:q+1].sum() / max((c[p+1:q+1] * v[p+1:q+1]).sum(), 1),
                      "fwd_ret7": c[q + FWD] / c[q] - 1, "fwd_min7": c[q+1:q+FWD+1].min() / c[q] - 1, "fwd_max7": c[q+1:q+FWD+1].max() / c[q] - 1,
                      "fwd_ret20": c[q + 20] / c[q] - 1, "fwd_min20": c[q+1:q+21].min() / c[q] - 1}
                if bi >= 0:
                    st.update({"b_r1": r1[bi], "b_vol_ratio": v[bi] / vol_drop if vol_drop > 0 else np.nan,
                               "b_clv": (c[bi] - l[bi]) / (h[bi] - l[bi]) if h[bi] > l[bi] else np.nan,
                               "b_gap": o[bi] / c[bi-1] - 1, "b_intraday": c[bi] / o[bi] - 1,
                               "b_age": q - bi, "giveback": (c[q] / c[bi] - 1) if q > bi else np.nan,
                               "b_frgn_r": fr[bi] / max(c[bi] * v[bi], 1), "b_inst_r": ins[bi] / max(c[bi] * v[bi], 1)})
                rows.append(st)
    s4 = pd.DataFrame(rows); s4.to_parquet(OUT / "stage4.parquet"); log(f"stage4: {len(s4):,} 행 (k별 {s4.groupby('k').size().to_dict()}) {int(time.time()-t0)}s")
    return s4

def analyze_stage4():
    s4 = pd.read_parquet(OUT / "stage4.parquet"); pd.set_option("display.width", 230)
    s4["crash"] = s4.label == "crash"; s4["fwd_bad"] = s4.fwd_min7 <= -0.15      # 판단일 이후 7일 내 추가 -15%
    def tab(df, by, name):
        g = df.groupby(by, observed=True)
        t = pd.DataFrame({"건수": g.size(), "crash%": g.crash.mean()*100, "추가-15%": g.fwd_bad.mean()*100,
                          "이후7일 평균수익%": g.fwd_ret7.mean()*100, "이후7일 중앙%": g.fwd_ret7.median()*100, "이후7일 최대낙폭 평균%": g.fwd_min7.mean()*100})
        print(f"\n[{name}]"); print(t.round(1).to_string())
    print("=== (가) 판단일별 기준선 ==="); tab(s4, "k", "k별 전체")
    print("\n=== (나) 상태별 — 반등이 나왔나 / 연속 하락인가 / 신저가인가 ===")
    for k in (1, 2, 3):
        d = s4[s4.k == k]
        tab(d, "bounce", f"k={k} · 반등(+3%↑) 유무")
        if k >= 2: tab(d, "consec_down", f"k={k} · s 이후 연속 하락일 수(끝에서)")
        tab(d, "newlow_k", f"k={k} · 판단일 종가가 s-2 이후 신저가")
    print("\n=== (다) 반등이 나온 경우 — 반등의 모양 (k=2,3 합산, 반등 다음날 이후 되돌림 포함) ===")
    b = s4[s4.bounce & (s4.k >= 2)].copy()
    b["b_vol_t"] = pd.qcut(b.b_vol_ratio, 3, labels=["반등 거래량 적음", "중", "반등 거래량 많음"])
    b["b_clv_t"] = pd.qcut(b.b_clv, 3, labels=["반등일 저가권 마감", "중", "고가권 마감"])
    b["gb_t"] = pd.cut(b.giveback, [-1, -0.05, -0.0001, 0.0001, 1], labels=["되돌림 ≤-5%", "되돌림 -5~0", "되돌림 없음(=반등일)", "추가 상승"])
    b["b_gap_t"] = pd.cut(b.b_gap, [-1, -0.0001, 0.02, 1], labels=["갭다운 출발", "갭 0~2%", "갭업 >2%"])
    b["b_shape"] = np.where(b.b_intraday > 0.02, "장중 상승형", np.where(b.b_intraday < -0.02, "갭업 후 장중 밀림", "혼합"))
    for col, nm in (("b_vol_t", "반등일 거래량 / 하락3일 평균"), ("b_clv_t", "반등일 종가 위치"), ("gb_t", "반등 후 되돌림(판단일 종가/반등일 종가)"), ("b_gap_t", "반등일 시가 갭"), ("b_shape", "반등일 모양")):
        tab(b, col, nm)
    tab(b, ["b_vol_t", "gb_t"], "거래량 × 되돌림")
    print("\n=== (라) 반등 없이 연속 하락 — 깊이별 ===")
    nb = s4[~s4.bounce & (s4.k >= 2)].copy(); nb["ret_t"] = pd.cut(nb.ret_s_k, [-1, -0.10, -0.05, 0, 1], labels=["s 이후 ≤-10%", "-10~-5%", "-5~0%", "s 이후 상승(반등<3%)"])
    tab(nb, ["k", "ret_t"], "k × s 이후 누적")
    print("\n=== (마) AUC (crash 기준) — 판단일별 상태 변수 ===")
    rows = []
    for k in (1, 2, 3):
        d = s4[s4.k == k]
        for col in ("ret_s_k", "cum_p0_k", "max_bounce", "n_up", "consec_down", "vol_k_ratio", "clv_k", "frgn_cum_r", "inst_cum_r", "b_vol_ratio", "b_clv", "giveback", "b_gap", "b_intraday"):
            if col in d: rows.append((k, col, auc(d[col], d.crash), auc(d[col], d.fwd_bad), d[col].notna().mean()*100))
    r = pd.DataFrame(rows, columns=["k", "변수", "AUC(crash)", "AUC(추가-15%)", "유효%"]).round(3); print(r.pivot(index="변수", columns="k", values=["AUC(crash)", "AUC(추가-15%)"]).round(3).to_string())

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("stage4", "analyze4"):
    if sys.argv[1] == "stage4": build_stage4()
    analyze_stage4()
