# -*- coding: utf-8 -*-
"""
gap_tail_market.py — 큰 갭업/갭다운 비율: 우리 유니버스 vs KOSPI 전체 vs KOSDAQ 전체 (연구 전용)

케인 질문 (2026-09-13): 유니버스(203) 의 ">+2% 갭업 비율" 이 9월 8.2% 로 역사 최저인데,
시장 전체(KOSPI 상장 전 종목)로 보면 어떤가 — 유니버스가 시장을 대표하는가, 아니면 유니버스만 마른 건가.

자료: pykrx `get_market_ohlcv_by_ticker(date, market)` (KRX 로그인 = LLV .env 의 KRX_ID/KRX_PW).
      거래일마다 KOSPI·KOSDAQ 전 종목 OHLCV 를 받아 build/krx_all_ohlcv.parquet 에 캐시(멱등 — 있는 날짜는 건너뜀).
      갭 = 다음 거래일 시가 / 당일 종가 − 1. 시가 0(거래정지)·종가 0 은 제외.
그룹: KOSPI 전체 / KOSDAQ 전체 / KOSPI 시총 상위 200 (그날 기준, KOSPI200 근사) / 우리 유니버스 (LLV core+extend, ETF·우선주 제외)
출력: build/gap_tail_market.csv (일별) + 콘솔 요약 (18일 롤링 · 월별 · 8/18 전후)

사용 (에어, 네트워크 필요):
  python3 gap_tail_market.py --start 2025-08-22 --end 2026-09-11      # 최초 약 8~10분 (256일 × 2시장)
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve(); MOP = HERE.parents[1]; STOLAB = HERE.parents[3]
LLV = STOLAB / "LongLiveVault"; OUT = MOP / "build"; CACHE = OUT / "krx_all_ohlcv.parquet"
THR = 2.0


def fetch(start, end):
    from dotenv import load_dotenv; load_dotenv(LLV / ".env")
    from pykrx import stock
    days = pd.bdate_range(start, end)
    have = set()
    if CACHE.exists():
        old = pd.read_parquet(CACHE); have = set(pd.to_datetime(old.Date).dt.strftime("%Y%m%d")); frames = [old]
    else:
        frames = []
    t0 = time.time(); n = 0
    for d in days:
        ds = d.strftime("%Y%m%d")
        if ds in have: continue
        for mkt in ("KOSPI", "KOSDAQ"):
            df = None
            for attempt in range(3):                       # KRX 가 연속 호출 ~120일째부터 오류 페이지를 돌려준다 → 간격 + 재시도
                try:
                    df = stock.get_market_ohlcv_by_ticker(ds, market=mkt); break
                except Exception as e:
                    print(f"  {ds} {mkt} 실패({attempt+1}/3): {str(e)[:60]}", flush=True); time.sleep(20 * (attempt + 1))
            time.sleep(0.8)
            if df is None or df.empty: continue
            df = df.rename(columns={"시가": "Open", "고가": "High", "저가": "Low", "종가": "Close", "거래량": "Volume", "거래대금": "Amount", "시가총액": "MarketCap"})
            df = df[["Open", "High", "Low", "Close", "Volume", "Amount", "MarketCap"]].reset_index().rename(columns={"티커": "Ticker"})
            df["Date"] = d; df["Market"] = mkt; frames.append(df)
        n += 1
        if n % 20 == 0:
            pd.concat(frames, ignore_index=True).to_parquet(CACHE, index=False)
            print(f"  [{n}] {ds} 경과 {(time.time()-t0)/60:.1f}분", flush=True)
    allf = pd.concat(frames, ignore_index=True).drop_duplicates(["Date", "Ticker"]) if frames else pd.DataFrame()
    OUT.mkdir(exist_ok=True); allf.to_parquet(CACHE, index=False)
    print(f"캐시 {len(allf):,}행 · {allf.Date.nunique()}일 → {CACHE}", flush=True)
    return allf


def load_raw():
    """LLV data/raw/krx_{kospi,kosdaq}_YYYYMMDD.parquet (daily_update 08:01 산출) — 미니엔 매일, 에어엔 드문드문.
    pykrx 캐시와 같은 스키마로 맞춰 합친다 (KRX 호출 한도에 걸릴 때의 대안 소스)."""
    import glob
    fs = sorted(glob.glob(str(LLV / "data/raw/krx_kospi_*.parquet")) + glob.glob(str(LLV / "data/raw/krx_kosdaq_*.parquet")))
    if not fs: return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f, columns=["Date", "Ticker", "Market", "Open", "High", "Low", "Close", "Volume", "Amount", "MarketCap"]) for f in fs], ignore_index=True)
    df["Date"] = pd.to_datetime(df.Date); df["Market"] = df.Market.str.upper()
    return df[df.Market.isin(["KOSPI", "KOSDAQ"])]


def merged_source():
    frames = [f for f in [pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame(), load_raw()] if len(f)]
    px = pd.concat(frames, ignore_index=True); px["Date"] = pd.to_datetime(px.Date)
    px = px.drop_duplicates(["Date", "Ticker"])
    print(f"소스 합산 {len(px):,}행 · {px.Date.nunique()}일 (pykrx 캐시 + LLV raw)", flush=True)
    return px


def universe_tickers():
    cols = ["Ticker"]
    u = set(pd.read_parquet(LLV / "data/ohlcv/core.parquet", columns=cols).Ticker) | set(pd.read_parquet(LLV / "data/ohlcv/extend.parquet", columns=cols).Ticker)
    import json
    cls = json.load(open(LLV / "data/ticker_classification.json", encoding="utf-8"))["classifications"]
    drop = {v["ticker"] for v in cls.values() if v["sector"].split(".")[0] in ("ETF", "우선주")}
    return {str(t).zfill(6) for t in u} - drop


def analyze(px, start, end):
    px = px.copy(); px["Date"] = pd.to_datetime(px.Date); px["Ticker"] = px.Ticker.astype(str).str.zfill(6)
    px = px[(px.Open > 0) & (px.Close > 0)].sort_values(["Ticker", "Date"])
    px["Open1"] = px.groupby("Ticker").Open.shift(-1); px["Date1"] = px.groupby("Ticker").Date.shift(-1)
    # 다음 행이 실제 다음 거래일인지 (캐시 결측일 방지): 전체 거래일 목록에서 바로 다음 날이어야 함
    days = np.sort(px.Date.unique()); nxt = dict(zip(days[:-1], days[1:]))
    px = px[px.Date1 == px.Date.map(nxt)]
    px["gap"] = (px.Open1 / px.Close - 1) * 100
    uni = universe_tickers()
    px["rank_mc"] = px.groupby(["Date", "Market"]).MarketCap.rank(ascending=False)
    groups = {"KOSPI 전체": px.Market == "KOSPI", "KOSDAQ 전체": px.Market == "KOSDAQ",
              "KOSPI 시총200": (px.Market == "KOSPI") & (px.rank_mc <= 200), "우리 유니버스": px.Ticker.isin(uni)}
    rows = {}
    for nm, m in groups.items():
        g = px[m].groupby("Date").gap
        rows[nm] = pd.DataFrame({"n": g.size(), "right": g.apply(lambda s: (s > THR).mean() * 100), "left": g.apply(lambda s: (s < -THR).mean() * 100), "mean": g.mean()})
    D = pd.concat(rows, axis=1)
    D = D[(D.index >= pd.Timestamp(start)) & (D.index <= pd.Timestamp(end))]
    D.to_csv(OUT / "gap_tail_market.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(f"\n일수 {len(D)} · 종목수(중앙): " + " · ".join(f"{nm} {int(D[(nm,'n')].median())}" for nm in groups))
    roll = D.xs("right", axis=1, level=1).rolling(18).mean().dropna(); rollL = D.xs("left", axis=1, level=1).rolling(18).mean().dropna()
    pre = roll.index < "2026-08-18"
    print("\n== >+2% 갭업 비율, 18일 롤링 (%) — 8/18 이전 최저 / 중앙 / 최고 | 최근값")
    for nm in groups:
        r = roll[nm]; print(f"  {nm:10s} {r[pre].min():5.1f} / {r[pre].median():5.1f} / {r[pre].max():5.1f} | {r.iloc[-1]:5.1f} ({r.index[-1].date()})  · 최근값이 이전 최저보다 {'낮음' if r.iloc[-1] < r[pre].min() else '높음'}")
    print("\n== <−2% 갭다운 비율, 18일 롤링 (%) — 8/18 이전 최저 / 중앙 / 최고 | 최근값")
    for nm in groups:
        r = rollL[nm]; print(f"  {nm:10s} {r[pre].min():5.1f} / {r[pre].median():5.1f} / {r[pre].max():5.1f} | {r.iloc[-1]:5.1f}")
    print("\n== 월별 >+2% 비율 (%)"); M = D.xs("right", axis=1, level=1).resample("ME").mean().round(1); M["n일"] = D.resample("ME").size(); print(M.to_string())
    print("\n== 월별 <−2% 비율 (%)"); print(D.xs("left", axis=1, level=1).resample("ME").mean().round(1).to_string())
    post = D.index >= "2026-08-18"
    print("\n== 8/18 전후 평균 (%)")
    for nm in groups:
        print(f"  {nm:10s} 갭업>+2%: 이전 {D.loc[~post,(nm,'right')].mean():5.1f} → 이후 {D.loc[post,(nm,'right')].mean():5.1f} | 갭다운<−2%: {D.loc[~post,(nm,'left')].mean():5.1f} → {D.loc[post,(nm,'left')].mean():5.1f} | 평균갭 {D.loc[~post,(nm,'mean')].mean():+.2f} → {D.loc[post,(nm,'mean')].mean():+.2f}")
    # 유니버스 − 시장 차이의 롤링
    print("\n== 우리 유니버스 − KOSPI 시총200, >+2% 비율 차 18일 롤링: 이전 최저/중앙/최고 | 최근")
    dd = (roll["우리 유니버스"] - roll["KOSPI 시총200"]); print(f"  {dd[pre].min():+5.1f} / {dd[pre].median():+5.1f} / {dd[pre].max():+5.1f} | {dd.iloc[-1]:+5.1f}")
    return D


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--start", default="2025-08-22"); ap.add_argument("--end", default="2026-09-11"); ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args()
    if not a.no_fetch:
        try: fetch(a.start, a.end)
        except Exception as e: print(f"pykrx 수집 실패(캐시+raw 로 계속): {str(e)[:80]}", flush=True)
    analyze(merged_source(), a.start, a.end)
