# -*- coding: utf-8 -*-
"""fetch_sector_indices.py — KRX 업종지수 일봉 + 연말 업종 구성종목 수집 (케인 2026-11-07 결정: 섹터 = 안 A, KRX 공식 업종).

용도   stock_gapshock.py 의 '범위' 축 — 섹터 쇼크(업종지수 자체)와 종목 쇼크의 섹터 맥락(같은 업종이 같이 빠졌나).
원천   pykrx (KRX 정보데이터시스템, LLV .env 의 KRX_ID/KRX_PW 로그인) — fetch_indices.py 와 같은 방식.
수집   ① 연말 스냅샷(2014-12-30, 2015~2025 각 해 마지막 거래일, 2026-10-29)마다 KOSPI·KOSDAQ 지수 목록 + 이름
       ② 목록에 한 번이라도 나온 지수 전부의 일봉 2015-01-02~2026-10-29 (지수 하나 = 기간 한 번 호출)
       ③ 합성·규모 지수(코스피/코스닥 본지수, 200·150·100·50 계열, 대·중·소형주)를 뺀 지수의 스냅샷별 구성종목
       → 연도 Y 의 종목→업종은 Y−1 연말 구성으로 매긴다 (유니버스 규칙과 같은 시점). 겹치는 계층(예: 제조업 ⊃ 화학)은
         분석 쪽에서 '구성종목이 가장 적은 업종'(가장 세분)으로 고른다 — 여기서는 원자료를 다 남긴다.
⚠     KRX 업종 분류가 중간에 개편됐으면 이름·코드가 스냅샷마다 달라진다 → krx_sector_list.csv 로 확인 후 분석에서 처리.
저장   data_idx/krx_sector_indices.parquet (long: date, market, code, name, open, high, low, close, volume, value, mktcap)
       data_idx/krx_sector_members.parquet (snap, market, code, name, ticker)
       data_idx/krx_sector_list.csv        (snap, market, code, name, n_members)
       캐시  _cache_sector/ (호출 단위 pickle — 중간에 끊겨도 다시 돌리면 이어서 받는다)
실행   맥(에어)에서:  cd korean_mkt_study && python3 fetch_sector_indices.py      (약 1,000콜 · 콜 간 0.5초 · 15~25분 예상)
"""
import re, sys, time, pickle
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parents[1] / "LongLiveVault" / ".env")
from pykrx import stock  # noqa: E402  (.env 로드 뒤 import — pykrx 가 import 시점에 자격증명을 읽는다)

START, END = "20150102", "20261029"
OUT = HERE / "data_idx"; CACHE = HERE / "_cache_sector"; CACHE.mkdir(exist_ok=True)
COLS = {"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume", "거래대금": "value", "상장시가총액": "mktcap"}
COMPOSITE = re.compile(r"^(코스피|코스닥)$|200|150|100|50|대형주|중형주|소형주")
SLEEP = 0.5

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def cached(key, fn):
    f = CACHE / f"{key}.pkl"
    if f.exists(): return pickle.loads(f.read_bytes())
    for k in range(3):
        try:
            v = fn(); f.write_bytes(pickle.dumps(v)); time.sleep(SLEEP); return v
        except Exception as e:
            log(f"  재시도 {k+1}/3 {key}: {type(e).__name__} {e}"); time.sleep(5 * (k + 1))
    raise SystemExit(f"실패: {key} — 몇 분 뒤 다시 실행하면 캐시부터 이어서 받는다")

def snapshots():
    k = pd.read_parquet(HERE / "data_2015" / "kospi200_daily.parquet"); d = pd.to_datetime(k.index)
    ye = pd.Series(d).groupby(pd.Series(d).dt.year).max()
    snaps = ["20141230"] + [x.strftime("%Y%m%d") for y, x in ye.items() if y <= 2025] + [END]
    return snaps

def main():
    snaps = snapshots(); log(f"스냅샷 {len(snaps)}개: {snaps[0]} … {snaps[-1]}")
    lst = []
    for s in snaps:
        for mkt in ("KOSPI", "KOSDAQ"):
            tk = cached(f"list_{mkt}_{s}", lambda: stock.get_index_ticker_list(s, market=mkt))
            for t in tk:
                nm = cached(f"name_{t}", lambda: stock.get_index_ticker_name(t)).strip()
                lst.append(dict(snap=s, market=mkt, code=t, name=nm))
    lst = pd.DataFrame(lst); log(f"지수 목록: 고유 코드 {lst.code.nunique()} (KOSPI {lst[lst.market=='KOSPI'].code.nunique()} · KOSDAQ {lst[lst.market=='KOSDAQ'].code.nunique()})")
    # ② 일봉
    out = []
    for (mkt, code), g in lst.groupby(["market", "code"]):
        nm = g.name.iloc[-1]
        df = cached(f"ohlcv_{code}", lambda: stock.get_index_ohlcv(START, END, code))
        if df is None or len(df) == 0: log(f"  일봉 없음 {code} {nm}"); continue
        df = df.rename(columns=COLS); df.index.name = "date"; df = df.reset_index()
        df["market"], df["code"], df["name"] = mkt, code, nm; out.append(df)
    ix = pd.concat(out, ignore_index=True); ix.to_parquet(OUT / "krx_sector_indices.parquet")
    log(f"일봉: {ix.code.nunique()}개 지수 · {len(ix):,}행 → {OUT/'krx_sector_indices.parquet'}")
    # ③ 구성종목
    mem = []; tgt = lst[~lst.name.str.contains(COMPOSITE)]
    log(f"구성종목 대상: {tgt.code.nunique()}개 업종 × 스냅샷 = {len(tgt)}콜")
    for i, r in enumerate(tgt.itertuples(), 1):
        tl = cached(f"mem_{r.code}_{r.snap}", lambda: stock.get_index_portfolio_deposit_file(r.code, r.snap))
        for t in (tl or []): mem.append(dict(snap=r.snap, market=r.market, code=r.code, name=r.name, ticker=t))
        if i % 50 == 0: log(f"  구성종목 {i}/{len(tgt)}")
    mem = pd.DataFrame(mem); mem.to_parquet(OUT / "krx_sector_members.parquet")
    n = mem.groupby(["snap", "market", "code"]).size().rename("n_members").reset_index()
    lst.merge(n, on=["snap", "market", "code"], how="left").to_csv(OUT / "krx_sector_list.csv", index=False)
    log(f"구성종목: {len(mem):,}행 · 업종 {mem.code.nunique()} → {OUT/'krx_sector_members.parquet'} · 목록 krx_sector_list.csv")
    # 점검: 스냅샷마다 업종 이름이 바뀐 코드
    ch = lst.groupby("code").name.nunique(); ch = ch[ch > 1]
    if len(ch): log(f"⚠ 이름이 바뀐 코드 {len(ch)}개: {list(ch.index)[:20]}")
    nn = lst.groupby("snap").code.nunique(); log("스냅샷별 지수 수: " + ", ".join(f"{k}:{v}" for k, v in nn.items()))

if __name__ == "__main__":
    main()
