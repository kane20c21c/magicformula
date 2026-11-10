# -*- coding: utf-8 -*-
"""fetch_indices.py — 국면 감지 비교 지수 수집 (케인 2026-11-05 승인 목록).

원천  pykrx (KRX 정보데이터시스템, LLV .env 의 KRX_ID/KRX_PW 로그인) — 지수 하나를 기간 한 번 호출
      KRX Open API(idx/krx_dd_trd)는 수집 당시(11-05) 이 키로 401 → KRX 계열은 pykrx 로 받았다
      ※ 2026-11-10 케인 신청 후 사용 가능 — KRX 중형 TMI 는 LLV tickers/KRX_TMI_MID.parquet 가 운영 정본
        (LLV krx_fetcher.fetch_krx_tmi_mid · probe_krx_dd_trd.py 에서 pykrx 값과 일치 확인)
      코스피200 동일가중 = KODEX 200동일가중 ETF(252650) 가격 그대로 (케인 16:56)
⚠    KRX 300(2018-02 발표)·KRX TMI 규모별(2025-01 발표)은 발표 전 구간이 소급 산출값 — 종가만 있고 시·고·저 0
저장  data_idx/krx_indices.parquet (long: date, name, code, open, high, low, close, volume, value, mktcap)
사용  python3 fetch_indices.py [시작 YYYYMMDD] [끝 YYYYMMDD]
"""
import sys, time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parents[1] / "LongLiveVault" / ".env")
from pykrx import stock  # noqa: E402  (.env 로드 뒤 import — pykrx 가 import 시점에 자격증명을 읽는다)

WANT = {
    "KRX": ["KTOP 30", "KRX 100", "KRX 300", "KRX TMI", "KRX 중대형 TMI", "KRX 중형 TMI", "KRX 소형 TMI", "KRX 초소형 TMI"],
    "KOSPI": ["코스피", "코스피 50", "코스피 100", "코스피 200", "코스피 200 중소형주", "코스피 200 초대형제외 지수",
              "코스피200제외 코스피지수", "코스피 대형주", "코스피 중형주", "코스피 소형주"],
    "KOSDAQ": ["코스닥", "코스닥 150", "코스닥 대형주", "코스닥 중형주", "코스닥 소형주"],
}
COLS = {"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume", "거래대금": "value", "상장시가총액": "mktcap"}

def main(start="20150102", end="20261104"):
    out = []; miss = []
    for mkt, names in WANT.items():
        tk = stock.get_index_ticker_list(end, market=mkt)
        m = {stock.get_index_ticker_name(t).strip(): t for t in tk}
        for nm in names:
            code = m.get(nm)
            if code is None: miss.append(f"{mkt}:{nm}"); continue
            df = stock.get_index_ohlcv(start, end, code).rename(columns=COLS)
            df.index.name = "date"; df = df.reset_index(); df["name"] = nm; df["code"] = code
            out.append(df); print(f"  {nm:<22} {code}  {len(df)}행  {df.date.min().date()}~{df.date.max().date()}", flush=True); time.sleep(0.5)
    e = stock.get_etf_ohlcv_by_date(start, end, "252650").rename(columns=COLS)
    e.index.name = "date"; e = e.reset_index(); e["name"] = "코스피200 동일가중(ETF 252650)"; e["code"] = "252650"
    out.append(e[[c for c in e.columns if c in ["date", "name", "code"] + list(COLS.values())]])
    print(f"  {'코스피200 동일가중(ETF)':<22} 252650  {len(e)}행  {e.date.min().date()}~{e.date.max().date()}")
    d = pd.concat(out, ignore_index=True)
    for c in COLS.values():
        if c in d: d[c] = pd.to_numeric(d[c], errors="coerce")
    p = HERE / "data_idx"; p.mkdir(exist_ok=True); d.to_parquet(p / "krx_indices.parquet", index=False)
    print("저장", p / "krx_indices.parquet", len(d), "행 · 못 찾음:", miss or "없음")

if __name__ == "__main__":
    main(*sys.argv[1:3])
