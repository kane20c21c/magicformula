# -*- coding: utf-8 -*-
"""probe_tmi_sameday.py — KRX 중형 TMI 지수가 그날 저녁 pykrx 로 나오는지 확인 (통일장 그림자 추적 1단계, 2026-11-10).
⑦ 의 C2 는 KRX 중형 TMI 가 필요하다 (코스피 중형주로 바꾸면 학습 수익이 283% → 219% 로 크게 바뀌어 대체 불가).
사용  python3 probe_tmi_sameday.py [YYYYMMDD]   — 인자 없으면 오늘
"""
import sys, time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / "LongLiveVault" / ".env")   # pykrx 는 import 때 자격증명을 읽는다
from pykrx import stock

day = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y%m%d")
print("조회 시각", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "· 대상일", day)
for mkt, nm in (("KRX", "KRX 중형 TMI"), ("KOSPI", "코스피 200")):
    tk = stock.get_index_ticker_list(day, market=mkt)
    m = {stock.get_index_ticker_name(t).strip(): t for t in tk}
    code = m.get(nm)
    if code is None: print(f"  {nm}: 코드 못 찾음 (목록 {len(tk)}개)"); continue
    df = stock.get_index_ohlcv(day, day, code)
    print(f"  {nm} ({code}): {len(df)}행", df.to_dict("records") if len(df) else "— 아직 없음")
    time.sleep(0.5)
