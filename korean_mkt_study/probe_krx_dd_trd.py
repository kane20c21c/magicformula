# -*- coding: utf-8 -*-
"""probe_krx_dd_trd.py — KRX Open API 'KRX 시리즈 일별시세'(idx/krx_dd_trd) 시험 (케인 2026-11-10 10:09 신청 → 시험 요청).
⑦ 의 C2 재료인 KRX 중형 TMI 를 pykrx 대신 공식 API 로 받을 수 있는지, 값이 pykrx·연구 자료와 같은지 본다.
키는 LongLiveVault/.env 의 KRX_AUTH_KEY (출력하지 않는다). 조회만 한다.
사용  python3 probe_krx_dd_trd.py [YYYYMMDD ...]
"""
import os, sys, time
from pathlib import Path
import requests, pandas as pd
from dotenv import load_dotenv
HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parents[1] / "LongLiveVault" / ".env")
URL = "https://data-dbg.krx.co.kr/svc/apis/idx/krx_dd_trd"
KEY = os.getenv("KRX_AUTH_KEY")
days = sys.argv[1:] or ["20261109", "20261110", "20250102", "20160104", "20140102"]
ref = pd.read_parquet(HERE / "data_idx" / "krx_indices.parquet")        # 연구 때 pykrx 로 받은 자료
ref = ref[ref.name == "KRX 중형 TMI"].assign(date=lambda d: pd.to_datetime(d.date)).set_index("date")
print("키 있음" if KEY else "키 없음", "· URL", URL)
for d in days:
    t0 = time.time()
    try:
        r = requests.post(URL, headers={"AUTH_KEY": KEY, "Content-Type": "application/json", "Accept": "application/json"}, json={"basDd": d}, timeout=30)
    except Exception as e:
        print(f"[{d}] 요청 실패: {type(e).__name__} {e}"); continue
    dt = time.time() - t0
    try: js = r.json()
    except Exception: print(f"[{d}] HTTP {r.status_code} · {dt:.1f}초 · JSON 아님: {r.text[:200]}"); continue
    rows = js.get("OutBlock_1", []) or []
    print(f"[{d}] HTTP {r.status_code} · {dt:.1f}초 · 행 {len(rows)}" + ("" if rows else f" · 응답 {str(js)[:200]}"))
    if not rows: continue
    if d == days[0]: print("   필드:", list(rows[0].keys())); print("   지수 이름:", [x.get("IDX_NM") for x in rows])
    mid = [x for x in rows if (x.get("IDX_NM") or "").strip() == "KRX 중형 TMI"]
    if not mid: print("   KRX 중형 TMI 없음"); continue
    x = mid[0]; c = float(str(x.get("CLSPRC_IDX", "nan")).replace(",", "") or "nan")
    rv = ref["close"].get(pd.Timestamp(d), float("nan"))
    print(f"   KRX 중형 TMI  BAS_DD {x.get('BAS_DD')} · 시가 {x.get('OPNPRC_IDX')} · 종가 {x.get('CLSPRC_IDX')} · 등락률 {x.get('FLUC_RT')}% · 연구 자료(pykrx) 종가 {rv}")
    time.sleep(0.5)
