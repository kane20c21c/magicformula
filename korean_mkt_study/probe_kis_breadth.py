# -*- coding: utf-8 -*-
"""probe_kis_breadth.py — 한국투자 업종 현재지수(FHPUP02100000)의 상승·하락 종목 수 필드 확인 (통일장 그림자 추적, 2026-11-10).
⑦·⑤ 의 시장 브레드스를 19:10 에 그날 값으로 판정하려면 전 종목 등락 수가 필요하다. 조회만 한다 (케인 10:20 허락).
저녁(장 마감 뒤) 값을 다음 날 아침 KRX 원본 집계와 대조하려고 결과를 out/regime/kis_breadth_probe.csv 에 쌓는다.
사용  python3 probe_kis_breadth.py
"""
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "LongLiveVault"))
from stolab_data import kis_fetcher as kf

FIELDS = ["bstp_nmix_prpr", "bstp_nmix_oprc", "bstp_nmix_prdy_ctrt", "ascn_issu_cnt", "uplm_issu_cnt", "stnr_issu_cnt",
          "down_issu_cnt", "lslm_issu_cnt", "acml_vol", "acml_tr_pbmn"]
tok = kf._get_token(); rows = []
for nm, code in (("KOSPI", "0001"), ("KOSDAQ", "1001"), ("KOSPI200", "2001")):   # 2001 = 애프터마켓 중 지수가 움직이는지 보려고 추가
    body = kf._kis_get(url=kf._INDEX_PRICE_URL, headers=kf._build_headers(tok, "FHPUP02100000"),
                       params={"FID_COND_MRKT_DIV_CODE": "U", "FID_INPUT_ISCD": code})
    out = (body or {}).get("output") or {}
    if nm == "KOSPI": print("응답 필드 수", len(out), "·", sorted(out.keys()))
    r = {"조회시각": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "시장": nm, **{f: out.get(f) for f in FIELDS}}
    rows.append(r); print(r)
f = HERE / "out" / "regime" / "kis_breadth_probe.csv"
pd.DataFrame(rows).to_csv(f, mode="a", header=not f.exists(), index=False, encoding="utf-8-sig")
print("저장", f)
