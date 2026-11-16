# -*- coding: utf-8 -*-
"""fetch_rates.py — 무위험수익률(Rf) 후보 단기금리 수집 (케인 2026-11-17 요청, 성과지표 스터디 1-4).

원천  한국은행 ECOS Open API — 통계표 817Y002 「1.3.2.1. 시장금리(일별)」
      인증키: LLV .env 의 ECOS_API_KEY (2026-11-17 케인 발급)
항목  KOFR(공시RFR)        010901000  2021-11-25~   공식 무위험지표금리 (국채·통안 익일물 RP)
      CD(91일)             010502000  1995~         전통적 단기 Rf
      통안증권(91일)        010400000  2006-09~      정부측 단기물
      콜금리(1일, 전체거래)  010101000  1995~         초단기 은행간
      CMA(수시형)           010504000  1996~         증권계좌 대기현금의 실제 기회비용에 가까움
Rf    연구 기본 Rf = CALL1D (케인 2026-11-17). KOFR 는 2021-11-25 부터만 있어 장기 연구에 짧다.
      겹치는 1,220일 KOFR−콜1일: 평균 +0.003%p · 표준편차 0.057 · 상관 0.997 (CD91 은 평균 −0.21%p, 상관 0.950)
단위  연% (rate). 일간 Rf 로 쓸 때: (1 + rate/100) ** (1/252) - 1  (스터디 1-3·1-4)
⚠    금리는 '그날 고시된 연율' — 수익률이 아니다. 휴장일 행은 없다. KOFR 는 전영업일분이 다음날 11시 공시.
저장  data_idx/ecos_rates.parquet (long: date, name, code, rate)
사용  python3 fetch_rates.py [시작 YYYYMMDD] [끝 YYYYMMDD]
"""
import sys, json, time, datetime as dt, urllib.request
from pathlib import Path
import pandas as pd
from dotenv import dotenv_values

HERE = Path(__file__).resolve().parent
KEY = dotenv_values(HERE.parents[1] / "LongLiveVault" / ".env")["ECOS_API_KEY"]
STAT = "817Y002"
ITEMS = {
    "010901000": "KOFR",
    "010502000": "CD91",
    "010400000": "MSB91",
    "010101000": "CALL1D",
    "010504000": "CMA",
}
OUT = HERE / "data_idx" / "ecos_rates.parquet"
PAGE = 10000


def fetch_item(code: str, start: str, end: str) -> pd.DataFrame:
    rows, s = [], 1
    while True:
        url = (f"https://ecos.bok.or.kr/api/StatisticSearch/{KEY}/json/kr/"
               f"{s}/{s + PAGE - 1}/{STAT}/D/{start}/{end}/{code}")
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.loads(r.read().decode("utf-8"))
        block = d.get("StatisticSearch")
        if not block:
            msg = d.get("RESULT", {})
            if msg.get("CODE") == "INFO-200":   # 해당 데이터 없음
                break
            raise RuntimeError(f"ECOS 오류 {code}: {msg}")
        rows += block["row"]
        if len(rows) >= int(block["list_total_count"]):
            break
        s += PAGE
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return pd.DataFrame({
        "date": pd.to_datetime(df["TIME"], format="%Y%m%d"),
        "name": ITEMS[code],
        "code": code,
        "rate": pd.to_numeric(df["DATA_VALUE"], errors="coerce"),
    }).dropna(subset=["rate"])


def main(start="20000101", end=None):
    end = end or dt.date.today().strftime("%Y%m%d")
    parts = []
    for code, name in ITEMS.items():
        df = fetch_item(code, start, end)
        print(f"{name:7s} {code}  {len(df):6d}행  "
              f"{df['date'].min().date() if len(df) else '-'} ~ {df['date'].max().date() if len(df) else '-'}")
        parts.append(df)
        time.sleep(0.3)
    out = pd.concat(parts, ignore_index=True).sort_values(["name", "date"]).reset_index(drop=True)
    out.to_parquet(OUT, index=False)
    import kms_data, kms_store                  # MANIFEST 는 연구 데이터 공통 경로로 (kms_store.write_manifest)
    spec = kms_data.SETS["indices"]
    kms_store.write_manifest(spec["path"], {k: v for k, v in spec.items() if k != "path"})
    print(f"저장 {OUT.relative_to(HERE)}  {len(out)}행")


if __name__ == "__main__":
    main(*sys.argv[1:3])
