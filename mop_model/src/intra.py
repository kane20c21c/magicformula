# -*- coding: utf-8 -*-
"""
intra.py — 화이트포트(장중) 타깃·피처. run_daily.py --target y_intra 가 사용.

experiments/intraday_eval.py 의 add_intra_target / add_intra_feats 를 그대로 옮긴 것
(검정 1단계 intra_plus 와 같은 정의). 실험 스크립트는 사전 등록본이라 건드리지 않고,
운영 코드가 실험과 어긋나지 않도록 tests 에서 두 구현의 결과 일치를 확인한다.

정의:
  intra_T1 = (1+C2C_T1)/(1+Gap_T1) − 1        익일 시가→종가 수익률 (features.parquet 두 라벨로 복원)
  y_intra  = intra_T1 > 그날 유니버스 중앙값   상대 이진 타깃 (데이포트 y_rel 의 장중판)
  피처 5개 (전부 날짜 t 의 O/C 까지만 사용):
    id1      당일 시가→종가
    id20     20일 평균 장중 수익률
    id_up20  20일 장중 양(+)의 날 비율
    gd20     20일 평균 갭
    gshare20 20일 갭 분산 / (갭 분산 + 장중 분산)
"""
import numpy as np, pandas as pd

INTRA_COLS = ["id20", "id_up20", "id1", "gd20", "gshare20"]
TARGET_INTRA = "y_intra"


def add_intra_target(d):
    d["intra_T1"] = (1.0 + d.C2C_T1) / (1.0 + d.Gap_T1) - 1.0
    med = d.groupby("Date").intra_T1.transform("median")
    d["y_intra"] = np.where(d.intra_T1.isna() | med.isna(), np.nan, (d.intra_T1 > med).astype(float))
    return d


def add_intra_feats(d):
    d = d.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    g = d.groupby("Ticker")
    d["id1"] = d.Close / d.Open - 1.0
    d["_gap0"] = d.Open / g.Close.shift(1) - 1.0
    d["id20"] = d.groupby("Ticker").id1.transform(lambda s: s.rolling(20, min_periods=15).mean())
    d["id_up20"] = d.groupby("Ticker").id1.transform(lambda s: (s > 0).astype(float).rolling(20, min_periods=15).mean())
    d["gd20"] = d.groupby("Ticker")._gap0.transform(lambda s: s.rolling(20, min_periods=15).mean())
    vg = d.groupby("Ticker")._gap0.transform(lambda s: s.rolling(20, min_periods=15).var())
    vi = d.groupby("Ticker").id1.transform(lambda s: s.rolling(20, min_periods=15).var())
    d["gshare20"] = vg / (vg + vi)
    return d.drop(columns=["_gap0"]).replace([np.inf, -np.inf], np.nan)


def prepare_intra(d, cols):
    """features → (타깃·피처 추가된 프레임, 확장 피처 목록). Date,Ticker 정렬로 돌려준다."""
    d = add_intra_feats(add_intra_target(d))
    d = d.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return d, list(cols) + INTRA_COLS
