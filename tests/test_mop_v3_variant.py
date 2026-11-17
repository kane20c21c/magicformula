# -*- coding: utf-8 -*-
"""
tests/test_mop_v3_variant.py — 데이포트 그림자 v3 (run_daily --variant v3 = 2.0 후보 BG1MD 97) 회귀 테스트 (2026-11-17)
=========================================================================================================
결정: 케인 2026-11-17 — 데이포트 2.0 피처 사다리에서 cur 과 함께 기준으로 둔 BG1MD 를 그림자 R11 로 붙인다.

지키는 것:
  1) v3 구성 = 정리 BASE 41 + 그 대섹터 순위 41 + 시장 10 + 요일 + 종목 드라이브 4 = 97, 중복 없음,
     미래 정보(vol_idx·wy_*)·기준점(close_idx)·같은 값 중복(vol_ratio·energy_index) 제외
  2) 드라이브 정의 (길이 부호·크기·속도·갭 몫)
  3) add_v3 는 행 수·순서를 바꾸지 않고, 시장 값은 날짜로, 대섹터 순위는 5종목 이상 섹터에만 붙인다
  4) --variant v3 는 output/signals_v3 로만 쓰고 데이포트 타깃 전용이다
모델 학습 없음 — 합성 패널, 시장·YZ 는 주입.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_HERE = Path(__file__).resolve().parent
_MOP = _HERE.parent / "mop_model"
if str(_MOP / "src") not in sys.path:
    sys.path.insert(0, str(_MOP / "src"))

import config as cfg  # noqa: E402
import build_features as bf  # noqa: E402
import features_v3 as F3  # noqa: E402

_spec = importlib.util.spec_from_file_location("_v2t", _HERE / "test_mop_v2_variant.py")
_v2t = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_v2t)


def _cols_all():
    px = _v2t._panel(n_days=30, n_tk=6)
    return {"BASE": list(bf._base(px).columns)}


def test_v3_cols_composition():
    cols = F3.v3_cols(_cols_all())
    assert len(cols) == 97 and len(set(cols)) == 97
    for c in ["vol_idx_c", "YZ_20", "YZ_60", "s1_YZ_20", "s1_vol_idx_c", "s1_rsi", "m_p7", "m_rv18", "m_drv_gap", "dow", "sdrv_gap"]:
        assert c in cols, c
    for c in ["vol_idx", "wy_label", "wy_phase", "close_idx", "vol_ratio", "energy_index", "smart_pressure",
              "retail_pressure", "hope_vector", "s1_dow", "xs_rsi", "s5_rsi"]:
        assert c not in cols, c
    b = F3.base41(_cols_all())
    assert len(b) == 41 and cols[:41] == b and cols[41:82] == ["s1_" + c for c in b]


def test_drive_cols_definition():
    df = pd.DataFrame({"k": "A", "Open": [100, 100.5, 101.5, 101.5, 100.5, 100, 102],
                       "Close": [100, 101, 102, 101, 100, 100, 103]})
    r = F3.drive_cols(df, "k")
    assert r["len"].tolist() == [0, 1, 2, -1, -2, 0, 1]
    assert r["size"].iloc[2] == pytest.approx(102 / 100 - 1)
    assert r["speed"].iloc[2] == pytest.approx((1 / 101) / 0.01)
    assert r["gap"].iloc[1] == pytest.approx(np.log(100.5 / 100) / np.log(101 / 100))
    assert pd.isna(r["speed"].iloc[1])                                        # 첫날은 속도 없음


def test_add_v3_keeps_rows_and_fills():
    px = _v2t._panel(n_days=40, n_tk=12)
    d = bf.build_features(panel_df=px, save=False).sample(frac=1, random_state=1)   # 순서 섞어서 유지 확인
    days = pd.DatetimeIndex(sorted(px.Date.unique()))
    mk = pd.DataFrame({c: np.arange(len(days), dtype=float) for c in F3.MCOL}, index=days)
    yz = px[["Date", "Ticker"]].copy(); yz["YZ_20"] = np.linspace(0.01, 0.05, len(yz)); yz["YZ_60"] = yz.YZ_20 * 2
    out = F3.add_v3(d, market=mk, yz=yz)
    assert len(out) == len(d) and (out.index == d.index).all()
    i = out.index[0]
    assert out.loc[i, "m_p7"] == days.get_loc(pd.Timestamp(out.loc[i, "Date"]))
    assert out["s1_YZ_20"].between(0, 1).all()                               # 섹터당 6종목(≥5) → 전부 값
    small = out.copy(); small.loc[small.Ticker == "000000", "sector_top"] = "ALONE"
    out2 = F3.add_v3(small.drop(columns=["s1_YZ_20", "s1_YZ_60"]), market=mk, yz=yz)
    assert out2.loc[out2.sector_top == "ALONE", "s1_YZ_20"].isna().all()       # 1종목 섹터 → 빈칸
    assert set(F3.SCOL) <= set(out.columns)


def test_v3_profile():
    import run_daily as rd
    assert rd.VARIANTS["v3"] == F3.__dict__.get("V3_KEY", "BG1MD") == rd.V3_KEY
    prof = rd.resolve_profile(cfg.TARGET, "v3")
    assert prof["signal_dir"].endswith("signals_v3") and prof["variant"] == "v3"
    with pytest.raises(SystemExit):
        rd.resolve_profile("y_intra", "v3")
