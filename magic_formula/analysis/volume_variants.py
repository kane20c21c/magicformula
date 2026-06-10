"""
analysis/volume_variants.py
---------------------------
거래량 영역(Area 3) 점수 산출 변형 — 지표별 단독 + 결합.

현 점수 spec (baseline)
----------------------
- 상대거래량 (Vol/Vol_MA20): 구간 +8/+5/+2/0/-3 (★ 비대칭: max+8, min-3)
- OBV 5일 기울기: 상승 +5 / 하락 -5
→ 이론 max (8+5)/2=6.5, min (-3-5)/2=-4.0 (비대칭)

거래량 지표의 본질
-----------------
- 상대거래량: 방향 없음 (높은 거래량 = 매수일수도 매도일수도). 단독 예측 약함.
- OBV / AD_Line / Chaikin: 가격×거래량 결합 (money flow) → 방향성 보유.

활용 가능 지표 (parquet)
-----------------------
- Rel_Volume (상대거래량), OBV (계산), AD_Line, AD_Line_Chg, Chaikin_Osc

변형
----
지표별 단독 평가 (모멘텀 방식). 방향 양쪽 (trend/contra).
- rel_vol  : 상대거래량 z-score
- obv      : OBV 5일 기울기 z-score
- ad_line  : AD_Line 5일 기울기 z-score
- ad_chg   : AD_Line_Chg (일간 변화) z-score
- chaikin  : Chaikin Osc z-score

데이터 부족 시 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from magic_formula.indicators import _obv, _clip

_MIN_ROWS = 25
_Z_LOOKBACK = 60


def _zscore_tanh(series: pd.Series, direction: int = 1, scale: float = 1.0) -> pd.Series:
    """series 의 60일 rolling z-score → tanh → ±10. direction +1/-1."""
    std = series.rolling(_Z_LOOKBACK).std()
    z = series / std.replace(0, np.nan)
    return _clip(direction * 10.0 * np.tanh(z / scale)).fillna(0.0)


# ---------------------------------------------------------------------------
# 지표별 단독 점수 (z-score tanh, 방향 파라미터)
# ---------------------------------------------------------------------------

def score_rel_vol(df, direction=1, scale=1.0):
    """상대거래량 (Vol/Vol_MA20 - 1) z-score. 방향 모호 → 양쪽 테스트."""
    if len(df) < _MIN_ROWS: return pd.Series(0.0, index=df.index)
    if "Rel_Volume" in df.columns and df["Rel_Volume"].notna().any():
        rv = df["Rel_Volume"] - 1.0   # 1.0 = 평균
    else:
        vol_ma20 = df["Volume"].rolling(20).mean()
        rv = df["Volume"] / vol_ma20.replace(0, np.nan) - 1.0
    return _zscore_tanh(rv, direction, scale)


def score_obv(df, direction=1, scale=1.0):
    """OBV 5일 기울기 z-score."""
    if len(df) < _MIN_ROWS: return pd.Series(0.0, index=df.index)
    obv = _obv(df["Close"], df["Volume"])
    slope = obv - obv.shift(5)
    return _zscore_tanh(slope, direction, scale)


def score_ad_line(df, direction=1, scale=1.0):
    """AD_Line 5일 기울기 z-score (매집/분산 방향)."""
    if len(df) < _MIN_ROWS: return pd.Series(0.0, index=df.index)
    if "AD_Line" not in df.columns:
        return pd.Series(0.0, index=df.index)
    slope = df["AD_Line"] - df["AD_Line"].shift(5)
    return _zscore_tanh(slope, direction, scale)


def score_ad_chg(df, direction=1, scale=1.0):
    """AD_Line_Chg (일간 매집/분산 변화) z-score."""
    if len(df) < _MIN_ROWS: return pd.Series(0.0, index=df.index)
    if "AD_Line_Chg" not in df.columns:
        return pd.Series(0.0, index=df.index)
    return _zscore_tanh(df["AD_Line_Chg"], direction, scale)


def score_chaikin(df, direction=1, scale=1.0):
    """Chaikin Oscillator z-score (AD_Line 모멘텀)."""
    if len(df) < _MIN_ROWS: return pd.Series(0.0, index=df.index)
    if "Chaikin_Osc" not in df.columns:
        return pd.Series(0.0, index=df.index)
    return _zscore_tanh(df["Chaikin_Osc"], direction, scale)


# 지표 레지스트리 (driver 에서 사용)
INDICATORS = {
    "rel_vol":  score_rel_vol,
    "obv":      score_obv,
    "ad_line":  score_ad_line,
    "ad_chg":   score_ad_chg,
    "chaikin":  score_chaikin,
}
