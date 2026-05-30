"""
analysis/trend_variants.py
--------------------------
추세 영역(Area 1) 점수 산출 변형 5종을 정의한다.

목적: 각 변형의 20거래일 forward direction 예측력을 검증해서
      "꾸준히 이어질 변화"를 가장 잘 잡아내는 공식을 채택.

변형 정의
---------
- baseline : 현재 score_trend (정배열±6 + 크로스±4 + 기울기±6) ÷3
            → 이론 max ±5.33 (영역 풀스케일의 53%)
- A        : sub-component cap을 ±10으로 (정±10 + 크±10 + 기±10) ÷3
            → 이론 max ±10
- C        : ÷N 제거, sum 후 클리핑. (정±6 + 크±4 + 기±6), clip(±10)
            → 이론 max ±10 (합산 16 → 클립)
- D        : 가중평균 (정 wa + 크 wc + 기 ws) / (wa+wc+ws)
            → 가중치 grid search 대상
- E        : Area 4 trend 방식의 연속 선형 (정 비율 평균 + 크±10 ffill + 기 ×50)
            → 이론 max ±10

데이터 부족(< 65행) 시 모두 0 시리즈 반환.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Sub-component scorers — 변형별 별도 정의
# ---------------------------------------------------------------------------

# === baseline 사용 sub (현행 score_trend과 동일) ===

def _align_baseline(ma5, ma20, ma60) -> pd.Series:
    """정배열 단계 ±6."""
    align = pd.Series(0.0, index=ma5.index)
    align[ma5 > ma20] = 3.0
    align[(ma5 > ma20) & (ma20 > ma60)] = 6.0
    align[(ma5 < ma20) & (ma20 < ma60)] = -6.0
    return align


def _cross_baseline(ma5, ma20) -> pd.Series:
    """골든/데드 크로스 ±4, 5일 ffill."""
    golden = ((ma5.shift(1) <= ma20.shift(1)) & (ma5 > ma20)).astype(float) * 4.0
    dead   = ((ma5.shift(1) >= ma20.shift(1)) & (ma5 < ma20)).astype(float) * -4.0
    cross_event = golden + dead
    return cross_event.replace(0.0, np.nan).ffill(limit=4).fillna(0.0)


def _slope_baseline(ma60) -> pd.Series:
    """MA60 5일 기울기% × 30, 클립 ±6."""
    slope_pct = (ma60 - ma60.shift(5)) / ma60.shift(5).replace(0, np.nan) * 100.0
    return (slope_pct * 30.0).clip(-6.0, 6.0)


# === A 변형 sub (cap ±10) ===

def _align_A(ma5, ma20, ma60) -> pd.Series:
    """정배열 단계 ±10 (베이스라인 ±6의 10/6 배)."""
    align = pd.Series(0.0, index=ma5.index)
    align[ma5 > ma20] = 5.0
    align[(ma5 > ma20) & (ma20 > ma60)] = 10.0
    align[(ma5 < ma20) & (ma20 < ma60)] = -10.0
    return align


def _cross_A(ma5, ma20) -> pd.Series:
    """크로스 ±10 (베이스라인 ±4의 10/4 배), 5일 ffill."""
    golden = ((ma5.shift(1) <= ma20.shift(1)) & (ma5 > ma20)).astype(float) * 10.0
    dead   = ((ma5.shift(1) >= ma20.shift(1)) & (ma5 < ma20)).astype(float) * -10.0
    cross_event = golden + dead
    return cross_event.replace(0.0, np.nan).ffill(limit=4).fillna(0.0)


def _slope_A(ma60) -> pd.Series:
    """MA60 5일 기울기% × 50, 클립 ±10 (베이스라인 ×30 ±6 → 거의 50/30 배)."""
    slope_pct = (ma60 - ma60.shift(5)) / ma60.shift(5).replace(0, np.nan) * 100.0
    return (slope_pct * 50.0).clip(-10.0, 10.0)


# === E 변형 sub (연속 선형, Area 4 trend 철학) ===

def _align_E(ma5, ma20, ma60) -> pd.Series:
    """
    정배열을 두 비율 평균의 연속 선형으로:
      ratio_5_20  = (MA5  / MA20 - 1) × 100   [%]
      ratio_20_60 = (MA20 / MA60 - 1) × 100   [%]
      score = clip( (ratio_5_20 + ratio_20_60) / 2 × 5, -10, +10 )

    해석: 두 비율 평균이 +1% → score +5, +2% → +10 (완전 정배열).
          단계 점프 없이 연속.
    """
    r1 = (ma5  / ma20.replace(0, np.nan) - 1.0) * 100.0
    r2 = (ma20 / ma60.replace(0, np.nan) - 1.0) * 100.0
    return (((r1 + r2) / 2.0) * 5.0).clip(-10.0, 10.0)


# E의 크로스/기울기는 A와 동일 (이미 풀 스케일)
_cross_E = _cross_A
_slope_E = _slope_A


# ---------------------------------------------------------------------------
# Variant 점수 함수
# ---------------------------------------------------------------------------

_MIN_ROWS = 65   # MA60 + 5일 기울기 여유


def _prep_mas(df: pd.DataFrame):
    """Close 기준 MA5/20/60 계산. 데이터 부족이면 None 반환."""
    if len(df) < _MIN_ROWS:
        return None
    close = df["Close"]
    return close.rolling(5).mean(), close.rolling(20).mean(), close.rolling(60).mean()


def score_baseline(df: pd.DataFrame) -> pd.Series:
    """현재 score_trend과 동일 (검증 기준점)."""
    mas = _prep_mas(df)
    if mas is None:
        return pd.Series(0.0, index=df.index)
    ma5, ma20, ma60 = mas
    a = _align_baseline(ma5, ma20, ma60)
    c = _cross_baseline(ma5, ma20)
    s = _slope_baseline(ma60)
    return ((a + c + s) / 3.0).clip(-10.0, 10.0).fillna(0.0)


def score_A(df: pd.DataFrame) -> pd.Series:
    """방안 A — sub cap ±10."""
    mas = _prep_mas(df)
    if mas is None:
        return pd.Series(0.0, index=df.index)
    ma5, ma20, ma60 = mas
    a = _align_A(ma5, ma20, ma60)
    c = _cross_A(ma5, ma20)
    s = _slope_A(ma60)
    return ((a + c + s) / 3.0).clip(-10.0, 10.0).fillna(0.0)


def score_C(df: pd.DataFrame) -> pd.Series:
    """방안 C — sub 베이스라인 그대로, ÷N 제거 후 클리핑."""
    mas = _prep_mas(df)
    if mas is None:
        return pd.Series(0.0, index=df.index)
    ma5, ma20, ma60 = mas
    a = _align_baseline(ma5, ma20, ma60)
    c = _cross_baseline(ma5, ma20)
    s = _slope_baseline(ma60)
    return (a + c + s).clip(-10.0, 10.0).fillna(0.0)


def score_D(df: pd.DataFrame, weights: tuple[float, float, float]) -> pd.Series:
    """
    방안 D (v1 — 호환용) — sub 베이스라인 그대로, 가중평균.

    공식: (wa·a + wc·c + ws·s) / Σw, 클립 ±10
    sub 베이스라인 max ±4~±6 이므로 가중평균도 그 범위에 머무름.
    Kane 2026-05-29: 이 변형은 ±10 풀스케일 달성 불가 — 폐기 권장.
    """
    mas = _prep_mas(df)
    if mas is None:
        return pd.Series(0.0, index=df.index)
    ma5, ma20, ma60 = mas
    a = _align_baseline(ma5, ma20, ma60)
    c = _cross_baseline(ma5, ma20)
    s = _slope_baseline(ma60)
    wa, wc, ws = weights
    w_sum = wa + wc + ws
    if w_sum <= 0:
        return pd.Series(0.0, index=df.index)
    return ((wa * a + wc * c + ws * s) / w_sum).clip(-10.0, 10.0).fillna(0.0)


def score_D_v2(df: pd.DataFrame, weights: tuple[float, float, float]) -> pd.Series:
    """
    방안 D v2 — A 변형의 sub (±10) 들의 가중평균.

    공식: wa·align_A + wc·cross_A + ws·slope_A,  단 Σw = 1
          → 모든 sub가 만점일 때 score = ±10 (풀 스케일)
          → 가중치를 통해 sub 간 중요도 조절 + ±10 도달 가능

    Parameters
    ----------
    weights : (w_align, w_cross, w_slope), Σ = 1.0, 각 > 0
    """
    mas = _prep_mas(df)
    if mas is None:
        return pd.Series(0.0, index=df.index)
    ma5, ma20, ma60 = mas
    a = _align_A(ma5, ma20, ma60)
    c = _cross_A(ma5, ma20)
    s = _slope_A(ma60)
    wa, wc, ws = weights
    w_sum = wa + wc + ws
    if w_sum <= 0:
        return pd.Series(0.0, index=df.index)
    # 가중합 = wa·(±10) + wc·(±10) + ws·(±10) = ±10·Σw, 정규화 후 ±10
    return ((wa * a + wc * c + ws * s) / w_sum).clip(-10.0, 10.0).fillna(0.0)


def score_E(df: pd.DataFrame) -> pd.Series:
    """방안 E — Area 4 trend 방식의 연속 선형 정배열 + A의 cross/slope."""
    mas = _prep_mas(df)
    if mas is None:
        return pd.Series(0.0, index=df.index)
    ma5, ma20, ma60 = mas
    a = _align_E(ma5, ma20, ma60)
    c = _cross_E(ma5, ma20)
    s = _slope_E(ma60)
    return ((a + c + s) / 3.0).clip(-10.0, 10.0).fillna(0.0)


# ---------------------------------------------------------------------------
# D 가중치 그리드 (3-simplex, step 0.1)
# ---------------------------------------------------------------------------

def d_weight_grid(step: int = 1) -> list[tuple[float, float, float]]:
    """
    3-simplex 위의 가중치 그리드 (v1, 호환용).

    step=1 (기본): w/10 단위, 합 = 1, 총 66개 조합 (0 가중치 허용).
    """
    total = 10 // step
    grid = []
    for i in range(total + 1):
        for j in range(total + 1 - i):
            k = total - i - j
            if i + j + k == total:
                grid.append((i / total, j / total, k / total))
    return grid


def d_weight_grid_v2(step: int = 1, min_step: int = 1) -> list[tuple[float, float, float]]:
    """
    모든 가중치가 양수(≥ min_step/total) 인 그리드 — v2 용.

    step=1, min_step=1 (기본): 각 가중치 ≥ 0.1, 합 = 1.0
                                → 총 36 조합  ( i,j,k ≥ 1, i+j+k=10 의 부분집합)
    step=1, min_step=2: 각 가중치 ≥ 0.2 → 더 좁은 grid
    """
    total = 10 // step
    grid = []
    for i in range(min_step, total + 1):
        for j in range(min_step, total + 1 - i):
            k = total - i - j
            if k >= min_step:
                grid.append((i / total, j / total, k / total))
    return grid
