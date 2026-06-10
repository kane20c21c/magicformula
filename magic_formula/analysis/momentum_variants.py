"""
analysis/momentum_variants.py
-----------------------------
모멘텀 영역(Area 2) 점수 산출 변형 5종 (+ D 그리드).

현 점수 spec (baseline)
----------------------
4 sub-component 단순 평균:
- RSI(14) 선형 변환  : (RSI - 50) / 5   → 이미 ±10 풀스케일
- Stoch %K(14) 선형 : (%K - 50) / 5    → 이미 ±10 풀스케일
- MACD vs Signal    : ±5  단계         → cap ±5 (이론 max 영향)
- MACD Hist 3일 방향 : ±2  단계         → cap ±2 (이론 max 영향)

이론 max = (10 + 10 + 5 + 2) / 4 = **6.75** (풀 범위의 67.5%)

변형
----
- baseline : 현 score_momentum
- A        : MACD vs Signal ±5→±10, Hist ±2→±10. max ±10.
- C        : ÷4 제거 후 클리핑. max sum 32 → clip 10.
- D        : A 변형 sub 의 4-simplex 가중평균 (각 가중치 ≥ 0.1, 합 = 1).
             → 56 조합.
- E        : MACD vs Signal, Hist 를 z-score 연속 선형으로.

데이터 부족(< 35행 = MACD 26 + signal 9) 시 모두 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from magic_formula.indicators import _rsi, _stoch_k, _macd, _clip


_MIN_ROWS = 35
_Z_LOOKBACK = 60   # E 변형의 z-score 표준편차 윈도우


# ---------------------------------------------------------------------------
# baseline sub (현 score_momentum 과 동일)
# ---------------------------------------------------------------------------

def _rsi_baseline(close: pd.Series) -> pd.Series:
    return _clip((_rsi(close) - 50.0) / 5.0)


def _stoch_baseline(high, low, close) -> pd.Series:
    return _clip((_stoch_k(high, low, close) - 50.0) / 5.0)


def _macd_vs_baseline(close: pd.Series) -> pd.Series:
    macd_line, signal_line, _ = _macd(close)
    valid = macd_line.notna() & signal_line.notna()
    s = pd.Series(np.nan, index=close.index)
    s.loc[valid & (macd_line >  signal_line)] =  5.0
    s.loc[valid & (macd_line <= signal_line)] = -5.0
    return s


def _hist_dir_baseline(close: pd.Series) -> pd.Series:
    _, _, hist = _macd(close)
    h1, h2, h3 = hist.shift(0), hist.shift(1), hist.shift(2)
    valid = h1.notna() & h2.notna() & h3.notna()
    s = pd.Series(np.nan, index=close.index)
    s.loc[valid] = 0.0
    s.loc[valid & (h1 > h2) & (h2 > h3)] =  2.0
    s.loc[valid & (h1 < h2) & (h2 < h3)] = -2.0
    return s


# ---------------------------------------------------------------------------
# A 변형 sub (cap ±10)
# ---------------------------------------------------------------------------

# RSI/Stoch 는 baseline 과 동일 (이미 ±10 선형)
_rsi_A   = _rsi_baseline
_stoch_A = _stoch_baseline


def _macd_vs_A(close: pd.Series) -> pd.Series:
    """baseline ±5 → A ±10."""
    macd_line, signal_line, _ = _macd(close)
    valid = macd_line.notna() & signal_line.notna()
    s = pd.Series(np.nan, index=close.index)
    s.loc[valid & (macd_line >  signal_line)] =  10.0
    s.loc[valid & (macd_line <= signal_line)] = -10.0
    return s


def _hist_dir_A(close: pd.Series) -> pd.Series:
    """baseline ±2 → A ±10."""
    _, _, hist = _macd(close)
    h1, h2, h3 = hist.shift(0), hist.shift(1), hist.shift(2)
    valid = h1.notna() & h2.notna() & h3.notna()
    s = pd.Series(np.nan, index=close.index)
    s.loc[valid] = 0.0
    s.loc[valid & (h1 > h2) & (h2 > h3)] =  10.0
    s.loc[valid & (h1 < h2) & (h2 < h3)] = -10.0
    return s


# ---------------------------------------------------------------------------
# E 변형 sub (연속 선형 — z-score)
# ---------------------------------------------------------------------------

_rsi_E   = _rsi_baseline   # RSI 는 이미 연속 선형
_stoch_E = _stoch_baseline


def _macd_vs_E(close: pd.Series) -> pd.Series:
    """
    (MACD - Signal) 의 60일 rolling z-score → 연속 선형 점수.

    z = (M-S) / rolling_std_60(M-S)
    score = clip(z, -10, +10)

    의미: z = 1 (1 std) → +1점, z = 3 (3 std) → +3점, z 큰 종목일수록 강함.
    단계 ±10 (A) 와 달리 신호 강도 표현.
    """
    macd_line, signal_line, _ = _macd(close)
    diff = macd_line - signal_line
    std  = diff.rolling(_Z_LOOKBACK).std()
    z    = diff / std.replace(0, np.nan)
    return z.clip(-10.0, 10.0)


def _hist_dir_E(close: pd.Series) -> pd.Series:
    """
    Hist 자체의 60일 z-score × 3 → 연속 선형. 3일 추세 대신 절대 강도.

    score = clip(hist / std_60(hist) × 3, -10, +10)
    """
    _, _, hist = _macd(close)
    std = hist.rolling(_Z_LOOKBACK).std()
    z   = hist / std.replace(0, np.nan)
    return (z * 3.0).clip(-10.0, 10.0)


# ---------------------------------------------------------------------------
# Variant 점수 함수
# ---------------------------------------------------------------------------

def _prep(df: pd.DataFrame):
    """OHLC 검증 + 데이터 부족 시 None."""
    if len(df) < _MIN_ROWS:
        return None
    return df["Close"], df["High"], df["Low"]


def score_baseline(df: pd.DataFrame) -> pd.Series:
    """현 score_momentum 과 동일."""
    p = _prep(df)
    if p is None: return pd.Series(0.0, index=df.index)
    close, high, low = p
    r = _rsi_baseline(close)
    s = _stoch_baseline(high, low, close)
    m = _macd_vs_baseline(close)
    h = _hist_dir_baseline(close)
    return _clip((r + s + m + h) / 4.0).fillna(0.0)


def score_A(df: pd.DataFrame) -> pd.Series:
    """방안 A — MACD/Hist sub cap ±10 으로 통일."""
    p = _prep(df)
    if p is None: return pd.Series(0.0, index=df.index)
    close, high, low = p
    r = _rsi_A(close)
    s = _stoch_A(high, low, close)
    m = _macd_vs_A(close)
    h = _hist_dir_A(close)
    return _clip((r + s + m + h) / 4.0).fillna(0.0)


def score_C(df: pd.DataFrame) -> pd.Series:
    """방안 C — baseline sub, ÷N 제거 후 클리핑."""
    p = _prep(df)
    if p is None: return pd.Series(0.0, index=df.index)
    close, high, low = p
    r = _rsi_baseline(close)
    s = _stoch_baseline(high, low, close)
    m = _macd_vs_baseline(close)
    h = _hist_dir_baseline(close)
    return _clip(r + s + m + h).fillna(0.0)


def score_D_v2(df: pd.DataFrame, weights: tuple[float, float, float, float]) -> pd.Series:
    """방안 D — A 변형 sub 가중평균. weights = (RSI, Stoch, MACD vs Sig, Hist).

    합 = 1.0, 각 > 0 → max ±10.
    """
    p = _prep(df)
    if p is None: return pd.Series(0.0, index=df.index)
    close, high, low = p
    r = _rsi_A(close)
    s = _stoch_A(high, low, close)
    m = _macd_vs_A(close)
    h = _hist_dir_A(close)
    wr, ws, wm, wh = weights
    wsum = wr + ws + wm + wh
    if wsum <= 0:
        return pd.Series(0.0, index=df.index)
    raw = (wr * r + ws * s + wm * m + wh * h) / wsum
    return _clip(raw).fillna(0.0)


def score_E(df: pd.DataFrame) -> pd.Series:
    """방안 E — RSI/Stoch baseline + MACD/Hist z-score 연속 선형."""
    p = _prep(df)
    if p is None: return pd.Series(0.0, index=df.index)
    close, high, low = p
    r = _rsi_E(close)
    s = _stoch_E(high, low, close)
    m = _macd_vs_E(close)
    h = _hist_dir_E(close)
    return _clip((r + s + m + h) / 4.0).fillna(0.0)


# ---------------------------------------------------------------------------
# F 변형 sub — 임계값 단계 (알려진 과매수/과매도 임계)
# ---------------------------------------------------------------------------
# 방향: dir=+1 추세추종 (과매수=+, 강세지속) / dir=-1 mean-reversion (과매수=-)

def _rsi_threshold(close: pd.Series, direction: int = 1) -> pd.Series:
    """RSI 임계 단계 (30/70). direction=+1 추세추종 / -1 meanrev."""
    rsi = _rsi(close)
    s = pd.Series(np.nan, index=close.index)
    valid = rsi.notna()
    s.loc[valid] = 0.0
    s.loc[valid & (rsi >= 70)]               =  8.0   # 과매수
    s.loc[valid & (rsi >= 55) & (rsi < 70)]  =  4.0
    s.loc[valid & (rsi > 45)  & (rsi < 55)]  =  0.0
    s.loc[valid & (rsi > 30)  & (rsi <= 45)] = -4.0
    s.loc[valid & (rsi <= 30)]               = -8.0   # 과매도
    return s * direction


def _stoch_threshold(high, low, close, direction: int = 1) -> pd.Series:
    """Stoch %K 임계 단계 (20/80). direction=+1 추세추종 / -1 meanrev."""
    k = _stoch_k(high, low, close)
    s = pd.Series(np.nan, index=close.index)
    valid = k.notna()
    s.loc[valid] = 0.0
    s.loc[valid & (k >= 80)]              =  8.0   # 과매수
    s.loc[valid & (k >= 60) & (k < 80)]  =  4.0
    s.loc[valid & (k > 40)  & (k < 60)]  =  0.0
    s.loc[valid & (k > 20)  & (k <= 40)] = -4.0
    s.loc[valid & (k <= 20)]             = -8.0   # 과매도
    return s * direction


# ---------------------------------------------------------------------------
# S 변형 sub — 시그모이드 (tanh) 비선형
# ---------------------------------------------------------------------------
# RSI: A·tanh((RSI-50)/k_rsi), k_rsi=20 → RSI 70 = tanh(1.0)=0.76 → +7.6
# Stoch: A·tanh((K-50)/k_stoch), k_stoch=30 → K 80 = tanh(1.0) → +7.6
# A = +10 추세추종 / -10 meanrev

K_RSI   = 20.0
K_STOCH = 30.0


def _rsi_sigmoid(close: pd.Series, direction: int = 1, k: float = K_RSI) -> pd.Series:
    rsi = _rsi(close)
    return direction * 10.0 * np.tanh((rsi - 50.0) / k)


def _stoch_sigmoid(high, low, close, direction: int = 1, k: float = K_STOCH) -> pd.Series:
    sk = _stoch_k(high, low, close)
    return direction * 10.0 * np.tanh((sk - 50.0) / k)


def _macd_vs_sigmoid(close: pd.Series, direction: int = 1) -> pd.Series:
    """(MACD - Signal) z-score → tanh. direction=+1 추세추종."""
    macd_line, signal_line, _ = _macd(close)
    diff = macd_line - signal_line
    std  = diff.rolling(_Z_LOOKBACK).std()
    z    = diff / std.replace(0, np.nan)
    return direction * 10.0 * np.tanh(z)


def _hist_sigmoid(close: pd.Series, direction: int = 1) -> pd.Series:
    """Hist z-score → tanh. direction=+1 추세추종."""
    _, _, hist = _macd(close)
    std = hist.rolling(_Z_LOOKBACK).std()
    z   = hist / std.replace(0, np.nan)
    return direction * 10.0 * np.tanh(z)


# ---------------------------------------------------------------------------
# F / S 변형 점수 함수 (4 sub 단순 평균)
# ---------------------------------------------------------------------------

def score_F_threshold(df: pd.DataFrame, direction: int = 1) -> pd.Series:
    """임계 단계 변형. direction=+1 추세추종 / -1 mean-reversion.

    MACD/Hist 는 baseline 단계 사용 (×direction).
    """
    p = _prep(df)
    if p is None: return pd.Series(0.0, index=df.index)
    close, high, low = p
    r = _rsi_threshold(close, direction)
    s = _stoch_threshold(high, low, close, direction)
    m = _macd_vs_baseline(close) * direction   # ±5 단계
    h = _hist_dir_baseline(close) * direction   # ±2 단계
    return _clip((r + s + m + h) / 4.0).fillna(0.0)


def score_S_sigmoid(df: pd.DataFrame, direction: int = 1) -> pd.Series:
    """시그모이드 (tanh) 변형. direction=+1 추세추종 / -1 mean-reversion."""
    p = _prep(df)
    if p is None: return pd.Series(0.0, index=df.index)
    close, high, low = p
    r = _rsi_sigmoid(close, direction)
    s = _stoch_sigmoid(high, low, close, direction)
    m = _macd_vs_sigmoid(close, direction)
    h = _hist_sigmoid(close, direction)
    return _clip((r + s + m + h) / 4.0).fillna(0.0)


# ---------------------------------------------------------------------------
# D 가중치 그리드 (4-simplex)
# ---------------------------------------------------------------------------

def d_weight_grid_v2(step: int = 1, min_step: int = 1) -> list[tuple[float, float, float, float]]:
    """
    4-simplex 그리드. 각 가중치 ≥ min_step/total, 합 = 1.

    step=1, min_step=1 (기본): C(9-1, 4-1) = C(8,3) = 56 조합.
                               각 sub 가중치 ∈ {0.1, 0.2, ..., 0.7}.
    step=1, min_step=2: 각 sub ≥ 0.2 → 더 좁은 그리드.
    """
    total = 10 // step
    grid = []
    for i in range(min_step, total + 1):
        for j in range(min_step, total + 1 - i):
            for k in range(min_step, total + 1 - i - j):
                l = total - i - j - k
                if l >= min_step:
                    grid.append((i / total, j / total, k / total, l / total))
    return grid
