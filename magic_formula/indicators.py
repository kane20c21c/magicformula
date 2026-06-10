"""
indicators.py
-------------
순수 기술지표 헬퍼 모음 (v2 단일화 시 scoring/scorer.py 에서 추출).

모든 함수는 과거 데이터만 사용하는 연산(rolling / ewm / shift)으로 구성되어
look-ahead bias 가 없다. 이름은 기존 scorer.py 와의 호환을 위해
underscore 형태를 유지한다 (_rsi, _macd, ...).

소비자
------
- magic_formula.analysis.area_scores      (v2 운영 점수)
- magic_formula.analysis.*_variants       (분석 연구용 변형)
- magic_formula.simulator / optimizer     (atr14)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clip(series: pd.Series, lo: float = -10.0, hi: float = 10.0) -> pd.Series:
    """시리즈 값을 [lo, hi]로 클리핑."""
    return series.clip(lower=lo, upper=hi)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder 방식 RSI (EWM alpha=1/period)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _stoch_k(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Stochastic %K."""
    lo = low.rolling(period).min()
    hi = high.rolling(period).max()
    denom = (hi - lo).replace(0.0, np.nan)
    return (close - lo) / denom * 100.0


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD, Signal, Histogram 반환."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(close: pd.Series, period: int = 20, std_mult: float = 2.0):
    """BB 중심선, 상단, 하단, %B 반환."""
    ma = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=1)
    upper = ma + std_mult * std
    lower = ma - std_mult * std
    denom = (upper - lower).replace(0.0, np.nan)
    pct_b = (close - lower) / denom
    return ma, upper, lower, pct_b


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ATR (Average True Range)."""
    hl = high - low
    hpc = (high - close.shift(1)).abs()
    lpc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()
