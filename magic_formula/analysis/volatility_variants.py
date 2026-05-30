"""
analysis/volatility_variants.py
-------------------------------
변동성·위치 영역(Area 4) 점수 변형 — BB %B + 52주 위치.

현 spec (baseline)
-----------------
- contrarian 모드: BB하단/52주저점 = 매수기회(+), BB상단/52주고점 = 위험(-)
- trend 모드: BB상단 = 강세지속(+), 연속선형 (%B-0.5)*20

핵심 질문: 위치 정보가 forward direction 예측하나? trend vs contra?

지표
----
- bb_pctb : 볼린저밴드 %B = (Close - BB_lower) / (BB_upper - BB_lower)
            0=하단, 0.5=중앙(MA20), 1=상단
- pos_52w : 52주 위치 = (Close - 52w_low) / (52w_high - 52w_low)
            0=52주저점, 1=52주고점

방향: trend (고점=+1) / contra (저점=+1)
인코딩: 연속선형 / 시그모이드(tanh) / 극단 임계
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from magic_formula.scoring.scorer import _bollinger, _clip

_MIN_ROWS = 60


def _bb_pctb(df: pd.DataFrame) -> pd.Series:
    """BB %B. parquet BB_upper/lower 있으면 사용, 없으면 계산."""
    close = df["Close"]
    if "BB_upper" in df.columns and "BB_lower" in df.columns and df["BB_upper"].notna().any():
        up, lo = df["BB_upper"], df["BB_lower"]
    else:
        up, _, lo, _ = _bollinger(close)
    denom = (up - lo).replace(0, np.nan)
    return (close - lo) / denom


def _pos_52w(df: pd.DataFrame) -> pd.Series:
    """52주(252일) 위치. 0=저점, 1=고점."""
    close = df["Close"]
    hi = close.rolling(252, min_periods=60).max()
    lo = close.rolling(252, min_periods=60).min()
    denom = (hi - lo).replace(0, np.nan)
    return (close - lo) / denom


# ---------------------------------------------------------------------------
# 인코딩 — 연속선형 / 시그모이드 / 극단 임계
# ---------------------------------------------------------------------------

def _linear(pos: pd.Series, direction: int) -> pd.Series:
    """위치 [0,1] → (pos-0.5)*20 → ±10. direction +1 trend(고점+) / -1 contra."""
    return _clip(direction * (pos - 0.5) * 20.0).fillna(0.0)


def _sigmoid(pos: pd.Series, direction: int, k: float = 0.2) -> pd.Series:
    """위치 → tanh((pos-0.5)/k) → ±10. k 작을수록 극단 민감."""
    return _clip(direction * 10.0 * np.tanh((pos - 0.5) / k)).fillna(0.0)


def _threshold(pos: pd.Series, direction: int, low: float = 0.2, high: float = 0.8) -> pd.Series:
    """5-band 극단 임계. high 위 / low 아래만 강신호. 중립=0 (선택적 베팅)."""
    mid_hi, mid_lo = (0.5 + high) / 2, (0.5 + low) / 2
    s = pd.Series(np.nan, index=pos.index)
    v = pos.notna(); s.loc[v] = 0.0
    s.loc[v & (pos >= high)]                   =  8.0
    s.loc[v & (pos >= mid_hi) & (pos < high)]  =  4.0
    s.loc[v & (pos > mid_lo) & (pos < mid_hi)] =  0.0
    s.loc[v & (pos > low) & (pos <= mid_lo)]   = -4.0
    s.loc[v & (pos <= low)]                    = -8.0
    return _clip(s * direction).fillna(0.0)


# ---------------------------------------------------------------------------
# 지표 추출 (driver 용)
# ---------------------------------------------------------------------------

def get_position(df: pd.DataFrame, kind: str) -> pd.Series:
    """kind ∈ {bb, pos52} → 위치 시리즈 [0,1]."""
    if len(df) < _MIN_ROWS:
        return pd.Series(np.nan, index=df.index)
    if kind == "bb":
        return _bb_pctb(df)
    elif kind == "pos52":
        return _pos_52w(df)
    raise ValueError(kind)


ENCODINGS = {
    "linear":    _linear,
    "sigmoid":   _sigmoid,
    "threshold": _threshold,
}


# ---------------------------------------------------------------------------
# BB × 52주 × 레짐 결합 점수표 (Kane 2026-05-30)
# ---------------------------------------------------------------------------
# 출처: BB위치×52주위치×레짐 4분면 forward 분석 (h=5).
# vs_bench → ±10 매핑: ≥+15→+10 / +8~15→+8 / +3~8→+5 / +1~3→+2 /
#            ±1→0 / -1~-3→-2 / -3~-8→-5 / <-8→-10

def _bb_bucket(x: float) -> str:
    if x > 1.0:   return "돌파상"
    if x >= 0.8:  return "상단"
    if x < 0.0:   return "돌파하"
    if x <= 0.2:  return "하단"
    return "중간"


def _p52_bucket(x: float) -> str:
    if x >= 0.8:  return "고점"
    if x <= 0.2:  return "저점"
    return "중간"


# (BB버킷, 52주버킷) → (강세점수, 하락조정점수)
# 강세장 조정 (Kane 2026-05-30): 강세장에선 down 베팅(-)이 상승 흐름과
# 싸워 손해. 강세장 음수/약신호를 0으로 (확실한 매수만 남김).
# 하락조정 컬럼은 4분면 vs_bench 그대로.
_JOINT_SCORE_TABLE = {
    # BB위치   52주    강세   하락조정   (강세 Kane 2026-05-30 수동 조정)
    ("돌파상", "고점"): ( +5,  -5),   # 강세 신고가추세 / 하락 막판강세 회피
    ("돌파상", "중간"): ( +7,  -2),   # Kane: 단기돌파+레인지내 → 강한 추세 시작
    ("돌파상", "저점"): (+10,   0),   # 강세 +30(n=7)
    ("상단",   "고점"): (  0,  -5),   # 강세 down베팅 금지
    ("상단",   "중간"): ( +2,  -5),   # Kane
    ("상단",   "저점"): ( +5, -10),   # 하락 n=3 (-25)
    ("중간",   "고점"): (  0,  -2),   # ★ 강세 최대표본 down베팅 금지
    ("중간",   "중간"): ( +2,  -2),   # Kane
    ("중간",   "저점"): (+10, +10),   # 전천후 최강 (+19.8 / +19.1)
    ("하단",   "고점"): (  0,  +8),
    ("하단",   "중간"): ( +4,  +5),
    ("하단",   "저점"): (+10,  +4),   # 강세 +22.9 / 하락 +4.1
    ("돌파하", "고점"): ( +8, +10),
    ("돌파하", "중간"): ( +4, +10),   # Kane: 강세 +4 (단기 눌림 매수) / 하락 +18.2
    ("돌파하", "저점"): (+10,   0),   # 강세 +30(n=7)
}


def score_joint_regime(df: pd.DataFrame, regime_ser: pd.Series) -> pd.Series:
    """
    BB %B × 52주 위치 × 레짐 결합 점수 (±10).

    regime_ser : 시점별 레짐 라벨 ('강세지속'/'강세약화'/'조정'/'하락'/'unknown').
                 강세지속·강세약화 → 강세 컬럼, 조정·하락 → 하락조정 컬럼.
    """
    if len(df) < _MIN_ROWS:
        return pd.Series(0.0, index=df.index)
    bb = _bb_pctb(df)
    p52 = _pos_52w(df)
    rg = regime_ser.reindex(df.index).ffill()

    out = pd.Series(0.0, index=df.index)
    valid = bb.notna() & p52.notna()
    for i in df.index[valid]:
        r = rg.get(i)
        if r in ("강세지속", "강세약화"):
            col = 0
        elif r in ("조정", "하락"):
            col = 1
        else:
            continue
        key = (_bb_bucket(bb.loc[i]), _p52_bucket(p52.loc[i]))
        sc = _JOINT_SCORE_TABLE.get(key)
        if sc is not None:
            out.loc[i] = float(sc[col])
    return _clip(out).fillna(0.0)
