"""
signals/rules.py
----------------
3가지 진입 규칙(R1·R2·R3)과 3가지 청산 규칙(C1·C_WY·C3)을 정의한다.

진입 규칙
---------
R1 : composite_score 가 +threshold를 상향 돌파한 첫날
     (이전 score ≤ threshold, 당일 score > threshold)
R2 : composite_score 가 +threshold 이상이면서 해당 종목 미보유 상태일 때
     (절대 수준 진입)
R3 : composite_score 가 음수→양수 부호 전환 AND ADX(14) > 20
     (추세 강도 확인 필터 추가 — 횡보장 허위 신호 제거)

청산 규칙 (simulator 에서 호출)
---------
C1   : 종가 < 손절가(진입가 - ATR×1)          → 다음날 시가 전량 청산 (손절)
C_WY : 와이코프 추세 전환 신호 (v3 신규)       → 다음날 시가 전량 청산 (익절/추세종료)
         - 신호1: composite_score 가 N일(기본 3) 연속 음수 전환
         - 신호2: 진입 점수 대비 Δ(기본 4.0) 이상 하락 + 현재 점수 음수
C3   : composite_score ≤ -3                   → 다음날 시가 잔량 전량 청산 (급락 안전망)

변경 이력
---------
v3 : C2(ATR 분할 익절) 제거 → C_WY(와이코프 점수 반전) 로 대체.
     단순 목표가 도달 매도 → 추세 전환 신호 발생 시 매도로 전환.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# ADX(14) 계산 헬퍼  — R3 추세 강도 필터에 사용
# ---------------------------------------------------------------------------

ADX_PERIOD    = 14      # Wilder 표준 기간
ADX_THRESHOLD = 20.0    # R3 필터 임계값 (ADX > 20 = 추세 확인)


def _compute_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.Series:
    """
    Wilder 방식 ADX(Average Directional Index) 계산.

    계산 순서:
      1. True Range (TR)   → Wilder EWM 평활로 ATR
      2. +DM / -DM         → +DI / -DI (각각 ATR으로 정규화, ×100)
      3. DX = |+DI - -DI| / (+DI + -DI) × 100
      4. ADX = DX의 Wilder EWM 평활

    Look-ahead bias 없음: 모두 과거 데이터(shift 또는 rolling)만 사용.

    Parameters
    ----------
    df     : High / Low / Close 컬럼이 있는 DataFrame
    period : 평활 기간 (기본 14)

    Returns
    -------
    pd.Series (0 ~ 100) — 초기 rows는 NaN
    """
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]

    # True Range
    hl  = high - low
    hpc = (high - close.shift(1)).abs()
    lpc = (low  - close.shift(1)).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)

    # Directional Movement
    up_move   = high - high.shift(1)
    dn_move   = low.shift(1) - low

    plus_dm  = pd.Series(
        np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0),
        index=close.index,
    )
    minus_dm = pd.Series(
        np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0),
        index=close.index,
    )

    # Wilder 평활 (EWM alpha = 1/period)
    alpha    = 1.0 / period
    atr_w    = tr.ewm(alpha=alpha, adjust=False).mean()
    denom    = atr_w.replace(0.0, np.nan)
    plus_di  = plus_dm.ewm(alpha=alpha, adjust=False).mean()  / denom * 100.0
    minus_di = minus_dm.ewm(alpha=alpha, adjust=False).mean() / denom * 100.0

    # DX → ADX
    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx     = (plus_di - minus_di).abs() / di_sum * 100.0
    adx    = dx.ewm(alpha=alpha, adjust=False).mean()

    return adx


# ---------------------------------------------------------------------------
# 진입 신호 생성
# ---------------------------------------------------------------------------

ENTRY_THRESHOLD = 5.0    # R1 / R2 기본 임계값 (CLI --threshold 로 재정의 가능)
SIGN_THRESHOLD  = 0.0    # R3 부호 전환 임계값 (고정)


def entry_signals(
    scored_df: pd.DataFrame,
    rule: str,
    already_in: pd.Series | None = None,
    threshold: float = ENTRY_THRESHOLD,
) -> pd.Series:
    """
    종합 점수 시리즈에서 진입 신호를 생성한다.

    Parameters
    ----------
    scored_df  : compute_scores() 가 반환한 DataFrame
                 (composite_score 컬럼 필요)
    rule       : 'R1', 'R2', 'R3' 중 하나
    already_in : 날짜별 보유 여부 (True=보유 중). None이면 항상 미보유로 간주.
                 R2 에서 이미 포지션이 있는 날에는 신호를 생성하지 않는다.
    threshold  : R1·R2 진입 임계값 (기본 5.0).
                 점수 분포에 따라 낮출 수 있음 (예: 1.5).
                 R3 는 항상 부호 전환(0 기준)을 사용하므로 영향 없음.

    Returns
    -------
    pd.Series[bool]  — True인 날이 진입 신호일
    """
    score = scored_df["composite_score"]
    signal = pd.Series(False, index=score.index)

    if already_in is None:
        already_in = pd.Series(False, index=score.index)

    if rule == "R1":
        # 전일 ≤ threshold 이고 당일 > threshold (상향 돌파)
        prev = score.shift(1)
        signal = (prev <= threshold) & (score > threshold)

    elif rule == "R2":
        # 당일 score > threshold 이면서 미보유
        signal = (score > threshold) & (~already_in)

    elif rule == "R3":
        # 부호 전환 AND ADX(14) > 25 — threshold 무관
        # ADX 필터: 횡보장(ADX 낮음)에서 발생하는 허위 부호전환 신호 제거
        prev = score.shift(1)
        sign_flip = (prev < SIGN_THRESHOLD) & (score >= SIGN_THRESHOLD)

        # scored_df 에 OHLCV 컬럼이 있으면 ADX 계산, 없으면 필터 미적용
        if all(c in scored_df.columns for c in ("High", "Low", "Close")):
            adx = _compute_adx(scored_df, period=ADX_PERIOD)
            adx_ok = adx > ADX_THRESHOLD          # True = 추세 있음
            adx_ok = adx_ok.reindex(score.index, fill_value=False)
            signal = sign_flip & adx_ok
        else:
            signal = sign_flip   # OHLCV 없으면 ADX 생략(호환성)

    else:
        raise ValueError(f"알 수 없는 진입 규칙: {rule}. 'R1', 'R2', 'R3' 중 택일.")

    # NaN이 생기는 첫 행 처리
    return signal.fillna(False)


# ---------------------------------------------------------------------------
# 청산 조건 평가 (포지션 단위로 호출)
# ---------------------------------------------------------------------------

# ATR 배수
ATR_STOP = 1.0   # C1 손절 (진입가 - ATR×1)

# C_WY (와이코프 추세 전환) 파라미터
WY_CONSEC_NEG_DAYS: int   = 3    # 연속 음전환 일수 임계
WY_SCORE_DROP:      float = 4.0  # 진입 점수 대비 낙폭 임계

# C3 점수 임계 (급락 안전망)
SCORE_EXIT_THRESHOLD = -3.0

def compute_stop_loss(entry_price: float, atr: float) -> float:
    """C1 손절가 계산 (진입가 - ATR×1)."""
    return entry_price - ATR_STOP * atr


def check_c1(close: float, stop_loss: float) -> bool:
    """C1: 종가 < 손절가 → True이면 다음날 시가 전량 청산."""
    return close < stop_loss


def check_wyckoff_exit(
    current_score:    float,
    entry_score:      float,
    consec_neg_days:  int,
    min_consec:       int   = WY_CONSEC_NEG_DAYS,
    max_drop:         float = WY_SCORE_DROP,
) -> bool:
    """
    C_WY: 와이코프 추세 전환 신호 판단.

    두 가지 조건 중 하나라도 충족하면 True → 다음날 시가 전량 청산.

    신호1 — 연속 음전환
        보유 기간 중 composite_score 가 min_consec 일 연속 0 미만.
        추세 종료 / 계단형 하락의 전형적 패턴.

    신호2 — 진입 대비 score 급락
        entry_score - current_score ≥ max_drop  AND  current_score < 0.
        진입 시점 대비 강도가 크게 떨어지고 이미 음수권 → 분배 국면 진입 가능성.

    Parameters
    ----------
    current_score   : 당일 composite_score
    entry_score     : 진입 당일 composite_score (Position 에 저장)
    consec_neg_days : 보유 기간 중 연속 음수 일수 카운터 (Position 에서 갱신)
    min_consec      : 신호1 임계 (기본 WY_CONSEC_NEG_DAYS = 3)
    max_drop        : 신호2 score 낙폭 임계 (기본 WY_SCORE_DROP = 4.0)
    """
    # 신호1: N일 연속 음수 전환
    if consec_neg_days >= min_consec:
        return True
    # 신호2: 진입 대비 score 급락 + 현재 음수
    if (entry_score - current_score >= max_drop) and (current_score < 0):
        return True
    return False


def check_c3(score: float) -> bool:
    """C3: 종합 점수 ≤ -3 → True이면 다음날 시가 잔량 전량 청산 (급락 안전망)."""
    return score <= SCORE_EXIT_THRESHOLD


