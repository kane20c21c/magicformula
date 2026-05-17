"""
signals/rules.py
----------------
3가지 진입 규칙(R1·R2·R3)과 4가지 청산 규칙(C1~C4)을 정의한다.

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
C1 : 종가 < 손절가(진입가 - ATR×1) → 다음날 시가 전량 청산
C2 : 분할 익절 (ATR×1 / ATR×2 / ATR×3) → 30% / 40% / 30%
C3 : composite_score ≤ -3 → 다음날 시가 잔량 전량 청산
C4 : 보유 20거래일 + 누적손익 ≤ 0% → 다음날 시가 잔량 전량 청산
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

# 분할 익절 비율
PARTIAL_RATIOS = (0.30, 0.40, 0.30)   # C2-1 / C2-2 / C2-3

# ATR 배수
ATR_STOP  = 1.0   # C1 손절
ATR_T1    = 1.0   # C2-1 1차 익절
ATR_T2    = 2.0   # C2-2 2차 익절
ATR_T3    = 3.0   # C2-3 3차 익절

# C3 점수 임계
SCORE_EXIT_THRESHOLD = -3.0

# C4 보유 기간 임계
MAX_HOLD_DAYS = 20


def compute_exit_prices(entry_price: float, atr: float) -> dict:
    """
    진입가 + ATR을 기반으로 손절가 / 3단계 익절가를 반환한다.

    Returns
    -------
    dict with keys: stop_loss, target1, target2, target3
    """
    return {
        "stop_loss": entry_price - ATR_STOP * atr,
        "target1":   entry_price + ATR_T1 * atr,
        "target2":   entry_price + ATR_T2 * atr,
        "target3":   entry_price + ATR_T3 * atr,
    }


def check_c1(close: float, stop_loss: float) -> bool:
    """C1: 종가 < 손절가 → True이면 다음날 시가 청산."""
    return close < stop_loss


def check_c2(
    high: float,
    targets: list[float],
    partial_done: list[bool],
) -> list[int]:
    """
    C2: 분할 익절 — 당일 고가가 각 익절 목표를 초과한 미완료 단계를 반환.

    Parameters
    ----------
    high         : 당일 고가
    targets      : [target1, target2, target3]
    partial_done : [done_1, done_2, done_3]

    Returns
    -------
    list of stage indices (0-based) that are newly triggered
    """
    triggered = []
    for i, (tgt, done) in enumerate(zip(targets, partial_done)):
        if not done and high >= tgt:
            triggered.append(i)
    return triggered


def check_c3(score: float) -> bool:
    """C3: 종합 점수 ≤ -3 → True이면 다음날 시가 잔량 청산."""
    return score <= SCORE_EXIT_THRESHOLD


def check_c4(
    days_held: int,
    unrealized_pnl_pct: float,
) -> bool:
    """
    C4: 보유 20거래일 + 누적손익 ≤ 0% → True이면 다음날 시가 청산.
    누적손익 > 0%이면 유지 (추세 추종).
    """
    return days_held >= MAX_HOLD_DAYS and unrealized_pnl_pct <= 0.0
