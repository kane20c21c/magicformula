"""
signals/rules.py
----------------
v2_combined 매매 규칙 — configs/active_strategy.yaml `trading:` 섹션의 코드 구현.

진입 규칙 (threshold_breakout)
------------------------------
종합점수가 threshold 를 상향 돌파한 첫날 (prev <= threshold < today).
게이트로 제외된 날(NaN)은 전일/당일 어느 쪽이든 신호를 만들지 않는다.
체결: 다음날 시가 (simulator).

청산 규칙 (simulator 에서 호출)
------------------------------
C1   : 종가 < 손절가(진입가 − ATR(14)×1)            → 다음날 시가 전량 청산 (손절)
TIME : 보유 TIME_STOP_DAYS(20거래일) 이상 + 누적손익 ≤ 0% → 다음날 시가 전량 청산
       (hold_if_profit: 누적손익 > 0% 면 시간청산 하지 않음)
END  : 평가종료일 미청산 잔량 → 종가 강제 청산

변경 이력
---------
2026-06-10 v2 단일화: v1 규칙(R2/R3/ADX 필터/C_WY/C3/ADAPTIVE) 완전 삭제.
           yaml trading.exit 의 time_stop 을 코드로 구현 (기존 미구현).
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# 파라미터 (yaml trading: 섹션과 일치 — 변경 시 양쪽 함께)
# ---------------------------------------------------------------------------

ENTRY_THRESHOLD = 6.0    # 기본 진입 임계값 (yaml scoring.threshold 가 정본)

ATR_STOP        = 1.0    # C1 손절 (진입가 − ATR×1)
TIME_STOP_DAYS  = 20     # TIME 청산 — 보유 거래일 임계 (진입 체결일 = 1일째)


# ---------------------------------------------------------------------------
# 진입 신호
# ---------------------------------------------------------------------------

def entry_signals(
    scored_df: pd.DataFrame,
    threshold: float = ENTRY_THRESHOLD,
) -> pd.Series:
    """
    threshold_breakout 진입 신호를 생성한다.

    Parameters
    ----------
    scored_df : composite_score 컬럼을 가진 DataFrame
                (게이트 제외 구간은 NaN — 신호 생성 안 함)
    threshold : 진입 임계값 (yaml scoring.threshold)

    Returns
    -------
    pd.Series[bool] — True 인 날이 진입 신호일 (체결은 다음날 시가)
    """
    score = scored_df["composite_score"]
    prev = score.shift(1)
    # NaN(게이트/워밍업) 은 비교 결과가 False 가 되어 자동 제외
    signal = (prev <= threshold) & (score > threshold)
    return signal.fillna(False)


def check_breakout_signal(scored: pd.DataFrame, threshold: float) -> bool:
    """
    최신 날짜 기준 돌파 신호 여부 (이전 ≤ threshold < 오늘). 데일리 트랙용.

    반환 타입은 Python ``bool`` — numpy.bool_ leak 방지 (JSON 직렬화 호환).
    """
    score = scored["composite_score"]
    if len(score) < 2:
        return False
    prev, cur = score.iloc[-2], score.iloc[-1]
    if pd.isna(prev) or pd.isna(cur):
        return False
    return bool(prev <= threshold < cur)


# ---------------------------------------------------------------------------
# 청산 조건 (포지션 단위)
# ---------------------------------------------------------------------------

def compute_stop_loss(entry_price: float, atr: float) -> float:
    """C1 손절가 계산 (진입가 − ATR×1)."""
    return entry_price - ATR_STOP * atr


def check_stop_loss(close: float, stop_loss: float) -> bool:
    """C1: 종가 < 손절가 → True 이면 다음날 시가 전량 청산."""
    return close < stop_loss


def check_time_stop(
    hold_days:      int,
    cum_return_pct: float,
    max_days:       int = TIME_STOP_DAYS,
) -> bool:
    """
    TIME: 보유 max_days 거래일 이상 + 누적손익 ≤ 0% → True.

    hold_if_profit — 누적손익 > 0% 면 보유일과 무관하게 청산하지 않는다.

    Parameters
    ----------
    hold_days      : 보유 거래일 수 (진입 체결일 = 1)
    cum_return_pct : 진입가 대비 당일 종가 수익률 (%)
    max_days       : 시간청산 임계 (기본 TIME_STOP_DAYS = 20)
    """
    return hold_days >= max_days and cum_return_pct <= 0.0
