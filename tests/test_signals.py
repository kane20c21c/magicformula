"""
tests/test_signals.py
---------------------
진입(threshold_breakout) + 청산(C1/TIME) 규칙 — vault 의존성 없는 빠른 검사.
"""

from __future__ import annotations

import pandas as pd

from magic_formula.signals.rules import (
    check_breakout_signal,
    check_stop_loss,
    check_time_stop,
    compute_stop_loss,
    entry_signals,
)


def _scored(values: list) -> pd.DataFrame:
    return pd.DataFrame({"composite_score": values})


# ---------------------------------------------------------------------------
# 진입 — 임계 상향 돌파 (prev <= threshold < today)
# ---------------------------------------------------------------------------

def test_breakout_on_upward_cross():
    """4.5 → 5.5 (threshold=5.0) 상향 돌파 → True."""
    assert check_breakout_signal(_scored([4.5, 5.5]), threshold=5.0) is True


def test_no_signal_when_already_above():
    """5.5 → 6.0 (둘 다 임계값 이상) → False."""
    assert check_breakout_signal(_scored([5.5, 6.0]), threshold=5.0) is False


def test_no_signal_when_decreasing():
    """5.5 → 4.5 (하향) → False."""
    assert check_breakout_signal(_scored([5.5, 4.5]), threshold=5.0) is False


def test_signal_exact_threshold_boundary():
    """5.0 → 5.1 (prev==threshold, today>threshold) → True."""
    assert check_breakout_signal(_scored([5.0, 5.1]), threshold=5.0) is True


def test_no_signal_too_short_series():
    """row 1개 → False (이전 점수 비교 불가)."""
    assert check_breakout_signal(_scored([5.5]), threshold=5.0) is False


def test_no_signal_when_gated_nan():
    """게이트 제외(NaN)가 전일/당일 어느 쪽이든 → False."""
    assert check_breakout_signal(_scored([float("nan"), 6.5]), threshold=5.0) is False
    assert check_breakout_signal(_scored([4.5, float("nan")]), threshold=5.0) is False


def test_entry_signals_series():
    """시리즈 전체에서 돌파일만 True, NaN 은 False."""
    df = _scored([3.0, 4.0, 6.5, 7.0, float("nan"), 7.0, 2.0, 6.1])
    sig = entry_signals(df, threshold=6.0)
    assert sig.tolist() == [False, False, True, False, False, False, False, True]


# ---------------------------------------------------------------------------
# 청산 — C1 (ATR 손절)
# ---------------------------------------------------------------------------

def test_stop_loss_level_and_trigger():
    stop = compute_stop_loss(entry_price=10_000.0, atr=500.0)
    assert stop == 9_500.0
    assert check_stop_loss(close=9_499.0, stop_loss=stop) is True
    assert check_stop_loss(close=9_500.0, stop_loss=stop) is False   # 같으면 유지


# ---------------------------------------------------------------------------
# 청산 — TIME (20거래일 + 손익≤0, 이익 중이면 유지)
# ---------------------------------------------------------------------------

def test_time_stop_triggers_when_underwater():
    assert check_time_stop(hold_days=20, cum_return_pct=-1.5) is True
    assert check_time_stop(hold_days=25, cum_return_pct=0.0) is True   # 0% 도 청산


def test_time_stop_holds_if_profit():
    """hold_if_profit — 누적손익 > 0% 면 보유일 무관 유지."""
    assert check_time_stop(hold_days=20, cum_return_pct=0.1) is False
    assert check_time_stop(hold_days=100, cum_return_pct=5.0) is False


def test_time_stop_not_before_max_days():
    assert check_time_stop(hold_days=19, cum_return_pct=-10.0) is False
