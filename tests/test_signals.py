"""
tests/test_signals.py
---------------------
진입 규칙(R1) 신호 감지 동작 — vault 의존성 없는 빠른 검사.
"""

from __future__ import annotations

import pandas as pd
import pytest

from magic_formula.daily.runner import check_r1_signal


def _make_score_series(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"composite_score": values})


# ---------------------------------------------------------------------------
# R1 신호 — 임계 상향 돌파 (prev <= threshold < today)
# ---------------------------------------------------------------------------

def test_r1_signal_on_upward_breakout():
    """4.5 → 5.5 (threshold=5.0) 상향 돌파 → True."""
    scored = _make_score_series([4.5, 5.5])
    assert check_r1_signal(scored, threshold=5.0) is True


def test_r1_no_signal_when_already_above():
    """5.5 → 6.0 (둘 다 임계값 이상) → False (이미 위에 있어 새 진입 X)."""
    scored = _make_score_series([5.5, 6.0])
    assert check_r1_signal(scored, threshold=5.0) is False


def test_r1_no_signal_when_decreasing():
    """5.5 → 4.5 (하향) → False."""
    scored = _make_score_series([5.5, 4.5])
    assert check_r1_signal(scored, threshold=5.0) is False


def test_r1_signal_exact_threshold_boundary():
    """5.0 → 5.1 (prev==threshold, today>threshold) → True."""
    scored = _make_score_series([5.0, 5.1])
    assert check_r1_signal(scored, threshold=5.0) is True


def test_r1_no_signal_too_short_series():
    """row 1개 → False (이전 점수 비교 불가)."""
    scored = _make_score_series([5.5])
    assert check_r1_signal(scored, threshold=5.0) is False
