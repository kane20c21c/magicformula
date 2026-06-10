"""
tests/test_area_scores.py
-------------------------
v2 운영 점수 정본(analysis/area_scores) — 합성 데이터 검사.

- 4영역 점수 범위(±10) / compute_area_scores 키
- combine_scores 가중 결합·게이트(NaN)·가중치 합 0 거부
- compute_combined_score == compute_area_scores + combine_scores 일관성
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from magic_formula.analysis import area_scores as A


# ---------------------------------------------------------------------------
# 합성 데이터
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 300, seed: int = 7) -> pd.DataFrame:
    """완만한 상승 추세 + 노이즈 합성 OHLCV."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-02", periods=n)
    drift = np.linspace(0, 0.4, n)
    noise = rng.normal(0, 0.01, n).cumsum()
    close = 10_000.0 * np.exp(drift + noise)
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.004, n)))
    vol = rng.integers(50_000, 200_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


@pytest.fixture(scope="module")
def df():
    return _make_ohlcv()


@pytest.fixture(scope="module")
def regimes(df):
    """상수 레짐 시리즈 2종 (조정 — 거래량 bear-only 경로도 활성화)."""
    rb = pd.Series("조정", index=df.index, dtype=object)
    rq = pd.Series("조정", index=df.index, dtype=object)
    return rb, rq


# ---------------------------------------------------------------------------
# 영역 점수
# ---------------------------------------------------------------------------

def test_compute_area_scores_keys_and_range(df, regimes):
    rb, rq = regimes
    areas = A.compute_area_scores(df, rb, rq)
    assert set(areas.keys()) == set(A.AREA_KEYS)
    for k, s in areas.items():
        assert isinstance(s, pd.Series), k
        assert len(s) == len(df), k
        valid = s.dropna()
        assert (valid.abs() <= 10.0 + 1e-9).all(), f"{k} 점수가 ±10 범위 초과"


def test_short_data_returns_zero_scores(regimes):
    """데이터 부족(<35행) 시 모멘텀/거래량은 0 시리즈."""
    short = _make_ohlcv(20)
    rb = pd.Series("조정", index=short.index, dtype=object)
    assert (A.score_momentum(short) == 0.0).all()
    assert (A.score_volume(short, rb) == 0.0).all()


# ---------------------------------------------------------------------------
# 결합 + 게이트
# ---------------------------------------------------------------------------

def test_combine_scores_weighted_sum(df, regimes):
    """단일 영역 가중치 1.0 → 그 영역 점수와 동일."""
    rb, rq = regimes
    areas = A.compute_area_scores(df, rb, rq)
    w = {"trend": 0.0, "momentum": 1.0, "volume": 0.0, "volatility": 0.0}
    comp = A.combine_scores(areas, w, phase_label=None, gate=False)
    pd.testing.assert_series_equal(
        comp.dropna(), areas["momentum"].dropna(), check_names=False)


def test_combine_scores_gate_sets_nan(df, regimes):
    """Markdown 국면 구간은 NaN (매수 후보 제외)."""
    rb, rq = regimes
    areas = A.compute_area_scores(df, rb, rq)
    phase = pd.Series("Markup", index=df.index, dtype=object)
    phase.iloc[100:120] = "Markdown"
    comp = A.combine_scores(areas, A.COMBINED_WEIGHTS, phase, gate=True)
    assert comp.iloc[100:120].isna().all()
    assert comp.iloc[50:60].notna().all()


def test_combine_scores_gate_off_keeps_values(df, regimes):
    rb, rq = regimes
    areas = A.compute_area_scores(df, rb, rq)
    phase = pd.Series("Markdown", index=df.index, dtype=object)
    comp = A.combine_scores(areas, A.COMBINED_WEIGHTS, phase, gate=False)
    assert comp.notna().all()


def test_combine_scores_zero_weight_sum_raises(df, regimes):
    rb, rq = regimes
    areas = A.compute_area_scores(df, rb, rq)
    w = {"trend": 0.0, "momentum": 0.0, "volume": 0.0, "volatility": 0.0}
    with pytest.raises(ValueError, match="가중치 합"):
        A.combine_scores(areas, w)


def test_compute_combined_score_consistency(df, regimes):
    """단일 진입점 == 캐시 경로(compute_area_scores + combine_scores)."""
    rb, rq = regimes
    phase = pd.Series("Markup", index=df.index, dtype=object)
    direct = A.compute_combined_score(df, rb, rq, phase)
    areas = A.compute_area_scores(df, rb, rq)
    cached = A.combine_scores(areas, None, phase)
    pd.testing.assert_series_equal(direct, cached, check_names=False)


# ---------------------------------------------------------------------------
# 레짐 빌더
# ---------------------------------------------------------------------------

def test_make_regimes_labels(df):
    """레짐 라벨이 허용 집합 안에 있어야 함."""
    stock_data = {"A": df, "B": _make_ohlcv(seed=11)}
    rb, rq = A.make_regimes(stock_data)
    allowed = {"강세지속", "강세약화", "조정", "하락", "unknown"}
    assert set(rb.dropna().unique()) <= allowed
    assert set(rq.dropna().unique()) <= allowed
