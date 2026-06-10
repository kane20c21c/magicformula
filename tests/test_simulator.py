"""
tests/test_simulator.py
-----------------------
v2 시뮬레이터 — 체결/비용 산식, C1/TIME/END 청산, 게이트 NaN 처리.
vault 의존성 없는 합성 데이터 검사.

비용 산식 검증이 핵심: 매수 슬리피지 이중 차감 버그(2026-06-10 수정 전)가
재발하면 test_cost_math_no_double_slippage 가 즉시 깨진다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from magic_formula.simulator.simulator import (
    CAPITAL_PER_TRADE,
    COMMISSION,
    SLIPPAGE_BUY,
    SLIPPAGE_SELL,
    TAX_SELL,
    run_simulation,
    simulate_ticker,
    trades_to_df,
)


# ---------------------------------------------------------------------------
# 합성 데이터 빌더
# ---------------------------------------------------------------------------

def _make_df(
    n: int,
    open_: float = 10_000.0,
    close: float = 10_000.0,
    scores: list | None = None,
    atr: float = 100.0,
) -> pd.DataFrame:
    """단순 합성 OHLCV + composite_score + atr14."""
    idx = pd.bdate_range("2026-01-05", periods=n)
    if scores is None:
        scores = [3.0] + [7.0] * (n - 1)   # 둘째 날 6.0 돌파
    df = pd.DataFrame({
        "Open":  open_, "High": max(open_, close) * 1.01,
        "Low":   min(open_, close) * 0.99, "Close": close,
        "composite_score": scores,
        "atr14": atr,
    }, index=idx)
    return df


def _window(df):
    return df.index[0], df.index[-1]


# ---------------------------------------------------------------------------
# 비용 산식 — 슬리피지 이중 차감 버그 회귀 방지
# ---------------------------------------------------------------------------

def test_cost_math_no_double_slippage():
    """
    Open=Close=10,000 / 신호 d1 → 진입 d2 / 청산 END(d4 종가).

    기대 net_pnl (자본 10M):
      체결가 = 10,000×1.001 = 10,010 → qty = 999
      매수비용 = 10,010×999×(1+0.00015)            = 10,001,489.9985
      매도수령 = 9,990×999×(1−0.00015−0.002)       =  9,958,552.9785
      net = −42,937.02
    (구버전 이중 차감 버그면 −52,937 부근 — 약 1만원 차이로 즉시 검출)
    """
    df = _make_df(5)
    trades, equity = simulate_ticker("TEST", df, *_window(df), entry_threshold=6.0)

    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "END"
    assert t.quantity == 999
    assert t.entry_price == pytest.approx(10_010.0)
    assert t.exit_price == pytest.approx(9_990.0)
    assert t.gross_pnl == pytest.approx((9_990.0 - 10_010.0) * 999)

    exp_cost = 10_010.0 * 999 * (1 + COMMISSION)
    exp_proceeds = 9_990.0 * 999 * (1 - COMMISSION - TAX_SELL)
    assert t.net_pnl == pytest.approx(exp_proceeds - exp_cost, abs=0.01)
    assert t.net_pnl == pytest.approx(-42_937.02, abs=0.1)
    assert equity.iloc[-1] == pytest.approx(t.net_pnl)


def test_quantity_respects_capital():
    """수량 = floor(자본 / 체결가) — 기본 자본 10M 정본."""
    df = _make_df(5, open_=50_000.0, close=50_000.0)
    trades, _ = simulate_ticker("TEST", df, *_window(df), entry_threshold=6.0)
    assert len(trades) == 1
    # 체결가 50,050 → floor(10M/50,050) = 199
    assert trades[0].quantity == int(CAPITAL_PER_TRADE // (50_000.0 * (1 + SLIPPAGE_BUY)))


# ---------------------------------------------------------------------------
# C1 — ATR 손절
# ---------------------------------------------------------------------------

def test_c1_stop_loss_exits_next_open():
    """진입가 10,010 − ATR 100 = 9,910. 종가 9,800 < 9,910 → 다음날 시가 청산."""
    n = 8
    df = _make_df(n)
    # d3 종가 급락 (손절선 9,910 아래)
    df.iloc[3, df.columns.get_loc("Close")] = 9_800.0
    df.iloc[3, df.columns.get_loc("Low")] = 9_700.0

    trades, _ = simulate_ticker("TEST", df, *_window(df), entry_threshold=6.0)
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "C1"
    assert t.entry_date == df.index[2]      # d1 신호 → d2 진입
    assert t.exit_date == df.index[4]       # d3 트리거 → d4 시가 청산


def test_no_exit_when_close_at_stop():
    """종가 == 손절가 → 청산 없음 (미만일 때만)."""
    df = _make_df(6)
    df.iloc[3, df.columns.get_loc("Close")] = 9_910.0   # 정확히 손절선
    trades, _ = simulate_ticker("TEST", df, *_window(df), entry_threshold=6.0)
    assert len(trades) == 1
    assert trades[0].exit_reason == "END"


# ---------------------------------------------------------------------------
# TIME — 시간 청산 (20거래일 + 손익≤0 / 이익 중 유지)
# ---------------------------------------------------------------------------

def test_time_stop_exits_underwater_position():
    """종가 10,000 < 진입가 10,010 (만년 약손실) → 보유 20일째 트리거, 21일째 청산."""
    df = _make_df(30)
    trades, _ = simulate_ticker("TEST", df, *_window(df), entry_threshold=6.0)
    assert len(trades) >= 1
    t = trades[0]
    assert t.exit_reason == "TIME"
    assert t.entry_date == df.index[2]
    # 진입 d2(보유 1일째) → 20일째 = d21 트리거 → d22 시가 청산
    assert t.exit_date == df.index[22]


def test_time_stop_holds_profitable_position():
    """종가 10,100 > 진입가 10,010 (이익 중) → 시간청산 없이 END 까지 보유."""
    df = _make_df(30, close=10_100.0)
    trades, _ = simulate_ticker("TEST", df, *_window(df), entry_threshold=6.0)
    assert len(trades) == 1
    assert trades[0].exit_reason == "END"
    assert trades[0].exit_date == df.index[-1]


# ---------------------------------------------------------------------------
# 게이트(NaN) / 데이터 결함 처리
# ---------------------------------------------------------------------------

def test_gated_nan_scores_never_enter():
    """composite 가 전부 NaN(게이트 제외) → 거래 없음."""
    df = _make_df(10, scores=[np.nan] * 10)
    trades, equity = simulate_ticker("TEST", df, *_window(df), entry_threshold=6.0)
    assert trades == []
    assert (equity == 0.0).all()


def test_entry_skipped_when_atr_nan():
    """진입 예정일 ATR NaN → 손절가 계산 불가 → 진입 스킵."""
    df = _make_df(6, atr=np.nan)
    trades, _ = simulate_ticker("TEST", df, *_window(df), entry_threshold=6.0)
    assert trades == []


# ---------------------------------------------------------------------------
# 멀티 종목 / 변환
# ---------------------------------------------------------------------------

def test_run_simulation_total_column():
    data = {"A": _make_df(6), "B": _make_df(6)}
    trades, eq = run_simulation(data, data["A"].index[0], data["A"].index[-1],
                                entry_threshold=6.0)
    assert "total" in eq.columns
    assert eq["total"].iloc[-1] == pytest.approx(eq["A"].iloc[-1] + eq["B"].iloc[-1])
    tdf = trades_to_df(trades)
    assert set(tdf["exit_reason"]) == {"END"}
    assert len(tdf) == 2
