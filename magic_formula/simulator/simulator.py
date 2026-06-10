"""
simulator/simulator.py
----------------------
단일 종목에 대해 시그널 → 체결 → 청산 흐름을 일별로 시뮬레이션한다.
(v2_combined — configs/active_strategy.yaml `trading:` 섹션과 1:1 일치)

거래 비용 가정
--------------
- 슬리피지 : 매수 +0.10%, 매도 -0.10%
- 수수료   : 매수·매도 각 0.015%
- 거래세   : 매도 0.20%

자본 관리
---------
- 종목당 투입 자본: capital_per_trade (기본 CAPITAL_PER_TRADE = 10,000,000 원
  — yaml trading.entry.position_size 정본)
- 수량 = floor(capital_per_trade / 체결가)
- 종목당 1포지션 (one_position_per_ticker)

청산 규칙 (v2)
--------------
- C1   : 종가 < 손절가(진입가 − ATR(14)×1)              → 다음날 시가 전량 청산
- TIME : 보유 20거래일 이상 + 누적손익 ≤ 0%             → 다음날 시가 전량 청산
         (누적손익 > 0% 면 유지 — hold_if_profit)
- END  : 평가종료일 미청산 잔량 → 종가 강제 청산

Look-ahead bias 방지
--------------------
- t일 시그널 → t+1일 시가로 체결
- t일 종가로 C1/TIME 판단 → t+1일 시가로 청산

변경 이력
---------
2026-06-10 v2 단일화:
- v1 규칙(R1/R2/R3/ADAPTIVE 분기, C_WY, C3) 삭제 — 진입은 threshold_breakout 단일.
- 매수 슬리피지 이중 차감 버그 수정 (entry_price 에 이미 포함된 슬리피지를
  net_pnl 계산에서 한 번 더 곱하던 결함 — 거래당 ~0.1%p 비용 과대 제거).
- yaml time_stop(20거래일 + 손익≤0) 구현. CAPITAL 20M → 10M (yaml 정본).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from magic_formula.signals.rules import (
    check_stop_loss, check_time_stop, compute_stop_loss, entry_signals,
)

# ---------------------------------------------------------------------------
# 거래 비용 상수
# ---------------------------------------------------------------------------

CAPITAL_PER_TRADE = 10_000_000   # 원 — yaml trading.entry.position_size 정본
SLIPPAGE_BUY      = 0.001        # 0.10%
SLIPPAGE_SELL     = 0.001        # 0.10%
COMMISSION        = 0.00015      # 0.015%
TAX_SELL          = 0.002        # 0.20%


# ---------------------------------------------------------------------------
# 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """청산된 거래 1건을 기록한다."""
    ticker:       str
    entry_date:   pd.Timestamp
    exit_date:    pd.Timestamp
    entry_price:  float          # 매수 체결가 (슬리피지 포함)
    exit_price:   float          # 매도 체결가 (슬리피지 포함)
    quantity:     int
    gross_pnl:    float          # (매도체결가 − 매수체결가) × 수량
    net_pnl:      float          # 수수료·거래세 차감 후 손익
    exit_reason:  str            # C1 / TIME / END

    @property
    def return_pct(self) -> float:
        """진입 원가(체결가 기준) 대비 순수익률 (%)."""
        cost = self.entry_price * self.quantity
        return (self.net_pnl / cost * 100.0) if cost else 0.0


@dataclass
class Position:
    """열린 포지션 상태를 추적한다."""
    ticker:      str
    entry_date:  pd.Timestamp
    entry_price: float           # 매수 체결가 (슬리피지 포함)
    quantity:    int
    stop_loss:   float
    hold_days:   int = 0         # 보유 거래일 수 (체결일 = 1)


# ---------------------------------------------------------------------------
# 비용 계산 유틸
# ---------------------------------------------------------------------------

def _exec_buy_price(open_price: float) -> float:
    """매수 체결가 (슬리피지 적용)."""
    return open_price * (1.0 + SLIPPAGE_BUY)


def _exec_sell_price(price: float) -> float:
    """매도 체결가 (슬리피지 적용)."""
    return price * (1.0 - SLIPPAGE_SELL)


def buy_cost(exec_price: float, qty: int) -> float:
    """매수 총비용 = 체결가×수량 + 수수료. (체결가에 슬리피지 기반영)"""
    notional = exec_price * qty
    return notional + notional * COMMISSION


def sell_proceeds(exec_price: float, qty: int) -> float:
    """매도 실수령액 = 체결가×수량 − 수수료 − 거래세. (체결가에 슬리피지 기반영)"""
    notional = exec_price * qty
    return notional - notional * COMMISSION - notional * TAX_SELL


# ---------------------------------------------------------------------------
# 내부 — 청산 1건 기록
# ---------------------------------------------------------------------------

def _close_position(
    position: Position,
    exit_date: pd.Timestamp,
    raw_price: float,
    reason: str,
) -> TradeRecord:
    """포지션 전량 청산 → TradeRecord 생성."""
    exec_sell = _exec_sell_price(raw_price)
    qty = position.quantity
    total_cost = buy_cost(position.entry_price, qty)
    proceeds = sell_proceeds(exec_sell, qty)
    return TradeRecord(
        ticker=position.ticker,
        entry_date=position.entry_date,
        exit_date=exit_date,
        entry_price=position.entry_price,
        exit_price=exec_sell,
        quantity=qty,
        gross_pnl=(exec_sell - position.entry_price) * qty,
        net_pnl=proceeds - total_cost,
        exit_reason=reason,
    )


# ---------------------------------------------------------------------------
# 단일 종목 시뮬레이터
# ---------------------------------------------------------------------------

def simulate_ticker(
    ticker:            str,
    scored_df:         pd.DataFrame,
    trade_start:       pd.Timestamp,
    trade_end:         pd.Timestamp,
    entry_threshold:   float = 6.0,
    capital_per_trade: float = CAPITAL_PER_TRADE,
) -> tuple[list[TradeRecord], pd.Series]:
    """
    단일 종목 백테스트 (진입 threshold_breakout / 청산 C1·TIME·END).

    Parameters
    ----------
    ticker            : 종목 코드
    scored_df         : Open/High/Low/Close + composite_score + atr14 컬럼 DataFrame
                        (composite_score 는 게이트 제외 구간 NaN 허용)
    trade_start       : 실거래 시작일 (이전은 워밍업 — 신호 생성 안 함)
    trade_end         : 백테스트 종료일
    entry_threshold   : 진입 임계값 (yaml scoring.threshold)
    capital_per_trade : 종목당 투입 자본 (yaml trading.entry.position_size)

    Returns
    -------
    (trades, equity_series)
    trades        : 완료된 거래 목록
    equity_series : 날짜별 누적 순손익 (0에서 시작)
    """
    trades: list[TradeRecord] = []
    position: Optional[Position] = None

    df = scored_df.loc[trade_start:trade_end]
    if df.empty:
        return trades, pd.Series(dtype=float)

    dates = df.index.tolist()
    cumulative_pnl = 0.0
    equity = pd.Series(0.0, index=df.index)

    pending_entry = False                          # t일 신호 → t+1일 시가 체결
    pending_exit_reason: Optional[str] = None      # t일 트리거 → t+1일 시가 청산

    # 전체 진입 신호 사전 계산 (워밍업 포함 전체 → 실거래 구간 필터)
    signals = entry_signals(scored_df, threshold=entry_threshold)
    signals = signals.reindex(df.index, fill_value=False)

    for date in dates:
        row = df.loc[date]
        o = row["Open"]
        c = row["Close"]
        atr = row.get("atr14", np.nan)

        # --- 1. 청산 예약 실행 (전날 C1/TIME 트리거 → 오늘 시가) ---
        if pending_exit_reason and position is not None:
            trade = _close_position(position, date, o, pending_exit_reason)
            trades.append(trade)
            cumulative_pnl += trade.net_pnl
            position = None
            pending_exit_reason = None
            pending_entry = False   # 청산 직후 진입 예약도 취소

        # --- 2. 진입 예약 실행 (전날 신호 → 오늘 시가) ---
        if pending_entry and position is None:
            exec_price = _exec_buy_price(o)
            qty = math.floor(capital_per_trade / exec_price)
            if qty > 0 and not np.isnan(atr) and atr > 0:
                position = Position(
                    ticker=ticker,
                    entry_date=date,
                    entry_price=exec_price,
                    quantity=qty,
                    stop_loss=compute_stop_loss(exec_price, atr),
                    hold_days=0,   # 아래 3단계에서 당일분 +1
                )
            pending_entry = False

        # --- 3. 보유 중인 포지션 처리 (당일 종가 기준 판단) ---
        if position is not None:
            position.hold_days += 1
            cum_ret_pct = (c - position.entry_price) / position.entry_price * 100.0

            # C1: 종가 손절 (최우선)
            if check_stop_loss(c, position.stop_loss):
                pending_exit_reason = "C1"
            # TIME: 20거래일 + 누적손익 ≤ 0% (이익 중이면 유지)
            elif check_time_stop(position.hold_days, cum_ret_pct):
                pending_exit_reason = "TIME"

        # --- 4. 신호 확인 → 다음날 진입 예약 ---
        if position is None and pending_exit_reason is None and signals.get(date, False):
            pending_entry = True

        equity.loc[date] = cumulative_pnl

    # --- 5. 백테스트 종료: 미청산 잔량 강제 청산 (종가) ---
    if position is not None:
        last_date = dates[-1]
        last_close = df.loc[last_date, "Close"]
        trade = _close_position(position, last_date, last_close, "END")
        trades.append(trade)
        cumulative_pnl += trade.net_pnl
        equity.loc[last_date] = cumulative_pnl

    return trades, equity


# ---------------------------------------------------------------------------
# 멀티 종목 시뮬레이터
# ---------------------------------------------------------------------------

def run_simulation(
    scored_data:       dict[str, pd.DataFrame],
    trade_start:       pd.Timestamp,
    trade_end:         pd.Timestamp,
    entry_threshold:   float = 6.0,
    capital_per_trade: float = CAPITAL_PER_TRADE,
) -> tuple[list[TradeRecord], pd.DataFrame]:
    """
    전 종목 순차 시뮬레이션.

    Returns
    -------
    (all_trades, equity_df)
    all_trades : 전체 거래 내역 리스트
    equity_df  : 종목별 누적 손익 컬럼 + 'total' 합계 (인덱스=날짜)
    """
    all_trades: list[TradeRecord] = []
    equity_dict: dict[str, pd.Series] = {}

    for ticker, df in scored_data.items():
        try:
            trades, eq = simulate_ticker(
                ticker, df, trade_start, trade_end,
                entry_threshold=entry_threshold,
                capital_per_trade=capital_per_trade,
            )
            all_trades.extend(trades)
            equity_dict[ticker] = eq
        except Exception as exc:
            print(f"  [simulator] {ticker} 시뮬레이션 오류: {exc} — 건너뜀")

    equity_df = pd.DataFrame(equity_dict)
    if not equity_df.empty:
        equity_df = equity_df.ffill().fillna(0.0)
        equity_df["total"] = equity_df.sum(axis=1)
    return all_trades, equity_df


def trades_to_df(trades: list[TradeRecord]) -> pd.DataFrame:
    """TradeRecord 리스트를 DataFrame 으로 변환."""
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([
        {
            "ticker":       t.ticker,
            "entry_date":   t.entry_date,
            "exit_date":    t.exit_date,
            "entry_price":  round(t.entry_price, 2),
            "exit_price":   round(t.exit_price, 2),
            "quantity":     t.quantity,
            "gross_pnl":    round(t.gross_pnl, 0),
            "net_pnl":      round(t.net_pnl, 0),
            "return_pct":   round(t.return_pct, 4),
            "exit_reason":  t.exit_reason,
        }
        for t in trades
    ])
