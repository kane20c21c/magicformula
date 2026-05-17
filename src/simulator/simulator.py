"""
simulator/simulator.py
----------------------
단일 종목에 대해 시그널 → 체결 → 청산 흐름을 일별로 시뮬레이션한다.

거래 비용 가정
--------------
- 슬리피지 : 매수 +0.10%, 매도 -0.10%
- 수수료   : 매수·매도 각 0.015%
- 거래세   : 매도 0.20%

자본 관리
---------
- 종목당 투입 자본: CAPITAL_PER_TRADE = 20_000_000 원
- 수량 = floor(CAPITAL_PER_TRADE / 체결가)

분할 청산 (C2)
--------------
- 1차: 원래 수량의 30% → target1(ATR×1) 달성 시 당일 target1 가격에 체결
- 2차: 원래 수량의 40% → target2(ATR×2) 달성 시 당일 target2 가격에 체결
- 3차: 잔여(30%)      → target3(ATR×3) 달성 시 당일 target3 가격에 체결

C1/C3/C4 는 조건 충족 다음날 시가에 잔량 전량 청산.
백테스트 종료일에 미청산 잔량은 종가로 강제 청산.

Look-ahead bias 방지
--------------------
- t일 시그널 → t+1일 시가로 체결
- t일 종가로 C1/C3/C4 판단 → t+1일 시가로 청산
- C2는 t일 고가가 목표가 이상이면 t일에 해당 목표가로 체결
  (일중 최고가 기준: 약간 낙관적이나 일별 백테스트 표준 가정)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from signals.rules import (
    PARTIAL_RATIOS,
    check_c1, check_c2, check_c3, check_c4,
    compute_exit_prices, entry_signals,
)

# ---------------------------------------------------------------------------
# 거래 비용 상수
# ---------------------------------------------------------------------------

CAPITAL_PER_TRADE = 20_000_000   # 원
SLIPPAGE_BUY      = 0.001        # 0.10%
SLIPPAGE_SELL     = 0.001        # 0.10%
COMMISSION        = 0.00015      # 0.015%
TAX_SELL          = 0.002        # 0.20%


# ---------------------------------------------------------------------------
# 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """청산된 부분 거래 1건을 기록한다."""
    ticker:       str
    entry_date:   pd.Timestamp
    exit_date:    pd.Timestamp
    entry_price:  float
    exit_price:   float
    quantity:     int
    gross_pnl:    float          # 수수료·세금 차감 전 손익
    net_pnl:      float          # 수수료·세금 차감 후 손익
    exit_reason:  str            # C1 / C2_1 / C2_2 / C2_3 / C3 / C4 / END

    @property
    def return_pct(self) -> float:
        """진입 원가 대비 수익률 (%)."""
        cost = self.entry_price * self.quantity
        return (self.net_pnl / cost * 100.0) if cost else 0.0


@dataclass
class Position:
    """열린 포지션 상태를 추적한다."""
    ticker:       str
    entry_date:   pd.Timestamp
    entry_price:  float
    orig_qty:     int            # 최초 매수 수량
    remaining:    int            # 잔여 수량
    stop_loss:    float
    targets:      list[float]    # [t1, t2, t3]
    partial_done: list[bool] = field(default_factory=lambda: [False, False, False])
    days_held:    int = 0

    # 분할 청산 비율별 수량
    @property
    def qty_each(self) -> list[int]:
        """[30%, 40%, 30%] 각 단계의 수량 (C2-1, C2-2 는 floor, C2-3 은 잔량)."""
        q1 = math.floor(self.orig_qty * PARTIAL_RATIOS[0])
        q2 = math.floor(self.orig_qty * PARTIAL_RATIOS[1])
        q3 = self.orig_qty - q1 - q2   # 잔량 전부
        return [q1, q2, q3]

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """미실현 손익률 (%)."""
        return (current_price - self.entry_price) / self.entry_price * 100.0


# ---------------------------------------------------------------------------
# 비용 계산 유틸
# ---------------------------------------------------------------------------

def _buy_cost(price: float, qty: int) -> float:
    """매수 체결가 포함 총 비용 (슬리피지 + 수수료 반영)."""
    exec_price = price * (1.0 + SLIPPAGE_BUY)
    commission = exec_price * qty * COMMISSION
    return exec_price * qty + commission


def _sell_proceeds(price: float, qty: int) -> float:
    """매도 실수령액 (슬리피지 + 수수료 + 거래세 차감)."""
    exec_price = price * (1.0 - SLIPPAGE_SELL)
    commission = exec_price * qty * COMMISSION
    tax        = exec_price * qty * TAX_SELL
    return exec_price * qty - commission - tax


def _exec_buy_price(open_price: float) -> float:
    """매수 체결가 (슬리피지 적용)."""
    return open_price * (1.0 + SLIPPAGE_BUY)


def _exec_sell_price(price: float) -> float:
    """매도 체결가 (슬리피지 적용)."""
    return price * (1.0 - SLIPPAGE_SELL)


def _net_pnl(entry_price_total: float, sell_proceeds: float) -> float:
    return sell_proceeds - entry_price_total


# ---------------------------------------------------------------------------
# 단일 종목 시뮬레이터
# ---------------------------------------------------------------------------

def simulate_ticker(
    ticker:          str,
    scored_df:       pd.DataFrame,
    rule:            str,
    trade_start:     pd.Timestamp,
    trade_end:       pd.Timestamp,
    entry_threshold: float = 5.0,
) -> tuple[list[TradeRecord], pd.Series]:
    """
    단일 종목에 대한 전체 백테스트를 수행한다.

    Parameters
    ----------
    ticker           : 종목 코드
    scored_df        : compute_scores() 가 반환한 DataFrame
                       (Open/High/Low/Close/Volume + area scores + composite_score + atr14)
    rule             : 'R1', 'R2', 'R3'
    trade_start      : 실거래 시작일 (이전은 워밍업 구간 — 신호 생성 안 함)
    trade_end        : 백테스트 종료일
    entry_threshold  : R1·R2 진입 임계값 (기본 5.0). R3 는 항상 0 기준.

    Returns
    -------
    (trades, equity_series)
    trades        : 완료된 거래 목록
    equity_series : 날짜별 누적 순손익 (0에서 시작)
    """
    trades: list[TradeRecord] = []
    position: Optional[Position] = None

    # 실거래 구간 데이터만
    df = scored_df.loc[trade_start:trade_end].copy()
    if df.empty:
        return trades, pd.Series(dtype=float)

    # 날짜 리스트
    dates = df.index.tolist()
    n = len(dates)

    cumulative_pnl = 0.0
    equity = pd.Series(0.0, index=df.index)

    # 진입 예약 (t일 시그널 → t+1일 체결)
    pending_entry: Optional[pd.Timestamp] = None

    # 청산 예약 (C1/C3/C4 → 다음날 시가 청산)
    pending_exit_reason: Optional[str] = None

    # 보유 중 여부 (R2 중복 진입 방지용)
    in_position = pd.Series(False, index=df.index)

    # 전체 진입 신호 사전 계산 (threshold 전달)
    all_signals = entry_signals(scored_df, rule, in_position, threshold=entry_threshold)
    # 실거래 구간으로 필터
    signals = all_signals.reindex(df.index, fill_value=False)

    for i, date in enumerate(dates):
        row    = df.loc[date]
        o      = row["Open"]
        h      = row["High"]
        lo_    = row["Low"]
        c      = row["Close"]
        score  = row["composite_score"]
        atr    = row.get("atr14", np.nan)

        # --- 1. 청산 예약 실행 (전날 C1/C3/C4 트리거 → 오늘 시가) ---
        if pending_exit_reason and position is not None:
            qty_to_sell = position.remaining
            if qty_to_sell > 0:
                # 진입 원가 비례 배분
                cost_basis = position.entry_price * qty_to_sell
                proceeds   = _sell_proceeds(o, qty_to_sell)
                net        = proceeds - cost_basis * (1.0 + COMMISSION + SLIPPAGE_BUY)
                gross      = (o * (1.0 - SLIPPAGE_SELL) - position.entry_price) * qty_to_sell

                trades.append(TradeRecord(
                    ticker=ticker,
                    entry_date=position.entry_date,
                    exit_date=date,
                    entry_price=position.entry_price,
                    exit_price=_exec_sell_price(o),
                    quantity=qty_to_sell,
                    gross_pnl=gross,
                    net_pnl=_net_pnl(
                        cost_basis * (1.0 + SLIPPAGE_BUY + COMMISSION),
                        proceeds,
                    ),
                    exit_reason=pending_exit_reason,
                ))
                cumulative_pnl += trades[-1].net_pnl
            position            = None
            pending_exit_reason = None
            pending_entry       = None   # 청산 직후 진입 예약도 취소

        # --- 2. 진입 예약 실행 (전날 신호 → 오늘 시가) ---
        if pending_entry and position is None:
            exec_price = _exec_buy_price(o)
            qty = math.floor(CAPITAL_PER_TRADE / exec_price)
            if qty > 0 and not np.isnan(atr) and atr > 0:
                exit_p = compute_exit_prices(exec_price, atr)
                position = Position(
                    ticker=ticker,
                    entry_date=date,
                    entry_price=exec_price,
                    orig_qty=qty,
                    remaining=qty,
                    stop_loss=exit_p["stop_loss"],
                    targets=[exit_p["target1"], exit_p["target2"], exit_p["target3"]],
                )
            pending_entry = None

        # --- 3. 보유 중인 포지션 처리 ---
        if position is not None:
            position.days_held += 1

            # C2: 분할 익절 (당일 고가 기준, 당일 목표가로 체결)
            triggered_stages = check_c2(h, position.targets, position.partial_done)
            qty_each = position.qty_each

            for stage in triggered_stages:
                tgt_price = position.targets[stage]
                qty_stage = qty_each[stage]
                # 마지막 단계는 잔량 전부
                if stage == 2:
                    qty_stage = position.remaining
                qty_stage = min(qty_stage, position.remaining)
                if qty_stage <= 0:
                    continue

                cost_basis = position.entry_price * qty_stage
                proceeds   = _sell_proceeds(tgt_price, qty_stage)
                gross      = (_exec_sell_price(tgt_price) - position.entry_price) * qty_stage

                trades.append(TradeRecord(
                    ticker=ticker,
                    entry_date=position.entry_date,
                    exit_date=date,
                    entry_price=position.entry_price,
                    exit_price=_exec_sell_price(tgt_price),
                    quantity=qty_stage,
                    gross_pnl=gross,
                    net_pnl=_net_pnl(
                        cost_basis * (1.0 + SLIPPAGE_BUY + COMMISSION),
                        proceeds,
                    ),
                    exit_reason=f"C2_{stage+1}",
                ))
                cumulative_pnl += trades[-1].net_pnl
                position.remaining   -= qty_stage
                position.partial_done[stage] = True

            # 잔량 소진 확인
            if position.remaining <= 0:
                position = None
            else:
                # C1: 종가 손절 판단
                if check_c1(c, position.stop_loss):
                    pending_exit_reason = "C1"
                # C3: 점수 반전 판단
                elif check_c3(score):
                    pending_exit_reason = "C3"
                # C4: 시간 청산 판단
                elif check_c4(position.days_held, position.unrealized_pnl_pct(c)):
                    pending_exit_reason = "C4"

        # --- 4. 신호 확인 → 다음날 진입 예약 ---
        if position is None and pending_exit_reason is None and signals.get(date, False):
            pending_entry = date   # 내일 시가로 진입 예약

        # 보유 여부 업데이트 (R2 중복 방지용)
        in_position[date] = position is not None

        # 자산 곡선 갱신
        equity.loc[date] = cumulative_pnl

    # --- 5. 백테스트 종료: 미청산 잔량 강제 청산 ---
    if position is not None and position.remaining > 0:
        last_date  = dates[-1]
        last_close = df.loc[last_date, "Close"]
        qty        = position.remaining
        cost_basis = position.entry_price * qty
        proceeds   = _sell_proceeds(last_close, qty)
        gross      = (_exec_sell_price(last_close) - position.entry_price) * qty

        trades.append(TradeRecord(
            ticker=ticker,
            entry_date=position.entry_date,
            exit_date=last_date,
            entry_price=position.entry_price,
            exit_price=_exec_sell_price(last_close),
            quantity=qty,
            gross_pnl=gross,
            net_pnl=_net_pnl(
                cost_basis * (1.0 + SLIPPAGE_BUY + COMMISSION),
                proceeds,
            ),
            exit_reason="END",
        ))
        cumulative_pnl += trades[-1].net_pnl
        equity.loc[last_date] = cumulative_pnl

    return trades, equity


# ---------------------------------------------------------------------------
# 멀티 종목 시뮬레이터
# ---------------------------------------------------------------------------

def run_simulation(
    scored_data:     dict[str, pd.DataFrame],
    rule:            str,
    trade_start:     pd.Timestamp,
    trade_end:       pd.Timestamp,
    entry_threshold: float = 5.0,
) -> tuple[list[TradeRecord], pd.DataFrame]:
    """
    10개 종목에 대해 병렬(순차)로 시뮬레이션을 실행한다.

    Parameters
    ----------
    scored_data      : {ticker: scored_df}
    rule             : 'R1', 'R2', 'R3'
    trade_start      : 실거래 시작일
    trade_end        : 백테스트 종료일
    entry_threshold  : R1·R2 진입 임계값 (기본 5.0)

    Returns
    -------
    (all_trades, equity_df)
    all_trades : 전체 거래 내역 리스트
    equity_df  : 종목별 누적 손익 컬럼을 가진 DataFrame (인덱스=날짜)
    """
    all_trades: list[TradeRecord] = []
    equity_dict: dict[str, pd.Series] = {}

    for ticker, df in scored_data.items():
        try:
            trades, eq = simulate_ticker(
                ticker, df, rule, trade_start, trade_end,
                entry_threshold=entry_threshold,
            )
            all_trades.extend(trades)
            equity_dict[ticker] = eq
        except Exception as exc:
            print(f"  [simulator] {ticker} 시뮬레이션 오류: {exc} — 건너뜀")

    equity_df = pd.DataFrame(equity_dict)
    equity_df["total"] = equity_df.sum(axis=1)
    return all_trades, equity_df


def trades_to_df(trades: list[TradeRecord]) -> pd.DataFrame:
    """TradeRecord 리스트를 DataFrame으로 변환."""
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
