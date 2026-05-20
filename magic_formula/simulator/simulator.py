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

청산 규칙 (v3 — 와이코프 익절 도입)
-------------------------------------
- C1    : 종가 < 손절가(진입가 - ATR×1) → 다음날 시가 전량 청산 (손절)
- C_WY  : 와이코프 추세 전환 신호 → 다음날 시가 전량 청산 (익절/추세종료)
            · 신호1: composite_score 가 3일 연속 0 미만
            · 신호2: 진입 점수 대비 4.0 이상 하락 + 현재 점수 음수
- C3    : composite_score ≤ -3 → 다음날 시가 전량 청산 (급락 안전망)
- END   : 백테스트 종료일 미청산 잔량 → 종가 강제 청산

Look-ahead bias 방지
--------------------
- t일 시그널 → t+1일 시가로 체결
- t일 종가로 C1/C_WY/C3 판단 → t+1일 시가로 청산

변경 이력
---------
v3 : C2(ATR 분할 익절) 제거 → C_WY(와이코프 점수 반전 익절) 도입.
     Position 에서 targets·partial_done 제거, entry_score·consec_neg_days 추가.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from magic_formula.signals.rules import (
    check_c1, check_wyckoff_exit, check_c3,
    compute_stop_loss, entry_signals,
)
from magic_formula.signals.adaptive_rule_selector import AdaptiveRuleSelector, RuleSelectionConfig

# ADAPTIVE 동적 분류에 사용할 lookback 기간 (거래일)
ADAPTIVE_LOOKBACK_DAYS = 60

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
    """청산된 거래 1건을 기록한다."""
    ticker:       str
    entry_date:   pd.Timestamp
    exit_date:    pd.Timestamp
    entry_price:  float
    exit_price:   float
    quantity:     int
    gross_pnl:    float          # 수수료·세금 차감 전 손익
    net_pnl:      float          # 수수료·세금 차감 후 손익
    exit_reason:  str            # C1 / C_WY / C3 / END

    @property
    def return_pct(self) -> float:
        """진입 원가 대비 수익률 (%)."""
        cost = self.entry_price * self.quantity
        return (self.net_pnl / cost * 100.0) if cost else 0.0


@dataclass
class Position:
    """열린 포지션 상태를 추적한다 (v3: C_WY 와이코프 익절 지원)."""
    ticker:          str
    entry_date:      pd.Timestamp
    entry_price:     float
    orig_qty:        int            # 최초 매수 수량
    remaining:       int            # 잔여 수량 (C_WY 단계에서 전량 청산)
    stop_loss:       float
    entry_score:     float          # 진입 당일 composite_score (C_WY 신호2 기준)
    consec_neg_days: int = 0        # 보유 중 연속 음수 score 일수 (C_WY 신호1 카운터)


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
# ADAPTIVE 동적 분류 헬퍼
# ---------------------------------------------------------------------------

def _compute_rolling_rule_series(
    scored_df: pd.DataFrame,
    trade_dates: list,
    lookback_days: int = ADAPTIVE_LOOKBACK_DAYS,
) -> pd.Series:
    """
    각 거래일마다 직전 lookback_days 거래일의 composite_score 패턴을 보고
    R1 / R3 / SKIP 중 하나를 동적으로 분류한다.

    Look-ahead bias 방지
    --------------------
    - date 당일 데이터를 포함하지 않음: `scores.index < date` 조건 사용
    - 워밍업 구간(scored_df에 포함된 trade_start 이전 데이터)이 lookback 창으로 활용됨

    R2 분류 처리
    ------------
    AdaptiveRuleSelector가 R2(직진 폭등형)를 반환할 경우 R1으로 통합.
    (R2는 RULES에서 제거됐으므로 R1 진입 신호를 대신 사용)

    Parameters
    ----------
    scored_df   : 전체 기간 DataFrame (워밍업 포함, composite_score 컬럼 필요)
    trade_dates : 실거래 구간 날짜 리스트 (루프 대상)
    lookback_days : 분류에 사용할 직전 거래일 수

    Returns
    -------
    pd.Series — index: date, value: 'R1' / 'R3' / 'SKIP'
    """
    cfg      = RuleSelectionConfig(lookback_days=lookback_days)
    selector = AdaptiveRuleSelector(cfg)
    scores   = scored_df["composite_score"]

    rule_map: dict = {}
    for date in trade_dates:
        past = scores[scores.index < date]          # look-ahead bias 방지
        raw  = selector.select_rule("_", past).selected_rule
        # R2 → R1 통합 (R2는 시뮬레이터에서 지원하지 않음)
        rule_map[date] = "R1" if raw == "R2" else raw

    return pd.Series(rule_map)


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

    청산 우선순위: C1(손절) → C_WY(와이코프) → C3(급락 안전망)

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

    # 청산 예약 (C1/C_WY/C3 → 다음날 시가 청산)
    pending_exit_reason: Optional[str] = None

    # 보유 중 여부 (R2 중복 진입 방지용)
    in_position = pd.Series(False, index=df.index)

    # 전체 진입 신호 사전 계산 (threshold 전달)
    if rule == "ADAPTIVE":
        # 동적 rolling 분류: 진입 신호 당일 직전 40거래일 패턴으로 R1/R3/SKIP 결정
        _rolling_rules = _compute_rolling_rule_series(scored_df, dates)
        _r1_sigs = entry_signals(scored_df, "R1", in_position,
                                 threshold=entry_threshold).reindex(df.index, fill_value=False)
        _r3_sigs = entry_signals(scored_df, "R3", in_position,
                                 threshold=entry_threshold).reindex(df.index, fill_value=False)
        signals = pd.Series(False, index=df.index)
        for _d in dates:
            _assigned = _rolling_rules.get(_d, "SKIP")
            if _assigned == "R1" and _r1_sigs.get(_d, False):
                signals[_d] = True
            elif _assigned == "R3" and _r3_sigs.get(_d, False):
                signals[_d] = True
    else:
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

        # --- 1. 청산 예약 실행 (전날 C1/C_WY/C3 트리거 → 오늘 시가) ---
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
                position = Position(
                    ticker=ticker,
                    entry_date=date,
                    entry_price=exec_price,
                    orig_qty=qty,
                    remaining=qty,
                    stop_loss=compute_stop_loss(exec_price, atr),
                    entry_score=score,   # 진입 당일 composite_score (C_WY 기준점)
                )
            pending_entry = None

        # --- 3. 보유 중인 포지션 처리 ---
        if position is not None:
            # C_WY 카운터: 연속 음수 score 추적 (신호1용)
            if score < 0:
                position.consec_neg_days += 1
            else:
                position.consec_neg_days = 0

            # C1: 종가 손절 판단 (최우선)
            if check_c1(c, position.stop_loss):
                pending_exit_reason = "C1"
            # C_WY: 와이코프 추세 전환 신호 (익절/추세종료)
            elif check_wyckoff_exit(score, position.entry_score, position.consec_neg_days):
                pending_exit_reason = "C_WY"
            # C3: score ≤ -3 급락 안전망 (C_WY 보다 score 가 더 급격히 떨어질 때)
            elif check_c3(score):
                pending_exit_reason = "C3"

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
