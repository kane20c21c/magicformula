"""
metrics/metrics.py
------------------
백테스트 성과 지표를 계산한다.

Primary (의사결정 기준)
-----------------------
- 총 수익률 (Total Return %)
- KOSPI 알파 (전략 수익률 − KOSPI 수익률)
- 거래당 평균 수익률 (Average Trade Return %)
- 승률 (Win Rate %)
- Profit Factor

Secondary (참고용)
------------------
- MDD (Maximum Drawdown %)
- Sharpe Ratio (연환산, 무위험금리 2.5% 가정)
- Sortino Ratio
- Calmar Ratio (CAGR / MDD)
- 평균 보유일
- 청산 사유 분포
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

INITIAL_CAPITAL    = 200_000_000   # 2억원
RISK_FREE_RATE_ANN = 0.025         # 연 2.5%
TRADING_DAYS_YEAR  = 252


# ---------------------------------------------------------------------------
# 내부 유틸
# ---------------------------------------------------------------------------

def _annualized_return(total_return: float, days: int) -> float:
    """총 수익률(소수)과 기간(거래일)으로 연환산 수익률 산출."""
    if days <= 0:
        return 0.0
    years = days / TRADING_DAYS_YEAR
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def _max_drawdown(equity: pd.Series) -> float:
    """
    자산 곡선에서 최대 낙폭(%) 산출.
    equity 는 절대금액(원) 또는 누적 손익 시리즈.
    """
    # 절대 자산으로 변환
    abs_equity = INITIAL_CAPITAL + equity.ffill().fillna(0)
    peak = abs_equity.cummax()
    drawdown = (abs_equity - peak) / peak
    return float(drawdown.min() * 100.0)   # 음수 반환


def _sharpe(daily_returns: pd.Series) -> float:
    """일별 수익률 시리즈로 연환산 Sharpe Ratio 계산."""
    if daily_returns.std() == 0 or len(daily_returns) < 2:
        return 0.0
    rf_daily = (1.0 + RISK_FREE_RATE_ANN) ** (1.0 / TRADING_DAYS_YEAR) - 1.0
    excess   = daily_returns - rf_daily
    return float(excess.mean() / excess.std() * math.sqrt(TRADING_DAYS_YEAR))


def _sortino(daily_returns: pd.Series) -> float:
    """일별 수익률 시리즈로 연환산 Sortino Ratio 계산 (하방 변동성 기준)."""
    if len(daily_returns) < 2:
        return 0.0
    rf_daily  = (1.0 + RISK_FREE_RATE_ANN) ** (1.0 / TRADING_DAYS_YEAR) - 1.0
    excess    = daily_returns - rf_daily
    downside  = excess[excess < 0]
    if downside.empty or downside.std() == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    return float(excess.mean() / downside.std() * math.sqrt(TRADING_DAYS_YEAR))


# ---------------------------------------------------------------------------
# 공개 함수
# ---------------------------------------------------------------------------

def compute_metrics(
    trades_df:   pd.DataFrame,
    equity_df:   pd.DataFrame,
    kospi_df:    Optional[pd.DataFrame],
    trade_start: pd.Timestamp,
    trade_end:   pd.Timestamp,
) -> dict:
    """
    전체 성과 지표를 계산하여 dict로 반환한다.

    Parameters
    ----------
    trades_df   : trades_to_df() 반환값
    equity_df   : run_simulation() 반환 equity_df (total 컬럼 포함)
    kospi_df    : KOSPI OHLCV DataFrame (없으면 알파 계산 생략)
    trade_start : 실거래 시작일
    trade_end   : 백테스트 종료일

    Returns
    -------
    dict — 각 지표 키:값
    """
    result = {}

    # -----------------------------------------------------------------------
    # 기본 정보
    # -----------------------------------------------------------------------
    n_trades = len(trades_df)
    result["n_trades"]    = n_trades
    result["trade_start"] = str(trade_start.date())
    result["trade_end"]   = str(trade_end.date())

    # -----------------------------------------------------------------------
    # 자산 곡선
    # -----------------------------------------------------------------------
    if equity_df.empty or "total" not in equity_df.columns:
        total_equity = pd.Series(0.0)
    else:
        total_equity = equity_df["total"].reindex(
            pd.date_range(trade_start, trade_end, freq="B"), method="ffill"
        ).fillna(0.0)

    # -----------------------------------------------------------------------
    # 총 수익률
    # -----------------------------------------------------------------------
    final_pnl          = float(total_equity.iloc[-1]) if len(total_equity) > 0 else 0.0
    total_return_pct   = final_pnl / INITIAL_CAPITAL * 100.0
    result["total_pnl_krw"]     = round(final_pnl, 0)
    result["total_return_pct"]  = round(total_return_pct, 4)

    # -----------------------------------------------------------------------
    # KOSPI 알파
    # -----------------------------------------------------------------------
    if kospi_df is not None and not kospi_df.empty:
        # 명시적 boolean 필터 — loc 슬라이싱 날짜 타입 불일치 방지
        kdf_all = kospi_df["Close"].dropna()
        mask = (kdf_all.index >= trade_start) & (kdf_all.index <= trade_end)
        kdf  = kdf_all[mask]

        if len(kdf) >= 2:
            k_start_date  = kdf.index[0]
            k_end_date    = kdf.index[-1]
            k_start_price = float(kdf.iloc[0])
            k_end_price   = float(kdf.iloc[-1])
            kospi_ret_pct = (k_end_price - k_start_price) / k_start_price * 100.0
            alpha         = total_return_pct - kospi_ret_pct

            result["kospi_return_pct"]  = round(kospi_ret_pct, 4)
            result["alpha_pct"]         = round(alpha, 4)
            # KOSPI 진단 정보 (리포트 출력용)
            result["kospi_diag"] = {
                "start_date":  str(k_start_date.date()),
                "end_date":    str(k_end_date.date()),
                "start_price": round(k_start_price, 2),
                "end_price":   round(k_end_price, 2),
                "n_days":      int(len(kdf)),
            }

            # 이상치 경고 (±50% 초과는 데이터 오류 가능성)
            if abs(kospi_ret_pct) > 50.0:
                import warnings
                warnings.warn(
                    f"[metrics] KOSPI 수익률 {kospi_ret_pct:+.1f}% — "
                    f"6개월 기준 비현실적 수치. "
                    f"실제 사용 가격: {k_start_price:,.0f}({k_start_date.date()}) → "
                    f"{k_end_price:,.0f}({k_end_date.date()}). "
                    "pykrx 코드 오류(1028=업종지수?) 또는 KOSPI 캐시를 삭제 후 재수집하세요.",
                    stacklevel=2,
                )
        else:
            result["kospi_return_pct"] = None
            result["alpha_pct"]        = None
            result["kospi_diag"]       = None
    else:
        result["kospi_return_pct"] = None
        result["alpha_pct"]        = None
        result["kospi_diag"]       = None

    # -----------------------------------------------------------------------
    # 거래 통계
    # -----------------------------------------------------------------------
    if n_trades == 0:
        result.update({
            "avg_trade_return_pct": 0.0,
            "win_rate_pct":         0.0,
            "profit_factor":        0.0,
            "avg_hold_days":        0.0,
            "exit_reason_dist":     {},
        })
    else:
        returns     = trades_df["return_pct"]
        net_pnls    = trades_df["net_pnl"]

        winners     = net_pnls[net_pnls > 0]
        losers      = net_pnls[net_pnls <= 0]

        win_rate    = len(winners) / n_trades * 100.0
        profit_factor = (
            winners.sum() / abs(losers.sum())
            if losers.sum() != 0 else float("inf")
        )

        # 평균 보유일
        hold_days = (
            (pd.to_datetime(trades_df["exit_date"]) -
             pd.to_datetime(trades_df["entry_date"])).dt.days
        )

        result["avg_trade_return_pct"] = round(float(returns.mean()), 4)
        result["win_rate_pct"]         = round(win_rate, 2)
        result["profit_factor"]        = round(profit_factor, 4)
        result["avg_hold_days"]        = round(float(hold_days.mean()), 1)

        # 청산 사유 분포
        dist = trades_df["exit_reason"].value_counts().to_dict()
        result["exit_reason_dist"] = dist

    # -----------------------------------------------------------------------
    # 일별 수익률 (자산 곡선 기반)
    # -----------------------------------------------------------------------
    abs_equity   = INITIAL_CAPITAL + total_equity
    daily_ret    = abs_equity.pct_change().dropna()

    result["mdd_pct"]      = round(_max_drawdown(total_equity), 4)
    result["sharpe"]       = round(_sharpe(daily_ret), 4)
    result["sortino"]      = round(_sortino(daily_ret), 4)

    # Calmar: CAGR / |MDD|
    trade_days  = len(total_equity)
    cagr        = _annualized_return(total_return_pct / 100.0, trade_days)
    mdd_abs     = abs(result["mdd_pct"]) / 100.0
    result["calmar"] = round(cagr / mdd_abs, 4) if mdd_abs > 0 else 0.0

    return result


def format_report(
    metrics: dict,
    weight_label: str,
    rule: str,
) -> str:
    """
    단일 조합(가중치 + 규칙)의 성과를 Markdown 섹션으로 포맷한다.

    Parameters
    ----------
    metrics      : compute_metrics() 반환값
    weight_label : 가중치 조합 레이블 (예: 'Basic', 'LL_trend_HL_vol')
    rule         : 'R1', 'R2', 'R3'
    """
    alpha_str = (
        f"{metrics.get('alpha_pct', 'N/A'):+.2f}%"
        if metrics.get("alpha_pct") is not None
        else "N/A (KOSPI 없음)"
    )

    # KOSPI 진단 행 생성
    diag = metrics.get("kospi_diag")
    if diag:
        kospi_diag_row = (
            f"| KOSPI 진단 | "
            f"{diag['start_date']} {diag['start_price']:,.0f}pt → "
            f"{diag['end_date']} {diag['end_price']:,.0f}pt "
            f"({diag['n_days']}거래일) |"
        )
        # 이상치 플래그
        kospi_ret = metrics.get("kospi_return_pct", 0) or 0
        if abs(kospi_ret) > 50:
            kospi_diag_row += "\n| ⚠ KOSPI 이상 | 수익률이 ±50% 초과 — 데이터 오류 의심 |"
    else:
        kospi_diag_row = ""

    dist_str = "  \n".join(
        f"  - {k}: {v}건"
        for k, v in sorted(
            metrics.get("exit_reason_dist", {}).items(),
            key=lambda x: -x[1],
        )
    ) or "  - 없음"

    return f"""
### {weight_label} / {rule}

| 지표 | 값 |
|---|---|
| 기간 | {metrics['trade_start']} ~ {metrics['trade_end']} |
| 총 거래 수 | {metrics['n_trades']}건 |
| **총 수익률** | **{metrics['total_return_pct']:+.2f}%** |
| KOSPI 수익률 | {metrics.get('kospi_return_pct') if metrics.get('kospi_return_pct') is not None else 'N/A'} |
| **KOSPI 알파** | **{alpha_str}** |
{kospi_diag_row}
| 거래당 평균 수익 | {metrics['avg_trade_return_pct']:+.4f}% |
| 승률 | {metrics['win_rate_pct']:.1f}% |
| Profit Factor | {metrics['profit_factor']:.4f} |
| MDD | {metrics['mdd_pct']:.2f}% |
| Sharpe | {metrics['sharpe']:.4f} |
| Sortino | {metrics['sortino']:.4f} |
| Calmar | {metrics['calmar']:.4f} |
| 평균 보유일 | {metrics['avg_hold_days']:.1f}일 |

**청산 사유 분포:**
{dist_str}
""".strip()
