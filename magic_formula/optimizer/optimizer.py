"""
optimizer/optimizer.py
----------------------
v2_combined 가중치 그리드 백테스트 실행기.

M4 분석(2026-05-30, docs/area_specs/combined.md) 방법론의 repo 내 구현:
4영역(T/M/Vu/Va) 가중치 그리드 × 임계값 조합을 전 종목에 대해 시뮬레이션하고
robust(상위 5건 제외) 지표 기준으로 랭킹을 만든다.

핵심 최적화
-----------
영역 점수는 가중치와 무관하므로 **종목당 1회만** 계산해 캐시하고,
조합 루프에서는 가중 결합(combine_scores)만 반복한다.
→ 56조합 × 2임계값 = 112회 실행이 영역 점수 1회 계산 비용에 수렴.

변경 이력
---------
2026-06-10 v2 단일화: v1(5영역 scorer 가중평균 + R1/R3/ADAPTIVE) 그리드 폐기.
"""

from __future__ import annotations

import sys
import traceback
from itertools import product
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 경로 설정 — Magic Formula 루트를 sys.path 에 추가
# ---------------------------------------------------------------------------

_PROJ_ROOT = Path(__file__).parent.parent.parent          # Magic Formula/
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from magic_formula.analysis import area_scores as A                      # noqa: E402
from magic_formula.data.collector import get_backtest_split             # noqa: E402
from magic_formula.indicators import _atr                                # noqa: E402
from magic_formula.metrics.metrics import compute_metrics                # noqa: E402
from magic_formula.simulator.simulator import (                          # noqa: E402
    CAPITAL_PER_TRADE, run_simulation, trades_to_df,
)

AREA_KEYS = A.AREA_KEYS   # ("trend", "momentum", "volume", "volatility")


# ---------------------------------------------------------------------------
# 가중치 그리드
# ---------------------------------------------------------------------------

def weights_label(weights: dict[str, float]) -> str:
    """{trend:0.2,...} → 'T20/M20/Vu0/Va60' (update_strategy 가 파싱하는 정형)."""
    return (f"T{round(weights['trend']*100)}"
            f"/M{round(weights['momentum']*100)}"
            f"/Vu{round(weights['volume']*100)}"
            f"/Va{round(weights['volatility']*100)}")


def generate_weight_grid(step: float = 0.2) -> list[dict]:
    """
    합이 1.0 인 4영역 가중치 조합 전수 생성.

    step=0.2 → 56조합 (M4 그리드와 동일 해상도),
    step=0.1 → 286조합 (정밀 탐색용).

    Returns
    -------
    list of {"label": "T20/M20/Vu0/Va60", "weights": {...}}
    """
    n = round(1.0 / step)
    combos = []
    for t, m, vu in product(range(n + 1), repeat=3):
        va = n - t - m - vu
        if va < 0:
            continue
        w = {
            "trend":      t * step,
            "momentum":   m * step,
            "volume":     vu * step,
            "volatility": va * step,
        }
        combos.append({"label": weights_label(w), "weights": w})
    return combos


# ---------------------------------------------------------------------------
# 전체 백테스트 실행기
# ---------------------------------------------------------------------------

def _trade_window(stock_data: dict[str, pd.DataFrame],
                  warmup_months: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """전 종목 공통 실거래 구간 (시작 = max, 끝 = min)."""
    trade_start = trade_end = None
    for df in stock_data.values():
        _, t_idx = get_backtest_split(df, warmup_months)
        if len(t_idx) == 0:
            continue
        ts, te = t_idx[0], t_idx[-1]
        trade_start = ts if trade_start is None else max(trade_start, ts)
        trade_end   = te if trade_end   is None else min(trade_end,   te)
    if trade_start is None:
        raise ValueError("실거래 구간이 없습니다. 데이터 기간을 확인하세요.")
    return trade_start, trade_end


def prepare_scoring_inputs(
    stock_data: dict[str, pd.DataFrame],
) -> tuple[dict, dict, dict, pd.Series, pd.Series]:
    """
    조합 루프 전에 종목당 1회만 계산하는 입력들.

    Returns
    -------
    (areas_by_ticker, atr_by_ticker, phase_by_ticker, regime_b, regime_q)
    """
    regime_b, regime_q = A.make_regimes(stock_data)
    areas_by_ticker: dict[str, dict] = {}
    atr_by_ticker:   dict[str, pd.Series] = {}
    phase_by_ticker: dict[str, pd.Series] = {}
    for t, df in stock_data.items():
        areas_by_ticker[t] = A.compute_area_scores(df, regime_b, regime_q)
        atr_by_ticker[t]   = _atr(df["High"], df["Low"], df["Close"])
        phase_by_ticker[t] = (df["Wyckoff_Label"] if "Wyckoff_Label" in df.columns
                              else pd.Series(index=df.index, dtype=object))
    return areas_by_ticker, atr_by_ticker, phase_by_ticker, regime_b, regime_q


def build_scored_data(
    stock_data:      dict[str, pd.DataFrame],
    areas_by_ticker: dict,
    atr_by_ticker:   dict,
    phase_by_ticker: dict,
    weights:         dict[str, float],
    gate:            bool = True,
    exclude_phases:  tuple[str, ...] = A.GATE_EXCLUDE_PHASES,
) -> dict[str, pd.DataFrame]:
    """캐시된 영역 점수 → 조합별 scored_data (simulator 입력)."""
    scored: dict[str, pd.DataFrame] = {}
    for t, df in stock_data.items():
        comp = A.combine_scores(
            areas_by_ticker[t], weights, phase_by_ticker[t],
            gate=gate, exclude_phases=exclude_phases,
        )
        sdf = df[["Open", "High", "Low", "Close"]].copy()
        sdf["composite_score"] = comp
        sdf["atr14"] = atr_by_ticker[t]
        scored[t] = sdf
    return scored


def run_all(
    stock_data:        dict[str, pd.DataFrame],
    kospi_df:          pd.DataFrame | None,
    thresholds:        tuple[float, ...] = (5.0, 6.0),
    step:              float = 0.2,
    warmup_months:     int = 12,
    gate:              bool = True,
    exclude_phases:    tuple[str, ...] = A.GATE_EXCLUDE_PHASES,
    capital_per_trade: float = CAPITAL_PER_TRADE,
    weights_list:      list[dict] | None = None,
    verbose:           bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    가중치 그리드 × 임계값 전체 백테스트.

    Parameters
    ----------
    stock_data    : collect_all() 반환값에서 KOSPI 를 뺀 종목 dict (full-column)
    kospi_df      : KOSPI OHLCV (알파 계산용, 없으면 None)
    thresholds    : 진입 임계값 후보들
    step          : 가중치 그리드 간격 (0.2 → 56조합)
    weights_list  : 명시하면 그리드 대신 이 조합들만 실행
                    (예: 현 운영 가중치 단일 검증)

    Returns
    -------
    (all_trades_df, summary_df)  — summary 는 robust 평균수익 기준 정렬
    """
    trade_start, trade_end = _trade_window(stock_data, warmup_months)
    combos = weights_list if weights_list is not None else generate_weight_grid(step)

    if verbose:
        print(f"\n실거래 구간: {trade_start.date()} ~ {trade_end.date()}")
        print(f"{len(combos)}조합 × {len(thresholds)}임계값 = "
              f"{len(combos)*len(thresholds)}회 실행 "
              f"(게이트={'ON' if gate else 'OFF'}, 종목당 {capital_per_trade:,.0f}원)\n")

    if verbose:
        print("[1/2] 영역 점수 사전 계산 (종목당 1회) ...")
    areas_bt, atr_bt, phase_bt, _, _ = prepare_scoring_inputs(stock_data)

    if verbose:
        print("[2/2] 조합 그리드 시뮬레이션 ...")

    all_trades_rows = []
    summary_rows = []
    total_runs = len(combos) * len(thresholds)
    run_count = 0

    for combo in combos:
        label = combo["label"]
        weights = combo["weights"]
        scored_data = build_scored_data(
            stock_data, areas_bt, atr_bt, phase_bt, weights,
            gate=gate, exclude_phases=exclude_phases,
        )

        for thr in thresholds:
            run_count += 1
            if verbose:
                print(f"[{run_count:3d}/{total_runs}] {label} thr={thr}",
                      end=" ... ", flush=True)
            try:
                trades_list, equity_df = run_simulation(
                    scored_data, trade_start, trade_end,
                    entry_threshold=thr, capital_per_trade=capital_per_trade,
                )
                tdf = trades_to_df(trades_list)
                metrics = compute_metrics(tdf, equity_df, kospi_df, trade_start, trade_end)

                if not tdf.empty:
                    tdf.insert(0, "weight_label", label)
                    tdf.insert(1, "threshold", thr)
                    all_trades_rows.append(tdf)

                summary_rows.append({
                    "weight_label":                 label,
                    "threshold":                    thr,
                    **{k: weights[k] for k in AREA_KEYS},
                    "total_return_pct":             metrics["total_return_pct"],
                    "kospi_return_pct":             metrics.get("kospi_return_pct"),
                    "alpha_pct":                    metrics.get("alpha_pct"),
                    "n_trades":                     metrics["n_trades"],
                    "win_rate_pct":                 metrics["win_rate_pct"],
                    "profit_factor":                metrics["profit_factor"],
                    "avg_trade_return_pct":         metrics["avg_trade_return_pct"],
                    "robust_avg_trade_return_pct":  metrics["robust_avg_trade_return_pct"],
                    "robust_total_pnl_krw":         metrics["robust_total_pnl_krw"],
                    "top5_pnl_share_pct":           metrics["top5_pnl_share_pct"],
                    "mdd_pct":                      metrics["mdd_pct"],
                    "sharpe":                       metrics["sharpe"],
                    "avg_hold_days":                metrics["avg_hold_days"],
                })

                if verbose:
                    print(f"ret={metrics['total_return_pct']:+.2f}% | "
                          f"robust평균={metrics['robust_avg_trade_return_pct']:+.2f}% | "
                          f"N={metrics['n_trades']}")
            except Exception as exc:
                if verbose:
                    print(f"오류: {exc}")
                    traceback.print_exc()
                summary_rows.append({
                    "weight_label": label, "threshold": thr, "error": str(exc),
                })

    all_trades_df = (pd.concat(all_trades_rows, ignore_index=True)
                     if all_trades_rows else pd.DataFrame())
    summary_df = pd.DataFrame(summary_rows)

    # robust 평균수익 기준 정렬 (없으면 총수익률) — 소수 대박 의존 조합 패널티
    if "robust_avg_trade_return_pct" in summary_df.columns and \
            summary_df["robust_avg_trade_return_pct"].notna().any():
        summary_df = summary_df.sort_values(
            ["robust_avg_trade_return_pct", "total_return_pct"],
            ascending=False).reset_index(drop=True)
    elif "total_return_pct" in summary_df.columns:
        summary_df = summary_df.sort_values(
            "total_return_pct", ascending=False).reset_index(drop=True)
    summary_df.insert(0, "rank", range(1, len(summary_df) + 1))

    return all_trades_df, summary_df


# ---------------------------------------------------------------------------
# 결과 리포트 생성
# ---------------------------------------------------------------------------

def make_weight_ranking_md(summary_df: pd.DataFrame, top_n: int = 60) -> str:
    """
    가중치 조합 랭킹 Markdown (robust 평균수익 기준).

    행 형식은 scripts/update_strategy.py 의 파서와 약속된 정형:
    | 순위 | T20/M20/Vu0/Va60 | 6.0 | ... |
    """
    lines = [
        "# 가중치 조합 랭킹 (v2_combined)\n",
        "> 정렬: **robust 평균수익** (거래 수익 상위 5건 제외 평균 — 소수 대박 의존 패널티)  ",
        "> 가중치 표기: T(추세)/M(모멘텀)/Vu(거래량)/Va(변동성) %  ",
        "> `scripts/update_strategy.py --from-ranking <이 파일>` 으로 1위를 yaml 에 반영\n",
        "| 순위 | 가중치 (T/M/Vu/Va) | 임계값 | robust평균수익 | 총수익률 | KOSPI알파 | 승률 | PF | MDD | 평균보유 | 거래수 | 상위5의존 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in summary_df.head(top_n).iterrows():
        if "error" in row and pd.notna(row.get("error")):
            lines.append(
                f"| {row['rank']} | {row['weight_label']} | {row.get('threshold','—')} "
                f"| ERROR | — | — | — | — | — | — | — | — |")
            continue
        alpha = (f"{row['alpha_pct']:+.2f}%"
                 if pd.notna(row.get("alpha_pct")) else "N/A")
        top5 = (f"{row['top5_pnl_share_pct']:.0f}%"
                if pd.notna(row.get("top5_pnl_share_pct")) else "—")
        lines.append(
            f"| {row['rank']} "
            f"| {row['weight_label']} "
            f"| {row['threshold']:.1f} "
            f"| {row['robust_avg_trade_return_pct']:+.2f}% "
            f"| {row['total_return_pct']:+.2f}% "
            f"| {alpha} "
            f"| {row['win_rate_pct']:.1f}% "
            f"| {row['profit_factor']:.2f} "
            f"| {row['mdd_pct']:.2f}% "
            f"| {row['avg_hold_days']:.1f}일 "
            f"| {int(row['n_trades'])} "
            f"| {top5} |"
        )
    return "\n".join(lines)


def make_backtest_report_md(
    summary_df:  pd.DataFrame,
    all_trades:  pd.DataFrame,
    trade_start: str,
    trade_end:   str,
    config_desc: str = "",
) -> str:
    """전체 백테스트 결과 요약 Markdown 보고서."""
    n_total = len(summary_df)
    ok = summary_df.dropna(subset=["total_return_pct"]) if not summary_df.empty else summary_df
    best = ok.iloc[0] if not ok.empty else None

    lines = [
        "# Magic Formula 백테스트 리포트 (v2_combined)\n",
        f"**기간**: {trade_start} ~ {trade_end}  ",
        f"**실행**: {len(ok)}/{n_total} 조합 성공  ",
    ]
    if config_desc:
        lines.append(f"**설정**: {config_desc}  ")
    lines.append("\n---\n")

    if best is not None:
        alpha_best = (f"{best['alpha_pct']:+.2f}%"
                      if pd.notna(best.get("alpha_pct")) else "N/A")
        lines += [
            "## 🏆 최고 성과 조합 (robust 기준)\n",
            f"- **가중치**: {best['weight_label']}  /  임계값 {best['threshold']:.1f}",
            f"- **robust 평균수익** (상위5 제외): {best['robust_avg_trade_return_pct']:+.2f}%",
            f"- **총 수익률**: {best['total_return_pct']:+.2f}%   |   KOSPI 알파: {alpha_best}",
            f"- **승률**: {best['win_rate_pct']:.1f}%   |   PF: {best['profit_factor']:.2f}"
            f"   |   MDD: {best['mdd_pct']:.2f}%",
            f"- **거래**: {int(best['n_trades'])}건, 평균 보유 {best['avg_hold_days']:.1f}일\n",
        ]

    lines += ["## 상위 5개 조합\n"]
    for _, row in ok.head(5).iterrows():
        alpha = (f"{row['alpha_pct']:+.2f}%"
                 if pd.notna(row.get("alpha_pct")) else "N/A")
        lines.append(
            f"**{row['rank']}위** `{row['weight_label']}` thr={row['threshold']:.1f}  "
            f"robust {row['robust_avg_trade_return_pct']:+.2f}% | "
            f"수익률 {row['total_return_pct']:+.2f}% | 알파 {alpha} | "
            f"승률 {row['win_rate_pct']:.1f}% | PF {row['profit_factor']:.2f}\n"
        )

    if not all_trades.empty and "net_pnl" in all_trades.columns:
        total_trades = len(all_trades)
        lines += [
            "\n---\n",
            "## 전체 거래 통계 (전 조합 합산)\n",
            f"- 총 거래 수: {total_trades:,}건",
            f"- 평균 손익: {all_trades['net_pnl'].mean():,.0f}원",
            f"- 최대 이익: {all_trades['net_pnl'].max():,.0f}원",
            f"- 최대 손실: {all_trades['net_pnl'].min():,.0f}원",
        ]
        if "exit_reason" in all_trades.columns:
            dist = all_trades["exit_reason"].value_counts()
            lines.append("\n**청산 사유 분포 (전체):**")
            for reason, cnt in dist.items():
                lines.append(f"- {reason}: {cnt}건 ({cnt/total_trades*100:.1f}%)")

    return "\n".join(lines)
