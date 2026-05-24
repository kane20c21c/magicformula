"""
optimizer/optimizer.py
----------------------
8개 가중치 조합 × 3개 진입 규칙(R1·R3·ADAPTIVE) = 24회 백테스트를 실행하고
결과를 집계한다.

ADAPTIVE 규칙: 진입 신호 당일 직전 40거래일 점수 패턴으로 종목별 규칙(R1/R3/SKIP)을
               동적으로 결정 (rolling 분류 — 정적 워밍업 분류 대체).

가중치 조합 구조 (설계 v3 — 규칙별 최적화)
-------------------------------------------
R1 최적화 조합 (combination11~14): Volume↑, Wyckoff↓
  - 임계 돌파형 종목에서 거래량 확인 강화, 와이코프 가중치 축소
R3 최적화 조합 (combination21~24): Trend↑, Volume↓
  - 계단형 추세 종목에서 추세 강도 강화, 거래량 가중치 축소
각 그룹은 기준→1단계→2단계 순으로 해당 방향 변화, 역방향은 반대 조정.
"""

from __future__ import annotations

import itertools
import sys
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# 경로 설정 — Magic Formula 루트를 sys.path 에 추가 (P5a)
# ---------------------------------------------------------------------------

_PROJ_ROOT = Path(__file__).parent.parent.parent          # Magic Formula/
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from magic_formula.data.collector import get_backtest_split, TICKERS
from magic_formula.signals.adaptive_rule_selector import select_rules_for_backtest

# ---------------------------------------------------------------------------
# 종목 universe — _vault 가 vault path + CORE_TICKERS + EXCLUDE 를 단일화 (P3)
# ---------------------------------------------------------------------------

from magic_formula._vault import get_universe   # noqa: E402

TICKER_LIST: list[str] = get_universe("core_excl_split")
if not TICKER_LIST:
    TICKER_LIST = sorted(TICKERS.keys())

from magic_formula.scoring.scorer import BASIC_WEIGHTS, compute_scores
from magic_formula.simulator.simulator import run_simulation, simulate_ticker, trades_to_df
from magic_formula.metrics.metrics import compute_metrics

# ---------------------------------------------------------------------------
# 영역 키 순서 (고정)
# ---------------------------------------------------------------------------

AREAS = ("trend", "momentum", "volume", "volatility", "wyckoff")


# ---------------------------------------------------------------------------
# 8개 가중치 조합 (v3 — 규칙별 최적화)
# ---------------------------------------------------------------------------

# 조합 코드 형식: T-M-V-P-W (각 영역 가중치 %, 합계 100)
# R1 최적화: Volume↑ Wyckoff↓ (분석 결과: R1 수익률 ∝ V+, W-)
# R3 최적화: Trend↑ Volume↓  (분석 결과: R3 수익률 ∝ T+, V-)

_COMBO_SPECS = [
    # label         T      M      V      P      W
    ("C11_R1_base", 0.20,  0.22,  0.33,  0.13,  0.12),  # R1 기준
    ("C12_R1_st1",  0.20,  0.20,  0.35,  0.15,  0.10),  # R1 1단계 (V↑↑ W↓↓)
    ("C13_R1_st2",  0.20,  0.18,  0.37,  0.17,  0.08),  # R1 2단계 (V↑↑↑ W↓↓↓)
    ("C14_R1_rev",  0.20,  0.24,  0.31,  0.11,  0.14),  # R1 역방향 (V↓ W↑)
    ("C21_R3_base", 0.23,  0.22,  0.27,  0.13,  0.15),  # R3 기준
    ("C22_R3_st1",  0.25,  0.20,  0.25,  0.15,  0.15),  # R3 1단계 (T↑↑ V↓↓)
    ("C23_R3_st2",  0.27,  0.18,  0.23,  0.17,  0.15),  # R3 2단계 (T↑↑↑ V↓↓↓)
    ("C24_R3_rev",  0.21,  0.24,  0.29,  0.11,  0.15),  # R3 역방향 (T↓ V↑)
]

# 가중치 합 검증
for _spec in _COMBO_SPECS:
    _total = sum(_spec[1:])
    assert abs(_total - 1.0) < 1e-9, f"{_spec[0]} 가중치 합 오류: {_total}"


def generate_weight_combinations() -> list[dict]:
    """
    8개 가중치 조합 목록을 반환한다 (v3 — 규칙별 최적화 설계).

    Returns
    -------
    list of dict with keys:
        label    : 조합 레이블 (C11_R1_base 등)
        weights  : {area: weight}
    """
    combos = []
    for label, t, m, v, p, w in _COMBO_SPECS:
        combos.append({
            "label": label,
            "weights": {
                "trend":      t,
                "momentum":   m,
                "volume":     v,
                "volatility": p,
                "wyckoff":    w,
            },
        })
    return combos


# ---------------------------------------------------------------------------
# 전체 백테스트 실행기
# ---------------------------------------------------------------------------

RULES = ("R1", "R3", "ADAPTIVE")   # R2 제거 (R1과 실질 동일)


def run_all(
    raw_data:       dict[str, pd.DataFrame],
    kospi_df:       Optional[pd.DataFrame],
    warmup_months:  int = 12,
    verbose:        bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    51조합 × 4규칙(R1·R2·R3·ADAPTIVE) 전체 백테스트를 실행한다.

    Parameters
    ----------
    raw_data       : collect_all() 반환값 (ticker → OHLCV DataFrame)
    kospi_df       : KOSPI OHLCV DataFrame (알파 계산용)
    warmup_months  : 워밍업 개월 수 (기본 12)
    verbose        : 진행 상황 출력 여부

    Returns
    -------
    (all_trades_df, summary_df)
    all_trades_df : 모든 거래 기록 (weight_label, rule 컬럼 포함)
    summary_df    : 조합별 성과 요약 (알파 기준 정렬)
    """
    combos = generate_weight_combinations()
    stock_tickers = [t for t in raw_data if t != "KOSPI"]

    # 각 종목의 실거래 구간 결정 (가장 짧은 데이터 기준)
    trade_start = None
    trade_end   = None
    for ticker in stock_tickers:
        df = raw_data[ticker]
        _, t_idx = get_backtest_split(df, warmup_months)
        if len(t_idx) == 0:
            continue
        ts = t_idx[0]
        te = t_idx[-1]
        trade_start = ts if trade_start is None else max(trade_start, ts)
        trade_end   = te if trade_end   is None else min(trade_end,   te)

    if trade_start is None:
        raise ValueError("실거래 구간이 없습니다. 데이터 기간을 확인하세요.")

    if verbose:
        print(f"\n실거래 구간: {trade_start.date()} ~ {trade_end.date()}")
        n_rules = len(RULES)
        print(f"51조합 × {n_rules}규칙 = {51*n_rules}회 실행 (ADAPTIVE 포함)\n")

    all_trades_rows = []
    summary_rows    = []
    total_runs      = len(combos) * len(RULES)
    run_count       = 0

    for combo in combos:
        label   = combo["label"]
        weights = combo["weights"]

        # 점수 사전 계산 (조합당 1회)
        scored_data: dict[str, pd.DataFrame] = {}
        for ticker in stock_tickers:
            try:
                scored_data[ticker] = compute_scores(raw_data[ticker], weights)
            except Exception as exc:
                if verbose:
                    print(f"  [scorer] {ticker} 오류: {exc} — 건너뜀")

        for rule in RULES:
            run_count += 1
            if verbose:
                print(f"[{run_count:3d}/{total_runs}] {label} / {rule}", end=" ... ", flush=True)

            try:
                # ── 모든 규칙 (R1 / R3 / ADAPTIVE) ──────────────────────
                # ADAPTIVE 는 simulate_ticker() 내부에서 진입 신호 당일
                # 직전 40거래일 점수로 rolling 동적 분류 수행 (look-ahead bias 없음)
                trades_list, equity_df = run_simulation(
                    scored_data, rule, trade_start, trade_end
                )

                tdf = trades_to_df(trades_list)
                metrics = compute_metrics(tdf, equity_df, kospi_df, trade_start, trade_end)

                # 거래 기록에 레이블 추가
                if not tdf.empty:
                    tdf.insert(0, "weight_label", label)
                    tdf.insert(1, "rule", rule)
                    all_trades_rows.append(tdf)

                # 요약
                summary_rows.append({
                    "weight_label":         label,
                    "rule":                 rule,
                    "total_return_pct":     metrics["total_return_pct"],
                    "kospi_return_pct":     metrics.get("kospi_return_pct"),
                    "alpha_pct":            metrics.get("alpha_pct"),
                    "n_trades":             metrics["n_trades"],
                    "win_rate_pct":         metrics["win_rate_pct"],
                    "profit_factor":        metrics["profit_factor"],
                    "avg_trade_return_pct": metrics["avg_trade_return_pct"],
                    "mdd_pct":              metrics["mdd_pct"],
                    "sharpe":              metrics["sharpe"],
                    "sortino":             metrics["sortino"],
                    "calmar":              metrics["calmar"],
                    "avg_hold_days":       metrics["avg_hold_days"],
                })

                if verbose:
                    alpha_str = (
                        f"alpha={metrics['alpha_pct']:+.2f}%"
                        if metrics.get("alpha_pct") is not None
                        else "alpha=N/A"
                    )
                    print(
                        f"ret={metrics['total_return_pct']:+.2f}% | "
                        f"{alpha_str} | "
                        f"win={metrics['win_rate_pct']:.0f}% | "
                        f"trades={metrics['n_trades']}"
                    )

            except Exception as exc:
                if verbose:
                    print(f"오류: {exc}")
                    traceback.print_exc()
                summary_rows.append({
                    "weight_label": label,
                    "rule":         rule,
                    "error":        str(exc),
                })

    # 결과 집계
    all_trades_df = pd.concat(all_trades_rows, ignore_index=True) if all_trades_rows else pd.DataFrame()
    summary_df    = pd.DataFrame(summary_rows)

    # 알파 기준 정렬 (알파 없을 시 총 수익률 기준)
    if "alpha_pct" in summary_df.columns and summary_df["alpha_pct"].notna().any():
        summary_df = summary_df.sort_values("alpha_pct", ascending=False).reset_index(drop=True)
    elif "total_return_pct" in summary_df.columns:
        summary_df = summary_df.sort_values("total_return_pct", ascending=False).reset_index(drop=True)

    summary_df.insert(0, "rank", range(1, len(summary_df) + 1))

    return all_trades_df, summary_df


# ---------------------------------------------------------------------------
# 조합 코드 생성 유틸
# ---------------------------------------------------------------------------

# 영역 순서 (5글자 코드의 위치 순서 고정)
_CODE_AREAS = ("trend", "momentum", "volume", "volatility", "wyckoff")
_CODE_AREA_ABBR = {
    "trend":      "T",
    "momentum":   "M",
    "volume":     "V",
    "volatility": "P",   # Position/Volatility
    "wyckoff":    "W",
}


def _weights_to_code(weights: dict[str, float]) -> str:
    """
    가중치 dict → 5글자 L/B/H 코드 + 가중치 수치 문자열.

    영역 순서: Trend(T) - Momentum(M) - Volume(V) - Volatility(P) - Wyckoff(W)

    각 영역:
      Low  (Basic - 0.05) → L
      Basic               → B
      High (Basic + 0.05) → H
      (그 외 ε 허용 0.001)

    반환 예시: "BBBBB (20/25/25/10/20)"
    """
    code_chars = []
    pct_parts  = []

    for area in _CODE_AREAS:
        w    = weights.get(area, BASIC_W[area])
        base = BASIC_W[area]
        pct_parts.append(f"{round(w * 100)}")

        if abs(w - (base - DELTA)) < 1e-6:
            code_chars.append("L")
        elif abs(w - (base + DELTA)) < 1e-6:
            code_chars.append("H")
        else:
            code_chars.append("B")

    code = "".join(code_chars)
    pct  = "/".join(pct_parts)
    return f"{code} ({pct})"


# ---------------------------------------------------------------------------
# 결과 리포트 생성
# ---------------------------------------------------------------------------

def make_weight_ranking_md(summary_df: pd.DataFrame) -> str:
    """
    가중치 조합 알파 기준 랭킹 Markdown 문자열을 생성한다.

    조합 코드: T-M-V-P-W 순서로 각 영역이 L(Low)/B(Basic)/H(High)인지 표시.
    예) BBBBB (20/25/25/10/20) — 전 영역 Basic
        HBLBH (25/25/15/5/25) — 추세·Wyckoff High, 거래량 Low
    """
    # summary_df 에 weight_label이 있으면 가중치 dict를 재구성해서 코드 생성
    combos_by_label: dict[str, dict] = {
        c["label"]: c["weights"]
        for c in generate_weight_combinations()
    }

    lines = [
        "# 가중치 조합 알파 기준 랭킹\n",
        "> 조합 코드 영역 순서: **T**(Trend) - **M**(Momentum) - **V**(Volume) - **P**(Volatility) - **W**(Wyckoff)  ",
        "> 각 영역: **L**=Low(Basic−3%) / **B**=Basic / **H**=High(Basic+3%)\n",
    ]
    lines.append("| 순위 | 조합 코드 (T-M-V-P-W) | 가중치% | 규칙 | 총 수익률 | KOSPI 알파 | 승률 | PF | Sharpe | Trades |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    for _, row in summary_df.iterrows():
        lbl = row.get("weight_label", "")

        if "error" in row and pd.notna(row.get("error")):
            lines.append(
                f"| {row['rank']} | {lbl} | — | {row['rule']} | ERROR | — | — | — | — | — |"
            )
            continue

        # 코드 생성
        if lbl in combos_by_label:
            w   = combos_by_label[lbl]
            chars = []
            pcts  = []
            for area in _CODE_AREAS:
                wv   = w.get(area, BASIC_W[area])
                base = BASIC_W[area]
                pcts.append(str(round(wv * 100)))
                if abs(wv - (base - DELTA)) < 1e-6:
                    chars.append("L")
                elif abs(wv - (base + DELTA)) < 1e-6:
                    chars.append("H")
                else:
                    chars.append("B")
            code_str = "".join(chars)
            pct_str  = "/".join(pcts)
        else:
            code_str = lbl[:5] if lbl else "?????"
            pct_str  = "—"

        alpha = f"{row['alpha_pct']:+.2f}%" if pd.notna(row.get("alpha_pct")) else "N/A"
        lines.append(
            f"| {row['rank']} "
            f"| `{code_str}` "
            f"| {pct_str} "
            f"| {row['rule']} "
            f"| {row['total_return_pct']:+.2f}% "
            f"| {alpha} "
            f"| {row['win_rate_pct']:.1f}% "
            f"| {row['profit_factor']:.2f} "
            f"| {row['sharpe']:.2f} "
            f"| {int(row['n_trades'])} |"
        )

    return "\n".join(lines)


def make_backtest_report_md(
    summary_df:  pd.DataFrame,
    all_trades:  pd.DataFrame,
    trade_start: str,
    trade_end:   str,
) -> str:
    """
    전체 백테스트 결과 요약 Markdown 보고서를 생성한다.
    """
    top5 = summary_df.head(5)

    n_total   = len(summary_df)
    n_success = summary_df.dropna(subset=["total_return_pct"]).shape[0] if not summary_df.empty else 0

    best = summary_df.iloc[0] if not summary_df.empty else None

    lines = [
        "# Magic Formula 백테스트 리포트\n",
        f"**기간**: {trade_start} ~ {trade_end}  ",
        f"**조합 수**: 51 가중치 × 3 규칙 = 153회  ",
        f"**성공 실행**: {n_success}/{n_total}\n",
        "---\n",
    ]

    if best is not None:
        alpha_best = f"{best['alpha_pct']:+.2f}%" if pd.notna(best.get("alpha_pct")) else "N/A"
        lines += [
            "## 🏆 최고 성과 조합\n",
            f"- **조합**: {best['weight_label']} / {best['rule']}",
            f"- **총 수익률**: {best['total_return_pct']:+.2f}%",
            f"- **KOSPI 알파**: {alpha_best}",
            f"- **승률**: {best['win_rate_pct']:.1f}%",
            f"- **Profit Factor**: {best['profit_factor']:.2f}",
            f"- **MDD**: {best['mdd_pct']:.2f}%\n",
        ]

    lines += ["## 상위 5개 조합\n"]
    for _, row in top5.iterrows():
        if "error" in row and pd.notna(row.get("error")):
            continue
        alpha = f"{row['alpha_pct']:+.2f}%" if pd.notna(row.get("alpha_pct")) else "N/A"
        lines.append(
            f"**{row['rank']}위** `{row['weight_label']}` / `{row['rule']}`  "
            f"수익률 {row['total_return_pct']:+.2f}% | 알파 {alpha} | "
            f"승률 {row['win_rate_pct']:.1f}% | PF {row['profit_factor']:.2f}\n"
        )

    # 거래 통계
    if not all_trades.empty and "net_pnl" in all_trades.columns:
        total_trades = len(all_trades)
        winners      = all_trades[all_trades["net_pnl"] > 0]
        lines += [
            "\n---\n",
            "## 전체 거래 통계 (153조합 합산)\n",
            f"- 총 거래 수: {total_trades:,}건",
            f"- 평균 수익: {all_trades['net_pnl'].mean():,.0f}원",
            f"- 최대 이익: {all_trades['net_pnl'].max():,.0f}원",
            f"- 최대 손실: {all_trades['net_pnl'].min():,.0f}원",
        ]
        if "exit_reason" in all_trades.columns:
            dist = all_trades["exit_reason"].value_counts()
            lines.append("\n**청산 사유 분포 (전체):**")
            for reason, cnt in dist.items():
                lines.append(f"- {reason}: {cnt}건 ({cnt/total_trades*100:.1f}%)")

    return "\n".join(lines)
