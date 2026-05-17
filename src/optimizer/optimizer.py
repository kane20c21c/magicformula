"""
optimizer/optimizer.py
----------------------
51개 가중치 조합 × 3개 진입 규칙(R1·R2·R3) × 57종목 = 8,721회 백테스트를 실행하고
결과를 집계한다.

가중치 조합 구조 (설계 v2 Section 9)
-------------------------------------
Basic : {trend:0.20, momentum:0.25, volume:0.30, volatility:0.10, wyckoff:0.15}
Low   : Basic - 0.03  /  High : Basic + 0.03

패턴 1 — 전 영역 Basic (1개)
패턴 2 — LL×1 + HL×1 + BL×3  →  C(5,1)×C(4,1) = 20개
패턴 3 — LL×2 + HL×2 + BL×1  →  C(5,2)×C(3,2) = 30개
합계: 51개

LL/HL 동수 → 가중치 합 자동 100% (정규화 불필요)
"""

from __future__ import annotations

import itertools
import sys
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# 경로 설정 (src/ 디렉터리를 sys.path에 추가)
# ---------------------------------------------------------------------------

_SRC = Path(__file__).parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data.collector import get_backtest_split, TICKERS
from signals.adaptive_rule_selector import select_rules_for_backtest

# ---------------------------------------------------------------------------
# 59종목 TICKER_LIST (longlivevault CORE_TICKERS 우선)
# ---------------------------------------------------------------------------

_VAULT_PATH = next(
    (p for p in [
        str(_SRC.parent.parent / "longlivevault"),
        "/Users/kaneyoun/DriveForALL/StoLab/longlivevault",
    ] if Path(p).exists()),
    str(_SRC.parent.parent / "longlivevault"),
)
if _VAULT_PATH not in sys.path:
    sys.path.insert(0, _VAULT_PATH)

# 사업 분할 등 분석 제외 종목
_EXCLUDE_TICKERS = {"207940", "0126Z0"}  # 삼성바이오로직스, 삼성에피스홀딩스

try:
    from stolab_data.ohlcv_store import CORE_TICKERS as _CORE_TICKERS  # type: ignore
    TICKER_LIST: list[str] = sorted(_CORE_TICKERS - _EXCLUDE_TICKERS)
except Exception:
    TICKER_LIST = sorted(k for k in TICKERS.keys() if k not in _EXCLUDE_TICKERS)
from scoring.scorer import BASIC_WEIGHTS, compute_scores
from simulator.simulator import run_simulation, trades_to_df
from metrics.metrics import compute_metrics

# ---------------------------------------------------------------------------
# 영역 키 순서 (고정)
# ---------------------------------------------------------------------------

AREAS = ("trend", "momentum", "volume", "volatility", "wyckoff")

BASIC_W: dict[str, float] = {
    "trend":      0.20,
    "momentum":   0.25,
    "volume":     0.30,
    "volatility": 0.10,
    "wyckoff":    0.15,
}

assert abs(sum(BASIC_W.values()) - 1.0) < 1e-9, f"BASIC_W 합계 오류: {sum(BASIC_W.values())}"

DELTA = 0.03   # Low = Basic - 0.03 / High = Basic + 0.03


# ---------------------------------------------------------------------------
# 51개 조합 생성
# ---------------------------------------------------------------------------

def _make_weights(low_areas: tuple, high_areas: tuple) -> dict[str, float]:
    """
    특정 영역에 Low/High를 적용한 가중치 dict를 반환한다.
    나머지 영역은 Basic을 유지.
    """
    w = dict(BASIC_W)
    for a in low_areas:
        w[a] = round(BASIC_W[a] - DELTA, 10)
    for a in high_areas:
        w[a] = round(BASIC_W[a] + DELTA, 10)
    return w


def _make_label(low_areas: tuple, high_areas: tuple) -> str:
    """조합의 사람이 읽기 쉬운 레이블."""
    if not low_areas and not high_areas:
        return "Basic"
    parts = [f"LL_{a}" for a in low_areas] + [f"HL_{a}" for a in high_areas]
    return "__".join(parts)


def generate_weight_combinations() -> list[dict]:
    """
    51개 가중치 조합 목록을 반환한다.

    Returns
    -------
    list of dict with keys:
        label    : 조합 레이블
        weights  : {area: weight}
    """
    combos = []

    # 패턴 1: 전 영역 Basic (1개)
    combos.append({
        "label":   "Basic",
        "weights": dict(BASIC_W),
    })

    # 패턴 2: LL×1 + HL×1 + BL×3 (20개)
    for ll_area in AREAS:
        remaining = [a for a in AREAS if a != ll_area]
        for hl_area in remaining:
            combos.append({
                "label":   _make_label((ll_area,), (hl_area,)),
                "weights": _make_weights((ll_area,), (hl_area,)),
            })

    # 패턴 3: LL×2 + HL×2 + BL×1 (30개)
    for ll_pair in itertools.combinations(AREAS, 2):
        remaining = [a for a in AREAS if a not in ll_pair]
        for hl_pair in itertools.combinations(remaining, 2):
            combos.append({
                "label":   _make_label(ll_pair, hl_pair),
                "weights": _make_weights(ll_pair, hl_pair),
            })

    assert len(combos) == 51, f"조합 수 오류: {len(combos)} (기대값: 51)"
    return combos


# ---------------------------------------------------------------------------
# 전체 백테스트 실행기
# ---------------------------------------------------------------------------

RULES = ("R1", "R2", "R3")


def run_all(
    raw_data:       dict[str, pd.DataFrame],
    kospi_df:       Optional[pd.DataFrame],
    warmup_months:  int = 12,
    verbose:        bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    51조합 × 3규칙 전체 백테스트를 실행한다.

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
                # ── ADAPTIVE 모드 ──────────────────────────────────────────
                if rule == "ADAPTIVE":
                    # 워밍업 구간(trade_start 이전) 종합 점수 추출
                    warmup_scores: dict[str, pd.Series] = {}
                    for ticker, df in scored_data.items():
                        warmup = df.loc[df.index < trade_start, "composite_score"]
                        if not warmup.empty:
                            warmup_scores[ticker] = warmup

                    # 종목별 규칙 자동 분류
                    rule_df = select_rules_for_backtest(warmup_scores)
                    assigned: dict[str, str] = dict(
                        zip(rule_df["ticker"], rule_df["rule"])
                    )
                    reasons: dict[str, str] = dict(
                        zip(rule_df["ticker"], rule_df["reason"])
                    )

                    if verbose:
                        print()  # 줄 바꿈
                        for ticker in sorted(assigned):
                            r = assigned[ticker]
                            print(f"  [ADAPTIVE] {ticker} → {r}  ({reasons[ticker]})")

                    # 규칙별로 서브셋 시뮬레이션 후 결합
                    all_adp_trades: list = []
                    equity_frames:  list = []
                    for sub_rule in ("R1", "R2", "R3"):
                        sub_scored = {
                            t: scored_data[t]
                            for t, r in assigned.items()
                            if r == sub_rule and t in scored_data
                        }
                        if not sub_scored:
                            continue
                        sub_trades, sub_equity = run_simulation(
                            sub_scored, sub_rule, trade_start, trade_end
                        )
                        all_adp_trades.extend(sub_trades)
                        equity_frames.append(sub_equity)

                    # 에쿼티 통합
                    if equity_frames:
                        combined_equity = pd.concat(equity_frames, axis=1)
                        combined_equity["total"] = combined_equity.drop(
                            columns=["total"], errors="ignore"
                        ).sum(axis=1)
                    else:
                        combined_equity = pd.DataFrame()

                    trades_list = all_adp_trades
                    equity_df   = combined_equity

                # ── 일반 규칙 (R1 / R2 / R3) ─────────────────────────────
                else:
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
