"""
main.py
-------
Magic Formula 분석 트랙 진입점 (v2_combined 백테스트 전체 파이프라인).

실행 방법
---------
# 전체 그리드 (56조합 × 2임계값 = 112회)
python scripts/run_analysis.py

# 현 운영 전략(yaml) 1조합만 빠른 검증
python scripts/run_analysis.py --quick-test

# 그리드 해상도/임계값 변경
python scripts/run_analysis.py --step 0.1 --thresholds 5.0,6.0,7.0

# 게이트 끄고 비교
python scripts/run_analysis.py --no-gate

파이프라인 순서
--------------
1. 데이터 수집 (collector → vault full-column, Wyckoff 포함)
2. 영역 점수 사전 계산 (종목당 1회) + 가중치 그리드 시뮬레이션 (optimizer)
3. output/analysis/YYYY-MM-DD/ 에 저장:
   - weight_ranking.md   ← scripts/update_strategy.py 가 읽음
   - backtest_report.md
   - summary.csv / trades.csv
   - equity_curves.png  (matplotlib 설치 시, 상위 5조합)

변경 이력
---------
2026-06-10 v2 단일화: v1 파이프라인(51조합 × R1/R3/ADAPTIVE) 폐기.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path 설정 — Magic Formula 루트(magic_formula 의 부모)를 추가
# ---------------------------------------------------------------------------

PKG_DIR    = Path(__file__).parent            # magic_formula/
PROJ_DIR   = PKG_DIR.parent                   # Magic Formula/

if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

from magic_formula._vault import get_universe                       # noqa: E402
from magic_formula.config import load_strategy                      # noqa: E402
from magic_formula.data.collector import collect_all                # noqa: E402
from magic_formula.metrics.metrics import INITIAL_CAPITAL           # noqa: E402
from magic_formula.optimizer.optimizer import (                     # noqa: E402
    build_scored_data,
    make_backtest_report_md,
    make_weight_ranking_md,
    prepare_scoring_inputs,
    run_all,
    weights_label,
    _trade_window,
)
from magic_formula.simulator.simulator import run_simulation        # noqa: E402

# ---------------------------------------------------------------------------
# 선택적 matplotlib (없어도 나머지 실행)
# ---------------------------------------------------------------------------

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates       # noqa: F401
    import matplotlib.pyplot as plt
    _MATPLOTLIB_OK = True
except ImportError:
    _MATPLOTLIB_OK = False


# ---------------------------------------------------------------------------
# 자산 곡선 플롯 (상위 N개 조합 재시뮬레이션)
# ---------------------------------------------------------------------------

def plot_equity_curves(
    summary_df,
    stock_data,
    kospi_df,
    gate: bool,
    capital_per_trade: float,
    warmup_months: int,
    output_path: Path,
    top_n: int = 5,
):
    """상위 N개 조합의 자산 곡선 vs KOSPI 를 PNG 로 저장."""
    if not _MATPLOTLIB_OK:
        print("[plot] matplotlib 미설치 — equity_curves.png 생략")
        return

    trade_start, trade_end = _trade_window(stock_data, warmup_months)
    areas_bt, atr_bt, phase_bt, _, _ = prepare_scoring_inputs(stock_data)

    fig, ax = plt.subplots(figsize=(12, 6))
    top = summary_df.dropna(subset=["total_return_pct"]).head(top_n)

    for _, row in top.iterrows():
        weights = {k: float(row[k]) for k in ("trend", "momentum", "volume", "volatility")}
        scored = build_scored_data(
            stock_data, areas_bt, atr_bt, phase_bt, weights, gate=gate)
        _, eq_df = run_simulation(
            scored, trade_start, trade_end,
            entry_threshold=float(row["threshold"]),
            capital_per_trade=capital_per_trade,
        )
        if "total" not in eq_df.columns or eq_df.empty:
            continue
        abs_eq = (INITIAL_CAPITAL + eq_df["total"]) / INITIAL_CAPITAL * 100.0
        abs_eq.plot(ax=ax, label=f"{row['weight_label']} thr={row['threshold']:.1f}")

    if kospi_df is not None and not kospi_df.empty:
        kdf = kospi_df.loc[trade_start:trade_end, "Close"]
        if len(kdf) >= 2:
            (kdf / kdf.iloc[0] * 100.0).plot(
                ax=ax, color="gray", linestyle="--", linewidth=1.5, label="KOSPI")

    ax.axhline(100, color="black", linestyle=":", linewidth=0.8)
    ax.set_title(f"상위 {top_n}개 조합 자산 곡선 vs KOSPI (v2_combined)", fontsize=14)
    ax.set_ylabel("수익률 기준 지수 (진입=100)", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  저장: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python scripts/run_analysis.py",
        description=(
            "Magic Formula v2_combined 백테스트 — 가중치 그리드 × 임계값 최적화\n"
            "결과는 output/analysis/YYYY-MM-DD/ 에 저장됩니다."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python scripts/run_analysis.py                       # 전체 그리드 (56×2)
  python scripts/run_analysis.py --quick-test          # 현 운영 전략 1조합 검증
  python scripts/run_analysis.py --step 0.1            # 정밀 그리드 (286조합)
  python scripts/run_analysis.py --thresholds 5.0,6.0,7.0
  python scripts/run_analysis.py --no-gate             # Wyckoff 게이트 OFF 비교
        """,
    )
    p.add_argument("--quick-test", action="store_true",
                   help="현 운영 전략(yaml 가중치/임계값) 1조합만 실행합니다.")
    p.add_argument("--months", type=int, default=18, metavar="N",
                   help="데이터 수집 기간(개월). 기본 18")
    p.add_argument("--warmup", type=int, default=12, metavar="N",
                   help="워밍업 기간(개월). 기본 12")
    p.add_argument("--step", type=float, default=0.2, metavar="F",
                   help="가중치 그리드 간격. 기본 0.2 (56조합)")
    p.add_argument("--thresholds", type=str, default="5.0,6.0", metavar="A,B",
                   help="진입 임계값 후보 (콤마 구분). 기본 '5.0,6.0'")
    p.add_argument("--no-gate", action="store_true",
                   help="Wyckoff Markdown 게이트를 끕니다.")
    p.add_argument("--output-dir", type=Path, default=None, metavar="PATH",
                   help="결과 저장 폴더. 기본: output/analysis/YYYY-MM-DD/")
    p.add_argument("--quiet", action="store_true", help="진행 출력 최소화")
    return p


# ---------------------------------------------------------------------------
# 메인 파이프라인
# ---------------------------------------------------------------------------

def main():
    args = build_parser().parse_args()
    verbose = not args.quiet
    start_time = datetime.now()

    thresholds = tuple(float(x) for x in args.thresholds.split(",") if x.strip())

    # 운영 yaml — universe / 게이트 / 자본 / (quick-test 시) 가중치·임계값
    cfg = load_strategy()
    gate = (not args.no_gate) and cfg.gate_enabled

    output_dir: Path = args.output_dir or (
        PROJ_DIR / "output" / "analysis" / datetime.today().strftime("%Y-%m-%d"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Magic Formula v2_combined 백테스트 시작")
    print(f"시각: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"전략 yaml: {cfg.summary()}")
    print("=" * 60)

    # ── Step 1: 데이터 수집 ──
    print("\n[Step 1] 데이터 수집")
    ticker_list = get_universe(cfg.universe)
    if not ticker_list:
        print("vault 미설치 — universe 를 해석할 수 없습니다. 종료.")
        sys.exit(1)

    raw_data = collect_all(months=args.months, ticker_list=ticker_list)
    if not raw_data:
        print("데이터 수집 실패. 종료합니다.")
        sys.exit(1)

    kospi_df = raw_data.pop("KOSPI", None)
    stock_data = raw_data
    if not stock_data:
        print("종목 데이터 없음. 종료합니다.")
        sys.exit(1)

    # ── Step 2: 백테스트 ──
    if args.quick_test:
        print(f"\n[Quick Test] 현 운영 전략 단일 실행 — "
              f"{weights_label(cfg.weights)} thr={cfg.threshold}")
        weights_list = [{"label": weights_label(cfg.weights), "weights": cfg.weights}]
        run_thresholds = (cfg.threshold,)
    else:
        weights_list = None
        run_thresholds = thresholds

    print(f"\n[Step 2] 백테스트 실행")
    all_trades_df, summary_df = run_all(
        stock_data, kospi_df,
        thresholds=run_thresholds,
        step=args.step,
        warmup_months=args.warmup,
        gate=gate,
        exclude_phases=cfg.gate_exclude_phases,
        capital_per_trade=cfg.position_size,
        weights_list=weights_list,
        verbose=verbose,
    )

    # ── Step 3: 결과 저장 ──
    print(f"\n[Step 3] 결과 저장 → {output_dir}")

    if not all_trades_df.empty:
        trades_path = output_dir / "trades.csv"
        all_trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")
        print(f"  저장: {trades_path}  ({len(all_trades_df):,}건)")

    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"  저장: {summary_path}")

    ranking_path = output_dir / "weight_ranking.md"
    ranking_path.write_text(make_weight_ranking_md(summary_df), encoding="utf-8")
    print(f"  저장: {ranking_path}")

    trade_start, trade_end = _trade_window(stock_data, args.warmup)
    config_desc = (f"universe={cfg.universe} ({len(stock_data)}종목), "
                   f"게이트={'ON' if gate else 'OFF'}, "
                   f"종목당 {cfg.position_size:,}원, "
                   f"임계값 {list(run_thresholds)}, step={args.step}")
    report_path = output_dir / "backtest_report.md"
    report_path.write_text(
        make_backtest_report_md(
            summary_df, all_trades_df,
            str(trade_start.date()), str(trade_end.date()),
            config_desc=config_desc,
        ),
        encoding="utf-8",
    )
    print(f"  저장: {report_path}")

    if _MATPLOTLIB_OK and not args.quick_test:
        print("  equity_curves.png 생성 중...")
        try:
            plot_equity_curves(
                summary_df, stock_data, kospi_df,
                gate=gate, capital_per_trade=cfg.position_size,
                warmup_months=args.warmup,
                output_path=output_dir / "equity_curves.png",
            )
        except Exception as exc:
            print(f"  equity_curves.png 생성 실패: {exc}")

    # ── 완료 요약 ──
    elapsed = datetime.now() - start_time
    print(f"\n{'='*60}")
    print(f"백테스트 완료 — 소요 시간: {elapsed}")

    ok = summary_df.dropna(subset=["total_return_pct"]) if not summary_df.empty else summary_df
    if not ok.empty:
        best = ok.iloc[0]
        print(f"\n🏆 최고 조합 (robust 기준): "
              f"[{best['weight_label']} / thr={best['threshold']:.1f}]")
        print(f"   robust평균: {best['robust_avg_trade_return_pct']:+.2f}%  |  "
              f"수익률: {best['total_return_pct']:+.2f}%  |  "
              f"승률: {best['win_rate_pct']:.1f}%  |  N={int(best['n_trades'])}")
        print(f"\n반영: python scripts/update_strategy.py --from-ranking {ranking_path}")

    print(f"\n결과 폴더: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
