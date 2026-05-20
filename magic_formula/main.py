"""
main.py
-------
Magic Formula 백테스트 전체 파이프라인 진입점.

실행 방법
---------
# 전체 실행 (51조합 × 3규칙 × 10종목)
python src/main.py

# 캐시 무시하고 새로 데이터 받기
python src/main.py --no-cache

# 빠른 테스트 (Basic 가중치 + R1 만 실행)
python src/main.py --quick-test

# 결과 저장 폴더 지정
python src/main.py --output-dir /path/to/output

# 워밍업 기간 조정 (기본 12개월)
python src/main.py --warmup 10

파이프라인 순서
--------------
1. 데이터 수집 (collector)
2. 51조합 × 3규칙 × 10종목 백테스트 (optimizer → simulator → metrics)
3. 결과 집계
4. output/ 폴더에 저장:
   - trades.csv
   - backtest_report.md
   - weight_ranking.md
   - equity_curves.png  (matplotlib 설치 시)
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path 설정 — Magic Formula 루트(magic_formula 의 부모)를 추가하면
# `import magic_formula.xxx` 가 동작. (P5a)
# ---------------------------------------------------------------------------

PKG_DIR    = Path(__file__).parent            # magic_formula/
PROJ_DIR   = PKG_DIR.parent                   # Magic Formula/
OUTPUT_DIR = PROJ_DIR / "output"

if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

# ---------------------------------------------------------------------------
# 모듈 임포트
# ---------------------------------------------------------------------------

from magic_formula.data.collector      import collect_all, get_backtest_split, _date_range, TICKERS as TICKER_NAMES_ALL
from magic_formula.scoring.scorer      import compute_scores, BASIC_WEIGHTS, AREA4_MODES

# ---------------------------------------------------------------------------
# 종목 universe — _vault 가 vault path + CORE_TICKERS + EXCLUDE 를 단일화 (P3)
# ---------------------------------------------------------------------------

from magic_formula._vault import get_universe   # noqa: E402

# 백테스트는 "core_57" — vault CORE_TICKERS 에서 분석 제외 종목(0126Z0/207940) 뺀 57개
TICKER_LIST: list[str] = get_universe("core_57")
if not TICKER_LIST:
    # vault 미설치 폴백 — collector.TICKERS 키 사용 (구성 동일)
    TICKER_LIST = sorted(TICKER_NAMES_ALL.keys())

from magic_formula.signals.rules                  import entry_signals, ENTRY_THRESHOLD
from magic_formula.signals.adaptive_rule_selector import select_rules_for_backtest
from magic_formula.simulator.simulator import run_simulation, trades_to_df
from magic_formula.metrics.metrics     import compute_metrics, format_report, INITIAL_CAPITAL
from magic_formula.optimizer.optimizer import (
    generate_weight_combinations,
    run_all,
    make_backtest_report_md,
    make_weight_ranking_md,
    RULES,
)


# ---------------------------------------------------------------------------
# 선택적 matplotlib (없어도 나머지 실행)
# ---------------------------------------------------------------------------

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    _MATPLOTLIB_OK = True
except ImportError:
    _MATPLOTLIB_OK = False
    print("[info] matplotlib 미설치 — equity_curves.png 생략")


# ---------------------------------------------------------------------------
# 자산 곡선 플롯
# ---------------------------------------------------------------------------

def plot_equity_curves(
    summary_df,
    scored_data_cache: dict,   # {combo_label: {ticker: scored_df}}
    kospi_df,
    trade_start,
    trade_end,
    output_path: Path,
    top_n: int = 5,
    entry_threshold: float = 5.0,
):
    """상위 N개 조합의 자산 곡선 vs KOSPI 를 PNG로 저장한다."""
    if not _MATPLOTLIB_OK:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    top = summary_df.dropna(subset=["total_return_pct"]).head(top_n)

    for _, row in top.iterrows():
        label   = row["weight_label"]
        rule    = row["rule"]
        key     = label

        if key not in scored_data_cache:
            continue
        scored_data = scored_data_cache[key]

        trades, eq_df = run_simulation(
            scored_data, rule, trade_start, trade_end,
            entry_threshold=entry_threshold,
        )
        if "total" not in eq_df.columns or eq_df.empty:
            continue

        abs_eq = (INITIAL_CAPITAL + eq_df["total"]) / INITIAL_CAPITAL * 100.0
        abs_eq.plot(ax=ax, label=f"{label}/{rule}")

    # KOSPI 기준선
    if kospi_df is not None and not kospi_df.empty:
        kdf = kospi_df.loc[trade_start:trade_end, "Close"]
        if len(kdf) >= 2:
            (kdf / kdf.iloc[0] * 100.0).plot(
                ax=ax, color="gray", linestyle="--", linewidth=1.5, label="KOSPI"
            )

    ax.axhline(100, color="black", linestyle=":", linewidth=0.8)
    ax.set_title("상위 5개 조합 자산 곡선 vs KOSPI", fontsize=14)
    ax.set_ylabel("수익률 기준 지수 (진입=100)", fontsize=11)
    ax.set_xlabel("날짜", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  저장: {output_path}")


# ---------------------------------------------------------------------------
# CLI 인수 파싱
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python src/main.py",
        description=(
            "Magic Formula 백테스트 — 51개 가중치 조합 × 3개 진입 규칙 × 10종목\n"
            "결과는 output/ 폴더에 저장됩니다."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python src/main.py                              # 전체 실행
  python src/main.py --no-cache                   # 데이터 새로 수집
  python src/main.py --quick-test                 # Basic/R1 빠른 테스트 + A/B 비교표
  python src/main.py --quick-test --threshold 1.5 # 낮은 임계값 테스트
  python src/main.py --quick-test --area4-mode trend  # 추세추종 Area4 테스트
  python src/main.py --threshold 1.5 --area4-mode trend  # 전체 실행 파라미터 지정
  python src/main.py --warmup 10                  # 워밍업 10개월
  python src/main.py --output-dir ./out           # 출력 폴더 변경
        """,
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="캐시를 무시하고 pykrx에서 데이터를 새로 수집합니다.",
    )
    p.add_argument(
        "--quick-test",
        action="store_true",
        help="Basic 가중치 + R1 규칙 1회만 실행합니다. 옵션 A/B 비교표도 함께 출력.",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=12,
        metavar="MONTHS",
        help="워밍업 기간(개월). 기본값: 12",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        metavar="PATH",
        help=f"결과 저장 폴더. 기본값: {OUTPUT_DIR}",
    )
    p.add_argument(
        "--months",
        type=int,
        default=18,
        metavar="MONTHS",
        help="데이터 수집 기간(개월). 기본값: 18",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=ENTRY_THRESHOLD,
        metavar="FLOAT",
        help=(
            f"R1·R2 진입 임계값. 기본값: {ENTRY_THRESHOLD}. "
            "점수 분포가 좁으면 낮추세요 (예: --threshold 1.5). "
            "R3(부호전환)는 항상 0 기준."
        ),
    )
    p.add_argument(
        "--area4-mode",
        dest="area4_mode",
        choices=list(AREA4_MODES),
        default="trend",
        help=(
            "Area 4 변동성·위치 점수 산출 방식. "
            "'trend'(기본, A안): BB상단=강세 지속, 연속 선형 공식. "
            "'contrarian': BB하단=매수기회(평균회귀). "
            "--quick-test 에서 두 모드 모두 비교 출력됨."
        ),
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="진행 상황 출력을 최소화합니다.",
    )
    return p


# ---------------------------------------------------------------------------
# 메인 파이프라인
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args   = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    verbose = not args.quiet
    start_time = datetime.now()

    print("=" * 60)
    print("Magic Formula 백테스트 시작")
    print(f"시각: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Step 1: 데이터 수집
    # -----------------------------------------------------------------------
    print("\n[Step 1] 데이터 수집")
    raw_data = collect_all(
        months=args.months,
        use_cache=not args.no_cache,
        ticker_list=TICKER_LIST,
    )

    if not raw_data:
        print("데이터 수집 실패. 종료합니다.")
        sys.exit(1)

    kospi_df = raw_data.pop("KOSPI", None)
    stock_data = raw_data   # 종목 데이터만

    if not stock_data:
        print("종목 데이터 없음. 종료합니다.")
        sys.exit(1)

    # 실거래 구간 결정
    # 수집 시작일 + warmup_months 를 고정 cutoff로 사용 (종목별 데이터 시작일 무관)
    # → 데이터가 짧은 종목이 있어도 trade_start가 뒤로 밀리지 않음
    import pandas as _pd
    _data_start_str, _ = _date_range(args.months)
    _fixed_cutoff = _pd.to_datetime(_data_start_str, format="%Y%m%d") + _pd.DateOffset(months=args.warmup)

    trade_start = None
    trade_end   = None
    for df in stock_data.values():
        # 고정 cutoff 기준으로 실거래 구간 인덱스 추출
        t_idx = df.index[df.index >= _fixed_cutoff]
        if len(t_idx) == 0:
            continue
        ts = t_idx[0]
        te = t_idx[-1]
        trade_start = ts if trade_start is None else max(trade_start, ts)
        trade_end   = te if trade_end   is None else min(trade_end,   te)

    if trade_start is None:
        print("실거래 구간 없음. 데이터가 부족합니다.")
        sys.exit(1)

    print(f"\n실거래 구간: {trade_start.date()} ~ {trade_end.date()}")

    # -----------------------------------------------------------------------
    # Step 2: quick-test 모드
    # -----------------------------------------------------------------------
    if args.quick_test:
        print("\n[Quick Test] Basic / R1 단일 실행")
        print(f"  설정: area4_mode={args.area4_mode!r}  threshold={args.threshold}")

        import pandas as pd
        TICKER_NAMES = TICKER_NAMES_ALL

        # -------------------------------------------------------------------
        # 헬퍼: 하나의 (area4_mode, threshold) 조합으로 scored_data + 신호 집계
        # -------------------------------------------------------------------
        def _score_and_count(
            a4_mode: str,
            thresh: float,
        ) -> tuple[dict, dict[str, dict[str, int]], dict]:
            """scored_data, signal_summary, score_stats 를 반환."""
            sd: dict = {}
            for _ticker, _df in stock_data.items():
                try:
                    sd[_ticker] = compute_scores(_df, BASIC_WEIGHTS, area4_mode=a4_mode)
                except Exception as exc:
                    print(f"  {_ticker} 점수 계산 오류: {exc}")

            sig_sum: dict[str, dict[str, int]] = {}
            stats:   dict[str, dict] = {}          # {ticker: {min, max}}

            for _ticker, _sdf in sd.items():
                _tsdf = _sdf.loc[trade_start:trade_end]
                if _tsdf.empty:
                    continue
                rc: dict[str, int] = {}
                for _rule in ("R1", "R2", "R3"):
                    try:
                        _sig = entry_signals(_sdf, _rule, threshold=thresh)
                        _sig_t = _sig.reindex(_tsdf.index, fill_value=False)
                        rc[_rule] = int(_sig_t.sum())
                    except Exception:
                        rc[_rule] = -1
                sig_sum[_ticker] = rc
                _cs = _tsdf["composite_score"].dropna()
                stats[_ticker] = {"min": _cs.min(), "max": _cs.max()}

            return sd, sig_sum, stats

        # -------------------------------------------------------------------
        # ★ 옵션 A / B / Current 비교 테이블
        # -------------------------------------------------------------------
        CONFIGS = [
            ("Current",  args.area4_mode,  args.threshold),
            ("Option-A (trend/5.0)",  "trend",      5.0),
            ("Option-B (contrarian/1.5)", "contrarian", 1.5),
        ]

        # 현재 설정과 동일한 경우 중복 계산 방지
        config_results: dict[str, tuple] = {}
        for cfg_name, a4m, thr in CONFIGS:
            _key = f"{a4m}|{thr}"
            if _key not in config_results:
                config_results[_key] = _score_and_count(a4m, thr)

        print("\n" + "=" * 80)
        print("[ 비교 ] Area4 모드 × 임계값 — 실거래 구간 전체 신호 수")
        print(f"  {'설정':<30} │  R1  │  R2  │  R3  │ 신호합계")
        print("  " + "-" * 60)

        comparison_rows = []
        for cfg_name, a4m, thr in CONFIGS:
            _key = f"{a4m}|{thr}"
            _, _ss, _ = config_results[_key]
            tot_r1 = sum(v.get("R1", 0) for v in _ss.values())
            tot_r2 = sum(v.get("R2", 0) for v in _ss.values())
            tot_r3 = sum(v.get("R3", 0) for v in _ss.values())
            total  = tot_r1 + tot_r2 + tot_r3
            comparison_rows.append((cfg_name, tot_r1, tot_r2, tot_r3, total))
            print(f"  {cfg_name:<30} │ {tot_r1:>3}  │ {tot_r2:>3}  │ {tot_r3:>3}  │ {total:>6}")

        print("=" * 80)

        # 가장 신호가 많은 설정 하이라이트
        best_cfg = max(comparison_rows, key=lambda r: r[4])
        print(f"\n  → 신호 가장 많은 설정: [{best_cfg[0]}]  (합계 {best_cfg[4]}건)")
        print("     --quick-test 시뮬레이션은 Current 설정으로 실행됩니다.\n")

        # -------------------------------------------------------------------
        # ★ 진단: Current 설정 — 종목별 최근 5거래일 점수 테이블
        # -------------------------------------------------------------------
        curr_key = f"{args.area4_mode}|{args.threshold}"
        scored_data, signal_summary, _ = config_results[curr_key]

        SCORE_COLS = [
            "area1_trend", "area2_momentum", "area3_volume",
            "area4_volatility", "area5_wyckoff", "composite_score",
        ]
        COL_LABELS = ["추세", "모멘텀", "거래량", "변동·위치", "Wyckoff", "종합"]
        W_FLAG     = "wyckoff_active"

        print("=" * 72)
        print(
            f"[ 진단 ] 종목별 최근 5거래일 점수"
            f"  (area4={args.area4_mode}, threshold={args.threshold})"
        )
        print("=" * 72)

        for ticker, sdf in scored_data.items():
            name    = TICKER_NAMES.get(ticker, ticker)
            wy_flag = bool(sdf[W_FLAG].iloc[-1]) if W_FLAG in sdf.columns else False

            trade_sdf = sdf.loc[trade_start:trade_end]
            if trade_sdf.empty:
                print(f"\n  {ticker} ({name}): 실거래 구간 데이터 없음")
                continue

            last5  = trade_sdf[SCORE_COLS].tail(5)
            rc     = signal_summary.get(ticker, {})

            print(f"\n  ▶ {ticker} ({name})  [Wyckoff: {'ON' if wy_flag else 'OFF — 가중치 재분배'}]")
            header = f"  {'날짜':>10} │" + "".join(f" {lbl:>8}" for lbl in COL_LABELS)
            print(header)
            print("  " + "-" * (len(header) - 2))

            for dt, row in last5.iterrows():
                vals = "".join(f" {row[c]:>+8.2f}" for c in SCORE_COLS)
                cs   = row["composite_score"]
                flag = " ▲" if cs > args.threshold else (" ▼" if cs < -3.0 else "  ")
                print(f"  {str(dt.date()):>10} │{vals}{flag}")

            score_range = trade_sdf["composite_score"].dropna()
            print(
                f"  {'':>10}  composite 범위: "
                f"{score_range.min():+.2f} ~ {score_range.max():+.2f}  │  "
                f"진입신호 R1:{rc.get('R1',0)}  "
                f"R2:{rc.get('R2',0)}  "
                f"R3:{rc.get('R3',0)}"
            )

        # 전체 신호 요약 (Current 설정)
        print("\n" + "=" * 72)
        print(f"[ 신호 요약 ] Current 설정 — 실거래 구간 전체 진입신호")
        print(f"  {'종목':>10} │  R1  │  R2  │  R3")
        print("  " + "-" * 36)
        total_sig = {"R1": 0, "R2": 0, "R3": 0}
        for ticker, rc in signal_summary.items():
            name = TICKER_NAMES.get(ticker, ticker)[:6]
            print(
                f"  {ticker}({name:>5}) │ "
                f"{rc.get('R1',0):>3}  │ "
                f"{rc.get('R2',0):>3}  │ "
                f"{rc.get('R3',0):>3}"
            )
            for r in total_sig:
                total_sig[r] += rc.get(r, 0)
        print("  " + "-" * 36)
        print(
            f"  {'합계':>10} │ {total_sig['R1']:>3}  │ "
            f"{total_sig['R2']:>3}  │ {total_sig['R3']:>3}"
        )
        print("=" * 72)

        # -------------------------------------------------------------------
        # R1 시뮬레이션 실행 (Current 설정)
        # -------------------------------------------------------------------
        print(f"\n[Quick Test] R1 시뮬레이션 — area4={args.area4_mode}, threshold={args.threshold}")
        trades_list, equity_df = run_simulation(
            scored_data, "R1", trade_start, trade_end,
            entry_threshold=args.threshold,
        )
        tdf     = trades_to_df(trades_list)
        metrics = compute_metrics(tdf, equity_df, kospi_df, trade_start, trade_end)

        print("\n=== Quick Test 결과 (R1) ===")
        print(format_report(metrics, "Basic", "R1"))

        if not tdf.empty:
            out = output_dir / "quick_test_trades.csv"
            tdf.to_csv(out, index=False, encoding="utf-8-sig")
            print(f"\n  R1 거래 기록: {out}")

        # -------------------------------------------------------------------
        # ★ ADAPTIVE 시뮬레이션 (Basic 가중치)
        # -------------------------------------------------------------------
        print(f"\n[Quick Test] ADAPTIVE 시뮬레이션 — 동적 rolling 분류 (40거래일)")
        # v3: 정적 워밍업 분류 제거 → simulator 내부에서 진입 신호 당일 rolling 분류

        _adp_trades, _adp_eq = run_simulation(
            scored_data, "ADAPTIVE", trade_start, trade_end,
            entry_threshold=args.threshold,
        )
        _adp_tdf     = trades_to_df(_adp_trades)
        _adp_metrics = compute_metrics(
            _adp_tdf, _adp_eq, kospi_df, trade_start, trade_end
        )

        print("\n=== Quick Test 결과 (ADAPTIVE) ===")
        print(format_report(_adp_metrics, "Basic", "ADAPTIVE"))

        if not _adp_tdf.empty:
            _adp_out = output_dir / "quick_test_adaptive_trades.csv"
            _adp_tdf.to_csv(_adp_out, index=False, encoding="utf-8-sig")
            print(f"\n  ADAPTIVE 거래 기록: {_adp_out}")

        print("\n[Quick Test 완료]")
        return

    # -----------------------------------------------------------------------
    # Step 3: 전체 백테스트 (51조합 × 4규칙)
    # -----------------------------------------------------------------------
    import pandas as _pd_loop
    print(f"\n[Step 2] 백테스트 실행 ({len(generate_weight_combinations())}조합 × {len(RULES)}규칙 = {len(generate_weight_combinations())*len(RULES)}회)")
    print(f"  파라미터: area4_mode={args.area4_mode!r}  threshold={args.threshold}")

    # 점수 캐시: 조합당 1회만 compute_scores 호출
    scored_cache: dict[str, dict[str, object]] = {}

    combos = generate_weight_combinations()
    all_trades_parts = []
    summary_rows     = []

    total_runs = len(combos) * len(RULES)
    run_count  = 0

    for combo in combos:
        label   = combo["label"]
        weights = combo["weights"]

        # 점수 계산 (조합당 1회) — area4_mode 반영
        if label not in scored_cache:
            sd = {}
            for ticker, df in stock_data.items():
                try:
                    sd[ticker] = compute_scores(df, weights, area4_mode=args.area4_mode)
                except Exception as exc:
                    if verbose:
                        print(f"  [scorer] {ticker}/{label} 오류: {exc}")
            scored_cache[label] = sd

        scored_data = scored_cache[label]

        for rule in RULES:
            run_count += 1
            if verbose:
                print(f"  [{run_count:3d}/{total_runs}] {label} / {rule}", end=" ... ", flush=True)

            try:
                # ── 시뮬레이션 실행 (ADAPTIVE 포함 — 동적 rolling 분류는 simulator 내부 처리)
                trades_list, equity_df = run_simulation(
                    scored_data, rule, trade_start, trade_end,
                    entry_threshold=args.threshold,
                )
                tdf = trades_to_df(trades_list)
                metrics = compute_metrics(
                    tdf, equity_df, kospi_df, trade_start, trade_end
                )

                if not tdf.empty:
                    tdf.insert(0, "weight_label", label)
                    tdf.insert(1, "rule", rule)
                    all_trades_parts.append(tdf)

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
                    "sharpe":               metrics["sharpe"],
                    "sortino":              metrics["sortino"],
                    "calmar":               metrics["calmar"],
                    "avg_hold_days":        metrics["avg_hold_days"],
                })

                if verbose:
                    alpha_s = (
                        f"alpha={metrics['alpha_pct']:+.2f}%"
                        if metrics.get("alpha_pct") is not None
                        else "alpha=N/A"
                    )
                    print(
                        f"ret={metrics['total_return_pct']:+.2f}% | "
                        f"{alpha_s} | "
                        f"win={metrics['win_rate_pct']:.0f}% | "
                        f"N={metrics['n_trades']}"
                    )

            except Exception as exc:
                if verbose:
                    print(f"오류: {exc}")
                    traceback.print_exc()

    # -----------------------------------------------------------------------
    # Step 4: 결과 집계
    # -----------------------------------------------------------------------
    import pandas as pd

    all_trades_df = (
        pd.concat(all_trades_parts, ignore_index=True)
        if all_trades_parts else pd.DataFrame()
    )
    summary_df = pd.DataFrame(summary_rows)

    # 알파 기준 정렬
    sort_col = "alpha_pct" if (
        "alpha_pct" in summary_df.columns and summary_df["alpha_pct"].notna().any()
    ) else "total_return_pct"
    summary_df = summary_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    summary_df.insert(0, "rank", range(1, len(summary_df) + 1))

    print(f"\n[Step 3] 결과 저장 → {output_dir}")

    # trades.csv
    if not all_trades_df.empty:
        trades_path = output_dir / "trades.csv"
        all_trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")
        print(f"  저장: {trades_path}  ({len(all_trades_df):,}건)")

    # weight_ranking.md
    ranking_md = make_weight_ranking_md(summary_df)
    ranking_path = output_dir / "weight_ranking.md"
    ranking_path.write_text(ranking_md, encoding="utf-8")
    print(f"  저장: {ranking_path}")

    # backtest_report.md
    report_md = make_backtest_report_md(
        summary_df,
        all_trades_df,
        str(trade_start.date()),
        str(trade_end.date()),
    )
    report_path = output_dir / "backtest_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"  저장: {report_path}")

    # equity_curves.png (matplotlib 있을 때만)
    if _MATPLOTLIB_OK:
        print("  equity_curves.png 생성 중...")
        try:
            plot_equity_curves(
                summary_df,
                scored_cache,
                kospi_df,
                trade_start,
                trade_end,
                output_dir / "equity_curves.png",
                top_n=5,
                entry_threshold=args.threshold,
            )
        except Exception as exc:
            print(f"  equity_curves.png 생성 실패: {exc}")

    # -----------------------------------------------------------------------
    # 완료 요약
    # -----------------------------------------------------------------------
    elapsed = datetime.now() - start_time
    print(f"\n{'='*60}")
    print(f"백테스트 완료 — 소요 시간: {elapsed}")

    if not summary_df.empty:
        best = summary_df.iloc[0]
        alpha_s = (
            f"{best['alpha_pct']:+.2f}%"
            if pd.notna(best.get("alpha_pct")) else "N/A"
        )
        print(f"\n🏆 최고 조합: [{best['weight_label']} / {best['rule']}]")
        print(f"   수익률: {best['total_return_pct']:+.2f}%  |  알파: {alpha_s}")
        print(f"   승률: {best['win_rate_pct']:.1f}%  |  PF: {best['profit_factor']:.2f}")

    print(f"\n결과 폴더: {output_dir}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
