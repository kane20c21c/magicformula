#!/usr/bin/env python3
"""
scripts/update_strategy.py
--------------------------
분석 결과 → configs/active_strategy.yaml 반영 도구.

월 1회 백테스트가 끝나면 도출된 최적 조합을 yaml 에 반영해서
데일리 트랙이 즉시 사용하도록 만든다.

세 가지 사용법
--------------

1) **자동 모드** — 분석 산출물 weight_ranking.md 의 N위를 그대로 적용

    python scripts/update_strategy.py \\
        --from-ranking output/analysis/2026-05-18/weight_ranking.md \\
        --top 1

2) **수동 모드** — 가중치/규칙을 직접 명시

    python scripts/update_strategy.py \\
        --weights "trend=0.23,momentum=0.22,volume=0.27,volatility=0.10,wyckoff=0.18" \\
        --rule R3

3) **드라이런** — 차이만 출력, 파일 변경 없음

    python scripts/update_strategy.py --from-ranking ... --top 1 --dry-run

처리 단계
---------
1. 현재 yaml 로드 → 기존 strategy 출력
2. 새 weights/rule 계산 (자동/수동)
3. 비변경 필드(threshold, area4_mode, universe, strategy_id, source_analysis 등) 적용
4. 변경 항목 diff 출력
5. dry-run 이 아니면:
   - configs/history/{YYYY-MM-DD}_{strategy_id}.yaml 에 기존 사본 백업
   - configs/active_strategy.yaml 덮어씀
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

# Magic Formula 루트를 sys.path 에 등록 (magic_formula 패키지 import 가능하게)
_MAGIC_ROOT = Path(__file__).resolve().parent.parent
if str(_MAGIC_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAGIC_ROOT))

from magic_formula.config import (   # noqa: E402
    ActiveStrategy,
    load_strategy,
    dump_strategy,
    DEFAULT_CONFIG_PATH,
)

# ---------------------------------------------------------------------------
# 분석 산출물 파서
# ---------------------------------------------------------------------------

_AREAS = ("trend", "momentum", "volume", "volatility", "wyckoff")
# weight_ranking.md 의 가중치 행 패턴
# 예) | 1 | `HLLBH` | 23/22/27/10/18 | R3 | +185.93% | +102.69% | 35.2% | 3.59 | 3.59 | 193 |
_ROW_RE = re.compile(
    r"^\|\s*(?P<rank>\d+)\s*\|"             # 순위
    r"\s*`?(?P<code>[A-Z]+)`?\s*\|"          # 조합코드 HLLBH
    r"\s*(?P<w>\d+/\d+/\d+/\d+/\d+)\s*\|"    # 가중치%
    r"\s*(?P<rule>R\d|ADAPTIVE)\s*\|"        # 규칙
)


def parse_ranking_md(path: Path, top: int = 1) -> tuple[dict[str, float], str]:
    """
    weight_ranking.md 에서 top 순위의 (weights, rule) 추출.

    Returns
    -------
    (weights_dict, rule_str)
    """
    if not path.exists():
        raise FileNotFoundError(f"weight_ranking.md 없음: {path}")

    with path.open(encoding="utf-8") as f:
        for line in f:
            m = _ROW_RE.match(line)
            if not m:
                continue
            if int(m.group("rank")) != top:
                continue

            w_pcts = [int(x) for x in m.group("w").split("/")]
            if len(w_pcts) != 5:
                raise ValueError(
                    f"가중치 컬럼 형식 이상: {m.group('w')} (5개 기대)"
                )
            if sum(w_pcts) != 100:
                raise ValueError(
                    f"가중치 합 != 100%: {m.group('w')} = {sum(w_pcts)}%"
                )

            weights = {area: pct / 100.0 for area, pct in zip(_AREAS, w_pcts)}
            return weights, m.group("rule")

    raise ValueError(f"순위 {top} 행을 찾을 수 없음: {path}")


def parse_weights_arg(s: str) -> dict[str, float]:
    """
    "trend=0.23,momentum=0.22,..." 형식 → dict.
    """
    out: dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--weights 항목 형식 오류: {part!r} (예: trend=0.23)")
        k, v = part.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


# ---------------------------------------------------------------------------
# diff 표시
# ---------------------------------------------------------------------------

def print_diff(old: ActiveStrategy, new: ActiveStrategy) -> bool:
    """
    두 전략의 차이를 깔끔하게 출력. **실질 변경** 여부 반환.

    실질 변경 판정에는 last_updated / source_analysis 같은 메타데이터는 제외.
    이 둘은 매번 갱신되어도 전략 내용은 그대로일 수 있기 때문.
    """
    changed = False

    def _line(label: str, old_v, new_v, *, count_change: bool = True) -> None:
        nonlocal changed
        diff = (old_v != new_v)
        if diff:
            print(f"  ~ {label:18}: {old_v}  →  {new_v}")
            if count_change:
                changed = True
        else:
            print(f"    {label:18}: {old_v}")

    print("\n[diff]")
    _line("strategy_id",      old.strategy_id,     new.strategy_id)
    _line("rule",             old.rule,            new.rule)
    _line("area4_mode",       old.area4_mode,      new.area4_mode)
    _line("threshold",        old.threshold,       new.threshold)
    _line("universe",         old.universe,        new.universe)
    # 메타데이터는 변경 카운트에서 제외 (실질 전략 변경이 아님)
    _line("last_updated",     old.last_updated,    new.last_updated,    count_change=False)
    _line("source_analysis",  old.source_analysis, new.source_analysis, count_change=False)

    # weights 비교
    print(f"\n    weights:")
    for k in _AREAS:
        ov = old.weights.get(k, 0.0)
        nv = new.weights.get(k, 0.0)
        marker = "  ~" if abs(ov - nv) > 1e-9 else "   "
        if abs(ov - nv) > 1e-9:
            changed = True
        print(f"    {marker} {k:12}: {ov:.4f}  →  {nv:.4f}")

    return changed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python scripts/update_strategy.py",
        description="분석 결과 → configs/active_strategy.yaml 반영 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 1) 자동 — weight_ranking.md 1위 적용
  python scripts/update_strategy.py \\
    --from-ranking output/analysis/2026-05-18/weight_ranking.md

  # 2) 수동 — 가중치/규칙 직접
  python scripts/update_strategy.py \\
    --weights "trend=0.23,momentum=0.22,volume=0.27,volatility=0.10,wyckoff=0.18" \\
    --rule R3

  # 3) 드라이런 — 차이만 출력
  python scripts/update_strategy.py --from-ranking ... --dry-run
""",
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--from-ranking",
        type=Path,
        metavar="PATH",
        help="weight_ranking.md 에서 자동 추출 (자동 모드)",
    )
    src.add_argument(
        "--weights",
        type=str,
        metavar="K=V,...",
        help='가중치 직접 명시 (예: "trend=0.23,momentum=0.22,...")',
    )

    p.add_argument(
        "--top",
        type=int,
        default=1,
        metavar="N",
        help="--from-ranking 사용 시 N번째 순위 선택 (기본 1)",
    )
    p.add_argument(
        "--rule",
        choices=["R1", "R2", "R3", "ADAPTIVE"],
        help="진입 규칙 (--from-ranking 가 자동 채움. 명시 시 덮어씀)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        help="R1·R2 진입 임계값 (생략 시 기존 yaml 값 유지)",
    )
    p.add_argument(
        "--area4-mode",
        dest="area4_mode",
        choices=["trend", "contrarian"],
        help="Area 4 산출 방식 (생략 시 기존 값 유지)",
    )
    p.add_argument(
        "--universe",
        choices=["core_all", "core_excl_split", "core_59", "core_57"],
        help="종목 universe (생략 시 기존 값 유지)",
    )
    p.add_argument(
        "--strategy-id",
        dest="strategy_id",
        type=str,
        help="새 전략 ID (생략 시 기존 ID 유지)",
    )
    p.add_argument(
        "--source-analysis",
        dest="source_analysis",
        type=str,
        help="감사 추적용 분석 산출물 경로. --from-ranking 사용 시 자동 채움",
    )
    p.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"대상 yaml 경로 (기본 {DEFAULT_CONFIG_PATH})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 파일은 안 바꾸고 차이만 출력",
    )
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="history 백업 생략 (권장하지 않음)",
    )

    return p


def main() -> int:
    args = build_parser().parse_args()

    # 1) 현재 yaml 로드
    try:
        old = load_strategy(args.config_path)
    except FileNotFoundError as e:
        print(f"[update_strategy] {e}", file=sys.stderr)
        return 1

    print("=== 현재 전략 ===")
    print(f"  {old.summary()}")

    # 2) 새 weights/rule 결정
    new_weights: dict[str, float]
    new_rule:    str

    if args.from_ranking is not None:
        new_weights, parsed_rule = parse_ranking_md(args.from_ranking, args.top)
        new_rule = args.rule or parsed_rule
        src_path = args.source_analysis or str(args.from_ranking)
        print(f"\n[자동] {args.from_ranking.name} #{args.top} → rule={parsed_rule}")
    else:
        new_weights = parse_weights_arg(args.weights)
        if not args.rule:
            print(
                "[update_strategy] --weights 와 함께 --rule 도 명시해야 합니다.",
                file=sys.stderr,
            )
            return 1
        new_rule = args.rule
        src_path = args.source_analysis

    # 3) 새 ActiveStrategy 생성 (비변경 필드는 기존 값 유지)
    today_str = datetime.today().strftime("%Y-%m-%d")
    new = ActiveStrategy(
        strategy_id     = args.strategy_id     or old.strategy_id,
        weights         = new_weights,
        rule            = new_rule,
        area4_mode      = args.area4_mode      or old.area4_mode,
        threshold       = args.threshold if args.threshold is not None else old.threshold,
        universe        = args.universe        or old.universe,
        last_updated    = today_str,
        source_analysis = src_path or old.source_analysis,
    )

    # 4) diff 출력
    changed = print_diff(old, new)

    # 5) 검증 (dry-run 도 포함 — 잘못된 값으로 적용하지 않도록 미리 확인)
    try:
        new.validate()
    except ValueError as e:
        print(f"\n[update_strategy] 검증 실패: {e}", file=sys.stderr)
        return 1

    if not changed:
        print("\n  실질 변경 사항 없음. 파일을 건드리지 않습니다.")
        return 0

    if args.dry_run:
        print("\n[dry-run] 검증 통과. 실제 파일은 변경되지 않았습니다.")
        return 0

    # 6) 저장 (history 백업 포함)
    out = dump_strategy(
        new,
        args.config_path,
        backup_history=not args.no_backup,
    )

    print(f"\n✅ 저장 완료: {out}")
    if not args.no_backup:
        print(f"   (기존 yaml 은 configs/history/ 에 백업됨)")
    print(f"\n다음 데일리 실행부터 새 전략이 적용됩니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
