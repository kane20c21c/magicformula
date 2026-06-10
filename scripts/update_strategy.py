#!/usr/bin/env python3
"""
scripts/update_strategy.py
--------------------------
분석 결과 → configs/active_strategy.yaml (v2_combined) 반영 도구.

월 1회 백테스트가 끝나면 도출된 최적 조합(4영역 가중치 + 임계값)을
yaml 에 반영해서 데일리 트랙이 즉시 사용하도록 만든다.

★ v2 가드: 대상 yaml 의 system_version 이 'v2_combined' 가 아니면 중단한다.
  (v1 스키마는 2026-06-10 폐기 — 구 yaml 을 덮어쓰는 사고 방지)

세 가지 사용법
--------------

1) **자동 모드** — 분석 산출물 weight_ranking.md 의 N위를 그대로 적용

    python scripts/update_strategy.py \\
        --from-ranking output/analysis/2026-06-15/weight_ranking.md \\
        --top 1

2) **수동 모드** — 가중치/임계값을 직접 명시

    python scripts/update_strategy.py \\
        --weights "trend=0.2,momentum=0.2,volume=0.0,volatility=0.6" \\
        --threshold 6.0

3) **드라이런** — 차이만 출력, 파일 변경 없음

    python scripts/update_strategy.py --from-ranking ... --dry-run

처리 단계
---------
1. 현재 yaml 로드 + v2 검증 → 기존 전략 출력
2. 새 weights/threshold 결정 (자동/수동)
3. 변경 항목 diff 출력 → 검증
4. dry-run 이 아니면 configs/history/ 백업 후 해당 필드만 갱신
   (ruamel.yaml 설치 시 주석 보존, 미설치 시 주석 소실 경고)
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

# Magic Formula 루트를 sys.path 에 등록
_MAGIC_ROOT = Path(__file__).resolve().parent.parent
if str(_MAGIC_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAGIC_ROOT))

from magic_formula.config import (   # noqa: E402
    DEFAULT_CONFIG_PATH,
    REQUIRED_WEIGHT_KEYS,
    StrategyV2,
    load_strategy,
    update_strategy_fields,
)

_AREAS = ("trend", "momentum", "volume", "volatility")

# ---------------------------------------------------------------------------
# 분석 산출물 파서 — weight_ranking.md (v2 포맷)
# ---------------------------------------------------------------------------
# 예) | 1 | T20/M20/Vu0/Va60 | 6.0 | +3.21% | +51.20% | ... |
_ROW_RE = re.compile(
    r"^\|\s*(?P<rank>\d+)\s*\|"
    r"\s*T(?P<t>\d+)/M(?P<m>\d+)/Vu(?P<vu>\d+)/Va(?P<va>\d+)\s*\|"
    r"\s*(?P<thr>[\d.]+)\s*\|"
)


def parse_ranking_md(path: Path, top: int = 1) -> tuple[dict[str, float], float]:
    """
    weight_ranking.md 에서 top 순위의 (weights, threshold) 추출.
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

            pcts = [int(m.group(k)) for k in ("t", "m", "vu", "va")]
            if sum(pcts) != 100:
                raise ValueError(f"가중치 합 != 100%: {pcts} = {sum(pcts)}%")
            weights = {area: pct / 100.0 for area, pct in zip(_AREAS, pcts)}
            return weights, float(m.group("thr"))

    raise ValueError(
        f"순위 {top} 행을 찾을 수 없음: {path}\n"
        "  (v1 포맷 ranking 파일이면 지원 종료 — v2 분석을 다시 실행하세요)"
    )


def parse_weights_arg(s: str) -> dict[str, float]:
    """ "trend=0.2,momentum=0.2,..." 형식 → dict. """
    out: dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--weights 항목 형식 오류: {part!r} (예: trend=0.2)")
        k, v = part.split("=", 1)
        out[k.strip()] = float(v.strip())
    missing = REQUIRED_WEIGHT_KEYS - set(out)
    extra   = set(out) - REQUIRED_WEIGHT_KEYS
    if missing or extra:
        raise ValueError(
            f"--weights 키 불일치  missing={missing or '∅'}  extra={extra or '∅'}  "
            f"(필요: {sorted(REQUIRED_WEIGHT_KEYS)})"
        )
    return out


# ---------------------------------------------------------------------------
# diff 표시
# ---------------------------------------------------------------------------

def print_diff(old: StrategyV2, new_weights: dict, new_threshold: float,
               new_id: str, new_src: str | None) -> bool:
    """기존 전략과의 차이 출력. 실질 변경 여부 반환."""
    changed = False

    def _line(label: str, old_v, new_v, *, count: bool = True) -> None:
        nonlocal changed
        if old_v != new_v:
            print(f"  ~ {label:18}: {old_v}  →  {new_v}")
            if count:
                changed = True
        else:
            print(f"    {label:18}: {old_v}")

    print("\n[diff]")
    _line("strategy_id", old.strategy_id, new_id)
    _line("threshold",   old.threshold,   new_threshold)
    _line("source_analysis", old.source_analysis, new_src or old.source_analysis,
          count=False)

    print("\n    weights:")
    for k in _AREAS:
        ov = old.weights.get(k, 0.0)
        nv = new_weights.get(k, 0.0)
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
        description="분석 결과 → configs/active_strategy.yaml (v2_combined) 반영 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 1) 자동 — weight_ranking.md 1위 적용
  python scripts/update_strategy.py \\
    --from-ranking output/analysis/2026-06-15/weight_ranking.md

  # 2) 수동 — 가중치/임계값 직접
  python scripts/update_strategy.py \\
    --weights "trend=0.2,momentum=0.2,volume=0.0,volatility=0.6" \\
    --threshold 6.0

  # 3) 드라이런 — 차이만 출력
  python scripts/update_strategy.py --from-ranking ... --dry-run
""",
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-ranking", type=Path, metavar="PATH",
                     help="weight_ranking.md 에서 자동 추출 (자동 모드)")
    src.add_argument("--weights", type=str, metavar="K=V,...",
                     help='가중치 직접 명시 (예: "trend=0.2,momentum=0.2,volume=0.0,volatility=0.6")')

    p.add_argument("--top", type=int, default=1, metavar="N",
                   help="--from-ranking 사용 시 N번째 순위 선택 (기본 1)")
    p.add_argument("--threshold", type=float,
                   help="진입 임계값 (--from-ranking 가 자동 채움. 명시 시 덮어씀)")
    p.add_argument("--strategy-id", dest="strategy_id", type=str,
                   help="새 전략 ID (생략 시 기존 ID 유지)")
    p.add_argument("--source-analysis", dest="source_analysis", type=str,
                   help="감사 추적용 분석 산출물 경로. --from-ranking 사용 시 자동 채움")
    p.add_argument("--config", dest="config_path", type=Path,
                   default=DEFAULT_CONFIG_PATH,
                   help=f"대상 yaml 경로 (기본 {DEFAULT_CONFIG_PATH})")
    p.add_argument("--dry-run", action="store_true",
                   help="실제 파일은 안 바꾸고 차이만 출력")
    p.add_argument("--no-backup", action="store_true",
                   help="history 백업 생략 (권장하지 않음)")
    return p


def main() -> int:
    args = build_parser().parse_args()

    # 1) 현재 yaml 로드 — load_strategy 가 v2 가드 겸함
    #    (v1 스키마면 ValueError: system_version != v2_combined)
    try:
        old = load_strategy(args.config_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"[update_strategy] 중단: {e}", file=sys.stderr)
        return 1

    print("=== 현재 전략 ===")
    print(f"  {old.summary()}")

    # 2) 새 weights/threshold 결정
    if args.from_ranking is not None:
        new_weights, parsed_thr = parse_ranking_md(args.from_ranking, args.top)
        new_threshold = args.threshold if args.threshold is not None else parsed_thr
        src_path = args.source_analysis or str(args.from_ranking)
        print(f"\n[자동] {args.from_ranking.name} #{args.top} → thr={parsed_thr}")
    else:
        new_weights = parse_weights_arg(args.weights)
        new_threshold = (args.threshold if args.threshold is not None
                         else old.threshold)
        src_path = args.source_analysis

    new_id = args.strategy_id or old.strategy_id

    # 3) diff + 검증 (합계는 parse 단계 + config.validate 에서 이중 확인)
    total = sum(new_weights.values())
    if abs(total - 1.0) > 1e-6:
        print(f"\n[update_strategy] 검증 실패: 가중치 합 {total:.6f} != 1.0",
              file=sys.stderr)
        return 1

    changed = print_diff(old, new_weights, new_threshold, new_id, src_path)

    if not changed:
        print("\n  실질 변경 사항 없음. 파일을 건드리지 않습니다.")
        return 0

    if args.dry_run:
        print("\n[dry-run] 검증 통과. 실제 파일은 변경되지 않았습니다.")
        return 0

    # 4) 저장 — 해당 필드만 갱신 (레짐/게이트/trading 등 나머지 구조 보존)
    out = update_strategy_fields(
        args.config_path,
        weights=new_weights,
        threshold=new_threshold,
        strategy_id=new_id,
        source_analysis=src_path,
        last_updated=datetime.today().strftime("%Y-%m-%d"),
        backup_history=not args.no_backup,
    )

    print(f"\n✅ 저장 완료: {out}")
    if not args.no_backup:
        print("   (기존 yaml 은 configs/history/ 에 백업됨)")
    print("\n다음 데일리 실행부터 새 전략이 적용됩니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
