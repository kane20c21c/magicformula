"""
config.py
---------
configs/active_strategy.yaml 로더/덤퍼.

데일리(scripts/daily_signal.py)와 분석(src/optimizer 등) 모두 이 모듈을 거쳐
전략 설정을 읽고 씁니다. 하드코딩 금지 — 모든 설정은 yaml 에서.

공개 API
--------
- load_strategy(path=None)  → ActiveStrategy
- dump_strategy(strategy, path=None, *, backup_history=True)
- ActiveStrategy.from_dict(d) / .to_dict()

검증 규칙
---------
- weights 키 5개 정확히 일치: trend / momentum / volume / volatility / wyckoff
- weights 합계 = 1.0 ± 1e-6
- rule ∈ {R1, R2, R3, ADAPTIVE}
- area4_mode ∈ {trend, contrarian}
- threshold: float (보통 -10 ~ +10)
- universe: 문자열 식별자 (해석은 _vault.get_universe — P3 에서 추가)

예시
----
    from magic_formula.config import load_strategy
    cfg = load_strategy()
    print(cfg.weights)       # {'trend': 0.30, ...}
    print(cfg.rule)          # 'R1'
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError as e:
    raise ImportError(
        "PyYAML 필요. 설치: pip install pyyaml --break-system-packages"
    ) from e

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

# Magic Formula 프로젝트 루트 (src/ 의 부모)
_PROJ_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG_PATH = _PROJ_ROOT / "configs" / "active_strategy.yaml"
HISTORY_DIR         = _PROJ_ROOT / "configs" / "history"

_REQUIRED_WEIGHT_KEYS = {"trend", "momentum", "volume", "volatility", "wyckoff"}
_ALLOWED_RULES        = {"R1", "R2", "R3", "ADAPTIVE"}
_ALLOWED_AREA4_MODES  = {"trend", "contrarian"}

_WEIGHT_SUM_TOL = 1e-6


# ---------------------------------------------------------------------------
# 데이터 모델
# ---------------------------------------------------------------------------

@dataclass
class ActiveStrategy:
    """active_strategy.yaml 의 in-memory 표현."""

    strategy_id:     str
    weights:         dict[str, float]
    rule:            str
    area4_mode:      str
    threshold:       float
    universe:        str
    last_updated:    Optional[str]  = None     # 'YYYY-MM-DD'
    source_analysis: Optional[str]  = None     # 보고서 경로 (감사 추적)

    # -----------------------------------------------------------------------
    # 직렬화
    # -----------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "ActiveStrategy":
        """dict (yaml 파싱 결과) → ActiveStrategy."""
        _validate_dict(d)

        # last_updated 가 date 객체로 파싱된 경우 문자열로 정규화
        lu = d.get("last_updated")
        if isinstance(lu, (date, datetime)):
            lu = lu.strftime("%Y-%m-%d")

        return cls(
            strategy_id     = str(d["strategy_id"]),
            weights         = {k: float(v) for k, v in d["weights"].items()},
            rule            = str(d["rule"]),
            area4_mode      = str(d["area4_mode"]),
            threshold       = float(d["threshold"]),
            universe        = str(d["universe"]),
            last_updated    = lu,
            source_analysis = d.get("source_analysis"),
        )

    def to_dict(self) -> dict:
        """ActiveStrategy → dict (yaml 덤프용)."""
        return {
            "strategy_id":     self.strategy_id,
            "last_updated":    self.last_updated,
            "source_analysis": self.source_analysis,
            "weights":         dict(self.weights),
            "rule":            self.rule,
            "area4_mode":      self.area4_mode,
            "threshold":       self.threshold,
            "universe":        self.universe,
        }

    # -----------------------------------------------------------------------
    # 검증
    # -----------------------------------------------------------------------
    def validate(self) -> None:
        """
        스키마 검증을 강제 실행. 잘못된 값이면 ValueError.

        dump_strategy() 가 저장 직전 자동으로 호출하지만,
        dry-run 같은 비저장 흐름에서 사전 검증하고 싶을 때 직접 호출한다.
        """
        _validate_dict(self.to_dict())

    # -----------------------------------------------------------------------
    # 유틸
    # -----------------------------------------------------------------------
    def summary(self) -> str:
        """사람이 읽기 좋은 한 줄 요약."""
        w = self.weights
        return (
            f"[{self.strategy_id}] rule={self.rule} thr={self.threshold} "
            f"area4={self.area4_mode} | "
            f"T={w['trend']:.2f} M={w['momentum']:.2f} V={w['volume']:.2f} "
            f"Vo={w['volatility']:.2f} W={w['wyckoff']:.2f} "
            f"(updated {self.last_updated})"
        )


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------

def _validate_dict(d: dict) -> None:
    """yaml dict 가 active_strategy 스키마를 따르는지 검증."""

    if not isinstance(d, dict):
        raise ValueError(f"yaml 최상위가 dict 가 아님: {type(d).__name__}")

    # 필수 키
    for k in ("strategy_id", "weights", "rule", "area4_mode", "threshold", "universe"):
        if k not in d:
            raise ValueError(f"필수 키 누락: {k!r}")

    # weights 구조
    w = d["weights"]
    if not isinstance(w, dict):
        raise ValueError(f"'weights' 가 dict 아님: {type(w).__name__}")

    keys = set(w.keys())
    missing = _REQUIRED_WEIGHT_KEYS - keys
    extra   = keys - _REQUIRED_WEIGHT_KEYS
    if missing or extra:
        raise ValueError(
            f"'weights' 키 불일치  missing={missing or '∅'}  extra={extra or '∅'}  "
            f"(필요: {sorted(_REQUIRED_WEIGHT_KEYS)})"
        )

    # weights 값 타입 + 합계
    try:
        vals = {k: float(v) for k, v in w.items()}
    except (TypeError, ValueError) as e:
        raise ValueError(f"'weights' 값을 float 로 변환 실패: {e}") from e

    total = sum(vals.values())
    if abs(total - 1.0) > _WEIGHT_SUM_TOL:
        raise ValueError(
            f"'weights' 합계가 1.0 아님: {total:.6f}  (허용오차 {_WEIGHT_SUM_TOL})"
        )

    # rule / area4_mode
    if d["rule"] not in _ALLOWED_RULES:
        raise ValueError(
            f"'rule'={d['rule']!r} 허용값 아님. 허용: {sorted(_ALLOWED_RULES)}"
        )
    if d["area4_mode"] not in _ALLOWED_AREA4_MODES:
        raise ValueError(
            f"'area4_mode'={d['area4_mode']!r} 허용값 아님. "
            f"허용: {sorted(_ALLOWED_AREA4_MODES)}"
        )

    # threshold — 숫자
    try:
        float(d["threshold"])
    except (TypeError, ValueError) as e:
        raise ValueError(f"'threshold' 가 숫자 아님: {e}") from e


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def load_strategy(path: str | Path | None = None) -> ActiveStrategy:
    """
    active_strategy.yaml 을 읽어 ActiveStrategy 객체로 반환한다.

    Parameters
    ----------
    path : 명시 안 하면 configs/active_strategy.yaml 사용
    """
    path = Path(path) if path else DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"active_strategy.yaml 없음: {path}\n"
            f"  → configs/ 디렉토리에 yaml 을 배치하거나 path 인자로 명시하세요."
        )

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return ActiveStrategy.from_dict(data)


def dump_strategy(
    strategy: ActiveStrategy,
    path: str | Path | None = None,
    *,
    backup_history: bool = True,
) -> Path:
    """
    ActiveStrategy 객체를 yaml 파일로 저장한다.

    Parameters
    ----------
    strategy       : 저장할 전략
    path           : 명시 안 하면 configs/active_strategy.yaml
    backup_history : True 면 저장 직전 기존 파일을 configs/history/ 에 복사
                     파일명: {YYYY-MM-DD}_{strategy_id}.yaml (충돌시 -N suffix)

    Returns
    -------
    저장된 yaml 의 Path
    """
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    # 저장 전 검증 (잘못된 객체로 yaml 덮어쓰기 방지)
    _validate_dict(strategy.to_dict())

    # 1) 기존 파일을 history 에 백업
    if backup_history and path.exists():
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        # 기존 yaml 내의 last_updated 를 백업 파일명에 반영
        with path.open(encoding="utf-8") as f:
            old = yaml.safe_load(f) or {}

        old_id   = old.get("strategy_id", "unknown")
        old_date = old.get("last_updated")
        if isinstance(old_date, (date, datetime)):
            old_date = old_date.strftime("%Y-%m-%d")
        if not old_date:
            old_date = datetime.today().strftime("%Y-%m-%d")

        backup_name = f"{old_date}_{old_id}.yaml"
        backup_path = HISTORY_DIR / backup_name

        # 충돌 시 suffix
        n = 1
        while backup_path.exists():
            backup_path = HISTORY_DIR / f"{old_date}_{old_id}-{n}.yaml"
            n += 1

        shutil.copy2(path, backup_path)

    # 2) 새 파일 작성
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            strategy.to_dict(),
            f,
            allow_unicode=True,
            sort_keys=False,    # 사람이 적은 순서 유지
            default_flow_style=False,
        )

    return path


# ---------------------------------------------------------------------------
# CLI (단독 실행 시 현재 설정 표시)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    try:
        cfg = load_strategy()
        print("=== active_strategy.yaml ===")
        print(cfg.summary())
        print()
        print("dump preview:")
        print(yaml.safe_dump(cfg.to_dict(), allow_unicode=True, sort_keys=False))
    except Exception as e:
        print(f"[config] 로드 실패: {e}", file=sys.stderr)
        sys.exit(1)
