"""
config.py
---------
configs/active_strategy.yaml (v2_combined) 로더 / 검증 / 부분 갱신.

2026-06-10 v2 단일화: v1 스키마(ActiveStrategy: 5가중치 + R1/R2/R3/ADAPTIVE)
는 완전 폐기. 이 모듈은 v2_combined 스키마만 다룬다.
(v1 정본 백업: configs/active_strategy_v1.yaml — 참고용으로만 보존)

v2 yaml 구조 (운영 정본)
------------------------
    strategy_id / last_updated / source_analysis / system_version: v2_combined
    scoring:
      weights: {trend, momentum, volume, volatility}   # 합 1.0
      threshold / candidate_threshold / universe
      regimes: {trend: {...}, volume_volatility: {...}}
      gate: {enabled, exclude_phases, ...}
    trading:
      entry: {rule: threshold_breakout, position_size, ...}
      exit:  {stop_loss, time_stop, hold_if_profit, force_exit}

공개 API
--------
- load_strategy(path=None)  → StrategyV2
- update_strategy_fields(path=None, *, weights=..., threshold=..., ...)
  → 지정 필드만 yaml 에 반영 (history 백업 포함, 주석 보존은 ruamel 설치 시)

예시
----
    from magic_formula.config import load_strategy
    cfg = load_strategy()
    print(cfg.weights)       # {'trend': 0.2, 'momentum': 0.2, ...}
    print(cfg.threshold)     # 6.0
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
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

_PROJ_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG_PATH = _PROJ_ROOT / "configs" / "active_strategy.yaml"
HISTORY_DIR         = _PROJ_ROOT / "configs" / "history"

SYSTEM_VERSION = "v2_combined"

REQUIRED_WEIGHT_KEYS = {"trend", "momentum", "volume", "volatility"}
_WEIGHT_SUM_TOL = 1e-6

DEFAULT_POSITION_SIZE = 10_000_000   # 종목당 투입 자본 (원) — yaml trading.entry 정본


# ---------------------------------------------------------------------------
# 데이터 모델
# ---------------------------------------------------------------------------

@dataclass
class StrategyV2:
    """active_strategy.yaml (v2_combined) 의 in-memory 표현."""

    strategy_id:          str
    weights:              dict[str, float]          # 4영역, 합 1.0
    threshold:            float
    candidate_threshold:  float
    universe:             str
    gate_enabled:         bool
    gate_exclude_phases:  tuple[str, ...]
    position_size:        int
    regimes:              dict = field(default_factory=dict)
    last_updated:         Optional[str] = None      # 'YYYY-MM-DD'
    source_analysis:      Optional[str] = None
    system_version:       str = SYSTEM_VERSION
    raw:                  dict = field(default_factory=dict, repr=False)

    # -----------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "StrategyV2":
        """dict (yaml 파싱 결과) → StrategyV2. 스키마 검증 포함."""
        validate_dict(d)

        sc = d["scoring"]
        gate = sc.get("gate", {}) or {}
        trading = d.get("trading", {}) or {}
        entry = trading.get("entry", {}) or {}

        lu = d.get("last_updated")
        if isinstance(lu, (date, datetime)):
            lu = lu.strftime("%Y-%m-%d")

        threshold = float(sc["threshold"])
        return cls(
            strategy_id         = str(d["strategy_id"]),
            weights             = {k: float(v) for k, v in sc["weights"].items()},
            threshold           = threshold,
            candidate_threshold = float(sc.get("candidate_threshold", threshold)),
            universe            = str(sc.get("universe", "core_excl_split")),
            gate_enabled        = bool(gate.get("enabled", True)),
            gate_exclude_phases = tuple(gate.get("exclude_phases", ["Markdown"])),
            position_size       = int(entry.get("position_size", DEFAULT_POSITION_SIZE)),
            regimes             = dict(sc.get("regimes", {}) or {}),
            last_updated        = lu,
            source_analysis     = d.get("source_analysis"),
            system_version      = str(d.get("system_version", "")).strip(),
            raw                 = d,
        )

    def validate(self) -> None:
        """스키마 검증 강제 실행. 잘못된 값이면 ValueError."""
        validate_dict(self.raw if self.raw else _to_minimal_dict(self))

    def summary(self) -> str:
        """사람이 읽기 좋은 한 줄 요약."""
        w = self.weights
        return (
            f"[{self.strategy_id}] v2_combined thr={self.threshold} "
            f"(cand={self.candidate_threshold}) gate={'ON' if self.gate_enabled else 'OFF'} | "
            f"T={w['trend']:.2f} M={w['momentum']:.2f} Vu={w['volume']:.2f} "
            f"Va={w['volatility']:.2f} | universe={self.universe} "
            f"(updated {self.last_updated})"
        )


def _to_minimal_dict(s: StrategyV2) -> dict:
    """StrategyV2 → 검증 가능한 최소 dict (raw 가 없을 때)."""
    return {
        "strategy_id":    s.strategy_id,
        "system_version": s.system_version,
        "scoring": {
            "weights":             dict(s.weights),
            "threshold":           s.threshold,
            "candidate_threshold": s.candidate_threshold,
            "universe":            s.universe,
            "gate": {
                "enabled":        s.gate_enabled,
                "exclude_phases": list(s.gate_exclude_phases),
            },
        },
        "trading": {"entry": {"position_size": s.position_size}},
    }


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------

def validate_dict(d: dict) -> None:
    """yaml dict 가 v2_combined 스키마를 따르는지 검증."""

    if not isinstance(d, dict):
        raise ValueError(f"yaml 최상위가 dict 가 아님: {type(d).__name__}")

    sv = str(d.get("system_version", "")).strip()
    if sv != SYSTEM_VERSION:
        raise ValueError(
            f"system_version={sv!r} — {SYSTEM_VERSION!r} 만 지원합니다. "
            "(v1 스키마는 2026-06-10 폐기. configs/active_strategy_v1.yaml 참고)"
        )

    for k in ("strategy_id", "scoring"):
        if k not in d:
            raise ValueError(f"필수 키 누락: {k!r}")

    sc = d["scoring"]
    if not isinstance(sc, dict):
        raise ValueError(f"'scoring' 이 dict 아님: {type(sc).__name__}")

    if "weights" not in sc:
        raise ValueError("필수 키 누락: 'scoring.weights'")
    if "threshold" not in sc:
        raise ValueError("필수 키 누락: 'scoring.threshold'")

    w = sc["weights"]
    if not isinstance(w, dict):
        raise ValueError(f"'scoring.weights' 가 dict 아님: {type(w).__name__}")

    keys = set(w.keys())
    missing = REQUIRED_WEIGHT_KEYS - keys
    extra   = keys - REQUIRED_WEIGHT_KEYS
    if missing or extra:
        raise ValueError(
            f"'scoring.weights' 키 불일치  missing={missing or '∅'}  extra={extra or '∅'}  "
            f"(필요: {sorted(REQUIRED_WEIGHT_KEYS)})"
        )

    try:
        vals = {k: float(v) for k, v in w.items()}
    except (TypeError, ValueError) as e:
        raise ValueError(f"'scoring.weights' 값을 float 로 변환 실패: {e}") from e

    total = sum(vals.values())
    if abs(total - 1.0) > _WEIGHT_SUM_TOL:
        raise ValueError(
            f"'scoring.weights' 합계가 1.0 아님: {total:.6f}  (허용오차 {_WEIGHT_SUM_TOL})"
        )

    for key in ("threshold", "candidate_threshold"):
        if key in sc:
            try:
                float(sc[key])
            except (TypeError, ValueError) as e:
                raise ValueError(f"'scoring.{key}' 가 숫자 아님: {e}") from e

    # position_size (선택 — 있으면 양의 정수)
    ps = ((d.get("trading") or {}).get("entry") or {}).get("position_size")
    if ps is not None:
        try:
            if int(ps) <= 0:
                raise ValueError("position_size 는 양수여야 함")
        except (TypeError, ValueError) as e:
            raise ValueError(f"'trading.entry.position_size' 오류: {e}") from e


# ---------------------------------------------------------------------------
# 공개 API — 로드
# ---------------------------------------------------------------------------

def load_strategy(path: str | Path | None = None) -> StrategyV2:
    """
    active_strategy.yaml (v2_combined) 을 읽어 StrategyV2 로 반환한다.

    v1 스키마 yaml 이면 ValueError (지원 종료 안내 포함).
    """
    path = Path(path) if path else DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"active_strategy.yaml 없음: {path}\n"
            f"  → configs/ 디렉토리에 yaml 을 배치하거나 path 인자로 명시하세요."
        )

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return StrategyV2.from_dict(data)


# ---------------------------------------------------------------------------
# 공개 API — 부분 갱신 (update_strategy.py 가 사용)
# ---------------------------------------------------------------------------

def _backup_to_history(path: Path) -> Path | None:
    """기존 yaml 을 configs/history/{date}_{id}.yaml 로 복사."""
    if not path.exists():
        return None
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    with path.open(encoding="utf-8") as f:
        old = yaml.safe_load(f) or {}
    old_id   = old.get("strategy_id", "unknown")
    old_date = old.get("last_updated")
    if isinstance(old_date, (date, datetime)):
        old_date = old_date.strftime("%Y-%m-%d")
    if not old_date:
        old_date = datetime.today().strftime("%Y-%m-%d")
    backup_path = HISTORY_DIR / f"{old_date}_{old_id}.yaml"
    n = 1
    while backup_path.exists():
        backup_path = HISTORY_DIR / f"{old_date}_{old_id}-{n}.yaml"
        n += 1
    shutil.copy2(path, backup_path)
    return backup_path


def update_strategy_fields(
    path: str | Path | None = None,
    *,
    weights:         dict[str, float] | None = None,
    threshold:       float | None = None,
    strategy_id:     str | None = None,
    source_analysis: str | None = None,
    last_updated:    str | None = None,
    backup_history:  bool = True,
) -> Path:
    """
    v2 yaml 의 지정 필드만 갱신한다. 나머지 구조(레짐/게이트/trading 등)는 보존.

    주석 보존: ruamel.yaml 이 설치되어 있으면 주석까지 그대로 유지하며 갱신,
    없으면 PyYAML safe_dump 로 재작성 (데이터는 보존되지만 주석은 사라짐 —
    이 경우 경고 출력. 원본 주석은 history 백업에 남는다).

    Returns
    -------
    저장된 yaml 의 Path
    """
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"yaml 없음: {path}")

    # 갱신 후 결과를 미리 검증 (PyYAML 기준)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    validate_dict(data)   # 대상이 v2 정본인지 가드

    sc = data.setdefault("scoring", {})
    if weights is not None:
        sc["weights"] = {k: float(v) for k, v in weights.items()}
    if threshold is not None:
        sc["threshold"] = float(threshold)
    if strategy_id is not None:
        data["strategy_id"] = strategy_id
    if source_analysis is not None:
        data["source_analysis"] = source_analysis
    data["last_updated"] = last_updated or datetime.today().strftime("%Y-%m-%d")

    validate_dict(data)   # 갱신 결과 재검증

    if backup_history:
        _backup_to_history(path)

    # ── 저장: ruamel(주석 보존) → PyYAML 폴백 ──
    try:
        from ruamel.yaml import YAML   # type: ignore

        ry = YAML()
        ry.preserve_quotes = True
        with path.open(encoding="utf-8") as f:
            rdata = ry.load(f)

        rsc = rdata["scoring"]
        if weights is not None:
            for k, v in weights.items():
                rsc["weights"][k] = float(v)
        if threshold is not None:
            rsc["threshold"] = float(threshold)
        if strategy_id is not None:
            rdata["strategy_id"] = strategy_id
        if source_analysis is not None:
            rdata["source_analysis"] = source_analysis
        rdata["last_updated"] = data["last_updated"]

        with path.open("w", encoding="utf-8") as f:
            ry.dump(rdata, f)

    except ImportError:
        print(
            "[config] ⚠ ruamel.yaml 미설치 — PyYAML 로 재작성합니다 (주석 소실). "
            "주석 보존을 원하면: pip install ruamel.yaml --break-system-packages"
        )
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                data, f,
                allow_unicode=True, sort_keys=False, default_flow_style=False,
            )

    return path


# ---------------------------------------------------------------------------
# CLI (단독 실행 시 현재 설정 표시)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    try:
        cfg = load_strategy()
        print("=== active_strategy.yaml (v2_combined) ===")
        print(cfg.summary())
    except Exception as e:
        print(f"[config] 로드 실패: {e}", file=sys.stderr)
        sys.exit(1)
