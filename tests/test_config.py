"""
tests/test_config.py
--------------------
StrategyV2 / load_strategy / update_strategy_fields / v2 가드 동작.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from magic_formula.config import (
    DEFAULT_CONFIG_PATH,
    StrategyV2,
    load_strategy,
    update_strategy_fields,
    validate_dict,
)


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _valid_dict():
    return {
        "strategy_id": "TEST-v2",
        "last_updated": "2026-06-01",
        "system_version": "v2_combined",
        "scoring": {
            "weights": {
                "trend":      0.2,
                "momentum":   0.2,
                "volume":     0.0,
                "volatility": 0.6,
            },
            "threshold": 6.0,
            "candidate_threshold": 5.0,
            "universe": "core_excl_split",
            "gate": {"enabled": True, "exclude_phases": ["Markdown"]},
        },
        "trading": {"entry": {"position_size": 10_000_000}},
    }


def _write_yaml(path: Path, d: dict) -> None:
    path.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")


# ---------------------------------------------------------------------------
# from_dict / 검증
# ---------------------------------------------------------------------------

def test_valid_dict_loads():
    cfg = StrategyV2.from_dict(_valid_dict())
    assert cfg.strategy_id == "TEST-v2"
    assert cfg.threshold == 6.0
    assert cfg.candidate_threshold == 5.0
    assert cfg.gate_enabled is True
    assert cfg.gate_exclude_phases == ("Markdown",)
    assert cfg.position_size == 10_000_000
    assert abs(sum(cfg.weights.values()) - 1.0) < 1e-9


def test_v1_schema_rejected():
    """v1 스키마(system_version 없음)는 명확한 에러로 거부."""
    v1 = {
        "strategy_id": "CompR15",
        "weights": {"trend": 0.3, "momentum": 0.15, "volume": 0.1,
                    "volatility": 0.25, "wyckoff": 0.2},
        "rule": "R1", "area4_mode": "trend",
        "threshold": 5.0, "universe": "core_all",
    }
    with pytest.raises(ValueError, match="system_version"):
        StrategyV2.from_dict(v1)


def test_weights_sum_not_one_raises():
    d = _valid_dict()
    d["scoring"]["weights"]["volatility"] = 0.5      # 합 0.90
    with pytest.raises(ValueError, match="합계가 1.0 아님"):
        StrategyV2.from_dict(d)


def test_missing_weight_key_raises():
    d = _valid_dict()
    del d["scoring"]["weights"]["volume"]
    with pytest.raises(ValueError, match="키 불일치"):
        StrategyV2.from_dict(d)


def test_wyckoff_weight_key_rejected():
    """v1 잔재(wyckoff 가중치 키)는 extra 로 거부."""
    d = _valid_dict()
    d["scoring"]["weights"]["wyckoff"] = 0.0
    with pytest.raises(ValueError, match="키 불일치"):
        StrategyV2.from_dict(d)


def test_threshold_not_number_raises():
    d = _valid_dict()
    d["scoring"]["threshold"] = "six"
    with pytest.raises(ValueError, match="숫자 아님"):
        StrategyV2.from_dict(d)


def test_position_size_default_when_missing():
    d = _valid_dict()
    del d["trading"]
    cfg = StrategyV2.from_dict(d)
    assert cfg.position_size == 10_000_000


# ---------------------------------------------------------------------------
# load / update round-trip
# ---------------------------------------------------------------------------

def test_load_strategy_from_file(tmp_path: Path):
    p = tmp_path / "s.yaml"
    _write_yaml(p, _valid_dict())
    cfg = load_strategy(p)
    assert cfg.threshold == 6.0
    assert cfg.universe == "core_excl_split"


def test_update_strategy_fields_roundtrip(tmp_path: Path):
    """weights/threshold 만 바뀌고 나머지 구조(게이트/trading)는 보존."""
    p = tmp_path / "s.yaml"
    _write_yaml(p, _valid_dict())

    new_w = {"trend": 0.4, "momentum": 0.2, "volume": 0.0, "volatility": 0.4}
    update_strategy_fields(
        p, weights=new_w, threshold=5.0,
        strategy_id="TEST-v2-NEW", backup_history=False,
    )

    cfg = load_strategy(p)
    assert cfg.weights == new_w
    assert cfg.threshold == 5.0
    assert cfg.strategy_id == "TEST-v2-NEW"
    # 비변경 필드 보존
    assert cfg.gate_enabled is True
    assert cfg.position_size == 10_000_000
    assert cfg.universe == "core_excl_split"


def test_update_strategy_fields_guards_v1(tmp_path: Path):
    """v1 yaml 을 대상으로 하면 갱신을 거부한다 (덮어쓰기 사고 방지)."""
    p = tmp_path / "v1.yaml"
    _write_yaml(p, {
        "strategy_id": "CompR15",
        "weights": {"trend": 0.3, "momentum": 0.15, "volume": 0.1,
                    "volatility": 0.25, "wyckoff": 0.2},
        "rule": "R1", "area4_mode": "trend",
        "threshold": 5.0, "universe": "core_all",
    })
    with pytest.raises(ValueError, match="system_version"):
        update_strategy_fields(p, threshold=6.0, backup_history=False)


def test_update_invalid_weights_rejected(tmp_path: Path):
    p = tmp_path / "s.yaml"
    _write_yaml(p, _valid_dict())
    bad = {"trend": 0.9, "momentum": 0.2, "volume": 0.0, "volatility": 0.6}  # 합 1.7
    with pytest.raises(ValueError, match="합계가 1.0 아님"):
        update_strategy_fields(p, weights=bad, backup_history=False)


# ---------------------------------------------------------------------------
# 운영 정본 yaml — 환경 의존 (있을 때만)
# ---------------------------------------------------------------------------

def test_real_active_strategy_loads_as_v2():
    """운영 정본 active_strategy.yaml 이 v2 로더를 통과해야 함."""
    if not DEFAULT_CONFIG_PATH.exists():
        pytest.skip(f"{DEFAULT_CONFIG_PATH} 없음 — 환경 의존 테스트 건너뜀")
    cfg = load_strategy()
    assert cfg.system_version == "v2_combined"
    assert abs(sum(cfg.weights.values()) - 1.0) < 1e-6
    assert cfg.position_size == 10_000_000   # yaml trading.entry 정본
    validate_dict(cfg.raw)
