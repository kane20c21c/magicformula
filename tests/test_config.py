"""
tests/test_config.py
--------------------
ActiveStrategy / load_strategy / dump_strategy / 검증 동작.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from magic_formula.config import (
    ActiveStrategy,
    DEFAULT_CONFIG_PATH,
    dump_strategy,
    load_strategy,
)


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _valid_dict():
    return {
        "strategy_id": "Test",
        "weights": {
            "trend":      0.30,
            "momentum":   0.20,
            "volume":     0.20,
            "volatility": 0.15,
            "wyckoff":    0.15,
        },
        "rule": "R1",
        "area4_mode": "trend",
        "threshold": 5.0,
        "universe": "core_59",
    }


# ---------------------------------------------------------------------------
# from_dict / 검증
# ---------------------------------------------------------------------------

def test_valid_dict_loads():
    cfg = ActiveStrategy.from_dict(_valid_dict())
    assert cfg.strategy_id == "Test"
    assert cfg.rule == "R1"
    assert cfg.area4_mode == "trend"
    assert abs(sum(cfg.weights.values()) - 1.0) < 1e-9


def test_weights_sum_not_one_raises():
    d = _valid_dict()
    d["weights"]["wyckoff"] = 0.05      # 합 0.90
    with pytest.raises(ValueError, match="합계가 1.0 아님"):
        ActiveStrategy.from_dict(d)


def test_missing_weight_key_raises():
    d = _valid_dict()
    del d["weights"]["wyckoff"]
    with pytest.raises(ValueError, match="키 불일치"):
        ActiveStrategy.from_dict(d)


def test_invalid_rule_raises():
    d = _valid_dict()
    d["rule"] = "R99"
    with pytest.raises(ValueError, match="허용값 아님"):
        ActiveStrategy.from_dict(d)


def test_invalid_area4_mode_raises():
    d = _valid_dict()
    d["area4_mode"] = "weird"
    with pytest.raises(ValueError, match="허용값 아님"):
        ActiveStrategy.from_dict(d)


def test_validate_method_works():
    cfg = ActiveStrategy.from_dict(_valid_dict())
    cfg.validate()   # 정상 — 예외 없음

    # 검증 실패 — mutate 후
    cfg.weights["trend"] = 0.99
    with pytest.raises(ValueError, match="합계가 1.0 아님"):
        cfg.validate()


# ---------------------------------------------------------------------------
# dump / load round-trip
# ---------------------------------------------------------------------------

def test_dump_and_load_roundtrip(tmp_path: Path):
    cfg = ActiveStrategy.from_dict(_valid_dict())
    out = tmp_path / "test.yaml"
    dump_strategy(cfg, out, backup_history=False)

    assert out.exists()
    cfg2 = load_strategy(out)
    assert cfg2.weights == cfg.weights
    assert cfg2.rule == cfg.rule
    assert cfg2.threshold == cfg.threshold
    assert cfg2.area4_mode == cfg.area4_mode


def test_dump_backups_history(tmp_path: Path):
    """backup_history=True 면 기존 파일을 history/ 에 복사한다."""
    target = tmp_path / "active_strategy.yaml"
    history = tmp_path / "history"

    # 모듈 상수 임시 변경 (history 디렉토리 위치)
    import magic_formula.config as cfg_mod
    orig_history = cfg_mod.HISTORY_DIR
    cfg_mod.HISTORY_DIR = history
    try:
        cfg1 = ActiveStrategy.from_dict({**_valid_dict(), "last_updated": "2026-01-01"})
        dump_strategy(cfg1, target, backup_history=False)   # 첫 저장 — 백업 없음

        cfg2 = ActiveStrategy.from_dict({**_valid_dict(), "strategy_id": "Test2"})
        dump_strategy(cfg2, target, backup_history=True)    # 두 번째 — cfg1 이 history 로

        backup_files = list(history.glob("*.yaml"))
        assert len(backup_files) >= 1, "백업 파일이 history 에 생성되어야 함"
    finally:
        cfg_mod.HISTORY_DIR = orig_history


# ---------------------------------------------------------------------------
# 현재 운영 yaml — 실제 active_strategy.yaml 로딩 가능한지
# ---------------------------------------------------------------------------

def test_real_active_strategy_loads():
    """저장소의 실제 active_strategy.yaml 이 검증을 통과해야 함."""
    if not DEFAULT_CONFIG_PATH.exists():
        pytest.skip(f"{DEFAULT_CONFIG_PATH} 없음 — 환경 의존 테스트 건너뜀")
    cfg = load_strategy()
    cfg.validate()
    assert cfg.rule in {"R1", "R2", "R3", "ADAPTIVE"}
    assert cfg.area4_mode in {"trend", "contrarian"}
