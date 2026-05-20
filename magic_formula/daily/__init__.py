"""
magic_formula.daily
===================
데일리 자동화 트랙 — 매일 실행되는 황금률 진입신호 리포트 생성.

진입점
------
- ``magic_formula.daily.runner.run(target_date=None, config_path=None)``
- CLI: ``scripts/daily_signal.py`` (얇은 wrapper)

설정 출처
---------
configs/active_strategy.yaml — 분석 트랙에서 도출된 최적 조합을 그대로 사용.
"""

from __future__ import annotations

from .runner import run

__all__ = ["run"]
