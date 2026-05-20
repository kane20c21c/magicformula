"""
magic_formula.analysis
======================
오프라인 분석 트랙 — 월 1회 수행하는 가중치/규칙 최적화 백테스트.

진입점
------
- ``magic_formula.analysis.backtest.main()`` — magic_formula.main 의 CLI 위임
- CLI: ``scripts/run_analysis.py`` (얇은 wrapper)

산출물
------
- ``output/analysis/YYYY-MM-DD/`` 하위에 trades.csv, weight_ranking.md, equity_curves.png 등
- 도출된 최적 조합을 ``configs/active_strategy.yaml`` 로 반영하면 데일리 트랙이 즉시 사용

참고
----
실제 백테스트 로직은 ``magic_formula.main`` 에 그대로 있으며, 본 모듈은
분석 트랙 진입점을 명시적으로 노출하기 위한 얇은 위임 레이어다.
향후 본격적인 분리 작업은 P5b 이후 단계에서 다룬다.
"""

from __future__ import annotations

from .backtest import main

__all__ = ["main"]
