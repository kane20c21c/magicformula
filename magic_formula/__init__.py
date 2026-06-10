"""
magic_formula
=============
황금률 백테스트·데일리 시그널 패키지 (v2_combined 단일 체제).

두 트랙
-------
- 데일리 자동화 : scripts/daily_signal.py 가 magic_formula.daily.runner 호출
- 월 1회 분석   : scripts/run_analysis.py → magic_formula.main (그리드 백테스트)

핵심 모듈
---------
- ``magic_formula._vault``              : longlivevault 진입점 통합 (CORE_TICKERS / universe / 종목명)
- ``magic_formula.config``              : configs/active_strategy.yaml (v2) 로더·부분갱신
- ``magic_formula.indicators``          : 기술지표 헬퍼 (RSI/MACD/BB/ATR/OBV ...)
- ``magic_formula.analysis.area_scores``: 4영역 점수 + 레짐 + 결합 (운영 점수 정본)
- ``magic_formula.signals.rules``       : threshold_breakout 진입 + C1/TIME 청산
- ``magic_formula.simulator.simulator`` : 매매 시뮬레이터 (yaml trading 스펙 1:1)
- ``magic_formula.metrics.metrics``     : 성과 지표 (robust 상위5제외 포함)
- ``magic_formula.optimizer.optimizer`` : 가중치 그리드 백테스트
- ``magic_formula.data.collector``      : OHLCV / KOSPI 수집 (vault 위임)

자주 쓰는 단축 import
---------------------
    from magic_formula import load_strategy
    from magic_formula import get_universe, get_ticker_name

변경 이력
---------
2026-06-10 v1 완전 폐기 (scoring/scorer, R1~R3/ADAPTIVE, 5영역 가중평균).
"""

from __future__ import annotations

# 공개 API 단축 import (자주 쓰는 것만)
from ._vault import (
    VAULT_AVAILABLE,
    VAULT_PATH,
    CORE_TICKERS,
    DEFAULT_EXCLUDE,
    SECTOR_ORDER,
    TICKER_SECTORS,
    TICKER_NAMES_FALLBACK,
    get_sector,
    get_ticker_name,
    get_universe,
)
from .config import (
    StrategyV2,
    load_strategy,
    update_strategy_fields,
    DEFAULT_CONFIG_PATH,
    HISTORY_DIR,
)

__all__ = [
    # _vault
    "VAULT_AVAILABLE", "VAULT_PATH",
    "CORE_TICKERS", "DEFAULT_EXCLUDE",
    "SECTOR_ORDER", "TICKER_SECTORS", "TICKER_NAMES_FALLBACK",
    "get_sector", "get_ticker_name", "get_universe",
    # config
    "StrategyV2", "load_strategy", "update_strategy_fields",
    "DEFAULT_CONFIG_PATH", "HISTORY_DIR",
]

__version__ = "1.0.0"   # 2026-06-10: v2_combined 단일 체제
