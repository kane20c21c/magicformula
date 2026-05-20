"""
magic_formula
=============
황금률 백테스트·데일리 시그널 패키지.

두 트랙
-------
- 데일리 자동화 : scripts/daily_signal.py 가 magic_formula 코어를 호출
- 월 1회 분석   : scripts/run_backtest 또는 magic_formula.main 진입점

핵심 모듈
---------
- ``magic_formula._vault``         : longlivevault 진입점 통합 (CORE_TICKERS / universe / 종목명)
- ``magic_formula.config``         : configs/active_strategy.yaml 로더·덤퍼
- ``magic_formula.scoring.scorer`` : 5영역 점수 + 종합 점수
- ``magic_formula.signals.rules``  : 진입 규칙 (R1/R2/R3) + 청산 규칙
- ``magic_formula.signals.adaptive_rule_selector`` : 종목별 동적 규칙 선택
- ``magic_formula.simulator.simulator`` : 매매 시뮬레이터
- ``magic_formula.metrics.metrics`` : 성과 지표
- ``magic_formula.optimizer.optimizer`` : 가중치 조합 최적화
- ``magic_formula.data.collector`` : OHLCV / KOSPI 수집 (vault 위임)

자주 쓰는 단축 import
---------------------
    from magic_formula import load_strategy
    from magic_formula import get_universe, get_ticker_name
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
    ActiveStrategy,
    load_strategy,
    dump_strategy,
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
    "ActiveStrategy", "load_strategy", "dump_strategy",
    "DEFAULT_CONFIG_PATH", "HISTORY_DIR",
]

__version__ = "0.5.0"   # P5a: 패키지화
