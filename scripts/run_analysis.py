#!/usr/bin/env python3
"""
scripts/run_analysis.py
-----------------------
오프라인 분석 트랙 진입점 (얇은 wrapper).

월 1회 수행하는 가중치/규칙 최적화 백테스트를 실행한다.
본체 로직은 ``magic_formula.analysis.backtest.main()`` 에 위임.

실행
----
    python scripts/run_analysis.py                       # 전체 실행
    python scripts/run_analysis.py --quick-test          # Basic/R1 빠른 검증
    python scripts/run_analysis.py --no-cache            # 데이터 새로 수집
    python scripts/run_analysis.py --output-dir ./out    # 출력 폴더 변경

도출된 최적 조합은 ``configs/active_strategy.yaml`` 에 반영해야
데일리 트랙(scripts/daily_signal.py) 이 즉시 사용한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

_MAGIC_ROOT = Path(__file__).resolve().parent.parent
_VAULT_PATH = Path("/Users/kaneyoun/DriveForALL/StoLab/longlivevault")
for p in [str(_MAGIC_ROOT), str(_VAULT_PATH)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from magic_formula.analysis.backtest import main   # noqa: E402


if __name__ == "__main__":
    main()
