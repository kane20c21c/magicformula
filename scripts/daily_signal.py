#!/usr/bin/env python3
"""
scripts/daily_signal.py
-----------------------
데일리 자동화 트랙 진입점 (얇은 wrapper).

본체 로직은 ``magic_formula.daily.runner.run()`` 에 있다.
이 스크립트는 외부 스케줄러(cron / launchd / Cowork 스케줄러) 가 호출하는
표준 명령으로, 이름과 인자 형식은 호환을 위해 그대로 유지한다.

실행
----
    python scripts/daily_signal.py            # 오늘 날짜, 기본 yaml
    python scripts/daily_signal.py 20260519   # 특정 날짜
    python scripts/daily_signal.py 20260519 configs/active_strategy.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

# Magic Formula 루트와 longlivevault 를 sys.path 에 등록
_MAGIC_ROOT = Path(__file__).resolve().parent.parent
_VAULT_PATH = Path("/Users/kaneyoun/DriveForALL/StoLab/longlivevault")
for p in [str(_MAGIC_ROOT), str(_VAULT_PATH)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from magic_formula.daily.runner import run   # noqa: E402


if __name__ == "__main__":
    target_date = sys.argv[1] if len(sys.argv) > 1 else None
    config_path = sys.argv[2] if len(sys.argv) > 2 else None
    result = run(target_date, config_path)
    print(f"\n✅ 완료: 신호 {result['signal_count']}종목 / 전체 {result['total_tickers']}종목")
