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
# vault 는 StoLab/ 아래 형제 저장소 — 머신(미니/에어) 무관하게 상대 경로로 찾는다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # MagicFormula/
STOLAB_ROOT = PROJECT_ROOT.parent                    # StoLab/
_MAGIC_ROOT = PROJECT_ROOT
_VAULT_PATH = STOLAB_ROOT / "LongLiveVault"
for p in [str(_MAGIC_ROOT), str(_VAULT_PATH)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from magic_formula.daily.runner import run   # noqa: E402


if __name__ == "__main__":
    target_date = sys.argv[1] if len(sys.argv) > 1 else None
    config_path = sys.argv[2] if len(sys.argv) > 2 else None

    # 휴장 가드 (2026-07-16 Kane 지시): plist 는 평일(월~금) 조건만 알므로
    # 공휴일 휴장은 LLV 거래일 캘린더(KIS chk_holiday 정본)로 스크립트가 직접
    # 판단한다. 날짜 인자를 명시한 수동/백필 실행은 가드를 건너뛴다.
    # 캘린더 미가용 시 fail-open (SP paper/engine.py 패턴) — 시그널 누락 방지 우선.
    if target_date is None:
        try:
            from datetime import date
            from stolab_data.trading_calendar import is_trading_day
            _today = date.today()
            if not is_trading_day(_today):
                print(f"⏸ {_today} 휴장일 — 데일리 시그널 skip (LLV 거래일 캘린더)")
                sys.exit(0)
        except Exception as e:
            print(f"⚠ 거래일 캘린더 확인 실패({e}) — 가드 없이 진행")

    result = run(target_date, config_path)
    print(f"\n✅ 완료: 신호 {result['signal_count']}종목 / 전체 {result['total_tickers']}종목")
