#!/usr/bin/env python3
"""
scripts/daily_extended_signal.py
--------------------------------
확장(extend) 유니버스 데일리 시그널 진입점 (얇은 wrapper).

기존 daily_signal.py(코어 69) 는 그대로 유지하고, 이 스크립트는 코어 ∪ 확장
= 시총 200종목을 황금율(v2_combined)로 스코어링해 **별도 파일**에 저장한다.
homalone Overview/메인 보드가 읽는 daily_signal_*.json 은 건드리지 않는다.

  - universe : "extended_all" (CORE_TICKERS ∪ EXTEND_TICKERS = 200)
  - 가중치/임계값/게이트 : configs/active_strategy.yaml 정본 그대로 공유 (재튜닝 없음)
  - 출력 : output/signals/daily_extended_signal_YYYYMMDD.json / .md
           output/signals/daily_extended_regimes_YYYYMMDD.json (코어 레짐과 분리)

실행
----
    python scripts/daily_extended_signal.py            # 오늘 날짜, 기본 yaml
    python scripts/daily_extended_signal.py 20260612   # 특정 날짜
    python scripts/daily_extended_signal.py 20260612 configs/active_strategy.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

_MAGIC_ROOT = Path(__file__).resolve().parent.parent
_VAULT_PATH = Path("/Users/kaneyoun/DriveForALL/StoLab/longlivevault")
for p in [str(_MAGIC_ROOT), str(_VAULT_PATH)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from magic_formula.daily.runner import run   # noqa: E402


if __name__ == "__main__":
    target_date = sys.argv[1] if len(sys.argv) > 1 else None
    config_path = sys.argv[2] if len(sys.argv) > 2 else None
    result = run(
        target_date,
        config_path,
        output_prefix="daily_extended_signal",
        regimes_prefix="daily_extended_regimes",
        universe_override="extended_all",
    )
    print(f"\n✅ 확장 완료: 신호 {result.get('signal_count', 0)}종목 "
          f"/ 전체 {result.get('total_tickers', 0)}종목")
