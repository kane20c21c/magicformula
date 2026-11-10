#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/market_layer_history.py — 통일장 국면 라벨 표 다시 만들기 (2026-11-10 케인).

    python3 scripts/market_layer_history.py            # LLV 확정 재료로 전 구간 → output/market_layer/regime_history.parquet
    python3 scripts/market_layer_history.py --dry-run  # 저장 없이 요약만

평소에는 08:40 확정 판정(market_layer_signal.py --confirm)이 끝에 이 갱신을 같이 부른다 — 따로 돌릴 일은
LLV 재료를 과거로 늘렸을 때(예: 브레드스 백필) 정도. 표의 정의·주의는 market_layer/history.py 머리말.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from market_layer.history import update_history          # noqa: E402
from market_layer.inputs import InputGapError            # noqa: E402


def main(argv: list[str]) -> int:
    try:
        m = update_history(dry_run="--dry-run" in argv)
    except InputGapError as e:
        print(f"✗ 국면 라벨 표 보류 — {e}"); return 1
    print(f"통일장 국면 라벨 표 {m['version']} · {m['first_date']} ~ {m['as_of']} · {m['rows']}일")
    print(f"  유효 시작 — 브레드스 {m['breadth_valid_from']} · C2 {m['c2_valid_from']} · ⑦ {m['p7_valid_from']}")
    print(f"  구간 수 — {m['spells']}")
    print(f"  최신 — {m['latest']}")
    print(f"  재료 — {m['breadth_source']}")
    print("저장: " + m["path"] if "path" in m else "[dry-run] 저장 안 함")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
