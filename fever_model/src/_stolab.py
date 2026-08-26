#!/usr/bin/env python3
"""_stolab.py — StoLab 루트 탐색 (경로 해석 공용)

**이관 안전장치.** 종전 두 스크립트는 `os.path.dirname(os.path.dirname(FM))` 로
두 단계를 올라가 StoLab 을 잡았다. 이 식은 **폴더 깊이에 묶여 있다** —

    이관 전:  StoLab/MagicFormula/fever_model/src  → FM=fever_model, 2단계 위 = StoLab   ✅
    이관 후:  StoLab/SpotGauge/src                 → FM=SpotGauge,   2단계 위 = StoLab 의 상위 ❌

후자에서는 LLV parquet(`longlivevault/data/ohlcv`)과 SMTP `.env`(StockPortfolio /
MorningBrief)를 못 찾는다. 16:30 배치는 입력 없음으로, 17:00 메일은 인증정보 없음으로
**조용히 실패**한다 — 예외가 아니라 무발송이라 알아채기 어렵다.

그래서 **깊이가 아니라 형제 프로젝트 존재 여부**로 StoLab 을 판정한다.
이 방식은 이관 전·후 양쪽에서 같은 값을 돌려주므로, 폴더를 옮기기 전에 먼저
적용해 두어도 동작이 바뀌지 않는다.

우선순위: 환경변수 `STOLAB_DIR` → 형제 프로젝트 탐색 → 종전 식(fallback)
"""
from __future__ import annotations

import os

# StoLab 루트임을 알려주는 표지 — 이 중 하나라도 하위에 있으면 StoLab 으로 본다.
SIBLINGS = ("longlivevault", "StockPortfolio", "MagicFormula", "MorningBrief")


def find_stolab(start: str, max_up: int = 6) -> str:
    """`start`(리포 루트) 에서 위로 올라가며 StoLab 루트를 찾는다.

    Parameters
    ----------
    start : 리포 루트 경로 (이관 전 `fever_model/`, 이관 후 `SpotGauge/`)
    max_up : 최대 상향 탐색 단계

    Returns
    -------
    StoLab 루트의 절대경로. 못 찾으면 종전 식(`start` 의 2단계 위)을 그대로 돌려준다.
    """
    env = os.environ.get("STOLAB_DIR")
    if env and os.path.isdir(env):
        return env

    d = os.path.abspath(start)
    for _ in range(max_up):
        if any(os.path.isdir(os.path.join(d, s)) for s in SIBLINGS):
            return d
        parent = os.path.dirname(d)
        if parent == d:          # 루트 도달
            break
        d = parent

    # 형제를 못 찾음 — 종전 동작 유지 (단독 체크아웃 등)
    return os.path.dirname(os.path.dirname(os.path.abspath(start)))
