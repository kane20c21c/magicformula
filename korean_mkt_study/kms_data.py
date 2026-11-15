# -*- coding: utf-8 -*-
"""kms_data.py — 연구 데이터 세트 로더 (케인 2026-11-15, 한 방향 원칙).

새 연구 스크립트는 경로를 하드코딩하지 말고 이 로더를 거친다 — 세트를 바꿀 때 한 곳만 고친다.
끝난 연구 스크립트(data/·data_ext/·data_1104/ 를 직접 읽는 것)는 재현을 위해 그대로 둔다.

    import kms_data as K
    px   = K.load("prices")                       # 연구 정본(current = data_2015/)
    meta = K.load("meta", columns=["mktcap", "foreign", "is_managed", "managed_known"])
    old  = K.load("prices", set="frozen_20260630")  # 지난 백테스트 재현용 동결본 (2010~)
    K.info()                                      # 세트 목록 · 상태 · 기간

세트 정의(SETS)가 정본이고, 각 폴더의 MANIFEST.json 은 kms_store.py manifest 가 여기서 만든다.
정의(단위·관례)는 DATA_DICTIONARY.md.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

HERE = Path(__file__).resolve().parent

SETS = {
    "current": {
        "path": HERE / "data_2015", "status": "current",
        "role": "연구 정본 — 2015-01-02~, 필요할 때 kms_store.py 로 KRX 에서 직접 연장",
        "sources": "KRX Open API(가격·시총·소속부) · pykrx(외인·펀더멘털·수급) · 한국투자(수급 대비책)",
    },
    "frozen_20260630": {
        "path": HERE / "data", "status": "frozen",
        "role": "동결본 — 2010~2026-06-30 스냅샷. 2026-11 이전 백테스트(backtest.py 등) 재현용. 고치지 않는다",
        "sources": "원 생성 스크립트는 'Kane repo'(이 리포 밖) — DATA_2015.md 가 정의를 역추적",
    },
    "retired_ext_0911": {
        "path": HERE / "data_ext", "status": "retired",
        "role": "은퇴 — data/ 를 LLV 패널로 ~2026-09-11 연장한 사본 (한 방향 원칙 이전 방식). 끝난 연구 재현용",
        "sources": "data/ + LLV panel.parquet",
    },
    "retired_1104": {
        "path": HERE / "data_1104", "status": "retired",
        "role": "은퇴 — data/ 를 LLV 패널·raw 로 ~2026-11-04 연장한 사본 (한 방향 원칙 이전 방식). 끝난 연구 재현용",
        "sources": "data/ + LLV panel.parquet + LLV data/raw",
    },
    "indices": {
        "path": HERE / "data_idx", "status": "current",
        "role": "지수 보조 — KOSPI200 2000~2014, KRX 지수·업종지수 (fetch_indices.py · fetch_sector_indices.py)",
        "sources": "pykrx · KRX",
    },
}


def path(name: str, set: str = "current") -> Path:
    spec = SETS[set]
    p = spec["path"] / (name if name.endswith(".parquet") else f"{name}.parquet")
    if not p.exists():
        raise FileNotFoundError(f"{set}: {p} 없음")
    return p


def load(name: str, set: str = "current", columns: Optional[list] = None) -> pd.DataFrame:
    """세트의 parquet 하나를 읽는다 (읽기 전용). 은퇴 세트는 경고를 찍는다."""
    if SETS[set]["status"] == "retired":
        print(f"⚠ kms_data: '{set}' 은 은퇴 세트다 — 새 연구는 'current' 를 쓸 것")
    return pd.read_parquet(path(name, set), columns=columns)


def manifest(set: str = "current") -> dict:
    mf = SETS[set]["path"] / "MANIFEST.json"
    return json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {}


def info() -> None:
    for k, v in SETS.items():
        m = manifest(k)
        px = m.get("files", {}).get("prices.parquet", {})
        span = f"{px.get('start', '?')} ~ {px.get('end', '?')}" if px else "-"
        print(f"{k:18s} [{v['status']:7s}] {v['path'].name:10s} {span:25s} {v['role']}")


__all__ = ["SETS", "path", "load", "manifest", "info"]
