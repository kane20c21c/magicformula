# -*- coding: utf-8 -*-
"""classification.py — 데이 모델 섹터 분류 로더 (2026-11-03, 케인 U7 확정).

정본은 LLV data/ticker_classification.json (git 미추적, 미니에서 생성). 그 위에 **mop_model/sector_overrides.json**
(git 추적 — 미니는 git pull 로만 바뀐다)의 케인 지정 섹터를 덮어쓴다:
  · LLV 멤버인데 분류가 없던 17종목 (U7 섹터 지정, 2026-11-02)
  · 기존 '바이오' 15종목 세분화 — 바이오.제약/바이오.신물질, 셀트리온·삼성바이오로직스는 생활.제약
반환: {ticker: {"ticker", "name", "sector"}} — build_panel·run_gauge 공용.
"""
import json
import os

import config as cfg

OVERRIDES = os.path.join(cfg.BASE_DIR, "sector_overrides.json")


def load_classification(classj=None, overrides=None) -> dict:
    j = json.load(open(classj or cfg.CLASSJ, encoding="utf-8"))
    out = {str(v["ticker"]).zfill(6): {"ticker": str(v["ticker"]).zfill(6), "name": v["name"], "sector": v["sector"]}
           for v in j["classifications"].values()}
    path = overrides or OVERRIDES
    if path and os.path.exists(path):
        for t, v in json.load(open(path, encoding="utf-8")).get("overrides", {}).items():
            t = str(t).zfill(6)
            out[t] = {"ticker": t, "name": v.get("name") or out.get(t, {}).get("name", ""), "sector": v["sector"]}
    return out
