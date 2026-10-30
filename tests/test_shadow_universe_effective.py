"""shadow_track.universe_for — 적용 시작일 기준 (2026-10-31 케인 U6 · 월 중간 적용)."""
import sys
from pathlib import Path

import pandas as pd

KMS = Path(__file__).resolve().parents[1] / "korean_mkt_study"
sys.path.insert(0, str(KMS))
import shadow_track as st  # noqa: E402

DOCS = [
    {"_file": "universe_2026-07_4jo_25pct.json", "apply_month": "2026-07", "base_date": "2026-06-30", "stocks": [1]},
    {"_file": "universe_2026-08_4jo_30pct.json", "apply_month": "2026-08", "base_date": "2026-06-30", "stocks": [1]},
    {"_file": "universe_2026-09_4jo_30pct.json", "apply_month": "2026-09", "base_date": "2026-08-31", "stocks": [1]},
    {"_file": "universe_2026-11-16_4jo_30pct_min30.json", "apply_month": "2026-11", "base_date": "2026-11-13",
     "effective_date": "2026-11-16", "stocks": [1]},
]


def _pick(monkeypatch, day):
    monkeypatch.setattr(st, "_universe_docs", lambda: [dict(x) for x in DOCS])
    model = next(m for m, v in st.MODELS.items() if not v["universe"].startswith("fixed:"))
    return st.universe_for(model, pd.Timestamp(day))["_file"]


def test_past_dates_same_as_before(monkeypatch):
    assert _pick(monkeypatch, "2026-08-14") == "universe_2026-08_4jo_30pct.json"
    assert _pick(monkeypatch, "2026-10-15") == "universe_2026-09_4jo_30pct.json"     # 10월분 없음 → 9월분 (종전과 같음)
    assert _pick(monkeypatch, "2026-06-15") == "universe_2026-07_4jo_25pct.json"     # 가장 이른 파일


def test_mid_month_switch(monkeypatch):
    assert _pick(monkeypatch, "2026-11-13") == "universe_2026-09_4jo_30pct.json"
    assert _pick(monkeypatch, "2026-11-16") == "universe_2026-11-16_4jo_30pct_min30.json"
