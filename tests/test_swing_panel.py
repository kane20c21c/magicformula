"""스윙 패널 (2026-10-10) — strategy_reference.build_panel / panel_spec 판정 회귀."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "korean_mkt_study"))
import strategy_reference as sr  # noqa: E402


def _wide(series: dict) -> pd.DataFrame:
    idx = pd.bdate_range("2025-01-02", periods=len(next(iter(series.values()))))
    return pd.DataFrame(series, index=idx)


def _case():
    n = 260
    up = np.linspace(100, 200, n)                      # 꾸준한 상승
    onset = up.copy(); onset[-1] = onset[-2] * 0.85    # 마지막 날 처음 10%↑ 눌림 → onset
    watch = up.copy(); watch[-3:] = watch[-4] * 0.85   # 3일째 눌림 지속 → 진입 조건 유지, 새 신호 아님
    down = np.linspace(200, 100, n)                    # MA200 아래
    cl = _wide({"A": onset, "B": watch, "C": down})
    ind = sr.compute_indicators(cl)
    return cl, ind, cl.index[-1]


def test_verdicts_without_filter():
    cl, ind, ts = _case()
    p = sr.build_panel(ind, None, ts, ["A", "B", "C"], cl.loc[ts])
    v = {t: r["verdict"] for t, r in p["rows"].items()}
    assert v == {"A": "signal", "B": "watch", "C": "none"}
    assert p["rows"]["A"]["values"]["r2"] is None          # 필터 값 없으면 제외 안 함
    assert p["spec_version"] == sr.STRATEGY_VERSION


def test_spec_follows_constants(monkeypatch):
    monkeypatch.setattr(sr, "FILTER_R2_MIN", 0.5)
    monkeypatch.setattr(sr, "PULLBACK_RATIO", 0.85)
    spec = {m["key"]: m for m in sr.panel_spec()}
    assert spec["r2"]["threshold"] == 0.5
    assert abs(spec["depth"]["threshold"] - 0.15) < 1e-9


def test_filtered_when_both_exclude_conditions_met():
    cl, ind, ts = _case()
    ef = sr.EntryFilter(
        r2=pd.DataFrame(0.99, index=cl.index, columns=cl.columns),
        yzr=pd.DataFrame(0.10, index=cl.index, columns=cl.columns),
        excluded=pd.DataFrame(True, index=cl.index, columns=cl.columns),
    )
    p = sr.build_panel(ind, ef, ts, ["A", "B"], cl.loc[ts])
    assert p["rows"]["A"]["verdict"] == "filtered"
    assert p["rows"]["B"]["verdict"] == "watch"            # onset 아니면 제외 판정 대상 아님
