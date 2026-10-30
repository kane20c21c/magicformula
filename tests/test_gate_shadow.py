"""황금률 Wyckoff 게이트 3변형 그림자 (2026-10-30 케인 WR-a — 와이코프 재작업 WR1 ①)."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from magic_formula.analysis.area_scores import AREA_KEYS
from magic_formula.daily import runner as R

CFG = SimpleNamespace(weights=None, gate_exclude_phases=("Markdown",), threshold=5.0, candidate_threshold=5.0)


def _case(n=30, last_score=6.0, prev_score=4.0, label_last="Markup", sig_at=None, d0=None):
    idx = pd.date_range("2026-09-01", periods=n, freq="B")
    base = np.full(n, 3.0); base[-1] = last_score; base[-2] = prev_score
    areas = {k: pd.Series(base, index=idx) for k in AREA_KEYS}          # 같은 값 → 가중평균도 같은 값
    lab = pd.Series(["Markup"] * n, index=idx, dtype=object); lab.iloc[-1] = label_last
    sig = pd.Series([None] * n, index=idx, dtype=object)
    if sig_at is not None:
        sig.iloc[sig_at] = "PANIC_ZONE"
    df = pd.DataFrame({"Close": np.arange(n, dtype=float) + 100, "Wyckoff_Signal": sig}, index=idx)
    if d0 is not None:
        df["Wyckoff_Phase_D0"] = d0
    return df, areas, lab


def test_label_gate_only_masks_label_variant():
    df, areas, lab = _case(label_last="Markdown")
    r = R.gate_shadow_row("000001", df, areas, CFG, lab)
    assert r["label_gate"] is True and r["label"]["score"] is None and r["label"]["candidate"] is False
    assert r["off"]["score"] == 6.0 and r["off"]["candidate"] and r["off"]["is_signal"]
    assert r["down"]["candidate"] and r["down"]["is_signal"]


def test_down_masks_within_n_days_only():
    df, areas, lab = _case(sig_at=-R.GATE_SHADOW_N)                  # 정확히 N 거래일 창의 첫날
    r = R.gate_shadow_row("000001", df, areas, CFG, lab)
    assert r["down_flag"] is True and r["down"]["score"] is None and r["label"]["candidate"]
    df, areas, lab = _case(sig_at=-(R.GATE_SHADOW_N + 1))            # 창 밖
    r = R.gate_shadow_row("000001", df, areas, CFG, lab)
    assert r["down_flag"] is False and r["down"]["candidate"]


def test_d0_flag_recorded_but_not_in_definition():
    n = 30
    d0 = ["Range"] * (n - 3) + ["Downtrend"] * 3
    df, areas, lab = _case(n=n, d0=d0)
    r = R.gate_shadow_row("000001", df, areas, CFG, lab)
    assert r["d0_flag"] is True and r["down"]["candidate"]          # (나) 정의엔 안 들어감
    df, areas, lab = _case()
    assert R.gate_shadow_row("000001", df, areas, CFG, lab)["d0_flag"] is None


def test_write_gate_shadow_separate_dir(tmp_path):
    df, areas, lab = _case(label_last="Markdown")
    rows = [R.gate_shadow_row("000001", df, areas, CFG, lab)]
    p = R.write_gate_shadow(rows, CFG, "20261030", "daily_extended_signal", out_dir=tmp_path / "gate_shadow")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert p.name == "daily_extended_signal_gate_shadow_20261030.json"
    assert d["summary"]["label"]["candidates"] == 0 and d["summary"]["off"]["candidates"] == 1
    assert R.GATE_SHADOW_DIR.name == "gate_shadow" and R.GATE_SHADOW_DIR != R.OUTPUT_DIR
