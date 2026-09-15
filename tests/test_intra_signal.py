# -*- coding: utf-8 -*-
"""
tests/test_intra_signal.py — 화이트포트 신호(run_daily --target y_intra) 회귀 테스트
====================================================================================
대상: mop_model/src/intra.py (타깃·5피처), run_daily.py 의 타깃 프로필·경로 분기.
전부 LLV/KIS 무관 — 합성 프레임으로 실행. 모델 학습은 하지 않는다.

지키는 것:
  1) intra.py 의 정의가 사전 등록 실험(experiments/intraday_eval.py)과 같다
  2) 기본 호출(옵션 없음)은 데이포트 그대로 — 타깃 y_rel, output/signals
  3) y_intra 는 별도 디렉터리(output/signals_white) 로만 쓴다
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_MOP = Path(__file__).resolve().parent.parent / "mop_model"
for p in (_MOP / "src", _MOP / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import intra  # noqa: E402


def _panel(n_days=40, n_tk=6, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2026-01-05", periods=n_days)
    rows = []
    for t in range(n_tk):
        px = 10000.0
        for d in days:
            o = px * (1 + rng.normal(0, 0.01)); c = o * (1 + rng.normal(0, 0.015))
            rows.append(dict(Date=d, Ticker=f"T{t}", Open=o, Close=c)); px = c
    d = pd.DataFrame(rows).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    g = d.groupby("Ticker")
    d["Gap_T1"] = g.Open.shift(-1) / d.Close - 1.0
    d["C2C_T1"] = g.Close.shift(-1) / d.Close - 1.0
    return d.sort_values(["Date", "Ticker"]).reset_index(drop=True)


class TestIntraDefinition:
    def test_target_matches_experiment(self):
        """운영 intra.py == 사전 등록 실험 add_intra_target (같은 입력 → 같은 y_intra)."""
        ie = pytest.importorskip("intraday_eval")     # lightgbm 없는 환경이면 건너뜀
        a = intra.add_intra_target(_panel()); b = ie.add_intra_target(_panel())
        pd.testing.assert_series_equal(a.y_intra, b.y_intra, check_names=False)
        pd.testing.assert_series_equal(a.intra_T1, b.intra_T1, check_names=False)

    def test_feats_match_experiment(self):
        ie = pytest.importorskip("intraday_eval")
        a = intra.add_intra_feats(_panel()); b = ie.add_intra_feats(_panel())
        for c in intra.INTRA_COLS:
            pd.testing.assert_series_equal(a[c], b[c], check_names=False)

    def test_intra_t1_reconstructs_open_to_close(self):
        """(1+C2C)/(1+Gap)−1 == 익일 Close/Open − 1."""
        d = intra.add_intra_target(_panel())
        d = d.sort_values(["Ticker", "Date"])
        g = d.groupby("Ticker")
        nxt = g.Close.shift(-1) / g.Open.shift(-1) - 1.0
        m = d.intra_T1.notna()
        assert np.allclose(d.intra_T1[m], nxt.loc[d.index[m]], atol=1e-12)

    def test_y_intra_is_relative_binary(self):
        d = intra.add_intra_target(_panel())
        ok = d.dropna(subset=["y_intra"])
        assert set(ok.y_intra.unique()) <= {0.0, 1.0}
        # 그날 중앙값 기준 → 날짜별 양성률은 0.5 근처 (6종목이면 정확히 3/6)
        assert (ok.groupby("Date").y_intra.mean().round(2) == 0.5).all()
        # 마지막 날은 T1 라벨 없음
        assert d[d.Date == d.Date.max()].y_intra.isna().all()

    def test_feats_use_only_today(self):
        """t 의 O/C 까지만 — 미래 행을 바꿔도 오늘 피처는 불변 (누출 방지)."""
        base = _panel(); a = intra.add_intra_feats(base.copy())
        fut = base.copy(); last = fut.Date.max(); fut.loc[fut.Date == last, ["Open", "Close"]] *= 1.5
        b = intra.add_intra_feats(fut)
        m = a.Date < last
        for c in intra.INTRA_COLS:
            pd.testing.assert_series_equal(a.loc[m, c], b.loc[m, c], check_names=False)

    def test_prepare_returns_sorted_and_extended_cols(self):
        d, cols = intra.prepare_intra(_panel(), ["f1", "f2"])
        assert cols == ["f1", "f2"] + intra.INTRA_COLS
        assert d.equals(d.sort_values(["Date", "Ticker"]).reset_index(drop=True))
        assert d.id1.notna().all() and d.id20.notna().sum() > 0


class TestRunDailyProfiles:
    def test_default_is_dayport_unchanged(self):
        rd = pytest.importorskip("run_daily")       # lightgbm/catboost 필요
        import config as cfg
        p = rd.PROFILES[cfg.TARGET]
        assert (p["signal_dir"], p["top_k"], p["strategy_id"]) == (cfg.SIGNAL_DIR, cfg.TOP_K, cfg.STRATEGY_ID)

    def test_intra_profile_is_separate_dir(self):
        rd = pytest.importorskip("run_daily")
        import config as cfg
        p = rd.PROFILES["y_intra"]
        assert p["signal_dir"].endswith("signals_white") and p["signal_dir"] != cfg.SIGNAL_DIR
        assert p["top_k"] == 10 and p["strategy_id"] == "white_ml_top10"

    def test_cli_has_target_and_signal_dir(self):
        src = (_MOP / "src" / "run_daily.py").read_text(encoding="utf-8")
        assert '"--target"' in src and '"--signal-dir"' in src
        assert "signal_dir=a.signal_dir" in src and "target=a.target" in src
