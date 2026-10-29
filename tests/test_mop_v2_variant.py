# -*- coding: utf-8 -*-
"""
tests/test_mop_v2_variant.py — 데이포트 그림자 v2 (run_daily --variant v2) 회귀 테스트 (2026-10-29)
=================================================================================================
결정: Kane 2026-10-29 (가) — WP9-1(wy_phase·wy_label 미래 정보) + WP9-2(vol_idx 비인과) 반영 그림자.

지키는 것:
  1) 운영 CHAMPION(148)·BASE/XS/G1/G5 목록은 v2 추가로 바뀌지 않는다
  2) CHAMPION_V2 = 146 = CHAMPION − {wy 2, vol_idx 3} + {vol_idx_c 3}
  3) vol_idx_c 는 LLV Vol_Idx_C 를 그대로 쓰고, 없으면 같은 정의로 직접 계산(인과)
  4) --variant v2 는 output/signals_v2 로만 쓰고, 화이트포트 타깃과는 섞이지 않는다
모델 학습 없음 — 합성 패널.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_MOP = Path(__file__).resolve().parent.parent / "mop_model"
if str(_MOP / "src") not in sys.path:
    sys.path.insert(0, str(_MOP / "src"))

import config as cfg  # noqa: E402
import build_features as bf  # noqa: E402

_NUM = ["MA5", "MA20", "MA60", "BB_upper", "BB_lower", "Rel_Volume", "Vol_Idx", "Vol_MA20", "RSI", "RSI_MA5",
        "MACD", "MACD_Signal", "MACD_Hist", "AD_Line_Chg", "Chaikin_Osc", "Inst_Net", "Foreign_Net", "Retail_Net",
        "Gap_Ratio", "Confidence_Index", "Energy_Index", "Hope_Vector", "Hope_Vector_Z", "Anxiety_Index",
        "Anxiety_Index_Z", "Smart_Pressure", "Retail_Pressure", "Supply_Score", "Close_Idx",
        "Weis_Dir", "Weis_Chg", "Weis_Days", "Weis_Vol", "Gap_T1", "C2C_T1", "y_rel", "y_abs", "mkt"]


def _panel(n_days=60, n_tk=12, seed=0, with_llv=True):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2026-01-05", periods=n_days)
    rows = []
    for t in range(n_tk):
        c = 10000 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days)))
        v = rng.integers(1_000, 100_000, n_days).astype(float)
        v[3] = 0.0
        for i, d in enumerate(days):
            r = dict(Date=d, Ticker=f"{t:06d}", Name=f"N{t}", sector_top=f"S{t % 2}", sector_sub=f"S{t % 2}.a",
                     Open=c[i], High=c[i] * 1.01, Low=c[i] * 0.99, Close=c[i], Volume=v[i], Amount=v[i] * c[i],
                     MarketCap=c[i] * 1e6, is_halt=False,
                     Wyckoff_Label="Markup", Wyckoff_Phase="Uptrend")
            r.update({k: rng.normal() for k in _NUM})
            r["MA5"] = r["MA20"] = r["MA60"] = c[i]
            r["BB_upper"], r["BB_lower"], r["Vol_MA20"] = c[i] * 1.05, c[i] * 0.95, 50_000.0
            rows.append(r)
    px = pd.DataFrame(rows).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    px["Ret_1d"] = px.groupby("Ticker").Close.pct_change()
    if with_llv:
        px["Vol_Idx_C"] = bf._local_vol_idx_c(px) * 1.0 + 0.5     # LLV 값 표식(+0.5) — 그대로 쓰는지 확인용
    return px


@pytest.fixture
def built(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "FEATURES", str(tmp_path / "features.parquet"))
    monkeypatch.setattr(cfg, "COLS_JSON", str(tmp_path / "cols.json"))
    d = bf.build_features(panel_df=_panel(), save=True)
    return d, json.load(open(cfg.COLS_JSON))


class TestFeatureSets:
    def test_champion_unchanged(self, built):
        _, C = built
        assert (len(C["BASE"]), len(C["XS"]), len(C["G1"]), len(C["G5"]), len(C["CHAMPION"])) == (48, 45, 45, 10, 148)
        assert not set(bf.V2_EXTRA) & set(C["CHAMPION"])
        assert {"wy_phase", "wy_label", "vol_idx", "xs_vol_idx", "s1_vol_idx"} <= set(C["CHAMPION"])

    def test_champion_v2_is_146(self, built):
        d, C = built
        v2 = C["CHAMPION_V2"]
        assert len(v2) == 146 and len(set(v2)) == 146
        assert set(v2) == (set(C["CHAMPION"]) - set(bf.V2_DROP)) | set(bf.V2_EXTRA)
        assert all(c in d.columns for c in v2)

    def test_matches_production_cols_json_shape(self):
        """저장소의 운영 cols.json CHAMPION 과 같은 목록이 나오는지 (순서까지)."""
        f = _MOP / "build" / "cols.json"            # git 제외 산출물 — 없는 환경(새 클론)은 건너뜀
        if not f.exists():
            pytest.skip("build/cols.json 없음")
        prod = json.load(open(f))["CHAMPION"]
        d = bf.build_features(panel_df=_panel(), save=False)
        base = [c for c in prod if not c.startswith(("xs_", "s1_", "s5_"))]
        assert all(c in d.columns for c in prod) and len(base) == 48


class TestVolIdxC:
    def test_uses_llv_column_when_present(self, built):
        d, _ = built
        px = _panel()
        exp = pd.Series(px.Vol_Idx_C.to_numpy(), index=px.index)
        got = d.set_index(["Ticker", "Date"]).vol_idx_c
        ref = px.set_index(["Ticker", "Date"]).Vol_Idx_C
        pd.testing.assert_series_equal(got.sort_index(), ref.sort_index(), check_names=False)
        assert exp.notna().sum() > 0

    def test_fallback_is_causal_and_matches_definition(self, capsys):
        px = _panel(with_llv=False)
        d = bf.build_features(panel_df=px, save=False)
        assert "Vol_Idx_C 없음" in capsys.readouterr().err
        g = px[px.Ticker == "000001"].reset_index(drop=True)
        vv = g.Volume.where(g.Volume > 0)
        t = 40
        exp = g.Volume[t] / vv[:t + 1].median() * 1000
        got = d[(d.Ticker == "000001") & (d.Date == g.Date[t])].vol_idx_c.iloc[0]
        assert got == pytest.approx(exp)
        assert d[(d.Ticker == "000001")].sort_values("Date").vol_idx_c.iloc[:19].isna().all()
        # 절단 불변: 앞 45일만으로 계산해도 t=40 값이 같다
        d2 = bf.build_features(panel_df=px[px.Date <= g.Date[44]], save=False)
        got2 = d2[(d2.Ticker == "000001") & (d2.Date == g.Date[t])].vol_idx_c.iloc[0]
        assert got2 == pytest.approx(got)

    def test_xs_rank_within_date(self, built):
        d, _ = built
        day = d.Date.max()
        x = d[d.Date == day]
        pd.testing.assert_series_equal(x.xs_vol_idx_c, x.vol_idx_c.rank(pct=True), check_names=False)


class TestRunDailyVariant:
    def test_v1_default_unchanged(self):
        rd = pytest.importorskip("run_daily")
        p = rd.resolve_profile()
        assert (p["cols_key"], p["signal_dir"], p["strategy_id"], p["model_version"]) == \
               ("CHAMPION", cfg.SIGNAL_DIR, cfg.STRATEGY_ID, cfg.MODEL_VERSION)

    def test_v2_separate_dir(self):
        rd = pytest.importorskip("run_daily")
        p = rd.resolve_profile(variant="v2")
        assert p["cols_key"] == "CHAMPION_V2"
        assert p["signal_dir"] == cfg.SIGNAL_DIR + "_v2" and p["signal_dir"] != cfg.SIGNAL_DIR
        assert p["strategy_id"] == cfg.STRATEGY_ID + "_v2" and p["model_version"].endswith("+v2")
        assert p["top_k"] == cfg.TOP_K

    def test_v2_rejects_white_target(self):
        rd = pytest.importorskip("run_daily")
        with pytest.raises(SystemExit):
            rd.resolve_profile(target="y_intra", variant="v2")

    def test_cli_has_variant(self):
        src = (_MOP / "src" / "run_daily.py").read_text(encoding="utf-8")
        assert '"--variant"' in src and "variant=a.variant" in src
