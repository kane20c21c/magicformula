# -*- coding: utf-8 -*-
"""korean_mkt_study/shadow_track.py — v1238 (stop_mode=close) 분기: 종가 확인 → 익일 시가 체결, peak 종가 기준.
LLV·유니버스 파일 없이 합성 가격으로 advance() 를 돌린다 (load_prices·유니버스·진입필터 monkeypatch)."""
from __future__ import annotations

import importlib.util, json, sys
from pathlib import Path

import numpy as np, pandas as pd, pytest

KMS = Path(__file__).resolve().parents[1] / "korean_mkt_study"
sys.path.insert(0, str(KMS))
spec = importlib.util.spec_from_file_location("shadow_track", KMS / "shadow_track.py")
st = importlib.util.module_from_spec(spec); spec.loader.exec_module(st)

TK = "000001"


def _prices(dates, open_, high, low, close):
    mk = lambda v: pd.DataFrame({TK: v}, index=dates)
    return {"open": mk(open_), "high": mk(high), "low": mk(low), "close": mk(close), "yz_20": mk([0.02] * len(dates))}


def _run(tmp_path, monkeypatch, model, prices, dates):
    monkeypatch.setattr(st, "DATA", tmp_path)
    monkeypatch.setattr(st, "load_prices", lambda tickers, start="2025-06-01": prices)
    monkeypatch.setattr(st, "all_universe_tickers", lambda m: ([TK], {TK: "테스트"}))
    monkeypatch.setattr(st, "universe_for", lambda m, d: {"stocks": [{"ticker": TK, "name": "테스트"}]})
    monkeypatch.setattr(st.SR, "compute_entry_filter", lambda o, h, l, c: type("X", (), {"excluded": pd.DataFrame(False, index=dates, columns=[TK])})())
    state = {"model": model, "cash": 0.0, "pending": [], "pending_buys": [], "last_sell_date": {},
             "positions": {TK: {"name": "테스트", "slots": 1, "shares": 10, "avg_price": 1000.0, "peak": 1000.0,
                                "entry_date": dates[0].strftime("%Y-%m-%d")}},
             "last_processed": dates[0].strftime("%Y-%m-%d")}
    return st.advance(model, state)


def test_models_v1238_differs_from_v1237_only_by_stop_mode():
    a, b = st.MODELS["v1237"], st.MODELS["v1238"]
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {"label", "stop_mode"}
    assert b["stop_mode"] == "close" and "stop_mode" not in a


def test_close_mode_marks_at_close_and_fills_next_open(tmp_path, monkeypatch):
    # D0 보유(평단 1000). D1: 저가 700(장중 급락)·종가 950 → close 모드는 **안 판다**(선 800). D2: 종가 790 ≤ 800 → 마킹. D3: 시가 820 체결.
    dates = pd.bdate_range("2026-07-06", periods=4)
    pr = _prices(dates, open_=[1000, 990, 940, 820], high=[1010, 1000, 960, 830], low=[990, 700, 780, 810], close=[1000, 950, 790, 815])
    s = _run(tmp_path, monkeypatch, "v1238", pr, dates)
    assert TK not in s["positions"], "D3 시가에 체결되어 포지션 소멸"
    eq = pd.read_csv(tmp_path / "shadow_v1238_equity.csv")
    assert len(eq) == 3
    assert eq.iloc[0]["n_pos"] == 1 and eq.iloc[1]["n_pos"] == 1 and eq.iloc[2]["n_pos"] == 0
    # 체결가 820 × 10주 × (1 − 수수료·세금), STOP_FILL(0.995) 할인 없음
    assert s["pending"][0]["amount"] == pytest.approx(820 * 10 * (1 - st.SELL_FEE - st.SELL_TAX))
    assert s["last_sell_date"][TK] == dates[3].strftime("%Y-%m-%d")


def test_intraday_mode_sells_on_low_touch_same_day(tmp_path, monkeypatch):
    dates = pd.bdate_range("2026-07-06", periods=3)
    pr = _prices(dates, open_=[1000, 990, 940], high=[1010, 1000, 960], low=[990, 700, 780], close=[1000, 950, 790])
    s = _run(tmp_path, monkeypatch, "v1237", pr, dates)
    eq = pd.read_csv(tmp_path / "shadow_v1237_equity.csv")
    assert eq.iloc[0]["n_pos"] == 0, "장중 모형: D1 저가 700 ≤ 800 → 그날 선 가격(×0.995) 체결"
    # D1 매도 대금은 D2 1) T+1 결제에서 현금화된다
    assert s["pending"] == [] and s["cash"] == pytest.approx(800 * st.STOP_FILL * 10 * (1 - st.SELL_FEE - st.SELL_TAX))


def test_close_mode_peak_from_close_not_high(tmp_path, monkeypatch):
    # D1 고가 1500·종가 1100 → peak 1100 (고가 아님). D+1 이라 조기 −20% → 선 880 (peak 기준). 종가 1100 > 880 보유.
    dates = pd.bdate_range("2026-07-06", periods=2)
    pr = _prices(dates, open_=[1000, 1050], high=[1010, 1500], low=[990, 1040], close=[1000, 1100])
    s = _run(tmp_path, monkeypatch, "v1238", pr, dates)
    assert s["positions"][TK]["peak"] == 1100.0 and not s["positions"][TK].get("exit_pending")
