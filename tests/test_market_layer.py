# -*- coding: utf-8 -*-
"""통일장 시장 층 판정기(market_layer) 회귀 테스트 (2026-11-10).

1) 연구 재현 — 연구 입력(필터 브레드스·pykrx TMI)을 넣으면 2015-01-02~2026-11-04 2,905일 동안
   브레드스·C2·⑦·⑤ 가 연구 판정과 하루도 다르지 않아야 한다 (고정본 tests/fixtures/market_layer_research.csv,
   생성 = korean_mkt_study/make_market_layer_fixture.py).
2) 운영 재료 — LLV 전 종목 브레드스(2025-05-02~)를 넣으면 보류 구간 2026-06-01~11-04 ⑦·⑤ 가 연구와 같아야 한다
   (케인 11-10 (가) 전 종목 채택 근거). 그 앞(2025-12~2026-04)은 짧은 이력·정의 차이로 갈리는 것이 알려져 있다.
3) 부품 — 히스테리시스·이평 상태·C2 복귀·갭쇼크 판정·관문 모드의 경계 동작.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_layer import DEFAULT, MarketLayerConfig, c2_state, gap_shock, gated, hyst, judge, ma_state

FIX = Path(__file__).resolve().parent / "fixtures" / "market_layer_research.csv"


@pytest.fixture(scope="module")
def fx():
    return pd.read_csv(FIX, index_col="date", parse_dates=["date"])


def _ratio(fx, a, d):
    s = fx[a] / (fx[a] + fx[d])
    return s.dropna()


def test_reproduces_research_all_days(fx):
    d = judge(fx.k_open, fx.k_close, fx.tmi, _ratio(fx, "adv_res", "dec_res"))
    assert len(d) == 2905
    for col, truth in (("breadth", "br"), ("c2", "c2"), ("p7", "p7"), ("p5", "p5")):
        diff = d[col] != fx[truth].reindex(d.index).astype(bool)
        assert not diff.any(), f"{col}: 연구와 다른 날 {list(d.index[diff][:5])}"


def test_llv_inputs_match_research_in_holdout(fx):
    d = judge(fx.k_open, fx.k_close, fx.tmi, _ratio(fx, "adv_llv", "dec_llv"))
    w = d.loc["2026-06-01":"2026-11-04"]
    assert len(w) == 106
    for col in ("p7", "p5"):
        assert (w[col] == fx[col].reindex(w.index).astype(bool)).all()


def test_holdout_states_known(fx):
    """보류 구간 판정 이력 고정 — 05-19 하락 → 08-04 상승 → 10-15 하락 (통일장 문서 3-3장)."""
    d = judge(fx.k_open, fx.k_close, fx.tmi, _ratio(fx, "adv_llv", "dec_llv"))
    p7 = d.p7.loc["2026-05-01":"2026-11-04"]
    flips = p7[p7 != p7.shift()].iloc[1:]
    assert [(t.strftime("%m-%d"), bool(v)) for t, v in flips.items()] == [("05-19", False), ("08-04", True), ("10-15", False)]


# ── 부품 ─────────────────────────────────────────────────────────────────
_IDX = pd.bdate_range("2026-01-01", periods=8)


def test_hyst_band_and_nan_keep():
    x = pd.Series([np.nan, 0.01, -0.05, -0.01, np.nan, 0.049, 0.05, 0.0], index=_IDX)
    assert hyst(x, 0.05, -0.05).tolist() == [True, True, False, False, False, False, True, True]


def test_ma_state_equal_keeps_previous():
    p = pd.Series([10, 10, 10, 9, 9, 12, 11, 11], index=_IDX, dtype=float)
    # 3일선: 결측·결측·10(같음→유지 True)·9.67(9<→False)·9.33(9<→False)·10(12>→True)·10.67(11>→True)·11.33(11<→False)
    assert ma_state(p, 3).tolist() == [True, True, True, False, False, True, True, False]


_K = pd.Series([100, 102, 104, 95, 90, 92, 99.5, 98], index=_IDX, dtype=float)


def test_c2_exit_then_rebound_return():
    """중형 지수가 이평 아래로 내려가 괴리형이 회복되지 않을 때 — 저점 +10% 반등으로만 복귀."""
    mid = pd.Series([1000, 1001, 1002, 990, 980, 970, 960, 950], index=_IDX, dtype=float)
    s = c2_state(_K, mid, MarketLayerConfig(c2_ma=2))
    # 4번째 괴리형 꺾임 → 퇴출(저점 95→90), 99.5 ≥ 90×1.10 에서 복귀, 그 뒤 괴리형이 '꺾이는' 날이 없으니 유지
    assert s.tolist() == [True, True, True, False, False, False, True, True]


def test_c2_returns_when_divergence_recovers():
    """두 지수가 다시 둘 다 이평 위가 되는 날 복귀하고, 다시 꺾이면 퇴출."""
    mid = pd.Series(1000.0, index=_IDX) + np.arange(8)            # 중형은 늘 이평 위
    s = c2_state(_K, mid, MarketLayerConfig(c2_ma=2))
    # 92 > 2일선 91 → 괴리형 회복 → 복귀 / 마지막 98 < 98.75 → 다시 꺾임 → 퇴출
    assert s.tolist() == [True, True, True, False, False, True, True, False]


def test_gap_shock_requires_gap_share():
    op = pd.Series([100, 100, 96.0, 100, 99.5], index=_IDX[:5])
    cl = pd.Series([100, 100, 96.5, 100, 96.0], index=_IDX[:5])
    win, shock = gap_shock(op, cl, _IDX[:5], MarketLayerConfig(gate_window=2))
    # 3일째: −3.5% 중 갭 −4% (갭 몫 > 1) → 쇼크 / 5일째: −4% 중 갭 −0.5% (몫 ≈ 12%) → 쇼크 아님
    assert shock.tolist() == [False, False, True, False, False]
    assert win.tolist() == [False, False, True, True, False]


def test_gated_enters_only_with_breadth_down_and_exits_on_breadth_up():
    gate = np.array([False, True, True, False, False, False, False, False])
    br = pd.Series([False, True, False, False, False, True, True, False], index=_IDX)
    base = pd.Series(True, index=_IDX)
    # 1일째 관문 없음 → 보유 / 2일째 관문인데 브레드스 상승 → 보유 / 3일째 관문 & 하락 → 현금 /
    # 관문이 닫혀도 브레드스가 상승할 때까지 현금 / 6일째 상승 → 해제 / 8일째 하락이지만 관문 없음 → 보유
    assert gated(gate, br, base).tolist() == [True, True, False, False, False, True, True, True]


def test_gated_follows_base_outside_mode():
    gate = np.zeros(8, bool)
    br = pd.Series(True, index=_IDX)
    base = pd.Series([True, False, False, True, True, False, True, True], index=_IDX)
    assert gated(gate, br, base).tolist() == base.tolist()


# ── 재료 읽기 (판정일 잠정 · 결측 사고) ─────────────────────────────────────
import importlib.util  # noqa: E402
import sys  # noqa: E402

from market_layer import inputs as MI  # noqa: E402

_LLV = Path(__file__).resolve().parents[2] / "LongLiveVault"


@pytest.fixture
def llv_root(tmp_path):
    """가짜 LLV — 코스피 200·TMI 300일, KRX 확정 브레드스는 마지막 날 빼고, 마지막 날은 한국투자 sig 만."""
    if str(_LLV) not in sys.path:
        sys.path.insert(0, str(_LLV))
    from stolab_data import market_breadth as mb
    days = pd.bdate_range("2025-09-01", periods=300)
    tk = tmp_path / "data" / "ohlcv" / "tickers"; tk.mkdir(parents=True)
    px = 800 + np.arange(300, dtype=float)
    pd.DataFrame({"Date": days, "Ticker": "KOSPI200", "Open": px, "Close": px}).to_parquet(tk / "KOSPI200.parquet")
    pd.DataFrame({"Date": days[:-1], "Ticker": "KRX_TMI_MID", "Close": px[:-1] * 2}).to_parquet(tk / "KRX_TMI_MID.parquet")
    rows = [{"Date": d, "Market": m, "Source": "KRX", "Phase": "close", "Adv": 500, "Dec": 500, "Flat": 10}
            for d in days[:-1] for m in mb.MARKETS]
    rows += [{"Date": days[-1], "Market": m, "Source": "KIS", "Phase": "sig", "Adv": 700, "Uplm": 3, "Dec": 300, "Lslm": 0,
              "Flat": 20} for m in mb.MARKETS]
    mb.save_breadth(pd.DataFrame(rows), tmp_path / "data" / "market_breadth.parquet")
    return tmp_path, days


def test_inputs_same_day_provisional(llv_root, monkeypatch):
    root, days = llv_root
    from stolab_data import market_breadth as mb
    monkeypatch.setattr(mb, "KIS_ADV_INCLUDES_LIMIT", True)
    i = MI.load_inputs(root=root)
    assert i.as_of == days[-1] and i.source["provisional"] is True
    assert i.source["breadth"] == "한국투자 sig (잠정)" and i.source["tmi"].startswith("그날 없음")
    assert i.ratio.loc[days[-1]] == pytest.approx(0.7) and i.ratio.loc[days[-2]] == pytest.approx(0.5)
    assert days[-1] not in i.tmi.index                     # TMI 그날 없음 → judge 의 200일선 상태가 전날 유지
    d = judge(i.k_open, i.k_close, i.tmi, i.ratio)
    assert d.index[-1] == days[-1]


def test_inputs_kis_definition_unset_keeps_previous(llv_root, monkeypatch):
    root, days = llv_root
    from stolab_data import market_breadth as mb
    monkeypatch.setattr(mb, "KIS_ADV_INCLUDES_LIMIT", None)
    i = MI.load_inputs(root=root)
    assert "사용 불가" in i.source["breadth"] and "전날 유지" in i.source["breadth"]
    assert days[-1] not in i.ratio.index


def test_inputs_past_gap_raises(llv_root):
    root, days = llv_root
    p = root / "data" / "ohlcv" / "tickers" / "KRX_TMI_MID.parquet"
    t = pd.read_parquet(p); t[t.Date != days[-50]].to_parquet(p)          # 50거래일 전 하루 결측
    with pytest.raises(MI.InputGapError, match="중형 TMI"):
        MI.load_inputs(root=root)


def test_inputs_missing_k200_day_raises(llv_root):
    root, days = llv_root
    with pytest.raises(MI.InputGapError, match="코스피 200"):
        MI.load_inputs("2030-01-02", root=root)


def _script():
    spec = importlib.util.spec_from_file_location("mls", Path(__file__).resolve().parents[1] / "scripts" / "market_layer_signal.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def test_merge_file_records_provisional_vs_confirmed(tmp_path):
    m = _script(); p = tmp_path / "x.json"
    a = {"states": {"p7": "현금", "p5": "현금"}}
    out = m.merge_file(p, "provisional", a, "2026-11-10")
    assert out["provisional_matches_confirmed"] is None
    p.write_text(__import__("json").dumps(out), encoding="utf-8")
    out2 = m.merge_file(p, "confirmed", {"states": {"p7": "보유", "p5": "현금"}}, "2026-11-10")
    assert out2["provisional"] == a and out2["provisional_matches_confirmed"] is False
