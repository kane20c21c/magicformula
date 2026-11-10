# -*- coding: utf-8 -*-
"""통일장 국면 라벨 표(market_layer/history.py) 테스트 (2026-11-10 케인).

1) 연구 재현 — 고정본 입력으로 만든 표의 브레드스·C2·⑦·⑤ 가 유효 구간에서 연구 판정과 하루도 다르지 않다.
2) 유효 구간 밖은 결측 — 판정기 시작값('보유')이 표에 새어 나오지 않는다.
3) 구간 번호·경과일·검열 표시, 쇼크 경과일, D1 붙이기(lag 0/1·휴장일), 확정 재료로 저장.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_layer import MarketLayerConfig, attach_regime, build_history, judge
from market_layer.history import breadth_valid_from, c2_valid_from, spells, update_history

FIX = Path(__file__).resolve().parent / "fixtures" / "market_layer_research.csv"
_LLV = Path(__file__).resolve().parents[2] / "LongLiveVault"


@pytest.fixture(scope="module")
def fx():
    return pd.read_csv(FIX, index_col="date", parse_dates=["date"])


@pytest.fixture(scope="module")
def hist(fx):
    r = (fx.adv_res / (fx.adv_res + fx.dec_res)).dropna()
    return build_history(fx.k_open, fx.k_close, fx.tmi, r)


def test_matches_research_on_valid_days(fx, hist):
    h, info = hist
    assert len(h) == 2905
    for col, truth in (("breadth", "br"), ("c2", "c2"), ("p7", "p7"), ("p5", "p5")):
        v = h[col].dropna()
        assert len(v) > 2000, col
        diff = v.astype(bool) != fx[truth].reindex(v.index).astype(bool)
        assert not diff.any(), f"{col}: 연구와 다른 날 {list(v.index[diff][:5])}"


def test_warmup_is_missing_not_hold(fx, hist):
    h, info = hist
    b0, c0 = pd.Timestamp(info["breadth_valid_from"]), pd.Timestamp(info["c2_valid_from"])
    # 연구 필터 브레드스는 2016-01-04 시작 → 250일 중앙값(최소 120일) 뒤에야 유효
    assert b0 > pd.Timestamp("2016-06-01") and c0 > pd.Timestamp("2015-10-01")
    assert h.loc[:b0].breadth.iloc[:-1].isna().all() and h.loc[b0, "breadth"] is not pd.NA
    assert h.loc[:c0].c2.iloc[:-1].isna().all()
    assert h.loc[:max(b0, c0)].p7.iloc[:-1].isna().all() and h.loc[:max(b0, c0)].regime.iloc[:-1].isna().all()
    # 판정기는 같은 날 '보유/상승' 시작값을 낸다 — 표는 그것을 지운다
    d = judge(fx.k_open, fx.k_close, fx.tmi, (fx.adv_res / (fx.adv_res + fx.dec_res)).dropna())
    assert d.loc[:b0].breadth.iloc[:-1].all()


def test_regime_is_two_by_two(hist):
    h, _ = hist
    v = h.dropna(subset=["regime"])
    exp = np.where(v.breadth.astype(bool), "상승", "하락") + "·" + np.where(v.c2.astype(bool), "보유", "퇴출")
    assert (v.regime.to_numpy() == exp).all()
    assert set(v.regime.unique()) == {"상승·보유", "상승·퇴출", "하락·보유", "하락·퇴출"}


def test_spell_columns_consistent(hist):
    h, _ = hist
    v = h.dropna(subset=["breadth"])
    g = v.groupby("breadth_spell")
    assert (g.breadth.nunique() == 1).all()                      # 구간 안은 같은 상태
    assert (g.breadth_day.max() == g.breadth_len.first()).all()  # 마지막 날 경과일 = 구간 길이
    assert v.breadth_cens.iloc[0] == "시작 미상" and v.breadth_cens.iloc[-1] == "진행 중"
    mid = v[(v.breadth_spell > v.breadth_spell.min()) & (v.breadth_spell < v.breadth_spell.max())]
    assert (mid.breadth_cens == "").all()


_IDX = pd.bdate_range("2026-01-05", periods=8)


def test_spells_unit():
    s = pd.Series([pd.NA, pd.NA, True, True, False, False, False, True], index=_IDX, dtype="boolean")
    sp = spells(s)
    assert sp.spell.tolist()[2:] == [1, 1, 2, 2, 2, 3] and sp.spell.isna().tolist()[:2] == [True, True]
    assert sp.day.tolist()[2:] == [1, 2, 1, 2, 3, 1]
    assert sp.len.tolist()[2:] == [2, 2, 3, 3, 3, 1]
    assert sp.cens.tolist()[2:] == ["시작 미상", "시작 미상", "", "", "", "진행 중"]
    one = spells(pd.Series([True] * 3, index=_IDX[:3], dtype="boolean"))
    assert one.cens.tolist() == ["시작 미상·진행 중"] * 3


def test_valid_from_units():
    bv = pd.Series([np.nan, 0.01, -0.049, -0.05, 0.2], index=_IDX[:5])
    assert breadth_valid_from(bv) == _IDX[3]
    k = pd.Series([10, 11, 12, 9, 13, 14, 15, 16], index=_IDX, dtype=float)
    mid = pd.Series([10, 9, 8, 7, 8, 9, 10, 11], index=_IDX, dtype=float)
    # 2일선: 둘 다 생기는 날 = 2번째. 괴리형(둘 다 위) 첫 성립 = k 13>10.5 & mid 8>7.5 → 5번째
    assert c2_valid_from(k, mid, MarketLayerConfig(c2_ma=2)) == _IDX[4]


def test_days_since_shock():
    op = pd.Series([100, 100, 96.0, 97, 97, 97, 97, 97], index=_IDX)
    cl = pd.Series([100, 100, 96.5, 97, 97, 97, 97, 97], index=_IDX)
    r = pd.Series(0.5, index=_IDX)
    h, _ = build_history(op, cl, cl * 2, r, MarketLayerConfig(c2_ma=2, breadth_median_min=1, breadth_median_win=3))
    assert h.shock.tolist() == [False, False, True] + [False] * 5
    assert h.days_since_shock.isna().tolist()[:2] == [True, True]
    assert h.days_since_shock.tolist()[2:] == [0, 1, 2, 3, 4, 5]


def test_attach_regime_d1_and_lag(hist):
    h, _ = hist
    v = h.dropna(subset=["regime"])
    t1, t2 = v.index[100], v.index[101]
    df = pd.DataFrame({"ticker": ["A", "B", "C"], "D1": [t2, t2 + pd.Timedelta(hours=15), pd.Timestamp("2030-01-05")]})
    a = attach_regime(df, "D1", hist=h)
    assert a.mkt_regime.iloc[0] == h.loc[t2, "regime"] and a.mkt_regime.iloc[1] == h.loc[t2, "regime"]  # 시각 무시
    assert pd.isna(a.mkt_regime.iloc[2])                                                               # 표 밖 = 결측
    assert a.mkt_breadth_day.iloc[0] == h.loc[t2, "breadth_day"]
    b = attach_regime(df, "D1", hist=h, lag=1)
    assert b.mkt_regime.iloc[0] == h.loc[t1, "regime"]                                                 # 전날 라벨
    assert str(a.mkt_breadth.dtype) == "boolean"


@pytest.fixture
def llv_root(tmp_path):
    """가짜 LLV — 300일 확정 재료 + 마지막 날은 한국투자 잠정만 (표에는 확정 마지막 날까지만 실려야 한다)."""
    if str(_LLV) not in sys.path:
        sys.path.insert(0, str(_LLV))
    from stolab_data import market_breadth as mb
    days = pd.bdate_range("2025-09-01", periods=300)
    tk = tmp_path / "data" / "ohlcv" / "tickers"; tk.mkdir(parents=True)
    px = 800 + np.arange(300, dtype=float)
    pd.DataFrame({"Date": days, "Ticker": "KOSPI200", "Open": px, "Close": px}).to_parquet(tk / "KOSPI200.parquet")
    pd.DataFrame({"Date": days[:-1], "Ticker": "KRX_TMI_MID", "Close": px[:-1] * 2}).to_parquet(tk / "KRX_TMI_MID.parquet")
    adv = np.where(np.arange(299) < 200, 400, 650)
    rows = [{"Date": d, "Market": m, "Source": "KRX", "Phase": "close", "Adv": int(a), "Dec": 1000 - int(a), "Flat": 10}
            for d, a in zip(days[:-1], adv) for m in mb.MARKETS]
    rows += [{"Date": days[-1], "Market": m, "Source": "KIS", "Phase": "sig", "Adv": 700, "Uplm": 3, "Dec": 300, "Lslm": 0,
              "Flat": 20} for m in mb.MARKETS]
    mb.save_breadth(pd.DataFrame(rows), tmp_path / "data" / "market_breadth.parquet")
    return tmp_path, days


def test_update_history_uses_confirmed_only(llv_root, tmp_path):
    root, days = llv_root
    out = tmp_path / "o" / "regime_history.parquet"
    m = update_history(out_path=out, root=root)
    assert m["as_of"] == days[-2].strftime("%Y-%m-%d")             # 마지막 날(잠정)은 싣지 않는다
    h = pd.read_parquet(out)
    assert pd.to_datetime(h.index).max() == days[-2] and len(h) == 299
    assert out.with_suffix(".meta.json").exists() and m["breadth_valid_from"] is not None
    assert h.breadth.dropna().astype(bool).iloc[-1]                 # 비율이 0.40 → 0.65 로 오른 뒤 상승
