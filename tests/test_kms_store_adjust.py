"""연구 정본 연장의 수정계수 (korean_mkt_study/kms_store.adjust_extension, 2026-11-15) — 합성 가격.

계수 f_t = 기준가 ÷ 전일 원종가. 새 행 = 원주가 × Π(이후 계수), 저장 이력 = 저장값 × Π(전체 계수) ÷ s
(s = 저장 seam 종가 ÷ 원 seam 종가 — 저장본이 이미 반영한 권리락을 두 번 곱하지 않는다).
실측 검증(2026-11-15): 9-11 까지로 잘라 9-14~10-29 를 이 식으로 다시 만들면 교정된 저장본과 70,279행 중
0.5% 넘게 다른 행은 30개 — 전부 10-30 권리락 종목(창 밖 사건, 기대된 차이).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "korean_mkt_study"))
import kms_store as K  # noqa: E402

S = pd.Timestamp("2026-10-29")


def _store(tk, close, s=1.0):
    idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2026-10-28"), tk), (S, tk)], names=["date", "ticker"])
    return pd.DataFrame({"open": [c * s for c in close], "high": [c * s for c in close], "low": [c * s for c in close],
                         "close": [c * s for c in close], "volume": [100.0 / s, 100.0 / s]}, index=idx)


def _raw(tk, rows):
    return pd.DataFrame([{"date": pd.Timestamp(d), "Ticker": tk, "Open": c, "High": c, "Low": c, "Close": c,
                          "Volume": 10.0, "Base": b} for d, c, b in rows])


def test_no_event_appends_raw():
    st = _store("000010", [1000, 1010])
    raw = _raw("000010", [("2026-10-30", 1020, 1010), ("2026-11-02", 1030, 1020)])
    new, hm = K.adjust_extension(st, raw, S, pd.DataFrame({"Ticker": ["000010"], "Close": [1010.0]}))
    assert hm == {} and new["close"].tolist() == [1020, 1030] and new["volume"].tolist() == [10, 10]


def test_reverse_split_in_window_scales_history_and_earlier_new_rows():
    # 11-02 에 5:1 병합 → 기준가 = 1030×5. 그 전 새 행(10-30)과 저장 이력이 ×5, 거래량 ÷5
    st = _store("000020", [1000, 1010])
    raw = _raw("000020", [("2026-10-30", 1030, 1010), ("2026-11-02", 5200, 5150)])
    new, hm = K.adjust_extension(st, raw, S, pd.DataFrame({"Ticker": ["000020"], "Close": [1010.0]}))
    assert np.isclose(hm["000020"], 5.0)
    assert np.allclose(new["close"].to_numpy(), [5150.0, 5200.0]) and np.allclose(new["volume"].to_numpy(), [2.0, 10.0])


def test_event_already_in_store_is_not_applied_twice():
    # 저장본(수집 시점 수정가)이 이미 ×5 를 반영(s=5) + 새 구간 첫날 기준가로 같은 사건이 보임 → 이력 배수 1
    st = _store("000030", [1000, 1010], s=5.0)
    raw = _raw("000030", [("2026-10-30", 5100, 5050)])
    new, hm = K.adjust_extension(st, raw, S, pd.DataFrame({"Ticker": ["000030"], "Close": [1010.0]}))
    assert hm == {} and new["close"].tolist() == [5100]


def test_small_base_gap_is_noise():
    # 스팩 기준가가 전일 종가와 0.15% 다른 실측 사례 — 사건으로 보지 않는다 (EVENT_TOL 0.3%)
    st = _store("0071M0", [2000, 2007])
    raw = _raw("0071M0", [("2026-10-30", 1998, 2010)])
    new, hm = K.adjust_extension(st, raw, S, pd.DataFrame({"Ticker": ["0071M0"], "Close": [2007.0]}))
    assert hm == {} and new["close"].tolist() == [1998]


def test_new_listing_has_no_history_factor():
    st = _store("000010", [1000, 1010])
    raw = _raw("999990", [("2026-11-02", 15000, 10000), ("2026-11-03", 14000, 15000)])
    new, hm = K.adjust_extension(st, raw, S, pd.DataFrame({"Ticker": ["000010"], "Close": [1010.0]}))
    assert hm == {} and new["close"].tolist() == [15000, 14000]
