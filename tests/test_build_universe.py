"""tests/test_build_universe.py — 스윙 유니버스 월 파일 생성 (WP4, 2026-10-29).

고정하는 계약: 기준일 이전 마지막 유효값으로 시총 ≥ 4조 & 외국인 ≥ 30% · 우선주·스팩·관리·ETF 제외 ·
기준일 근처 값이 없는 종목 제외 · 패널 밖 4조↑ 는 경고만 · apply_month = 다음 달(12월 → 다음 해 1월).
실데이터가 있으면 9월 파일(8/31 기준 44종목)을 그대로 재현해야 한다.
"""
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

KMS = Path(__file__).resolve().parents[1] / "korean_mkt_study"
sys.path.insert(0, str(KMS))
import build_universe as BU  # noqa: E402


def _setup(tmp_path):
    llv = tmp_path / "llv"; (llv / "ohlcv").mkdir(parents=True); (llv / "raw").mkdir()
    rows = []
    def add(tk, nm, cap, fr, last="2026-10-30"):
        for d in pd.bdate_range("2026-10-01", last):
            rows.append({"Date": d, "Ticker": tk, "Name": nm, "MarketCap": cap, "Foreign_Ratio": fr})
    add("000010", "대형외인", 5e12, 35.0)
    add("000020", "대형내국", 5e12, 20.0)          # 외인 미달
    add("000030", "중형외인", 3e12, 50.0)          # 시총 미달
    add("000040", "스팩", 6e12, 40.0)              # meta 스팩
    add("000055", "우선주", 6e12, 40.0)            # 끝자리 ≠ 0
    add("102110", "TIGER200", 9e12, 40.0)          # ETF
    add("000060", "옛값", 6e12, 40.0, last="2026-10-14")   # 기준일 근처 값 없음
    pd.DataFrame(rows).to_parquet(llv / "ohlcv" / "panel.parquet")
    meta = pd.DataFrame([{"date": pd.Timestamp("2026-09-30"), "ticker": t, "is_pref": False,
                          "is_spac": t == "000040", "is_managed": False}
                         for t in ("000010", "000020", "000030", "000040", "000055", "000060")]).set_index(["date", "ticker"])
    mp = tmp_path / "meta.parquet"; meta.to_parquet(mp)
    raw = pd.DataFrame([{"Ticker": "000010", "Name": "대형외인", "MarketCap": 5e12},
                        {"Ticker": "000990", "Name": "패널밖대형", "MarketCap": 7e12},
                        {"Ticker": "000995", "Name": "패널밖우선", "MarketCap": 7e12}])
    raw.to_parquet(llv / "raw" / "krx_kospi_20261028.parquet")
    return llv, mp


def test_filters_and_warnings(tmp_path):
    llv, mp = _setup(tmp_path)
    r = BU.build(date(2026, 10, 30), llv=llv, meta_path=mp, min_count=0)
    assert [s["ticker"] for s in r["stocks"]] == ["000010"]
    assert r["apply_month"] == "2026-11" and r["base_date"] == "2026-10-30"
    pv = r["data_provenance"]
    assert pv["stale_excluded"] == ["000060"]
    assert [o["ticker"] for o in pv["outside_panel_4jo"]] == ["000990"]
    assert pv["krx_raw_date"] == "2026-10-28"


def test_december_rolls_to_january():
    assert BU.next_month(date(2026, 12, 30)) == "2027-01"


def test_uses_value_on_or_before_base(tmp_path):
    llv, mp = _setup(tmp_path)
    r = BU.build(date(2026, 10, 27), llv=llv, meta_path=mp, min_count=0)
    assert r["apply_month"] == "2026-10" and r["effective_date"] == "2026-10-28" and r["count"] == 1


@pytest.mark.skipif(not (KMS / "data" / "universe_2026-09_4jo_30pct.json").exists()
                    or not (BU.LLV / "ohlcv" / "panel.parquet").exists(), reason="실데이터 없음")
def test_reproduces_september_file():
    old = json.loads((KMS / "data" / "universe_2026-09_4jo_30pct.json").read_text(encoding="utf-8"))
    r = BU.build(date(2026, 8, 31))
    assert sorted(s["ticker"] for s in r["stocks"]) == sorted(s["ticker"] for s in old["stocks"])


# ── U6 확정 (케인 2026-10-31): 최소 30종목 · 채움은 LLV 패널 안에서만 · 월 중간 적용(effective_date) ──

WD = lambda d: d.weekday() < 5          # noqa: E731 — 거래일 캘린더 대용(주말만 휴장)


def test_floor_fills_from_panel_by_mktcap(tmp_path):
    llv, mp = _setup(tmp_path)
    r = BU.build(date(2026, 10, 30), llv=llv, meta_path=mp, min_count=2)
    assert [s["ticker"] for s in r["stocks"]] == ["000010", "000030"]          # 4조/30% 1 + 바닥(외인 50%·3조) 1
    assert r["count_rule"] == 1 and r["floor_filled"] == ["000030"] and r["floor_shortfall"] == 0
    assert [s["rule"] for s in r["stocks"]] == ["4jo_30pct", "floor"]
    assert "000020" not in r["floor_filled"]                                   # 외인 20% 는 바닥 후보 아님


def test_floor_never_goes_outside_panel(tmp_path):
    llv, mp = _setup(tmp_path)
    r = BU.build(date(2026, 10, 30), llv=llv, meta_path=mp, min_count=30)
    tk = {s["ticker"] for s in r["stocks"]}
    assert "000990" not in tk                                                  # KRX 원본에만 있는 종목은 채우지 않음
    assert tk == {"000010", "000030"} and r["floor_shortfall"] == 28
    assert r["filters"]["min_count"] == 30


def test_floor_noop_when_rule_has_enough(tmp_path):
    llv, mp = _setup(tmp_path)
    r = BU.build(date(2026, 10, 30), llv=llv, meta_path=mp, min_count=1)
    assert r["floor_filled"] == [] and r["count"] == 1


def test_expiry_week_friday_and_effective():
    assert BU.expiry_week_friday(2026, 11, WD) == date(2026, 11, 13)          # 둘째 목 11-12 → 금 11-13
    assert BU.expiry_week_friday(2026, 10, WD) == date(2026, 10, 9)
    hol = lambda d: WD(d) and d != date(2026, 11, 13)                          # noqa: E731 — 금요일 휴장이면 목요일
    assert BU.expiry_week_friday(2026, 11, hol) == date(2026, 11, 12)
    assert BU.next_trading_day(date(2026, 11, 13), WD) == date(2026, 11, 16)


def test_effective_date_mid_month(tmp_path):
    llv, mp = _setup(tmp_path)
    r = BU.build(date(2026, 10, 30), llv=llv, meta_path=mp, min_count=0, effective=date(2026, 11, 16))
    assert r["effective_date"] == "2026-11-16" and r["apply_month"] == "2026-11" and r["base_date"] == "2026-10-30"
