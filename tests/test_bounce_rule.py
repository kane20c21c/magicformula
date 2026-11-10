# -*- coding: utf-8 -*-
"""반등 룰 회귀 테스트 — 합성 시계열 + 연구 코드 재현(파일 있을 때만)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bounce_rule import scan_drives, BounceConfig, DEFAULT


def _series(ticker, rets, start="2026-01-01", base=10000.0):
    dates = pd.bdate_range(start, periods=len(rets) + 1)
    close = base * np.cumprod(np.concatenate([[1.0], 1 + np.asarray(rets)]))
    return pd.DataFrame({"Date": dates, "Ticker": ticker, "Name": ticker, "Close": close})


def _calm(n=80, seed=0):
    return list(np.random.default_rng(seed).normal(0, 0.01, n))     # σ ≈ 1%


def test_signal_fires_on_deep_drive_and_strong_bounce():
    rets = _calm() + [-0.07, -0.08, -0.07, -0.05] + [+0.06]           # 4일 -24.5%, D+1 +6% (≈6σ)
    out = scan_drives(_series("A", rets))
    assert len(out) == 1 and bool(out.signal.iloc[0]) and out.drive_days.iloc[0] == 4
    assert out.drive_drop.iloc[0] < -0.20 and out.strength.iloc[0] >= 1.65 and not out.deadcat.iloc[0]


def test_weak_bounce_is_deadcat_not_signal():
    rets = _calm() + [-0.07, -0.08, -0.07, -0.05] + [+0.001]
    out = scan_drives(_series("B", rets))
    assert len(out) == 1 and not out.signal.iloc[0] and bool(out.deadcat.iloc[0])


def test_shallow_drive_no_signal_even_if_strong():
    rets = _calm() + [-0.02, -0.02, -0.02] + [+0.06]                   # 낙폭 -5.9%
    out = scan_drives(_series("C", rets))
    assert len(out) == 1 and not out.signal.iloc[0]


def test_drive_in_progress_or_short_not_listed():
    assert scan_drives(_series("D", _calm() + [-0.07, -0.08, -0.07, -0.05])).empty          # 오늘도 하락
    assert scan_drives(_series("E", _calm() + [-0.10, -0.12] + [+0.08])).empty                # 2일 — 드라이브 아님
    assert scan_drives(_series("F", _calm(60))).empty                                          # 이력 부족


def test_sigma_uses_history_before_bounce_and_asof_filter():
    rets = _calm() + [-0.07, -0.08, -0.07, -0.05] + [+0.06] + [+0.01]
    df = _series("G", rets)
    asof = df.Date.iloc[-2]
    out = scan_drives(df, asof=asof)
    assert len(out) == 1 and out.date.iloc[0] == asof.strftime("%Y-%m-%d")
    r = np.array(rets); expect_sigma = np.std(r[len(r)-2-60:len(r)-2], ddof=1)   # D+1(끝에서 둘째) 제외, 드라이브 종료일까지 60일
    assert abs(out.sigma60.iloc[0] - expect_sigma) < 1e-5                              # sigma60 은 소수 5자리 반올림


RESEARCH = ROOT / "korean_mkt_study" / "out" / "crash" / "drive_hold2.parquet"
PRICES = ROOT / "korean_mkt_study" / "data_2015" / "prices.parquet"


@pytest.mark.skipif(not (RESEARCH.exists() and PRICES.exists()), reason="연구 산출물 없음 (git 제외)")
def test_reproduces_research_on_a_signal_day():
    """연구(crash_study 5단계)가 신호로 분류한 날짜·종목을 운영 스캔이 그대로 낸다."""
    e = pd.read_parquet(RESEARCH); e = e[(e.B_sig >= 1.65) & (e.tot <= -0.20)]
    e["d1"] = pd.to_datetime(e.date)
    day = e.groupby("d1").size().sort_values().index[-3]            # 신호가 여럿인 날 하나
    expect = set(e[e.d1 == day].ticker)
    px = pd.read_parquet(PRICES, columns=["close"]).reset_index().rename(columns={"date": "Date", "ticker": "Ticker", "close": "Close"})
    px = px[px.Ticker.isin(expect) & (px.Date <= day + pd.Timedelta(days=1)) & (px.Date >= day - pd.Timedelta(days=200))]
    # 연구의 D+1 = 드라이브 종료 다음 날 — 운영은 그날(D+1)을 asof 로 본다
    nxt = sorted(px.Date.unique()); d1 = [d for d in nxt if d > day][0]
    px_all = pd.read_parquet(PRICES, columns=["close"]).reset_index().rename(columns={"date": "Date", "ticker": "Ticker", "close": "Close"})
    px_all = px_all[px_all.Ticker.isin(expect) & (px_all.Date <= d1) & (px_all.Date >= day - pd.Timedelta(days=200))]
    out = scan_drives(px_all, asof=d1)
    got = set(out[out.signal].ticker)
    assert expect <= got, f"연구 신호 {len(expect)} 중 운영 누락 {sorted(expect - got)[:5]}"


# ── 신호일 시장 국면 표기 (2026-11-10 케인 (가)) ──────────────────────────────────
import json as _json  # noqa: E402
from bounce_rule.regime import market_regime, regime_html  # noqa: E402


def _ml_json(d, day, prov=None, conf=None):
    j = {"as_of": day, "provisional": None, "confirmed": None}
    if prov: j["provisional"] = {"components": prov}
    if conf: j["confirmed"] = {"components": conf}
    (d / f"{day.replace('-', '')}.json").write_text(_json.dumps(j, ensure_ascii=False), encoding="utf-8")


def test_regime_prefers_confirmed_then_provisional(tmp_path):
    _ml_json(tmp_path, "2026-11-10", prov={"breadth": "하락", "c2": "보유", "breadth_value": -0.06},
             conf={"breadth": "상승", "c2": "퇴출", "breadth_value": 0.06})
    r = market_regime("2026-11-10", tmp_path)
    assert r["stage"] == "확정" and r["breadth"] == "상승" and r["c2"] == "퇴출"
    _ml_json(tmp_path, "2026-11-11", prov={"breadth": "하락", "c2": "보유", "breadth_value": -0.06})
    r = market_regime("2026-11-11", tmp_path)
    assert r["stage"] == "잠정" and r["breadth"] == "하락"


def test_regime_falls_back_to_history_then_none(tmp_path):
    idx = pd.to_datetime(["2026-11-06", "2026-11-09"])
    h = pd.DataFrame({"breadth": pd.array([True, pd.NA], dtype="boolean"), "c2": pd.array([False, True], dtype="boolean"),
                      "breadth_value": [0.07, np.nan]}, index=idx)
    h.to_parquet(tmp_path / "regime_history.parquet")
    r = market_regime("2026-11-06", tmp_path)
    assert r == {"breadth": "상승", "c2": "퇴출", "stage": "이력", "breadth_value": 0.07}
    assert market_regime("2026-11-09", tmp_path) is None          # 브레드스 결측(유효 구간 밖) → 없음
    assert market_regime("2026-11-12", tmp_path) is None


def test_regime_html_warns_only_on_breadth_up():
    up = regime_html({"breadth": "상승", "c2": "보유", "stage": "잠정"})
    dn = regime_html({"breadth": "하락", "c2": "보유", "stage": "잠정"})
    assert "⚠" in up and "40.0%" in up and "⚠" not in dn
    assert "판정 없음" in regime_html(None)
