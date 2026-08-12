# -*- coding: utf-8 -*-
"""
tests/test_mop_gauge.py — 섹터 오버나이트 게이지 회귀 테스트
=============================================================
대상: mop_model/src/gauge_core.py(집계), gauge_config.py(세트 정의),
      gauge_notify.py(본문 빌더 — 발송 없음), launchd plist 정합.
전부 LLV/KIS/모델 무관 — 네트워크·parquet 없이 실행 가능.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import pytest

_MOP_SRC = Path(__file__).resolve().parent.parent / "mop_model" / "src"
if str(_MOP_SRC) not in sys.path:
    sys.path.insert(0, str(_MOP_SRC))

import gauge_config as gcfg          # noqa: E402
from gauge_core import aggregate_sets  # noqa: E402
from gauge_notify import build_gauge_email, build_gauge_push  # noqa: E402


# ════════════════════════════════════════════════════════
# gauge_config — 세트 정의 무결성
# ════════════════════════════════════════════════════════
class TestGaugeConfig:
    def test_10sets_62tickers(self):
        """Kane 확정 2026-07-30 8세트 44종목 → 2026-08-12 10세트 62종목."""
        assert len(gcfg.SECTOR_SETS) == 10
        assert sum(len(v) for v in gcfg.SECTOR_SETS.values()) == 62

    def test_ticker_format(self):
        for name, tks in gcfg.SECTOR_SETS.items():
            for t in tks:
                assert isinstance(t, str) and len(t) == 6 and t.isdigit(), \
                    f"{name}/{t} — 6자리 숫자 문자열이어야 함"

    def test_no_dup_within_set(self):
        # ⚠ 세트 '간' 중복은 의도된 것 (Kane 2026-08-12) — 세트별 독립 집계라
        #    같은 종목이 여러 세트에 들어가도 무방하다. 여기선 세트 '안' 만 본다.
        for name, tks in gcfg.SECTOR_SETS.items():
            assert len(tks) == len(set(tks)), f"{name} 내 중복 티커"


# ════════════════════════════════════════════════════════
# gauge_core — 시총 가중평균 집계
# ════════════════════════════════════════════════════════
def _score_df(rows):
    df = pd.DataFrame(rows, columns=["Ticker", "Name", "p", "rank", "Close", "is_halt"])
    return df.set_index("Ticker")


class TestAggregateSets:
    def test_weighted_mean(self):
        """가중평균 = Σ(시총비중 × p). 시총 3:1 → 0.75·0.8 + 0.25·0.4 = 0.7"""
        by = _score_df([("000001", "A", 0.8, 1, 100.0, False),
                        ("000002", "B", 0.4, 2, 50.0, False)])
        mcap = pd.Series({"000001": 300.0, "000002": 100.0})
        out = aggregate_sets(by, mcap, {"S": ["000001", "000002"]},
                             {"000001": "A", "000002": "B"})
        s = out[0]
        assert s["weighted_p"] == pytest.approx(0.7)
        assert s["mean_p"] == pytest.approx(0.6)
        assert s["n_scored"] == 2 and s["n_members"] == 2
        assert s["top_member"]["ticker"] == "000001"
        assert s["members"][0]["weight"] == pytest.approx(0.75)

    def test_missing_member_renormalized(self):
        """미스코어 종목 제외 후 잔여 시총으로 재정규화. 표기는 p=None 으로 유지."""
        by = _score_df([("000001", "A", 0.6, 1, 100.0, False)])
        mcap = pd.Series({"000001": 100.0})
        out = aggregate_sets(by, mcap, {"S": ["000001", "999999"]},
                             {"999999": "누락종목"})
        s = out[0]
        assert s["weighted_p"] == pytest.approx(0.6)
        assert s["n_scored"] == 1 and s["n_members"] == 2
        none_m = [m for m in s["members"] if m["p"] is None][0]
        assert none_m["ticker"] == "999999" and none_m["name"] == "누락종목"

    def test_equal_weight_fallback_when_no_mcap(self):
        """시총 전부 결측 → 동일가중 폴백."""
        by = _score_df([("000001", "A", 0.8, 1, 100.0, False),
                        ("000002", "B", 0.4, 2, 50.0, False)])
        mcap = pd.Series(dtype=float)
        out = aggregate_sets(by, mcap, {"S": ["000001", "000002"]}, {})
        assert out[0]["weighted_p"] == pytest.approx(0.6)

    def test_zfill_normalization(self):
        """설정에 5자리로 들어와도 zfill 보정."""
        by = _score_df([("000660", "SK하이닉스", 0.9, 1, 100.0, False)])
        mcap = pd.Series({"000660": 100.0})
        out = aggregate_sets(by, mcap, {"S": [660]}, {})
        assert out[0]["n_scored"] == 1

    def test_empty_set(self):
        out = aggregate_sets(_score_df([]), pd.Series(dtype=float),
                             {"S": ["111111"]}, {})
        s = out[0]
        assert s["weighted_p"] is None and s["mean_p"] is None
        assert s["n_scored"] == 0


# ════════════════════════════════════════════════════════
# gauge_notify — 본문 빌더 (발송 없음)
# ════════════════════════════════════════════════════════
def _doc():
    return {
        "as_of": "2026-07-30", "price_time": "15:10", "universe_count": 195,
        "sets": [
            {"name": "K_방산레", "weighted_p": 0.861, "mean_p": 0.84,
             "n_members": 5, "n_scored": 5,
             "top_member": {"ticker": "012450", "name": "한화에어로스페이스", "p": 0.99},
             "members": [{"ticker": "012450", "name": "한화에어로스페이스",
                          "p": 0.99, "rank": 1, "close": 1000000.0,
                          "mcap": 1e12, "weight": 0.6, "is_halt": False}]},
            {"name": "K_은행", "weighted_p": 0.462, "mean_p": 0.45,
             "n_members": 4, "n_scored": 3,
             "top_member": {"ticker": "105560", "name": "KB금융", "p": 0.6},
             "members": [{"ticker": "105560", "name": "KB금융", "p": 0.6,
                          "rank": 40, "close": 90000.0, "mcap": 1e11,
                          "weight": 1.0, "is_halt": False},
                         {"ticker": "316140", "name": "우리금융지주", "p": None,
                          "rank": None, "close": None, "mcap": None,
                          "weight": None, "is_halt": False}]},
        ],
    }


class TestNotifyBuilders:
    def test_email_content(self):
        subj, html = build_gauge_email(_doc())
        assert "2026-07-30" in subj and "잠정" in subj
        assert "K_방산레" in html and "86.1%" in html
        assert "K_은행" in html and "46.2%" in html
        # Kane 표기 규칙 — 50% 위 빨강 / 아래 파랑
        assert "#ef5350" in html and "#1976D2" in html
        # 미스코어 종목 경고 표기
        assert "우리금융지주" in html
        # 잠정 고지 + 정본 안내
        assert "16:20" in html

    def test_push_content(self):
        title, msg = build_gauge_push(_doc())
        assert "잠정" in title
        assert "K_방산레" in msg and "86.1%" in msg
        # Pushover 는 <font color> 만 지원 — <span> 금지
        assert "<span" not in msg
        assert '<font color="#ef5350">' in msg


# ════════════════════════════════════════════════════════
# launchd plist — 15:10 스케줄 정합
# ════════════════════════════════════════════════════════
class TestPlist:
    def test_gauge_plist(self):
        p = (Path(__file__).resolve().parent.parent / "configs" / "launchd"
             / "com.kane.magicformula-mop-gauge.plist")
        assert p.exists()
        root = ET.parse(p).getroot()
        text = ET.tostring(root, encoding="unicode")
        assert "run_gauge.py" in text
        assert "<integer>15</integer>" in text and "<integer>10</integer>" in text
