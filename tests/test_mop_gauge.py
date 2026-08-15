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
    def test_10sets_75tickers(self):
        """8세트 44종목(07-30) → 10세트 62종목(08-12) → 10세트 75종목(08-15)."""
        assert len(gcfg.SECTOR_SETS) == 10
        assert sum(len(v) for v in gcfg.SECTOR_SETS.values()) == 75

    def test_all_sets_are_weight_dicts(self):
        """2026-08-15 고정가중 전환 — 세트는 {티커: 가중치} dict 여야 한다."""
        for name, spec in gcfg.SECTOR_SETS.items():
            assert isinstance(spec, dict), f"{name} — dict 여야 함 (고정가중)"

    def test_ticker_format(self):
        for name, tks in gcfg.SECTOR_SETS.items():
            for t in tks:
                assert isinstance(t, str) and len(t) == 6 and t.isdigit(), \
                    f"{name}/{t} — 6자리 숫자 문자열이어야 함"

    def test_weights_positive_and_near_one(self):
        """가중치는 양수. 세트 합은 정규화하지 않으므로 반올림 오차 ±0.02 허용."""
        for name, spec in gcfg.SECTOR_SETS.items():
            for t, w in spec.items():
                assert isinstance(w, (int, float)) and w > 0, f"{name}/{t} 가중치"
            s = sum(spec.values())
            assert 0.98 <= s <= 1.02, f"{name} 가중치 합 {s:.4f} — 1.0 에서 이탈"

    def test_renamed_and_removed(self):
        """2026-08-15 개편 — K_조선레 개칭 + 삭제 종목 반영."""
        assert "S_조선레" in gcfg.SECTOR_SETS
        assert "K_조선레" not in gcfg.SECTOR_SETS
        assert "000150" not in gcfg.SECTOR_SETS["K_반.핵심장비"]   # 두산
        assert "000500" not in gcfg.SECTOR_SETS["T_전력기기"]      # 가온전선
        for t in ("403870", "357780", "095340"):                   # HPSP·솔브레인·ISC
            assert t not in gcfg.SECTOR_SETS["T_반도체레"]
        assert "403870" in gcfg.SECTOR_SETS["K_반.핵심장비"]        # HPSP 는 이동

    def test_no_dup_within_set(self):
        # ⚠ 세트 '간' 중복은 의도된 것 (Kane 2026-08-12) — 세트별 독립 집계라
        #    같은 종목이 여러 세트에 들어가도 무방하다. 여기선 세트 '안' 만 본다.
        #    dict 라 키 중복은 구조적으로 불가하나, list 회귀 시를 대비해 남긴다.
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
# gauge_core — 고정가중 (Kane 전환 2026-08-15, 정본 경로)
# ════════════════════════════════════════════════════════
class TestFixedWeights:
    def test_fixed_weight_ignores_mcap(self):
        """dict 세트면 시총을 무시하고 지정 가중치를 쓴다. 0.7·0.8+0.3·0.4=0.68"""
        by = _score_df([("000001", "A", 0.8, 1, 100.0, False),
                        ("000002", "B", 0.4, 2, 50.0, False)])
        mcap = pd.Series({"000001": 1e12, "000002": 1.0})   # 시총은 극단적으로 편향
        out = aggregate_sets(by, mcap, {"S": {"000001": 0.7, "000002": 0.3}},
                             {"000001": "A", "000002": "B"})
        s = out[0]
        assert s["weight_mode"] == "fixed"
        assert s["weighted_p"] == pytest.approx(0.68)
        assert s["weight_sum"] == pytest.approx(1.0)
        assert s["members"][0]["weight"] == pytest.approx(0.7)
        assert s["members"][0]["weight_def"] == pytest.approx(0.7)

    def test_set_sum_not_normalized(self):
        """합이 0.99 인 세트는 그대로 0.99 배가 실린다 (Kane 결정 — 정규화 없음)."""
        by = _score_df([("000001", "A", 1.0, 1, 100.0, False)])
        out = aggregate_sets(by, pd.Series({"000001": 100.0}),
                             {"S": {"000001": 0.99}}, {})
        assert out[0]["weighted_p"] == pytest.approx(0.99)
        assert out[0]["weight_sum"] == pytest.approx(0.99)

    def test_missing_member_weight_redistributed(self):
        """유니버스 밖 종목의 가중치는 남은 종목에 비례 재분배 (세트 합 보존)."""
        by = _score_df([("000001", "A", 0.8, 1, 100.0, False),
                        ("000002", "B", 0.4, 2, 50.0, False)])
        mcap = pd.Series({"000001": 100.0, "000002": 100.0})
        out = aggregate_sets(
            by, mcap,
            {"S": {"000001": 0.5, "000002": 0.3, "999999": 0.2}},
            {"999999": "유니버스밖"})
        s = out[0]
        assert s["n_members"] == 3 and s["n_scored"] == 2
        assert s["weight_sum"] == pytest.approx(1.0)          # 0.8 → 1.0 으로 복원
        assert s["members"][0]["weight"] == pytest.approx(0.5 / 0.8)
        assert s["weighted_p"] == pytest.approx((0.5 * 0.8 + 0.3 * 0.4) / 0.8)
        gone = [m for m in s["members"] if m["p"] is None][0]
        assert gone["name"] == "유니버스밖" and gone["weight"] is None
        assert gone["weight_def"] == pytest.approx(0.2)

    def test_fixed_zfill(self):
        by = _score_df([("000660", "SK하이닉스", 0.9, 1, 100.0, False)])
        out = aggregate_sets(by, pd.Series({"000660": 100.0}), {"S": {660: 1.0}}, {})
        assert out[0]["n_scored"] == 1 and out[0]["weighted_p"] == pytest.approx(0.9)

    def test_fixed_all_missing(self):
        out = aggregate_sets(_score_df([]), pd.Series(dtype=float),
                             {"S": {"111111": 1.0}}, {})
        s = out[0]
        assert s["weighted_p"] is None and s["weight_sum"] == 0.0

    def test_real_config_runs(self):
        """실제 gauge_config 로 집계가 통과하는지 (스모크)."""
        rows, mc = [], {}
        for i, t in enumerate(sorted({t for s in gcfg.SECTOR_SETS.values() for t in s})):
            rows.append((t, f"N{i}", 0.5, i + 1, 1000.0, False))
            mc[t] = 1e9
        out = aggregate_sets(_score_df(rows), pd.Series(mc),
                             gcfg.SECTOR_SETS, {})
        assert len(out) == 10
        for s in out:
            assert s["weight_mode"] == "fixed"
            # 전원 스코어 → 지정 합이 그대로, p 전부 0.5 → weighted_p = 합 × 0.5
            assert s["weighted_p"] == pytest.approx(s["weight_sum"] * 0.5)


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
