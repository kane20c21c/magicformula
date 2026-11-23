# -*- coding: utf-8 -*-
"""tests/test_jdh_rule.py — 장단현포트(룰 #6) 판정 회귀 (2026-11-23 케인)

  - 이동 규칙: 선행 3개 ±2 · 정확히 2개 ±1 · 불완전 0 / KOSPI200 부호 ±1
  - 틀: ⑦ 보유 → cash(롱/현금, K200 신호) · ⑦ 현금 → short(롱/숏, 선행 신호) · ⑦ 낡음 → None(틀 유지)
  - ⑦ 낡음 판정 (판정일 = 직전 거래일만 사용)
  - 연구 재현: LLV 자료로 고른 2026-10·11월 선행종목 = 연구(k200_lead_walkforward.py 월별 선정)와 같다
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from jdh_rule import judge, lead_move, k200_move, latest_p7
from jdh_rule.config import PANEL, KOSPI200_PARQUET, LLV_ROOT

TODAY = date(2026, 11, 24)
PREV = [date(2026, 11, 23), date(2026, 11, 20), date(2026, 11, 19), date(2026, 11, 18), date(2026, 11, 17)]


def test_lead_move():
    assert lead_move([0.01, 0.02, 0.03]) == (2, 3, 0)
    assert lead_move([-0.01, -0.02, -0.03]) == (-2, 0, 3)
    assert lead_move([0.01, 0.02, -0.03]) == (1, 2, 1)
    assert lead_move([0.01, -0.02, -0.03]) == (-1, 1, 2)
    assert lead_move([0.01, 0.0, -0.03])[0] == 0            # 1:1 (0 은 방향 없음)
    assert lead_move([0.01, None, 0.03])[0] == 0            # 불완전 → 매매 안 함


def test_k200_move():
    assert k200_move(0.003) == 1 and k200_move(-0.01) == -1 and k200_move(0.0) == 0 and k200_move(None) == 0


def _p7(v, usable=True):
    return {"date": "2026-11-23", "p7": v, "label": "보유" if v else "현금", "stale_td": 0, "usable": usable}


def _leads(r):
    return [{"ticker": f"00000{i}", "name": "", "base_close": 100.0, "now": 100.0 * (1 + x)} for i, x in enumerate(r)]


def test_judge_frames():
    k_up = {"base_close": 800.0, "now": 820.0}
    a = judge(TODAY, _leads([-0.01, -0.02, -0.03]), k_up, _p7(True))
    assert a["frame"] == "cash" and a["move"] == 1 and a["move_source"] == "k200"       # ⑦ 보유 → K200
    b = judge(TODAY, _leads([-0.01, -0.02, -0.03]), k_up, _p7(False))
    assert b["frame"] == "short" and b["move"] == -2 and b["move_source"] == "lead"     # ⑦ 현금 → 선행
    c = judge(TODAY, _leads([0.01, 0.02, 0.03]), k_up, _p7(True, usable=False))
    assert c["frame"] is None and c["move"] is None and c["lead"]["move"] == 2 and c["k200"]["move"] == 1
    assert abs(a["k200"]["ret5"] - 0.025) < 1e-12


def test_latest_p7_stale(tmp_path):
    idx = pd.to_datetime(["2026-11-19", "2026-11-20", "2026-11-23", "2026-11-24"])
    pd.DataFrame({"p7": [False, True, False, True]}, index=idx).to_parquet(tmp_path / "h.parquet")
    r = latest_p7(TODAY, PREV, tmp_path / "h.parquet")
    assert r["date"] == "2026-11-23" and r["p7"] is False and r["usable"] and r["stale_td"] == 0   # 오늘 행은 안 본다
    pd.DataFrame({"p7": [False, True]}, index=idx[:2]).to_parquet(tmp_path / "h2.parquet")
    r2 = latest_p7(TODAY, PREV, tmp_path / "h2.parquet")
    assert r2["date"] == "2026-11-20" and r2["stale_td"] == 1 and not r2["usable"]


@pytest.mark.skipif(not PANEL.exists() or not KOSPI200_PARQUET.exists(), reason="LLV 패널 없음")
def test_picks_reproduce_research():
    from jdh_rule.picks import select_leads, load_closes, load_k200
    u = json.loads((LLV_ROOT / "groups" / "universe_2026-11-16.json").read_text(encoding="utf-8"))["members"]
    px, k = load_closes([str(t).zfill(6) for t in u]), load_k200()
    assert [x["ticker"] for x in select_leads(px, k, "2026-11-02")["leads"]] == ["041510", "052690", "120110"]
    assert [x["ticker"] for x in select_leads(px, k, "2026-10-01")["leads"]] == ["030000", "041510", "036460"]


def test_ops_does_not_read_research():
    """한 방향 원칙 — 운영 코드는 연구 폴더를 읽지 않는다."""
    root = Path(__file__).resolve().parents[1]
    for f in list((root / "jdh_rule").glob("*.py")) + [root / "scripts" / "jdh_signal.py"]:
        assert "korean_mkt_study" not in f.read_text(encoding="utf-8"), f
