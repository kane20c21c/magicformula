"""데이포트 유니버스 규칙 (2026-11-03 케인 U7 확정 → v1.2.2.5): 거래대금 50억 · 2회 연속 미달 편출 · 만기 주 금요일 점검 · 11-16 적용."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "mop_model" / "src"
sys.path.insert(0, str(SRC))
import day_universe as DU          # noqa: E402
import classification as CL        # noqa: E402


def test_check_dates_expiry_week_friday():
    days = pd.bdate_range("2026-09-01", "2026-11-30")
    c = [str(x.date()) for x in DU.check_dates(days)]
    assert c == ["2026-09-11", "2026-10-09", "2026-11-13"]
    days2 = days[days != pd.Timestamp("2026-11-13")]                    # 금요일 휴장 → 그 주 마지막 거래일
    assert str(DU.check_dates(days2)[-1].date()) == "2026-11-12"


def _px(amounts_by_ticker, days):
    rows = []
    for t, f in amounts_by_ticker.items():
        for i, d in enumerate(days):
            rows.append({"Date": d, "Ticker": t, "Amount": f(d, i), "Close": 1.0, "Volume": 1.0})
    return pd.DataFrame(rows)


def test_two_consecutive_below_then_reentry():
    days = pd.bdate_range("2026-06-01", "2026-12-31")
    low = lambda d, i: 30e8 if pd.Timestamp("2026-07-01") <= d <= pd.Timestamp("2026-09-30") else 80e8   # noqa: E731
    px = _px({"000010": low, "000020": lambda d, i: 80e8}, days)
    S, A = DU.status_table(px)
    st = S["000010"]; cd = [str(x.date()) for x in S.index]
    assert st[cd.index("2026-07-10")]                  # 20일 평균 60억 → 통과
    assert st[cd.index("2026-08-14")]                  # 1회 미달 → 유지
    assert not st[cd.index("2026-09-11")]              # 2회 연속 미달 → 편출
    assert not st[cd.index("2026-10-09")]              # 47.5억 → 여전히 미달
    assert st[cd.index("2026-11-13")]                  # 50억 회복 → 재편입
    assert round(A["000010"][cd.index("2026-10-09")] / 1e8, 1) == 47.5
    assert S["000020"].all()


def test_mask_applies_from_next_trading_day_and_active_from():
    days = pd.bdate_range("2026-06-01", "2026-12-31")
    low = lambda d, i: 30e8 if pd.Timestamp("2026-07-01") <= d <= pd.Timestamp("2026-09-30") else 80e8   # noqa: E731
    px = _px({"000010": low}, days)
    m = DU.membership_mask(px, today="2026-11-16")
    on = dict(zip(px.Date.dt.strftime("%Y-%m-%d"), m))
    assert on["2026-09-11"] == True                                      # 점검 당일은 직전 상태
    assert on["2026-09-14"] == False and on["2026-11-13"] == False        # 다음 거래일부터 편출, 재편입 점검 당일까지
    assert on["2026-11-16"] == True
    assert DU.membership_mask(px, today="2026-11-13").all()              # 적용 전(ACTIVE_FROM 이전) → 거르지 않음


def test_classification_overrides(tmp_path):
    base = {"classifications": {"a": {"ticker": "068270", "name": "셀트리온", "sector": "바이오"},
                                "b": {"ticker": "005930", "name": "삼성전자", "sector": "반도체.핵심"}}}
    (tmp_path / "c.json").write_text(json.dumps(base, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "o.json").write_text(json.dumps({"overrides": {"068270": {"name": "셀트리온", "sector": "생활.제약"},
                                                               "096530": {"name": "씨젠", "sector": "바이오.서비스"}}}, ensure_ascii=False), encoding="utf-8")
    c = CL.load_classification(tmp_path / "c.json", tmp_path / "o.json")
    assert c["068270"]["sector"] == "생활.제약" and c["096530"]["sector"] == "바이오.서비스" and c["005930"]["sector"] == "반도체.핵심"


def test_repo_overrides_file_is_valid():
    d = json.load(open(SRC.parent / "sector_overrides.json", encoding="utf-8"))["overrides"]
    assert len(d) == 32 and d["068270"]["sector"] == "생활.제약" and d["207940"]["sector"] == "생활.제약"
    assert all("." in v["sector"] or v["sector"] in {"엔터", "건설", "조선", "로봇", "방산", "지주", "IT서비스", "기계장비", "자동차", "자동차소부장", "기타"} for v in d.values())
