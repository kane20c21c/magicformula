# -*- coding: utf-8 -*-
"""day_universe.py — 데이포트 유니버스 규칙 (2026-11-03 케인 U7 확정 → v1.2.2.5, 11-16 적용).

⚠ 2026-11-14 (케인 — 명단 정본 = LLV): **규칙 코드·명단 파일은 LLV `stolab_data/ticker_groups.py` 가 정본이다.**
  이 파일은 그 함수를 다시 내보내는 껍데기 + MF 고유값(ACTIVE_FROM) 만 갖는다. 규칙을 고치려면 LLV 를 고친다.
  명단 파일 = LLV data/groups/universe_{적용일}.json (LLV daily_update 08:01 이 점검일 다음 날 생성).
  종전 output/day_universe/day_universe_*.json 은 더 만들지 않는다 (git 미추적·에어 전용이라 미니에 없었다).

용어 (2026-11-14 케인): 이 결과를 대화·문서에서는 **유니버스**(관심 대상), 모집단 LLV 멤버(core+extend)는 **와이드버스** 라 부른다.

규칙 (근거: 볼트 '2026-11-01 U7 — 데이포트 유니버스 규칙'):
  · 대상 = 와이드버스 중 ETF·우선주가 아닌 종목 — **편입은 케인 판단(ScoutingTheNext)**, 시총 상하한 없음
  · 유지 조건 = 20일 평균 거래대금 ≥ **50억** — 점검일에 **2회 연속 미달이면 편출**, 다시 50억 이상이면 재편입
  · 점검일 = 매월 옵션 만기(둘째 목요일)가 든 주의 금요일(휴장이면 그 주 마지막 거래일), **다음 거래일부터** 적용
  · 관리 범위 260~310종목 · 대분류 섹터당 5종목 이상 (미달은 경고 — 자동 편출 없음)
학습·신호 모두 같은 멤버십으로 거른다(횡단면 피처·y_rel 이 그 유니버스 안에서 계산되게). 과거 날짜도 같은 규칙.
ACTIVE_FROM 이전 날짜에 신호를 만들면 필터를 걸지 않는다(배포 시점과 무관하게 11-16 부터 적용).

CLI (K7 회차 기록): python3 day_universe.py --report [--asof YYYY-MM-DD]
  → LLV data/groups/universe_{적용일}.json (멤버·편출·재편입·주의·섹터 수)
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

import config as cfg

if cfg.LLV_PATH not in sys.path:
    sys.path.insert(0, cfg.LLV_PATH)
from stolab_data import ticker_groups as _tg          # noqa: E402  ← 규칙 정본 (LLV)

AMOUNT_MIN = _tg.AMOUNT_MIN
AMOUNT_WIN, AMOUNT_MINP = _tg.AMOUNT_WIN, _tg.AMOUNT_MINP
OUT_AFTER = _tg.OUT_AFTER
RANGE_MIN, RANGE_MAX = _tg.RANGE_MIN, _tg.RANGE_MAX
ACTIVE_FROM = "2026-11-16"  # 11-13(11월 만기 주 금요일) 점검분부터 운영 적용 (MF 고유값)
RULE_ID = _tg.UNIVERSE_RULE_ID

check_dates = _tg.check_dates
status_table = _tg.status_table


def membership_mask(px: pd.DataFrame, today=None) -> pd.Series:
    """px 행별 멤버 여부 (LLV ticker_groups.membership_mask + ACTIVE_FROM)."""
    return _tg.membership_mask(px, today=today, active_from=ACTIVE_FROM)


def report(px: pd.DataFrame, cls: dict, asof=None, save: bool = False) -> dict:
    """마지막 점검(asof 이전) 결과 — LLV build_universe. save=True 면 LLV data/groups 에 덮어쓴다."""
    S, _ = status_table(px)
    if asof is not None:
        S = S[S.index <= pd.Timestamp(asof)]
    c = pd.Timestamp(S.index[-1]).date()
    doc = _tg.build_universe(c, px=px, sectors={t: v["sector"] for t, v in cls.items()})
    doc["active_from"] = ACTIVE_FROM
    if save:
        _tg.save(doc, overwrite=True)
    return doc


def _load_px(cls):
    px = pd.concat([pd.read_parquet(cfg.CORE), pd.read_parquet(cfg.EXTEND)], ignore_index=True)
    px["Ticker"] = px.Ticker.astype(str).str.zfill(6)
    keep = {t for t, v in cls.items() if v["sector"] not in cfg.DROP_SUBSECTORS}
    return px[px.Ticker.isin(keep)]


if __name__ == "__main__":
    from classification import load_classification
    ap = argparse.ArgumentParser(); ap.add_argument("--report", action="store_true"); ap.add_argument("--asof", default=None)
    a = ap.parse_args()
    cls = load_classification(); px = _load_px(cls)
    r = report(px, cls, asof=a.asof, save=a.report)
    print(f"[day_universe] 점검 {r['check_date']} → 적용 {r['effective_date']} · {r['count']}종목 (범위 {RANGE_MIN}~{RANGE_MAX} {'OK' if r['range_ok'] else '⚠'})")
    print(f"  편출 {len(r['removed'])}: " + ", ".join(f"{x['name']}({x['a20_eok']}억)" for x in r["removed"]))
    print(f"  재편입 {len(r['reentered'])}: " + ", ".join(x["name"] for x in r["reentered"]))
    print(f"  주의(1회 미달) {len(r['watch_below_once'])}: " + ", ".join(f"{x['name']}({x['a20_eok']}억)" for x in r["watch_below_once"]))
    if r["sector_under_min"]:
        print(f"  ⚠ 섹터 {cfg.MIN_SECTOR_N}종목 미만: {r['sector_under_min']}")
