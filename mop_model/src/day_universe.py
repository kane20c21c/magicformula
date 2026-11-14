# -*- coding: utf-8 -*-
"""day_universe.py — 데이포트 유니버스 규칙 (2026-11-03 케인 U7 확정 → v1.2.2.5, 11-16 적용).

규칙 (근거: 볼트 '2026-11-01 U7 — 데이포트 유니버스 규칙'):
  · 대상 = LLV 멤버(core+extend) 중 분류가 있고 ETF·우선주가 아닌 종목 — **편입은 케인 판단(ScoutingTheNext)**, 시총 상하한 없음
  · 유지 조건 = 20일 평균 거래대금 ≥ **50억** — 점검일에 **2회 연속 미달이면 편출**, 다시 50억 이상이면 재편입
  · 점검일 = 매월 옵션 만기(둘째 목요일)가 든 주의 금요일(휴장이면 그 주 마지막 거래일), **다음 거래일부터** 적용
  · 관리 범위 260~310종목 (2026-11-14 유니버스 조정으로 약 285종목 — 종전 190~240, 케인 2026-11-11) · 대분류 섹터당 5종목 이상
    (미달은 K7 회차 보고 경고 — 자동 편출 없음)
학습·신호 모두 같은 멤버십으로 거른다(횡단면 피처·y_rel 이 그 유니버스 안에서 계산되게). 과거 날짜도 같은 규칙.
ACTIVE_FROM 이전 날짜에 신호를 만들면 필터를 걸지 않는다(배포 시점과 무관하게 11-16 부터 적용).

CLI (K7 회차 기록): python3 day_universe.py --report [--asof YYYY-MM-DD]
  → output/day_universe/day_universe_{적용일}.json (멤버·편출·재편입·주의·섹터 수)
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

import config as cfg

AMOUNT_MIN = 50e8          # 20일 평균 거래대금 하한 (원)
AMOUNT_WIN, AMOUNT_MINP = 20, 15
OUT_AFTER = 2              # 연속 미달 횟수 → 편출
RANGE_MIN, RANGE_MAX = 260, 310   # 관리 범위 (경고용 — 자동 편출 없음). 2026-11-14 유니버스 조정, 종전 190~240
ACTIVE_FROM = "2026-11-16"  # 11-13(11월 만기 주 금요일) 점검분부터 운영 적용
RULE_ID = "day_universe_v1 (U7 2026-11-03: LLV멤버 ∩ 20일평균거래대금≥50억, 2회 연속 미달 편출, 만기 주 금요일 점검)"


def check_dates(trading_days) -> list[pd.Timestamp]:
    """거래일 목록 안에서 매월 '옵션 만기가 든 주의 금요일(휴장이면 그 주 마지막 거래일)'."""
    td = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(trading_days)).unique()))
    out = []
    for per in pd.period_range(td.min(), td.max(), freq="M"):
        first = date(per.year, per.month, 1)
        thu = first + timedelta(days=(3 - first.weekday()) % 7 + 7)
        mon, fri = pd.Timestamp(thu - timedelta(days=3)), pd.Timestamp(thu + timedelta(days=1))
        wk = td[(td >= mon) & (td <= fri)]
        if len(wk):
            out.append(wk[-1])
    return out


def status_table(px: pd.DataFrame) -> pd.DataFrame:
    """점검일 × 종목 → 멤버 여부(bool) 와 20일 평균 거래대금. px: Date, Ticker, Amount (Close·Volume 있으면 결측 보충)."""
    p = px[["Date", "Ticker"] + [c for c in ("Amount", "Close", "Volume") if c in px.columns]].copy()
    p["Date"] = pd.to_datetime(p.Date)
    amt = p.Amount.where(p.Amount > 0) if "Amount" in p else pd.Series(np.nan, index=p.index)
    if {"Close", "Volume"} <= set(p.columns):
        amt = amt.fillna(p.Close * p.Volume)
    p["amt"] = amt
    p = p.sort_values(["Ticker", "Date"])
    p["a20"] = p.groupby("Ticker").amt.transform(lambda s: s.rolling(AMOUNT_WIN, min_periods=AMOUNT_MINP).mean())
    A = p.pivot_table(index="Date", columns="Ticker", values="a20")
    checks = [c for c in check_dates(A.index) if c in A.index]
    st, below = {}, {t: 0 for t in A.columns}
    rows = []
    for c in checks:
        v = A.loc[c]
        for t in A.columns:
            x = v[t]
            if pd.isna(x):                       # 상장 직후 등 값 없음 → 상태 유지
                pass
            elif x >= AMOUNT_MIN:
                below[t] = 0
            else:
                below[t] += 1
            st[t] = below[t] < OUT_AFTER
        rows.append(pd.Series(st, name=c))
    S = pd.DataFrame(rows) if rows else pd.DataFrame(columns=A.columns)
    S.index.name = "check"
    return S, A.reindex(checks)


def membership_mask(px: pd.DataFrame, today=None) -> pd.Series:
    """px 행별 멤버 여부. 날짜 D 의 상태 = D 보다 앞선 마지막 점검일의 상태(점검 다음 거래일부터 적용). 첫 점검 전은 True.
    today(신호 기준일)가 ACTIVE_FROM 이전이면 전부 True (운영 적용 전)."""
    if today is not None and pd.Timestamp(today) < pd.Timestamp(ACTIVE_FROM):
        return pd.Series(True, index=px.index)
    S, _ = status_table(px)
    if S.empty:
        return pd.Series(True, index=px.index)
    d = pd.to_datetime(px.Date).to_numpy()
    k = np.searchsorted(S.index.to_numpy(), d, side="left") - 1      # 점검일 '당일'은 직전 상태
    tk = px.Ticker.astype(str).to_numpy()
    col = {t: i for i, t in enumerate(S.columns)}
    M = S.to_numpy(dtype=bool)
    out = np.ones(len(px), dtype=bool)
    for i in range(len(px)):
        if k[i] >= 0 and tk[i] in col:
            out[i] = M[k[i], col[tk[i]]]
    return pd.Series(out, index=px.index)


def report(px: pd.DataFrame, cls: dict, asof=None, out_dir=None) -> dict:
    """마지막 점검(asof 이전) 결과 기록 — K7 회차용."""
    S, A = status_table(px)
    if asof is not None:
        S, A = S[S.index <= pd.Timestamp(asof)], A[A.index <= pd.Timestamp(asof)]
    c = S.index[-1]; prev = S.iloc[-2] if len(S) > 1 else pd.Series(True, index=S.columns)
    cur = S.iloc[-1]
    td = sorted(pd.to_datetime(px.Date).unique()); nxt = [x for x in td if x > c]
    eff = (pd.Timestamp(nxt[0]) if nxt else pd.Timestamp(c) + pd.offsets.BDay(1)).date()
    members = sorted(t for t, v in cur.items() if v)
    removed = sorted(t for t in S.columns if prev[t] and not cur[t])
    reentered = sorted(t for t in S.columns if (not prev[t]) and cur[t])
    watch = sorted(t for t in members if pd.notna(A.loc[c, t]) and A.loc[c, t] < AMOUNT_MIN)
    sec = pd.Series([cls.get(t, {}).get("sector", "?").split(".")[0] for t in members]).value_counts()
    doc = {"rule": RULE_ID, "check_date": str(pd.Timestamp(c).date()), "effective_date": str(eff), "active_from": ACTIVE_FROM,
           "count": len(members), "range_ok": RANGE_MIN <= len(members) <= RANGE_MAX,
           "removed": [{"ticker": t, "name": cls.get(t, {}).get("name", ""), "a20_eok": round(float(A.loc[c, t]) / 1e8, 1)} for t in removed],
           "reentered": [{"ticker": t, "name": cls.get(t, {}).get("name", "")} for t in reentered],
           "watch_below_once": [{"ticker": t, "name": cls.get(t, {}).get("name", ""), "a20_eok": round(float(A.loc[c, t]) / 1e8, 1)} for t in watch],
           "sector_counts": {k: int(v) for k, v in sec.items()},
           "sector_under_min": {k: int(v) for k, v in sec.items() if v < cfg.MIN_SECTOR_N},
           "members": members}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"day_universe_{eff}.json"), "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
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
    r = report(px, cls, asof=a.asof, out_dir=os.path.join(cfg.BASE_DIR, "output", "day_universe") if a.report else None)
    print(f"[day_universe] 점검 {r['check_date']} → 적용 {r['effective_date']} · {r['count']}종목 (범위 {RANGE_MIN}~{RANGE_MAX} {'OK' if r['range_ok'] else '⚠'})")
    print(f"  편출 {len(r['removed'])}: " + ", ".join(f"{x['name']}({x['a20_eok']}억)" for x in r["removed"]))
    print(f"  재편입 {len(r['reentered'])}: " + ", ".join(x["name"] for x in r["reentered"]))
    print(f"  주의(1회 미달) {len(r['watch_below_once'])}: " + ", ".join(f"{x['name']}({x['a20_eok']}억)" for x in r["watch_below_once"]))
    if r["sector_under_min"]:
        print(f"  ⚠ 섹터 {cfg.MIN_SECTOR_N}종목 미만: {r['sector_under_min']}")
