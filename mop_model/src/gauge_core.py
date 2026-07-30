# -*- coding: utf-8 -*-
"""
gauge_core.py — 섹터 게이지 집계 (순수 로직, LLV/네트워크 무관)
================================================================
run_gauge.py 의 세트별 시총 가중평균 집계를 분리한 모듈.
tests/test_mop_gauge.py 가 이 모듈만 import 해 검증한다 (모델·KIS 불필요).

가중치 규칙 (Kane 확정 2026-07-30):
  - 가중치 = 세트 내 시가총액 비중 (잠정 종가 × ListShrs, 세트 내 정규화)
  - p 없는 종목(스냅샷 누락·유니버스 밖)은 제외하고 잔여 시총으로 재정규화
  - 시총 전부 결측이면 동일가중 폴백
"""
from __future__ import annotations

import numpy as np


def _num(x):
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def aggregate_sets(by_ticker, mcap, sector_sets, names) -> list:
    """세트별 시총 가중평균 p 집계.

    Args:
        by_ticker:   Ticker 인덱스 DataFrame — 컬럼 p, Name, rank, Close, is_halt
        mcap:        Ticker 인덱스 Series — 당일 시가총액 (원)
        sector_sets: {세트명: [티커, ...]}
        names:       {티커: 종목명} (미스코어 종목 표기용)

    Returns:
        [{name, weighted_p, mean_p, n_members, n_scored, top_member, members}, ...]
    """
    import pandas as pd

    out = []
    for set_name, tks in sector_sets.items():
        tks = [str(t).zfill(6) for t in tks]
        members, scored = [], []
        for t in tks:
            if t in by_ticker.index and pd.notna(by_ticker.at[t, "p"]):
                r = by_ticker.loc[t]
                m = {"ticker": t, "name": r["Name"], "p": round(float(r["p"]), 6),
                     "rank": int(r["rank"]), "close": _num(r["Close"]),
                     "mcap": _num(mcap.get(t)),
                     "is_halt": bool(r.get("is_halt", False))}
                scored.append(m)
            else:
                m = {"ticker": t, "name": names.get(t, t), "p": None,
                     "rank": None, "close": None, "mcap": None,
                     "is_halt": False, "weight": None}
            members.append(m)

        w_sum = sum(m["mcap"] or 0.0 for m in scored)
        for m in scored:
            m["weight"] = (m["mcap"] / w_sum) if (w_sum > 0 and m["mcap"]) else (
                1.0 / len(scored) if scored else 0.0)
        weighted_p = (sum(m["weight"] * m["p"] for m in scored)
                      if scored else None)
        mean_p = (float(np.mean([m["p"] for m in scored])) if scored else None)
        top = max(scored, key=lambda m: m["p"]) if scored else {}
        out.append({
            "name": set_name,
            "weighted_p": round(weighted_p, 6) if weighted_p is not None else None,
            "mean_p": round(mean_p, 6) if mean_p is not None else None,
            "n_members": len(tks), "n_scored": len(scored),
            "top_member": {"ticker": top.get("ticker"), "name": top.get("name"),
                           "p": top.get("p")},
            "members": members,
        })
    return out
