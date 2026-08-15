# -*- coding: utf-8 -*-
"""
gauge_core.py — 섹터 게이지 집계 (순수 로직, LLV/네트워크 무관)
================================================================
run_gauge.py 의 세트별 시총 가중평균 집계를 분리한 모듈.
tests/test_mop_gauge.py 가 이 모듈만 import 해 검증한다 (모델·KIS 불필요).

가중치 규칙 (Kane 전환 2026-08-15 — 종전 시총 자동가중 폐지):
  - 세트 정의가 dict({티커: 가중치}) 면 **Kane 지정 고정가중**을 그대로 쓴다.
    세트 합은 정규화하지 않는다 (엑셀 반올림 탓에 0.99~1.01 인 세트가 있고,
    그 배율이 weighted_p 에 그대로 실린다 — Kane 결정).
  - p 없는 종목(스냅샷 누락·유니버스 밖)은 제외하고, 남은 종목의 가중치를
    **세트 정의 합에 맞춰 비례 재분배**한다 (결손분이 p 를 끌어내리지 않도록).
    → 전원 스코어되면 재분배 배율 1.0 이라 지정값이 그대로 보존된다.
  - 세트 정의가 list 면 종전 시총 가중 경로 (호환 유지):
    시총 비중으로 정규화, 시총 전부 결측이면 동일가중 폴백.

출력에 진단 필드 2개를 함께 싣는다:
  weight_mode  "fixed" | "mcap"
  weight_sum   실제 적용된 가중치 합 (= 세트 정의 합, 전원 미스코어면 0.0)
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
        sector_sets: {세트명: {티커: 가중치}}  — 고정가중 (정본)
                     {세트명: [티커, ...]}     — 시총가중 (구 경로, 호환)
        names:       {티커: 종목명} (미스코어 종목 표기용)

    Returns:
        [{name, weighted_p, mean_p, n_members, n_scored, weight_mode,
          weight_sum, top_member, members}, ...]
    """
    import pandas as pd

    out = []
    for set_name, spec in sector_sets.items():
        if isinstance(spec, dict):
            wmap = {str(t).zfill(6): float(w) for t, w in spec.items()}
            tks = list(wmap)
        else:
            wmap = None
            tks = [str(t).zfill(6) for t in spec]

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
            if wmap is not None:
                m["weight_def"] = wmap[t]
            members.append(m)

        if wmap is not None:
            # 고정가중 — 미스코어분을 남은 종목에 비례 재분배 (세트 합 보존).
            w_def = sum(wmap.values())
            w_hit = sum(wmap[m["ticker"]] for m in scored)
            scale = (w_def / w_hit) if w_hit > 0 else 0.0
            for m in scored:
                m["weight"] = wmap[m["ticker"]] * scale
            weight_mode = "fixed"
        else:
            # 시총가중 (구 경로) — 잔여 시총으로 재정규화, 전부 결측이면 동일가중.
            w_sum = sum(m["mcap"] or 0.0 for m in scored)
            for m in scored:
                m["weight"] = (m["mcap"] / w_sum) if (w_sum > 0 and m["mcap"]) else (
                    1.0 / len(scored) if scored else 0.0)
            weight_mode = "mcap"

        weighted_p = (sum(m["weight"] * m["p"] for m in scored)
                      if scored else None)
        mean_p = (float(np.mean([m["p"] for m in scored])) if scored else None)
        top = max(scored, key=lambda m: m["p"]) if scored else {}
        out.append({
            "name": set_name,
            "weighted_p": round(weighted_p, 6) if weighted_p is not None else None,
            "mean_p": round(mean_p, 6) if mean_p is not None else None,
            "n_members": len(tks), "n_scored": len(scored),
            "weight_mode": weight_mode,
            "weight_sum": round(sum(m["weight"] for m in scored), 6),
            "top_member": {"ticker": top.get("ticker"), "name": top.get("name"),
                           "p": top.get("p")},
            "members": members,
        })
    return out
