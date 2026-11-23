"""jdh_rule/signal.py — 15:10 잠정 판정 (케인 2026-11-23).

  5일 수익률 = 15:10 현재가(잠정) ÷ 5거래일 전 종가 − 1  → 완전 4일 + 잠정 1일. 선행 3종목·KOSPI200 모두 같은 방식.
  ⑦ = regime_history 의 최신 확정 판정(전날, 08:40 확정분). 판정일 ≠ 직전 거래일이면 stale.
  틀: ⑦ 보유 → 'cash'(롱/현금) / ⑦ 현금 → 'short'(롱/숏) / ⑦ 낡음·없음 → None (SP 가 지금 틀 유지)
  이동: 틀 cash → KOSPI200 부호 × k200_move / 틀 short → 선행 합의(3개 ±2, 정확히 2개 ±1)
가격 조회·저장은 scripts/jdh_signal.py — 여기는 순수 계산(테스트 가능).
"""
from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from jdh_rule.config import DEFAULT, JdhConfig, REGIME_HISTORY, RULE_VERSION


def ret(now: Optional[float], base: Optional[float]) -> Optional[float]:
    if now is None or base is None or not (now > 0) or not (base > 0):
        return None
    return float(now) / float(base) - 1.0


def lead_move(rets: list[Optional[float]], cfg: JdhConfig = DEFAULT) -> tuple[int, int, int]:
    """(이동 슬롯, 상승 수, 하락 수). 수익률이 하나라도 없으면 이동 0 (불완전 판정은 매매하지 않는다)."""
    if len(rets) < cfg.n_leads or any(r is None for r in rets):
        up = sum(1 for r in rets if r is not None and r > 0); dn = sum(1 for r in rets if r is not None and r < 0)
        return 0, up, dn
    up = sum(1 for r in rets if r > 0); dn = sum(1 for r in rets if r < 0)
    if up == 3: return cfg.lead_move_3, up, dn
    if dn == 3: return -cfg.lead_move_3, up, dn
    if up == 2: return cfg.lead_move_2, up, dn
    if dn == 2: return -cfg.lead_move_2, up, dn
    return 0, up, dn


def k200_move(r: Optional[float], cfg: JdhConfig = DEFAULT) -> int:
    if r is None or r == 0:
        return 0
    return cfg.k200_move if r > 0 else -cfg.k200_move


def latest_p7(today: date, prev_tds: list[date], path: Path = REGIME_HISTORY, cfg: JdhConfig = DEFAULT) -> dict:
    """today 이전 가장 최근 ⑦. prev_tds = today 직전 거래일들(최근 먼저)."""
    try:
        h = pd.read_parquet(path, columns=["p7"])
    except Exception as e:  # noqa: BLE001
        return {"date": None, "p7": None, "stale_td": None, "usable": False, "why": f"regime_history 읽기 실패: {e}"}
    h.index = pd.to_datetime(h.index)
    h = h[(h.index < pd.Timestamp(today)) & h.p7.notna()]
    if h.empty:
        return {"date": None, "p7": None, "stale_td": None, "usable": False, "why": "⑦ 이력 없음"}
    d = h.index[-1].date(); v = bool(h.p7.iloc[-1])
    stale = next((i for i, x in enumerate(prev_tds) if x <= d), len(prev_tds))   # 0 = 직전 거래일
    usable = stale < cfg.p7_max_stale_td
    return {"date": str(d), "p7": v, "label": "보유" if v else "현금", "stale_td": int(stale), "usable": bool(usable),
            "why": "" if usable else f"⑦ 판정일 {d} 이 직전 거래일보다 {stale}거래일 낡음 — 틀 유지"}


def judge(today: date, leads: list[dict], k200: dict, p7: dict, cfg: JdhConfig = DEFAULT) -> dict:
    """leads = [{ticker,name,base_close,now}] · k200 = {base_close, now} · p7 = latest_p7 결과."""
    rows = []
    for x in leads:
        r = ret(x.get("now"), x.get("base_close"))
        rows.append({**x, "ret5": r, "dir": None if r is None else (1 if r > 0 else -1 if r < 0 else 0)})
    lm, up, dn = lead_move([x["ret5"] for x in rows], cfg)
    kr = ret(k200.get("now"), k200.get("base_close")); km = k200_move(kr, cfg)
    frame = ("cash" if p7.get("p7") else "short") if p7.get("usable") else None
    if frame == "cash":
        move, source = km, "k200"
    elif frame == "short":
        move, source = lm, "lead"
    else:
        move, source = None, None          # SP 가 지금 틀에 맞는 신호를 고른다 (둘 다 실어 보냄)
    incomplete = [x["ticker"] for x in rows if x["ret5"] is None]
    return {
        "rule": "jdh", "rule_version": RULE_VERSION, "as_of": str(today),
        "leads": rows, "lead": {"move": lm, "up": up, "dn": dn, "incomplete": incomplete},
        "k200": {**k200, "ret5": kr, "move": km},
        "p7": p7, "frame": frame, "move": move, "move_source": source,
        "frame_label": {"cash": "롱/현금 (⑦ 보유)", "short": "롱/숏 (⑦ 현금)", None: "⑦ 낡음 — 틀 유지"}[frame],
    }


__all__ = ["ret", "lead_move", "k200_move", "latest_p7", "judge"]
