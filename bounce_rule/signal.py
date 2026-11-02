# -*- coding: utf-8 -*-
"""반등 룰 스캔 — 종목별 일별 종가 패널에서 '오늘 끝난 드라이브' 를 찾고 강도를 매긴다.

입력: DataFrame[Date, Ticker, Name?, Close] (여러 종목, 정렬 불문). 출력: 오늘(asof) 드라이브가 끝난 종목 표 —
  ticker name date drive_days drive_drop d1_ret sigma60 strength low_close signal deadcat
연구 코드(korean_mkt_study/crash_study.py, drive_hold*) 와 같은 정의다 — 회귀 테스트가 같은 날짜 재현을 검사한다.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import BounceConfig, DEFAULT


def scan_drives(panel: pd.DataFrame, asof=None, cfg: BounceConfig = DEFAULT, all_ended: bool = True) -> pd.DataFrame:
    """asof(기본 = 패널 마지막 날짜) 에 끝난 드라이브를 전부 돌려준다. all_ended=False 면 신호만."""
    p = panel.copy()
    p["Date"] = pd.to_datetime(p["Date"]); p["Ticker"] = p["Ticker"].astype(str)
    if asof is not None:
        p = p[p["Date"] <= pd.Timestamp(asof)]
    p = p.sort_values(["Ticker", "Date"])
    p = p[p["Close"] > 0]
    rows = []
    for t, d in p.groupby("Ticker", sort=False):
        c = d["Close"].to_numpy(dtype=float)
        if len(c) < cfg.min_history + 1:
            continue
        if asof is not None and d["Date"].iloc[-1] != pd.Timestamp(asof):
            continue                                   # 그날 행이 없는 종목(거래정지 등)
        r = c[1:] / c[:-1] - 1                         # r[i] = 날 i+1 의 수익률
        if r[-1] < 0:
            continue                                   # 오늘도 하락 — 드라이브 진행 중
        k = 0
        while k < len(r) - 1 and r[-2 - k] < 0:
            k += 1                                     # 어제부터 거꾸로 센 연속 하락일
        if k < cfg.min_drive_days:
            continue
        win = r[-1 - cfg.sigma_window:-1]              # 드라이브 종료일까지 60일 (D+1 제외)
        if len(win) < cfg.sigma_window:
            continue
        sig = float(np.std(win, ddof=1))
        if not np.isfinite(sig) or sig <= 0:
            continue
        drop = float(np.prod(1 + r[-1 - k:-1]) - 1)
        d1 = float(r[-1]); strength = d1 / sig
        signal = (drop <= cfg.min_drop) and (strength >= cfg.strong_sigma)
        if not all_ended and not signal:
            continue
        rows.append({
            "ticker": t, "name": str(d["Name"].iloc[-1]) if "Name" in d else "",
            "date": d["Date"].iloc[-1].strftime("%Y-%m-%d"),
            "drive_days": int(k), "drive_drop": round(drop, 4), "d1_ret": round(d1, 4),
            "sigma60": round(sig, 5), "strength": round(strength, 2),
            "low_close": float(c[-2]),                 # 드라이브 저점 = 마지막 하락일 종가 (10일 재이탈 기준선)
            "signal": bool(signal), "deadcat": bool(strength < cfg.deadcat_sigma),
        })
    cols = ["ticker", "name", "date", "drive_days", "drive_drop", "d1_ret", "sigma60", "strength", "low_close", "signal", "deadcat"]
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["signal", "strength"], ascending=[False, False]).reset_index(drop=True)
