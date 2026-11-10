# -*- coding: utf-8 -*-
"""반등 룰 신호일의 시장 국면 표기 (2026-11-10 케인 (가) — 표기만, 신호는 그대로).

근거 korean_mkt_study/DRIVE_STUDY.md §9: 신호의 92%(학습)·99%(보류)가 브레드스 하락 국면에서 나오고,
브레드스 상승 국면 신호(학습 110건)는 5일 뒤 플러스 40.0% vs 하락 국면 67.7%. 보류 구간은 상승 국면 신호 4건이라 판정 불가 →
규칙은 바꾸지 않고 메일·JSON 에 국면만 적는다.

읽는 곳 (MagicFormula 통일장 시장 층 판정 — market_layer)
  1) output/market_layer/YYYYMMDD.json — 확정(confirmed) 블록이 있으면 확정, 없으면 19:10 잠정(provisional)
  2) 없으면 output/market_layer/regime_history.parquet 의 그날 행 (확정 이력)
  3) 둘 다 없으면 None — 메일에 '판정 없음' 으로 적는다 (신호 발송은 막지 않는다)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1] / "output" / "market_layer"
# 학습 구간(2016-01~2026-05) 신호일 국면별 성적 — DRIVE_STUDY §9.1. 메일 주석용 (값을 바꾸면 그 절도 같이)
STUDY_NOTE = {"상승": "연구(2016~2026.05) 브레드스 상승 국면 신호 110건: 5일 뒤 플러스 40.0% · 하락 국면 1,325건 67.7%",
              "하락": "연구(2016~2026.05) 브레드스 하락 국면 신호 1,325건: 5일 뒤 플러스 67.7%"}


def market_regime(asof, ml_dir: Optional[Path] = None) -> Optional[dict]:
    """{'breadth': '상승'|'하락', 'c2': '보유'|'퇴출', 'stage': '확정'|'잠정'|'이력', 'breadth_value': float|None} 또는 None."""
    d = Path(ml_dir) if ml_dir else ML_DIR
    day = pd.Timestamp(asof)
    p = d / f"{day:%Y%m%d}.json"
    if p.exists():
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            for key, stage in (("confirmed", "확정"), ("provisional", "잠정")):
                b = j.get(key)
                if b and b.get("components"):
                    c = b["components"]
                    return {"breadth": c.get("breadth"), "c2": c.get("c2"), "stage": stage, "breadth_value": c.get("breadth_value")}
        except Exception:                                          # noqa: BLE001 — 깨진 파일이면 이력으로
            pass
    h = d / "regime_history.parquet"
    if h.exists():
        try:
            t = pd.read_parquet(h, columns=["breadth", "c2", "breadth_value"])
            t.index = pd.to_datetime(t.index)
            if day in t.index and pd.notna(t.loc[day, "breadth"]) and pd.notna(t.loc[day, "c2"]):
                r = t.loc[day]
                return {"breadth": "상승" if bool(r.breadth) else "하락", "c2": "보유" if bool(r.c2) else "퇴출", "stage": "이력",
                        "breadth_value": None if pd.isna(r.breadth_value) else round(float(r.breadth_value), 4)}
        except Exception:                                          # noqa: BLE001
            pass
    return None


def regime_html(reg: Optional[dict]) -> str:
    """메일 한 단락. 브레드스 상승이면 연구 성적을 경고색으로 붙인다."""
    if not reg or not reg.get("breadth"):
        return "<p style='color:#666'>신호일 시장 국면: 판정 없음 (통일장 시장 층 파일을 찾지 못함)</p>"
    head = f"신호일 시장 국면: <b>브레드스 {reg['breadth']}</b> · C2 {reg.get('c2') or '-'} ({reg.get('stage')})"
    note = STUDY_NOTE.get(reg["breadth"], "")
    if reg["breadth"] == "상승":
        return (f"<p>{head}<br><span style='color:#1976D2'>⚠ {note} — 표기만, 신호는 그대로 (케인 2026-11-10)</span></p>")
    return f"<p>{head}<br><span style='color:#666'>{note}</span></p>"
