"""
magic_formula.daily.report
==========================
데일리 시그널 결과 dict → Markdown 리포트 렌더링.

runner.py 에서 분리 (2026-06-10 v2 단일화) — 파이프라인(runner)과
표현(report)의 책임 분리. JSON 직렬화 헬퍼도 여기에 둔다.
"""

from __future__ import annotations

from datetime import datetime
from itertools import groupby
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# JSON 직렬화 헬퍼
# ---------------------------------------------------------------------------

def json_default(obj):
    """numpy bool_ / int_ / float_ → Python 기본 타입 변환."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ---------------------------------------------------------------------------
# 셀 포맷 헬퍼
# ---------------------------------------------------------------------------

def _phase_short(label: str) -> str:
    """국면 라벨 → 약어 (md 표 가독성)."""
    mapping = {
        "Accumulation": "Accum",
        "Markup": "Markup",
        "Distribution": "Dist",
        "Markdown": "Mark↓",
    }
    return mapping.get(label, label[:6] if label else "")


def _signal_short(sig: str, strength) -> str:
    """전환 신호 + 강도 → 표 셀 텍스트. 예: ACC_COMPLETE strength=2 → 'ACC↑(2)'."""
    if not sig:
        return ""
    label_map = {
        "ACC_COMPLETE":    "ACC↑",
        "DIST_CONFIRM":    "DIST↓",
        "MARKUP_WATCH":    "MU↘",
        "MARKDOWN_WATCH":  "MD↗",
        "PANIC_ZONE":      "PANIC",
    }
    short = label_map.get(sig, sig[:6])
    if strength is not None:
        return f"{short}({strength})"
    return short


# ---------------------------------------------------------------------------
# Markdown 리포트
# ---------------------------------------------------------------------------

def write_md(result: dict, path: Path) -> None:
    """v2 결과 dict → Markdown 리포트 저장.

    표 컬럼 순서 (Kane 지시 2026-05-31):
      종목 | 국면 | 전환신호 | 종합점수 | 추세 | 모멘텀 | 거래량 | 변동성 | 종가 | 등락 | 거래량변동
    국면은 게이트 역할이므로 점수 앞에 배치. 전환신호는 5종(ACC/DIST/MU/MD/PANIC) 모두 기록.
    """
    d = datetime.strptime(result["date"], "%Y%m%d")
    date_ko = f"{d.year}.{d.month:02d}.{d.day:02d}"
    w = result["weights"]
    lines = [
        f"# 황금률 진입신호 리포트 (v2 결합) — {date_ko}", "",
        f"- **진입 신호**: {result['signal_count']}종목",
        f"- **전략**: {result['strategy_id']} (threshold={result['threshold']:.1f}, "
        f"게이트={'ON' if result['gate'] else 'OFF'})",
        f"- **가중치**: T={w.get('trend',0)*100:.0f}% M={w.get('momentum',0)*100:.0f}% "
        f"Vu={w.get('volume',0)*100:.0f}% Va={w.get('volatility',0)*100:.0f}%",
        f"- **대상 종목**: {result['total_tickers']}개", "",
    ]

    # 진입 신호 표 (섹터별 그룹)
    if result["signals"]:
        lines += ["## 📈 진입 신호 종목", ""]
        for sector, rows in groupby(result["signals"], key=lambda r: r["sector"]):
            rows = list(rows)
            lines += [
                f"### [{sector}]", "",
                "| 종목 | 국면 | 전환신호 | 종합점수 | 추세 | 모멘텀 | 거래량 | 변동성 | 종가 | 등락 | 거래량변동 |",
                "|------|------|---------|---------|------|--------|--------|--------|------|------|-----------|",
            ]
            for s in rows:
                chg = s.get("change_pct", 0)
                vol_chg = s.get("vol_chg_pct", 0)
                lines.append(
                    f"| {s['name']}({s['ticker']}) "
                    f"| {_phase_short(s.get('wyckoff_phase',''))} "
                    f"| {_signal_short(s.get('wyckoff_signal',''), s.get('wyckoff_signal_strength'))} "
                    f"| **{s['composite_score']:.2f}** "
                    f"| {s['area1_trend']:.1f} | {s['area2_momentum']:.1f} "
                    f"| {s['area3_volume']:.1f} | {s['area4_volatility']:.1f} "
                    f"| {s.get('close','-'):,.0f} "
                    f"| {'+' if chg>=0 else ''}{chg:.1f}% "
                    f"| {'+' if vol_chg>=0 else ''}{vol_chg:.0f}% |"
                )
            lines += [""]
    else:
        lines += ["## 📭 오늘 진입 신호 없음", ""]

    # 전체 종목 현황
    lines += ["## 전체 종목 현황", "",
              "| 섹터 | 종목 | 국면 | 전환신호 | 종합점수 | 추세 | 모멘텀 | 거래량 | 변동성 |",
              "|------|------|------|---------|---------|------|--------|--------|--------|"]
    for r in result["all_scores"]:
        mark = " 🔔" if r["is_signal"] else (" ⛔" if r.get("gated_out") else "")
        cs = f"{r['composite_score']:.2f}" if r["composite_score"] is not None else "—(게이트)"
        lines.append(
            f"| {r['sector']} | {r['name']}({r['ticker']}){mark} "
            f"| {_phase_short(r.get('wyckoff_phase',''))} "
            f"| {_signal_short(r.get('wyckoff_signal',''), r.get('wyckoff_signal_strength'))} "
            f"| {cs} "
            f"| {r['area1_trend']:.1f} | {r['area2_momentum']:.1f} "
            f"| {r['area3_volume']:.1f} | {r['area4_volatility']:.1f} |"
        )
    lines += [""]
    path.write_text("\n".join(lines), encoding="utf-8")
