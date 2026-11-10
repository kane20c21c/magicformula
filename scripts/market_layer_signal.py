#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/market_layer_signal.py — 통일장 시장 층 판정 (⑦ 운영 후보 · ⑤ 그림자, 2026-11-10 케인).

    python3 scripts/market_layer_signal.py              # LLV 코스피 200 마지막 날 기준 (휴장 가드)
    python3 scripts/market_layer_signal.py 20261109     # 특정 날짜
    python3 scripts/market_layer_signal.py --dry-run    # 저장 없이 출력만

단계는 재료가 정한다 — 판정일의 TMI·브레드스가 모두 KRX 확정이면 '확정', 하나라도 잠정이면 '잠정'.
  19:10 저녁 판정  : 브레드스 = 한국투자 잠정, TMI = 전날 200일선 상태 유지 → 잠정
  다음 날 아침 재판정: 08:01 LLV 가 KRX 확정치를 채운 뒤 → 확정 (같은 파일에 덧붙이고 잠정과 같은지 기록)
출력 = output/market_layer/YYYYMMDD.json  — 실행 시각 연결·알림은 ③ 그림자 추적과 함께 따로 붙인다.
재료 결측(판정일 아닌 날) = InputGapError → exit 1 (배치 실패로 드러나게).
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from market_layer import DEFAULT, judge                                   # noqa: E402
from market_layer.inputs import InputGapError, load_inputs               # noqa: E402

OUT_DIR = PROJECT_ROOT / "output" / "market_layer"
NAMES = {"p7": "⑦ C2+갭쇼크 (운영 후보)", "p5": "⑤ 갭쇼크·브레드스·평소보유 (그림자)"}


def _lab(v: bool) -> str:
    return "보유" if bool(v) else "현금"


def build_block(inp, d: pd.DataFrame) -> dict:
    day = inp.as_of
    row, prev = d.loc[day], d.loc[:day].iloc[-2]
    k, ko = float(inp.k_close.loc[day]), float(inp.k_open.loc[day])
    kp = float(inp.k_close.loc[:day].iloc[-2])
    tmi = inp.tmi.loc[day] if day in inp.tmi.index else None
    ratio = inp.ratio.loc[day] if day in inp.ratio.index else None
    tail = d.loc[:day].tail(10)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "states": {k_: _lab(row[k_]) for k_ in ("p7", "p5")},
        "prev_states": {k_: _lab(prev[k_]) for k_ in ("p7", "p5")},
        "changed": {k_: bool(row[k_] != prev[k_]) for k_ in ("p7", "p5")},
        "components": {"breadth": "상승" if row.breadth else "하락",
                       "breadth_value": None if pd.isna(row.breadth_value) else round(float(row.breadth_value), 4),
                       "breadth_band": DEFAULT.breadth_band, "c2": "보유" if row.c2 else "퇴출",
                       "gate_window": bool(row.gate_window), "shock_today": bool(row.shock), "gate_mode": bool(row["mode"])},
        "inputs": {"k200_open": round(ko, 2), "k200_close": round(k, 2), "k200_ret_pct": round((k / kp - 1) * 100, 2),
                   "k200_gap_pct": round((ko / kp - 1) * 100, 2),
                   "tmi_close": None if tmi is None or pd.isna(tmi) else round(float(tmi), 2),
                   "breadth_ratio": None if ratio is None else round(float(ratio), 4)},
        "source": inp.source,
        "recent": [{"date": t.strftime("%Y-%m-%d"), "p7": _lab(r.p7), "p5": _lab(r.p5), "breadth_value":
                    None if pd.isna(r.breadth_value) else round(float(r.breadth_value), 4)} for t, r in tail.iterrows()],
    }


def merge_file(path: Path, stage: str, block: dict, as_of: str) -> dict:
    """같은 날 파일에 단계별 블록을 넣는다 — 잠정은 잠정 칸, 확정은 확정 칸. 둘 다 있으면 같은지 기록."""
    out = {"version": DEFAULT.version, "as_of": as_of, "provisional": None, "confirmed": None, "provisional_matches_confirmed": None}
    if path.exists():
        try:
            out.update({k: v for k, v in json.loads(path.read_text(encoding="utf-8")).items() if k in out})
        except Exception:                                          # noqa: BLE001 — 깨진 파일은 새로 쓴다
            pass
    out["version"] = DEFAULT.version
    out[stage] = block
    if out["provisional"] and out["confirmed"]:
        out["provisional_matches_confirmed"] = out["provisional"]["states"] == out["confirmed"]["states"]
    return out


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]; flags = {a for a in argv if a.startswith("--")}
    target = args[0] if args else None
    if target is None:
        try:
            sys.path.insert(0, str(PROJECT_ROOT.parent / "LongLiveVault"))
            from stolab_data.trading_calendar import is_trading_day
            if not is_trading_day(date.today()):
                print(f"⏸ {date.today()} 휴장일 — 시장 층 판정 skip"); return 0
        except Exception as e:                                     # noqa: BLE001
            print(f"⚠ 거래일 캘린더 확인 실패({e}) — 가드 없이 진행")
    try:
        inp = load_inputs(target)
    except InputGapError as e:
        print(f"✗ 판정 보류 — {e}"); return 1
    d = judge(inp.k_open, inp.k_close, inp.tmi, inp.ratio)
    stage = "provisional" if inp.source["provisional"] else "confirmed"
    block = build_block(inp, d)
    as_of = inp.as_of.strftime("%Y-%m-%d")
    print(f"통일장 시장 층 {DEFAULT.version} · {as_of} · {'잠정' if stage == 'provisional' else '확정'}")
    for k_, nm in NAMES.items():
        ch = " ← 바뀜" if block["changed"][k_] else ""
        print(f"  {nm}: {block['states'][k_]} (전날 {block['prev_states'][k_]}){ch}")
    c = block["components"]
    print(f"  브레드스 {c['breadth']} ({c['breadth_value']} / ±{c['breadth_band']}) · C2 {c['c2']} · 관문 창 {c['gate_window']}"
          f" · 오늘 쇼크 {c['shock_today']} · 관문 모드 {c['gate_mode']}")
    print(f"  재료: {inp.source['breadth']} · TMI {inp.source['tmi']}")
    if "--dry-run" in flags:
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{inp.as_of:%Y%m%d}.json"
    out = merge_file(path, stage, block, as_of)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"); tmp.replace(path)
    print(f"저장: {path}" + ("" if out["provisional_matches_confirmed"] is None
                              else f" · 잠정=확정 {out['provisional_matches_confirmed']}"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
