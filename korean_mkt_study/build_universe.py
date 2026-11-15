# -*- coding: utf-8 -*-
"""
build_universe.py — 스윙 유니버스 월 파일 생성 (STRATEGY.md §2 · §8-1, 작업 계획 WP4 — 2026-10-29)

⚠ 2026-11-14 (케인 — 명단 정본 = LLV): 이 결과의 이름은 **메가갤럭시**다. 규칙 코드는 LLV
  `stolab_data/ticker_groups.build_megagalaxy` 가 정본이고 이 파일은 그것을 부르는 껍데기다. 생성 파일도
  LLV groups/megagalaxy_{적용일}.json 으로 간다 (git 추적 — 정기 조정 절차서 §1-2 에어 계산 → §1-3 **LLV 커밋** →
  케인 push · 미니 pull). D 외인을 토요일에 채운 뒤 돌려야 하므로 자동 생성하지 않는다.
  이 폴더의 universe_*.json 은 2026-11 까지의 이력으로만 남는다.
====================================================================================================
지금까지 universe_YYYY-MM_*.json 은 손으로 만들었다(9월분 = 2026-09-12, 클로이). 10월분은 만들어지지 않아
10월 내내 9월 파일(8/31 기준)로 폴백했다 (2026-10-29 확인). 같은 절차를 스크립트로 옮긴다.

규칙 (유니버스 2번째, 2026-08-17 케인 확정): 기준일(월말) 시총 ≥ 4조원 & 외국인지분율 ≥ 30%,
  우선주·스팩·관리종목·ETF 제외. 다음 달 첫 거래일부터 적용, 월중 유지.
원천 (9월분과 같게): LLV panel.parquet 의 MarketCap·Foreign_Ratio — 기준일 이전 마지막 유효값.
  제외 플래그 = LLV issue_status(관리·스팩·우선주·정리매매, 2026-11-15~ — 종전 data_ext/meta.parquet) + 티커 끝자리≠0 우선주 보조.
  패널 밖 누락 점검 = LLV data/raw/krx_{kospi,kosdaq}_*.parquet 중 기준일 이전 최신 파일의 시총 4조↑ 가운데
  패널에 없는 종목을 경고로 보고(편입은 사람이 판단 — 패널에 없으면 가격·지표도 없다).

2026-10-31 케인 확정 (U6 · 월 중간 적용 · 채움 후보는 LLV 안에서만):
  ① **최소 30종목** — 4조 & 30% 가 MIN_COUNT 미만이면 **같은 패널 안의** 외국인 ≥ 30% 종목을 시총 큰 순서로 채운다
     (LLV 패널 밖 종목으로는 채우지 않는다 — 가격·지표가 없다). 채워도 모자라면 floor_shortfall 경고.
     근거: 볼트 '2026-10-30 U6 재테스트' — 30~35 는 현행과 같은 평지, 37 부터 절벽. 바닥 30 은 2016~26 의 24개월(폭락·약세장)에만 걸림.
  ② **조정 = 옵션 만기 주 금요일 종가 → 다음 거래일(월요일)부터 적용** (10-29 결정). JSON 에 effective_date 를 적고,
     소비자(SP paper/universe · MF shadow_track · LLV vol_scale)는 effective_date ≤ 그날 인 파일 중 가장 늦은 것을 쓴다.
     effective_date 없는 옛 파일은 'apply_month 1일' 로 본다. 파일명 = universe_{effective_date}_4jo_30pct_min30.json.

사용:
  python3 build_universe.py --expiry 2026-11                 → 기준 = 11월 만기 주 금요일(11-13) · 적용 11-16
  python3 build_universe.py --base 2026-11-13                → 기준일 직접 지정 (적용 = 다음 거래일)
  python3 build_universe.py --base 2026-08-31 --min-count 0 --dry-run --check data/universe_2026-09_4jo_30pct.json   (9월분 재현 검증)
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
STOLAB = HERE.parents[1]
LLV = STOLAB / "LongLiveVault" / "data"
MKTCAP_MIN = 4_000_000_000_000
FOREIGN_MIN = 30.0
ETF = {"102110", "252650"}
STRATEGY_ID = "kr_pullback_largecap_foreign"
MIN_COUNT = 30          # 2026-10-31 케인 U6 확정 — 최소 30종목 유지


def _tg():
    import sys
    if str(STOLAB / "LongLiveVault") not in sys.path:
        sys.path.insert(0, str(STOLAB / "LongLiveVault"))
    from stolab_data import ticker_groups
    return ticker_groups


def _is_trading_day(d: date) -> bool:
    try:
        return bool(_tg()._is_td(d))
    except Exception:
        return d.weekday() < 5


def next_trading_day(d: date, is_td=None) -> date:
    return _tg().next_trading_day(d, is_td or _is_trading_day)


def expiry_week_friday(year: int, month: int, is_td=None) -> date:
    """그 달 옵션 만기(둘째 목요일)가 든 주의 금요일 — 휴장이면 그 주의 마지막 거래일 (LLV 정본)."""
    return _tg().expiry_week_friday(year, month, is_td or _is_trading_day)


def next_month(d: date) -> str:
    y, m = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
    return f"{y:04d}-{m:02d}"


def build(base: date, max_stale_days: int = 7, llv: Path = LLV, issue_status_path: Path | None = None,
          min_count: int = MIN_COUNT, effective: date | None = None) -> dict:
    """메가갤럭시 문서 — LLV ticker_groups.build_megagalaxy 위임 (llv 인자 = 테스트용 data 폴더).

    2026-11-15: 제외 표시 = LLV issue_status (llv/issue_status.parquet). 종전 data_ext/meta.parquet 인자는 제거
    (한 방향 원칙 — LLV 가 연구 폴더를 읽지 않는다)."""
    panel = pd.read_parquet(llv / "ohlcv" / "panel.parquet", columns=["Date", "Ticker", "Name", "MarketCap", "Foreign_Ratio"])
    eff = effective or next_trading_day(base)
    return _tg().build_megagalaxy(base, max_stale_days=max_stale_days, min_count=min_count, effective=eff,
                                  panel=panel, issue_status_path=issue_status_path or (llv / "issue_status.parquet"),
                                  raw_dir=llv / "raw")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--base", help="기준일 YYYY-MM-DD (적용 = 다음 거래일)")
    g.add_argument("--expiry", help="YYYY-MM — 그 달 옵션 만기 주 금요일을 기준일로 (정규 경로)")
    ap.add_argument("--min-count", type=int, default=MIN_COUNT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", default=None, help="비교할 기존 universe json")
    a = ap.parse_args(argv)
    if a.expiry:
        y, m = (int(x) for x in a.expiry.split("-"))
        base = expiry_week_friday(y, m)
    else:
        base = date.fromisoformat(a.base)
    res = build(base, min_count=a.min_count)
    tk = [s["ticker"] for s in res["stocks"]]
    print(f"[universe] base {res['base_date']} → 적용 {res['effective_date']} · {res['count']}종목 "
          f"(4조/30% {res['count_rule']} + 바닥 채움 {len(res['floor_filled'])})")
    if res["floor_filled"]:
        nm = {s["ticker"]: s["name"] for s in res["stocks"]}
        print("  바닥 채움: " + ", ".join(f"{nm[t]}({t})" for t in res["floor_filled"]))
    if res["floor_shortfall"]:
        print(f"  ⚠ LLV 패널 안 후보가 모자라 {res['floor_shortfall']}종목 부족 — 패널 밖으로는 채우지 않음(케인 10-31)")
    pv = res["data_provenance"]
    if pv["stale_excluded"]:
        print(f"  ⚠ 기준일 근처 값 없음(제외): {pv['stale_excluded']}")
    if pv["outside_panel_4jo"]:
        print(f"  ⚠ 패널 밖 4조↑ (KRX {pv['krx_raw_date']}): " + ", ".join(f"{o['name']}({o['ticker']})" for o in pv["outside_panel_4jo"]))
    tg = _tg()
    prev = [x for x in tg.list_docs("megagalaxy") if x.get("effective_date") != res["effective_date"]]
    old = json.loads(Path(a.check).read_text(encoding="utf-8")) if a.check else (prev[-1] if prev else None)
    if old:
        ot = {s["ticker"]: s["name"] for s in old["stocks"]}; nt = {s["ticker"]: s["name"] for s in res["stocks"]}
        add = [f"{nt[t]}({t})" for t in nt if t not in ot]; rem = [f"{ot[t]}({t})" for t in ot if t not in nt]
        print(f"  vs {a.check or old.get('_file')} ({old.get('count')}종목): 공통 {len(set(ot) & set(nt))} · "
              f"편입 {len(add)} {add} · 편출 {len(rem)} {rem}")
    if a.dry_run:
        return 0
    out = tg.save(res, overwrite=True)          # LLV groups/megagalaxy_{적용일}.json (명단 정본)
    print(f"  → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
