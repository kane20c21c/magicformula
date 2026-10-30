# -*- coding: utf-8 -*-
"""
build_universe.py — 스윙 유니버스 월 파일 생성 (STRATEGY.md §2 · §8-1, 작업 계획 WP4 — 2026-10-29)
====================================================================================================
지금까지 universe_YYYY-MM_*.json 은 손으로 만들었다(9월분 = 2026-09-12, 클로이). 10월분은 만들어지지 않아
10월 내내 9월 파일(8/31 기준)로 폴백했다 (2026-10-29 확인). 같은 절차를 스크립트로 옮긴다.

규칙 (유니버스 2번째, 2026-08-17 케인 확정): 기준일(월말) 시총 ≥ 4조원 & 외국인지분율 ≥ 30%,
  우선주·스팩·관리종목·ETF 제외. 다음 달 첫 거래일부터 적용, 월중 유지.
원천 (9월분과 같게): LLV panel.parquet 의 MarketCap·Foreign_Ratio — 기준일 이전 마지막 유효값.
  제외 플래그 = data_ext/meta.parquet(is_pref·is_spac·is_managed, 기준일 이전 최신 월) + 티커 끝자리≠0 우선주 보조.
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


def _is_trading_day(d: date) -> bool:
    """LLV 거래일 캘린더(가능할 때) — 없으면 주말만 제외."""
    try:
        import sys
        if str(STOLAB / "LongLiveVault") not in sys.path:
            sys.path.insert(0, str(STOLAB / "LongLiveVault"))
        from stolab_data.trading_calendar import is_trading_day
        return bool(is_trading_day(d))
    except Exception:
        return d.weekday() < 5


def next_trading_day(d: date, is_td=None) -> date:
    is_td = is_td or _is_trading_day
    x = d + timedelta(days=1)
    while not is_td(x):
        x += timedelta(days=1)
    return x


def expiry_week_friday(year: int, month: int, is_td=None) -> date:
    """그 달 옵션 만기(둘째 목요일)가 든 주의 금요일 — 휴장이면 그 주의 마지막 거래일."""
    is_td = is_td or _is_trading_day
    first = date(year, month, 1)
    thu = first + timedelta(days=(3 - first.weekday()) % 7 + 7)
    for k in (1, 0, -1, -2, -3):                     # 금 → 목 → 수 → 화 → 월
        x = thu + timedelta(days=k)
        if is_td(x):
            return x
    raise ValueError(f"{year}-{month:02d} 만기 주에 거래일 없음")


def next_month(d: date) -> str:
    y, m = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
    return f"{y:04d}-{m:02d}"


def build(base: date, max_stale_days: int = 7, llv: Path = LLV, meta_path: Path | None = None,
          min_count: int = MIN_COUNT, effective: date | None = None) -> dict:
    p = pd.read_parquet(llv / "ohlcv" / "panel.parquet", columns=["Date", "Ticker", "Name", "MarketCap", "Foreign_Ratio"])
    p["Date"] = pd.to_datetime(p.Date); p["Ticker"] = p.Ticker.astype(str).str.zfill(6)
    p = p[p.Date <= pd.Timestamp(base)].dropna(subset=["MarketCap", "Foreign_Ratio"])
    last = p.sort_values("Date").groupby("Ticker").last()
    stale = last[last.Date < pd.Timestamp(base) - pd.Timedelta(days=max_stale_days)]
    last = last.drop(stale.index)

    meta = pd.read_parquet(meta_path or (HERE / "data_ext" / "meta.parquet")).reset_index()
    meta["ticker"] = meta.ticker.astype(str).str.zfill(6); meta["date"] = pd.to_datetime(meta.date)
    meta = meta[meta.date <= pd.Timestamp(base)].sort_values("date").groupby("ticker").last()
    flag = meta[["is_pref", "is_spac", "is_managed"]].astype("boolean").fillna(False).astype(bool).any(axis=1)
    excl = set(flag[flag].index) | ETF | {t for t in last.index if not t.endswith("0")}

    cand = last[~last.index.isin(excl)]
    sel = cand[(cand.MarketCap >= MKTCAP_MIN) & (cand.Foreign_Ratio >= FOREIGN_MIN)].sort_values("MarketCap", ascending=False)
    n_rule = len(sel)
    # 최소 개수 바닥 (U6, 2026-10-31) — 채움 후보는 **이 패널(LLV) 안**의 외국인 ≥ 30% 종목만, 시총 큰 순서
    floor = cand[(cand.Foreign_Ratio >= FOREIGN_MIN) & ~cand.index.isin(sel.index)].sort_values("MarketCap", ascending=False)
    fill = floor.head(max(0, min_count - n_rule))
    sel = pd.concat([sel, fill])
    shortfall = max(0, min_count - len(sel))

    # 패널 밖 4조↑ 점검 (KRX raw)
    outside, raw_date = [], None
    raws = sorted((llv / "raw").glob("krx_kospi_*.parquet"))
    raws = [f for f in raws if f.stem.split("_")[-1] <= base.strftime("%Y%m%d")]
    if raws:
        rd = raws[-1].stem.split("_")[-1]; raw_date = f"{rd[:4]}-{rd[4:6]}-{rd[6:]}"
        k = pd.concat([pd.read_parquet(f) for f in (llv / "raw" / f"krx_kospi_{rd}.parquet", llv / "raw" / f"krx_kosdaq_{rd}.parquet") if f.exists()])
        k["Ticker"] = k.Ticker.astype(str).str.zfill(6)
        big = k[(k.MarketCap >= MKTCAP_MIN) & k.Ticker.str.endswith("0") & ~k.Ticker.isin(set(last.index) | set(stale.index) | excl)]
        outside = [{"ticker": r.Ticker, "name": r.Name, "mktcap_krw": int(r.MarketCap)} for r in big.sort_values("MarketCap", ascending=False).itertuples()]

    filled = set(fill.index)
    stocks = [{"ticker": t, "name": r.Name, "mktcap_krw": int(round(r.MarketCap)), "foreign_ratio_pct": round(float(r.Foreign_Ratio), 2),
               "rule": "floor" if t in filled else "4jo_30pct"}
              for t, r in sel.iterrows()]
    eff = effective or next_trading_day(base)
    return {
        "strategy_id": STRATEGY_ID, "base_date": base.isoformat(),
        "effective_date": eff.isoformat(), "apply_month": eff.strftime("%Y-%m"),
        "filters": {"mktcap_min_krw": MKTCAP_MIN, "foreign_ratio_min_pct": FOREIGN_MIN, "min_count": min_count,
                    "floor_rule": "4조 & 30% 가 min_count 미만이면 LLV 패널 안 외국인 ≥ 30% 종목을 시총순으로 채움"},
        "created_by": f"build_universe.py ({datetime.now():%Y-%m-%d %H:%M}) — 유니버스 2번째(4조/30%) + 최소 {min_count} (U6 2026-10-31)",
        "count": len(stocks), "count_rule": n_rule, "floor_filled": sorted(filled), "floor_shortfall": shortfall,
        "data_provenance": {
            "derived_from": "LLV panel.parquet (MarketCap·Foreign_Ratio, 기준일 이전 마지막 유효값) + data_ext/meta 우선주·스팩·관리 플래그",
            "panel_tickers": int(len(last) + len(stale)), "stale_excluded": sorted(stale.index.tolist()),
            "krx_raw_date": raw_date, "outside_panel_4jo": outside,
        },
        "stocks": stocks,
    }


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
    prev_files = sorted(HERE.glob("data/universe_*.json"))
    cmp = Path(a.check) if a.check else (prev_files[-1] if prev_files else None)
    if cmp and cmp.exists():
        old = json.loads(cmp.read_text(encoding="utf-8"))
        ot = {s["ticker"]: s["name"] for s in old["stocks"]}; nt = {s["ticker"]: s["name"] for s in res["stocks"]}
        add = [f"{nt[t]}({t})" for t in nt if t not in ot]; rem = [f"{ot[t]}({t})" for t in ot if t not in nt]
        print(f"  vs {cmp.name} ({old.get('count')}종목): 공통 {len(set(ot) & set(nt))} · 편입 {len(add)} {add} · 편출 {len(rem)} {rem}")
    if a.dry_run:
        return 0
    out = HERE / "data" / f"universe_{res['effective_date']}_4jo_30pct_min{res['filters']['min_count']}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
