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

사용:
  python3 build_universe.py --base 2026-10-30            → data/universe_2026-11_4jo_30pct.json (apply 2026-11)
  python3 build_universe.py --base 2026-08-31 --dry-run --check data/universe_2026-09_4jo_30pct.json   (재현 검증)
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
STOLAB = HERE.parents[1]
LLV = STOLAB / "LongLiveVault" / "data"
MKTCAP_MIN = 4_000_000_000_000
FOREIGN_MIN = 30.0
ETF = {"102110", "252650"}
STRATEGY_ID = "kr_pullback_largecap_foreign"


def next_month(d: date) -> str:
    y, m = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
    return f"{y:04d}-{m:02d}"


def build(base: date, max_stale_days: int = 7, llv: Path = LLV, meta_path: Path | None = None) -> dict:
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

    stocks = [{"ticker": t, "name": r.Name, "mktcap_krw": int(round(r.MarketCap)), "foreign_ratio_pct": round(float(r.Foreign_Ratio), 2)}
              for t, r in sel.iterrows()]
    return {
        "strategy_id": STRATEGY_ID, "base_date": base.isoformat(), "apply_month": next_month(base),
        "filters": {"mktcap_min_krw": MKTCAP_MIN, "foreign_ratio_min_pct": FOREIGN_MIN},
        "created_by": f"build_universe.py ({datetime.now():%Y-%m-%d %H:%M}) — 유니버스 2번째(4조/30%)",
        "count": len(stocks),
        "data_provenance": {
            "derived_from": "LLV panel.parquet (MarketCap·Foreign_Ratio, 기준일 이전 마지막 유효값) + data_ext/meta 우선주·스팩·관리 플래그",
            "panel_tickers": int(len(last) + len(stale)), "stale_excluded": sorted(stale.index.tolist()),
            "krx_raw_date": raw_date, "outside_panel_4jo": outside,
        },
        "stocks": stocks,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="기준일(월말 마지막 거래일) YYYY-MM-DD")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", default=None, help="비교할 기존 universe json")
    a = ap.parse_args(argv)
    res = build(date.fromisoformat(a.base))
    tk = [s["ticker"] for s in res["stocks"]]
    print(f"[universe] base {res['base_date']} → apply {res['apply_month']} · {res['count']}종목")
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
    out = HERE / "data" / f"universe_{res['apply_month']}_4jo_30pct.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
