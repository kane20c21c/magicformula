#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/jdh_signal.py — 장단현포트(룰 #6) 15:10 잠정 판정 (케인 2026-11-23).

    python3 scripts/jdh_signal.py              # 오늘 (거래일·시각 가드)
    python3 scripts/jdh_signal.py --force      # 가드 생략 (수동 확인용 — 장 마감 뒤엔 현재가 = 종가)
    python3 scripts/jdh_signal.py --dry-run    # 저장 없이 출력만

1) 오늘 적용 유니버스의 선행 3종목 (없으면 만든다 — 유니버스 조정 뒤 첫 실행에서 자가치유)
2) 선행 3종목·KOSPI200 의 15:10 현재가(잠정) ÷ 5거래일 전 종가 → 5일 수익률 (완전 4일 + 잠정 1일)
3) ⑦ = regime_history 최신 확정(전날) → 틀·이동 슬롯
출력 = output/signals_jdh/YYYYMMDD.json → StockPortfolio app/jdh 가 15:17 에 읽어 체결.
가격: 종목 = LLV data_service.get_current_price (KIS → 토스 → store), 지수 = LLV kis_fetcher.fetch_index_today("KOSPI200").
   'store' 폴백(실시간 아님)은 현재가로 쓰지 않는다 → 그 종목 수익률 없음 → 선행 이동 0.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]; STOLAB_ROOT = PROJECT_ROOT.parent
for p in (PROJECT_ROOT, STOLAB_ROOT / "LongLiveVault"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from jdh_rule import DEFAULT, ensure_picks, judge, latest_p7       # noqa: E402
from jdh_rule.config import PANEL, KOSPI200_PARQUET, SIGNAL_DIR  # noqa: E402

KST = timezone(timedelta(hours=9))
log = logging.getLogger("jdh_signal")
WINDOW = ("15:05", "15:19")


def _llv():
    try:
        from dotenv import load_dotenv
        load_dotenv(STOLAB_ROOT / "LongLiveVault" / ".env")
    except Exception:
        pass
    from stolab_data.trading_calendar import is_trading_day, get_recent_trading_days
    return is_trading_day, get_recent_trading_days


def stock_now(t: str) -> tuple[float | None, str]:
    from stolab_data.data_service import get_current_price
    try:
        r = get_current_price(t)
    except Exception as e:  # noqa: BLE001
        return None, f"실패: {e}"
    if not r or r.get("Source") == "store" or not r.get("Price"):
        return None, (r or {}).get("Source", "없음")
    return float(r["Price"]), r.get("Source", "")


def k200_now() -> tuple[float | None, str]:
    try:
        from stolab_data.kis_fetcher import fetch_index_today
        df = fetch_index_today("KOSPI200")
        if df is not None and not df.empty and float(df.iloc[0]["Close"]) > 0:
            return float(df.iloc[0]["Close"]), "KIS"
    except Exception as e:  # noqa: BLE001
        return None, f"실패: {e}"
    return None, "없음"


def base_closes(tickers: list[str], base_day) -> tuple[dict, float | None, str | None]:
    p = pd.read_parquet(PANEL, columns=["Date", "Ticker", "Close"])
    p["Ticker"] = p["Ticker"].astype(str).str.zfill(6)
    last = str(pd.to_datetime(p.Date).max().date())
    p = p[(p.Ticker.isin(tickers)) & (pd.to_datetime(p.Date) == pd.Timestamp(base_day))]
    st = {r.Ticker: float(r.Close) for r in p.itertuples()}
    k = pd.read_parquet(KOSPI200_PARQUET).set_index("Date")["Close"]; k.index = pd.to_datetime(k.index)
    kb = float(k.loc[pd.Timestamp(base_day)]) if pd.Timestamp(base_day) in k.index else None
    return st, kb, last


def recent_closes(tickers: list[str], days: list) -> dict:
    """화면용 — 완전 4일 종가 (종목·KOSPI200)."""
    p = pd.read_parquet(PANEL, columns=["Date", "Ticker", "Close"])
    p["Ticker"] = p["Ticker"].astype(str).str.zfill(6); p["Date"] = pd.to_datetime(p.Date)
    p = p[p.Ticker.isin(tickers) & p.Date.isin([pd.Timestamp(d) for d in days])]
    out = {t: {str(r.Date.date()): float(r.Close) for r in g.itertuples()} for t, g in p.groupby("Ticker")}
    k = pd.read_parquet(KOSPI200_PARQUET).set_index("Date")["Close"]; k.index = pd.to_datetime(k.index)
    out["KOSPI200"] = {str(d): float(k.loc[pd.Timestamp(d)]) for d in days if pd.Timestamp(d) in k.index}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    is_td, recent_tds = _llv()
    now = datetime.now(KST); today = now.date()
    if not a.force:
        if not is_td(today):
            log.info("%s 휴장 — 종료", today); return 0
        if not (WINDOW[0] <= now.strftime("%H:%M") <= WINDOW[1]):
            log.info("판정 시간대(%s~%s) 밖 — 종료 (수동은 --force)", *WINDOW); return 0
    prev = recent_tds(today, 10)                       # 최근 먼저
    base_day, complete = prev[DEFAULT.sig_ret_days - 1], prev[:DEFAULT.sig_ret_days - 1][::-1]
    picks = ensure_picks(today)
    tk = [x["ticker"] for x in picks["leads"]]
    st_base, k_base, panel_last = base_closes(tk, base_day)
    leads = []
    for x in picks["leads"]:
        px, src = stock_now(x["ticker"])
        leads.append({"ticker": x["ticker"], "name": x.get("name", ""), "base_close": st_base.get(x["ticker"]),
                      "now": px, "now_source": src})
    kn, ksrc = k200_now()
    res = judge(today, leads, {"base_close": k_base, "now": kn, "now_source": ksrc}, latest_p7(today, prev))
    res.update({"generated_at": now.isoformat(timespec="seconds"), "base_date": str(base_day),
                "complete_days": [str(d) for d in complete], "panel_last": panel_last, "panel_fresh": panel_last == str(prev[0]),
                "picks": {"universe_effective": picks["universe_effective"], "window": picks.get("window"),
                          "leads": [{k: v for k, v in x.items() if k in ("ticker", "name", "bal_recent", "slope")} for x in picks["leads"]]},
                "closes": recent_closes(tk, complete)})
    txt = json.dumps(res, ensure_ascii=False, indent=2, default=str)
    if a.dry_run:
        print(txt); return 0
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    f = SIGNAL_DIR / f"{today:%Y%m%d}.json"; tmp = f.with_suffix(".tmp")
    tmp.write_text(txt, encoding="utf-8"); tmp.replace(f)
    log.info("저장 %s — 틀 %s · 이동 %s (선행 %s↑%s↓ → %+d, K200 %s → %+d) · ⑦ %s(%s)",
             f.name, res["frame"], res["move"], res["lead"]["up"], res["lead"]["dn"], res["lead"]["move"],
             None if res["k200"]["ret5"] is None else f"{res['k200']['ret5']*100:+.2f}%", res["k200"]["move"],
             res["p7"].get("label"), res["p7"].get("date"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
