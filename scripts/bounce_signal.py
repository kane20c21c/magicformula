#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/bounce_signal.py — 반등 룰(룰 #5) 일일 신호. 팩트 체인 13단계 (20:30 LLV 확정 뒤, 평일).

    python3 scripts/bounce_signal.py              # 패널 마지막 날짜 기준 (휴장 가드)
    python3 scripts/bounce_signal.py 20260731     # 특정 날짜 (가드 생략)
    python3 scripts/bounce_signal.py --no-mail    # JSON 만
    python3 scripts/bounce_signal.py --dry-run    # 저장·메일 없이 표만 출력

입력 = LLV data/ohlcv/panel.parquet (수정주가, 20:30 확정) 에서 ETF 섹터 제외.
출력 = output/signals_bounce/YYYYMMDD.json — 그날 끝난 드라이브 전부(signal·deadcat 플래그) + 메타.
메일 = 신호가 1건 이상일 때만 (GMAIL_*/ALERT_EMAIL — mop_model gauge_notify 와 같은 .env). 실패해도 JSON 은 남는다.
"""
from __future__ import annotations
import json, sys, logging
from datetime import date, datetime
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]; STOLAB_ROOT = PROJECT_ROOT.parent
for p in (PROJECT_ROOT, STOLAB_ROOT / "LongLiveVault", PROJECT_ROOT / "mop_model" / "src"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
from bounce_rule import scan_drives, DEFAULT                       # noqa: E402
PANEL = STOLAB_ROOT / "LongLiveVault" / "data" / "ohlcv" / "panel.parquet"
OUT_DIR = PROJECT_ROOT / "output" / "signals_bounce"
RED, BLUE = "#ef5350", "#1976D2"
log = logging.getLogger("bounce_signal")


def load_panel() -> pd.DataFrame:
    p = pd.read_parquet(PANEL, columns=["Date", "Ticker", "Name", "Close"])
    p["Ticker"] = p["Ticker"].astype(str).str.zfill(6)
    try:                                                            # ETF 제외 — 섹터 정본은 LLV 티커 리스트
        from stolab_data.core_tickers import TICKER_LIST
        from stolab_data.extend_tickers import EXTEND_LIST
        etf = {t for t, _n, sec in list(TICKER_LIST) + list(EXTEND_LIST) if sec in DEFAULT.exclude_sectors}
    except Exception as e:
        log.warning("섹터 목록 로드 실패(%s) — 알려진 ETF 2종만 제외", e); etf = {"102110", "252650"}
    return p[~p["Ticker"].isin(etf)]


def fmt_pct(x: float) -> str:
    color = RED if x > 0 else BLUE
    return f'<span style="color:{color}">{x*100:+.1f}%</span>'


def build_mail(df: pd.DataFrame, asof: str) -> tuple[str, str]:
    sig = df[df.signal]
    next_note = "내일 09:00 시가 진입 후보 · 권고 보유 5거래일 · 저점(마지막 하락일 종가) 종가 이탈 시 익일 시가 정리"
    rows = "".join(
        f"<tr><td>{r.ticker}</td><td>{r.name}</td><td style='text-align:right'>{r.drive_days}일</td>"
        f"<td style='text-align:right'>{fmt_pct(r.drive_drop)}</td><td style='text-align:right'>{fmt_pct(r.d1_ret)}</td>"
        f"<td style='text-align:right'>{r.strength:.2f}σ</td><td style='text-align:right'>{r.low_close:,.0f}원</td></tr>"
        for r in sig.itertuples())
    others = df[~df.signal]
    rows2 = "".join(
        f"<tr><td>{r.ticker}</td><td>{r.name}</td><td style='text-align:right'>{r.drive_days}일</td><td style='text-align:right'>{fmt_pct(r.drive_drop)}</td>"
        f"<td style='text-align:right'>{fmt_pct(r.d1_ret)}</td><td style='text-align:right'>{r.strength:.2f}σ</td><td>{'데드캣형' if r.deadcat else ''}</td></tr>"
        for r in others.itertuples())
    html = f"""<div style="font-family:-apple-system,sans-serif;font-size:14px">
<h3>반등 룰 신호 — {asof} 종가 기준 · {len(sig)}건</h3><p>{next_note}</p>
<table border="1" cellpadding="4" style="border-collapse:collapse"><tr><th>종목</th><th>이름</th><th>드라이브</th><th>낙폭</th><th>D+1</th><th>강도</th><th>저점(종가)</th></tr>{rows}</table>
<p style="margin-top:14px;color:#666">같은 날 끝난 다른 드라이브 (참고, 신호 아님)</p>
<table border="1" cellpadding="4" style="border-collapse:collapse;color:#666"><tr><th>종목</th><th>이름</th><th>드라이브</th><th>낙폭</th><th>D+1</th><th>강도</th><th></th></tr>{rows2}</table>
<p style="color:#999;font-size:12px">규칙 {DEFAULT.version}: 3일↑ 연속 하락 드라이브 종료 · 낙폭 ≤ −20% · D+1 ≥ 1.65σ60. 근거 MagicFormula/bounce_rule/README.md</p></div>"""
    names = sig.name.tolist(); head = ", ".join(names[:5]) + (f" 외 {len(names)-5}" if len(names) > 5 else "")
    return f"[반등 룰] {asof} 신호 {len(sig)}건 — {head}", html


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = [a for a in argv if not a.startswith("--")]; flags = {a for a in argv if a.startswith("--")}
    target = args[0] if args else None
    if target is None:
        try:
            from stolab_data.trading_calendar import is_trading_day
            if not is_trading_day(date.today()):
                print(f"⏸ {date.today()} 휴장일 — 반등 룰 skip"); return 0
        except Exception as e:
            print(f"⚠ 거래일 캘린더 확인 실패({e}) — 가드 없이 진행")
    panel = load_panel()
    asof = pd.Timestamp(target) if target else panel["Date"].max()
    if target is None and asof.date() != date.today():
        print(f"⚠ 패널 마지막 날짜 {asof.date()} ≠ 오늘 {date.today()} — LLV 20:30 확정 전일 수 있다 (그대로 진행)")
    df = scan_drives(panel, asof=asof)
    sig = df[df.signal]
    pd.set_option("display.width", 200)
    print(f"반등 룰 {DEFAULT.version} · {asof.date()} · 패널 {panel.Ticker.nunique()}종목 · 끝난 드라이브 {len(df)}건 · 신호 {len(sig)}건")
    print(df.to_string(index=False) if len(df) else "(끝난 드라이브 없음)")
    if "--dry-run" in flags: return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"version": DEFAULT.version, "as_of": asof.strftime("%Y-%m-%d"), "generated_at": datetime.now().isoformat(timespec="seconds"),
           "rule": {"min_drive_days": DEFAULT.min_drive_days, "min_drop": DEFAULT.min_drop, "strong_sigma": DEFAULT.strong_sigma, "deadcat_sigma": DEFAULT.deadcat_sigma, "sigma_window": DEFAULT.sigma_window},
           "entry": "next_open", "hold_days": DEFAULT.hold_days, "signals": sig.to_dict(orient="records"), "ended_drives": df.to_dict(orient="records")}
    path = OUT_DIR / f"{asof.strftime('%Y%m%d')}.json"
    tmp = path.with_suffix(".json.tmp"); tmp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"); tmp.replace(path)
    print(f"저장: {path}")
    if len(sig) and "--no-mail" not in flags:
        try:
            from gauge_notify import send_email
            subject, html = build_mail(df, asof.strftime("%Y-%m-%d"))
            print("메일:", "발송" if send_email(subject, html) else "실패(미설정 또는 오류) — JSON 은 저장됨")
        except Exception as e:
            print(f"메일 모듈 실패({e}) — JSON 은 저장됨")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
