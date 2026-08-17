"""shadow_track.py — 직전 운영 모델 v1.1.1.2 그림자 추적 (Kane 지시 2026-08-17).

v1.2.2.3 전환 시점의 **실제 가상계좌 상태에서 분기**해, 구모델 규칙으로 매일
EOD 재현한다. 다음 달 모델 평가에서 실계정(v1.2.2.3)과 비교하는 용도.

그림자 모델 v1.1.1.2
  유니버스 : 4조/25% (universe_2026-07_4jo_25pct.json — 분기 시점 운영본 고정)
  진입     : 60일 고점 −10% 눌림 + MA120 위, onset 만 (v1 진입규칙 1번째)
  청산     : 시간연동 4분기, 배율 없음 (2026-08-01 규칙 = 청산규칙 2번째)
             D+0~5 ref×(1−0.20) / D+6~ 피크>평단 피크×(1−0.05) · 이하 평단×(1−0.15)
  계좌     : 가상계좌와 동일 — 슬롯 200만 × 20 · 종목당 max2 · 쿨다운 1일 ·
             고가주 1주 허용 · 매수 0.015% / 매도 0.215% · T+1

근사 (실엔진과의 차이 — 비교 시 감안)
  · 매수 = 다음 거래일 시가 (실엔진은 09:10 실시간가)
  · 손절 감지 = 일중 저가, 체결 = min(시가, 손절선) × 0.995 (실엔진은 15분 샘플 + 5분 후 시장가)
  · peak = 일별 고가 누적 (전일까지 기준으로 당일 판정 — 룩어헤드 방지)

데이터: LLV core/extend parquet (유니버스 60종목 전부 포함 확인, 2026-08-17).
상태:   data/shadow_v1112_state.json / 로그: data/shadow_v1112_equity.csv

실행 (맥에서, 평일 20:35 이후 권장 — 20:30 KIS 배치로 당일 종가 확정 후):
  python3 shadow_track.py --init     # 최초 1회: 가상계좌 state.json 에서 분기
  python3 shadow_track.py            # 마지막 처리일 이후 ~ LLV 최신일까지 전진 (멱등)
  python3 shadow_track.py --report   # 실계정(equity.csv)과 나란히 비교 출력
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

KMS = Path(__file__).resolve().parent
STOLAB = KMS.parents[1]
LLV = STOLAB / "longlivevault"
SP_PAPER = STOLAB / "StockPortfolio" / "data" / "paper"

STATE_F = KMS / "data" / "shadow_v1112_state.json"
EQUITY_F = KMS / "data" / "shadow_v1112_equity.csv"
UNIVERSE_F = KMS / "data" / "universe_2026-07_4jo_25pct.json"   # 분기 시점 운영본 고정

# ── v1.1.1.2 파라미터 ──────────────────────────────────────────
HIGH_WINDOW, HIGH_MIN_P = 60, 30
TREND_MA, TREND_MIN_P = 120, 80
PULLBACK = 0.90
STOP_SWITCH_DAY = 5
STOP_EARLY, STOP_LATE_UP, STOP_LATE_FLAT = 0.20, 0.05, 0.15    # 배율 없음 (v1.1.0)
SLOT_KRW, MAX_POS, MAX_SLOTS = 2_000_000.0, 20, 2
COOLDOWN_TD = 1
BUY_FEE, SELL_FEE, SELL_TAX = 0.00015, 0.00015, 0.002
STOP_FILL = 0.995
ALLOW_SINGLE_SHARE = True


def load_prices(tickers: list[str], start: str = "2025-06-01") -> dict[str, pd.DataFrame]:
    """LLV core+extend 에서 유니버스 종목 OHLC 로드 → {col: wide DataFrame}."""
    frames = []
    for f in ("core.parquet", "extend.parquet"):
        d = pd.read_parquet(LLV / "data" / "ohlcv" / f,
                            columns=["Date", "Ticker", "Open", "High", "Low", "Close"])
        frames.append(d[d.Ticker.isin(tickers)])
    d = pd.concat(frames).drop_duplicates(["Date", "Ticker"])
    d = d[d.Date >= pd.Timestamp(start)]
    out = {}
    for c in ("Open", "High", "Low", "Close"):
        out[c.lower()] = d.pivot(index="Date", columns="Ticker", values=c).sort_index()
    missing = [t for t in tickers if t not in out["close"].columns]
    if missing:
        print(f"⚠ LLV 에 없는 종목 {missing} — 그림자에서 제외")
    return out


def init_state() -> dict:
    """실제 가상계좌 state.json 에서 분기."""
    sp = json.loads((SP_PAPER / "state.json").read_text(encoding="utf-8"))
    positions = {}
    for tk, p in (sp.get("positions") or {}).items():
        positions[tk] = {
            "name": p.get("name", ""), "slots": int(p.get("slots", 1)),
            "shares": int(p.get("shares", 0)),
            "avg_price": float(p.get("avg_price", 0.0)),
            "peak": float(p.get("peak", p.get("avg_price", 0.0))),
            "entry_date": p.get("entry_date"),
        }
    pending = sp.get("pending_settlements") or []
    if isinstance(pending, dict):
        pending = [{"date": k, "amount": v} for k, v in pending.items()]
    st = {
        "model": "v1.1.1.2 (shadow)",
        "forked_from": "StockPortfolio/data/paper/state.json",
        "forked_at": datetime.now().isoformat(timespec="seconds"),
        "fork_note": f"실계정 updated_at={sp.get('updated_at')} 기준 분기",
        "cash": float(sp.get("cash", 0.0)),
        "pending": [{"date": str(x.get("date")), "amount": float(x.get("amount", 0.0))}
                    for x in pending],
        "positions": positions,
        "last_sell_date": dict(sp.get("last_sell_date") or {}),
        "pending_buys": [],                   # [{ticker, queued_on}]
        "last_processed": sp.get("last_signal_date") or "2026-08-14",
    }
    STATE_F.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    if not EQUITY_F.exists():
        EQUITY_F.write_text("date,cash,pending,mv,total,n_pos,note\n", encoding="utf-8")
    print(f"분기 완료 — 현금 {st['cash']:,.0f}원 · 보유 {len(positions)}종목 · "
          f"기준일 {st['last_processed']}")
    return st


def stop_line(pos: dict, d_plus: int) -> float:
    avg = pos["avg_price"]
    peak = pos["peak"]
    above = peak > avg
    if d_plus <= STOP_SWITCH_DAY:
        return (peak if above else avg) * (1 - STOP_EARLY)
    if above:
        return peak * (1 - STOP_LATE_UP)
    return avg * (1 - STOP_LATE_FLAT)


def advance(st: dict) -> dict:
    uni = json.loads(UNIVERSE_F.read_text(encoding="utf-8"))
    tickers = [s["ticker"] for s in uni["stocks"]]
    names = {s["ticker"]: s["name"] for s in uni["stocks"]}
    px = load_prices(tickers)
    cl, op, hi, lo = px["close"], px["open"], px["high"], px["low"]
    dates = cl.index

    high_n = cl.rolling(HIGH_WINDOW, min_periods=HIGH_MIN_P).max()
    ma = cl.rolling(TREND_MA, min_periods=TREND_MIN_P).mean()
    depth = (high_n - cl) / high_n
    cond = ((cl <= PULLBACK * high_n) & (cl > ma)).fillna(False)
    onset = cond & ~cond.shift(1).fillna(False)

    last = pd.Timestamp(st["last_processed"])
    todo = [d for d in dates if d > last]
    if not todo:
        print(f"처리할 새 거래일 없음 (마지막 {st['last_processed']})")
        return st

    didx = {d: i for i, d in enumerate(dates)}
    rows = []
    for d in todo:
        ds = d.strftime("%Y-%m-%d")
        # 1) T+1 결제
        settle = [p for p in st["pending"] if p["date"] <= ds]
        st["cash"] += sum(p["amount"] for p in settle)
        st["pending"] = [p for p in st["pending"] if p["date"] > ds]

        # 2) 손절 (전일까지의 peak 기준, 당일 저가 감지)
        for tk in list(st["positions"]):
            pos = st["positions"][tk]
            if tk not in didx or tk not in cl.columns or pd.isna(lo.loc[d, tk]):
                continue
            e_i = didx.get(pd.Timestamp(pos["entry_date"]))
            d_plus = (didx[d] - e_i) if e_i is not None else 99
            line = stop_line(pos, d_plus)
            if lo.loc[d, tk] <= line:
                fill = min(op.loc[d, tk], line) if not pd.isna(op.loc[d, tk]) else line
                fill *= STOP_FILL
                net = fill * pos["shares"] * (1 - SELL_FEE - SELL_TAX)
                st["pending"].append({"date": ds, "amount": net})   # T+1 (다음 거래일 결제)
                st["last_sell_date"][tk] = ds
                del st["positions"][tk]

        # 3) 매수 집행 (전일 신호 → 오늘 시가)
        used = sum(p["slots"] for p in st["positions"].values())
        for q in st["pending_buys"]:
            tk = q["ticker"]
            if used >= MAX_POS or tk not in cl.columns or pd.isna(op.loc[d, tk]):
                continue
            pos = st["positions"].get(tk)
            if pos and pos["slots"] >= MAX_SLOTS:
                continue
            price = float(op.loc[d, tk])
            shares = int(SLOT_KRW // price)
            if shares == 0 and ALLOW_SINGLE_SHARE:
                shares = 1
            cost = price * shares * (1 + BUY_FEE)
            if shares <= 0 or cost > st["cash"]:
                continue
            st["cash"] -= cost
            used += 1
            if pos:
                tot = pos["avg_price"] * pos["shares"] + price * shares
                pos["shares"] += shares
                pos["avg_price"] = tot / pos["shares"]
                pos["slots"] += 1
                pos["entry_date"] = ds                    # 불타기 → D+ 리셋
                pos["peak"] = max(pos["peak"], price)
            else:
                st["positions"][tk] = {"name": names.get(tk, ""), "slots": 1,
                                       "shares": shares, "avg_price": price,
                                       "peak": price, "entry_date": ds}
        st["pending_buys"] = []

        # 4) 오늘 신호 → 내일 매수 큐 (눌림 깊은 순)
        sig = [(float(depth.loc[d, tk]), tk) for tk in cl.columns
               if bool(onset.loc[d, tk])]
        sig.sort(reverse=True)
        for dep, tk in sig:
            ls = st["last_sell_date"].get(tk)
            if ls == ds:                                   # 쿨다운 1거래일
                continue
            pos = st["positions"].get(tk)
            if pos and pos["slots"] >= MAX_SLOTS:
                continue
            st["pending_buys"].append({"ticker": tk, "queued_on": ds})

        # 5) peak 갱신 + 평가
        mv = 0.0
        for tk, pos in st["positions"].items():
            if tk in hi.columns and not pd.isna(hi.loc[d, tk]):
                pos["peak"] = max(pos["peak"], float(hi.loc[d, tk]))
            c = cl.loc[d, tk] if tk in cl.columns else np.nan
            mv += (float(c) if not pd.isna(c) else pos["avg_price"]) * pos["shares"]
        pend = sum(p["amount"] for p in st["pending"])
        rows.append({"date": ds, "cash": round(st["cash"], 2), "pending": round(pend, 2),
                     "mv": round(mv, 2), "total": round(st["cash"] + pend + mv, 2),
                     "n_pos": len(st["positions"]), "note": ""})
        st["last_processed"] = ds

    STATE_F.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.DataFrame(rows).to_csv(EQUITY_F, mode="a", header=not EQUITY_F.exists(),
                              index=False)
    for r in rows:
        print(f"{r['date']}  총자산 {r['total']:>14,.0f}원  보유 {r['n_pos']}종목")
    return st


def report():
    sh = pd.read_csv(EQUITY_F)
    real = pd.read_csv(SP_PAPER / "equity.csv")   # date,cash,pending_cash,...,total,...
    m = sh.merge(real[["date", "total"]], on="date", suffixes=("_그림자", "_실계정"))
    if not len(m):
        print("겹치는 날짜 없음 — 실계정 equity.csv 스키마 확인 필요")
        return
    base_s, base_r = m.iloc[0]["total_그림자"], m.iloc[0]["total_실계정"]
    print(f"{'날짜':10s} {'그림자 v1.1.1.2':>16s} {'실계정 v1.2.2.3':>16s} {'격차':>10s}")
    for _, r in m.iterrows():
        gap = (r["total_실계정"] / base_r - r["total_그림자"] / base_s) * 100
        print(f"{r['date']:10s} {r['total_그림자']:>15,.0f}원 {r['total_실계정']:>15,.0f}원 "
              f"{gap:>+9.2f}%p")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true", help="가상계좌 state 에서 분기 (최초 1회)")
    ap.add_argument("--report", action="store_true", help="실계정과 비교 출력")
    a = ap.parse_args()
    if a.init:
        init_state()
    elif a.report:
        report()
    else:
        if not STATE_F.exists():
            print("상태 파일 없음 — 먼저 --init 을 실행하세요.")
        else:
            advance(json.loads(STATE_F.read_text(encoding="utf-8")))
