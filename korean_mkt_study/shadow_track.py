"""shadow_track.py — 스윙 포트 그림자 추적 (다중 모델, Kane 지시 2026-09-12).

운영 모델이 v1.2.3.4 로 바뀌면서 **직전 두 모델**을 그림자로 돌린다:

  v1223  직전 운영 v1.2.2.3 — 규칙 효과만 보기 위해 **사이징은 신모델과 같은 10×400만**
         유니버스 4조/30% (월별 universe_*.json, 실계정과 동일 갱신)
         진입 W1: 40일 고점 −10% + MA200 위, onset 만
         청산 ③: 시간연동 4분기 × 변동배율(클립) — D+6 검사 **없음**
         분기: 2026-09-14 전환 시점의 가상계좌 상태 (--init --model v1223)
  v1112  그 전 모델 v1.1.1.2 — 2026-08-17 분기 그대로 계속 (20×200만, 4조/25% 고정,
         60일/MA120, 4분기 배율 없음). 상태 파일·이력 유지.

근사 (실엔진과의 차이 — 비교 시 감안)
  · 매수 = 다음 거래일 시가 (실엔진은 09:10 실시간가)
  · 손절 감지 = 일중 저가, 체결 = min(시가, 손절선) × 0.995 (실엔진은 15분 샘플 + 5분 후 시장가)
  · peak = 일별 고가 누적 (전일까지 기준으로 당일 판정 — 룩어헤드 방지)
  · 변동배율(v1223) = LLV panel YZ_20 ÷ 직전 252거래일 유니버스 일별 중앙값의 중앙값(전일까지)
    — LLV vol_scale.json 과 같은 정의를 EOD 재현 (과거일 재생이 가능하도록 자체 계산)

데이터: LLV core/extend parquet (OHLC) + panel.parquet (YZ_20).
상태:   data/shadow_{model}_state.json / 로그: data/shadow_{model}_equity.csv

실행 (맥미니, 평일 20:35 이후 — 20:30 KIS 배치로 당일 종가 확정 후):
  python3 shadow_track.py --init --model v1223    # 최초 1회: 가상계좌 state.json 에서 분기
  python3 shadow_track.py                          # 두 모델 모두 마지막 처리일 이후 전진 (멱등)
  python3 shadow_track.py --model v1112            # 한 모델만
  python3 shadow_track.py --report                 # 실계정(equity.csv)과 나란히 비교
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

KMS = Path(__file__).resolve().parent
STOLAB = KMS.parents[1]
LLV = STOLAB / "longlivevault"
SP_PAPER = STOLAB / "StockPortfolio" / "data" / "paper"
DATA = KMS / "data"

BUY_FEE, SELL_FEE, SELL_TAX = 0.00015, 0.00015, 0.002
STOP_FILL = 0.995
ALLOW_SINGLE_SHARE = True

MODELS = {
    "v1112": dict(
        label="v1.1.1.2", universe="fixed:universe_2026-07_4jo_25pct.json",
        high_window=60, high_min_p=30, trend_ma=120, trend_min_p=80, pullback=0.90,
        slot_krw=2_000_000.0, max_pos=20, max_slots=2, count_by="slots",
        stop_switch=5, stop_early=0.20, stop_late_up=0.05, stop_late_flat=0.15,
        vol_linked=False, clip_early=(0.10, 0.40), clip_late_up=(0.03, 0.15), clip_late_flat=(0.05, 0.25),
        check_day=None,
    ),
    "v1223": dict(
        label="v1.2.2.3", universe="monthly",
        high_window=40, high_min_p=20, trend_ma=200, trend_min_p=140, pullback=0.90,
        slot_krw=4_000_000.0, max_pos=10, max_slots=2, count_by="stocks",
        stop_switch=5, stop_early=0.20, stop_late_up=0.05, stop_late_flat=0.10,
        vol_linked=True, clip_early=(0.10, 0.40), clip_late_up=(0.03, 0.15), clip_late_flat=(0.05, 0.25),
        check_day=None,
    ),
}


def files(model: str) -> tuple[Path, Path]:
    return DATA / f"shadow_{model}_state.json", DATA / f"shadow_{model}_equity.csv"


# ── 유니버스 ────────────────────────────────────────────────────────────────
def _universe_docs() -> list[dict]:
    docs = []
    for f in sorted(DATA.glob("universe_*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if d.get("stocks"):
                d["_file"] = f.name
                docs.append(d)
        except Exception:
            pass
    return docs


def universe_for(model: str, d: pd.Timestamp) -> dict:
    """모델·날짜별 유니버스 문서. fixed: 지정 파일 / monthly: apply_month 매칭, 없으면 base_date ≤ d 최신."""
    spec = MODELS[model]["universe"]
    docs = _universe_docs()
    if spec.startswith("fixed:"):
        name = spec.split(":", 1)[1]
        return next(x for x in docs if x["_file"] == name)
    ym = d.strftime("%Y-%m")
    match = [x for x in docs if x.get("apply_month") == ym]
    if match:
        return match[-1]
    by_base = sorted(docs, key=lambda x: str(x.get("base_date", "")))
    older = [x for x in by_base if str(x.get("base_date", "")) <= d.strftime("%Y-%m-%d")]
    return older[-1] if older else by_base[0]          # 가장 이른 파일보다 앞선 날짜 → 그 파일


def all_universe_tickers(model: str) -> tuple[list[str], dict]:
    tks, names = set(), {}
    docs = _universe_docs()
    spec = MODELS[model]["universe"]
    if spec.startswith("fixed:"):
        docs = [x for x in docs if x["_file"] == spec.split(":", 1)[1]]
    for doc in docs:
        for s in doc["stocks"]:
            tks.add(s["ticker"]); names[s["ticker"]] = s["name"]
    return sorted(tks), names


# ── 데이터 ──────────────────────────────────────────────────────────────────
def load_prices(tickers: list[str], start: str = "2025-06-01") -> dict[str, pd.DataFrame]:
    """LLV core+extend 에서 OHLC → {col: wide DataFrame}. panel 의 YZ_20 도 함께."""
    frames = []
    for f in ("core.parquet", "extend.parquet"):
        d = pd.read_parquet(LLV / "data" / "ohlcv" / f,
                            columns=["Date", "Ticker", "Open", "High", "Low", "Close", "YZ_20"])
        frames.append(d[d.Ticker.isin(tickers)])
    d = pd.concat(frames).drop_duplicates(["Date", "Ticker"])
    d = d[d.Date >= pd.Timestamp(start)]
    out = {}
    for c in ("Open", "High", "Low", "Close", "YZ_20"):
        out[c.lower()] = d.pivot(index="Date", columns="Ticker", values=c).sort_index()
    missing = [t for t in tickers if t not in out["close"].columns]
    if missing:
        print(f"⚠ LLV 에 없는 종목 {missing} — 그림자에서 제외")
    return out


def vol_scale_panel(yz: pd.DataFrame, elig: pd.DataFrame) -> pd.DataFrame:
    """LLV vol_scale.json 정의 재현: 종목 YZ_20 ÷ (유니버스 일별 중앙값의 직전 252일 중앙값, t−1까지)."""
    pool_med = yz.where(elig.reindex_like(yz).fillna(False)).median(axis=1)
    denom = pool_med.rolling(252, min_periods=126).median().shift(1)
    return yz.div(denom, axis=0).clip(0.2, 5.0)


# ── 상태 ────────────────────────────────────────────────────────────────────
def init_state(model: str) -> dict:
    """실제 가상계좌 state.json 에서 분기."""
    m = MODELS[model]
    state_f, equity_f = files(model)
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
        "model": f"{m['label']} (shadow)",
        "forked_from": "StockPortfolio/data/paper/state.json",
        "forked_at": datetime.now().isoformat(timespec="seconds"),
        "fork_note": f"실계정 updated_at={sp.get('updated_at')} 기준 분기 (사이징 {m['max_pos']}×{m['slot_krw']:,.0f})",
        "cash": float(sp.get("cash", 0.0)),
        "pending": [{"date": str(x.get("available_on") or x.get("date")), "amount": float(x.get("amount", 0.0))}
                    for x in pending],
        "positions": positions,
        "last_sell_date": dict(sp.get("last_sell_date") or {}),
        "pending_buys": [],
        "last_processed": sp.get("last_signal_date") or datetime.now().strftime("%Y-%m-%d"),
    }
    state_f.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    if not equity_f.exists():
        equity_f.write_text("date,cash,pending,mv,total,n_pos,note\n", encoding="utf-8")
    print(f"[{m['label']}] 분기 완료 — 현금 {st['cash']:,.0f}원 · 보유 {len(positions)}종목 · "
          f"기준일 {st['last_processed']}")
    return st


def stop_line(pos: dict, d_plus: int, m: dict, scale: float) -> float:
    avg, peak = pos["avg_price"], pos["peak"]
    above = peak > avg
    late = d_plus > m["stop_switch"]
    if not late:
        base, clip = m["stop_early"], m["clip_early"]
    elif above:
        base, clip = m["stop_late_up"], m["clip_late_up"]
    else:
        base, clip = m["stop_late_flat"], m["clip_late_flat"]
    pct = min(max(base * scale, clip[0]), clip[1]) if m["vol_linked"] else base
    return (peak if above else avg) * (1 - pct)


def used_count(st: dict, m: dict) -> int:
    if m["count_by"] == "stocks":
        return len(st["positions"])
    return sum(p["slots"] for p in st["positions"].values())


def advance(model: str, st: dict) -> dict:
    m = MODELS[model]
    state_f, equity_f = files(model)
    tickers, names = all_universe_tickers(model)
    held = [t for t in st["positions"] if t not in tickers]
    px = load_prices(sorted(set(tickers) | set(st["positions"])))
    cl, op, hi, lo, yz = px["close"], px["open"], px["high"], px["low"], px["yz_20"]
    dates = cl.index

    # 일별 유니버스 편입 (monthly 모델은 달마다 다른 파일)
    elig = pd.DataFrame(False, index=dates, columns=cl.columns)
    for d in dates:
        doc = universe_for(model, d)
        cols = [s["ticker"] for s in doc["stocks"] if s["ticker"] in elig.columns]
        elig.loc[d, cols] = True

    high_n = cl.rolling(m["high_window"], min_periods=m["high_min_p"]).max()
    ma = cl.rolling(m["trend_ma"], min_periods=m["trend_min_p"]).mean()
    depth = (high_n - cl) / high_n
    cond = ((cl <= m["pullback"] * high_n) & (cl > ma) & elig).fillna(False)
    onset = cond & ~cond.shift(1).fillna(False)
    scale_p = vol_scale_panel(yz, elig) if m["vol_linked"] else None

    last = pd.Timestamp(st["last_processed"])
    todo = [d for d in dates if d > last]
    if not todo:
        print(f"[{m['label']}] 처리할 새 거래일 없음 (마지막 {st['last_processed']})")
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
            if tk not in cl.columns or pd.isna(lo.loc[d, tk]):
                continue
            e_i = didx.get(pd.Timestamp(pos["entry_date"]))
            d_plus = (didx[d] - e_i) if e_i is not None else 99
            sc = 1.0
            if scale_p is not None:
                v = scale_p.loc[d, tk] if tk in scale_p.columns else np.nan
                sc = float(v) if not pd.isna(v) else 1.0
            line = stop_line(pos, d_plus, m, sc)
            if lo.loc[d, tk] <= line:
                fill = min(op.loc[d, tk], line) if not pd.isna(op.loc[d, tk]) else line
                fill *= STOP_FILL
                net = fill * pos["shares"] * (1 - SELL_FEE - SELL_TAX)
                st["pending"].append({"date": ds, "amount": net})
                st["last_sell_date"][tk] = ds
                del st["positions"][tk]

        # 3) 매수 집행 (전일 신호 → 오늘 시가)
        for q in st["pending_buys"]:
            tk = q["ticker"]
            if used_count(st, m) >= m["max_pos"] and tk not in st["positions"]:
                continue
            if tk not in cl.columns or pd.isna(op.loc[d, tk]):
                continue
            pos = st["positions"].get(tk)
            if pos and pos["slots"] >= m["max_slots"]:
                continue
            if m["count_by"] == "slots" and used_count(st, m) >= m["max_pos"]:
                continue
            price = float(op.loc[d, tk])
            shares = int(m["slot_krw"] // price)
            if shares == 0 and ALLOW_SINGLE_SHARE:
                shares = 1
            cost = price * shares * (1 + BUY_FEE)
            if shares <= 0 or cost > st["cash"]:
                continue
            st["cash"] -= cost
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
        sig = [(float(depth.loc[d, tk]), tk) for tk in cl.columns if bool(onset.loc[d, tk])]
        sig.sort(reverse=True)
        for dep, tk in sig:
            if st["last_sell_date"].get(tk) == ds:            # 쿨다운 1거래일
                continue
            pos = st["positions"].get(tk)
            if pos and pos["slots"] >= m["max_slots"]:
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

    state_f.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.DataFrame(rows).to_csv(equity_f, mode="a", header=not equity_f.exists(), index=False)
    for r in rows:
        print(f"[{m['label']}] {r['date']}  총자산 {r['total']:>14,.0f}원  보유 {r['n_pos']}종목")
    return st


def report():
    real = pd.read_csv(SP_PAPER / "equity.csv")[["date", "total"]].rename(columns={"total": "실계정"})
    m = real
    for model in MODELS:
        _, eq = files(model)
        if eq.exists():
            sh = pd.read_csv(eq)[["date", "total"]].rename(columns={"total": MODELS[model]["label"]})
            m = m.merge(sh, on="date", how="left")
    if not len(m):
        print("겹치는 날짜 없음"); return
    cols = [c for c in m.columns if c != "date"]
    print("날짜        " + "".join(f"{c:>16s}" for c in cols) + "   (실계정=v1.2.3.4, 각 열은 분기일=100 기준 지수)")
    base = {c: m[c].dropna().iloc[0] if m[c].notna().any() else np.nan for c in cols}
    for _, r in m.tail(30).iterrows():
        print(f"{r['date']:10s} " + "".join(f"{(r[c]/base[c]*100 if pd.notna(r[c]) else float('nan')):>15.2f} " for c in cols))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true", help="가상계좌 state 에서 분기 (--model 필수)")
    ap.add_argument("--model", choices=list(MODELS), help="대상 모델 (생략 시 전체)")
    ap.add_argument("--report", action="store_true", help="실계정과 비교 출력")
    a = ap.parse_args()
    if a.init:
        if not a.model:
            raise SystemExit("--init 은 --model 을 지정해야 합니다")
        init_state(a.model)
    elif a.report:
        report()
    else:
        for model in ([a.model] if a.model else list(MODELS)):
            state_f, _ = files(model)
            if not state_f.exists():
                print(f"[{MODELS[model]['label']}] 상태 파일 없음 — 먼저 --init --model {model} 을 실행하세요.")
                continue
            advance(model, json.loads(state_f.read_text(encoding="utf-8")))
