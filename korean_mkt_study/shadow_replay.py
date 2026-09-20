# -*- coding: utf-8 -*-
"""shadow_replay.py — 스윙 포트 규칙 재생: v1.1.1.2 vs v1.2.2.3 (Kane 지시 2026-09-20).

shadow_track.py 의 엔진(init_state · advance · stop_line)을 **수정 없이** 가져다, 같은 출발 상태에서
여러 규칙 변형을 하루씩 전진시키며 거래를 기록한다. 운영 파일(data/shadow_*)은 건드리지 않는다.

출발 상태  out/shadow_replay/fork/state_20260817.json
          = 미니 StockPortfolio/data/paper/backups/state_20260817.json
            (8/17 대체공휴일, last_signal 2026-08-14 — 라이브 v1.1.1.2 그림자가 분기한 실계정 8/14 상태)
변형      v1112 · v1223 × 사이징 S20(20×200만, 슬롯 카운트) / S10(10×400만, 종목 카운트)
          v1112 에서 한 자리씩: +U(유니버스 4조/30% 월별) · +E(40일 고점·MA200) · +X(변동배율 + 평단후기 −10%)
          live_repro — v1112 S20 인데 2026-09-11 까지 손절 없음. 0ad233f 판 advance 의 손절 루프 첫 조건
                       `tk not in didx`(didx 키는 날짜, tk 는 종목코드 → 항상 참)를 재현하고 9/14 부터 정상.
                       라이브 shadow_v1112_equity.csv 와 일치하면 그 결함이 라이브 기록의 원인임이 확인된다.
하루씩 전진  load_prices 를 날짜 컷오프 버전으로 바꿔 advance 가 하루만 처리하게 한다 (지표는 전부 과거값만
          쓰므로 컷오프해도 그날 값이 같다). 전후 상태를 비교해 매수·매도·신호 큐를 기록한다.
실행      python3 shadow_replay.py [--end 2026-09-18]
산출      out/shadow_replay/{equity.csv, trades.csv, queue.csv, summary.md, equity.png}
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import shadow_track as T

KMS = Path(__file__).resolve().parent
OUT = KMS / "out" / "shadow_replay"
FORK = OUT / "fork" / "state_20260817.json"
LIVE_V1112 = OUT / "fork" / "live_shadow_v1112_equity.csv"
REAL_EQ = OUT / "fork" / "real_equity.csv"
BUG_UNTIL = pd.Timestamp("2026-09-11")          # 0ad233f 판이 돈 마지막 날

S20 = dict(slot_krw=2_000_000.0, max_pos=20, count_by="slots")
S10 = dict(slot_krw=4_000_000.0, max_pos=10, count_by="stocks")
COMP = {
    "U": dict(universe="monthly"),
    "E": dict(high_window=40, high_min_p=20, trend_ma=200, trend_min_p=140),
    "X": dict(vol_linked=True, stop_late_flat=0.10),
}


def build_variants() -> dict:
    b1, b2 = T.MODELS["v1112"], T.MODELS["v1223"]
    # 두 모델의 차이가 정확히 U·E·X·사이징 네 묶음인지 확인 (label 제외)
    full = {**b1, **COMP["U"], **COMP["E"], **COMP["X"], **S10}
    diff = {k for k in b2 if k != "label" and full.get(k) != b2[k]}
    assert not diff, f"v1112+U+E+X+S10 ≠ v1223: {diff}"
    V = {}
    for sn, sz in (("S20", S20), ("S10", S10)):
        V[f"v1112_{sn}"] = {**b1, **sz, "label": f"v1.1.1.2 {sn}"}
        V[f"v1223_{sn}"] = {**b2, **sz, "label": f"v1.2.2.3 {sn}"}
        for cn, comp in COMP.items():
            V[f"v1112+{cn}_{sn}"] = {**b1, **sz, **comp, "label": f"v1.1.1.2+{cn} {sn}"}
    V["live_repro"] = {**b1, "label": "v1.1.1.2 라이브 재현(9/11까지 손절 없음)"}
    return V


# ── shadow_track 패치 (모듈 전역만 교체, 로직은 그대로) ─────────────────────────
_LLV: pd.DataFrame | None = None
CUTOFF = pd.Timestamp("2100-01-01")
NOSTOP = {"on": False}
_orig_stop_line = T.stop_line


def _llv() -> pd.DataFrame:
    global _LLV
    if _LLV is None:
        frames = [pd.read_parquet(T.LLV / "data" / "ohlcv" / f,
                                  columns=["Date", "Ticker", "Name", "Open", "High", "Low", "Close", "YZ_20"])
                  for f in ("core.parquet", "extend.parquet")]
        d = pd.concat(frames).drop_duplicates(["Date", "Ticker"])
        d["Date"] = pd.to_datetime(d["Date"])
        _LLV = d[d.Date >= pd.Timestamp("2025-06-01")].reset_index(drop=True)
    return _LLV


def load_prices_cut(tickers, start: str = "2025-06-01"):
    d = _llv()
    d = d[d.Ticker.isin(tickers) & (d.Date >= pd.Timestamp(start)) & (d.Date <= CUTOFF)]
    return {c.lower(): d.pivot(index="Date", columns="Ticker", values=c).sort_index()
            for c in ("Open", "High", "Low", "Close", "YZ_20")}


def stop_line_wrap(pos, d_plus, m, scale):
    return -np.inf if NOSTOP["on"] else _orig_stop_line(pos, d_plus, m, scale)


def patch(variants: dict, fork_dir: Path) -> None:
    docs = T._universe_docs()
    T._universe_docs = lambda: docs
    T.MODELS = variants
    T.load_prices = load_prices_cut
    T.stop_line = stop_line_wrap
    T.SP_PAPER = fork_dir
    (OUT / "state").mkdir(parents=True, exist_ok=True)
    T.files = lambda model: (OUT / "state" / f"{model}_state.json", OUT / "state" / f"{model}_equity.csv")


# ── 재생 ────────────────────────────────────────────────────────────────────
def _d_plus(td_idx: dict, d: pd.Timestamp, entry: str) -> int:
    e = td_idx.get(pd.Timestamp(entry))
    return td_idx[d] - e if e is not None else 99


def run_variant(key: str, days: list, td_idx: dict, seed_prev: pd.Timestamp) -> tuple[pd.DataFrame, list, list]:
    """days[0] = 출발일(8/14) — 시드 처리: 손절 없음·매수 없음(큐 비어 있음), 그날 신호만 큐에 쌓고 peak·평가 갱신.
    라이브 v1.1.1.2 는 8/18 첫날 8/14 신호(NC 불타기·SK하이닉스·삼성SDI)를 샀다 — 분기 시 8/14 큐가
    들어가 있었다는 뜻이라 모든 변형에 같은 방식(각자 규칙의 8/14 신호)으로 맞춘다."""
    global CUTOFF
    m = T.MODELS[key]
    state_f, eq_f = T.files(key)
    for f in (state_f, eq_f):
        f.unlink(missing_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        st = T.init_state(key)
    st["last_processed"] = seed_prev.strftime("%Y-%m-%d")
    trades, queue = [], []
    for d in days:
        CUTOFF = d
        NOSTOP["on"] = d == days[0] or (key == "live_repro" and d <= BUG_UNTIL)
        ds = d.strftime("%Y-%m-%d")
        before = copy.deepcopy(st)
        with contextlib.redirect_stdout(io.StringIO()):
            st = T.advance(key, st)

        # 매도: 오늘 last_sell_date 가 찍힌 종목 ↔ 오늘 날짜로 새로 쌓인 pending (같은 순서로 append 됨)
        sold = [tk for tk in before["positions"]
                if st["last_sell_date"].get(tk) == ds and before["last_sell_date"].get(tk) != ds]
        new_p = [p for p in st["pending"] if p["date"] == ds]
        assert len(sold) == len(new_p), (key, ds, sold, new_p)
        for tk, p in zip(sold, new_p):
            pos = before["positions"][tk]
            sh, net = pos["shares"], p["amount"]
            fill = net / (sh * (1 - T.SELL_FEE - T.SELL_TAX))
            dp = _d_plus(td_idx, d, pos["entry_date"])
            regime = ("early" if dp <= m["stop_switch"]
                      else "late_up" if pos["peak"] > pos["avg_price"] else "late_flat")
            cost = pos["avg_price"] * sh * (1 + T.BUY_FEE)
            trades.append(dict(model=key, date=ds, ticker=tk, name=pos["name"], side="SELL", kind=regime,
                               shares=sh, price=round(fill, 1), avg=round(pos["avg_price"], 1),
                               peak=pos["peak"], entry=pos["entry_date"], d_plus=dp,
                               pnl=round(net - cost), ret=net / cost - 1))
        # 매수: 새 종목 또는 주식 수 증가(불타기)
        bought = set()
        for tk, pos in st["positions"].items():
            b = None if tk in sold else before["positions"].get(tk)
            if b is None:
                kind, add, price = "new", pos["shares"], pos["avg_price"]
            elif pos["shares"] > b["shares"]:
                add = pos["shares"] - b["shares"]
                price = (pos["avg_price"] * pos["shares"] - b["avg_price"] * b["shares"]) / add
                kind = "pyramid"
            else:
                continue
            bought.add(tk)
            trades.append(dict(model=key, date=ds, ticker=tk, name=pos["name"], side="BUY", kind=kind,
                               shares=add, price=round(price, 1), avg=round(pos["avg_price"], 1),
                               peak=np.nan, entry=pos["entry_date"], d_plus=0, pnl=np.nan, ret=np.nan))
        for q in before["pending_buys"]:
            queue.append(dict(model=key, date=ds, ticker=q["ticker"], queued_on=q["queued_on"],
                              status="bought" if q["ticker"] in bought else "skipped"))
        for q in st["pending_buys"]:
            queue.append(dict(model=key, date=ds, ticker=q["ticker"], queued_on=ds, status="queued"))

    eq = pd.read_csv(eq_f)
    eq = eq[eq.date != days[0].strftime("%Y-%m-%d")]      # 시드일 평가행은 기준값(base) 확인용으로만
    eq.insert(0, "model", key)
    return eq, trades, queue


# ── 요약 ────────────────────────────────────────────────────────────────────
def fork_base(fork: dict, d0: pd.Timestamp) -> float:
    """출발일(8/14) 종가 기준 총자산 — 모든 변형의 공통 기준(=100)."""
    c = _llv()
    c = c[c.Date == d0].set_index("Ticker").Close
    pend = fork.get("pending_settlements") or []
    pend = sum(float(x.get("amount", 0)) for x in pend) if isinstance(pend, list) else sum(pend.values())
    return float(fork["cash"]) + pend + sum(float(c[tk]) * int(p["shares"]) for tk, p in fork["positions"].items())


def stats(key: str, eq: pd.DataFrame, tr: pd.DataFrame, base: float) -> dict:
    tot = np.r_[base, eq.total.to_numpy()]
    s = tr[tr.side == "SELL"]
    b = tr[tr.side == "BUY"]
    return dict(model=key, final=tot[-1], ret=tot[-1] / base - 1, mdd=(tot / np.maximum.accumulate(tot) - 1).min(),
                buys_new=int((b.kind == "new").sum()), buys_pyr=int((b.kind == "pyramid").sum()),
                sells=len(s), early=int((s.kind == "early").sum()), late_up=int((s.kind == "late_up").sum()),
                late_flat=int((s.kind == "late_flat").sum()), win=(s.ret > 0).mean() if len(s) else np.nan,
                realized=s.pnl.sum(), n_pos_end=int(eq.n_pos.iloc[-1]), cash_end=eq.cash.iloc[-1] + eq.pending.iloc[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default="2026-09-18")
    a = ap.parse_args()
    fork = json.loads(FORK.read_text(encoding="utf-8"))
    d0 = pd.Timestamp(fork["last_signal_date"])
    fork_dir = OUT / "fork" / "_as_state"
    fork_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(FORK, fork_dir / "state.json")
    V = build_variants()
    patch(V, fork_dir)

    td = sorted(_llv().Date.unique())
    td_idx = {pd.Timestamp(d): i for i, d in enumerate(td)}
    days = [pd.Timestamp(d) for d in td if d0 <= d <= pd.Timestamp(a.end)]
    seed_prev = pd.Timestamp(td[td_idx[d0] - 1])
    base = fork_base(fork, d0)
    print(f"출발 {d0.date()} 총자산(LLV 종가) {base:,.0f}원 · 시드 {days[0].date()} · 재생 {days[1].date()}~{days[-1].date()} {len(days) - 1}거래일")

    eqs, trs, qs = [], [], []
    for key in V:
        eq, tr, q = run_variant(key, days, td_idx, seed_prev)
        eqs.append(eq); trs += tr; qs += q
        print(f"  {V[key]['label']:<40s} 최종 {eq.total.iloc[-1]:>14,.0f}원  거래 {len(tr)}")
    eq = pd.concat(eqs, ignore_index=True)
    tr = pd.DataFrame(trs)
    q = pd.DataFrame(qs)
    eq.to_csv(OUT / "equity.csv", index=False)
    tr.to_csv(OUT / "trades.csv", index=False)
    q.to_csv(OUT / "queue.csv", index=False)
    summ = pd.DataFrame([stats(k, eq[eq.model == k], tr[tr.model == k], base) for k in V])
    summ.to_csv(OUT / "summary.csv", index=False)
    write_md(V, eq, tr, summ, base, d0)
    plot(V, eq, base, d0)


def write_md(V, eq, tr, summ, base, d0) -> None:
    o = [f"# 스윙 포트 규칙 재생 — v1.1.1.2 vs v1.2.2.3", "",
         f"출발 상태 `{FORK.name}` (last_signal {d0.date()}) · 출발 총자산(LLV 종가) {base:,.0f}원 · "
         f"엔진 = shadow_track.py (a3c5061) 그대로", ""]
    # 1) 라이브 재현
    live = pd.read_csv(LIVE_V1112)[["date", "total"]].rename(columns={"total": "live"})
    rep = eq[eq.model == "live_repro"][["date", "total"]].rename(columns={"total": "repro"})
    fix = eq[eq.model == "v1112_S20"][["date", "total"]].rename(columns={"total": "fixed"})
    j = live.merge(rep, on="date", how="outer").merge(fix, on="date", how="outer")
    j["gap"] = j.repro - j.live
    o += ["## 1. 라이브 v1.1.1.2 재현 (9/11까지 손절 없음 → 9/14부터 정상)", "",
          f"최대 |repro−live| = {j.gap.abs().max():,.0f}원", "",
          "| 날짜 | 라이브 | 재현 | 차이 | 정상 엔진(손절 있음) |", "|---|---:|---:|---:|---:|"]
    o += [f"| {r.date} | {r.live:,.0f} | {r.repro:,.0f} | {r.gap:+,.0f} | {r.fixed:,.0f} |" for r in j.itertuples()]
    # 2) 실계정(운영 v1.2.2.3, 8/18~9/11) vs 재생 v1223 S20
    real = pd.read_csv(REAL_EQ)[["date", "total"]].rename(columns={"total": "real"})
    r23 = eq[eq.model == "v1223_S20"][["date", "total"]].rename(columns={"total": "replay"})
    k = real.merge(r23, on="date")
    k = k[k.date <= "2026-09-11"]
    o += ["", "## 2. 엔진 대조 — 실계정(운영 v1.2.2.3, 20×200만) vs 재생 v1.2.2.3 S20 (8/18~9/11)", "",
          "| 날짜 | 실계정 | 재생 | 재생−실계정 |", "|---|---:|---:|---:|"]
    o += [f"| {r.date} | {r.real:,.0f} | {r.replay:,.0f} | {r.replay - r.real:+,.0f} |" for r in k.itertuples()]
    # 3) 요약
    o += ["", "## 3. 변형별 요약 (출발=100)", "",
          "| 변형 | 최종 | 수익률 | MDD | 신규 | 불타기 | 매도 | early | late_up | late_flat | 승률 | 실현손익 | 말일 보유 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summ.itertuples():
        o.append(f"| {V[r.model]['label']} | {r.final:,.0f} | {r.ret:+.1%} | {r.mdd:.1%} | {r.buys_new} | {r.buys_pyr} | "
                 f"{r.sells} | {r.early} | {r.late_up} | {r.late_flat} | {r.win:.0%} | {r.realized:+,.0f} | {r.n_pos_end} |")
    # 4) 자리별 분해
    fin = summ.set_index("model").final
    o += ["", "## 4. 자리별 분해 (v1.1.1.2 대비 최종 총자산 차이, 원)", "",
          "| 사이징 | v1.2.2.3 전체 | +U 유니버스 | +E 진입 | +X 청산 | 상호작용(잔차) |", "|---|---:|---:|---:|---:|---:|"]
    for sn in ("S20", "S10"):
        tot = fin[f"v1223_{sn}"] - fin[f"v1112_{sn}"]
        parts = [fin[f"v1112+{c}_{sn}"] - fin[f"v1112_{sn}"] for c in "UEX"]
        o.append(f"| {sn} | {tot:+,.0f} | " + " | ".join(f"{p:+,.0f}" for p in parts) + f" | {tot - sum(parts):+,.0f} |")
    (OUT / "summary.md").write_text("\n".join(o) + "\n", encoding="utf-8")
    print("\n".join(o))


def plot(V, eq, base, d0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "AppleGothic"; plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for key, sty in (("v1112_S20", "-"), ("v1223_S20", "-"), ("v1112_S10", "--"), ("v1223_S10", "--"), ("live_repro", ":")):
        s = eq[eq.model == key]
        x = [d0] + list(pd.to_datetime(s.date)); y = [100] + list(s.total / base * 100)
        ax.plot(x, y, sty, lw=1.8 if "S20" in key else 1.2, label=V[key]["label"])
    real = pd.read_csv(REAL_EQ)
    real = real[(real.date >= d0.strftime("%Y-%m-%d")) & (real.date <= "2026-09-11")]
    ax.plot(pd.to_datetime(real.date), real.total / real.total.iloc[0] * 100, color="#444", lw=1.2, label="실계정 (운영 v1.2.2.3)")
    ax.axhline(100, color="#999", lw=0.6); ax.grid(alpha=0.3); ax.legend(fontsize=9, frameon=False)
    ax.set_title(f"스윙 포트 규칙 재생 — 출발 {d0.date()} 총자산=100", loc="left")
    fig.savefig(OUT / "equity.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
