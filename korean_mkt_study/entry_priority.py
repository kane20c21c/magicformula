# -*- coding: utf-8 -*-
"""entry_priority.py — 작업계획 1단계: 진입 우선순위·현금 진단 (케인 2026-11-05).

질문  ① 체결된 신호 vs 놓친 신호 (놓친 이유별) — 포트 평균이 신호 전체보다 낮은 이유
      ② 같은 날 후보 우선순위: 눌림 깊은 순(현행) / 얕은 순 / 모멘텀 20·60 / 거래대금 / 저·고변동 / 무작위(MC)
      ③ 슬롯: 1,000만 고정 ×10·15·20 / 자산÷10·15·20
조건  버팀 + T40·T60 (stop_port.run_port 그대로). 보류 구간 2026-06-01~ 은 규칙 선택에서 제외 → END=2026-05-29 기준
      (판정 수치는 2015-01 ~ 2026-05 진입분. 2026-06~ 확인은 마지막 단계)
사용  STOP_UNIV=swing|day python3 entry_priority.py [mc회수=50]
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import stop_research as sr, stop_port as sp

TRAIN_END = "2026-05-29"
OUT = HERE / "out" / "entry" / sr.UNIV

def fwd(p, s, T):
    """신호(t) → t+1 시가 매수 → t+1+T+1 시가 매도 (포트 시간 청산과 같은 시점), 비용 차감 %."""
    op = p.open.to_numpy(float); D = op.shape[0]; H = T + 1
    m = s.reindex(index=p.dates, columns=p.tickers).fillna(False).to_numpy(bool)
    te = int(np.searchsorted(p.dates, pd.Timestamp(TRAIN_END), side="right")) - 1   # 매도일도 학습 구간 안 (보류 구간 가격 미사용)
    ti, tj = np.nonzero(m); f = ti + 1; ok = (f + H < D) & (f + H <= te)
    ti, tj, f = ti[ok], tj[ok], f[ok]
    r = (op[f + H, tj] / op[f, tj] * (1 - sp.SELL_FEE - sp.TAX) / (1 + sp.BUY_FEE) - 1) * 100
    return pd.DataFrame(dict(date=p.dates[ti], ticker=np.array(p.tickers)[tj], ret=r)).dropna()

def dstat(r):
    return pd.Series({"건수": len(r), "평균%": r.mean(), "중앙%": r.median(), "플러스%": (r > 0).mean() * 100,
                      "하위10%": r.quantile(.1), "상위10%": r.quantile(.9), "상위5%": r.quantile(.95), "≥+20%": (r >= 20).mean() * 100})

def prios(p):
    c = p.close; hi40 = c.rolling(40, min_periods=20).max(); depth = (hi40 - c) / hi40
    return {"눌림 깊은 순 (현행)": None, "눌림 얕은 순": -depth, "모멘텀20 높은 순": c / c.shift(20) - 1,
            "모멘텀60 높은 순": c / c.shift(60) - 1, "거래대금20 큰 순": (c * p.volume).rolling(20, min_periods=10).mean(),
            "저변동 순 (YZ20 낮은)": -p.yz20, "고변동 순 (YZ20 높은)": p.yz20}

def row(eq, tr, extra=None):
    m = sp.metrics(eq, tr); se = tr[tr.side == "SELL"].ret * 100
    d = {"CAGR%": m["cagr"] * 100, "Sharpe": m["sharpe"], "MDD%": m["mdd"] * 100, "투자비중%": m["invested"] * 100,
         "청산수": m["n_trade"], "승률%": m["win"] * 100, "거래평균%": se.mean(), "거래상위5%": se.quantile(.95)}
    if extra: d.update(extra)
    return d

def main(n_mc=50, seed=20261105):
    p, s, _, _, atr = sr.load(); OUT.mkdir(parents=True, exist_ok=True); pd.set_option("display.width", 260)
    s = s.loc[:TRAIN_END]; P = prios(p)
    print(f"[{sr.UNIV}] 종목 {len(p.tickers)} · 신호 {int(s.to_numpy().sum())} · 학습 구간 2015-01 ~ {TRAIN_END} 진입")
    for T in (40, 60):
        kw = dict(T=T, end=TRAIN_END)
        # ① 체결 vs 놓침
        F = fwd(p, s, T); L = []
        eq, tr = sp.run_port(p, s, atr, None, log=L, **kw)
        lg = pd.DataFrame(L, columns=["date", "ticker", "상태"]).drop_duplicates(["date", "ticker"])
        J = F.merge(lg, on=["date", "ticker"], how="left").fillna({"상태": "기간밖"})
        a = J.groupby("상태").ret.apply(dstat).unstack(); a.loc["신호 전체"] = dstat(J.ret)
        print(f"\n== [T{T}] ① 신호별 T{T} 수익률 — 운영 조건(1,000만×10, 눌림 깊은 순)에서 체결/놓침 ==\n", a.round(1).to_string())
        a.to_csv(OUT / f"p1_filled_vs_missed_T{T}.csv")
        # ② 우선순위 (운영 조건 고정 슬롯 · 자산÷10 각각) + 무작위 MC
        rows = []
        for mode in ("fixed", "prop"):
            rnd = [sp.metrics(*sp.run_port(p, s, atr, None, slot_mode=mode, rng=np.random.default_rng(seed + i), **kw))["cagr"] * 100 for i in range(n_mc)]
            rnd = np.array(rnd)
            for nm, pr in P.items():
                eq, tr = sp.run_port(p, s, atr, None, slot_mode=mode, prio=pr, **kw); r = row(eq, tr)
                r.update({"슬롯": "1,000만×10" if mode == "fixed" else "자산÷10", "우선순위": nm, "무작위 대비 백분위": (rnd < r["CAGR%"]).mean() * 100})
                rows.append(r)
            rows.append({"슬롯": "1,000만×10" if mode == "fixed" else "자산÷10", "우선순위": f"무작위 MC{n_mc} 중앙 (10~90%)",
                         "CAGR%": np.median(rnd), "Sharpe": np.nan, "MDD%": np.nan, "무작위 대비 백분위": 50.0,
                         "투자비중%": np.nan, "거래평균%": np.nan, "범위": f"{np.quantile(rnd,.1):.1f}~{np.quantile(rnd,.9):.1f}"})
        b = pd.DataFrame(rows).set_index(["슬롯", "우선순위"])
        print(f"\n== [T{T}] ② 같은 날 후보 우선순위 ==\n", b.round(2).to_string()); b.to_csv(OUT / f"p2_priority_T{T}.csv")
        # ③ 슬롯 크기·개수 (현행 우선순위 + 무작위 MC 중앙)
        rows = []
        for mode, ns in [("fixed", 10), ("fixed", 15), ("fixed", 20), ("prop", 10), ("prop", 15), ("prop", 20)]:
            eq, tr = sp.run_port(p, s, atr, None, slot_mode=mode, nslot=ns, **kw); r = row(eq, tr)
            mc = [sp.metrics(*sp.run_port(p, s, atr, None, slot_mode=mode, nslot=ns, rng=np.random.default_rng(seed + i), **kw)) for i in range(n_mc)]
            r.update({"슬롯": ("1,000만×" if mode == "fixed" else "자산÷") + str(ns), "MC CAGR중앙%": np.median([m["cagr"] for m in mc]) * 100,
                      "MC MDD중앙%": np.median([m["mdd"] for m in mc]) * 100, "MC 투자비중%": np.median([m["invested"] for m in mc]) * 100})
            rows.append(r)
        c = pd.DataFrame(rows).set_index("슬롯")
        print(f"\n== [T{T}] ③ 슬롯 (단일 경로 = 눌림 깊은 순 · MC = 무작위 {n_mc}회) ==\n", c.round(2).to_string()); c.to_csv(OUT / f"p3_slots_T{T}.csv")

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
