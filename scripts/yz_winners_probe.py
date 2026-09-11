#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yz_winners_probe.py — 대박의 반복성 · 변동성 vs 수익성의 설명력
================================================================
케인 질문 2026-09-11 (둘)

  ① "같은 종목이 여러 분기에 걸쳐 대박을 내기는 쉽지 않을 듯. 찾아봐, 나는 별로
     기대 없음."
  ② "전분기의 변동성과 수익성 중에서 다음 분기를 1%라도 더 설명하는 것은 변동성?"

전제 (앞 작업과 동일)
  · 유니버스에서 102110(BM ETF) · 005930 삼성전자 · 000660 SK하이닉스 제외
  · 분기 전이 12쌍 (2023Q4~2026Q3) — 252000 상장일(2023-08-14) 제약
  · 변동성 = 그 분기 데이터만으로 계산한 Yang-Zhang σ

⚠ ① 을 볼 때 반드시 통제할 것 — **'대박의 반복' 과 '변동성 그룹의 지속' 은 다르다.**
  고변동 종목은 어느 분기에나 상위 5% 에 들 확률이 높다(산포). 그래서 통제 없이
  "대박이 반복된다" 고 세면 사실은 "고변동 종목이 계속 고변동" 을 본 것일 수 있다.
  → 변동성 그룹 **안에서** 다시 센다.

실행
  python3 scripts/yz_winners_probe.py
"""
from __future__ import annotations

import sys
import warnings
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import yz_group as G                                     # noqa: E402
import yz_group_quarterly as Q                           # noqa: E402

DROP = ["102110", "005930", "000660"]
Q0 = pd.Period("2023Q4")
TOP = 0.05                     # '대박' 정의 — 그 분기 상위 5%


def _binom_p(k: int, n: int) -> float:
    if n == 0:
        return np.nan
    pmf = [comb(n, i) / 2 ** n for i in range(n + 1)]
    return float(sum(p for p in pmf if p <= pmf[k] + 1e-12))


def prep(df: pd.DataFrame):
    piv = lambda c: df.pivot_table(index="Date", columns="Ticker",
                                   values=c, aggfunc="last").astype(float)
    O, H, L, C = (piv(c) for c in ("Open", "High", "Low", "Close"))
    Cx, Ox, Hx, Lx = (t.drop(columns=DROP, errors="ignore") for t in (C, O, H, L))
    o, cc = np.log(Ox / Cx.shift(1)), np.log(Cx / Ox)
    u, dn = np.log(Hx / Ox), np.log(Lx / Ox)
    rs = u * (u - cc) + dn * (dn - cc)
    qs = sorted(set(C.index.to_period("Q")))

    def qret(q):
        s = C.index[C.index.to_period("Q") == q]
        return Cx.loc[s].apply(lambda c_: (c_.dropna().iloc[-1] / c_.dropna().iloc[0] - 1)
                               if c_.notna().sum() > 15 else np.nan)

    out = []
    for a, b in zip(qs, qs[1:]):
        if b < Q0:
            continue
        s = C.index[C.index.to_period("Q") == a]
        n = len(s)
        if n < Q.MIN_Q_DAYS:
            continue
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        sig = np.sqrt(o.loc[s].var(ddof=1) + k * cc.loc[s].var(ddof=1)
                      + (1 - k) * rs.loc[s].mean())
        ra, rb = qret(a), qret(b)
        ok = (sig.notna() & ra.notna() & rb.notna()
              & (Cx.loc[s].notna().sum() >= n * Q.MIN_COVER))
        out.append((str(b), sig[ok], ra[ok], rb[ok]))
    return out, Cx


def explain(P) -> pd.DataFrame:
    """② 변동성 vs 수익성 — 다음 분기의 '수준' 과 '크기' 각각에 대해."""
    rows = []
    for qb, sig, ra, rb in P:
        rows.append({
            "분기": qb, "종목": len(rb),
            "변동성→수익": sig.rank().corr(rb.rank()),
            "수익성→수익": ra.rank().corr(rb.rank()),
            "변동성→|수익|": sig.rank().corr(rb.abs().rank()),
            "수익성→|수익|": ra.rank().corr(rb.abs().rank()),
        })
    T = pd.DataFrame(rows)
    summ = []
    for c in ["변동성→수익", "수익성→수익", "변동성→|수익|", "수익성→|수익|"]:
        w = int((T[c] > 0).sum())
        summ.append({"관계": c, "평균 ρ": float(T[c].mean()),
                     "중앙 ρ": float(T[c].median()),
                     "양수 분기": f"{w}/{len(T)}", "부호검정 p": _binom_p(w, len(T))})
    return T, pd.DataFrame(summ)


def repeat(P) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """① 대박(상위 5%)의 반복성 — 통제 전/후."""
    rec = []
    for qb, sig, ra, rb in P:
        gv = Q._t3(sig, Q.VOL3)
        ha = ra.rank(pct=True, ascending=False) <= TOP
        hb = rb.rank(pct=True, ascending=False) <= TOP
        for t_ in rb.index:
            rec.append({"분기": qb, "티커": t_, "변동성군": gv[t_],
                        "전분기 대박": bool(ha[t_]), "다음분기 대박": bool(hb[t_])})
    X = pd.DataFrame(rec)

    rows = []
    for g in ["고", "중", "저", "전체"]:
        x = X if g == "전체" else X[X["변동성군"] == g]
        u = x["다음분기 대박"].mean()
        s = x[x["전분기 대박"]]
        c = s["다음분기 대박"].mean() if len(s) else np.nan
        rows.append({"변동성군": g, "표본": len(x), "무조건부 대박률": float(u),
                     "전분기 대박이면": float(c) if len(s) else np.nan,
                     "배수": float(c / u) if (len(s) and u) else np.nan,
                     "조건 표본": len(s)})
    cond = pd.DataFrame(rows)

    # 종목별 대박 횟수 분포 vs 무작위(이항)
    hits = X.pivot_table(index="티커", columns="분기", values="다음분기 대박", aggfunc="first")
    part = hits.notna().sum(axis=1)
    cnt = hits.sum(axis=1)[part >= 8].astype(int)
    nq = hits.shape[1]
    dist = pd.DataFrame([{
        "대박 횟수": f"{k}회" if k < 5 else "5회+",
        "실측 종목": int((cnt == k).sum()) if k < 5 else int((cnt >= 5).sum()),
        "무작위 기대": (len(cnt) * comb(nq, k) * TOP ** k * (1 - TOP) ** (nq - k)
                   if k < 5 else
                   sum(len(cnt) * comb(nq, j) * TOP ** j * (1 - TOP) ** (nq - j)
                       for j in range(5, nq + 1))),
    } for k in range(6)])
    return cond, dist, X


def main() -> int:
    print("· 패널 로드")
    df = G.load_raw()
    P, _ = prep(df)
    print(f"· 분기 전이 {len(P)}쌍 ({P[0][0]} ~ {P[-1][0]})")
    pd.set_option("display.width", 200)

    T, S = explain(P)
    print("\n■ ② 전분기 변동성 vs 수익성 — 다음 분기 설명력 (분기별 스피어만)")
    print(T.round(3).to_string(index=False))
    print("\n요약")
    print(S.round(4).to_string(index=False))

    cond, dist, X = repeat(P)
    print("\n■ ① 대박(상위 5%) 반복성 — 변동성 통제 전/후")
    print(cond.round(4).to_string(index=False))
    print("\n종목별 대박 횟수 분포 (8분기 이상 참여 종목)")
    print(dist.round(1).to_string(index=False))

    nm = (df.dropna(subset=["Name"]).sort_values("Date")
          .groupby("Ticker")["Name"].last())
    h = X[X["다음분기 대박"]].groupby("티커")["분기"].apply(list)
    h = h[h.apply(len) >= 3].sort_values(key=lambda s: s.apply(len), ascending=False)
    print(f"\n상위 5% 를 3회 이상 찍은 종목 {len(h)}개")
    for t_, qq in h.items():
        print(f"  {nm.get(t_, t_)[:12]:<13} {len(qq)}회 — {', '.join(qq)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
