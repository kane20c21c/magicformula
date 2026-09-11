#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yz_appearance_probe.py — 일간 상위/하위 등장비율의 예측력 (검증 1·2·3)
======================================================================
케인 지시 2026-09-11.

  "특정일(d)에서 d−1까지의 40거래일 윈도우, 상위/하위 40위의 등장비율과
   d일의 상승율의 상관관계를 봐줘.
     검증1) 7월1일부터 40거래일을 run forward 방식으로
     검증2) 40 / 20 / 60 거래일 윈도우로
     검증3) 일간 · 주간 · 격주 단위가 의미가 있을까"

정의
  등장비율 = (윈도우 내 **일간 상승률 상위 40위** 진입 횟수)
             ÷ (상위 40위 + **하위 40위** 진입 횟수)
  윈도우는 **d−1 까지** — 룩어헤드 없음.
  목표수익률 = C[d+H−1] / C[d−1] − 1  (H=1 이면 곧 d일 수익률)

⚠⚠ 목표가 H>1 이면 이웃한 d 의 목표가 겹친다 — 매일 계산해 부호검정하면
  유사반복이 된다. 그래서 **d 를 H 간격으로 띄워 비중첩 표본**만 쓴다.
  (표본 확보를 위해 평가 구간을 2023-04 ~ 2026-09 전 기간으로 늘렸다.)

결론 (셋 다 귀무)
  검증1  40일 윈도우 → d일 : 25/50, p=1.000
  검증2  20/40/60 전부 0. 윈도우가 길수록 비율이 매끄러워질 뿐(sd 0.164→0.099)
         예측력은 그대로다.
  검증3  12칸 격자 전부 0. 최저 p=0.188 (본페로니 임계 0.0042).
  ★ 원인: **등장비율은 그 기간 수익률의 재표현**이다 — 같은 창의 단순 수익률과
    동어반복도 0.71~0.75, 일별 ρ 계열 상관 0.93~0.96. 그 수익률 자체가 미래를
    설명하지 못하니 당연한 귀결이다.
  ※ 유일한 p<0.05 는 `20일 윈도우 × 1일 목표` 의 **단순 수익률**(382/838, p=0.012,
    ρ=−0.016) — 음의 방향이라 단기 반전이다. 크기가 거래비용 아래고 본페로니도
    못 넘어 쓸 수 없다.

실행
  python3 scripts/yz_appearance_probe.py
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

BM = "102110"
RANK_N = 40                    # 상위/하위 몇 위까지를 '등장' 으로 볼지
MIN_APPEAR = 5                 # 등장 합계가 이보다 적으면 비율이 잡음 — 제외
MIN_STOCKS = 50
WINDOWS = (20, 40, 60)
HORIZONS = {1: "1일", 5: "1주", 10: "2주", 20: "4주"}
EVAL_FROM = "2023-04-01"


def _binom_p(k: int, n: int) -> float:
    if n == 0:
        return np.nan
    pmf = [comb(n, i) / 2 ** n for i in range(n + 1)]
    return float(sum(p for p in pmf if p <= pmf[k] + 1e-12))


def prep(df: pd.DataFrame):
    C = (df.pivot_table(index="Date", columns="Ticker", values="Close", aggfunc="last")
         .astype(float).drop(columns=[BM], errors="ignore"))
    ret = C / C.shift(1) - 1
    up = (ret.rank(axis=1, ascending=False, method="first").le(RANK_N) & ret.notna()).astype(float)
    dn = (ret.rank(axis=1, ascending=True, method="first").le(RANK_N) & ret.notna()).astype(float)
    return C, ret, up, dn


def grid(df: pd.DataFrame, windows=WINDOWS, horizons=tuple(HORIZONS)) -> pd.DataFrame:
    C, ret, up, dn = prep(df)
    idx = C.index
    # ⚠ shift(1) 을 먼저 걸어 창이 d−1 에서 끝나게 한다 (룩어헤드 차단)
    pre = {w: (up.shift(1).rolling(w).sum(), dn.shift(1).rolling(w).sum(),
               C.shift(1) / C.shift(w + 1) - 1) for w in windows}
    rows = []
    for w in windows:
        upR, dnR, pastR = pre[w]
        for h in horizons:
            fwd = C.shift(-(h - 1)) / C.shift(1) - 1
            i0 = max(idx.searchsorted(pd.Timestamp(EVAL_FROM)), w + 2)
            rb, rp = [], []
            for i in range(i0, len(idx) - h, h):        # 비중첩
                t = idx[i]
                tt = upR.loc[t] + dnR.loc[t]
                ratio = upR.loc[t] / tt.replace(0, np.nan)
                y, p_ = fwd.loc[t], pastR.loc[t]
                m = ratio.notna() & y.notna() & (tt >= MIN_APPEAR)
                if m.sum() < MIN_STOCKS:
                    continue
                rb.append(ratio[m].rank().corr(y[m].rank()))
                rp.append(p_[m].rank().corr(y[m].rank()))
            rb, rp = np.array(rb), np.array(rp)
            if not len(rb):
                continue
            kb, kp = int((rb > 0).sum()), int((rp > 0).sum())
            rows.append({
                "윈도우": w, "목표": HORIZONS[h], "표본일": len(rb),
                "ρ 등장비율": float(rb.mean()), "양수(비율)": f"{kb}/{len(rb)}",
                "p(비율)": _binom_p(kb, len(rb)),
                "ρ 단순수익": float(rp.mean()), "양수(수익)": f"{kp}/{len(rp)}",
                "p(수익)": _binom_p(kp, len(rp)),
            })
    return pd.DataFrame(rows)


def window_detail(df: pd.DataFrame, start="2026-07-01", windows=WINDOWS) -> pd.DataFrame:
    """검증2 — 윈도우가 바꾸는 것이 신호인지 측정 품질인지."""
    C, ret, up, dn = prep(df)
    idx = C.index
    i0 = idx.searchsorted(pd.Timestamp(start))
    rows = []
    for w in windows:
        upR, dnR = up.shift(1).rolling(w).sum(), dn.shift(1).rolling(w).sum()
        pastR = C.shift(1) / C.shift(w + 1) - 1
        acc = []
        for i in range(i0, len(idx)):
            t = idx[i]
            tt = upR.loc[t] + dnR.loc[t]
            ratio = upR.loc[t] / tt.replace(0, np.nan)
            y, p_ = ret.loc[t], pastR.loc[t]
            m = ratio.notna() & y.notna() & (tt >= MIN_APPEAR)
            if m.sum() < MIN_STOCKS:
                continue
            acc.append({"종목": int(m.sum()), "등장": float(tt[m].mean()),
                        "비율sd": float(ratio[m].std()),
                        "ρ비율": ratio[m].rank().corr(y[m].rank()),
                        "비율vs수익": ratio[m].rank().corr(p_[m].rank())})
        A = pd.DataFrame(acc)
        k = int((A["ρ비율"] > 0).sum())
        rows.append({"윈도우": w, "대상일": len(A), "종목": A["종목"].mean(),
                     "종목당 등장": A["등장"].mean(), "비율 sd": A["비율sd"].mean(),
                     "평균 ρ": A["ρ비율"].mean(), "양수": f"{k}/{len(A)}",
                     "p": _binom_p(k, len(A)),
                     "동어반복도": A["비율vs수익"].mean()})
    return pd.DataFrame(rows)


def main() -> int:
    df = G.load_raw()
    pd.set_option("display.width", 220)
    print("■ 검증2 — 윈도우가 바꾸는 것 (2026-07~09, 목표 d일 고정)")
    print(window_detail(df).round(4).to_string(index=False))
    print("\n  ⇒ 윈도우가 길수록 비율이 매끄러워지고 대상 종목이 늘 뿐, 예측력은 0 그대로다.")
    print("\n■ 검증3 — 윈도우 × 목표 격자 (비중첩, 2023-04~2026-09)")
    T = grid(df)
    print(T.round(4).to_string(index=False))
    print(f"\n  12칸 다중비교 — 본페로니 임계 p < {0.05/len(T):.4f}")
    print(f"  등장비율 최저 p = {T['p(비율)'].min():.4f} · 단순수익 최저 p = {T['p(수익)'].min():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
