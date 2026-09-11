#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yz_hit_definition.py — '대박' 을 어떻게 정의할 것인가
======================================================
케인 지시 2026-09-11. 정의를 바꿔가며 개수와 변별력을 비교한 기록.

  "수익율 상위 5% 라고 정의 했으면 조금 애매하네..
   분기 수익율이 동일가중 수익율보다 20% 높은 경우를 대박이라고 정의하면
   분기에 몇 종목쯤 나와? 50% 높다고 하면?"
   → "20%p 말고 20%.. 동일가중이 9.7%이면 9.7% × 120% = 11.6% 이상.
      만약 동일가중이 0% 이하이면 0% 보다 큰 경우"
   → "100%, 150%, 200%는?"
   → "각 분기의 상위 20위, 상위 40위를 기준으로 자른다면 문턱 수익율은 몇%?"

결론 (아래 표가 근거)
  **배수 방식은 이 데이터에서 안 통한다.** 동일가중 수익률이 0 근처인 분기가
  12분기 중 5개라, ×3.0 을 해도 문턱이 3~4% 에 그친다. 문턱이 무너지는 분기가
  표를 지배해 분기당 62종목(31%)이 '대박' 이 된다 — 변별력이 사라진다.
  **순위 기준이 낫다.** 상위 20위면 문턱이 12분기 전부 동일가중을 +15.8~66.2%p
  초과하고, 고변동군 비중이 52.9% 로 가장 선명하다.

전제: 유니버스에서 102110(BM ETF) 제외. 분기 전이 12쌍(2023Q4~2026Q3).
⚠ 삼성전자·SK하이닉스는 **여기서는 빼지 않는다** — 대박 정의를 고르는 문제라
  유니버스를 좁히면 개수 비교가 흐려진다(제외 효과는 §11-4 에서 ≈0 로 확인됨).

실행
  python3 scripts/yz_hit_definition.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import yz_group as G                                     # noqa: E402
import yz_group_quarterly as Q                           # noqa: E402
import yz_winners_probe as W                             # noqa: E402

REPO = G.REPO
BM_EQ, BM_CAP = "252000", "102110"
MULTS = (1.2, 1.5, 2.0, 2.5, 3.0)
ABS_PP = (0.20, 0.50)
RANKS = (20, 40)


def _bm(ser: pd.Series, qb: str, C: pd.DataFrame) -> float:
    s = C.index[C.index.to_period("Q") == pd.Period(qb)]
    x = ser.reindex(s).dropna()
    return float(x.iloc[-1] / x.iloc[0] - 1) if len(x) >= len(s) * 0.95 else np.nan


def build(df: pd.DataFrame) -> dict:
    """⚠ 변동성 순위와 수익률을 **같은 유니버스**에서 만든다.
    (초판은 변동성만 W.prep 의 제외본으로 계산하고 수익률은 포함본으로 계산해
     삼성·하이닉스가 분모에는 있고 고변동 카운트에는 없었다 — 고쳤다.)"""
    piv = lambda c: df.pivot_table(index="Date", columns="Ticker",
                                   values=c, aggfunc="last").astype(float)
    O, H, L, C = (piv(c) for c in ("Open", "High", "Low", "Close"))
    Cx, Ox, Hx, Lx = (t.drop(columns=[BM_CAP], errors="ignore") for t in (C, O, H, L))
    o, cc = np.log(Ox / Cx.shift(1)), np.log(Cx / Ox)
    u, dn = np.log(Hx / Ox), np.log(Lx / Ox)
    rs = u * (u - cc) + dn * (dn - cc)

    eq_p = (G.STOLAB / "longlivevault" / "data" / "ohlcv" / "tickers" / f"{BM_EQ}.parquet")
    if not eq_p.exists():
        raise SystemExit(f"❌ {BM_EQ} 없음 — yz_cells_etf.py 머리의 수집 명령 참조")
    e = pd.read_parquet(eq_p)
    e["Date"] = pd.to_datetime(e["Date"])
    EQ = e.set_index("Date")["Close"].astype(float)

    qs = sorted(set(C.index.to_period("Q")))
    rows = []
    for a, b in zip(qs, qs[1:]):
        if b < pd.Period("2023Q4"):
            continue
        sa = C.index[C.index.to_period("Q") == a]
        n = len(sa)
        if n < Q.MIN_Q_DAYS:
            continue
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        sig = np.sqrt(o.loc[sa].var(ddof=1) + k * cc.loc[sa].var(ddof=1)
                      + (1 - k) * rs.loc[sa].mean()).dropna()
        qb = str(b)
        s = C.index[C.index.to_period("Q") == b]
        rb = Cx.loc[s].apply(lambda c_: (c_.dropna().iloc[-1] / c_.dropna().iloc[0] - 1)
                             if c_.notna().sum() > 15 else np.nan).dropna()
        rb = rb[rb.index.isin(sig.index)]        # 변동성 판정이 있는 종목만
        gv = Q._t3(sig, Q.VOL3).reindex(rb.index)
        beq, bcap = _bm(EQ, qb, C), _bm(C[BM_CAP], qb, C)
        r = {"분기": qb, "종목": len(rb), "동일가중": beq, "시총가중": bcap,
             "1위": float(rb.max())}
        hi = lambda mask: int((mask & (gv == "고")).sum())
        for m in MULTS:                     # 배수 (동일가중 ≤0 이면 문턱 0)
            t = beq * m if beq > 0 else 0.0
            k = rb >= t
            r[f"배수{m}_문턱"], r[f"배수{m}_n"], r[f"배수{m}_고"] = t, int(k.sum()), hi(k)
        for m in MULTS:                     # 배수 (시총가중)
            t = bcap * m if bcap > 0 else 0.0
            k = rb >= t
            r[f"캡배수{m}_문턱"], r[f"캡배수{m}_n"], r[f"캡배수{m}_고"] = t, int(k.sum()), hi(k)
        for pp in ABS_PP:                   # 절대 초과폭
            k = rb >= beq + pp
            r[f"절대{pp}_n"], r[f"절대{pp}_고"] = int(k.sum()), hi(k)
        srt = rb.sort_values(ascending=False)
        for n in RANKS:                     # 순위
            t = float(srt.iloc[n - 1])
            k = rb >= t
            r[f"순위{n}_문턱"], r[f"순위{n}_대비"] = t, t - beq
            r[f"순위{n}_고"] = int((gv.reindex(srt.index[:n]) == "고").sum())
        rows.append(r)
    return {"T": pd.DataFrame(rows)}


def summarize(T: pd.DataFrame) -> pd.DataFrame:
    tot = T["종목"].sum()
    out = []
    for m in MULTS:
        out.append({"정의": f"동일가중 ×{m}", "분기당 종목": T[f"배수{m}_n"].mean(),
                    "비중": T[f"배수{m}_n"].sum() / tot,
                    "고변동 비중": T[f"배수{m}_고"].sum() / T[f"배수{m}_n"].sum(),
                    "분기 최소": int(T[f"배수{m}_n"].min()),
                    "분기 최대": int(T[f"배수{m}_n"].max())})
    for m in (2.0, 3.0):
        out.append({"정의": f"시총가중 ×{m}", "분기당 종목": T[f"캡배수{m}_n"].mean(),
                    "비중": T[f"캡배수{m}_n"].sum() / tot,
                    "고변동 비중": T[f"캡배수{m}_고"].sum() / T[f"캡배수{m}_n"].sum(),
                    "분기 최소": int(T[f"캡배수{m}_n"].min()),
                    "분기 최대": int(T[f"캡배수{m}_n"].max())})
    for pp in ABS_PP:
        out.append({"정의": f"동일가중 +{pp:.0%}p", "분기당 종목": T[f"절대{pp}_n"].mean(),
                    "비중": T[f"절대{pp}_n"].sum() / tot,
                    "고변동 비중": T[f"절대{pp}_고"].sum() / T[f"절대{pp}_n"].sum(),
                    "분기 최소": int(T[f"절대{pp}_n"].min()),
                    "분기 최대": int(T[f"절대{pp}_n"].max())})
    for n in RANKS:
        out.append({"정의": f"상위 {n}위", "분기당 종목": float(n), "비중": n / T["종목"].mean(),
                    "고변동 비중": T[f"순위{n}_고"].sum() / (n * len(T)),
                    "분기 최소": n, "분기 최대": n})
    return pd.DataFrame(out)


def main() -> int:
    df = G.load_raw()
    R = build(df)
    T = R["T"]
    pd.set_option("display.width", 220)
    print("■ 분기별 문턱 — 배수 vs 순위\n")
    cols = ["분기", "종목", "동일가중", "시총가중", "1위",
            "배수2.0_문턱", "배수2.0_n", "배수3.0_문턱", "배수3.0_n",
            "순위20_문턱", "순위20_대비", "순위40_문턱", "순위40_대비"]
    print(T[cols].round(4).to_string(index=False))
    print("\n■ 정의별 요약 (12분기)")
    print(summarize(T).round(4).to_string(index=False))
    print("\n  ⇒ 배수 방식은 동일가중이 0 근처인 분기에서 문턱이 무너져 변별력을 잃는다.")
    print("    순위 기준(상위 20위)이 고변동 비중 52.9% 로 가장 선명하다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
