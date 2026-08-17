"""verify_entry.py — 진입 개선 후보 최종 검증.

축 A/B/C 스윕에서 살아남은 후보를 대상으로
  1) 레짐별 분해
  2) 블록 부트스트랩 (일별 수익률 차이의 평균 > 0 인가)
  3) 민감도 — 슬리피지·유니버스·기간 시작점
를 돌린다. 판단 근거는 MDD·Sharpe (수익률은 대개 유의차가 안 난다).

실행: python3 verify_entry.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from backtest import (EntryParams, ExitParams, PortfolioParams, UniverseParams,
                      VolScaleParams, load_panel, metrics, run_backtest)

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

MA200 = dict(trend_ma=200, trend_min_periods=140)

CANDIDATES = {
    "기준선 v1.0.0": EntryParams(),
    "P1 우선순위=여력큰순": EntryParams(priority="trend"),
    "T1 추세 MA200": EntryParams(**MA200),
    "T1+P1 MA200+여력큰순": EntryParams(**MA200, priority="trend"),
    "T2 추세 MA180": EntryParams(trend_ma=180, trend_min_periods=126),
    "T3 추세 MA220": EntryParams(trend_ma=220, trend_min_periods=154),
    "(참고) A1 반등확인 mid/3": EntryParams(confirm="mid", confirm_max_wait=3),
    "(참고) R1a 추세필터 제거": EntryParams(trend_margin=-1e9),
}

REGIMES = {
    "2015~2019 평시": ("2015-01-01", "2019-12-31"),
    "2020~2021 코로나": ("2020-01-01", "2021-12-31"),
    "2022~2023 하락": ("2022-01-01", "2023-12-31"),
    "2024~2026 멜트업": ("2024-01-01", "2026-06-30"),
}


def block_bootstrap(a: pd.Series, b: pd.Series, block: int = 20,
                    n_boot: int = 5000, seed: int = 20260817) -> dict:
    """일별 수익률 차이(a−b)의 평균이 0보다 큰지 — 이동블록 부트스트랩 양측 P값."""
    d = (a - b).dropna().to_numpy()
    n = len(d)
    if n < block * 3:
        return {"mean_diff": np.nan, "p": np.nan}
    rng = np.random.default_rng(seed)
    n_blk = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blk))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    means = d[idx].mean(axis=1)
    obs = d.mean()
    # 귀무가설: 평균 차이 0 → 부트스트랩 분포를 중심화해 관측치의 극단성 평가
    centered = means - means.mean()
    p = float((np.abs(centered) >= abs(obs)).mean())
    return {"mean_diff": float(obs), "p": p,
            "ann_diff": float(obs * 252), "boot_std": float(means.std())}


if __name__ == "__main__":
    panel = load_panel(UniverseParams(), VolScaleParams())
    print(f"패널 종목 {len(panel.tickers)} · 거래일 {len(panel.dates)}\n")

    pp = PortfolioParams()
    res = {k: run_backtest(panel, ep, ExitParams(), pp) for k, ep in CANDIDATES.items()}

    print("── 전구간 2015-01~2026-06 ──")
    rows = []
    for k, r in res.items():
        m = metrics(r)
        rows.append({"안": k, "최종(억)": round(m["final"] / 1e8, 2),
                     "CAGR%": round(m["cagr"] * 100, 2), "Sharpe": round(m["sharpe"], 2),
                     "MDD%": round(m["mdd"] * 100, 1),
                     "투자비중%": round(m["invested"] * 100, 1),
                     "매수": m["n_buy"], "승률%": round(m["win_rate"] * 100, 1),
                     "건당수익%": round(m["avg_ret"] * 100, 2)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n── 블록 부트스트랩 (vs 기준선, 20일 블록 · 5,000회) ──")
    base_eq = res["기준선 v1.0.0"]["equity"]["equity"].pct_change()
    rows = []
    for k, r in res.items():
        if k.startswith("기준선"):
            continue
        bs = block_bootstrap(r["equity"]["equity"].pct_change(), base_eq)
        rows.append({"안": k, "일평균 차이(bp)": round(bs["mean_diff"] * 1e4, 2),
                     "연환산 차이%": round(bs["ann_diff"] * 100, 2),
                     "P값": round(bs["p"], 3),
                     "유의(5%)": "○" if bs["p"] < 0.05 else "×"})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n── 레짐별 손익(만원) / MDD% ──")
    rows = []
    for k, ep in CANDIDATES.items():
        row = {"안": k}
        for name, (s, e) in REGIMES.items():
            m = metrics(run_backtest(panel, ep, ExitParams(), pp, start=s, end=e))
            row[name] = f"{m['profit']/1e4:,.0f} / {m['mdd']*100:.1f}"
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n── 민감도 ──")
    rows = []
    for k, ep in CANDIDATES.items():
        row = {"안": k}
        # 슬리피지 0%
        m = metrics(run_backtest(panel, ep, ExitParams(), PortfolioParams(stop_slippage=0.0)))
        row["슬리피지0%"] = f"{m['final']/1e8:.2f}억/{m['sharpe']:.2f}"
        # 기간 시작 2018
        m = metrics(run_backtest(panel, ep, ExitParams(), pp, start="2018-01-01"))
        row["2018~"] = f"{m['final']/1e8:.2f}억/{m['sharpe']:.2f}"
        # 전반/후반 분리
        m = metrics(run_backtest(panel, ep, ExitParams(), pp, start="2015-01-01", end="2020-12-31"))
        row["전반 15~20"] = f"{m['final']/1e8:.2f}억/{m['sharpe']:.2f}"
        m = metrics(run_backtest(panel, ep, ExitParams(), pp, start="2021-01-01", end="2026-06-30"))
        row["후반 21~26"] = f"{m['final']/1e8:.2f}억/{m['sharpe']:.2f}"
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n── 유니버스 5조/30% (STRATEGY.md 문서값) ──")
    p2 = load_panel(UniverseParams(mktcap_min_krw=5e12, foreign_min_pct=30.0), VolScaleParams())
    rows = []
    for k, ep in CANDIDATES.items():
        m = metrics(run_backtest(p2, ep, ExitParams(), pp))
        rows.append({"안": k, "최종(억)": round(m["final"] / 1e8, 2),
                     "Sharpe": round(m["sharpe"], 2), "MDD%": round(m["mdd"] * 100, 1),
                     "투자비중%": round(m["invested"] * 100, 1)})
    print(pd.DataFrame(rows).to_string(index=False))
