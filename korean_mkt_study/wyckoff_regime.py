"""와이코프 시그널 × 시장 국면 (2026-11-10 케인 — 유형 규칙에 국면 통제를 넣는 점검 2번째, (A) LLV 패널).

입력  LLV data/ohlcv/panel.parquet — Wyckoff_Signal · Wyckoff_Signal_Strength · Wyckoff_Signal_Desc (HillStorm 위임 계산, 운영 정의 그대로)
      ETF(102110·252650) 제외 · 219종목 · 2023-01-02~
측정  시그널일 t 종가에 판단 → t+1 시가 진입. r_h = t+h 종가 / t+1 시가 − 1 − 0.3% (h = 5·10·20 거래일, 진입일 포함 보유일수)
      시장 m_h = 코스피 200 같은 창 (t+1 시가 → t+h 종가). **초과 x_h = r_h − m_h 가 기본** (반등 룰 §9.4 교훈 — 국면은 시장 방향을 같이 바꾼다)
국면  시그널일 t 종가 라벨 (regime_history) — 브레드스 · C2 · 갭쇼크 창
구간  학습: t+20 ≤ 2026-05-29 (2023~2024 / 2025~2026.05 분할) · 보류: t ≥ 2026-06-01 (--holdout, 검증 때만)
신뢰구간 = 시그널일 단위 부트스트랩 1,000회
출력  out/regime/wyckoff_regime{_holdout}.csv
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from market_layer import attach_regime, load_history            # noqa: E402

LLV = HERE.parents[1] / "LongLiveVault" / "data" / "ohlcv"
LEARN_END = pd.Timestamp("2026-05-29"); HOLD_START = pd.Timestamp("2026-06-01")
COST = 0.003; HS = (5, 10, 20); NB = 1000
RNG = np.random.default_rng(20261110)
ETF = {"102110", "252650"}
AX = {"breadth": {True: "상승", False: "하락"}, "c2": {True: "보유", False: "퇴출"},
      "gate_window": {True: "갭쇼크창 안", False: "갭쇼크창 밖"}}


def build() -> pd.DataFrame:
    p = pd.read_parquet(LLV / "panel.parquet", columns=["Date", "Ticker", "Open", "Close", "Wyckoff_Signal",
                                                         "Wyckoff_Signal_Strength", "Wyckoff_Signal_Desc"])
    p["Date"] = pd.to_datetime(p.Date); p["Ticker"] = p.Ticker.astype(str)
    p = p[~p.Ticker.isin(ETF)].sort_values(["Ticker", "Date"]).reset_index(drop=True)
    g = p.groupby("Ticker", sort=False)
    p["o1"] = g.Open.shift(-1)
    for h in HS:
        p[f"c{h}"] = g.Close.shift(-h); p[f"d{h}"] = g.Date.shift(-h)
    ev = p.dropna(subset=["Wyckoff_Signal"]).copy()
    K = pd.read_parquet(LLV / "tickers" / "KOSPI200.parquet"); K["Date"] = pd.to_datetime(K.Date)
    K = K.drop_duplicates("Date", keep="last").set_index("Date").sort_index()
    days = K.index; pos = days.searchsorted(ev.Date.values)
    for h in HS:
        ok = pos + h < len(days)
        mo = np.full(len(ev), np.nan); mc = mo.copy()
        mo[ok] = K.Open.to_numpy(float)[pos[ok] + 1]; mc[ok] = K.Close.to_numpy(float)[pos[ok] + h]
        good = (ev.o1 > 0).to_numpy()
        r = np.where(good, ev[f"c{h}"].to_numpy(float) / ev.o1.to_numpy(float) - 1 - COST, np.nan)
        ev[f"r{h}"] = r; ev[f"m{h}"] = mc / mo - 1; ev[f"x{h}"] = r - ev[f"m{h}"]
    ev["sig"] = ev.Wyckoff_Signal.astype(str)
    ev["strength"] = ev.Wyckoff_Signal_Strength
    ev["weis"] = ev.Wyckoff_Signal_Desc.fillna("").str.contains("Weis동반")
    return ev.drop(columns=["Wyckoff_Signal", "Wyckoff_Signal_Strength"]).reset_index(drop=True)


METRICS = {"x5평균": ("x5", "mean"), "x10평균": ("x10", "mean"), "x20평균": ("x20", "mean"), "x10>0%": ("x10", "pos"),
           "r10평균": ("r10", "mean"), "시장m10": ("m10", "mean")}


def stats(d: pd.DataFrame):
    cols = {}
    for m, (c, k) in METRICS.items():
        x = d[c].astype(float)
        a = ((x > 0).astype(float) if k == "pos" else x).where(x.notna(), 0.0)
        cols[m + "_n"], cols[m + "_d"] = a.to_numpy(), x.notna().astype(float).to_numpy()
    day = pd.DataFrame(cols).groupby(d.Date.to_numpy()).sum(); tot = day.sum()
    pt = {"건수": len(d), "신호일": day.shape[0], **{m: tot[m + "_n"] / tot[m + "_d"] for m in METRICS}}
    a = day.to_numpy(); cl = list(day.columns); s = a[RNG.integers(0, len(a), size=(NB, len(a)))].sum(axis=1)
    return pt, {m: s[:, cl.index(m + "_n")] / s[:, cl.index(m + "_d")] for m in METRICS}


def groups(ev: pd.DataFrame) -> dict:
    out = {"전 시그널": ev}
    for s in sorted(ev.sig.unique()):
        e = ev[ev.sig == s]; out[s] = e
        if e.strength.nunique() > 1:
            for v in sorted(e.strength.dropna().unique()):
                out[f"{s} 강도{int(v)}"] = e[e.strength == v]
        if s == "ACC_COMPLETE":
            out[f"{s} Weis동반"] = e[e.weis]; out[f"{s} Weis없음"] = e[~e.weis]
    return out


def table(ev: pd.DataFrame, hist: pd.DataFrame, periods: dict, min_n: int = 20) -> pd.DataFrame:
    e0 = attach_regime(ev, "Date", cols=list(AX), hist=hist, prefix="")
    out = []
    for per, (a, b) in periods.items():
        ep = e0[(e0.Date >= a) & (e0.Date <= b)]
        for gname, e in groups(ep).items():
            if len(e) >= min_n:
                pt, _ = stats(e)
                out.append(pd.DataFrame([pt], index=["(전체)"]).assign(기간=per, 묶음=gname, 축="-").reset_index(names="그룹"))
            for ax, labs in AX.items():
                rows, bss = {}, {}
                for v, lab in labs.items():
                    gg = e[e[ax] == v]
                    if len(gg) >= min_n:
                        rows[lab], bss[lab] = stats(gg)
                if not rows:
                    continue
                t = pd.DataFrame(rows).T
                if len(bss) == 2:
                    x, y = list(bss)
                    t.loc[f"차({x}−{y})"] = [np.nan, np.nan] + [t.loc[x, m] - t.loc[y, m] for m in METRICS]
                    t.loc["CI하"] = [np.nan, np.nan] + [np.percentile(bss[x][m] - bss[y][m], 2.5) for m in METRICS]
                    t.loc["CI상"] = [np.nan, np.nan] + [np.percentile(bss[x][m] - bss[y][m], 97.5) for m in METRICS]
                out.append(t.assign(기간=per, 묶음=gname, 축=ax).reset_index(names="그룹"))
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    t0 = time.time()
    hist = load_history(); ev = build()
    if "--holdout" in sys.argv:
        print("⚠ 보류 구간(2026-06~) — 검증 때만")
        ev = ev[(ev.Date >= HOLD_START) & ev.d20.notna()]; periods = {"보류": ("2026-06-01", "2026-12-31")}; tag = "_holdout"
    else:
        ev = ev[ev.d20.notna() & (pd.to_datetime(ev.d20) <= LEARN_END)]
        periods = {"전체": ("2023-01-01", "2026-05-29"), "2023~2024": ("2023-01-01", "2024-12-31"), "2025~2026.05": ("2025-01-01", "2026-05-29")}
        tag = ""
    print(f"시그널 {len(ev):,} · {ev.Date.min():%Y-%m-%d}~{ev.Date.max():%Y-%m-%d} · {ev.sig.value_counts().to_dict()}")
    T = table(ev, hist, periods)
    out = HERE / "out" / "regime" / f"wyckoff_regime{tag}.csv"; out.parent.mkdir(parents=True, exist_ok=True); T.to_csv(out, index=False)
    print(f"저장 {out} · {time.time() - t0:.0f}s")
