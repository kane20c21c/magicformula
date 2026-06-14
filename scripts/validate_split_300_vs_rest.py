#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_split_300_vs_rest.py
==============================
유니버스 198종목(성장율 계산가능)을 2개 그룹으로 나눠 가설1·2를 비교한다.

  Group A (고성장)  : 2025-04-30 → 2026-06-12 총 성장율 > 300%  (45종목)
  Group B (그외)    : 총 성장율 ≤ 300%                          (153종목)

가설1) 종합점수(v2_combined, 운영본 가중치 T0.2/M0.2/Vu0/Va0.6)가 5/10일 후 수익률과 상관
가설2) Wyckoff_Phase 국면이 5/10일 후 수익률과 상관

방법은 scripts/validate_score_wyckoff_200.py 와 동일.
차이점: 점수·레짐은 전체 200 유니버스로 계산(운영본과 동일, 룩어헤드 없음),
       그 다음 IC/국면 통계만 각 그룹으로 한정해 비교.
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import scipy.stats as st
warnings.filterwarnings("ignore")

from magic_formula.analysis import area_scores as A
from magic_formula.analysis import ic_framework as IC

BASE = Path("/sessions/funny-stoic-cray/mnt")
VAULT = BASE / "longlivevault" / "data" / "ohlcv"
CORE = VAULT / "core.parquet"; EXTEND = VAULT / "extend.parquet"
KOSPI200 = VAULT / "tickers" / "KOSPI200.parquet"
XLSX = BASE / "uploads" / "유니버스200_섹터분류용_20260611.xlsx"
OUT = BASE / "outputs" / "split300"; OUT.mkdir(parents=True, exist_ok=True)

WEIGHTS = A.COMBINED_WEIGHTS
HORIZONS = [5, 10]; MODES = ["raw", "alpha"]
BASE_D = pd.Timestamp("2025-04-30"); END_D = pd.Timestamp("2026-06-12")


def _prep(df):
    g = df.sort_values("Date").copy(); g.index = pd.DatetimeIndex(g["Date"]); return g

def load_universe():
    panel = pd.concat([pd.read_parquet(CORE), pd.read_parquet(EXTEND)], ignore_index=True)
    return {t: _prep(g) for t, g in panel.groupby("Ticker")}

def load_kospi200():
    k = pd.read_parquet(KOSPI200).sort_values("Date"); k.index = pd.DatetimeIndex(k["Date"]); return k

def build_scores(stock):
    rb, rq = A.make_regimes(stock)
    scores_off, phases = {}, {}
    for t, df in stock.items():
        areas = A.compute_area_scores(df, rb, rq)
        wy = df["Wyckoff_Label"] if "Wyckoff_Label" in df.columns else None
        scores_off[t] = A.combine_scores(areas, WEIGHTS, wy, gate=False)
        phases[t] = df["Wyckoff_Phase"] if "Wyckoff_Phase" in df.columns else pd.Series(dtype=object)
    return scores_off, phases

def build_fwd(stock, kospi, h, mode):
    return {t: IC.compute_fwd_return(df, horizon_days=h, mode=mode, kospi_df=kospi) for t, df in stock.items()}


def _nw_t(x, lags):
    x = x[~np.isnan(x)]; n = len(x)
    if n < 3: return (np.nan, np.nan, np.nan)
    mu = x.mean(); e = x - mu; s = (e @ e) / n
    for L in range(1, min(lags, n - 1) + 1):
        w = 1.0 - L / (lags + 1.0); s += 2.0 * w * (e[L:] @ e[:-L]) / n
    se = np.sqrt(s / n) if s > 0 else np.nan
    return (mu, se, mu / se if se and se > 0 else np.nan)

def daily_ic(scores, fwd, tickers, horizon):
    rows = []
    for t in tickers:
        s, f = scores.get(t), fwd.get(t)
        if s is None or f is None: continue
        common = s.index.intersection(f.index)
        sub = pd.DataFrame({"score": s.loc[common], "fwd": f.loc[common]}).dropna()
        sub["date"] = sub.index; rows.append(sub)
    if not rows: return {}
    panel = pd.concat(rows, ignore_index=True)
    ics = {}
    for d, g in panel.groupby("date"):
        if g["score"].nunique() < 5 or len(g) < 5: continue
        ic = st.spearmanr(g["score"], g["fwd"]).correlation
        if not np.isnan(ic): ics[d] = ic
    ser = pd.Series(ics).sort_index()
    if len(ser) < 3: return {"n_days": int(len(ser)), "mean_ic": None, "nw_tstat": None}
    mu, se, tv = _nw_t(ser.values, lags=horizon)
    return {"n_days": int(len(ser)), "mean_ic": float(mu), "ic_std": float(ser.std()),
            "nw_tstat": float(tv) if not np.isnan(tv) else None,
            "pct_pos_days": float((ser > 0).mean()), "_ser": ser}

def pooled_ic(scores, fwd, tickers):
    ss, ff = [], []
    for t in tickers:
        s, f = scores.get(t), fwd.get(t)
        if s is None or f is None: continue
        common = s.index.intersection(f.index)
        sub = pd.DataFrame({"s": s.loc[common], "f": f.loc[common]}).dropna()
        ss.append(sub["s"]); ff.append(sub["f"])
    if not ss: return {}
    s_all = pd.concat(ss); f_all = pd.concat(ff)
    sp = st.spearmanr(s_all, f_all).correlation
    pe = st.pearsonr(s_all, f_all)[0]
    hit = float((np.sign(s_all - s_all.median()) == np.sign(f_all)).mean())
    # decile
    try:
        q = pd.qcut(s_all.reset_index(drop=True), 10, labels=False, duplicates="drop")
        fa = f_all.reset_index(drop=True)
        means = fa.groupby(q).mean()
        spread = float(means.loc[means.index.max()] - means.loc[means.index.min()])
        mono = bool(means.is_monotonic_increasing)
    except Exception:
        spread, mono = None, None
    return {"n_obs": int(len(s_all)), "spearman_ic": float(sp), "pearson_ic": float(pe),
            "hit_rate": float((f_all > 0).mean()), "decile_spread": spread, "monotonic_up": mono}

def phase_analysis(phases, fwd, tickers):
    rows = []
    for t in tickers:
        p, f = phases.get(t), fwd.get(t)
        if p is None or f is None or len(p) == 0: continue
        common = p.index.intersection(f.index)
        sub = pd.DataFrame({"phase": p.loc[common], "fwd": f.loc[common]}).dropna()
        rows.append(sub)
    if not rows: return {}
    panel = pd.concat(rows, ignore_index=True)
    panel = panel[panel["phase"].isin(["Uptrend", "Downtrend", "Range", "Base"])]
    grp = panel.groupby("phase")["fwd"]
    stats = {ph: {"n": int(len(g)), "mean": float(g.mean()), "median": float(g.median()),
                  "hit_rate": float((g > 0).mean())} for ph, g in grp}
    groups = [g.values for _, g in grp if len(g) >= 20]
    kw = st.kruskal(*groups) if len(groups) >= 2 else None
    best_med = max(stats, key=lambda k: stats[k]["median"]) if stats else None
    return {"by_phase": stats, "kruskal_pvalue": float(kw.pvalue) if kw else None,
            "kruskal_H": float(kw.statistic) if kw else None, "best_median_phase": best_med}


def main():
    print("[1/3] 패널·점수 구축..."); stock = load_universe(); kospi = load_kospi200()
    scores, phases = build_scores(stock)

    # 그룹 분할
    xl = pd.read_excel(XLSX, sheet_name="유니버스200"); xl["티커"] = xl["티커"].astype(str).str.zfill(6)
    growth = {}
    for t, df in stock.items():
        try:
            bp = df.loc[df["Date"] == BASE_D, "Close"]; ep = df.loc[df["Date"] == END_D, "Close"]
            if len(bp) and len(ep): growth[t] = ep.iloc[0] / bp.iloc[0] - 1
        except Exception: pass
    g = pd.Series(growth)
    A_tk = sorted(g[g > 3.0].index); B_tk = sorted(g[g <= 3.0].index)
    print(f"  Group A(>300%)={len(A_tk)}  Group B(<=300%)={len(B_tk)}  계산가능={len(g)}")

    result = {"groupA_n": len(A_tk), "groupB_n": len(B_tk), "weights": WEIGHTS,
              "groupA_tickers": A_tk, "h1": {}, "h2": {}}
    ser_store = {}
    for mode in MODES:
        for h in HORIZONS:
            key = f"h{h}_{mode}"; fwd = build_fwd(stock, kospi, h, mode)
            result["h1"][key] = {}
            result["h2"][key] = {}
            for gname, tks in [("A", A_tk), ("B", B_tk)]:
                p = pooled_ic(scores, fwd, tks); d = daily_ic(scores, fwd, tks, h)
                ser_store[f"{key}_{gname}"] = d.pop("_ser", None)
                result["h1"][key][gname] = {**p, "daily": d}
                result["h2"][key][gname] = phase_analysis(phases, fwd, tks)

    (OUT / "metrics_split.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # 콘솔 요약
    print("\n===== 가설1: 종합점수 → 포워드리턴 (그룹 비교) =====")
    print(f"{'key':10}{'grp':4}{'pooledIC':>10}{'hit':>7}{'dailyIC':>10}{'NW_t':>8}{'decile':>9}{'mono':>6}")
    for key, gd in result["h1"].items():
        for gn, v in gd.items():
            d = v.get("daily", {})
            print(f"{key:10}{gn:4}{v.get('spearman_ic',float('nan')):>10.4f}{v.get('hit_rate',float('nan')):>7.3f}"
                  f"{(d.get('mean_ic') or float('nan')):>10.4f}{(d.get('nw_tstat') or float('nan')):>8.2f}"
                  f"{(v.get('decile_spread') or float('nan')):>9.4f}{str(v.get('monotonic_up')):>6}")
    print("\n===== 가설2: Wyckoff 국면 → 10일 raw 포워드리턴 =====")
    v = result["h2"]["h10_raw"]
    for gn in ["A", "B"]:
        d = v[gn]; print(f"  [Group {gn}] KW_p={d.get('kruskal_pvalue'):.2e}  최고중앙값국면={d.get('best_median_phase')}")
        for ph in ["Downtrend", "Range", "Uptrend", "Base"]:
            if ph in d["by_phase"]:
                s = d["by_phase"][ph]
                print(f"      {ph:10} n={s['n']:6} med={s['median']*100:+.2f}% hit={s['hit_rate']:.3f} mean={s['mean']*100:+.2f}%")
    return result, ser_store

if __name__ == "__main__":
    main()
