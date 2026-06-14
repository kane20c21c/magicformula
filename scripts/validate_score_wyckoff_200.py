#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_score_wyckoff_200.py
=============================
200종목(core 69 + extend 131) 패널로 두 가설을 검증한다.

가설1) 종합점수(v2_combined, 운영본 가중치)가 5/10일 후 수익률과 상관
가설2) Wyckoff_Phase 국면이 5/10일 후 수익률과 상관

방법
----
1. 패널 구축
   - core.parquet + extend.parquet → 200종목 stock_data{ticker: df(Date index)}
   - make_regimes(전 종목 횡단면) → compute_area_scores → combine_scores
     · 종합점수는 gate=False(순수 점수 예측력, H1 주분석)와
       gate=True(운영 게이트 적용분, 보조) 둘 다 계산
   - 포워드 수익률: ic_framework.compute_fwd_return, horizon 5/10, mode raw/alpha
     · alpha 벤치마크 = KOSPI200 (tickers/KOSPI200.parquet)
   - 룩어헤드 없음: 종합점수는 backward 지표·backward breadth (검증필)
2. 가설1
   - pooled Spearman/Pearson IC (evaluate_variant), decile 버킷 평균 포워드리턴 + 상하위 스프레드
   - 일별 단면 IC 시계열 + Newey-West(HAC) t값 → IC IR (윈도 중첩 자기상관 보정)
3. 가설2
   - 국면별 포워드리턴 평균/중앙값/승률, Kruskal-Wallis, 쌍별 Mann-Whitney
4. robustness: 기간 전·후반 분할 IC 재현

출력: out_dir/ 에 JSON(metrics) + PNG(차트 3종)
실행: PYTHONPATH=<pylibs>:<Magic Formula> python3 scripts/validate_score_wyckoff_200.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

warnings.filterwarnings("ignore")

from magic_formula.analysis import area_scores as A
from magic_formula.analysis import ic_framework as IC

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
VAULT = Path("/sessions/practical-friendly-mccarthy/mnt/longlivevault/data/ohlcv")
CORE = VAULT / "core.parquet"
EXTEND = VAULT / "extend.parquet"
KOSPI200 = VAULT / "tickers" / "KOSPI200.parquet"
OUT = Path("/sessions/practical-friendly-mccarthy/mnt/outputs/validation")
OUT.mkdir(parents=True, exist_ok=True)

WEIGHTS = A.COMBINED_WEIGHTS          # 운영본: T0.2/M0.2/Vu0/Va0.6
HORIZONS = [5, 10]
MODES = ["raw", "alpha"]


# ---------------------------------------------------------------------------
# 1. 패널 구축
# ---------------------------------------------------------------------------
def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Date 인덱스 정렬 + 정렬."""
    g = df.sort_values("Date").copy()
    g.index = pd.DatetimeIndex(g["Date"])
    return g


def load_universe() -> dict[str, pd.DataFrame]:
    core = pd.read_parquet(CORE)
    ext = pd.read_parquet(EXTEND)
    panel = pd.concat([core, ext], ignore_index=True)
    stock = {t: _prep(g) for t, g in panel.groupby("Ticker")}
    return stock


def load_kospi200() -> pd.DataFrame | None:
    if not KOSPI200.exists():
        return None
    k = pd.read_parquet(KOSPI200).sort_values("Date")
    k.index = pd.DatetimeIndex(k["Date"])
    return k


def build_scores(stock: dict[str, pd.DataFrame]):
    """종합점수(gate off/on) 시계열 재계산."""
    regime_b, regime_q = A.make_regimes(stock)
    scores_off, scores_on, phases = {}, {}, {}
    for t, df in stock.items():
        areas = A.compute_area_scores(df, regime_b, regime_q)
        wy_label = df["Wyckoff_Label"] if "Wyckoff_Label" in df.columns else None
        scores_off[t] = A.combine_scores(areas, WEIGHTS, wy_label, gate=False)
        scores_on[t] = A.combine_scores(areas, WEIGHTS, wy_label, gate=True)
        phases[t] = df["Wyckoff_Phase"] if "Wyckoff_Phase" in df.columns else pd.Series(dtype=object)
    return scores_off, scores_on, phases


def build_fwd(stock: dict[str, pd.DataFrame], kospi: pd.DataFrame | None,
              horizon: int, mode: str) -> dict[str, pd.Series]:
    return {t: IC.compute_fwd_return(df, horizon_days=horizon, mode=mode, kospi_df=kospi)
            for t, df in stock.items()}


# ---------------------------------------------------------------------------
# 2. 가설1 — 일별 단면 IC + Newey-West
# ---------------------------------------------------------------------------
def _newey_west_tstat(x: np.ndarray, lags: int) -> tuple[float, float, float]:
    """평균 mean(x)의 HAC(Newey-West) 표준오차 기반 t값. 반환 (mean, se, t)."""
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return (np.nan, np.nan, np.nan)
    mu = x.mean()
    e = x - mu
    gamma0 = (e @ e) / n
    s = gamma0
    for L in range(1, min(lags, n - 1) + 1):
        w = 1.0 - L / (lags + 1.0)          # Bartlett kernel
        cov = (e[L:] @ e[:-L]) / n
        s += 2.0 * w * cov
    var_mean = s / n                        # var of sample mean under HAC
    se = np.sqrt(var_mean) if var_mean > 0 else np.nan
    tval = mu / se if se and se > 0 else np.nan
    return (mu, se, tval)


def daily_cross_sectional_ic(scores: dict[str, pd.Series], fwd: dict[str, pd.Series],
                             horizon: int) -> dict:
    """각 거래일 단면 Spearman IC → 시계열. 평균 IC + Newey-West t (lag=horizon)."""
    # long panel
    rows = []
    for t in scores:
        s, f = scores[t], fwd.get(t)
        if f is None:
            continue
        common = s.index.intersection(f.index)
        sub = pd.DataFrame({"score": s.loc[common], "fwd": f.loc[common]}).dropna()
        sub["ticker"] = t
        sub["date"] = sub.index
        rows.append(sub)
    if not rows:
        return {}
    panel = pd.concat(rows, ignore_index=True)
    ic_series = {}
    for d, g in panel.groupby("date"):
        if g["score"].nunique() < 5 or len(g) < 5:
            continue
        ic = st.spearmanr(g["score"], g["fwd"]).correlation
        if not np.isnan(ic):
            ic_series[d] = ic
    ic_ser = pd.Series(ic_series).sort_index()
    mu, se, tval = _newey_west_tstat(ic_ser.values, lags=horizon)
    ir = mu / ic_ser.std() if ic_ser.std() > 0 else np.nan   # IC IR (정보비율)
    return {
        "n_days": int(len(ic_ser)),
        "mean_ic": float(mu) if not np.isnan(mu) else None,
        "ic_std": float(ic_ser.std()),
        "ic_ir": float(ir) if not np.isnan(ir) else None,
        "nw_se": float(se) if not np.isnan(se) else None,
        "nw_tstat": float(tval) if not np.isnan(tval) else None,
        "pct_positive_days": float((ic_ser > 0).mean()),
        "_ic_series": ic_ser,        # 차트용 (JSON 직렬화 시 제외)
    }


def decile_spread(scores: dict[str, pd.Series], fwd: dict[str, pd.Series],
                  nbin: int = 10) -> dict:
    """pooled 점수 decile별 평균 포워드리턴 + 상하위 스프레드."""
    ss, ff = [], []
    for t in scores:
        s, f = scores[t], fwd.get(t)
        if f is None:
            continue
        common = s.index.intersection(f.index)
        sub = pd.DataFrame({"s": s.loc[common], "f": f.loc[common]}).dropna()
        ss.append(sub["s"]); ff.append(sub["f"])
    if not ss:
        return {}
    s_all = pd.concat(ss).reset_index(drop=True)
    f_all = pd.concat(ff).reset_index(drop=True)
    try:
        q = pd.qcut(s_all, nbin, labels=False, duplicates="drop")
    except ValueError:
        return {}
    means = f_all.groupby(q).mean()
    counts = f_all.groupby(q).size()
    lo, hi = means.index.min(), means.index.max()
    spread = float(means.loc[hi] - means.loc[lo])
    # 스프레드 유의성: 상위빈 vs 하위빈 t검정
    top = f_all[q == hi]; bot = f_all[q == lo]
    tt = st.ttest_ind(top, bot, equal_var=False)
    return {
        "n_obs": int(len(s_all)),
        "decile_mean_fwd": {int(k): float(v) for k, v in means.items()},
        "decile_count": {int(k): int(v) for k, v in counts.items()},
        "top_minus_bottom": spread,
        "tb_tstat": float(tt.statistic),
        "tb_pvalue": float(tt.pvalue),
        "monotonic_up": bool(means.is_monotonic_increasing),
    }


# ---------------------------------------------------------------------------
# 3. 가설2 — 국면별 포워드리턴
# ---------------------------------------------------------------------------
def phase_analysis(phases: dict[str, pd.Series], fwd: dict[str, pd.Series]) -> dict:
    rows = []
    for t in phases:
        p, f = phases[t], fwd.get(t)
        if f is None or p is None or len(p) == 0:
            continue
        common = p.index.intersection(f.index)
        sub = pd.DataFrame({"phase": p.loc[common], "fwd": f.loc[common]}).dropna()
        rows.append(sub)
    if not rows:
        return {}
    panel = pd.concat(rows, ignore_index=True)
    panel = panel[panel["phase"].isin(["Uptrend", "Downtrend", "Range", "Base"])]
    grp = panel.groupby("phase")["fwd"]
    stats = {}
    for ph, g in grp:
        stats[ph] = {
            "n": int(len(g)),
            "mean": float(g.mean()),
            "median": float(g.median()),
            "hit_rate": float((g > 0).mean()),
            "std": float(g.std()),
        }
    # Kruskal-Wallis (국면 그룹간 분포 차이)
    groups = [g.values for _, g in grp if len(g) >= 20]
    kw = st.kruskal(*groups) if len(groups) >= 2 else None
    # 쌍별 Mann-Whitney (주요 대비)
    pair = {}
    labels = list(grp.groups.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            ga, gb = grp.get_group(a), grp.get_group(b)
            if len(ga) >= 20 and len(gb) >= 20:
                u = st.mannwhitneyu(ga, gb, alternative="two-sided")
                pair[f"{a}_vs_{b}"] = {
                    "mean_diff": float(ga.mean() - gb.mean()),
                    "pvalue": float(u.pvalue),
                }
    return {
        "by_phase": stats,
        "kruskal_H": float(kw.statistic) if kw else None,
        "kruskal_pvalue": float(kw.pvalue) if kw else None,
        "pairwise": pair,
    }


# ---------------------------------------------------------------------------
# robustness: 기간 분할
# ---------------------------------------------------------------------------
def split_ic(scores, fwd, horizon):
    """전·후반 분할 평균 IC."""
    rows = []
    for t in scores:
        s, f = scores[t], fwd.get(t)
        if f is None:
            continue
        common = s.index.intersection(f.index)
        sub = pd.DataFrame({"s": s.loc[common], "f": f.loc[common]}).dropna()
        sub["date"] = sub.index
        rows.append(sub)
    if not rows:
        return {}
    panel = pd.concat(rows, ignore_index=True)
    mid = panel["date"].quantile(0.5)
    out = {}
    for name, seg in [("first_half", panel[panel.date <= mid]),
                      ("second_half", panel[panel.date > mid])]:
        if len(seg) > 50:
            ic = st.spearmanr(seg["s"], seg["f"]).correlation
            out[name] = {"n": int(len(seg)), "spearman_ic": float(ic)}
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("[1/4] 패널 구축...")
    stock = load_universe()
    kospi = load_kospi200()
    print(f"  종목 {len(stock)}개 / KOSPI200 {'O' if kospi is not None else 'X'}")

    scores_off, scores_on, phases = build_scores(stock)

    result = {"universe_n": len(stock), "weights": WEIGHTS,
              "h1_score": {}, "h2_wyckoff": {}}

    ic_series_store = {}
    for mode in MODES:
        for h in HORIZONS:
            key = f"h{h}_{mode}"
            fwd = build_fwd(stock, kospi, h, mode)

            # ── 가설1 (gate off = 순수 점수) ──
            ev = IC.evaluate_variant(scores_off, fwd)
            dx = daily_cross_sectional_ic(scores_off, fwd, h)
            ds = decile_spread(scores_off, fwd)
            sp = split_ic(scores_off, fwd, h)
            ic_series_store[key] = dx.pop("_ic_series", None)
            # gate on 보조 (운영 적용분 IC만)
            ev_on = IC.evaluate_variant(scores_on, fwd)
            result["h1_score"][key] = {
                "pooled_ic_spearman": ev.get("ic_spearman"),
                "pooled_ic_pearson": ev.get("ic_pearson"),
                "hit_rate": ev.get("hit_rate"),
                "n_obs": ev.get("n_obs"),
                "daily_ic": dx,
                "decile": ds,
                "split_robustness": sp,
                "gated_pooled_ic_spearman": ev_on.get("ic_spearman"),
                "gated_n_obs": ev_on.get("n_obs"),
            }

            # ── 가설2 ──
            result["h2_wyckoff"][key] = phase_analysis(phases, fwd)

    # JSON 저장
    (OUT / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[done] metrics → {OUT/'metrics.json'}")

    # 콘솔 요약
    print("\n===== 가설1: 종합점수 → 포워드리턴 =====")
    for key, v in result["h1_score"].items():
        d = v["daily_ic"]
        print(f"  [{key}] pooled Spearman IC={v['pooled_ic_spearman']:.4f} "
              f"hit={v['hit_rate']:.3f} | 일별 mean_IC={d.get('mean_ic')} "
              f"NW_t={d.get('nw_tstat')} IR={d.get('ic_ir')} "
              f"| decile스프레드={v['decile'].get('top_minus_bottom')}")
    print("\n===== 가설2: Wyckoff 국면 → 포워드리턴 =====")
    for key, v in result["h2_wyckoff"].items():
        print(f"  [{key}] KW_p={v.get('kruskal_pvalue')}")
        for ph, s in v.get("by_phase", {}).items():
            print(f"      {ph:10} n={s['n']:6} mean={s['mean']*100:+.2f}% "
                  f"med={s['median']*100:+.2f}% hit={s['hit_rate']:.3f}")

    # 차트
    make_charts(result, ic_series_store)
    return result


def make_charts(result, ic_series_store):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["axes.unicode_minus"] = False

    UP, DOWN = "#ef5350", "#1976D2"

    # (1) decile 스프레드 (h5/h10 raw)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, h in zip(axes, HORIZONS):
        dec = result["h1_score"][f"h{h}_raw"]["decile"]["decile_mean_fwd"]
        ks = sorted(dec.keys()); vs = [dec[k] * 100 for k in ks]
        colors = [UP if x >= 0 else DOWN for x in vs]
        ax.bar(ks, vs, color=colors)
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_title(f"종합점수 decile별 {h}일 포워드리턴 (raw)")
        ax.set_xlabel("점수 decile (0=최저 ~ 9=최고)"); ax.set_ylabel("평균 수익률 %")
    fig.tight_layout(); fig.savefig(OUT / "h1_decile_spread.png", dpi=130); plt.close(fig)

    # (2) 국면별 박스플롯스러운 막대 (mean ± std, h5/h10 raw)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    order = ["Base", "Range", "Uptrend", "Downtrend"]
    for ax, h in zip(axes, HORIZONS):
        bp = result["h2_wyckoff"][f"h{h}_raw"]["by_phase"]
        phs = [p for p in order if p in bp]
        means = [bp[p]["mean"] * 100 for p in phs]
        colors = [UP if m >= 0 else DOWN for m in means]
        ax.bar(phs, means, color=colors)
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_title(f"Wyckoff 국면별 {h}일 평균 포워드리턴 (raw)")
        ax.set_ylabel("평균 수익률 %")
    fig.tight_layout(); fig.savefig(OUT / "h2_phase_returns.png", dpi=130); plt.close(fig)

    # (3) 일별 IC 시계열 (h5 raw 누적)
    ser = ic_series_store.get("h5_raw")
    if ser is not None and len(ser):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(ser.index, ser.values, color="#888", lw=0.6, alpha=0.6, label="일별 IC")
        ax.plot(ser.index, ser.rolling(20).mean(), color=UP, lw=1.6, label="20일 이동평균")
        ax.axhline(0, color="#333", lw=0.8)
        ax.set_title("종합점수 일별 단면 IC (5일 포워드, raw)")
        ax.legend(); fig.tight_layout()
        fig.savefig(OUT / "h1_daily_ic.png", dpi=130); plt.close(fig)
    print(f"[charts] → {OUT}")


if __name__ == "__main__":
    main()
