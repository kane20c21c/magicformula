#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_h3_capsize_wyckoff.py
==============================
가설3) 시가총액이 작을수록 와이코프 국면이 실제 가격행동과 더 잘 "일치"한다.

체급 (raw KRX 2026-06-11 시총, 단위 조)
  50조+ / 20~50조 / 10~20조 / 5~10조 / 3~5조 / <3조

정합도(alignment) 정의 — 방향성 국면(Uptrend·Downtrend) 한정
  · Uptrend  → fwd_ret > 0 이면 일치(국면=상승, 실제도 상승)
  · Downtrend→ fwd_ret < 0 이면 일치(국면=하락, 실제도 하락)
  alignment_rate(bucket) = P(일치 | 국면∈{Up,Down})   (0.5=무관, >0.5 모멘텀 정합, <0.5 역추세)

보조 지표
  · momentum_gap = hit(Uptrend) − hit(Downtrend)     (양수 클수록 모멘텀 정합)
  · KW effect    = Kruskal-Wallis H / N               (국면이 수익률을 가르는 효과크기)
  · 국면별 hit / median 테이블

추세검정: 버킷(소형=0…대형=5)과 정합도 지표의 Spearman → 가설3이면 음의 상관(소형↑정합)

forward return = raw (방향 일치가 핵심이라 raw 사용). horizon 5·10.
룩어헤드 없음(Wyckoff_Phase·MarketCap 모두 t시점 정보).
"""
from __future__ import annotations
import json, glob, warnings
from pathlib import Path
import numpy as np, pandas as pd, scipy.stats as st
warnings.filterwarnings("ignore")

from magic_formula.analysis import ic_framework as IC

VAULT = Path("/sessions/practical-friendly-mccarthy/mnt/longlivevault/data")
OHLCV = VAULT / "ohlcv"
RAW = VAULT / "raw"
OUT = Path("/sessions/practical-friendly-mccarthy/mnt/outputs/validation")
OUT.mkdir(parents=True, exist_ok=True)

ASOF = "20260611"
BINS = [0, 3, 5, 10, 20, 50, 1e9]
LABELS = ["<3조", "3~5조", "5~10조", "10~20조", "20~50조", "50조+"]
ORDER_SMALL_TO_BIG = LABELS                # index 0=소형
HORIZONS = [5, 10]
DIR_PHASES = ["Uptrend", "Downtrend"]


def _prep(df):
    g = df.sort_values("Date").copy()
    g.index = pd.DatetimeIndex(g["Date"])
    return g


def load_panel():
    core = pd.read_parquet(OHLCV / "core.parquet")
    ext = pd.read_parquet(OHLCV / "extend.parquet")
    panel = pd.concat([core, ext], ignore_index=True)
    return {t: _prep(g) for t, g in panel.groupby("Ticker")}


def cap_buckets(tickers):
    raw = pd.concat([pd.read_parquet(f) for f in glob.glob(str(RAW / f"krx_*{ASOF}*.parquet"))],
                    ignore_index=True).drop_duplicates("Ticker")
    mc = raw.set_index("Ticker")["MarketCap"]
    cap_jo = {t: float(mc[t]) / 1e12 for t in tickers if t in mc.index and mc[t] > 0}
    s = pd.Series(cap_jo)
    bucket = pd.cut(s, bins=BINS, labels=LABELS, right=False)
    return {t: str(bucket[t]) for t in s.index}, cap_jo


def build_fwd(stock, horizon):
    return {t: IC.compute_fwd_return(df, horizon_days=horizon, mode="raw")
            for t, df in stock.items()}


def bucket_panel(stock, phases_col, fwd, bmap):
    """(phase, fwd, bucket) long table."""
    rows = []
    for t, df in stock.items():
        if t not in bmap:
            continue
        p = df[phases_col] if phases_col in df.columns else None
        f = fwd.get(t)
        if p is None or f is None:
            continue
        common = p.index.intersection(f.index)
        sub = pd.DataFrame({"phase": p.loc[common], "fwd": f.loc[common]}).dropna()
        sub["bucket"] = bmap[t]
        rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def analyze_bucket(g):
    """버킷 내 정합도 지표."""
    gd = g[g["phase"].isin(DIR_PHASES)]
    # alignment
    consistent = ((gd["phase"] == "Uptrend") & (gd["fwd"] > 0)) | \
                 ((gd["phase"] == "Downtrend") & (gd["fwd"] < 0))
    align = float(consistent.mean()) if len(gd) else np.nan
    # momentum gap
    up = g[g["phase"] == "Uptrend"]["fwd"]; dn = g[g["phase"] == "Downtrend"]["fwd"]
    hit_up = float((up > 0).mean()) if len(up) else np.nan
    hit_dn = float((dn > 0).mean()) if len(dn) else np.nan
    mgap = hit_up - hit_dn if not (np.isnan(hit_up) or np.isnan(hit_dn)) else np.nan
    # KW effect (all 4 phases present in bucket)
    grp = [v.values for _, v in g.groupby("phase")["fwd"] if len(v) >= 20]
    kw = st.kruskal(*grp) if len(grp) >= 2 else None
    kw_eff = float(kw.statistic) / len(g) if kw else np.nan
    # per-phase
    by = {}
    for ph, v in g.groupby("phase")["fwd"]:
        by[ph] = {"n": int(len(v)), "median": float(v.median()), "hit": float((v > 0).mean())}
    return {
        "n_obs": int(len(g)), "n_directional": int(len(gd)),
        "alignment_rate": align, "momentum_gap": mgap,
        "hit_up": hit_up, "hit_dn": hit_dn,
        "kw_pvalue": float(kw.pvalue) if kw else None, "kw_effect": kw_eff,
        "by_phase": by,
    }


def main():
    stock = load_panel()
    bmap, cap_jo = cap_buckets(list(stock.keys()))
    counts = pd.Series(bmap).value_counts().reindex(LABELS).to_dict()
    print("[버킷 개수]", counts)

    result = {"asof": ASOF, "bucket_counts": counts, "by_horizon": {}}
    chart_data = {}

    for h in HORIZONS:
        fwd = build_fwd(stock, h)
        panel = bucket_panel(stock, "Wyckoff_Phase", fwd, bmap)
        per = {}
        for lab in LABELS:
            g = panel[panel["bucket"] == lab]
            if len(g) >= 50:
                per[lab] = analyze_bucket(g)
        # 추세검정: 소형→대형 순서 index vs 지표
        labs_present = [l for l in ORDER_SMALL_TO_BIG if l in per]
        idx = list(range(len(labs_present)))
        trend = {}
        for metric in ["alignment_rate", "momentum_gap", "kw_effect"]:
            ys = [per[l][metric] for l in labs_present]
            if len(ys) >= 3 and not any(np.isnan(ys)):
                rho, p = st.spearmanr(idx, ys)
                trend[metric] = {"spearman_rho": float(rho), "pvalue": float(p)}
        result["by_horizon"][f"h{h}"] = {"per_bucket": per, "trend_small_to_big": trend}
        chart_data[h] = (labs_present, per)

    (OUT / "metrics_h3.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # 콘솔 요약
    print("\n===== 가설3: 시총 체급별 와이코프 정합도 =====")
    for h in HORIZONS:
        print(f"\n--- {h}일 포워드 (raw) ---")
        per = result["by_horizon"][f"h{h}"]["per_bucket"]
        print(f"{'체급':8} {'n':>7} {'정합율':>7} {'모멘텀갭':>8} {'hitUp':>6} {'hitDn':>6} {'KW_p':>9}")
        for lab in LABELS:
            if lab in per:
                v = per[lab]
                print(f"{lab:8} {v['n_obs']:>7} {v['alignment_rate']:>7.3f} "
                      f"{v['momentum_gap']:>+8.3f} {v['hit_up']:>6.3f} {v['hit_dn']:>6.3f} "
                      f"{(v['kw_pvalue'] or 1):>9.1e}")
        tr = result["by_horizon"][f"h{h}"]["trend_small_to_big"]
        print("  추세(소형0→대형5 Spearman):", {k: f"rho={v['spearman_rho']:+.3f},p={v['pvalue']:.3f}"
                                                  for k, v in tr.items()})
    make_chart(chart_data)
    return result


def make_chart(chart_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["axes.unicode_minus"] = False
    UP, DOWN = "#ef5350", "#1976D2"
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, h in zip(axes, HORIZONS):
        labs, per = chart_data[h]
        align = [per[l]["alignment_rate"] for l in labs]
        mgap = [per[l]["momentum_gap"] for l in labs]
        x = np.arange(len(labs))
        ax.axhline(0.5, color="#888", lw=0.8, ls="--")
        bars = ax.bar(x, align, color=[UP if a >= 0.5 else DOWN for a in align], alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(labs, rotation=20)
        ax.set_ylim(0.35, 0.65)
        ax.set_title(f"{h}일 포워드 — 체급별 와이코프 방향 정합율")
        ax.set_ylabel("정합율 (0.5=무관)")
        for xi, a in zip(x, align):
            ax.text(xi, a + 0.004, f"{a:.3f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "h3_capsize_alignment.png", dpi=130); plt.close(fig)
    print(f"[chart] → {OUT/'h3_capsize_alignment.png'}")


if __name__ == "__main__":
    main()
