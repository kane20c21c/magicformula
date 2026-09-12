# -*- coding: utf-8 -*-
"""
entry_eval.py — entry_hypotheses.py 결과(build/eh_*.parquet) 비교표.
판정 기준은 entry_hypotheses.py 헤더에 사전 등록된 것 그대로 (결과 보고 바꾸지 않는다).

  python3 entry_eval.py                    # build/eh_*.parquet 전부
  python3 entry_eval.py --split 2026-08-18 # 최근 구간 기준일 (기본 2026-08-18)
"""
import argparse, glob, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as cfg  # noqa: E402


def daily(R):
    rows = []
    for D, g in R.dropna(subset=["Gap_T1"]).groupby("Date"):
        if len(g) < 50: continue
        ug = g.Gap_T1.mean(); t10 = g[g["rank"] <= 10]; t5 = g[g["rank"] <= 5]
        rows.append({"d": D, "ic": spearmanr(g.p, g.Gap_T1).correlation,
                     "ex10": (t10.Gap_T1.mean() - ug) * 100, "gap5": t5.Gap_T1.mean() * 100,
                     "intra5": (t5.close_T1 / t5.open_T1 - 1).mean() * 100,
                     "ret5": (t5.close_T1 / t5.Close - 1).mean() * 100,
                     "uret": (g.close_T1 / g.Close - 1).mean() * 100,
                     "sec_max10": t10.sector_top.value_counts().iloc[0] if len(t10) else np.nan})
    return pd.DataFrame(rows)


def summarize(W):
    roll = W.set_index("d").ic.rolling(18).mean() if len(W) >= 18 else pd.Series(dtype=float)
    return {"days": len(W), "ic": W.ic.mean(), "ic_neg%": (W.ic < 0).mean() * 100,
            "roll18_min": roll.min() if len(roll) else np.nan,
            "ex10": W.ex10.mean(), "lose10%": (W.ex10 < 0).mean() * 100,
            "gap5": W.gap5.mean(), "intra5": W.intra5.mean(), "ret5": W.ret5.mean(),
            "ret5-uni": (W.ret5 - W.uret).mean(), "sec_max10": W.sec_max10.mean()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="2026-08-18"); a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(cfg.OUT_DIR, "eh_*.parquet")))
    if not files: sys.exit("build/eh_*.parquet 없음")
    res = {}
    for f in files:
        R = pd.read_parquet(f); R["Date"] = pd.to_datetime(R.Date)
        h = os.path.basename(f)[3:-8]          # 파일명 기준 라벨 (eh_mkt_ens → mkt_ens): 같은 hyp 의 LGBM/앙상블 결과가 덮어쓰지 않게
        W = daily(R); W["d"] = pd.to_datetime(W.d)
        res[h] = {"full": summarize(W), "post": summarize(W[W.d >= a.split]), "pre": summarize(W[W.d < a.split])}
    for per in ["full", "pre", "post"]:
        print(f"\n== {per} ==")
        print(pd.DataFrame({h: v[per] for h, v in res.items()}).T.round(3).to_string())
    bases = [h for h in res if h.startswith("base")]
    for bh in bases:
        suf = bh[4:]; b = res[bh]; print(f"\n== 사전 등록 판정 ({bh} 대비, 같은 접미사끼리) ==")
        for h, v in res.items():
            if h.startswith("base") or h[h.find("_"):] != suf and not (suf == "" and "_" not in h): continue
            c1 = v["full"]["ic"] >= b["full"]["ic"] - 0.010
            c2 = v["full"]["ex10"] >= b["full"]["ex10"]
            c3 = v["post"]["ic"] > b["post"]["ic"]
            c4 = v["full"]["ic_neg%"] <= b["full"]["ic_neg%"] + 2.0
            c5 = v["full"]["roll18_min"] >= b["full"]["roll18_min"] - 0.010
            print(f"  {h:7s} v1: 전구간IC {'O' if c1 else 'X'} · 초과갭 {'O' if c2 else 'X'} · 최근IC {'O' if c3 else 'X'} → "
                  f"{'채택 후보' if (c1 and c2 and c3) else '기각'}  |  v2(+음수일 {'O' if c4 else 'X'} · 롤링최저 {'O' if c5 else 'X'}) → "
                  f"{'채택 후보' if (c1 and c2 and c3 and c4 and c5) else '기각'}")
