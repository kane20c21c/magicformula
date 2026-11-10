# -*- coding: utf-8 -*-
"""stop_random_paths.py — 메가갤럭시 무작위 1종목 T60 기대수익·경로 진폭 (케인 2026-11-05).

모집단  신호일 t 에 메가갤럭시 편입(p.elig)인 모든 (t, 종목) — '무작위로 하나 고른다' 의 기댓값 = 이 전체 분포
체결    t+1 시가 매수 → 보유 61거래일 → t+62 시가 매도 (버팀+T60 포트와 같은 시점) · 비용 왕복 0.23% (손절 슬리피지 없음)
경로    t+1 ~ t+61 종가 / 매수가. 최저(MAE)·최고(MFE) 와 어느 쪽이 먼저 왔는지
비교    스윙 신호 전체(슬롯 제한 없이 모든 신호) · 포트 실제 284건
사용    STOP_UNIV=swing python3 stop_random_paths.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import stop_research as sr, stop_port as sp

H = 61
OUT = HERE / "out" / "stop_research" / ("port_" + sr.UNIV)

def paths(p, mask):
    op = p.open.to_numpy(float); cl = p.close.to_numpy(float); D, N = op.shape
    dates = p.dates; lo = np.searchsorted(dates, pd.Timestamp("2015-01-01")); hi = D - (H + 2)
    m = mask.reindex(index=dates, columns=p.tickers).fillna(False).to_numpy(bool)
    m[:lo] = False; m[hi:] = False
    ti, tj = np.nonzero(m); f = ti + 1
    e = op[f, tj]; x = op[f + H, tj]
    W = np.stack([cl[f + k, tj] for k in range(H)], axis=1) / e[:, None] - 1
    ok = np.isfinite(e) & (e > 0) & np.isfinite(x) & (x > 0) & np.isfinite(W).all(axis=1)
    W = W[ok]; ret = (x[ok] / e[ok] * (1 - sp.SELL_FEE - sp.TAX) / (1 + sp.BUY_FEE) - 1) * 100
    mae = W.min(1) * 100; mfe = W.max(1) * 100; amin = W.argmin(1); amax = W.argmax(1)
    out = pd.DataFrame(dict(date=dates[ti[ok]], ret=ret, mae=mae, mfe=mfe, t_min=amin + 1, t_max=amax + 1))
    for b in (5, 10, 20):
        dn = W <= -b / 100; upm = W >= b / 100
        out[f"dn{b}"] = np.where(dn.any(1), dn.argmax(1), 999); out[f"up{b}"] = np.where(upm.any(1), upm.argmax(1), 999)
    return out

def q(x, a=(.10, .25, .5, .75, .90)): return " / ".join(f"{v:+.1f}" for v in np.quantile(x, a))

def dist(d):
    r = d.ret
    return pd.Series({"표본(종목·일)": len(d), "평균%": r.mean(), "중앙%": r.median(), "표준편차%": r.std(), "플러스%": (r > 0).mean() * 100,
                      "하위5%": r.quantile(.05), "하위10%": r.quantile(.10), "하위25%": r.quantile(.25), "상위25%": r.quantile(.75),
                      "상위10%": r.quantile(.90), "상위5%": r.quantile(.95), "≤−20%": (r <= -20).mean() * 100, "≥+20%": (r >= 20).mean() * 100,
                      "≥+50%": (r >= 50).mean() * 100, "보유중최저 중앙%": d.mae.median(), "보유중최고 중앙%": d.mfe.median(),
                      "진폭(최고−최저) 중앙%": (d.mfe - d.mae).median()})

def shapes(d):
    g = np.select([d.mae >= 0, d.mfe <= 0, d.t_min < d.t_max], ["① 진입가 아래로 한 번도 안 감", "② 진입가 위로 한 번도 못 감", "③ 하락 → 상승 (저점이 먼저)"], "④ 상승 → 하락 (고점이 먼저)")
    rows = []
    for k in sorted(set(g)):
        x = d[g == k]
        rows.append({"모양": k, "비율%": len(x) / len(d) * 100, "최저 10/25/50/75/90%": q(x.mae), "최고 10/25/50/75/90%": q(x.mfe),
                     "최종 10/25/50/75/90%": q(x.ret), "최종 평균%": round(x.ret.mean(), 1), "최종 플러스%": round((x.ret > 0).mean() * 100, 1),
                     "저점일 중앙": int(x.t_min.median()), "고점일 중앙": int(x.t_max.median())})
    return pd.DataFrame(rows).set_index("모양")

def barrier(d, b):
    """종가 기준 ±b% 중 먼저 닿은 쪽 → 그 뒤 최종 수익률."""
    dn, up = d[f"dn{b}"], d[f"up{b}"]
    g = np.select([(dn == 999) & (up == 999), dn < up], ["없음 (±%d%% 안에서만 움직임)" % b, f"−{b}% 먼저"], f"+{b}% 먼저")
    rows = []
    for k in [f"−{b}% 먼저", f"+{b}% 먼저", "없음 (±%d%% 안에서만 움직임)" % b]:
        x = d[g == k]
        if not len(x): continue
        both = ((x[f"up{b}"] < 999) & (x[f"dn{b}"] < 999)).mean() * 100
        rows.append({"먼저 닿은 쪽": k, "비율%": len(x) / len(d) * 100, "반대쪽도 닿음%": both, "최종 평균%": x.ret.mean(), "최종 중앙%": x.ret.median(),
                     "최종 플러스%": (x.ret > 0).mean() * 100, "최종 10/25/50/75/90%": q(x.ret)})
    return pd.DataFrame(rows).set_index("먼저 닿은 쪽")

def main():
    p, s, _, _, atr = sr.load()
    el = p.elig.fillna(False).astype(bool); sb = s.fillna(False).astype(bool)
    print("elig 평균 종목수/일", round(el.loc["2015":].sum(1).mean(), 1), "· 신호 중 elig 밖", int((sb & ~el.reindex_like(sb).fillna(False)).to_numpy().sum()))
    R = paths(p, el); S = paths(p, sb)
    tr = pd.read_csv(OUT / "regime_trades_holdT60.csv")
    P = pd.DataFrame(dict(ret=tr.r, mae=tr.mae, mfe=tr.mfe))
    pd.set_option("display.width", 320); pd.set_option("display.max_colwidth", 60)
    t = pd.DataFrame({"무작위 1종목 (메가갤럭시 전체)": dist(R), "스윙 신호 전체 (슬롯 무제한)": dist(S)})
    pc = dist(P.assign(ret=P.ret)); pc["표본(종목·일)"] = len(P); t["포트 실제 284건"] = pc
    print("== T60 수익률 분포 (비용 차감, %) ==\n", t.round(2).to_string())
    bins = [-100, -30, -20, -10, -5, 0, 5, 10, 20, 30, 50, 1e9]
    h = pd.DataFrame({k: pd.cut(v.ret, bins).value_counts(normalize=True).sort_index() * 100 for k, v in [("무작위", R), ("신호 전체", S), ("포트 284", P)]})
    print("\n== 구간별 비율 (%) ==\n", h.round(1).to_string())
    for nm, d in [("무작위 1종목", R), ("스윙 신호 전체", S)]:
        print(f"\n== [{nm}] 경로 모양 ==\n", shapes(d).to_string())
        for b in (5, 10, 20): print(f"\n-- [{nm}] ±{b}% 먼저 닿기 --\n", barrier(d, b).round(1).to_string())
    OUT.mkdir(parents=True, exist_ok=True); t.to_csv(OUT / "random_paths_dist.csv"); h.to_csv(OUT / "random_paths_hist.csv")
    shapes(R).to_csv(OUT / "random_paths_shapes.csv"); shapes(S).to_csv(OUT / "signal_paths_shapes.csv")

if __name__ == "__main__":
    main()
