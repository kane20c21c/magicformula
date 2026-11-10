# -*- coding: utf-8 -*-
"""regime_gapshock.py — 하루 −3% 쇼크일: 갭 하락 주도 vs 장중 하락 주도, 어느 쪽 영향이 오래 가나 (케인 2026-11-06 21:40).

자료    코스피 200 일봉 2000-01-04 ~ 2026-05-29 (2000~2014 pykrx 1028 → data_idx/kospi200_2000_2014.parquet, 2015~ data_1104). 보류 구간 미사용
쇼크일  종가 기준 하루 ≤ −3%
분해    갭 g = 시가/전일종가 − 1, 장중 i = 종가/시가 − 1  (1+r = (1+g)(1+i))
        갭 몫 = ln(1+g) / ln(1+r)  → 갭 주도 ≥ 0.6 · 장중 주도 ≤ 0.4 · 섞임
지속    ① 방향: 쇼크일 종가 대비 +1·3·5·10·20일 수익률 (평균·중앙·하락 비율)
        ② 변동성: 이후 1~5일 · 6~20일 평균 일중 폭(고가−저가)/전일종가 ÷ 쇼크 전 20일 평균 폭
        ③ 회복: 쇼크 전날 종가를 종가로 되찾기까지 거래일 (20·60일 안 회복 비율) · 장중 고가로 전날 종가를 터치(갭 메움)한 비율
표본    ⓐ 모든 쇼크일  ⓑ 앞 10거래일 안에 쇼크가 없던 첫 쇼크일만 (연쇄 쇼크 섞임 제거)
사용    python3 regime_gapshock.py [임계=-0.03]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; OUT = HERE / "out" / "regime"; END = pd.Timestamp("2026-05-29")

def load():
    a = pd.read_parquet(HERE / "data_idx" / "kospi200_2000_2014.parquet")[["Open", "High", "Low", "Close"]]
    b = pd.read_parquet(HERE / "data_1104" / "kospi200_daily.parquet")[["Open", "High", "Low", "Close"]]
    d = pd.concat([a, b[b.index > a.index.max()]]).astype(float).sort_index(); d.index = pd.to_datetime(d.index)
    return d.loc[:END]

def build(d, th):
    c, o, h, l = d.Close, d.Open, d.High, d.Low; cp = c.shift(1)
    r = c / cp - 1; g = o / cp - 1; i = c / o - 1; rng = (h - l) / cp
    share = np.log1p(g) / np.log1p(r)
    rng20 = rng.rolling(20).mean().shift(1)
    idx = np.flatnonzero((r <= th).to_numpy()); n = len(d); rows = []; prev_shock = -999
    for t in idx:
        first = (t - prev_shock) > 10; prev_shock = t
        if t + 20 >= n: continue
        row = dict(date=d.index[t], r=r.iloc[t] * 100, gap=g.iloc[t] * 100, intra=i.iloc[t] * 100, share=share.iloc[t],
                   gap_n=abs(g.iloc[t]) / (abs(g).rolling(20).mean().shift(1).iloc[t]), rng_n=rng.iloc[t] / rng20.iloc[t], first=first)
        for k in (1, 3, 5, 10, 20): row[f"+{k}일"] = (c.iloc[t + k] / c.iloc[t] - 1) * 100
        row["다음날 갭"] = g.iloc[t + 1] * 100
        row["변동 1~5일"] = rng.iloc[t + 1:t + 6].mean() / rng20.iloc[t]; row["변동 6~20일"] = rng.iloc[t + 6:t + 21].mean() / rng20.iloc[t]
        pre = cp.iloc[t]; seg = c.iloc[t + 1:min(n, t + 61)].to_numpy(); kk = np.flatnonzero(seg >= pre)
        row["회복일"] = int(kk[0] + 1) if kk.size else np.nan
        row["20일내 회복"] = bool(kk.size and kk[0] + 1 <= 20); row["60일내 회복"] = bool(kk.size) and (t + 60 < n)
        hh = h.iloc[t + 1:t + 6].to_numpy(); row["5일내 전날종가 터치"] = bool((hh >= pre).any())
        rows.append(row)
    D = pd.DataFrame(rows)
    D["유형"] = np.select([D.share >= 0.6, D.share <= 0.4], ["갭 주도", "장중 주도"], "섞임")
    return D

def summ(D):
    f = {"건수": ("r", "size"), "쇼크일 평균%": ("r", "mean"), "갭 평균%": ("gap", "mean"), "장중 평균%": ("intra", "mean"),
         "일중폭 배수": ("rng_n", "median")}
    out = D.groupby("유형").agg(**f)
    for k in ("+1일", "+3일", "+5일", "+10일", "+20일"):
        out[f"{k} 중앙%"] = D.groupby("유형")[k].median(); out[f"{k} 하락%"] = D.groupby("유형")[k].apply(lambda x: (x < 0).mean() * 100)
    out["다음날 갭 중앙%"] = D.groupby("유형")["다음날 갭"].median()
    out["변동 1~5일 배수"] = D.groupby("유형")["변동 1~5일"].median(); out["변동 6~20일 배수"] = D.groupby("유형")["변동 6~20일"].median()
    out["회복일 중앙"] = D.groupby("유형")["회복일"].apply(lambda x: x.fillna(61).median())
    out["20일내 회복%"] = D.groupby("유형")["20일내 회복"].mean() * 100; out["5일내 전날종가 터치%"] = D.groupby("유형")["5일내 전날종가 터치"].mean() * 100
    return out.reindex(["갭 주도", "섞임", "장중 주도"])

def test(D, col):
    try:
        from scipy.stats import mannwhitneyu
        a, b = D[D.유형 == "갭 주도"][col].dropna(), D[D.유형 == "장중 주도"][col].dropna()
        return mannwhitneyu(a, b).pvalue if len(a) > 2 and len(b) > 2 else np.nan
    except Exception: return np.nan

def main(th=-0.03):
    d = load(); D = build(d, th); pd.set_option("display.width", 320)
    print(f"코스피 200 {d.index[0].date()} ~ {d.index[-1].date()} · 하루 ≤ {th*100:.1f}% 쇼크일 {len(D)}개 (20일 경로 완결분)")
    print("연도별:", D.date.dt.year.value_counts().sort_index().to_dict())
    for lbl, X in (("ⓐ 모든 쇼크일", D), ("ⓑ 연쇄 아닌 첫 쇼크일", D[D["first"]])):
        S = summ(X); print(f"\n== {lbl} ==\n", S.T.round(2).to_string())
        print("  갭 주도 vs 장중 주도 차이 검정(맨-휘트니 p):", {c: round(test(X, c), 3) for c in ("+5일", "+20일", "변동 1~5일", "변동 6~20일", "회복일")})
        S.to_csv(OUT / f"gapshock_{'all' if lbl.startswith('ⓐ') else 'first'}.csv")
    D.to_csv(OUT / "gapshock_days.csv", index=False)

if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else -0.03)
