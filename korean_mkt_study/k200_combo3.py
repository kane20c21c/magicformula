"""3종목(현대로템·신한지주·HD현대중공업) 10일 수익률 방향 합의 → 다음 날 KOSPI200 방향 (2026-11-23 케인)
셋 다 10일 수익률 > 0 → 상승 예측 / 셋 다 < 0 → 하락 예측 / 섞이면 예측 없음.
⚠ 세 종목은 2024~2026 상관으로 고른 것 → 같은 구간 결과는 표본 안(선택 편향). 2015~2023 은 고르는 데 안 쓴 구간.
"""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "out", "k200_nextday")
T = {"064350": "현대로템", "055550": "신한지주", "329180": "HD현대중공업"}
px = pd.read_parquet(f"{ROOT}/data_2015/prices.parquet", columns=["close"])["close"].unstack("ticker")[list(T)]
k = pd.read_parquet(f"{ROOT}/data_2015/kospi200_daily.parquet")["Close"]; k.index = pd.to_datetime(k.index)
idx = px.index.intersection(k.index); px, k = px.loc[idx], k.loc[idx]
s = np.log(px / px.shift(10)); y = np.log(k.shift(-1) / k)
df = s.rename(columns=T); df["y"] = y; df["k10"] = np.log(k / k.shift(10)); df = df.dropna(subset=["y", "현대로템", "신한지주"])
def evalp(pred, d):
    m = (pred != 0) & (d.y != 0); p, yy = pred[m], d.y[m]
    h = np.sign(p) == np.sign(yy); up = yy > 0
    nu, nd = (p > 0).sum(), (p < 0).sum()
    return dict(일수=len(d), 예측일=int(m.sum()), 예측비율=m.sum() / len(d), 적중률=h.mean(),
                균형적중률=0.5 * (h[up].mean() + h[~up].mean()),
                상승예측일=int(nu), 상승예측_적중=(h[p > 0]).mean() if nu else np.nan,
                하락예측일=int(nd), 하락예측_적중=(h[p < 0]).mean() if nd else np.nan,
                상승예측일_지수평균=yy[p > 0].mean() * 100 if nu else np.nan, 하락예측일_지수평균=yy[p < 0].mean() * 100 if nd else np.nan,
                지수상승비율=(d.y > 0).mean())
rows = []
periods = {"2024~2026.10 (선정 구간)": df.index.year >= 2024, "2021.09~2023 (선정 밖)": (df.index.year <= 2023) & df["HD현대중공업"].notna(),
           "2015~2023 (선정 밖)": df.index.year <= 2023}
for pn, pm in periods.items():
    d = df[pm]
    sig3 = np.sign(d[["현대로템", "신한지주", "HD현대중공업"]])
    all3 = np.where((sig3 > 0).all(1), 1, np.where((sig3 < 0).all(1), -1, 0))
    sig2 = np.sign(d[["현대로템", "신한지주"]]); all2 = np.where((sig2 > 0).all(1), 1, np.where((sig2 < 0).all(1), -1, 0))
    vote = np.sign(sig3.fillna(0).sum(1)).values
    cands = {"3종목 모두 같은 방향": all3, "2종목(로템·신한) 모두 같은 방향": all2, "3종목 다수결(항상 예측)": vote,
             "현대로템 단독": np.sign(d["현대로템"]).values, "신한지주 단독": np.sign(d["신한지주"]).values,
             "HD현대중공업 단독": np.sign(d["HD현대중공업"]).fillna(0).values,
             "(기준) KOSPI200 자기 10일": np.sign(d["k10"]).values, "(기준) 항상 상승 예측": np.ones(len(d))}
    if pn.startswith("2015"):
        for c in ["3종목 모두 같은 방향", "3종목 다수결(항상 예측)", "HD현대중공업 단독"]: cands.pop(c)
    for cn, pr in cands.items():
        rows.append(dict(구간=pn, 규칙=cn, **evalp(pd.Series(pr, index=d.index), d)))
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/combo3.csv", index=False, encoding="utf-8-sig")
# 연도별 (3종목 합의)
yr = []
for y_, d in df[df["HD현대중공업"].notna()].groupby(df[df["HD현대중공업"].notna()].index.year):
    sig3 = np.sign(d[["현대로템", "신한지주", "HD현대중공업"]]); a = np.where((sig3 > 0).all(1), 1, np.where((sig3 < 0).all(1), -1, 0))
    yr.append(dict(연도=y_, **evalp(pd.Series(a, index=d.index), d)))
Y = pd.DataFrame(yr); Y.to_csv(f"{OUT}/combo3_yearly.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(R.round(3).to_string()); print(Y.round(3).to_string())
