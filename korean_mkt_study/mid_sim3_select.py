# -*- coding: utf-8 -*-
"""mid_sim3_select.py — 급락 모듈 슬롯 선택의 결정론적 대체 확인
'무작위'는 규칙이 못 되므로: 그날 후보에서 vol20 상위 25% + 거래대금 하위 25% 제외 후 종목코드순 / 거래대금순 / vol순 / 무작위(시드 3)를
30슬롯 · 손절 슬리피지 1% · 당일~20 창으로 비교. 산출: out/mid/sim3_select.csv + 로그
"""
import numpy as np, pandas as pd
import mid_gate as g, mid_sim as ms

df = pd.read_parquet(g.OUT / "states2.parquet")
W, mktcap, foreign, managed, static, sector, fw = g.load(False)
C = W["close"]; dates = C.index
r_path, h, lab, en_arr, close = ms.labels_with_hold(W)
vol = g.pc(C).rolling(20, min_periods=10).std().to_numpy()
amt = (C * W["volume"]).rolling(20, min_periods=10).mean().to_numpy()
di = pd.Series(np.arange(len(dates)), index=dates); ti = pd.Series(np.arange(len(C.columns)), index=C.columns)
df["di"] = di.reindex(df.date).to_numpy(); df["ti"] = ti.reindex(df.ticker).to_numpy()
df = df.dropna(subset=["di", "ti"]); df["di"] = df.di.astype(int); df["ti"] = df.ti.astype(int)
df["vol20n"] = vol[df.di, df.ti]; df["amt20n"] = amt[df.di, df.ti]; df["h"] = h[df.di, df.ti]
df = df.dropna(subset=["h"])
b = df[df.mkt_200.isin(["아래+상승", "아래+하락"]) & df.dd20.isin(["<-20", "-20~-12"]) & df.crash_days.isin(["당일", "1~5", "6~20"])
       & (df.date >= g.EVAL_START)].rename(columns={"rpath10": "r_path", "lab10": "lab"}).copy()
qv = b.groupby("date").vol20n.transform(lambda x: x.quantile(.75)); qa = b.groupby("date").amt20n.transform(lambda x: x.quantile(.25))
b = b[(b.vol20n < qv) & (b.amt20n > qa)]; b["rule"] = "R2"; b["prio"] = 0
d0 = int(di[dates[dates >= g.EVAL_START][0]])
g.log(f"후보 {len(b):,}셀 · {len(b)/b.date.nunique():.0f}/일 · 셀평균 {b.r_path.mean():+.2%}")
rows = []


def run(key, name, slots=30, slip=0.01):
    s = b.copy(); s["key"] = key(s)
    nav, tr = ms.simulate(s, dates, close, ti, en_arr, slots=slots, stop_slip=slip); nav = nav.iloc[d0:]; nav = nav / nav.iloc[0]
    m = ms.metrics(nav, tr, s.date.nunique(), len(nav), tr.h.sum() / slots / len(nav))
    nb = nav[nav.index.year >= 2021]; nb = nb / nb.iloc[0]; m["mdd_B"] = (nb / nb.cummax() - 1).min(); m["name"] = name
    rows.append(m)
    g.log(f"{name:26s} CAGR {m['cagr']:+.1%} MDD {m['mdd']:.1%} Sharpe {m['sharpe']:.2f} 승률 {m['win']:.2f} avg {m['avg_r']:+.2%} 거래 {m['trades']} 노출 {m['slot_use']:.1%} | 후반 MDD {m['mdd_B']:.1%}")


run(lambda s: s.ticker.astype(str), "종목코드순")
run(lambda s: -s.amt20n, "거래대금 큰 순")
run(lambda s: s.amt20n, "거래대금 작은 순")
run(lambda s: s.vol20n, "vol20 낮은 순")
run(lambda s: -s.vol20n, "vol20 높은 순(상한 안에서)")
for sd in range(3):
    run(lambda s, sd=sd: np.random.default_rng(sd).random(len(s)), f"무작위 seed{sd}")
run(lambda s: s.ticker.astype(str), "종목코드순 · 20슬롯", slots=20)
pd.DataFrame(rows).to_csv(g.OUT / "sim3_select.csv", index=False)
g.log("완료")
