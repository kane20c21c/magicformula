# -*- coding: utf-8 -*-
"""second_drop.py — 단독 쇼크 뒤 '2차 하락' 다음 경로 (케인 2026-11-07 22:22: "2차 하락 후에는 상승하는지, 주목하고 지켜봐야 하는지").

대상   stock_shock_events_ctx.parquet 의 종목 쇼크 (≤−5%, 연구판 유니버스·제외 규칙) — 학습 ~2026-05 / 보류 2026-06~
2차 하락 D2 (케인 동의) = 쇼크 다음 날부터 10거래일 안에 종가가 쇼크일 장중 저가를 처음 깬 날
       (확인용 D2' = 10거래일 안에 다시 하루 ≤ −5% 인 첫날)
측정   D2 다음 날 시가 진입(왕복 0.23% 차감) → +5·+10·+20일 종가 수익 · 코스피200 대비 초과 ·
       20일 안 쇼크 전날 종가 회복 · D2 저가 10일 유지(종가) · D2 종가 → 이후 10일 최저 종가(추가 하락)
비교   저거래량 단독(케인 관심) vs 중·고거래량 단독 / 같은 저거래량 단독 중 D2 없던 쇼크(쇼크 다음 날 시가 진입) / 시장 동반 D2
"""
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; OUT = HERE / "out" / "regime"; COST = 0.0023
def main():
    ev = pd.read_parquet(OUT / "stock_shock_events_ctx.parquet"); ev = ev[ev.ctx != "맥락 미상"]
    ev["V"] = pd.cut(ev.vol_n, [0, 1.0, 2.0, np.inf], labels=["저", "중", "고"], right=False).astype(str)
    p = pd.read_parquet(HERE / "data_2015" / "prices.parquet").reset_index().sort_values(["ticker", "date"]).reset_index(drop=True)
    k = pd.read_parquet(HERE / "data_2015" / "kospi200_daily.parquet").Close.astype(float); k.index = pd.to_datetime(k.index)
    p["k"] = p.date.map(k); n_after = p.groupby("ticker").cumcount(ascending=False).to_numpy()
    key = pd.Series(np.arange(len(p)), index=pd.MultiIndex.from_frame(p[["ticker", "date"]]))
    pos = key.reindex(pd.MultiIndex.from_frame(ev[["ticker", "date"]])).to_numpy()
    O, H, L, C, K = (p[c].to_numpy(float) for c in ("open", "high", "low", "close", "k"))
    R = np.r_[np.nan, C[1:] / C[:-1] - 1]
    rows = []
    for (_, e), i in zip(ev.iterrows(), pos):
        if np.isnan(i): continue
        i = int(i)
        if n_after[i] < 32: continue
        lo, pc = e.low, e.pc
        seg = C[i + 1:i + 11]; hit = np.flatnonzero(seg < lo)
        hit2 = np.flatnonzero(R[i + 1:i + 11] <= -0.05)
        base = dict(date=e.date, ticker=e.ticker, ctx=e.ctx, V=e.V, S=e.S if "S" in e else None, hold=e.hold, r=e.r)
        for kind, h in (("D2", hit), ("D2'", hit2)):
            if h.size:
                d = i + 1 + h[0]; ent = O[d + 1]
                if not np.isfinite(ent) or ent <= 0: continue
                fut = C[d + 1:d + 21]; rec = np.flatnonzero(fut >= pc)
                rows.append({**base, "종류": kind, "D2까지 일": h[0] + 1, "D2 누적(쇼크 종가 대비)": C[d] / C[i] - 1,
                             "f5": C[d + 5] / ent - 1 - COST, "f10": C[d + 10] / ent - 1 - COST, "f20": C[d + 20] / ent - 1 - COST,
                             "x20": (C[d + 20] / ent - 1) - (K[d + 20] / K[d + 1] - 1), "rec20": rec.size > 0,
                             "hold10": C[d + 1:d + 11].min() >= L[d], "dd10": C[d + 1:d + 11].min() / C[d] - 1})
            elif kind == "D2":
                ent = O[i + 1]
                if np.isfinite(ent) and ent > 0:
                    rows.append({**base, "종류": "D2 없음", "D2까지 일": np.nan, "D2 누적(쇼크 종가 대비)": np.nan,
                                 "f5": C[i + 5] / ent - 1 - COST, "f10": C[i + 10] / ent - 1 - COST, "f20": C[i + 20] / ent - 1 - COST,
                                 "x20": (C[i + 20] / ent - 1) - (K[i + 20] / K[i + 1] - 1), "rec20": np.flatnonzero(C[i + 1:i + 21] >= pc).size > 0,
                                 "hold10": C[i + 1:i + 11].min() >= L[i], "dd10": C[i + 1:i + 11].min() / C[i] - 1})
    d = pd.DataFrame(rows); d["era"] = pd.cut(d.date.dt.year, [2014, 2019, 2022, 2026], labels=["15–19", "20–22", "23–26.5"]).astype(str)
    d.to_parquet(OUT / "second_drop_events.parquet")
    def tab(x):
        g = x.groupby(["ctx", "V", "종류"], observed=True)
        return pd.DataFrame({"건수": g.size(), "날짜": g.date.nunique(), "D2까지 일(중앙)": g["D2까지 일"].median(),
            "D2 누적%": g["D2 누적(쇼크 종가 대비)"].median() * 100, "+5일%": g.f5.median() * 100, "+10일%": g.f10.median() * 100,
            "+20일%": g.f20.median() * 100, "+20일 평균%": g.f20.mean() * 100, "+20 승률%": (g.f20 > 0).mean() * 100 if False else g.f20.apply(lambda s: (s > 0).mean() * 100),
            "+20 초과%": g.x20.median() * 100, "20일 안 회복%": g.rec20.mean() * 100, "D2 저가 유지%": g.hold10.mean() * 100, "추가 하락%": g.dd10.median() * 100})
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 200)
    tr = d[~d.hold]; ho = d[d.hold]
    A = tab(tr[tr.ctx.isin(["단독", "시장 동반"])]); print("== 학습 2015~2026-05\n", A.round(1).to_string())
    # 시기별: 단독 저거래량 D2 vs 단독 고거래량 D2 vs 단독 저거래량 D2 없음
    Bx = tr[(tr.ctx == "단독") & (tr.종류 != "D2'")].groupby(["era", "V", "종류"]).agg(건수=("f20", "size"), 중앙20=("f20", "median"), 초과20=("x20", "median"), 회복=("rec20", "mean"))
    Bx[["중앙20", "초과20"]] *= 100; Bx["회복"] *= 100; print("\n== 단독 · 시기별\n", Bx.round(1).to_string())
    C2 = tab(ho[ho.ctx.isin(["단독", "시장 동반"])]); print("\n== 보류 2026-06~ (참고)\n", C2.round(1).to_string())
    with pd.ExcelWriter(OUT / "second_drop.xlsx") as w:
        A.round(2).reset_index().to_excel(w, "학습", index=False); Bx.round(2).reset_index().to_excel(w, "단독_시기별", index=False); C2.round(2).reset_index().to_excel(w, "보류", index=False)
if __name__ == "__main__":
    main()
