# -*- coding: utf-8 -*-
"""dayport_shock_check.py — 데이포트 후보에 쇼크 태그를 붙여 다음 날 시가 손익 비교 (케인 2026-11-07 22:22 "같이 보자").

후보   mop_model/build/eh_base_ens.parquet (현행 y_rel 모델 워크포워드 491일, 2024-09-02~2026-09-10) — rank ≤ 10
손익   Gap_T1 = 다음 날 시가 ÷ 신호일 종가 − 1 (v1.1.2.4: 저녁 매수 → 09:04 시가 전량 청산 근사) · 왕복 0.23% 차감은 표에 따로
태그   shock_tags.py (인과) — 신호일 당일(창 1) / 최근 10거래일(창 10). 게이트 = 해당 후보를 빼고 rank 11~15 로 채우지 않음(빈 슬롯=현금) / 채움 두 가지
"""
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; OUT = HERE / "out" / "regime"
def main():
    e = pd.read_parquet(HERE.parent / "mop_model" / "build" / "eh_base_ens.parquet"); e["Date"] = pd.to_datetime(e.Date)
    t = pd.read_parquet(OUT / "shock_tags.parquet")
    dates = pd.Index(sorted(e.Date.unique())); cal = pd.Index(sorted(pd.read_parquet(HERE / "data_1104" / "kospi200_daily.parquet").index.map(pd.Timestamp)))
    conds = {"단독": t.ctx == "단독", "저거래량 단독": (t.ctx == "단독") & (t.volbin == "저"), "고거래량 단독": (t.ctx == "단독") & (t.volbin == "고"),
             "큰 단독": (t.ctx == "단독") & (t["size"] == "큰"), "섹터동반 ①": (t.ctx == "섹터 동반") & (t["shape"] == "①"), "시장 동반": t.ctx == "시장 동반", "쇼크 전부": t.ctx != "미상"}
    pos = pd.Series(np.arange(len(cal)), index=cal)
    rows, port = [], []
    for win in (1, 10):
        for nm, c in conds.items():
            x = t[c][["date", "ticker"]].copy(); x["pi"] = x.date.map(pos)
            # 창 안 태그: 신호일 d 의 위치 pd 에 대해 pi ∈ [pd−win+1, pd]
            e["pi"] = e.Date.map(pos)
            m = e[["Ticker", "pi"]].merge(x.rename(columns={"ticker": "Ticker"})[["Ticker", "pi"]].rename(columns={"pi": "ti"}), on="Ticker")
            m = m[(m.ti <= m.pi) & (m.ti > m.pi - win)]
            tagged = set(zip(m.Ticker, m.pi)); e[f"tag"] = [(a, b) in tagged for a, b in zip(e.Ticker, e.pi)]
            top = e[e["rank"] <= 10]
            for flag, g in top.groupby("tag"):
                rows.append({"창": win, "태그": nm, "후보": "태그 있음" if flag else "태그 없음", "건수": len(g), "날짜": g.Date.nunique(),
                             "다음날 갭 평균%": g.Gap_T1.mean() * 100, "중앙%": g.Gap_T1.median() * 100, "승률%": (g.Gap_T1 > 0.0023).mean() * 100})
            # 포트: 기준(top10 동일가중) vs 게이트(태그 빼고 현금) vs 게이트+채움(rank 11~15 에서 태그 없는 순)
            def daily(df):
                return df.groupby("Date").Gap_T1.mean() - 0.0023
            base = daily(top)
            gc = top[~top.tag].groupby("Date").Gap_T1.sum().reindex(dates).fillna(0) / 10 - (top[~top.tag].groupby("Date").size().reindex(dates).fillna(0) / 10) * 0.0023
            fill = e[(e["rank"] <= 15) & ~e.tag].sort_values(["Date", "rank"]).groupby("Date").head(10); gf = daily(fill)
            for lbl, r in (("기준 top10", base), ("게이트(빈 슬롯 현금)", gc), ("게이트+채움(≤15)", gf)):
                r = r.reindex(dates).fillna(0)
                for a, b, per in (("2024-09-01", "2026-05-29", "학습"), ("2026-06-01", "2026-09-30", "보류")):
                    rr = r.loc[a:b]; port.append({"창": win, "태그": nm, "포트": lbl, "구간": per, "일평균%": rr.mean() * 100, "누적%": ((1 + rr).prod() - 1) * 100,
                                                  "Sharpe": rr.mean() / rr.std() * np.sqrt(252) if rr.std() > 0 else np.nan})
    R = pd.DataFrame(rows); P = pd.DataFrame(port)
    R.to_csv(OUT / "dayport_shock_candidates.csv", index=False, encoding="utf-8-sig"); P.to_csv(OUT / "dayport_shock_port.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    print(R.round(2).to_string(index=False))
    P = P.drop_duplicates(["창", "태그", "포트", "구간"]); print(P.pivot_table(index=["창", "태그", "포트"], columns="구간", values=["누적%", "Sharpe"], sort=False).round(2).to_string())
if __name__ == "__main__":
    main()
