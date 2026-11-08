# -*- coding: utf-8 -*-
"""dayport_surprise_check.py — 데이포트 후보에 급등 태그를 붙여 다음 날 갭 비교 (급등 연구 6단계, 케인 2026-11-08 20:06).
dayport_shock_check.py 의 거울상. 후보 = mop_model/build/eh_base_ens.parquet (워크포워드 491일, 2024-09-02~2026-09-10) rank ≤ 10
손익 Gap_T1 = 다음 날 시가 ÷ 신호일 종가 − 1 (v1.1.2.4 09:04 시가 청산 근사), 승률은 왕복 0.23% 초과 기준
태그 surprise_tags.py (인과, 주 정의 ≥ +6%) — 신호일 당일(창 1) / 최근 10거래일(창 10)
구간 학습 2024-09-02~2026-04-30 · 보류 2026-05-01~ (급등 연구 보류 구간)
게이트 = 태그 후보를 빼고 빈 슬롯 현금 / rank 11~15 로 채움"""
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; OUT = HERE / "out" / "regime"
def main():
    e = pd.read_parquet(HERE.parent / "mop_model" / "build" / "eh_base_ens.parquet"); e["Date"] = pd.to_datetime(e.Date)
    t = pd.read_parquet(OUT / "surprise_tags.parquet"); t = t[t.main]
    dates = pd.Index(sorted(e.Date.unique())); cal = pd.Index(sorted(pd.read_parquet(HERE / "data_1104" / "kospi200_daily.parquet").index.map(pd.Timestamp)))
    s1 = t.ctx2 == "단독"
    conds = {"급등 전부": t.ctx2 != "미상", "단독": s1, "단독 큰": s1 & (t["size"] == "큰"), "단독 ⑤": s1 & (t["shape"] == "⑤"),
             "단독 저거래량": s1 & (t.volbin == "저"), "단독 고거래량": s1 & (t.volbin == "고"), "섹터 동반": t.ctx2 == "섹터 동반",
             "시장·반등형": t.ctx2 == "시장·반등형", "시장·순수": t.ctx2 == "시장·순수"}
    pos = pd.Series(np.arange(len(cal)), index=cal); e["pi"] = e.Date.map(pos)
    SPL = (("2024-09-01", "2026-04-30", "학습"), ("2026-05-01", "2026-09-30", "보류"))
    rows, port = [], []
    for win in (1, 10):
        for nm, c in conds.items():
            x = t[c][["date", "ticker"]].copy(); x["ti"] = x.date.map(pos)
            m = e[["Ticker", "pi"]].merge(x.rename(columns={"ticker": "Ticker"})[["Ticker", "ti"]], on="Ticker")
            m = m[(m.ti <= m.pi) & (m.ti > m.pi - win)]
            tagged = set(zip(m.Ticker, m.pi)); e["tag"] = [(a, b) in tagged for a, b in zip(e.Ticker, e.pi)]
            top = e[e["rank"] <= 10]
            for a, b, per in SPL:
                tp = top[(top.Date >= a) & (top.Date <= b)]
                for flag, g in tp.groupby("tag"):
                    rows.append({"창": win, "태그": nm, "구간": per, "후보": "태그 있음" if flag else "태그 없음", "건수": len(g), "날짜": g.Date.nunique(),
                                 "다음날 갭 평균%": g.Gap_T1.mean() * 100, "중앙%": g.Gap_T1.median() * 100, "승률%": (g.Gap_T1 > 0.0023).mean() * 100})
            daily = lambda df: df.groupby("Date").Gap_T1.mean() - 0.0023
            base = daily(top)
            gc = top[~top.tag].groupby("Date").Gap_T1.sum().reindex(dates).fillna(0) / 10 - (top[~top.tag].groupby("Date").size().reindex(dates).fillna(0) / 10) * 0.0023
            fill = e[(e["rank"] <= 15) & ~e.tag].sort_values(["Date", "rank"]).groupby("Date").head(10); gf = daily(fill)
            for lbl, r in (("기준 top10", base), ("게이트(빈 슬롯 현금)", gc), ("게이트+채움(≤15)", gf)):
                r = r.reindex(dates).fillna(0)
                for a, b, per in SPL:
                    rr = r.loc[a:b]; port.append({"창": win, "태그": nm, "포트": lbl, "구간": per, "일평균%": rr.mean() * 100, "누적%": ((1 + rr).prod() - 1) * 100,
                                                  "Sharpe": rr.mean() / rr.std() * np.sqrt(252) if rr.std() > 0 else np.nan})
    R = pd.DataFrame(rows); P = pd.DataFrame(port).drop_duplicates(["창", "태그", "포트", "구간"])
    R.to_csv(OUT / "dayport_surprise_candidates.csv", index=False, encoding="utf-8-sig"); P.to_csv(OUT / "dayport_surprise_port.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    print(R.round(2).to_string(index=False))
    pv = P.pivot_table(index=["창", "태그", "포트"], columns="구간", values=["누적%", "Sharpe"], sort=False).round(2); print(pv.to_string())
    import surprise_xlsx
    Ri = R.set_index(["창", "태그", "구간", "후보"])
    pv.columns = [f"{a} {b}" for a, b in pv.columns]
    surprise_xlsx.write_generic(OUT / "급등연구_6단계_데이포트.xlsx",
        [("급등 연구 6단계 — 데이포트 후보 급등 태그별 다음 날 갭 (eh_base_ens 워크포워드, rank ≤ 10)", True),
         ("창 1 = 신호일 당일 급등 · 창 10 = 최근 10거래일 안 급등 · 승률 = 갭 > 0.23% · 학습 2024-09-02~2026-04-30 / 보류 2026-05-01~", False)],
        [("후보", [("태그 있음 vs 없음", Ri, ["다음날 갭 평균%", "중앙%"], False, None, ["창", "태그", "구간", "후보"])]),
         ("포트", [("기준 vs 게이트 (누적 %, Sharpe)", pv, [c for c in pv.columns if c.startswith("누적")], False, None, ["창", "태그", "포트"])])])
if __name__ == "__main__":
    main()
