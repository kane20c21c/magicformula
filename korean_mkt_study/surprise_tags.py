# -*- coding: utf-8 -*-
"""surprise_tags.py — 일별 종목 급등 태그 (인과 — 그날 장 마감까지 정보만). 게이트·보유 규칙·알림용 (케인 2026-11-08 20:06 6단계).
shock_tags.py 의 거울상. 연구판(stock_surprise.py)과 같은 정의, 미래 정보를 쓰는 제외 규칙(R4·R5)은 뺐다.
자료    data_1104/prices (~2026-11-04) · data_1104/kospi200_daily · KRX 업종 (data_idx, 연말 구성, 미매핑 (나) 소급)
급등    종가 기준 하루 ≥ +5% 행 저장 (주 정의는 ≥ +6% — main 열) · 그날·전일 거래량 > 0 · 시가 > 0 · 제한폭 이내
범위    시장 동반 = 코스피200 ≥ +2% (반등형 = 직전 10거래일 안 코스피200 ≤ −2% 있음 / 순수) · 섹터 동반 = 업종 동료 평균 ≥ +3% · 단독 · 미상
크기    작은 +6~+9 · 중간 +9~+14 · 큰 ≥ +14% (주, 같은 드묾)   거래량  저 <1.5 · 중 1.5~3 · 고 ≥3 배   형태 ①~⑤ (⑤ = 갭상승→밀림)
저장    out/regime/surprise_tags.parquet"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from stock_gapshock import sector_map
D = HERE / "data_1104"; OUT = HERE / "out" / "regime"

def build():
    p = pd.read_parquet(D / "prices.parquet").reset_index()
    p = p[p.date >= "2014-11-01"].sort_values(["ticker", "date"]).reset_index(drop=True)
    g = p.groupby("ticker", sort=False); p["pc"] = g.close.shift(1); pv = g.volume.shift(1)
    p["r"] = p.close / p.pc - 1; p["gap"] = p.open / p.pc - 1
    lim = np.where(p.date < "2015-06-15", 0.1505, 0.3005)
    ok = (pv > 0) & (p.volume > 0) & (p.open > 0) & (p.r.abs() <= lim) & (p.gap.abs() <= lim)
    v = p.volume.where(ok)
    p["vol20"] = v.groupby(p.ticker, sort=False).rolling(20, min_periods=15).mean().reset_index(level=0, drop=True).sort_index()
    p["vol20"] = p.groupby("ticker", sort=False).vol20.shift(1); p["year"] = p.date.dt.year
    p = p.merge(sector_map()[["ticker", "year", "code", "name"]], on=["ticker", "year"], how="left")
    agg = p[ok & p.code.notna()].groupby(["date", "code"]).r.agg(["sum", "count"]).reset_index()
    k = pd.read_parquet(D / "kospi200_daily.parquet").Close.astype(float); k.index = pd.to_datetime(k.index); kr = k.pct_change()
    kmin10 = kr.shift(1).rolling(10, min_periods=1).min()
    t = p[ok & (p.r >= 0.05) & (p.date >= "2015-01-01")].copy().merge(agg, on=["date", "code"], how="left")
    t["peer_n"] = t["count"] - 1; t["peer_r"] = np.where(t.peer_n >= 5, (t["sum"] - t.r) / t.peer_n, np.nan)
    t["k_r"] = t.date.map(kr); t["pre_mkt"] = t.date.map(kmin10) <= -0.02
    t["ctx"] = np.select([t.k_r >= 0.02, t.peer_r >= 0.03, t.peer_r.notna()], ["시장 동반", "섹터 동반", "단독"], "미상")
    t["ctx2"] = np.where(t.ctx == "시장 동반", np.where(t.pre_mkt, "시장·반등형", "시장·순수"), t.ctx)
    t["main"] = t.r >= 0.06
    t["size"] = pd.cut(t.r, [0.05, 0.06, 0.09, 0.14, 1], right=False, labels=["(+5~+6)", "작은", "중간", "큰"]).astype(str)
    t["vol_n"] = t.volume / t.vol20
    t["volbin"] = pd.cut(t.vol_n, [0, 1.5, 3, np.inf], right=False, labels=["저", "중", "고"]).astype(str); t.loc[t.vol_n.isna(), "volbin"] = "미상"
    t["share"] = np.log1p(t.gap) / np.log1p(t.r)
    t["shape"] = pd.cut(t.share, [-np.inf, 0, .4, .6, 1, np.inf], right=False, labels=list("①②③④⑤")).astype(str)
    cols = ["date", "ticker", "r", "gap", "share", "shape", "size", "main", "vol_n", "volbin", "k_r", "pre_mkt", "peer_r", "peer_n", "code", "name", "ctx", "ctx2", "open", "high", "low", "close"]
    t = t[cols].rename(columns={"name": "sector"}); t.to_parquet(OUT / "surprise_tags.parquet")
    m = t[t.main]
    print(f"tags: {len(t):,} 행 (주 정의 ≥+6% {len(m):,}) · {t.date.min().date()}~{t.date.max().date()} · 맥락 {m.ctx2.value_counts().to_dict()} · 거래량 {m.volbin.value_counts().to_dict()}")

if __name__ == "__main__":
    build()
