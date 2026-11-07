# -*- coding: utf-8 -*-
"""shock_tags.py — 일별 종목 쇼크 태그 (인과 — 그날 장 마감까지 정보만). 게이트·알림용 (케인 2026-11-07 22:22).

연구판(stock_gapshock.py)과 같은 정의를 쓰되 미래 정보를 쓰는 제외 규칙(R4 상폐 전 10일·R5 이후 20일 이상 행)은 뺐다.
자료    data_1104/prices (2010~2026-11-04, 전 종목 수정주가) · data_1104/kospi200_daily · KRX 업종 (data_idx, 연말 구성 — 미매핑은 (나) 소급)
쇼크    종가 기준 하루 ≤ −5% · 그날·전일 거래량 > 0 · 시가 > 0 · 제한폭 이내
범위    시장 동반 = 코스피200 ≤ −2% / 섹터 동반 = 시장 아님 & 같은 업종 다른 종목(정상 행, 자기 제외, 5종목 이상) 평균 ≤ −3% / 단독 = 그 밖 / 미상
크기    작은 −5~−7% · 중간 −7~−10% · 큰 ≤−10%
거래량  그날 거래량 ÷ 직전 20거래일 평균(정상 행) — 저 <1.0 · 중 1.0~2.0 · 고 ≥2.0
형태    갭 몫 = ln(1+갭)/ln(1+하루) — ① <0 · ② 0~0.4 · ③ 0.4~0.6 · ④ 0.6~1 · ⑤ >1
저장    out/regime/shock_tags.parquet (쇼크 행만) · 사용 python3 shock_tags.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from stock_gapshock import sector_map
D = HERE / "data_1104"; OUT = HERE / "out" / "regime"

def build():
    p = pd.read_parquet(D / "prices.parquet").reset_index()
    p = p[p.date >= "2014-11-01"].sort_values(["ticker", "date"]).reset_index(drop=True)
    g = p.groupby("ticker", sort=False)
    p["pc"] = g.close.shift(1); pv = g.volume.shift(1)
    p["r"] = p.close / p.pc - 1; p["gap"] = p.open / p.pc - 1
    lim = np.where(p.date < "2015-06-15", 0.1505, 0.3005)
    ok = (pv > 0) & (p.volume > 0) & (p.open > 0) & (p.r.abs() <= lim) & (p.gap.abs() <= lim)
    v = p.volume.where(ok)
    p["vol20"] = v.groupby(p.ticker, sort=False).rolling(20, min_periods=15).mean().reset_index(level=0, drop=True).sort_index()
    p["vol20"] = p.groupby("ticker", sort=False).vol20.shift(1)
    p["year"] = p.date.dt.year
    sm = sector_map()
    p = p.merge(sm[["ticker", "year", "code", "name"]], on=["ticker", "year"], how="left")
    q = p[ok & p.code.notna()]
    agg = q.groupby(["date", "code"]).r.agg(["sum", "count"]).reset_index()
    k = pd.read_parquet(D / "kospi200_daily.parquet").Close.astype(float); k.index = pd.to_datetime(k.index); kr = k.pct_change()
    t = p[ok & (p.r <= -0.05) & (p.date >= "2015-01-01")].copy()
    t = t.merge(agg, on=["date", "code"], how="left")
    t["peer_n"] = t["count"] - 1
    t["peer_r"] = np.where(t.peer_n >= 5, (t["sum"] - t.r) / t.peer_n, np.nan)
    t["k_r"] = t.date.map(kr)
    t["ctx"] = np.select([t.k_r <= -0.02, t.peer_r <= -0.03, t.peer_r.notna()], ["시장 동반", "섹터 동반", "단독"], "미상")
    t["size"] = pd.cut(t.r, [-1, -0.10, -0.07, -0.05], labels=["큰", "중간", "작은"]).astype(str)
    t["vol_n"] = t.volume / t.vol20
    t["volbin"] = pd.cut(t.vol_n, [0, 1.0, 2.0, np.inf], labels=["저", "중", "고"], right=False).astype(str)
    t.loc[t.vol_n.isna(), "volbin"] = "미상"
    t["share"] = np.log1p(t.gap) / np.log1p(t.r)
    t["shape"] = pd.cut(t.share, [-np.inf, 0, .4, .6, 1, np.inf], labels=list("①②③④⑤")).astype(str)
    cols = ["date", "ticker", "r", "gap", "share", "shape", "size", "vol_n", "volbin", "k_r", "peer_r", "peer_n", "code", "name", "ctx", "close", "low"]
    t = t[cols].rename(columns={"name": "sector"}); t.to_parquet(OUT / "shock_tags.parquet")
    print(f"tags: {len(t):,} 쇼크 행 · {t.date.min().date()}~{t.date.max().date()} · 맥락 {t.ctx.value_counts().to_dict()} · 거래량 {t.volbin.value_counts().to_dict()}")

if __name__ == "__main__":
    build()
