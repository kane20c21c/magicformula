"""3일차 상승 드라이브의 초상화 (2026-11-04 케인) — k=3 시점에 서 있는 모든 드라이브를 D0..D3 정보로 묘사하고
다음날 지속(D4 상승) 여부로 가른다. 출력 out/crash/drive_story_k3.parquet + drive_story_k3_YYYYMMDD.xlsx"""
from __future__ import annotations
import sys, time, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from crash_study import DATA, OUT, eligibility
from drive_types import add_daily_fields, load_ohlcv, classify

K = 3

def build() -> pd.DataFrame:
    t0 = time.time()
    px = load_ohlcv(DATA / "prices.parquet"); px = add_daily_fields(px, +1)
    g = px.groupby("ticker", sort=False, observed=True)
    for w in (5, 10, 20, 60):
        px[f"ma{w}"] = g.close.transform(lambda s, w=w: s.rolling(w, min_periods=w).mean()).astype("float32")
    px["rng20"] = (px.high - px.low).groupby(px.ticker, observed=True).transform(lambda s: s.rolling(20, min_periods=15).mean()).astype("float32")
    i3 = np.flatnonzero((px.run.values == K) & px.next_r1.notna().values)
    i0, i1, i2 = i3 - 3, i3 - 2, i3 - 1
    tk = px.ticker.values
    ok = (i0 >= 0); i3, i0, i1, i2 = i3[ok], i0[ok], i1[ok], i2[ok]
    ok = (tk[i0] == tk[i3]) & px.valid.values[i1] & px.valid.values[i2] & px.valid.values[i3] & (px.close.values[i0] > 0)
    i3, i0, i1, i2 = i3[ok], i0[ok], i1[ok], i2[ok]
    O, H, L, C, V = (px[c].values.astype("float64") for c in ("open", "high", "low", "close", "volume"))
    r1 = px.r1.values.astype("float64"); ov = px.ov.values.astype("float64")
    d = pd.DataFrame({"ticker": tk[i3].astype(str), "date": px.date.values[i3], "date_d1": px.date.values[i1]})
    d["year"] = d.date.dt.year.astype("int16")
    d["cont"] = px.next_r1.values[i3] > 0                      # 다음날도 상승 = 지속
    d["next_r1"] = px.next_r1.values[i3]; d["r10"] = px.c10.values[i3] / C[i3] - 1
    d["tot"] = C[i3] / C[i0] - 1
    # 3변수 (D1..D3)
    d["speed"] = np.abs(r1[i3]) / np.abs(r1[i1]); d["v_trend"] = V[i3] / V[i1]; d["overlap"] = (ov[i2] + ov[i3]) / 2
    d["v_level"] = (V[i1] + V[i2] + V[i3]) / 3 / px.v20.values[i0]; d["v_last"] = V[i3] / px.v20.values[i0]
    for key in ("speed", "v_trend", "overlap"): d[f"{key}_c"] = classify(d[key], key, +1)
    # 봉 구조
    for j, (ia, ip) in enumerate(((i1, i0), (i2, i1), (i3, i2)), 1):
        d[f"gap{j}"] = O[ia] / C[ip] - 1                                  # 시가 갭
        d[f"intra{j}"] = C[ia] / O[ia] - 1                                 # 장중(시가→종가)
        rg = H[ia] - L[ia]
        d[f"clv{j}"] = np.where(rg > 0, (C[ia] - L[ia]) / rg, np.nan)      # 종가 위치 0~1
        d[f"ushadow{j}"] = np.where(rg > 0, (H[ia] - np.maximum(O[ia], C[ia])) / rg, np.nan)
        d[f"hh{j}"] = H[ia] > H[ip]; d[f"hl{j}"] = L[ia] > L[ip]            # 고가 갱신 / 저가 상승
        d[f"body_up{j}"] = C[ia] > O[ia]
        d[f"rngx{j}"] = rg / px.rng20.values[i0]                           # 봉 길이 / 직전 20일 평균 봉
    d["gap_up_n"] = (d.gap1 > 0).astype(int) + (d.gap2 > 0).astype(int) + (d.gap3 > 0).astype(int)
    d["hh_n"] = d.hh2.astype(int) + d.hh3.astype(int); d["hl_n"] = d.hl2.astype(int) + d.hl3.astype(int)
    d["body_up_n"] = d.body_up1.astype(int) + d.body_up2.astype(int) + d.body_up3.astype(int)
    d["gap_share"] = ((1 + d.gap1) * (1 + d.gap2) * (1 + d.gap3) - 1) / d.tot.replace(0, np.nan)
    # 이동평균 위치
    for w in (5, 10, 20, 60):
        ma = px[f"ma{w}"].values
        d[f"d0_ma{w}"] = C[i0] > ma[i0]; d[f"d3_ma{w}"] = C[i3] > ma[i3]
        d[f"d3_ma{w}_gap"] = C[i3] / ma[i3] - 1
    d["ma_ok"] = ~np.isnan(px.ma60.values[i0])
    def stack(prefix):
        return (d[f"{prefix}_ma5"].astype(int).astype(str) + d[f"{prefix}_ma10"].astype(int).astype(str)
                + d[f"{prefix}_ma20"].astype(int).astype(str) + d[f"{prefix}_ma60"].astype(int).astype(str)).where(d.ma_ok)
    d["d0_stack"] = stack("d0"); d["d3_stack"] = stack("d3")   # 자리 = MA5·10·20·60 위(1)/아래(0)
    d["d0_above_n"] = d[["d0_ma5", "d0_ma10", "d0_ma20", "d0_ma60"]].sum(axis=1).where(d.ma_ok)
    d["d3_above_n"] = d[["d3_ma5", "d3_ma10", "d3_ma20", "d3_ma60"]].sum(axis=1).where(d.ma_ok)
    d["ma_order"] = ((px.ma5.values[i3] > px.ma20.values[i3]) & (px.ma20.values[i3] > px.ma60.values[i3]))  # 정배열(5>20>60) at D3
    d["d0_ma_order"] = ((px.ma5.values[i0] > px.ma20.values[i0]) & (px.ma20.values[i0] > px.ma60.values[i0]))
    # 자격·시총
    el = eligibility(px[["ticker", "close", "volume"]].assign(year=px.date.dt.year.astype("int16")))
    d = d.merge(el[["ticker", "year", "eligible", "cap_bucket", "market_prev"]], on=["ticker", "year"], how="left")
    d = d[d.eligible.fillna(False) & (d.year >= 2016)].drop(columns="eligible").replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    print(f"build: {len(d):,} 건 {time.time()-t0:.0f}s, 지속 {d.cont.mean()*100:.1f}%")
    return d

if __name__ == "__main__":
    d = build(); d.to_parquet(OUT / "drive_story_k3.parquet")
