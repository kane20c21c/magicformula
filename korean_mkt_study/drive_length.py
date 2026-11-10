"""상승 드라이브 길이 연구 — 드라이브 단위 표 + D0 시점 설명 변수 (2026-11-03, 케인 지시).

규칙
  · 드라이브 정의·3변수(속도 A·거래량 T·겹침 ō)는 drive_types 그대로. 단 여기서는 PIT 가 아니라
    드라이브 **전 구간(D1..Dn)** 으로 계산한다 — 길이 n 자체가 사후값이라 이 표는 '설명' 용도이지 신호용이 아니다.
  · 설명 변수는 전부 **D0 = D1 의 전 거래일 종가 시점**까지의 정보만 쓴다 (케인: "당일 조건으로 넣지 말고 하루 어긋나게").
    월말 스냅샷(시총·외국인·섹터)은 D1 보다 앞선 가장 최근 월말.
출력: out/crash/drive_len_up.parquet
"""
from __future__ import annotations
import sys, time
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from crash_study import DATA, OUT, eligibility, load_index
from drive_types import add_daily_fields, load_ohlcv, classify, MIN_DRIVE


def roll(g, col, w, fn="mean", minp=None):
    minp = minp or w
    return g[col].transform(lambda s: getattr(s.rolling(w, min_periods=minp), fn)())


def build(direction: int = +1, px: pd.DataFrame | None = None) -> pd.DataFrame:
    """px 를 주면 그 가격으로 만든다 (drive_regime 보류 구간 검증 — LLV 원본 이어 붙인 가격). 기본 = prices.parquet."""
    t0 = time.time()
    px = load_ohlcv(DATA / "prices.parquet") if px is None else px.copy()
    px = add_daily_fields(px, direction)                       # r1, run, ov, valid, v20, next_r1, c10, ext2_10
    g = px.groupby("ticker", sort=False, observed=True)
    # ---- D0 시점 일별 피처 (그날 종가까지) ----
    px["ma20"] = roll(g, "close", 20); px["ma60"] = roll(g, "close", 60, minp=40)
    px["hi60"] = roll(g, "close", 60, "max", 40); px["hi252"] = roll(g, "close", 252, "max", 120)
    px["sig60"] = g.r1.transform(lambda s: s.rolling(60, min_periods=40).std()).astype("float32")
    px["sig20"] = g.r1.transform(lambda s: s.rolling(20, min_periods=15).std()).astype("float32")
    px["c20"] = g.close.shift(20); px["c5"] = g.close.shift(5)
    px["v5"] = roll(g, "volume", 5)
    px["amt"] = (px.close.astype("float64") * px.volume).astype("float64")
    px["amt20"] = roll(g, "amt", 20, minp=15)
    # 직전 하락 런 길이 (D0 에서 끝나는 연속 하락일 수)
    dn = px.r1 < 0; brk = (~dn) | (px.ticker != px.ticker.shift())
    px["dn_run"] = dn.groupby(brk.cumsum()).cumsum().astype("int16")
    # 시장 폭: 그날 종가 > MA20 인 종목 비율 (전 종목)
    above = (px.close > px.ma20).where(px.ma20.notna())
    breadth = above.groupby(px.date).mean().rename("breadth20")
    # ---- 드라이브 종료 행 찾기 ----
    run = px.run.values.astype(np.int32)
    nxt = np.r_[run[1:], 0]; same_next = np.r_[(px.ticker.values[1:] == px.ticker.values[:-1]), False]
    end = (run >= MIN_DRIVE) & ~((nxt > run) & same_next)
    idx_end = np.flatnonzero(end); n = run[idx_end]; s = idx_end - n + 1; d0 = s - 1
    ok = d0 >= 0
    idx_end, n, s, d0 = idx_end[ok], n[ok], s[ok], d0[ok]
    tk = px.ticker.values
    ok = (tk[d0] == tk[s]) & (tk[idx_end] == tk[s])
    idx_end, n, s, d0 = idx_end[ok], n[ok], s[ok], d0[ok]
    # ---- 드라이브 전 구간 3변수 ----
    r1 = px.r1.values.astype("float64"); vol = px.volume.values.astype("float64"); close = px.close.values.astype("float64")
    ovv = px.ov.values.astype("float64"); valid = px.valid.values
    ov_cs = np.nancumsum(np.where(valid, ovv, 0.0)); va_cs = np.cumsum(valid.astype(np.int64))
    def seg_sum(cs, a, b):  # a..b 포함
        return cs[b] - np.where(a > 0, cs[a - 1], 0.0)
    nvalid = seg_sum(va_cs, s, idx_end)
    overlap = np.where(nvalid == n, seg_sum(ov_cs, s + 1, idx_end) / (n - 1), np.nan)
    d = pd.DataFrame({
        "ticker": tk[s].astype(str), "date": px.date.values[s], "date_end": px.date.values[idx_end],
        "n": n.astype("int16"),
        "tot": close[idx_end] / close[d0] - 1,
        "speed": np.abs(r1[idx_end]) / np.abs(r1[s]),
        "v_trend": vol[idx_end] / vol[s],
        "overlap": overlap,
        "v_level": (seg_sum(np.cumsum(vol), s, idx_end) / n) / px.v20.values[d0],
        "r_d1": r1[s], "r_dn": r1[idx_end],
        # 사후(검증용)
        "next_r1": px.next_r1.values[idx_end], "r10": px.c10.values[idx_end] / close[idx_end] - 1,
        "rebreak": px.ext2_10.values[idx_end] > close[idx_end],
    })
    # ---- D0 시점 설명 변수 ----
    D = px.iloc[d0]
    d["close_d0"] = close[d0]
    d["sig60"] = D.sig60.values; d["sig20"] = D.sig20.values
    d["ret20"] = close[d0] / D.c20.values - 1; d["ret5"] = close[d0] / D.c5.values - 1
    d["ma20_gap"] = close[d0] / D.ma20.values - 1; d["ma60_gap"] = close[d0] / D.ma60.values - 1
    d["hi60_gap"] = close[d0] / D.hi60.values - 1; d["hi252_gap"] = close[d0] / D.hi252.values - 1
    d["v5_20"] = D.v5.values / D.v20.values
    d["amt20"] = D.amt20.values
    d["dn_run_pre"] = D.dn_run.values
    d["breadth20"] = breadth.reindex(D.date.values).values
    kospi = load_index()["KOSPI"]
    d["kospi_ret20"] = (kospi / kospi.shift(20) - 1).reindex(D.date.values).values
    d["kospi_ret60"] = (kospi / kospi.shift(60) - 1).reindex(D.date.values).values
    # 월말 스냅샷: D1 이전 가장 최근 월말
    meta = pd.read_parquet(DATA / "meta.parquet", columns=["mktcap", "foreign", "sector", "market"]).reset_index().sort_values("date")
    d = d.sort_values("date")
    d = pd.merge_asof(d, meta.rename(columns={"date": "snap_date"}), left_on="date", right_on="snap_date", by="ticker",
                      allow_exact_matches=False, direction="backward")
    d["year"] = d.date.dt.year.astype("int16")
    el = eligibility(px[["ticker", "close", "volume"]].assign(year=px.date.dt.year.astype("int16")))
    d = d.merge(el[["ticker", "year", "eligible", "cap_bucket", "market_prev"]], on=["ticker", "year"], how="left")
    d = d[d.eligible.fillna(False) & (d.year >= 2016)].drop(columns="eligible")
    d = d.replace([np.inf, -np.inf], np.nan)
    for key in ("speed", "v_trend", "overlap"):
        d[f"{key}_c"] = classify(d[key], key, direction)
    d = d.reset_index(drop=True)
    print(f"build: {len(d):,} 드라이브 {time.time()-t0:.0f}s")
    return d


if __name__ == "__main__":
    d = build(+1)
    d.to_parquet(OUT / "drive_len_up.parquet")
    print(d.n.value_counts().sort_index().to_string())
