"""드라이브 공통 정의 — 하락형/상승형 연구가 같은 함수·같은 경계를 쓰도록 한 곳에 모음.

용어 (2026-11-03 케인과 확정)
  드라이브   : sign(r_t) 가 같은 날이 연속 n>=3 일 이어지는 최대 구간. r_t = C_t/C_{t-1}-1. 종가 기준.
               r_t = 0 인 날은 양쪽 모두 드라이브를 끊는다. D1..Dn = 드라이브 날짜, D+1 = 종료 확인일.
  속도 A     : |r_Dn| / |r_D1|            가속 A > 1.8  / 감쇄 A < 0.65
  거래량 T   : V_Dn / V_D1                 증가 T > 1.25 / 감소 T < 0.75
  겹침 ō     : mean_{t=D2..Dn} o_t,  o_t = max(0, min(H_t,H_{t-1}) - max(L_t,L_{t-1})) / (H_t - L_t)
                                          계단 ō < 0.40 / 미끄럼 ō > 0.57
  경계는 하락형 2016~2026 point-in-time 281,696건의 3분위(0.639/1.833, 0.767/1.277, 0.414/0.571)를 반올림한 고정값.

빌더는 point-in-time: k일차(드라이브 진행 중)에 서 있는 모든 (ticker, date) 행을 만들고,
지표는 D1..Dk 까지의 정보만 쓴다. 사후 컬럼(next_r1, hold10 ...)은 검증용이며 신호 계산에 쓰면 안 된다.
"""
from __future__ import annotations
import numpy as np, pandas as pd

THRESH = {  # (하위 경계, 상위 경계)
    "speed":   (0.65, 1.80),   # 감쇄 / 가속
    "v_trend": (0.75, 1.25),   # 감소 / 증가
    "overlap": (0.40, 0.57),   # 계단 / 미끄럼
}
LABELS = {
    "speed":   ("감쇄", "평탄", "가속"),
    "v_trend": ("감소", "평탄", "증가"),
    "overlap": ("계단", "중간", "미끄럼"),
}
MIN_DRIVE = 3
PRE_WINDOW = 20          # 거래량 정규화: 드라이브 직전 20일 평균
HOLD_FROM, HOLD_TO = 2, 10   # 사후 타깃: D+2..D+10 종가가 드라이브 끝 종가를 지키는가


def classify(x: pd.Series, key: str) -> pd.Series:
    lo, hi = THRESH[key]; a, b, c = LABELS[key]
    return pd.Series(np.select([x < lo, x > hi], [a, c], b), index=x.index).where(x.notna()).astype("category")


def add_daily_fields(px: pd.DataFrame, direction: int) -> pd.DataFrame:
    """px: ticker(category)·date 정렬된 OHLCV. direction +1 상승형 / -1 하락형. 일별 보조 컬럼 추가."""
    g = px.groupby("ticker", sort=False, observed=True)
    pc = g.close.shift(1)
    px["r1"] = (px.close / pc - 1).astype("float32")
    same = (px.r1 > 0) if direction > 0 else (px.r1 < 0)
    brk = (~same) | (px.ticker != px.ticker.shift())
    px["run"] = same.groupby(brk.cumsum()).cumsum().astype("int16")
    h, l = px.high, px.low
    ph, pl = g.high.shift(1), g.low.shift(1)
    rlen = (h - l).where(lambda s: s > 0)
    px["ov"] = ((np.minimum(h, ph) - np.maximum(l, pl)).clip(lower=0) / rlen).astype("float32")
    px["valid"] = (px.open > 0) & (h > 0) & (l > 0) & (ph > 0) & (pl > 0)
    px["v20"] = g.volume.transform(lambda s: s.rolling(PRE_WINDOW, min_periods=15).mean()).astype("float32")
    px["next_r1"] = g.r1.shift(-1)
    px["c10"] = g.close.shift(-HOLD_TO)
    w = HOLD_TO - HOLD_FROM + 1
    fwd = g.close.transform(lambda s: s[::-1].rolling(w, min_periods=w).min()[::-1].shift(-HOLD_FROM)) if direction < 0 \
        else g.close.transform(lambda s: s[::-1].rolling(w, min_periods=w).max()[::-1].shift(-HOLD_FROM))
    px["ext2_10"] = fwd.astype("float32")   # 하락형: D+2..10 최저 종가 / 상승형: 최고 종가
    return px


def build_pit(px: pd.DataFrame, direction: int, ks=(3, 4, 5)) -> pd.DataFrame:
    """k일차 시점 행 생성. 반환 컬럼: ticker,date,year,k,close,drop,speed,v_trend,overlap,v_level,v_last,
    next_r1,ended,hold10,r10 (+ 분류 컬럼 speed_c,v_trend_c,overlap_c)."""
    px = add_daily_fields(px, direction)
    g = px.groupby("ticker", sort=False, observed=True)
    out = []
    for k in ks:
        r_first = g.r1.shift(k - 1); v_first = g.volume.shift(k - 1); base = g.v20.shift(k)
        ov_mean = g.ov.transform(lambda s, w=k - 1: s.rolling(w).mean())
        nvalid = g.valid.transform(lambda s, w=k: s.rolling(w).sum()) == k
        vmean = g.volume.transform(lambda s, w=k: s.rolling(w).mean())
        sel = (px.run == k) & nvalid & px.next_r1.notna() & base.gt(0)
        d = px.loc[sel, ["ticker", "date", "close", "next_r1", "c10", "ext2_10"]].copy()
        d["year"] = d.date.dt.year.astype("int16"); d["k"] = np.int8(k)
        d["drop"] = (px.close / g.close.shift(k) - 1)[sel].astype("float32").values
        d["speed"] = (px.r1.abs() / r_first.abs())[sel].astype("float32").values
        d["v_trend"] = (px.volume / v_first)[sel].astype("float32").values
        d["overlap"] = ov_mean[sel].astype("float32").values
        d["v_level"] = (vmean / base)[sel].astype("float32").values
        d["v_last"] = (px.volume / base)[sel].astype("float32").values
        out.append(d)
    d = pd.concat(out).replace([np.inf, -np.inf], np.nan)
    d["ended"] = (d.next_r1 >= 0) if direction < 0 else (d.next_r1 <= 0)   # 하락형: 다음날 r1>=0 이면 종료
    d["hold10"] = (d.ext2_10 >= d.close) if direction < 0 else (d.ext2_10 <= d.close)   # 상승형: 고점 미돌파 = '지킴'
    d["r10"] = (d.c10 / d.close - 1).astype("float32")
    for key in ("speed", "v_trend", "overlap"):
        d[f"{key}_c"] = classify(d[key], key)
    return d.drop(columns=["c10", "ext2_10"]).reset_index(drop=True)


def load_ohlcv(path) -> pd.DataFrame:
    px = pd.read_parquet(path, columns=["open", "high", "low", "close", "volume"]).reset_index()
    px["ticker"] = px.ticker.astype("category")
    for c in ("open", "high", "low", "close"): px[c] = px[c].astype("float32")
    return px.sort_values(["ticker", "date"]).reset_index(drop=True)


def tercile_bounds(d: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({k: d[k].quantile([1 / 3, 2 / 3]).values for k in ("speed", "v_trend", "overlap")}, index=["1/3", "2/3"]).T


if __name__ == "__main__":
    import sys, time
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from crash_study import DATA, OUT, eligibility
    direction = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    t0 = time.time()
    px = load_ohlcv(DATA / "prices.parquet"); px["year"] = px.date.dt.year.astype("int16")
    el = eligibility(px)
    d = build_pit(px, direction)
    d["ticker"] = d.ticker.astype(str)
    d = d.merge(el, on=["ticker", "year"], how="left"); d = d[d.eligible.fillna(False) & (d.year >= 2016)]
    name = "down" if direction < 0 else "up"
    d.to_parquet(OUT / f"drive_pit_{name}.parquet")
    print(f"{name} {len(d):,}행 {time.time()-t0:.0f}s")
    print(tercile_bounds(d).round(3).to_string())
    for key in ("speed", "v_trend", "overlap"):
        print(key, (d[f"{key}_c"].value_counts(normalize=True) * 100).round(1).to_dict())
