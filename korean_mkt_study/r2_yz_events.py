"""r2_yz_events.py — 스윙 포트 눌림목 onset 신호에 R²·YZ 상태를 붙인 이벤트 테이블 생성.

연구 질문 (2026-09-12 Kane): "MA 눌림목 신호에 R²(추세 건전성)와 YZ(변동성 상태)를 얹으면
신호의 질(H일 후 수익)이 갈리는가" — 세 지표의 관계(S1)와 창 스윕(S2)의 공통 입력.

정의 (S0, 2026-09-12 확정)
  이벤트   : backtest.compute_signals 와 같은 onset — close > SMA(ma) & 40/60일 고점 대비
             −10% 눌림이 전일 False → 당일 True 로 바뀐 날. 유니버스 필터 없음(전 종목).
  조건변수 : t일 종가까지의 정보만 (룩어헤드 금지)
             R2_{w}, SLOPE_{w}  w∈{30,60,100,120}  — 로그종가 vs 시간 선형회귀
             YZ_{n}             n∈{10,20,60}       — Yang-Zhang 일간 σ
             YZR = YZ_20/YZ_60 , YZP = YZ_20 의 자기 252일 백분위
             depth(눌림 깊이), margin(close/MA−1), mktcap 티어, 운영 유니버스 편입(4조/30%)
  타깃     : t+1 시가 매수 → t+H 종가 청산 수익률 (H=5/10/20/40 — 케인 지시: 스윙은 짧은 창 우선),
             MAE_H = min(low[t+1..t+H])/open[t+1]−1,  MFE_H = max(high)/open[t+1]−1
  대조군   : 비신호일 무작위 표본 2종 — ctrl_all(전체), ctrl_up(close>MA, 눌림 아님)

출력: out/r2yz/events_{ma}_{hw}.parquet, out/r2yz/control_{ma}.parquet
실행: python3 r2_yz_events.py   (korean_mkt_study/ 에서)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", FutureWarning)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from backtest import DATA, UniverseParams, _monthly_eligibility, _yang_zhang  # noqa: E402

OUT = HERE / "out" / "r2yz"
OUT.mkdir(parents=True, exist_ok=True)

START, END = "2012-01-01", "2026-06-30"
EVENT_FLOOR = pd.Timestamp("2014-01-01")       # STRATEGY.md §12 데이터 floor
R2_WINDOWS = (30, 60, 100, 120)
YZ_WINDOWS = (10, 20, 60)
MA_LIST = (150, 200, 250)
HW_LIST = (40, 60)
PULLBACK = 0.10
HORIZONS = (5, 10, 15, 20, 40)
CHUNK = 350                                    # 종목 청크 (메모리 3.9GB VM 대응)
RNG = np.random.default_rng(20260912)
CTRL_FRAC = 0.02


def rolling_r2_slope(y: pd.DataFrame, w: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """로그가격 y 의 창 w 선형회귀 R²·기울기(일당 로그수익). 누적합 벡터화.
    창 안에 NaN 이 하나라도 있으면 NaN (부분창 금지). np.polyfit 대비 1e-10 일치 확인."""
    m = y.notna().to_numpy()
    yv = np.where(m, y.to_numpy(), 0.0)
    # 크기 축소(누적합 오차) — 열별 첫 유효값 기준 상대화
    first = np.where(m.any(0), yv[np.argmax(m, axis=0), np.arange(yv.shape[1])], 0.0)
    yv = np.where(m, yv - first, 0.0)
    T = yv.shape[0]
    t = np.arange(T, dtype=np.float64)[:, None]

    def wsum(a):
        c = np.cumsum(a, axis=0)
        out = c.copy()
        out[w:] = c[w:] - c[:-w]
        out[: w - 1] = np.nan
        return out

    cnt = wsum(m.astype(np.float64))
    Sy = wsum(yv)
    Syy = wsum(yv * yv)
    Sty = wsum(t * yv)
    # i = t − (t_end − w + 1)  →  Σ i·y = Σ t·y − (t_end − w + 1)·Σ y
    t_end = t[:, 0]
    Siy = Sty - ((t_end - w + 1)[:, None]) * Sy
    Si = w * (w - 1) / 2.0
    Sii = (w - 1) * w * (2 * w - 1) / 6.0
    sxx = w * Sii - Si * Si
    sxy = w * Siy - Si * Sy
    syy = w * Syy - Sy * Sy
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = (sxy * sxy) / (sxx * syy)
        slope = sxy / sxx
    bad = ~(cnt == w)
    r2[bad] = np.nan
    slope[bad] = np.nan
    r2 = np.clip(r2, 0.0, 1.0)
    return (pd.DataFrame(r2, index=y.index, columns=y.columns),
            pd.DataFrame(slope, index=y.index, columns=y.columns))


def fwd_stats(op: pd.DataFrame, hi: pd.DataFrame, lo: pd.DataFrame, cl: pd.DataFrame,
              H: int) -> dict[str, pd.DataFrame]:
    entry = op.shift(-1)
    ret = cl.shift(-H) / entry - 1.0
    lo_min = lo.iloc[::-1].rolling(H, min_periods=H).min().iloc[::-1].shift(-1)
    hi_max = hi.iloc[::-1].rolling(H, min_periods=H).max().iloc[::-1].shift(-1)
    return {f"RET_{H}": ret, f"MAE_{H}": lo_min / entry - 1.0, f"MFE_{H}": hi_max / entry - 1.0}


def main() -> None:
    t0 = time.time()
    px = pd.read_parquet(DATA / "prices.parquet")
    d = px.index.get_level_values("date")
    px = px[(d >= pd.Timestamp(START)) & (d <= pd.Timestamp(END))]
    wide = {c: px[c].astype("float64").unstack("ticker").sort_index()
            for c in ("open", "high", "low", "close")}
    del px
    for c in wide:                                  # 가격 0 (거래정지·오류) → NaN
        wide[c] = wide[c].where(wide[c] > 0)
    dates = wide["close"].index
    tickers = wide["close"].columns
    print(f"panel {len(dates)} days × {len(tickers)} tickers  ({time.time()-t0:.0f}s)")

    # 운영 유니버스 편입(4조/30%) + 시총 — 이벤트 라벨용
    elig_m = _monthly_eligibility(UniverseParams())
    elig_d = (elig_m.reindex(columns=tickers).reindex(index=dates, method="ffill")
              .fillna(False))
    meta = pd.read_parquet(DATA / "meta.parquet", columns=["mktcap"])
    cap_m = meta["mktcap"].astype("float64").unstack("ticker")
    cap_d = cap_m.reindex(columns=tickers).reindex(index=dates, method="ffill")

    events: dict[tuple[int, int], list[pd.DataFrame]] = {(ma, hw): [] for ma in MA_LIST for hw in HW_LIST}
    controls: dict[int, list[pd.DataFrame]] = {ma: [] for ma in MA_LIST}

    for c0 in range(0, len(tickers), CHUNK):
        cols = tickers[c0:c0 + CHUNK]
        op, hi, lo, cl = (wide[k][cols] for k in ("open", "high", "low", "close"))
        feat: dict[str, pd.DataFrame] = {}
        ly = np.log(cl)
        for w in R2_WINDOWS:
            r2, sl = rolling_r2_slope(ly, w)
            feat[f"R2_{w}"] = r2
            feat[f"SLOPE_{w}"] = sl
        for n in YZ_WINDOWS:
            feat[f"YZ_{n}"] = _yang_zhang(op, hi, lo, cl, n)
        feat["YZR"] = feat["YZ_20"] / feat["YZ_60"]
        feat["YZP"] = feat["YZ_20"].rolling(252, min_periods=126).rank(pct=True)
        for H in HORIZONS:
            feat.update(fwd_stats(op, hi, lo, cl, H))
        feat["ELIG"] = elig_d[cols].astype(float)
        feat["MKTCAP"] = cap_d[cols]

        mas = {ma: cl.rolling(ma, min_periods=int(ma * 0.7)).mean() for ma in MA_LIST}
        highs = {hw: cl.rolling(hw, min_periods=hw // 2).max() for hw in HW_LIST}

        def collect(mask: pd.DataFrame, extra: dict[str, pd.DataFrame]) -> pd.DataFrame:
            mask = mask.copy()
            mask.loc[mask.index < EVENT_FLOOR] = False
            idx = np.argwhere(mask.to_numpy())
            if len(idx) == 0:
                return pd.DataFrame()
            rows = {"date": mask.index[idx[:, 0]], "ticker": mask.columns[idx[:, 1]]}
            for k, df in {**feat, **extra}.items():
                rows[k] = df.to_numpy()[idx[:, 0], idx[:, 1]]
            return pd.DataFrame(rows)

        for ma in MA_LIST:
            margin = cl / mas[ma] - 1.0
            up = (margin > 0).fillna(False)
            any_dip = pd.DataFrame(False, index=cl.index, columns=cl.columns)
            for hw in HW_LIST:
                depth = (highs[hw] - cl) / highs[hw]
                cond = ((depth >= PULLBACK) & up).fillna(False)
                onset = cond & ~cond.shift(1).fillna(False)
                any_dip |= cond
                ev = collect(onset, {"DEPTH": depth, "MARGIN": margin})
                if len(ev):
                    ev["MA"], ev["HW"] = ma, hw
                    events[(ma, hw)].append(ev)
            # 대조군: 비신호일 무작위 (전체 / 상승추세·비눌림)
            valid = cl.notna() & op.shift(-1).notna()
            samp = pd.DataFrame(RNG.random(cl.shape) < CTRL_FRAC, index=cl.index, columns=cl.columns)
            ctrl_all = collect(valid & samp & ~any_dip, {"MARGIN": margin})
            ctrl_up = collect(valid & samp & up & ~any_dip, {"MARGIN": margin})
            if len(ctrl_all):
                ctrl_all["KIND"] = "ctrl_all"
                ctrl_up["KIND"] = "ctrl_up"
                controls[ma].append(pd.concat([ctrl_all, ctrl_up]))
        print(f"  chunk {c0//CHUNK+1}/{(len(tickers)-1)//CHUNK+1} done ({time.time()-t0:.0f}s)", flush=True)

    for (ma, hw), lst in events.items():
        df = pd.concat(lst, ignore_index=True)
        df.to_parquet(OUT / f"events_{ma}_{hw}.parquet", index=False)
        print(f"events MA{ma}/HW{hw}: {len(df):,} rows, {df.ticker.nunique()} tickers, "
              f"{df.date.min().date()}~{df.date.max().date()}")
    for ma, lst in controls.items():
        df = pd.concat(lst, ignore_index=True)
        df.to_parquet(OUT / f"control_{ma}.parquet", index=False)
        print(f"control MA{ma}: {len(df):,} rows")
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
