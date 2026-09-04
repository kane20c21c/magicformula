"""backtest.py — 스윙 포트(`kr_pullback_largecap_foreign`) 백테스트 엔진.

목적: 진입·청산 규칙 변경을 **같은 조건에서** 비교하기 위한 재현 가능한 엔진.

⚠ 이 파일이 생긴 이유 — 2026-08-16 청산 v1.2.0 검증에 쓴 코드가 어느 리포에도
   남지 않아(리포트 HTML만 존재) 재현이 불가능했다. 앞으로 규칙 실험은 이 파일로 한다.

정본 관계
  - 진입 신호 정의 : `strategy_reference.py` (v1.0.0) — 여기서는 그 정의를 재현하고
                     **변형(variant)** 을 얹는다. 채택된 변형만 정본에 반영한다.
  - 청산 규칙      : `StockPortfolio/app/paper/config.py` (v1.2.0) — 아래 ExitParams 가
                     그 값을 복사한 것이다. 원본이 바뀌면 여기도 맞춰야 한다.

검증 조건 (2026-08-17 Kane 확정)
  - 유니버스 : 월말 시총 **4조**↑ & 외국인지분 **25%**↑ (실사용 universe_*.json 과 동일)
               ⚠ STRATEGY.md §2 의 문서값은 5조/30% — 실사용과 불일치. 실사용을 따른다.
  - 매수     : 신호 t일 종가 확정 → **t+1일 시가**
  - 매도     : 청산 규칙 도달 시 **청산가격 × 99.5%** 로 전량 (슬리피지 0.5%)
  - 비용     : 왕복 **0.23%** (매수 0.015% / 매도 0.015% + 세금 0.2%)
  - 매도대금 T+1 현금화
  ⚠ STRATEGY.md §8 의 구 관례(종가 체결·왕복 0.6%)로 재면 고회전 규칙이 부당하게
    불리해진다 — 규칙 비교는 반드시 위 조건으로 한다.
  ⚠ STRATEGY.md §8 의 구 관례(종가 체결·왕복 0.6%)로 재면 고회전 규칙이 부당하게
    불리해진다 — 규칙 비교는 반드시 라이브 조건으로 한다.

데이터
  data/prices.parquet   (date, ticker) → open/high/low/close/volume   2010-01~2026-06
  data/meta.parquet     (월말, ticker) → mktcap, shares, ...          2010-01~2026-05
  data/foreign.parquet  (월말, ticker) → foreign_ratio(%)             2010-01~2026-06

실행: python3 backtest.py            (기준선 1회 실행 + 요약 출력)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"

# ────────────────────────────────────────────────────────────────────────────
# 파라미터
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class UniverseParams:
    """월말 판정 → 다음 달 적용 (일별 ffill)."""
    # v1.2.2.3 확정 (Kane 2026-08-17) — 유니버스 2번째: 4조/30%.
    # (①4조/25% 실운영 → ②4조/30%. 완화 스윕: 어느 축이든 완화 시 단조 악화,
    #  외인 필터는 위기 방어 필터 — 완화 시 코로나 손익 +2,774만→+201만 붕괴.)
    mktcap_min_krw: float = 4_000_000_000_000     # 4조
    foreign_min_pct: float = 30.0                 # 30%
    exclude_pref: bool = True                     # 우선주 제외
    exclude_spac: bool = True                     # 스팩 제외
    exclude_managed: bool = True                  # 관리종목 제외


@dataclass(frozen=True)
class EntryParams:
    """진입 신호. baseline = v1.0.0 (§3).

    variant 축
      confirm   : 눌림 구간 안에서 '하락 멈춤'을 확인한 첫날에 진입 (축 A)
      vol_depth : 눌림 임계를 변동배율로 정규화 (축 B)
    """
    # v1.2.2.3 확정 (Kane 2026-08-17) — 진입규칙 2번째 (W1): 40일 고점 · MA200.
    # v1.0.0 기준선 재현은 EntryParams(high_window=60, high_min_periods=30,
    # trend_ma=120, trend_min_periods=80) 로 명시 지정.
    high_window: int = 40
    high_min_periods: int = 20
    trend_ma: int = 200
    trend_min_periods: int = 140
    pullback_pct: float = 0.10                    # close <= (1-0.10) * high_n

    # ── 축 M: 추세선 종류 (2026-09-04 Kane 요청) ──
    #   "sma"   : 종가 단순이동평균 — v1.0.0~v1.2.2.3 정본
    #   "evwma" : Elastic Volume Weighted MA (LLV `_evwma_one` 정본 공식)
    #             유동물량 V(=n일 거래량 합) 대비 그날 거래된 만큼만 평단이 교체된다.
    #   ⚠ evwma 는 `trend_min_periods` 가 **V 의 min_periods** 로 쓰인다.
    #     LLV 운영 컬럼(EVWMA_200)과 값을 맞추려면 trend_min_periods=trend_ma (엄격).
    trend_ma_kind: str = "sma"

    # ── 눌림 기준 (pattern_study.py 격자 연구, 2026-08-17) ──
    #   "high"     : 종가 ≤ (1−pullback_pct) × N일 최고종가   (현행)
    #   "low_prox" : 종가 ≤ (1+low_tol) × 직전 N일 최저종가   (저점 근접 — Kane 안)
    dip_basis: str = "high"
    low_tol: float = 0.20

    # ── 깊이별 계단 진입 (pattern_study 후속, 2026-08-17) ──
    #   () 이면 단일 임계(pullback_pct). (0.10, 0.20, 0.30) 이면 각 깊이를
    #   '처음 뚫는 날' 마다 onset — 같은 하락에서 최대 len(tiers) 회 신호.
    #   격자 실측: 깊을수록 건당 품질 급등 (−20% 정밀도 39~45%, −30% 승률 72%)
    #   하지만 단독으론 신호가 희소해 자본이 놈 → 계단으로 결합.
    pullback_tiers: tuple = ()

    #   True 면 깊은 티어(2번째~)는 신규/재진입에만 쓰고, 보유 중 불타기 트리거로는
    #   1차 티어 onset 만 허용 — "물타기" 성격 차단 (Kane 우려 2026-08-17).
    tiers_new_only: bool = False

    # ── 축 A: 하락 멈춤 확인 ──
    #   None      = v1.0.0 (눌림 조건 성립 첫날 = onset 즉시 진입)
    #   "up1"     = 종가 > 전일 종가
    #   "up2"     = 2일 연속 상승
    #   "mid"     = 종가 > (당일 고가+저가)/2   (당일 종가 위치 상단)
    #   "up1_mid" = up1 AND mid
    #   "ma5"     = 종가 > MA5
    confirm: Optional[str] = None
    confirm_max_wait: int = 10                    # 눌림 시작 후 N거래일 안에 확인 없으면 포기

    # ── 축 R: 재현율 (diag_recall.py 진단으로 신설, Kane 지시 2026-08-17) ──
    #   진단: 눌림 후 반등 기회 2,605건 중 **20.7%만** 잡았다.
    #         놓친 이유 1위 = MA120 아래 (60.3%), 2위 = onset 제한 (12.6%),
    #         슬롯/현금 부족은 6.4%뿐 (20슬롯 만재일 0.2%).
    #   → 사이징이 아니라 트리거가 기회의 4/5 를 구조적으로 배제하고 있었다.
    #
    #   resignal_gap : 0  = onset 만 (v1.0.0)
    #                  N>0 = 같은 눌림 구간 안에서도 N거래일 간격으로 재신호 허용
    resignal_gap: int = 0

    # ── 축 B: 변동성 정규화 눌림 ──
    #   False = 고정 pullback_pct
    #   True  = clip(pullback_pct × 변동배율, vol_depth_clip)
    vol_depth: bool = False
    vol_depth_clip: tuple = (0.05, 0.20)

    # ── 축 C: 추세 여력 (diag_entry.py 진단 결과로 신설) ──
    #   v1.0.0 은 close > MA120 만 요구해 '간신히 위' 도 통과시킨다.
    #   진단: MA120 대비 여력 최상위 5분위 fwd10 +3.43%(승률 56.0%) vs
    #         나머지 +0.30~0.89%(49~53%). 여력이 신호 품질의 최강 분리축이었다.
    trend_margin: float = 0.0                     # close > MA120 × (1 + margin)

    # ── 후보 우선순위 (§6) ──
    #   "depth"      : 눌림 깊은 순 (v1.0.0)
    #   "depth_asc"  : 눌림 얕은 순
    #   "trend"      : MA120 대비 여력 큰 순
    #   "vol"        : 변동배율 큰 순
    priority: str = "depth"


@dataclass(frozen=True)
class ExitParams:
    """청산 v1.2.0 — StockPortfolio/app/paper/config.py 복사본."""
    switch_day: int = 5                           # D+0~5 early, D+6~ late
    early_pct: float = 0.20
    late_up_pct: float = 0.05                     # 피크 > 평단
    late_flat_pct: float = 0.10                   # 피크 ≤ 평단
    vol_linked: bool = True
    clip_early: tuple = (0.10, 0.40)
    clip_late_up: tuple = (0.03, 0.15)
    clip_late_flat: tuple = (0.05, 0.25)


@dataclass(frozen=True)
class PortfolioParams:
    capital_krw: float = 100_000_000.0
    max_positions: int = 20
    slot_krw: float = 5_000_000.0
    max_slots_per_stock: int = 2
    cooldown_trading_days: int = 1
    buy_fee: float = 0.00015
    sell_fee: float = 0.00015
    sell_tax: float = 0.002
    # 손절 체결 = 청산가격 × 99.5% (Kane 확정 2026-08-17).
    # 청산가격 = min(당일 시가, 손절선) — 갭하락으로 시가가 손절선보다 낮으면 시가.
    stop_slippage: float = 0.005


@dataclass(frozen=True)
class VolScaleParams:
    """변동배율 — 청산 v1.2.0 정의 (LLV data/vol_scale.json 과 동일 산식).

    배율 = 종목 YZ_20(t) ÷ D(t)
    D(t) = 직전 252거래일 '스윙풀 YZ_20 일별 중앙값' 의 중앙값
    ⚠ 분모가 '그날 시장 중앙값'(상대)이 아니라 '직전 1년 중앙값'(절대)인 것이 핵심.
    """
    yz_window: int = 20
    pool_lookback: int = 252
    clip: tuple = (0.2, 5.0)


# ────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class Panel:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame        # EVWMA 추세선용 (SMA 경로에서는 안 쓴다)
    elig: pd.DataFrame          # 일별 유니버스 편입 bool
    yz20: pd.DataFrame
    vol_scale: pd.DataFrame
    dates: pd.DatetimeIndex
    tickers: pd.Index
    names: dict


def _monthly_eligibility(up: UniverseParams) -> pd.DataFrame:
    meta = pd.read_parquet(DATA / "meta.parquet")
    fo = pd.read_parquet(DATA / "foreign.parquet")

    m = meta.copy()
    if up.exclude_pref and "is_pref" in m:
        m = m[~m["is_pref"].fillna(False).astype(bool)]
    if up.exclude_spac and "is_spac" in m:
        m = m[~m["is_spac"].fillna(False).astype(bool)]
    if up.exclude_managed and "is_managed" in m:
        m = m[~m["is_managed"].fillna(False).astype(bool)]

    cap = m["mktcap"].astype("float64").unstack("ticker")
    fr = fo["foreign_ratio"].astype("float64").unstack("ticker")

    cols = cap.columns.intersection(fr.columns)
    idx = cap.index.union(fr.index)
    cap = cap.reindex(index=idx, columns=cols)
    fr = fr.reindex(index=idx, columns=cols)
    elig = (cap >= up.mktcap_min_krw) & (fr >= up.foreign_min_pct)
    return elig.fillna(False)


def _yang_zhang(op: pd.DataFrame, hi: pd.DataFrame, lo: pd.DataFrame,
                cl: pd.DataFrame, n: int) -> pd.DataFrame:
    """Yang–Zhang 일간 σ (LLV indicator_calculator 와 동일 산식, 벡터화 판)."""
    prev_c = cl.shift(1)
    o = np.log(op / prev_c)          # overnight
    c = np.log(cl / op)              # open-to-close
    u = np.log(hi / op)
    d = np.log(lo / op)

    v_o = o.rolling(n, min_periods=n).var(ddof=1)
    v_c = c.rolling(n, min_periods=n).var(ddof=1)
    rs = u * (u - c) + d * (d - c)
    v_rs = rs.rolling(n, min_periods=n).mean()

    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var = v_o + k * v_c + (1 - k) * v_rs
    return np.sqrt(var.clip(lower=0))


def _evwma_col(close: np.ndarray, vol: np.ndarray, n: int, min_periods: int) -> np.ndarray:
    """단일 종목 EVWMA. **LLV `indicator_calculator._evwma_one()` 이식** (2026-09-04).

        EVWMA_t = EVWMA_{t-1} · (V_t − v_t)/V_t  +  Close_t · v_t/V_t
        V_t = 직전 n거래일 거래량 합 (= '유동물량')

    "그날 거래된 물량만큼만 평단이 교체된다" 는 해석. 거래가 없으면 값이 그대로
    유지되고, 하루 거래량이 유동물량에 가까울수록 종가에 붙는다.

    ⚠ LLV 정본과 다른 점은 `min_periods` 를 인자로 받는다는 것 하나뿐이다.
      LLV 는 엄격(=n) 고정인데, 여기서는 SMA(140) 와 조건을 맞춘 대조군을 함께
      재기 위해 열어 뒀다. **운영 컬럼(LLV EVWMA_200)과 값을 맞추려면 n 을 줄 것.**

    ⚠ 재귀식이라 벡터화하지 않았다 (LLV 주석과 동일 사유 — cumprod 로 풀면
      v_t/V_t 가 1 에 가까울 때 곱누적이 0 으로 붕괴해 나눗셈이 발산한다).

    ⚠ **첫 유효일 값은 그날 종가 자체다.** 따라서 워밍업 직후 구간은
      `close > EVWMA` 가 거의 항상 참이 된다 — 백테스트 시작일보다 앞선 데이터로
      워밍업을 흡수시켜야 이 편향이 성과에 안 섞인다 (load_panel 의 warmup 참조).
    """
    N = len(close)
    out = np.full(N, np.nan)
    if N == 0:
        return out
    V = pd.Series(vol).rolling(n, min_periods=min_periods).sum().to_numpy(dtype=float)
    prev = np.nan
    for i in range(N):
        if not np.isfinite(V[i]) or V[i] <= 0 or not np.isfinite(close[i]):
            continue
        if not np.isfinite(prev):
            prev = close[i]                    # 첫 유효일은 종가에서 출발
        v = vol[i] if np.isfinite(vol[i]) else 0.0
        v = min(v, V[i])                       # 하루 거래량이 창 합을 넘지 않게
        prev = prev * (V[i] - v) / V[i] + close[i] * v / V[i]
        out[i] = prev
    return out


def _evwma(cl: pd.DataFrame, vol: pd.DataFrame, n: int, min_periods: int) -> pd.DataFrame:
    """종목별 EVWMA (파이썬 루프 — 종목 수 × 거래일 수)."""
    v = vol.reindex_like(cl)
    out = {t: _evwma_col(cl[t].to_numpy(dtype=float), v[t].to_numpy(dtype=float),
                         n, min_periods)
           for t in cl.columns}
    return pd.DataFrame(out, index=cl.index, columns=cl.columns)


def _trend_ma(p: "Panel", ep: "EntryParams") -> pd.DataFrame:
    """추세선. 종류 축은 여기 한 곳에서만 갈린다."""
    if ep.trend_ma_kind == "sma":
        return p.close.rolling(ep.trend_ma, min_periods=ep.trend_min_periods).mean()
    if ep.trend_ma_kind == "evwma":
        return _evwma(p.close, p.volume, ep.trend_ma, ep.trend_min_periods)
    raise ValueError(f"unknown trend_ma_kind: {ep.trend_ma_kind}")


def load_panel(up: UniverseParams, vp: VolScaleParams,
               start: str = "2013-01-01", end: str = "2026-06-30") -> Panel:
    elig_m = _monthly_eligibility(up)
    ever = elig_m.columns[elig_m.any(axis=0)]

    px = pd.read_parquet(DATA / "prices.parquet")
    px = px[px.index.get_level_values("ticker").isin(ever)]
    px = px[(px.index.get_level_values("date") >= pd.Timestamp(start)) &
            (px.index.get_level_values("date") <= pd.Timestamp(end))]

    wide = {c: px[c].astype("float64").unstack("ticker").sort_index()
            for c in ("open", "high", "low", "close", "volume")}
    dates = wide["close"].index
    tickers = wide["close"].columns

    elig_d = (elig_m.reindex(columns=tickers)
              .reindex(index=dates, method="ffill").fillna(False))

    yz = _yang_zhang(wide["open"], wide["high"], wide["low"], wide["close"], vp.yz_window)

    # 스윙풀 = 그날 유니버스 편입 종목. 일별 중앙값 → 직전 252거래일 중앙값 = D(t)
    pool_med = yz.where(elig_d).median(axis=1)
    denom = pool_med.rolling(vp.pool_lookback, min_periods=vp.pool_lookback // 2).median()
    denom = denom.shift(1)                       # ⚠ 룩어헤드 방지 — t 시점엔 t−1 까지만 안다
    scale = yz.div(denom, axis=0).clip(*vp.clip)

    try:
        meta = pd.read_parquet(DATA / "meta.parquet")
        names = (meta["name"].groupby(level="ticker").last().reindex(tickers)
                 .fillna("").to_dict())
    except Exception:
        names = {}

    return Panel(wide["open"], wide["high"], wide["low"], wide["close"],
                 wide["volume"], elig_d, yz, scale, dates, tickers, names)


# ────────────────────────────────────────────────────────────────────────────
# 진입 신호
# ────────────────────────────────────────────────────────────────────────────


def _confirm_mask(p: Panel, kind: str) -> pd.DataFrame:
    cl, op, hi, lo = p.close, p.open, p.high, p.low
    if kind == "up1":
        return cl > cl.shift(1)
    if kind == "up2":
        up = cl > cl.shift(1)
        return up & up.shift(1).fillna(False)
    if kind == "mid":
        return cl > (hi + lo) / 2.0
    if kind == "up1_mid":
        return (cl > cl.shift(1)) & (cl > (hi + lo) / 2.0)
    if kind == "ma5":
        return cl > cl.rolling(5, min_periods=5).mean()
    raise ValueError(f"unknown confirm: {kind}")


def compute_signals(p: Panel, ep: EntryParams) -> dict:
    """진입 신호 계산. 반환 {'signal': bool DF, 'depth': DF}.

    signal = 그날 종가 기준 매수 신호 (t+1 시가 체결).
    v1.0.0 baseline 은 눌림 조건 성립 첫날(onset) 이 곧 signal.
    confirm 을 주면 같은 눌림 구간(episode) 안에서 확인 조건이 처음 참인 날로 미룬다.
    """
    cl = p.close
    ma = _trend_ma(p, ep)
    margin = cl / ma - 1.0                        # 추세 MA 대비 여력

    if ep.dip_basis == "high":
        high_n = cl.rolling(ep.high_window, min_periods=ep.high_min_periods).max()
        depth = (high_n - cl) / high_n
        if ep.vol_depth:
            req = (ep.pullback_pct * p.vol_scale).clip(*ep.vol_depth_clip)
        else:
            req = pd.DataFrame(ep.pullback_pct, index=cl.index, columns=cl.columns)
        dip_ok = depth >= req

        # 깊이별 계단: 각 임계의 onset 을 union (추세·유니버스는 공통 AND)
        if ep.pullback_tiers:
            gate = (margin > ep.trend_margin) & p.elig
            sig_u = sig_base = None
            for t in sorted(ep.pullback_tiers):
                c_t = ((depth >= t) & gate).fillna(False)
                o_t = c_t & ~c_t.shift(1).fillna(False)
                if sig_base is None:
                    sig_base = o_t                # 1차(가장 얕은) 티어 onset
                sig_u = o_t if sig_u is None else (sig_u | o_t)
            cond_any = ((depth >= min(ep.pullback_tiers)) & gate).fillna(False)
            return {"signal": sig_u, "depth": depth, "cond": cond_any,
                    "rank": depth if ep.priority == "depth" else margin,
                    "signal_base": sig_base}
    elif ep.dip_basis == "low_prox":
        low_n = cl.shift(1).rolling(ep.high_window,
                                    min_periods=ep.high_min_periods).min()
        thr = (1.0 + ep.low_tol) * low_n
        depth = (thr - cl) / thr                  # 임계 대비 깊이 (랭킹용, ≥0 이 신호)
        dip_ok = cl <= thr
    else:
        raise ValueError(f"unknown dip_basis: {ep.dip_basis}")

    cond = dip_ok & (margin > ep.trend_margin) & p.elig
    cond = cond.fillna(False)

    if ep.priority == "depth":
        rank = depth
    elif ep.priority == "depth_asc":
        rank = -depth
    elif ep.priority == "trend":
        rank = margin
    elif ep.priority == "vol":
        rank = p.vol_scale
    else:
        raise ValueError(f"unknown priority: {ep.priority}")

    onset = cond & ~cond.shift(1).fillna(False)

    # 재신호 — 같은 눌림 구간 안에서 resignal_gap 거래일마다 다시 신호
    if ep.resignal_gap > 0 and ep.confirm is None:
        c = cond.to_numpy()
        out = np.zeros_like(c)
        since = np.full(c.shape[1], 10**6, dtype=np.int64)
        for i in range(c.shape[0]):
            new_ep = c[i] & ~(c[i - 1] if i else np.zeros(c.shape[1], bool))
            since = np.where(new_ep, 10**6, since + 1)
            fire = c[i] & (new_ep | (since >= ep.resignal_gap))
            out[i] = fire
            since = np.where(fire, 0, since)
            since = np.where(~c[i], 10**6, since)
        return {"signal": pd.DataFrame(out, index=cond.index, columns=cond.columns),
                "depth": depth, "cond": cond, "rank": rank}

    if ep.confirm is None:
        return {"signal": onset, "depth": depth, "cond": cond, "rank": rank}

    ok = _confirm_mask(p, ep.confirm).fillna(False)

    # episode 안에서 확인 첫날만 True. numpy 루프(종목 수 ~수백, 날짜 ~3천 — 충분히 빠름)
    c = cond.to_numpy()
    o = onset.to_numpy()
    k = ok.to_numpy()
    out = np.zeros_like(c)
    n_d, n_t = c.shape
    waiting = np.zeros(n_t, dtype=bool)
    waited = np.zeros(n_t, dtype=np.int32)
    for i in range(n_d):
        waiting &= c[i]                       # 눌림 구간 끊기면 대기 종료
        started = o[i]
        waiting |= started
        waited = np.where(started, 0, waited + 1)
        fire = waiting & c[i] & k[i] & (waited <= ep.confirm_max_wait)
        out[i] = fire
        waiting &= ~fire                       # 한 episode 당 1회
        waiting &= waited <= ep.confirm_max_wait

    return {"signal": pd.DataFrame(out, index=cond.index, columns=cond.columns),
            "depth": depth, "cond": cond, "rank": rank}


# ────────────────────────────────────────────────────────────────────────────
# 포트폴리오 시뮬레이션
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class Position:
    ticker: str
    slots: int
    shares: int
    cost: float                 # 총 매입금액(수수료 포함)
    entry_idx: int              # 최근 매수일 인덱스 (D+0)
    first_entry_idx: int
    peak: float                 # 보유 중 관측 최고 (high 누적)
    scale: float                # 진입 신호일 변동배율


def _stop_line(pos: Position, i: int, xp: ExitParams) -> float:
    avg = pos.cost / max(pos.shares, 1)
    late = (i - pos.entry_idx) > xp.switch_day
    above = pos.peak > avg
    if not late:
        base, clip = xp.early_pct, xp.clip_early
    elif above:
        base, clip = xp.late_up_pct, xp.clip_late_up
    else:
        base, clip = xp.late_flat_pct, xp.clip_late_flat
    pct = base * (pos.scale if xp.vol_linked else 1.0)
    pct = min(max(pct, clip[0]), clip[1])
    ref = pos.peak if above else avg
    return ref * (1.0 - pct)


def run_backtest(p: Panel, ep: EntryParams, xp: ExitParams = ExitParams(),
                 pp: PortfolioParams = PortfolioParams(),
                 start: str = "2015-01-01", end: str = "2026-06-30") -> dict:
    sig = compute_signals(p, ep)
    signal, depth, rank = sig["signal"], sig["depth"], sig["rank"]
    # tiers_new_only: 보유 중 불타기는 1차 티어 onset 만 허용
    pyr_sig = (sig["signal_base"] if ep.tiers_new_only and "signal_base" in sig
               else signal)
    ps = pyr_sig.to_numpy()

    mask = (p.dates >= pd.Timestamp(start)) & (p.dates <= pd.Timestamp(end))
    idxs = np.flatnonzero(mask)
    dates = p.dates

    op = p.open.to_numpy(); hi = p.high.to_numpy()
    lo = p.low.to_numpy(); cl = p.close.to_numpy()
    sg = signal.to_numpy(); dp = depth.to_numpy(); sc = p.vol_scale.to_numpy()
    rk = rank.to_numpy()
    tick = list(p.tickers)
    col = {t: j for j, t in enumerate(tick)}

    cash = pp.capital_krw
    pending_cash = 0.0                     # T+1 결제 대기
    pos: dict[str, Position] = {}
    last_sell_i: dict[str, int] = {}
    trades: list[dict] = []
    equity: list[tuple] = []

    for i in idxs:
        cash += pending_cash
        pending_cash = 0.0

        # ── 1) 손절 (당일 저가로 감지, 전일까지의 peak 기준) ──
        for tk in list(pos):
            j = col[tk]
            pz = pos[tk]
            line = _stop_line(pz, i, xp)
            if np.isnan(lo[i, j]):
                continue
            if lo[i, j] <= line:
                fill = min(op[i, j], line) if not np.isnan(op[i, j]) else line
                fill *= (1.0 - pp.stop_slippage)
                gross = fill * pz.shares
                net = gross * (1.0 - pp.sell_fee - pp.sell_tax)
                pending_cash += net
                trades.append(dict(date=dates[i], ticker=tk, side="SELL",
                                   shares=pz.shares, price=fill, amount=net,
                                   pnl=net - pz.cost, hold_days=i - pz.first_entry_idx,
                                   ret=(net - pz.cost) / pz.cost))
                last_sell_i[tk] = i
                del pos[tk]

        # ── 2) 매수 (전일 신호 → 오늘 시가) ──
        if i > 0:
            fired = np.flatnonzero(sg[i - 1])
            if fired.size:
                cands = []
                for j in fired:
                    tk = tick[j]
                    if np.isnan(op[i, j]) or op[i, j] <= 0:
                        continue
                    held = pos.get(tk)
                    if held is not None:
                        if held.slots >= pp.max_slots_per_stock:
                            continue
                        if not ps[i - 1, j]:      # 불타기 허용 신호가 아니면 skip
                            continue
                    else:
                        ls = last_sell_i.get(tk)
                        if ls is not None and (i - ls) < pp.cooldown_trading_days:
                            continue
                    r = rk[i - 1, j]
                    cands.append((r if np.isfinite(r) else -1e18, j, tk))
                cands.sort(reverse=True)          # ep.priority 기준 내림차순

                used = sum(q.slots for q in pos.values())
                for d_, j, tk in cands:
                    if used >= pp.max_positions:
                        break
                    price = op[i, j]
                    shares = int(pp.slot_krw // price)
                    if shares <= 0:
                        continue
                    cost = price * shares * (1.0 + pp.buy_fee)
                    if cost > cash:
                        shares = int(cash / (price * (1.0 + pp.buy_fee)))
                        if shares <= 0:
                            continue
                        cost = price * shares * (1.0 + pp.buy_fee)
                    cash -= cost
                    used += 1
                    s = sc[i - 1, j]
                    s = 1.0 if not np.isfinite(s) else float(s)
                    if tk in pos:
                        q = pos[tk]
                        q.slots += 1
                        q.shares += shares
                        q.cost += cost
                        q.entry_idx = i                       # 불타기 → D+ 리셋
                        q.peak = max(q.peak, price)
                        q.scale = s
                    else:
                        pos[tk] = Position(tk, 1, shares, cost, i, i, price, s)
                    trades.append(dict(date=dates[i], ticker=tk, side="BUY",
                                       shares=shares, price=price, amount=cost,
                                       pnl=np.nan, hold_days=np.nan, ret=np.nan))

        # ── 3) peak 갱신 + 평가 ──
        mv = 0.0
        for tk, q in pos.items():
            j = col[tk]
            if not np.isnan(hi[i, j]):
                q.peak = max(q.peak, hi[i, j])
            c = cl[i, j]
            mv += (c if not np.isnan(c) else q.cost / max(q.shares, 1)) * q.shares
        equity.append((dates[i], cash + pending_cash + mv, cash, mv, len(pos)))

    eq = pd.DataFrame(equity, columns=["date", "equity", "cash", "mv", "n_pos"]).set_index("date")
    tr = pd.DataFrame(trades)
    return {"equity": eq, "trades": tr, "params": ep}


# ────────────────────────────────────────────────────────────────────────────
# 성과 지표
# ────────────────────────────────────────────────────────────────────────────


def metrics(res: dict, capital: float = 100_000_000.0) -> dict:
    eq = res["equity"]["equity"]
    tr = res["trades"]
    ret = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / capital) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    dd = eq / eq.cummax() - 1
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan
    sells = tr[tr.side == "SELL"] if len(tr) else pd.DataFrame()
    # ⚠ 평균 투자비중 — 진입을 조이면 슬롯이 놀아 현금비중이 커지고, MDD·Sharpe 가
    #   '덜 투자해서' 좋아 보이는 착시가 생긴다 (OOS_2026-07_review §5 의 함정).
    #   규칙 비교 시 반드시 함께 볼 것.
    invested = (res["equity"]["mv"] / res["equity"]["equity"]).mean()
    return {
        "final": float(eq.iloc[-1]),
        "profit": float(eq.iloc[-1] - capital),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "mdd": float(dd.min()),
        "invested": float(invested),
        "n_buy": int((tr.side == "BUY").sum()) if len(tr) else 0,
        "n_sell": int(len(sells)),
        "win_rate": float((sells.pnl > 0).mean()) if len(sells) else np.nan,
        "avg_hold": float(sells.hold_days.mean()) if len(sells) else np.nan,
        "avg_ret": float(sells.ret.mean()) if len(sells) else np.nan,
    }


def fmt(m: dict, label: str = "") -> str:
    return (f"{label:<28} 최종 {m['final']/1e8:5.2f}억  CAGR {m['cagr']*100:6.2f}%  "
            f"Sharpe {m['sharpe']:5.2f}  MDD {m['mdd']*100:7.2f}%  "
            f"매수 {m['n_buy']:4d} 청산 {m['n_sell']:4d}  승률 {m['win_rate']*100:5.1f}%  "
            f"평균보유 {m['avg_hold']:5.1f}일")


if __name__ == "__main__":
    up, vp = UniverseParams(), VolScaleParams()
    print("패널 로드 중...")
    panel = load_panel(up, vp)
    print(f"  종목 {len(panel.tickers)} · 거래일 {len(panel.dates)} "
          f"({panel.dates[0].date()}~{panel.dates[-1].date()})")
    base = run_backtest(panel, EntryParams())
    print(fmt(metrics(base), "v1.2.2.3 (4조30%·W1·청산③)"))
    # 기대값 (2026-08-17 확정): 최종 2.03억 · Sharpe 0.68 · MDD −23.2% · 매수 1,500
