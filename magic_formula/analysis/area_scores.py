"""
analysis/area_scores.py
-----------------------
4영역(추세·모멘텀·거래량·변동성) 확정 점수 함수 통합 모듈.

각 영역 분석(scripts/test_*) 에서 도출한 best 신호를 한 곳에 모아
5영역 종합 시뮬레이션에서 호출한다. (Wyckoff 는 별도 hillstorm.)

영역별 확정 spec (docs/area_specs/*.md)
---------------------------------------
- 추세 (trend):     Dv2(정30/크30/기40) + invert_dist_off_bull (breadth 레짐)
- 모멘텀 (momentum): RSI 10/90 극단 trend 단독 (레짐 없음)
- 거래량 (volume):   bear-only (Q2+Q3+OBV_contra) (quickregime)
- 변동성 (volatility): BB×52주×레짐 결합 점수표 (quickregime)

모두 ±10 풀스케일. 레짐 인자는 시점별 라벨 Series 로 주입.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from magic_formula.scoring.scorer import _rsi, _obv, _clip, _macd, _stoch_k
from magic_formula.analysis.trend_variants import score_D_v2
from magic_formula.analysis import volatility_variants as VLV


# ===========================================================================
# 추세 (Trend) — Dv2 + invert_dist_off_bull
# ===========================================================================

def score_trend(df: pd.DataFrame, regime_ser: pd.Series) -> pd.Series:
    """
    Dv2(정30/크30/기40) 기본 점수에 breadth 레짐 적응:
    강세지속→off(0) / 강세약화→invert(×-1) / 조정·하락→그대로.

    regime_ser : breadth 레짐 (추세 영역 breadth 10/10/0.60).
    """
    base = score_D_v2(df, (0.3, 0.3, 0.4))
    rg = regime_ser.reindex(base.index).ffill()
    out = base.copy()
    out.loc[rg == "강세지속"] = 0.0
    out.loc[rg == "강세약화"] = -base.loc[rg == "강세약화"]
    return _clip(out).fillna(0.0)


# ===========================================================================
# 모멘텀 (Momentum) — RSI 10/90 극단 trend 단독
# ===========================================================================

def score_momentum(df: pd.DataFrame) -> pd.Series:
    """RSI 5-band 극단(10/90) trend, ±10. 레짐 없음 (상시)."""
    if len(df) < 35:
        return pd.Series(0.0, index=df.index)
    rsi = _rsi(df["Close"])
    low, high = 10, 90
    mid_hi, mid_lo = (50 + high) / 2, (50 + low) / 2
    s = pd.Series(np.nan, index=df.index)
    v = rsi.notna(); s.loc[v] = 0.0
    s.loc[v & (rsi >= high)]                   =  10.0
    s.loc[v & (rsi >= mid_hi) & (rsi < high)]  =   5.0
    s.loc[v & (rsi > mid_lo) & (rsi < mid_hi)] =   0.0
    s.loc[v & (rsi > low) & (rsi <= mid_lo)]   =  -5.0
    s.loc[v & (rsi <= low)]                    = -10.0
    return _clip(s).fillna(0.0)


# ===========================================================================
# 거래량 (Volume) — bear-only (Q2+Q3+OBV_contra), quickregime
# ===========================================================================

_PC = 5
_VOL_HIGH, _VOL_LOW = 1.5, 0.7


def _rel_vol(df):
    if "Rel_Volume" in df.columns and df["Rel_Volume"].notna().any():
        return df["Rel_Volume"]
    vm = df["Volume"].rolling(20).mean()
    return df["Volume"] / vm.replace(0, np.nan)


def _q2(df):
    ret = df["Close"].pct_change(_PC); rv = _rel_vol(df)
    s = pd.Series(0.0, index=df.index)
    s.loc[(ret > 0) & (rv < _VOL_LOW)] = -10.0   # 관심 식은 상승
    return s


def _q3(df):
    ret = df["Close"].pct_change(_PC); rv = _rel_vol(df)
    s = pd.Series(0.0, index=df.index)
    s.loc[(ret < 0) & (rv > _VOL_HIGH)] = -10.0  # 투매
    return s


def _obv_contra(df):
    obv = _obv(df["Close"], df["Volume"]); slope = obv - obv.shift(5)
    std = slope.rolling(60).std(); z = slope / std.replace(0, np.nan)
    return _clip(-10.0 * np.tanh(z)).fillna(0.0)


def score_volume(df: pd.DataFrame, regime_ser: pd.Series) -> pd.Series:
    """
    bear-only: 강세장 → 0, 하락·조정장 → (Q2+Q3+OBV_contra)/3.

    regime_ser : quickregime (3/5/0.52).
    """
    if len(df) < 25:
        return pd.Series(0.0, index=df.index)
    rg = regime_ser.reindex(df.index).ffill()
    bear = _clip((_q2(df) + _q3(df) + _obv_contra(df)) / 3.0)
    out = pd.Series(0.0, index=df.index)
    out.loc[rg.isin(["조정", "하락"])] = bear.loc[rg.isin(["조정", "하락"])]
    return _clip(out).fillna(0.0)


# ===========================================================================
# 변동성 (Volatility) — BB×52주×레짐 결합 점수표, quickregime
# ===========================================================================

def score_volatility(df: pd.DataFrame, regime_ser: pd.Series) -> pd.Series:
    """BB %B × 52주 위치 × 레짐 결합 점수표 (±10). quickregime."""
    return VLV.score_joint_regime(df, regime_ser)


# ===========================================================================
# 레짐 판별기 — 영역별 2종 (추세=breadth, 거래량/변동성=quickregime)
# ===========================================================================

from magic_formula.analysis.ic_framework import compute_breadth_series  # noqa: E402


def _make_regime(stock_data: dict[str, pd.DataFrame],
                 lookback: int, b_horizon: int, high_thr: float,
                 low_thr: float = 0.40, trend_lb: int = 5) -> pd.Series:
    """
    breadth 기반 4-mode 레짐 라벨 시계열.
    강세지속 / 강세약화 / 조정 / 하락 / unknown.
    """
    b = compute_breadth_series(stock_data, lookback=lookback, horizon=b_horizon)
    trend = b.diff(trend_lb)
    labels = pd.Series(index=b.index, dtype=object)
    for ts, v in b.items():
        if pd.isna(v):
            labels[ts] = "unknown"
        elif v > high_thr:
            labels[ts] = "강세지속" if (pd.isna(trend.get(ts)) or trend.get(ts) >= 0) else "강세약화"
        elif v < low_thr:
            labels[ts] = "하락"
        else:
            labels[ts] = "조정"
    return labels


def make_regimes(stock_data: dict[str, pd.DataFrame]) -> tuple[pd.Series, pd.Series]:
    """
    영역별 레짐 2종 반환.

    Returns
    -------
    (regime_breadth, regime_quick)
        regime_breadth : 추세 영역용 (lookback=10, horizon=10, HIGH=0.60)
        regime_quick   : 거래량·변동성 영역용 (lookback=3, horizon=5, HIGH=0.52)
    """
    regime_breadth = _make_regime(stock_data, 10, 10, 0.60)
    regime_quick   = _make_regime(stock_data, 3, 5, 0.52)
    return regime_breadth, regime_quick


# ===========================================================================
# 종합 점수 — robust 가중치 + Markdown 게이트 (결합 시스템 단일 진입점)
# ===========================================================================

# 확정 가중치 (M4 분석 2026-05-30, robust grid search 최적)
# 그리드 robust(상위5제외) 최적: T20/M20/Vu0/Va60
COMBINED_WEIGHTS = {"trend": 0.2, "momentum": 0.2, "volume": 0.0, "volatility": 0.6}
COMBINED_THRESHOLD = 6.0   # 확정 (5.0 후보)
GATE_EXCLUDE_PHASES = ("Markdown",)   # 매수 제외 국면


def compute_combined_score(
    df:             pd.DataFrame,
    regime_breadth: pd.Series,
    regime_quick:   pd.Series,
    phase_label:    pd.Series,
    weights:        dict[str, float] | None = None,
    gate:           bool = True,
) -> pd.Series:
    """
    4영역 가중 결합 종합 점수 + Wyckoff 국면 게이트.

    종합점수 = (wt·추세 + wm·모멘텀 + wvu·거래량 + wva·변동성) / Σw, ±10 클립.
    게이트 ON 이면 Wyckoff 국면이 GATE_EXCLUDE_PHASES (기본 Markdown) 일 때
    점수를 NaN 으로 (매수 후보 제외).

    Parameters
    ----------
    df             : OHLCV (full-column, 거래량/BB/심리지표 포함)
    regime_breadth : 추세 영역 레짐 (make_regimes 첫번째)
    regime_quick   : 거래량·변동성 영역 레짐 (make_regimes 두번째)
    phase_label    : hillstorm Wyckoff_Label Series (게이트용)
    weights        : {trend,momentum,volume,volatility}. None 이면 COMBINED_WEIGHTS.
    gate           : Markdown 게이트 적용 여부.

    Returns
    -------
    종합 점수 Series (게이트 제외 구간은 NaN).
    """
    if weights is None:
        weights = COMBINED_WEIGHTS
    st = score_trend(df, regime_breadth)
    sm = score_momentum(df)
    sv = score_volume(df, regime_quick)
    sp = score_volatility(df, regime_quick)
    wt, wm, wvu, wva = (weights["trend"], weights["momentum"],
                        weights["volume"], weights["volatility"])
    wsum = wt + wm + wvu + wva
    comp = _clip((wt * st + wm * sm + wvu * sv + wva * sp) / wsum)
    if gate and phase_label is not None:
        comp = comp.where(~phase_label.reindex(comp.index).isin(GATE_EXCLUDE_PHASES))
    return comp
