"""
analysis/area_scores.py
-----------------------
4영역(추세·모멘텀·거래량·변동성) 확정 점수 함수 통합 모듈.

각 영역 분석(scripts/test_*) 에서 도출한 best 신호를 한 곳에 모아
5영역 종합 시뮬레이션에서 호출한다. (Wyckoff 는 별도 hillstorm.)

영역별 확정 spec (docs/area_specs/*.md)
---------------------------------------
- 추세 (trend):     Dv2(정30/크30/기40) + invert_dist_off_bull (breadth 레짐)
- 모멘텀 (momentum): (RSI(14) − 50) / 5 — 연속 선형 (레짐 없음)        ← v3
- 거래량 (volume):   bear-only (Q2+Q3+OBV_contra) (quickregime)
- 변동성 (volatility): 52주 위치 0.6 + BB %B 꺾임 0.4 (레짐 미사용)     ← v3

모두 ±10 풀스케일. 레짐 인자는 시점별 라벨 Series 로 주입.

v3 (2026-08-25) — COMBINED-v3-2026-08
--------------------------------------
근거: `StockPortfolio/reports/황금률_전구간_재검증_20260824.html` §5·§6·§14
재현·검증: `scripts/validate_area_redesign.py`

v2 는 상위 3% 20일 상대수익이 4개 연도 모두 음수였다(코어 67 기준 −1.29%, 0/4).
주원인은 변동성 점수표가 52주 **저점** 버킷에 최고점을 준 것 — 만든 구간에서도
틀린 평가 기준 오류다. v3 로 +4.30% (4/4, p<0.001).

⚠ **이 종합점수는 예측 모델이 아니라 상위 선별기다.**
   20일 상대수익과의 순위상관(IC)은 +0.007(p=0.25)로 유의하지 않다. 실질 이득은
   10분위 중 최상위에 몰려 있다. "점수가 높을수록 수익이 좋다" 로 읽으면 안 된다.
   (v2 는 IC −0.015, p=0.0004 로 **유의하게 음수**였다 — 부호는 뒤집혔다.)

⚠ 구버전 함수는 롤백용으로 남겨 뒀다 —
   score_momentum_step() / score_volatility_table() / COMBINED_WEIGHTS_V2.

⚠ 모든 검증 수치는 유니버스 205종목 **현재 명단**(생존편향) · 2023-05~2026-08
   **초강세장** 구간에서 나왔다. point-in-time 재검증과 약세장 백필 전에는
   확정값으로 쓰지 말 것 (HANDOFF §5.1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from magic_formula.indicators import _rsi, _obv, _clip, _macd, _stoch_k  # noqa: F401 (_macd/_stoch_k 는 변형 연구 호환)
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
    """
    RSI(14) 를 ±10 으로 선형 매핑. (RSI − 50) / 5. 레짐 없음 (상시).

    v3 (2026-08-25) — 구 5-band 계단에서 연속 선형으로 교체.
    자유 파라미터가 없다: RSI 0~100 이 그대로 −10~+10 에 얹힌다.

    왜 바꿨나
    ---------
    계단은 고유값이 0/±5/±10 셋뿐이라 모멘텀 비중 80% 에서 **꼭대기 변별이
    안 된다**. 상위 3% 20일 상대수익 +3.00% → +4.45% (4/4 연도 양수).
    기울기 k 를 3.5~6.0 으로 흔들어도 +4.42~4.48% 로 평탄 — 칼날 최적점이 아니다.

    ⚠ 이 축은 20일 상대수익과의 **순위상관(IC)이 0 이다** (+0.001, p=0.86).
      전체 순위를 맞히는 축이 아니라 **꼭대기를 고르는 축**이다. 점수가 높을수록
      수익이 좋다고 읽으면 안 된다 — 8~10분위에서만 올라간다.

    구 계단은 score_momentum_step() 으로 보존 (롤백용).
    """
    if len(df) < 35:
        return pd.Series(0.0, index=df.index)
    return _clip((_rsi(df["Close"]) - 50.0) / 5.0).fillna(0.0)


def score_momentum_step(df: pd.DataFrame) -> pd.Series:
    """구 v2 — RSI 5-band 극단(10/90) trend, ±10. 롤백용 보존."""
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
    """
    52주 위치(0.6) + BB %B 꺾임 반영(0.4). ±10. **레짐을 쓰지 않는다.**

    v3 (2026-08-25) — 구 BB×52주×레짐 결합 점수표에서 교체 (08-24 리포트 후보 A2).

    왜 바꿨나
    ---------
    구 점수표는 52주 **저점** 버킷에 최고점을 주고 있었고, 그 방향이 만든 구간
    (in-sample) 에서도 틀렸다. 종합점수 IC 가 4개 연도 모두 음수였던 주원인이다.
    교체 후 상위 3% 20일 상대수익 −0.89% → +2.41%, 4/4 연도 양수, p<0.001.

    BB 구간 임계(1.0/0.8/0.6/0.2/0)는 실측 꺾임 지점에서 왔다. 밴드를 **돌파한**
    구간(>1.0)에 3점만 주는 게 핵심 — 상위 5% 의 38% 가 밴드 돌파 종목인데
    이들 수익(+1.72%)이 밴드 상단 구간(+2.03%)보다 낮다.

    ⚠ regime_ser 는 호출 계약 유지를 위해 받기만 하고 쓰지 않는다.
    ⚠ 52주:BB 배분은 둔감하다 — 0.8:0.2~0.4:0.6 에서 MDD −18.3~−20.7%,
      Sharpe 2.21~2.48 로 평탄. 0.6:0.4 는 그 고원 안의 한 점이다.

    구 점수표는 score_volatility_table() 으로 보존 (롤백용).
    """
    p52 = VLV._pos_52w(df)
    bb = VLV._bb_pctb(df)
    s52 = (p52 - 0.5) * 20.0
    sbb = pd.Series(
        np.select(
            [bb > 1.0, bb >= 0.8, bb >= 0.6, bb >= 0.2, bb >= 0.0],
            [    3.0,     10.0,      5.0,      0.0,     -3.0],
            default=2.0,
        ),
        index=df.index,
    )
    return _clip(0.6 * s52 + 0.4 * sbb).fillna(0.0)


def score_volatility_table(df: pd.DataFrame, regime_ser: pd.Series) -> pd.Series:
    """구 v2 — BB %B × 52주 위치 × 레짐 결합 점수표 (±10). 롤백용 보존."""
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

# 확정 가중치 v3 (2026-08-25) — T0/M80/Vu0/Va20
#
# 08-24 리포트 §6 walk-forward 5/5 fold 동일 배합. 실측으로도 고원이다 —
# M50→M80 은 단조 개선이고 M80~M100 은 평탄(누적 +1,281~1,302%). M80 이
# MDD 최저(−18.7%)·Sharpe 최고(2.36) 라 그 자리를 택했다.
# 추세를 되살리면 단조 악화: T0 −18.7% → T10 −22.2% → T20 −29.4% (MDD).
#
# ⚠ 추세·거래량은 계산은 계속 하되 종합점수에 들어가지 않는다 (가중 0).
#   Quickview 4영역 배지는 그대로 보이지만 뒤 둘은 참고값이다.
# ⚠ 구 v2 배합은 T20/M20/Vu0/Va60 — configs/active_strategy_v2.yaml 에 보존.
COMBINED_WEIGHTS = {"trend": 0.0, "momentum": 0.8, "volume": 0.0, "volatility": 0.2}
COMBINED_WEIGHTS_V2 = {"trend": 0.2, "momentum": 0.2, "volume": 0.0, "volatility": 0.6}

# 임계는 v2 그대로 유지한다 — 실측이 지지한다.
# 임계별 실운용(슬롯10·t+1시가·왕복0.43%): 4.5 −34.0% / 5.0 −25.8% / 5.5 −21.4%
# / 6.0 −20.4% / 6.5 −19.9% / 7.0 −16.7% (MDD). 누적·Sharpe 는 6.0 이 최고
# (+1,365%, 2.41). MDD 가 임계에 대해 **단조**라 이 스윕은 믿을 만하다.
# 신호량은 일평균 11.4 → 5.0 종목으로 줄지만 신호 있는 날이 81.6% 라 무방.
COMBINED_THRESHOLD = 6.0   # 확정 (5.0 후보)
GATE_EXCLUDE_PHASES = ("Markdown",)   # 매수 제외 국면

AREA_KEYS = ("trend", "momentum", "volume", "volatility")


def compute_area_scores(
    df:             pd.DataFrame,
    regime_breadth: pd.Series,
    regime_quick:   pd.Series,
) -> dict[str, pd.Series]:
    """
    4영역 점수를 한 번에 계산해서 dict 로 반환한다.

    가중치 조합이 바뀌어도 영역 점수는 동일하므로, 그리드 백테스트나
    데일리 리포트에서는 이 결과를 캐시하고 combine_scores() 로 결합만
    반복하면 중복 계산이 없다.
    """
    return {
        "trend":      score_trend(df, regime_breadth),
        "momentum":   score_momentum(df),
        "volume":     score_volume(df, regime_quick),
        "volatility": score_volatility(df, regime_quick),
    }


def combine_scores(
    areas:          dict[str, pd.Series],
    weights:        dict[str, float] | None = None,
    phase_label:    pd.Series | None = None,
    gate:           bool = True,
    exclude_phases: tuple[str, ...] = GATE_EXCLUDE_PHASES,
) -> pd.Series:
    """
    compute_area_scores() 결과를 가중 결합 + Wyckoff 게이트 적용.

    종합점수 = Σ(w_i · area_i) / Σw, ±10 클립.
    게이트 ON 이면 Wyckoff 국면이 exclude_phases (기본 Markdown) 일 때
    점수를 NaN 으로 (매수 후보 제외).
    """
    if weights is None:
        weights = COMBINED_WEIGHTS
    wsum = sum(weights[k] for k in AREA_KEYS)
    if wsum <= 0:
        raise ValueError(f"가중치 합이 0 이하: {weights}")
    acc = None
    for k in AREA_KEYS:
        term = weights[k] * areas[k]
        acc = term if acc is None else acc + term
    comp = _clip(acc / wsum)
    if gate and phase_label is not None:
        comp = comp.where(~phase_label.reindex(comp.index).isin(exclude_phases))
    return comp


def compute_combined_score(
    df:             pd.DataFrame,
    regime_breadth: pd.Series,
    regime_quick:   pd.Series,
    phase_label:    pd.Series,
    weights:        dict[str, float] | None = None,
    gate:           bool = True,
    exclude_phases: tuple[str, ...] = GATE_EXCLUDE_PHASES,
) -> pd.Series:
    """
    4영역 가중 결합 종합 점수 + Wyckoff 국면 게이트 (단일 진입점).

    내부적으로 compute_area_scores() + combine_scores() 를 호출한다.
    영역 점수를 재사용하려면 두 함수를 직접 쓰는 편이 효율적이다.

    Returns
    -------
    종합 점수 Series (게이트 제외 구간은 NaN).
    """
    areas = compute_area_scores(df, regime_breadth, regime_quick)
    return combine_scores(areas, weights, phase_label, gate, exclude_phases)
