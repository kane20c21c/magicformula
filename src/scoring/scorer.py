"""
scoring/scorer.py
-----------------
5개 영역(추세·모멘텀·거래량·변동성·심리·Wyckoff)별 점수(-10~+10)를 산출하고,
가중치(weights dict)로 합산하여 종합 점수(-10~+10)를 반환한다.

Look-ahead bias 방지 원칙
- 모든 지표는 rolling/ewm 등 과거 데이터만 사용하는 연산으로 계산한다.
- t시점 점수는 index[t]까지의 데이터만 반영한다.

Wyckoff hillstorm 경로: /Users/kaneyoun/DriveForALL/StoLab/hillstorm/
- import 실패 시 Area 5 점수를 0으로 대체하고 경고 출력 후 계속 실행.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Wyckoff hillstorm import (optional — fallback to stub)
# ---------------------------------------------------------------------------

HILLSTORM_PATH = "/Users/kaneyoun/DriveForALL/StoLab/hillstorm"

_wyckoff_available = False
compute_indicators = classify_wyckoff = detect_signals = None

try:
    if HILLSTORM_PATH not in sys.path:
        sys.path.insert(0, HILLSTORM_PATH)
    from wyckoff_analysis import (  # type: ignore
        compute_indicators,
        classify_wyckoff,
        detect_signals,
    )
    _wyckoff_available = True
    print("[scorer] Wyckoff hillstorm 모듈 로드 완료")
except ImportError:
    warnings.warn(
        f"[scorer] wyckoff_analysis 모듈을 찾을 수 없습니다 ({HILLSTORM_PATH}). "
        "Area 5 점수를 0으로 대체합니다.",
        stacklevel=2,
    )

# ---------------------------------------------------------------------------
# 기본 가중치 (Basic)
# ---------------------------------------------------------------------------

BASIC_WEIGHTS: dict[str, float] = {
    "trend":      0.20,
    "momentum":   0.25,
    "volume":     0.30,
    "volatility": 0.10,
    "wyckoff":    0.15,
}

assert abs(sum(BASIC_WEIGHTS.values()) - 1.0) < 1e-9, (
    f"BASIC_WEIGHTS 합계 오류: {sum(BASIC_WEIGHTS.values())}"
)


# ---------------------------------------------------------------------------
# 내부 유틸
# ---------------------------------------------------------------------------

def _clip(series: pd.Series, lo: float = -10.0, hi: float = 10.0) -> pd.Series:
    """시리즈 값을 [lo, hi]로 클리핑."""
    return series.clip(lower=lo, upper=hi)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder 방식 RSI (EWM alpha=1/period)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _stoch_k(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Stochastic %K."""
    lo = low.rolling(period).min()
    hi = high.rolling(period).max()
    denom = (hi - lo).replace(0.0, np.nan)
    return (close - lo) / denom * 100.0


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD, Signal, Histogram 반환."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(close: pd.Series, period: int = 20, std_mult: float = 2.0):
    """BB 중심선, 상단, 하단, %B 반환."""
    ma = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=1)
    upper = ma + std_mult * std
    lower = ma - std_mult * std
    denom = (upper - lower).replace(0.0, np.nan)
    pct_b = (close - lower) / denom
    return ma, upper, lower, pct_b


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ATR (Average True Range)."""
    hl = high - low
    hpc = (high - close.shift(1)).abs()
    lpc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


# ---------------------------------------------------------------------------
# Area 1 — 추세 (Trend)
# ---------------------------------------------------------------------------

def score_trend(df: pd.DataFrame) -> pd.Series:
    """
    추세 영역 점수 (-10 ~ +10).

    구성 요소:
      - MA 정배열: MA5>MA20>MA60 → +6 / MA5>MA20 → +3 / MA5<MA20<MA60 → -6 / 기타 → 0
      - MA 크로스(최근 5일): 골든크로스 → +4 / 데드크로스 → -4
      - MA60 기울기: 5일 기울기 백분율 × 30 → [-6, +6] 클리핑
    """
    close = df["Close"]

    # --- Moving averages ---
    ma5  = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    # MA 정배열 점수
    align = pd.Series(0.0, index=close.index)
    align[ma5 > ma20] = 3.0
    align[(ma5 > ma20) & (ma20 > ma60)] = 6.0
    align[(ma5 < ma20) & (ma20 < ma60)] = -6.0

    # MA 크로스 (골든/데드 크로스, 최근 5일 지속)
    golden = ((ma5.shift(1) <= ma20.shift(1)) & (ma5 > ma20)).astype(float) * 4.0
    dead   = ((ma5.shift(1) >= ma20.shift(1)) & (ma5 < ma20)).astype(float) * (-4.0)
    cross_event = golden + dead
    # 이벤트 발생일로부터 최대 5일간 신호 유지 (가장 최신 크로스가 우선)
    cross_score = (
        cross_event.replace(0.0, np.nan)
        .ffill(limit=4)
        .fillna(0.0)
    )

    # MA60 기울기 (5일 변화율 %)
    ma60_slope_pct = (ma60 - ma60.shift(5)) / ma60.shift(5).replace(0, np.nan) * 100.0
    slope_score = (ma60_slope_pct * 30.0).clip(-6.0, 6.0)

    raw = (align + cross_score + slope_score) / 3.0
    return _clip(raw)


# ---------------------------------------------------------------------------
# Area 2 — 모멘텀 (Momentum)  ★ 추세추종 방향
# ---------------------------------------------------------------------------

def score_momentum(df: pd.DataFrame) -> pd.Series:
    """
    모멘텀 영역 점수 (-10 ~ +10) — **추세추종(trend-following)** 관점.

    구성 요소 (4개 단순 평균):
      ① RSI(14) 선형 변환: (RSI - 50) / 5
         → RSI 80 = +6,  RSI 50 = 0,  RSI 20 = -6  (클램핑 -10~+10)
         → 해석: 높은 RSI = 강한 상승 모멘텀 (과매수 경고 아님)

      ② Stoch %K(14) 선형 변환: (%K - 50) / 5
         → 동일 스케일, RSI와 같은 추세추종 해석

      ③ MACD vs Signal: MACD > Signal → +5 / < Signal → -5
         (기존 유지 — 이미 추세추종 방향)

      ④ MACD Histogram 3일 방향: 연속 상승 → +2 / 연속 하락 → -2
         (기존 유지 — 이미 추세추종 방향)

    변경 전(역추세)과 비교:
      RSI 70 이상 추세 상승 중: 이전 -7~-10  →  이제 +4~+6
      → Area2 가 Area1/Area4 와 같은 방향으로 종합점수 기여
    """
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    # ① RSI → 추세추종 선형 점수
    rsi_vals  = _rsi(close)
    rsi_score = _clip((rsi_vals - 50.0) / 5.0)

    # ② Stochastic %K → 추세추종 선형 점수
    stoch_vals  = _stoch_k(high, low, close)
    stoch_score = _clip((stoch_vals - 50.0) / 5.0)

    # ③ MACD vs Signal (기존 유지)
    macd_line, signal_line, hist = _macd(close)
    macd_vs_signal = pd.Series(
        np.where(macd_line > signal_line, 5.0, -5.0), index=close.index
    )

    # ④ MACD Histogram 3일 방향 (기존 유지)
    h1 = hist.shift(0)
    h2 = hist.shift(1)
    h3 = hist.shift(2)
    hist_dir = pd.Series(0.0, index=close.index)
    hist_dir[(h1 > h2) & (h2 > h3)] =  2.0
    hist_dir[(h1 < h2) & (h2 < h3)] = -2.0

    raw = (rsi_score + stoch_score + macd_vs_signal + hist_dir) / 4.0
    return _clip(raw)


# ---------------------------------------------------------------------------
# Area 3 — 거래량 (Volume)
# ---------------------------------------------------------------------------

def score_volume(df: pd.DataFrame) -> pd.Series:
    """
    거래량 영역 점수 (-10 ~ +10).

    구성 요소:
      - 상대거래량 (Vol / Vol_MA20) → 구간별 점수
      - OBV 방향 (5일 기울기): 상승 → +5 / 하락 → -5
    """
    close  = df["Close"]
    volume = df["Volume"]

    # 상대거래량
    vol_ma20 = volume.rolling(20).mean()
    rel_vol  = volume / vol_ma20.replace(0, np.nan)

    rel_score = pd.Series(0.0, index=close.index)
    rel_score[rel_vol > 2.0]                           = 8.0
    rel_score[(rel_vol > 1.5) & (rel_vol <= 2.0)]     = 5.0
    rel_score[(rel_vol > 1.0) & (rel_vol <= 1.5)]     = 2.0
    rel_score[(rel_vol > 0.7) & (rel_vol <= 1.0)]     = 0.0
    rel_score[rel_vol <= 0.7]                          = -3.0

    # OBV 방향
    obv_vals  = _obv(close, volume)
    obv_slope = obv_vals - obv_vals.shift(5)
    obv_score = pd.Series(np.where(obv_slope > 0, 5.0, -5.0), index=close.index)
    obv_score[obv_slope.isna()] = 0.0

    raw = (rel_score + obv_score) / 2.0
    return _clip(raw)


# ---------------------------------------------------------------------------
# Area 4 — 변동성·위치 (Volatility / Position)
# ---------------------------------------------------------------------------

def _bb_pos_raw(df: pd.DataFrame):
    """BB %B와 52주 위치 공통 계산. (pct_b, pos_52w) 반환."""
    close = df["Close"]

    _, _, _, pct_b = _bollinger(close)

    w52_high = close.rolling(252, min_periods=60).max()
    w52_low  = close.rolling(252, min_periods=60).min()
    denom    = (w52_high - w52_low).replace(0, np.nan)
    pos_52w  = (close - w52_low) / denom

    return pct_b, pos_52w


def score_volatility(df: pd.DataFrame) -> pd.Series:
    """
    변동성·위치 영역 점수 — **평균회귀(mean-reversion)** 관점 (-10 ~ +10).

    해석: BB 하단 근처 / 52주 저점 근처 = 과매도 = 매수 기회 (+)
          BB 상단 근처 / 52주 고점 근처 = 과매수 = 위험 (-)

    BB %B  : <0.1→+8, 0.1~0.2→+5, 0.2~0.4→+2, 0.4~0.6→0,
             0.6~0.8→-2, 0.8~0.9→-5, >0.9→-8
    52주   : <0.2→+4, 0.2~0.4→+2, 0.4~0.6→0, 0.6~0.8→-2, >0.8→-4
    """
    pct_b, pos_52w = _bb_pos_raw(df)
    idx = df["Close"].index

    bb_score = pd.Series(0.0, index=idx)
    bb_score[pct_b < 0.1]                         =  8.0
    bb_score[(pct_b >= 0.1) & (pct_b < 0.2)]     =  5.0
    bb_score[(pct_b >= 0.2) & (pct_b < 0.4)]     =  2.0
    bb_score[(pct_b >= 0.4) & (pct_b < 0.6)]     =  0.0
    bb_score[(pct_b >= 0.6) & (pct_b < 0.8)]     = -2.0
    bb_score[(pct_b >= 0.8) & (pct_b < 0.9)]     = -5.0
    bb_score[pct_b >= 0.9]                        = -8.0

    pos_score = pd.Series(0.0, index=idx)
    pos_score[pos_52w < 0.2]                          =  4.0
    pos_score[(pos_52w >= 0.2) & (pos_52w < 0.4)]    =  2.0
    pos_score[(pos_52w >= 0.4) & (pos_52w < 0.6)]    =  0.0
    pos_score[(pos_52w >= 0.6) & (pos_52w < 0.8)]    = -2.0
    pos_score[pos_52w >= 0.8]                         = -4.0

    return _clip((bb_score + pos_score) / 2.0)


def score_volatility_trend(df: pd.DataFrame) -> pd.Series:
    """
    변동성·위치 영역 점수 — **추세추종(trend-following)** 관점 (-10 ~ +10).

    해석: BB 상단 근처 = 강한 상승 추세 진행 중 (+)
          52주 신고점 근처 = 추세 지속 신호 (+)
          BB 하단 = 하락 추세 (-)
          52주 저점 근처 = 약세 구간 (-)

    연속 선형 공식 (단계 함수 대신 — 더 부드러운 점수 분포):
      BB %B  score = (%B   - 0.5) × 20  →  %B=1.0 → +10, %B=0.5 → 0, %B=0.0 → -10
      52w pos score = (pos  - 0.5) × 20  →  pos=1.0 → +10, pos=0.5 → 0, pos=0.0 → -10
      Area4 = (bb_score + pos_score) / 2          → 범위 -10 ~ +10

    설계 의도: 추세추종 전략에서 "고점 = 위험"이 아니라
               "고점 = 추세 강도 확인"으로 해석하여
               상승 추세 중 Area4가 다른 영역(추세·모멘텀)과
               같은 방향을 지지하도록 한다.
    """
    pct_b, pos_52w = _bb_pos_raw(df)

    bb_score  = _clip((pct_b  - 0.5) * 20.0)
    pos_score = _clip((pos_52w - 0.5) * 20.0)

    return _clip((bb_score + pos_score) / 2.0)


# ---------------------------------------------------------------------------
# Area 5 — 심리·Wyckoff
# ---------------------------------------------------------------------------

def _wyckoff_label_score(label_series: pd.Series) -> pd.Series:
    """Wyckoff 국면 레이블 → 점수."""
    mapping = {
        "Markup":        6.0,
        "Accumulation":  3.0,
        "Distribution": -3.0,
        "Markdown":     -6.0,
    }
    return label_series.map(mapping).fillna(0.0)


def _hope_vector_score(hv: pd.Series) -> pd.Series:
    """Hope Vector → 점수."""
    s = pd.Series(0.0, index=hv.index)
    s[hv > 0.05]                        = 6.0
    s[(hv > 0.01) & (hv <= 0.05)]      = 3.0
    s[(hv >= -0.01) & (hv <= 0.01)]    = 0.0
    s[(hv >= -0.05) & (hv < -0.01)]    = -3.0
    s[hv < -0.05]                       = -6.0
    return s


def _anxiety_score(ax: pd.Series) -> pd.Series:
    """
    Anxiety_Index 방향 반영.
    양수 고변동(>0.3) → Hope 방향 +4
    음수 고변동(<-0.3) → Fear 방향 -4
    중립 구간 → 크기 비례 선형 매핑 [-2, +2]
    """
    s = ax.clip(-1.0, 1.0) * 4.0   # [-4, +4] 범위로 선형 스케일
    return s


def score_wyckoff(df: pd.DataFrame) -> pd.Series:
    """
    심리·Wyckoff 영역 점수 (-10 ~ +10).

    hillstorm 모듈 미설치 시 0.0을 반환한다.
    """
    if not _wyckoff_available:
        return pd.Series(0.0, index=df.index)

    try:
        # hillstorm API 호출
        df_ind = compute_indicators(df.copy())
        df_wy  = classify_wyckoff(df_ind)
        df_sig = detect_signals(df_wy)

        # --- Wyckoff 국면 레이블 ---
        label_col = next(
            (c for c in df_sig.columns
             if "phase" in c.lower() or "label" in c.lower() or "wyckoff" in c.lower()),
            None,
        )
        if label_col:
            lbl_score = _wyckoff_label_score(df_sig[label_col])
        else:
            lbl_score = pd.Series(0.0, index=df.index)

        # --- Hope Vector ---
        hope_col = next(
            (c for c in df_sig.columns if "hope" in c.lower()),
            None,
        )
        if hope_col:
            hv_score = _hope_vector_score(df_sig[hope_col].fillna(0.0))
        else:
            hv_score = pd.Series(0.0, index=df.index)

        # --- Anxiety Index ---
        anx_col = next(
            (c for c in df_sig.columns if "anxiety" in c.lower()),
            None,
        )
        if anx_col:
            ax_score = _anxiety_score(df_sig[anx_col].fillna(0.0))
        else:
            ax_score = pd.Series(0.0, index=df.index)

        # 인덱스 정렬 (hillstorm이 인덱스를 바꿀 수도 있으므로)
        lbl_score = lbl_score.reindex(df.index, fill_value=0.0)
        hv_score  = hv_score.reindex(df.index, fill_value=0.0)
        ax_score  = ax_score.reindex(df.index, fill_value=0.0)

        raw = (lbl_score + hv_score + ax_score) / 3.0
        return _clip(raw)

    except Exception as exc:
        warnings.warn(f"[scorer] Wyckoff 점수 계산 실패: {exc} → 0으로 대체", stacklevel=2)
        return pd.Series(0.0, index=df.index)


# ---------------------------------------------------------------------------
# 종합 점수 산출 (메인 진입점)
# ---------------------------------------------------------------------------

def _effective_weights(weights: dict[str, float]) -> dict[str, float]:
    """
    유효 가중치를 계산한다.

    Wyckoff hillstorm 미설치 시 'wyckoff' 가중치를 나머지 4개 영역에
    비례 재분배하여 composite_score 의 실질 최대값이 줄어드는 문제를 방지한다.

    예) Basic 가중치 + Wyckoff 미사용:
        wyckoff(0.20) → 나머지 4개에 비례 배분
        trend:  0.20 → 0.25  | momentum: 0.25 → 0.3125
        volume: 0.25 → 0.3125 | volatility: 0.10 → 0.125
        composite 최대값: ~6.2 (수정 전 ~4.98, +5 임계 달성 가능)
    """
    w = dict(weights)

    # 1단계: 전체 합 1.0 정규화
    w_total = sum(w.values())
    if abs(w_total - 1.0) > 1e-9:
        w = {k: v / w_total for k, v in w.items()}

    # 2단계: Wyckoff 미사용 시 가중치 재분배
    if not _wyckoff_available:
        wy_w = w.pop("wyckoff", 0.0)
        if wy_w > 0:
            remaining_sum = sum(w.values())
            if remaining_sum > 0:
                scale = 1.0 / remaining_sum   # 나머지를 1.0으로 스케일업
                w = {k: v * scale for k, v in w.items()}
        w["wyckoff"] = 0.0   # 점수 계산에서 제외

    return w


AREA4_MODES = ("contrarian", "trend")   # 허용 Area4 모드 값


def compute_scores(
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
    area4_mode: str = "trend",
) -> pd.DataFrame:
    """
    OHLCV DataFrame을 받아 5개 영역 점수 + 종합 점수를 계산한다.

    Parameters
    ----------
    df          : Open/High/Low/Close/Volume 컬럼을 가진 DatetimeIndex DataFrame
    weights     : 가중치 dict. None이면 BASIC_WEIGHTS 사용.
                  키: trend / momentum / volume / volatility / wyckoff
    area4_mode  : Area 4 변동성·위치 점수 산출 방식
                  - "contrarian" (기본) : BB하단·52주저점 = 매수 기회 (평균회귀)
                  - "trend"            : BB상단·52주고점 = 강세 지속 (추세추종)

    Returns
    -------
    pd.DataFrame  — 원본 컬럼 + 아래 컬럼 추가:
        area1_trend, area2_momentum, area3_volume,
        area4_volatility, area5_wyckoff, composite_score,
        wyckoff_active (bool), area4_mode (str),
        atr14  (C1/C2 청산가 계산용)

    Notes
    -----
    Wyckoff hillstorm 미설치 시 area5_wyckoff = 0 이 되지만,
    _effective_weights() 가 나머지 4개 영역 가중치를 재분배하므로
    composite_score 의 달성 가능 범위는 유지된다.
    """
    if weights is None:
        weights = BASIC_WEIGHTS

    if area4_mode not in AREA4_MODES:
        raise ValueError(
            f"area4_mode='{area4_mode}' 는 유효하지 않습니다. "
            f"허용값: {AREA4_MODES}"
        )

    # 유효 가중치 계산 (Wyckoff 미사용 시 재분배 포함)
    w = _effective_weights(weights)

    result = df.copy()

    # 각 영역 점수 계산
    result["area1_trend"]      = score_trend(df).values
    result["area2_momentum"]   = score_momentum(df).values
    result["area3_volume"]     = score_volume(df).values

    # Area 4: 모드에 따라 함수 선택
    if area4_mode == "trend":
        result["area4_volatility"] = score_volatility_trend(df).values
    else:
        result["area4_volatility"] = score_volatility(df).values

    result["area5_wyckoff"]    = score_wyckoff(df).values
    result["wyckoff_active"]   = _wyckoff_available
    result["area4_mode"]       = area4_mode

    # 가중합 → 종합 점수
    composite = (
        result["area1_trend"]      * w.get("trend",      0.0) +
        result["area2_momentum"]   * w.get("momentum",   0.0) +
        result["area3_volume"]     * w.get("volume",     0.0) +
        result["area4_volatility"] * w.get("volatility", 0.0) +
        result["area5_wyckoff"]    * w.get("wyckoff",    0.0)
    )
    result["composite_score"] = composite.clip(-10.0, 10.0)

    # ATR (청산가 계산용으로 함께 제공)
    result["atr14"] = _atr(df["High"], df["Low"], df["Close"]).values

    return result
