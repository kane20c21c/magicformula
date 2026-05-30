"""
analysis/ic_framework.py
------------------------
영역별 점수 변형의 forward direction 예측력을 측정하는 generic 프레임워크.

핵심 가설: "좋은 점수 = 향후 N거래일 가격 변화 방향을 잘 예측한다"

평가 지표
---------
- hit_rate          : sign(score_t) == sign(fwd_alpha_{t+N}) 비율 (전체)
- hc_hit_rate       : |score_t| ≥ τ 의 high-conviction subset 에서의 hit rate
- ic_pearson        : Pearson correlation (score, fwd_alpha)
- ic_spearman       : Spearman rank correlation
- bucket_alpha      : 점수 십분위별 fwd_alpha 평균 (monotone 확인용)
- mean / std / max  : 점수 분포 요약

설계 원칙
---------
- look-ahead bias 차단: score는 t 시점까지 데이터만 사용, fwd_alpha는 t+1 ~ t+N
- ticker 풀링: 모든 (ticker, date) 관측치를 한 풀로 합쳐 통계 산출
- KOSPI alpha: 시장 베타 제거 — fwd_alpha = stock_ret - kospi_ret
"""

from __future__ import annotations

from typing import Callable, Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Forward alpha 계산
# ---------------------------------------------------------------------------

def compute_fwd_alpha(
    df:           pd.DataFrame,
    kospi_df:     Optional[pd.DataFrame],
    horizon_days: int = 20,
) -> pd.Series:
    """
    종목 df 의 각 t 시점에서 t+horizon_days 까지의 KOSPI 대비 알파 수익률.

    fwd_alpha_t = (Close_t+N / Close_t - 1) - (KOSPI_t+N / KOSPI_t - 1)

    kospi_df 가 None 이면 단순 수익률 반환 (시장 alpha 없음).
    """
    if "Close" not in df.columns:
        raise ValueError("df 에 Close 컬럼 필요")

    close = df["Close"]
    stock_ret = close.shift(-horizon_days) / close - 1.0

    if kospi_df is None or kospi_df.empty:
        return stock_ret

    # KOSPI 를 종목 인덱스에 정렬
    k = kospi_df["Close"].reindex(close.index, method="ffill")
    kospi_ret = k.shift(-horizon_days) / k - 1.0

    return stock_ret - kospi_ret


def compute_fwd_return(
    df:           pd.DataFrame,
    horizon_days: int = 20,
    mode:         str = "raw",
    kospi_df:     Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    forward return 일반화 진입점.

    Parameters
    ----------
    mode : 'raw'   → 단순 종목 수익률  (Close_{t+N} / Close_t - 1)
           'alpha' → KOSPI 대비 알파    (stock_ret - kospi_ret)

    Kane 2026-05-29: 강세장에서는 alpha 가 음으로 편향되어 hit_rate 가
    구조적으로 < 0.5 가 됨. 점수의 방향 예측력 자체를 보려면 'raw' 권장.
    """
    if mode == "raw":
        if "Close" not in df.columns:
            raise ValueError("df 에 Close 컬럼 필요")
        return df["Close"].shift(-horizon_days) / df["Close"] - 1.0
    elif mode == "alpha":
        return compute_fwd_alpha(df, kospi_df, horizon_days)
    else:
        raise ValueError(f"mode={mode!r} 는 유효하지 않음. 'raw' 또는 'alpha'")


# ---------------------------------------------------------------------------
# Variant evaluator — pooled (ticker × date)
# ---------------------------------------------------------------------------

def evaluate_variant(
    scores_by_ticker:  dict[str, pd.Series],
    fwd_by_ticker:     dict[str, pd.Series],
    hc_threshold:      float = 5.0,
    bench_pos_ratio:   float | None = None,
) -> dict:
    """
    여러 종목의 score / fwd 를 풀링해서 단일 metric 산출.

    Parameters
    ----------
    scores_by_ticker : {ticker: pd.Series of scores}
    fwd_by_ticker    : {ticker: pd.Series of fwd_return}
    hc_threshold     : |score| ≥ τ 의 high-conviction subset 기준 (기본 5.0)
    bench_pos_ratio  : 평가 모집단의 양수 비율 (= no-signal bench).
                       제공되면 realistic_hit 계산: score=0 인 (신호 없는)
                       날은 'long 디폴트 → bench 확률로 적중' 가정.
                       → off 모드의 평가 모집단 축소 effect 보정.

    Returns
    -------
    dict with keys:
        n_obs           : 전체 풀 관측치 수
        n_directional_f : sign(fwd) != 0 인 관측치 수 (평가 모집단)
        n_bet           : 베팅한 관측치 수 (sign(score) != 0 AND sign(fwd) != 0)
        coverage        : n_bet / n_directional_f
        hit_rate        : 베팅한 관측치 중 적중 비율 (conditional)
        realistic_hit   : coverage × hit_rate + (1-coverage) × bench_pos_ratio
                          → 신호 없을 때 bench 베팅 가정 시 전체 hit
        hc_hit_rate, n_hc, ic_pearson, ic_spearman,
        mean_score, std_score, max_score, min_score, bucket_alpha
    """
    pooled_scores = []
    pooled_fwd    = []

    for ticker in scores_by_ticker:
        s = scores_by_ticker[ticker]
        f = fwd_by_ticker.get(ticker)
        if f is None:
            continue
        # 공통 인덱스 + NaN 제거
        common = s.index.intersection(f.index)
        s_ = s.loc[common]
        f_ = f.loc[common]
        mask = s_.notna() & f_.notna()
        pooled_scores.append(s_.loc[mask])
        pooled_fwd.append(f_.loc[mask])

    if not pooled_scores:
        return {"n_obs": 0}

    s_all = pd.concat(pooled_scores)
    f_all = pd.concat(pooled_fwd)

    if len(s_all) == 0:
        return {"n_obs": 0}

    # ─ Hit rate — 분모/분자 명시
    sign_s = np.sign(s_all)
    sign_f = np.sign(f_all)
    f_directional = (sign_f != 0)              # 평가 가능한 obs (fwd 가 방향성 있음)
    bettable      = (sign_s != 0) & f_directional   # 점수 신호 있음 AND fwd 방향 있음

    n_directional_f = int(f_directional.sum())
    n_bet           = int(bettable.sum())
    hit_rate = ((sign_s == sign_f) & bettable).sum() / max(n_bet, 1)
    coverage = n_bet / max(n_directional_f, 1)

    # realistic_hit: score=0 인 날은 'long 디폴트' 가정 → bench 확률로 적중
    if bench_pos_ratio is not None:
        realistic_hit = coverage * hit_rate + (1.0 - coverage) * bench_pos_ratio
    else:
        realistic_hit = float("nan")

    # ─ High-conviction subset
    hc_mask = s_all.abs() >= hc_threshold
    hc_sign_s = sign_s[hc_mask]
    hc_sign_f = sign_f[hc_mask]
    hc_directional = (hc_sign_s != 0) & (hc_sign_f != 0)
    hc_hit_rate = (
        ((hc_sign_s == hc_sign_f) & hc_directional).sum()
        / max(hc_directional.sum(), 1)
    )

    # ─ IC
    if len(s_all) > 1 and s_all.std() > 0 and f_all.std() > 0:
        ic_pearson  = float(s_all.corr(f_all, method="pearson"))
        ic_spearman = float(s_all.corr(f_all, method="spearman"))
    else:
        ic_pearson = ic_spearman = float("nan")

    # ─ 점수 십분위별 fwd_alpha 평균 (monotone 확인용)
    try:
        buckets = pd.qcut(s_all, 10, duplicates="drop", labels=False)
        bucket_alpha = f_all.groupby(buckets).mean().to_dict()
    except (ValueError, IndexError):
        bucket_alpha = {}

    return {
        "n_obs":           int(len(s_all)),
        "n_directional_f": n_directional_f,
        "n_bet":           n_bet,
        "n_hc":            int(hc_mask.sum()),
        "coverage":        float(coverage),
        "hit_rate":        float(hit_rate),
        "realistic_hit":   float(realistic_hit),
        "hc_hit_rate":     float(hc_hit_rate),
        "ic_pearson":      ic_pearson,
        "ic_spearman":     ic_spearman,
        "mean_score":      float(s_all.mean()),
        "std_score":       float(s_all.std()),
        "max_score":       float(s_all.max()),
        "min_score":       float(s_all.min()),
        "bucket_alpha":    bucket_alpha,
    }


# ---------------------------------------------------------------------------
# Worker — picklable (Pool 호출용)
# ---------------------------------------------------------------------------

def compute_scores_one(
    ticker:       str,
    df:           pd.DataFrame,
    score_fn:     Callable[[pd.DataFrame], pd.Series],
) -> tuple[str, pd.Series]:
    """단일 종목의 점수 Series 를 계산해서 (ticker, scores) 반환.

    Pool 워커에서 호출되도록 module-level.
    """
    try:
        return ticker, score_fn(df)
    except Exception:
        return ticker, pd.Series(0.0, index=df.index)


# ---------------------------------------------------------------------------
# Market regime — breadth (시점 t 의 횡단면 양수 비율)
# ---------------------------------------------------------------------------

import numpy as np   # noqa: E402 (블록 분리 후 추가)


def compute_breadth_series(
    stock_data:  dict[str, pd.DataFrame],
    lookback:    int = 20,
    horizon:     int = 5,
) -> pd.Series:
    """
    시점 t 별 시장 breadth (횡단면 양수 비율) 시계열 반환.

    정의
    ----
    breadth(t) = (시점 [t-lookback+1, t] 의 모든 (종목, 날짜)) 중
                 horizon 일 backward 수익률이 > 0 인 비율.

    예) lookback=20, horizon=5
        시점 t 에서 지난 20거래일 × 67종목 = 약 1340개 5일 수익률 데이터.
        그 중 양수 비율.

    backward pct_change 이므로 시점 t 에 즉시 사용 가능 (look-ahead 없음).

    Parameters
    ----------
    stock_data : {ticker: pd.DataFrame with Close column, DatetimeIndex}
    lookback   : 윈도우 크기 (거래일)
    horizon    : 각 데이터 점의 측정 간격 (거래일)

    Returns
    -------
    pd.Series[date → breadth ∈ [0, 1]]
        index 는 모든 종목의 합집합 거래일. lookback 이전은 NaN.
    """
    # 각 종목의 horizon 일 수익률 (backward)
    rets = pd.DataFrame({
        t: df["Close"].pct_change(horizon)
        for t, df in stock_data.items()
    })

    # 각 (date, ticker) → 1 (양수) / 0 (음수) / NaN (결측)
    is_pos = rets.gt(0).astype(float)
    is_pos[rets.isna()] = float("nan")

    # 각 시점에서 지난 lookback 거래일 × 모든 종목의 양수 비율
    breadth = pd.Series(index=rets.index, dtype=float)
    arr = is_pos.values  # shape (n_dates, n_tickers)
    n_dates = arr.shape[0]
    for i in range(lookback - 1, n_dates):
        window = arr[i - lookback + 1 : i + 1]   # (lookback, n_tickers)
        flat = window.flatten()
        valid = ~np.isnan(flat)
        if valid.sum() > 0:
            breadth.iloc[i] = flat[valid].mean()

    return breadth


def classify_regime(
    breadth: float,
    low_threshold:  float = 0.40,
    high_threshold: float = 0.60,
) -> str:
    """breadth 값을 3 레짐 (강세/조정/하락) 으로 분류."""
    if pd.isna(breadth):
        return "unknown"
    if breadth >= high_threshold:
        return "강세"
    if breadth <= low_threshold:
        return "하락"
    return "조정"
