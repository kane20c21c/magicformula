"""strategy_reference.py — 대형·고외국인 눌림목 진입 전략 참조 구현 (v1.0.0, 확정 2026-07-05).

목적: Kane의 맥미니 데이터수집/실시간 모니터링 프로젝트가 그대로 import 하여
      '오늘 어떤 종목이 유니버스에 있고 / 진입신호가 떴고 / 보유 종목이 손절선에 닿았는지'를
      계산하는 데 쓰는 순수 함수 모음. 백테스트와 동일한 신호 정의를 보장한다.

의존성: pandas, numpy 뿐. (외부 상태·네트워크·파일 없음)

핵심 규약 (STRATEGY.md / strategy_spec.json 과 일치):
  - 신호는 t일 종가로 확정 → 매수는 t+1일 종가 (룩어헤드 방지).
  - 손절은 당일(t) 종가 체결 (Kane 확정: 당일 손절).
  - 유니버스: 월말 시총 5조 & 외국인지분 30% (다음 달 적용, 일별 ffill).
  - 진입: 120일선 위 + 60일 고점 대비 10%↑ 눌림, '새 눌림(onset)'만.
  - 후보 우선순위: 눌림 깊은 순.
  - 재진입 쿨다운: 1거래일 (판 그날만 금지).
  - 포트: N=20, 종목당 max 2슬롯(불타기), 슬롯 500만 / 총 1억, 정수주, 비용 0.3%/편도, T+1.

데이터 입력 스키마 (Kane repo data/*.parquet 과 동일):
  - close_wide      : DataFrame, index=거래일(DatetimeIndex), columns=ticker, 값=종가(원). 최소 120거래일 워밍업.
  - mktcap_monthly  : DataFrame, index=월말일, columns=ticker, 값=시가총액(원).
  - foreign_monthly : DataFrame, index=월말일, columns=ticker, 값=외국인지분율(%). 0~100 스케일.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────────────────────
# 확정 파라미터 (strategy_spec.json 과 동일. JSON을 로드해 덮어써도 됨)
# ────────────────────────────────────────────────────────────────────────────
STRATEGY_VERSION = "1.0.0"

MKTCAP_MIN_KRW = 5_000_000_000_000     # 유니버스 시총 하한 (5조원)
FOREIGN_MIN_PCT = 30.0                 # 유니버스 외국인지분율 하한 (%)

HIGH_WINDOW = 60                       # 눌림 기준 고점 창
HIGH_MIN_PERIODS = 30
TREND_MA = 120                         # 추세 필터 이동평균
TREND_MIN_PERIODS = 80
PULLBACK_RATIO = 0.90                  # close <= 0.90*high60  (10% 눌림)

TRAIL_STOP_PCT = 0.20                  # 고점 대비 20% 추적손절
CAPITAL_KRW = 100_000_000              # 1억
MAX_POSITIONS = 20                     # N 슬롯
SLOT_KRW = 5_000_000                   # 슬롯당 500만
MAX_SLOTS_PER_STOCK = 2                # 불타기 max2
REENTRY_COOLDOWN_DAYS = 1              # 판 그날만 금지
FEE_PER_SIDE = 0.003                   # 편도 0.3%


# ────────────────────────────────────────────────────────────────────────────
# 1) 유니버스 판정
# ────────────────────────────────────────────────────────────────────────────
def compute_monthly_eligibility(
    mktcap_monthly: pd.DataFrame,
    foreign_monthly: pd.DataFrame,
    cap_min_krw: float = MKTCAP_MIN_KRW,
    foreign_min_pct: float = FOREIGN_MIN_PCT,
) -> pd.DataFrame:
    """월말 기준 유니버스 편입 여부(bool). index=월말일, columns=ticker.

    두 조건 AND: 시총 >= cap_min_krw 그리고 외국인지분율 >= foreign_min_pct.
    """
    cols = mktcap_monthly.columns.union(foreign_monthly.columns)
    cap = mktcap_monthly.reindex(columns=cols)
    fr = foreign_monthly.reindex(index=cap.index, columns=cols)
    elig = (cap >= cap_min_krw) & (fr >= foreign_min_pct)
    return elig.fillna(False)


def to_daily_eligibility(monthly_elig: pd.DataFrame, daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    """월말 편입 flag를 거래일 인덱스로 forward-fill.

    월말(t)에 판정된 유니버스가 '다음 달'에 적용되도록, 월말 이후 거래일에 값이 채워진다
    (reindex+ffill). 백테스트와 동일하게 월말 라벨을 그대로 ffill 한다.
    """
    return monthly_elig.reindex(index=daily_index, method="ffill").fillna(False)


# ────────────────────────────────────────────────────────────────────────────
# 2) 지표 & 진입/청산 신호
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class Indicators:
    high60: pd.DataFrame
    ma120: pd.DataFrame
    depth: pd.DataFrame        # (high60 - close)/high60 : 눌림 깊이(0~1)
    entry: pd.DataFrame        # 진입 조건 충족(bool)  — 유니버스 반영 전/후는 인자로 결정
    onset: pd.DataFrame        # '새 눌림' 첫날(bool)


def compute_indicators(close_wide: pd.DataFrame, elig_daily: Optional[pd.DataFrame] = None) -> Indicators:
    """가격 패널로부터 60일고점·120일선·눌림깊이·진입·onset 계산.

    elig_daily 를 주면 진입조건에 유니버스 편입 여부까지 AND 한다(권장).
    """
    high60 = close_wide.rolling(HIGH_WINDOW, min_periods=HIGH_MIN_PERIODS).max()
    ma120 = close_wide.rolling(TREND_MA, min_periods=TREND_MIN_PERIODS).mean()
    depth = (high60 - close_wide) / high60

    cond = (close_wide <= PULLBACK_RATIO * high60) & (close_wide > ma120)
    if elig_daily is not None:
        cond = cond & elig_daily.reindex(index=close_wide.index, columns=close_wide.columns).fillna(False)
    entry = cond.fillna(False)
    onset = entry & ~entry.shift(1).fillna(False)   # 전일 False → 오늘 True
    return Indicators(high60, ma120, depth, entry, onset)


def check_trailing_stop(peak_price: float, current_close: float, trail_pct: float = TRAIL_STOP_PCT) -> bool:
    """추적손절 발동 여부. 보유중 최고가(peak) 대비 current_close 가 trail_pct 이상 하락하면 True."""
    if peak_price is None or np.isnan(peak_price) or np.isnan(current_close):
        return False
    return current_close <= peak_price * (1.0 - trail_pct)


def shares_for_slot(price: float, slot_krw: float = SLOT_KRW, fee: float = FEE_PER_SIDE) -> int:
    """슬롯 예산으로 살 수 있는 정수 주식수(수수료 무시한 단순 floor). 실제 현금검증은 호출측에서."""
    if price is None or np.isnan(price) or price <= 0:
        return 0
    return int(slot_krw // price)


# ────────────────────────────────────────────────────────────────────────────
# 3) 일일 스크린 — 모니터링 코드가 매일 종가 확정 후 호출
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class DailyScreen:
    as_of: pd.Timestamp
    buy_candidates: pd.DataFrame       # 내일(t+1) 매수 후보: [ticker, depth, close] 눌림깊은 순
    pyramid_candidates: list           # 보유중 & onset 재발생 → 불타기 후보 ticker
    stop_hits: list                    # 보유중 & 오늘 종가로 손절선 도달 → 오늘 청산 대상 ticker
    note: str = ""


def screen_today(
    as_of: pd.Timestamp,
    close_wide: pd.DataFrame,
    elig_daily: pd.DataFrame,
    positions: Optional[dict] = None,          # {ticker: {"peak": float, "slots": int}}
    last_sell_day: Optional[dict] = None,       # {ticker: pd.Timestamp}  최근 매도일
    cooldown_days: int = REENTRY_COOLDOWN_DAYS,
) -> DailyScreen:
    """as_of(오늘) 종가 확정 후의 액션 스크린.

    반환:
      - buy_candidates : onset(오늘 새 눌림) & 유니버스 & 쿨다운 통과 & 미보유 → t+1 매수 후보(눌림깊은 순).
      - pyramid_candidates : 보유중 & 오늘 onset 재발생 & 슬롯<2 → t+1 추가매수 후보.
      - stop_hits : 보유중 & 오늘 종가가 보유중고점*(1-0.20) 이하 → 오늘 청산 대상.
    주의: 매수는 t+1 종가 체결, 손절은 당일 종가 체결(규약).
    """
    positions = positions or {}
    last_sell_day = last_sell_day or {}
    ind = compute_indicators(close_wide, elig_daily)

    if as_of not in close_wide.index:
        raise KeyError(f"as_of {as_of} 가 close_wide 인덱스에 없음")
    d = as_of
    px_today = close_wide.loc[d]
    onset_today = ind.onset.loc[d]
    depth_today = ind.depth.loc[d]

    held = set(positions.keys())

    # 손절 대상 (보유중, 당일 종가 기준)
    stop_hits = []
    for tk, pos in positions.items():
        c = px_today.get(tk, np.nan)
        peak = max(pos.get("peak", c), c) if not np.isnan(c) else pos.get("peak", np.nan)
        if check_trailing_stop(peak, c):
            stop_hits.append(tk)

    # 불타기 후보 (보유중 & onset & 슬롯 여유)
    pyramid = [
        tk for tk, pos in positions.items()
        if bool(onset_today.get(tk, False)) and pos.get("slots", 1) < MAX_SLOTS_PER_STOCK
    ]

    # 신규 매수 후보 (onset & 미보유 & 쿨다운 통과)
    def cooldown_ok(tk: str) -> bool:
        ls = last_sell_day.get(tk)
        if ls is None:
            return True
        gap = np.busday_count(np.datetime64(ls, "D"), np.datetime64(d, "D"))
        return gap >= cooldown_days

    cand = [
        tk for tk in onset_today.index
        if bool(onset_today.get(tk, False)) and tk not in held and cooldown_ok(tk)
        and not np.isnan(px_today.get(tk, np.nan))
    ]
    buy = pd.DataFrame({
        "ticker": cand,
        "depth": [float(depth_today.get(tk, np.nan)) for tk in cand],
        "close": [float(px_today.get(tk, np.nan)) for tk in cand],
    }).sort_values("depth", ascending=False).reset_index(drop=True)

    return DailyScreen(
        as_of=d, buy_candidates=buy, pyramid_candidates=pyramid, stop_hits=stop_hits,
        note="buy_candidates 는 t+1 종가 매수 대상(눌림깊은 순, 빈 슬롯까지). stop_hits 는 오늘 종가 청산 대상.",
    )


# ────────────────────────────────────────────────────────────────────────────
# 4) 자체 점검용 데모 (Kane repo 에서 실행)
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]   # korea_momentum_research/
    data = root / "data"
    try:
        close = pd.read_parquet(data / "prices.parquet")["close"].astype("float64").unstack("ticker").sort_index()
        meta = pd.read_parquet(data / "meta.parquet")
        cap_m = meta["mktcap"].unstack("ticker")
        fo = pd.read_parquet(data / "foreign.parquet")
        fr_m = fo["foreign_ratio"].astype("float64").unstack("ticker")
    except Exception as e:
        print(f"[데모] 데이터 로드 실패: {e}\n  (모니터링 프로젝트에서는 자체 데이터로 함수만 import 하세요.)")
        sys.exit(0)

    close = close[close.index >= pd.Timestamp("2018-06-01")]
    elig_m = compute_monthly_eligibility(cap_m, fr_m)
    elig_d = to_daily_eligibility(elig_m, close.index)

    as_of = close.index[-1]
    scr = screen_today(as_of, close, elig_d, positions={}, last_sell_day={})
    print(f"strategy_reference v{STRATEGY_VERSION} 데모 — as_of={as_of.date()}")
    print(f"유니버스 편입 종목수(오늘): {int(elig_d.loc[as_of].sum())}")
    print(f"신규 진입신호(onset & 유니버스, t+1 매수후보) {len(scr.buy_candidates)}종목 — 상위 10(눌림깊은 순):")
    print(scr.buy_candidates.head(10).to_string(index=False))
