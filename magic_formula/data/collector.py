"""
data/collector.py
-----------------
OHLCV / KOSPI 데이터 수집 — **longlivevault 위임 단일화** (P4).

이전 (845줄)
------------
- 종목 OHLCV: vault → pykrx 폴백 + src/cache/ 로컬 캐시
- KOSPI:      vault raw → pykrx → KIS REST → yfinance 4단 폴백

현재 (이 파일)
------------
- 종목 OHLCV: vault `data_service.get_ohlcv` 위임만
- KOSPI:      vault `ohlcv_store.get_ohlcv("KOSPI", ...)` 위임만
- 캐시:        vault 가 이미 캐시 역할 — src/cache/ 폐기

vault 미설치 환경에서는 빈 DataFrame 을 반환. 폴백 데이터 소스를 가지지 않으므로
KOSPI 알파 계산이 생략될 수 있고, 그 경우 metrics 가 'N/A' 로 출력한다.

호환성
-------
이전 import 경로/시그니처는 모두 유지:
- `fetch_ohlcv(ticker, start, end, use_cache=True)`
- `fetch_kospi(start, end, use_cache=True)`
- `collect_all(months=18, use_cache=True, ticker_list=None)`
- `get_backtest_split(df, warmup_months=12)`
- `TICKERS`, `_date_range`

`use_cache` 파라미터는 호환을 위해 남겨두지만, vault 가 자체 캐시(parquet) 를
관리하므로 이 모듈에서는 별도 처리 없이 무시한다.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# longlivevault 연결 — _vault 가 sys.path 등록 + 가용성 판정 단일화 (P3)
# P5a: 같은 패키지 안의 _vault 를 import (절대 경로 magic_formula._vault)
# ---------------------------------------------------------------------------
# Magic Formula 루트가 sys.path 에 없으면 추가 (대개 main / daily_signal 진입점이 이미 추가)
_PROJ_ROOT = str(Path(__file__).parent.parent.parent)
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from magic_formula._vault import VAULT_PATH, VAULT_AVAILABLE as _VAULT_AVAILABLE   # noqa: E402
from magic_formula._vault import TICKER_NAMES_FALLBACK as _NAMES_ALL               # noqa: E402
from magic_formula._vault import DEFAULT_EXCLUDE as _EXCLUDE                       # noqa: E402

if _VAULT_AVAILABLE:
    try:
        from stolab_data.data_service import get_ohlcv as _vault_get_ohlcv  # type: ignore
        from stolab_data.ohlcv_store  import get_ohlcv as _vault_store_get_ohlcv  # type: ignore
    except ImportError:
        _VAULT_AVAILABLE = False
        _vault_get_ohlcv       = None   # type: ignore
        _vault_store_get_ohlcv = None   # type: ignore
else:
    _vault_get_ohlcv       = None   # type: ignore
    _vault_store_get_ohlcv = None   # type: ignore

# ---------------------------------------------------------------------------
# 57종목 종목명 딕셔너리 (호환용 — _vault 에서 EXCLUDE 빼서 빌드, P3)
# ---------------------------------------------------------------------------

TICKERS: dict[str, str] = {
    t: n for t, n in _NAMES_ALL.items() if t not in _EXCLUDE
}

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

KOSPI_TICKER = "KOSPI"   # vault tickers/KOSPI.parquet (종합지수 1001) 식별자


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------

def _date_range(months: int = 18) -> tuple[str, str]:
    """오늘 기준 `months`개월 전 ~ 오늘 날짜를 'YYYYMMDD' 형식으로 반환."""
    end = datetime.today()
    start = end - timedelta(days=int(months * 30.4375))
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def _vault_df_to_indexed(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    vault 반환값(컬럼: Date, Ticker, Open, High, Low, Close, Volume, ...) 을
    Magic Formula 표준(DatetimeIndex + OHLCV) 으로 변환.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Date 컬럼 → DatetimeIndex
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        df.index.name = "Date"
    elif not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame()

    # OHLCV 5개 컬럼만 유지 (지수는 Volume 없을 수 있어 4개만 남는 케이스 OK)
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    if not keep:
        return pd.DataFrame()
    df = df[keep].copy()

    # 숫자 변환
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Volume 컬럼이 있으면 0인 행(비거래일) 제거 — 지수는 Volume 없을 수 있으므로 조건부
    if "Volume" in df.columns:
        df = df[df["Volume"] > 0]

    df = df.sort_index()

    # 날짜 범위 필터
    start_dt = pd.to_datetime(start, format="%Y%m%d")
    end_dt   = pd.to_datetime(end,   format="%Y%m%d")
    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
    return df


# ---------------------------------------------------------------------------
# 공개 API — 종목 OHLCV
# ---------------------------------------------------------------------------

def fetch_ohlcv(
    ticker: str,
    start: str,
    end: str,
    use_cache: bool = True,   # noqa: ARG001  (호환 — vault 자체 캐시 사용)
) -> pd.DataFrame:
    """
    단일 종목 OHLCV 데이터를 반환한다 (longlivevault 위임).

    Parameters
    ----------
    ticker    : 종목 코드 (예: '000660')
    start     : 'YYYYMMDD'
    end       : 'YYYYMMDD'
    use_cache : 호환용 (무시)

    Returns
    -------
    DatetimeIndex 인 OHLCV DataFrame. vault 미설치/실패 시 빈 DataFrame.
    """
    name = TICKERS.get(ticker, ticker)

    if not _VAULT_AVAILABLE or _vault_get_ohlcv is None:
        print(f"  [skip] {ticker} ({name}): vault 미설치")
        return pd.DataFrame()

    try:
        raw = _vault_get_ohlcv(ticker, start_date=start, end_date=end)
    except Exception as exc:
        print(f"  [vault] {ticker} 실패: {exc}")
        return pd.DataFrame()

    df = _vault_df_to_indexed(raw, start, end)
    if df.empty:
        print(f"  [vault] {ticker} ({name}): empty")
    else:
        print(f"  [vault] {ticker} ({name}): {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# 공개 API — KOSPI 종합지수
# ---------------------------------------------------------------------------

def fetch_kospi(
    start: str,
    end: str,
    use_cache: bool = True,   # noqa: ARG001  (호환 — vault 자체 캐시 사용)
) -> pd.DataFrame:
    """
    KOSPI 종합지수(1001) OHLCV 를 반환한다 (longlivevault 위임).

    vault `tickers/KOSPI.parquet` 에서 직접 조회.
    Volume 컬럼이 없어도 OK (지수).
    """
    if not _VAULT_AVAILABLE or _vault_store_get_ohlcv is None:
        print(f"  [skip] KOSPI: vault 미설치 — 알파 계산 생략")
        return pd.DataFrame()

    try:
        raw = _vault_store_get_ohlcv(KOSPI_TICKER, start_date=start, end_date=end)
    except Exception as exc:
        print(f"  [vault] KOSPI 실패: {exc}")
        return pd.DataFrame()

    df = _vault_df_to_indexed(raw, start, end)
    if df.empty:
        print(f"  [vault] KOSPI: empty — 알파 계산 생략")
    else:
        c_min, c_max = df["Close"].min(), df["Close"].max()
        print(f"  [vault] KOSPI: {len(df)} rows  Close {c_min:,.0f}~{c_max:,.0f}pt")
    return df


# ---------------------------------------------------------------------------
# 공개 API — 일괄 수집
# ---------------------------------------------------------------------------

def collect_all(
    months: int = 18,
    use_cache: bool = True,   # noqa: ARG001  (호환)
    ticker_list: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    지정 종목 + KOSPI 지수를 일괄 수집한다 (vault 위임).

    Parameters
    ----------
    months      : 조회 기간 (기본 18개월)
    use_cache   : 호환용 (무시)
    ticker_list : 수집할 종목 코드 리스트. None 이면 TICKERS(57종목) 사용.

    Returns
    -------
    dict {ticker_or_'KOSPI': DataFrame}
    실패한 종목은 결과에서 제외된다.
    """
    if ticker_list is None:
        ticker_list = list(TICKERS.keys())

    n_total = len(ticker_list)
    start, end = _date_range(months)

    print(f"\n{'='*60}")
    print(f"데이터 수집 (vault 위임): {start} ~ {end}  ({months}개월)")
    print(f"{'='*60}")

    data: dict[str, pd.DataFrame] = {}

    # KOSPI 지수
    kospi = fetch_kospi(start, end)
    if not kospi.empty:
        data["KOSPI"] = kospi

    # 개별 종목
    for ticker in ticker_list:
        df = fetch_ohlcv(ticker, start, end)
        if not df.empty:
            data[ticker] = df

    n_stocks = len(data) - (1 if "KOSPI" in data else 0)
    print(f"\n수집 완료: {n_stocks}/{n_total} 종목, KOSPI {'포함' if 'KOSPI' in data else '제외'}")
    return data


# ---------------------------------------------------------------------------
# 공개 API — 백테스트 구간 분할
# ---------------------------------------------------------------------------

def get_backtest_split(
    df: pd.DataFrame,
    warmup_months: int = 12,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    전체 인덱스를 워밍업 / 실거래 구간으로 분리한다.

    Returns
    -------
    (warmup_idx, trade_idx)
    warmup_idx : 지표 계산용 (처음 warmup_months 개월)
    trade_idx  : 실제 매매 시뮬레이션 구간 (나머지)
    """
    cutoff = df.index[0] + pd.DateOffset(months=warmup_months)
    warmup_idx = df.index[df.index < cutoff]
    trade_idx  = df.index[df.index >= cutoff]
    return warmup_idx, trade_idx
