"""
data/collector.py
-----------------
OHLCV 데이터 수집 모듈.  59종목 + KOSPI 지수의 18개월 데이터를 수집한다.
캐시 폴더(src/cache/)에 Parquet으로 저장하여 재실행 시 재사용한다.

캐시 유효 기간: 1일 (당일 내 재실행 시 API 호출 생략)

종목 OHLCV 수집 우선순위:
  1. longlivevault (stolab_data.data_service.get_ohlcv) — 로컬 core.parquet + KIS 연동
  2. pykrx  — 기존 방식 (폴백)

KOSPI 데이터 수집 우선순위:
  1. longlivevault data/raw/krx_kospi_*.parquet — 가장 최신 raw 파일 취합
  2. pykrx  get_index_ohlcv_by_date("1001", ...)
  3. KIS REST API 직접 호출  — yaml 설정 파일에서 인증 정보 읽음
  4. yfinance  ^KS11
  5. 모두 실패 시 경고 후 빈 DataFrame 반환 (알파 계산 생략)
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from pykrx import stock as krx_stock
except ImportError:
    raise ImportError("pykrx 패키지가 필요합니다. 설치: pip install pykrx")

# ---------------------------------------------------------------------------
# longlivevault 연결 (1순위 데이터 소스)
# ---------------------------------------------------------------------------

# collector.py 위치: .../StoLab/Magic Formula/src/data/collector.py
# longlivevault 위치: .../StoLab/longlivevault/
# → __file__ 기반 상대경로로 자동 탐색, 없으면 절대경로 fallback
_VAULT_CANDIDATES = [
    str(Path(__file__).parent.parent.parent.parent / "longlivevault"),  # 상대 경로 (환경 독립)
    "/Users/kaneyoun/DriveForALL/StoLab/longlivevault",                 # 절대 경로 fallback
]
VAULT_PATH: str = next(
    (p for p in _VAULT_CANDIDATES if Path(p).exists()),
    _VAULT_CANDIDATES[0],  # 못 찾으면 첫 번째 후보 (import 실패로 처리됨)
)

if VAULT_PATH not in sys.path:
    sys.path.insert(0, VAULT_PATH)

try:
    from stolab_data.data_service import get_ohlcv as vault_get_ohlcv  # type: ignore
    from stolab_data.ohlcv_store import CORE_TICKERS as _VAULT_CORE_TICKERS  # type: ignore
    _VAULT_AVAILABLE = True
except ImportError:
    _VAULT_AVAILABLE = False
    _VAULT_CORE_TICKERS: frozenset = frozenset()

# ---------------------------------------------------------------------------
# KIS 설정 파일 탐색 경로 (우선순위 순)
# ---------------------------------------------------------------------------

_KIS_YAML_CANDIDATES = [
    os.path.expanduser("~/KIS/config/kis_devlp.yaml"),
    os.path.expanduser("~/open-trading-api/kis_devlp.yaml"),
    "/Users/kaneyoun/open-trading-api/kis_devlp.yaml",
]

# KIS open-trading-api 라이브러리 경로 (기존 폴백 방식 유지)
_KIS_EXAMPLES_PATH = "/Users/kaneyoun/open-trading-api/examples_llm"
_KIS_INDEX_PATH    = (
    "/Users/kaneyoun/open-trading-api/examples_llm"
    "/domestic_stock/inquire_index_daily_price"
)

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 59종목 종목명 딕셔너리 (longlivevault core.parquet Name 컬럼 기반)
# ---------------------------------------------------------------------------

TICKERS: dict[str, str] = {
    # 반도체
    "000660": "SK하이닉스",
    "005930": "삼성전자",
    "042700": "한미반도체",
    "058470": "리노공업",
    "240810": "원익IPS",
    "039030": "이오테크닉스",
    "000990": "DB하이텍",
    "403870": "HPSP",
    "357780": "솔브레인",
    "005290": "동진쎄미켐",
    # 반도체 핵심장비
    "000150": "두산",
    "007660": "이수페타시스",
    "095340": "ISC",
    # 로봇
    "058610": "에스피지",
    "277810": "레인보우로보틱스",
    "108490": "로보티즈",
    "454910": "두산로보틱스",
    "348340": "뉴로메카",
    "056080": "유진로봇",
    # 에너지 전송
    "010120": "LS ELECTRIC",
    "298040": "효성중공업",
    "267260": "HD현대일렉트릭",
    "001440": "대한전선",
    # 에너지 생산
    "000720": "현대건설",
    "047040": "대우건설",
    "034020": "두산에너빌리티",
    "052690": "한전기술",
    "032820": "우리기술",
    # 에너지 보관
    "373220": "LG에너지솔루션",
    "006400": "삼성SDI",
    "005490": "POSCO홀딩스",
    "247540": "에코프로비엠",
    "051910": "LG화학",
    "086520": "에코프로",
    # 바이오
    "068270": "셀트리온",
    "196170": "알테오젠",
    # "207940": 삼성바이오로직스 — 사업 분할 종목, 분석 제외
    # "0126Z0": 삼성에피스홀딩스 — 사업 분할 종목, 분석 제외
    # 조선
    "329180": "HD현대중공업",
    "042660": "한화오션",
    "010140": "삼성중공업",
    # 방산
    "012450": "한화에어로스페이스",
    "047810": "한국항공우주",
    "064350": "현대로템",
    "272210": "한화시스템",
    "079550": "LIG디펜스앤에어로스페이스",
    # 은행
    "055550": "신한지주",
    "086790": "하나금융지주",
    "105560": "KB금융",
    "316140": "우리금융지주",
    # 증권
    "006800": "미래에셋증권",
    "071050": "한국금융지주",
    "016360": "삼성증권",
    "039490": "키움증권",
    # 인터넷통신
    "017670": "SK텔레콤",
    "030200": "KT",
    "032640": "LG유플러스",
    "035420": "NAVER",
    "035720": "카카오",
}

KOSPI_INDEX_CODE      = "1001"   # pykrx KOSPI 종합지수 (코스피) 코드
_PYKRX_KOSPI_FALLBACK = "1028"   # 일부 pykrx 버전 폴백 코드 (KOSPI200 또는 구버전 코드)
# ※ "1028"은 IT 업종 지수일 수 있음 — 데이터 범위 검증 필수
CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_MAX_AGE_DAYS = 1             # 하루 이상 된 캐시는 갱신

# KOSPI 종합지수 합리적 Close 범위 (1001 기준)
# 역사적 최저 ~500pt, 최고 ~3,300pt (2024~2026 기준 2,200~2,900)
# 5,000pt 초과 = 분명히 업종 지수 (1028 IT 등) → 무효
_KOSPI_CLOSE_MIN = 1_000.0
_KOSPI_CLOSE_MAX = 5_000.0

# pykrx 한글 컬럼 → 영문 매핑
_COL_MAP = {
    "시가": "Open",
    "고가": "High",
    "저가": "Low",
    "종가": "Close",
    "거래량": "Volume",
    "등락률": "Change",
}


# ---------------------------------------------------------------------------
# 내부 유틸
# ---------------------------------------------------------------------------

def _is_valid_kospi(df: pd.DataFrame) -> bool:
    """
    KOSPI 종합지수(1001) 데이터인지 Close 범위로 검증한다.

    1001 기준 Close 는 통상 1,000 ~ 5,000pt 내에 있다.
    5,000 초과 → IT 업종지수(1028) 등 오수집 가능성 높음 → False.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return False
    close = df["Close"].dropna()
    if close.empty:
        return False
    return bool(
        close.min() >= _KOSPI_CLOSE_MIN and close.max() <= _KOSPI_CLOSE_MAX
    )


def _date_range(months: int = 18) -> tuple[str, str]:
    """오늘 기준 `months`개월 전 ~ 오늘 날짜를 'YYYYMMDD' 형식으로 반환."""
    end = datetime.today()
    # 월 단위 근사: 30.4375일 × months
    start = end - timedelta(days=int(months * 30.4375))
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def _cache_path(key: str) -> Path:
    """캐시 파일 경로 반환 (폴더가 없으면 생성)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.parquet"


def _is_cache_valid(path: Path) -> bool:
    """캐시 파일이 존재하고 CACHE_MAX_AGE_DAYS 이내면 True."""
    if not path.exists():
        return False
    age_sec = datetime.now().timestamp() - path.stat().st_mtime
    return age_sec < CACHE_MAX_AGE_DAYS * 86_400


def _from_vault_to_ohlcv(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    longlivevault get_ohlcv() 반환값을 Magic Formula 표준 형식으로 변환한다.

    vault 형식: Date(datetime 컬럼), Ticker, Open, High, Low, Close, Volume, ...
    MF 형식:    DatetimeIndex('Date'), Open, High, Low, Close, Volume
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

    # OHLCV 5개 컬럼만 유지
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    if not keep:
        return pd.DataFrame()
    df = df[keep].copy()

    # 숫자 변환 + Volume=0 행 제거
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df.get("Volume", pd.Series(1, index=df.index)) > 0]
    df = df.sort_index()

    # 날짜 범위 필터
    start_dt = pd.to_datetime(start, format="%Y%m%d")
    end_dt   = pd.to_datetime(end,   format="%Y%m%d")
    df = df[(df.index >= start_dt) & (df.index <= end_dt)]

    return df


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    pykrx에서 받은 DataFrame을 표준 형식으로 정리한다.
    - 한글 컬럼 → 영문
    - 인덱스 → DatetimeIndex
    - OHLCV 5개 컬럼만 유지
    - Volume = 0인 행(비거래일) 제거
    """
    df = df.rename(columns=_COL_MAP)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    df = df[keep].copy()
    df = df[df.get("Volume", pd.Series(1, index=df.index)) > 0]
    df = df.sort_index()
    return df


# ---------------------------------------------------------------------------
# 공개 함수
# ---------------------------------------------------------------------------

def fetch_ohlcv(
    ticker: str,
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    단일 종목 OHLCV 데이터를 반환한다.

    Parameters
    ----------
    ticker   : 종목 코드 (예: '000660')
    start    : 조회 시작일 'YYYYMMDD'
    end      : 조회 종료일 'YYYYMMDD'
    use_cache: True이면 유효한 캐시가 있을 때 API를 호출하지 않는다.

    Returns
    -------
    pd.DataFrame  Open/High/Low/Close/Volume, DatetimeIndex
    """
    cache_path = _cache_path(ticker)
    name       = TICKERS.get(ticker, ticker)

    if use_cache and _is_cache_valid(cache_path):
        df = pd.read_parquet(cache_path)
        print(f"  [cache] {ticker} ({name}): {len(df)} rows")
        return df

    # --- Stage 1: longlivevault (core.parquet + KIS) ---
    if _VAULT_AVAILABLE:
        try:
            raw = vault_get_ohlcv(ticker, start_date=start, end_date=end)
            df  = _from_vault_to_ohlcv(raw, start, end)
            if not df.empty:
                df.to_parquet(cache_path)
                print(f"  [vault] {ticker} ({name}): {len(df)} rows ✓")
                return df
        except Exception as exc:
            print(f"  [vault] {ticker} 실패: {exc} → pykrx 폴백")

    # --- Stage 2: pykrx ---
    print(f"  [fetch] {ticker} ({name}): {start} ~ {end} ...", end=" ", flush=True)
    try:
        raw = krx_stock.get_market_ohlcv_by_date(start, end, ticker)
        if raw is None or raw.empty:
            print("⚠ empty")
            return pd.DataFrame()

        df = _normalize_ohlcv(raw)
        df.to_parquet(cache_path)
        print(f"✓ {len(df)} rows")
        return df

    except Exception as exc:
        print(f"✗ ERROR: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# KIS REST API 직접 호출 (Stage 2)
# ---------------------------------------------------------------------------

def _load_kis_config() -> dict:
    """
    KIS 설정 yaml을 여러 경로에서 탐색하여 유효한 앱키가 있는 것을 반환.
    유효한 yaml이 없으면 ValueError를 발생시킨다.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("pyyaml 미설치. pip install pyyaml --break-system-packages")

    for path in _KIS_YAML_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="UTF-8") as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)
            app_key    = str(cfg.get("my_app", "")).strip()
            app_secret = str(cfg.get("my_sec", "")).strip()
            # 플레이스홀더("앱키") 또는 짧은 더미 값 제외
            if app_key and "앱키" not in app_key and len(app_key) >= 16:
                return cfg
        except Exception:
            continue

    raise ValueError(
        "유효한 KIS 설정 파일을 찾을 수 없습니다.\n"
        "아래 경로 중 하나에 kis_devlp.yaml 을 배치하고\n"
        "my_app / my_sec 항목에 실제 앱키를 입력하세요:\n"
        + "\n".join(f"  {p}" for p in _KIS_YAML_CANDIDATES)
    )


def _get_kis_access_token(app_key: str, app_secret: str) -> str:
    """KIS OAuth2 접근 토큰 발급 (실전투자 기준)."""
    import requests
    url  = "https://openapi.koreainvestment.com:9443/oauth2/tokenP"
    body = {
        "grant_type": "client_credentials",
        "appkey":     app_key,
        "appsecret":  app_secret,
    }
    resp = requests.post(url, json=body, timeout=15)
    resp.raise_for_status()
    token = resp.json().get("access_token", "")
    if not token:
        raise ValueError("KIS 토큰 발급 실패: " + str(resp.json()))
    return token


def _fetch_kospi_kis_rest(start: str, end: str) -> pd.DataFrame:
    """
    KIS REST API 직접 호출로 KOSPI 일별 데이터를 수집한다.

    yaml 설정 파일에서 앱키/앱시크릿을 읽고,
    /uapi/domestic-stock/v1/quotations/inquire-index-daily-price 엔드포인트를
    최대 5회 페이징하여 18개월치(~390거래일) 데이터를 가져온다.
    """
    import requests

    cfg        = _load_kis_config()
    app_key    = cfg["my_app"]
    app_secret = cfg["my_sec"]
    token      = _get_kis_access_token(app_key, app_secret)

    api_url = (
        "https://openapi.koreainvestment.com:9443"
        "/uapi/domestic-stock/v1/quotations/inquire-index-daily-price"
    )
    headers = {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appkey":        app_key,
        "appsecret":     app_secret,
        "tr_id":         "FHPUP02120000",
        "custtype":      "P",
    }
    params = {
        "FID_PERIOD_DIV_CODE":   "D",
        "FID_COND_MRKT_DIV_CODE": "U",
        "FID_INPUT_ISCD":        "0001",   # KOSPI 종합지수
        "FID_INPUT_DATE_1":      end,
    }

    all_rows: list = []
    tr_cont = ""

    for _ in range(5):   # 최대 5페이지 (약 500거래일)
        if tr_cont:
            headers["tr_cont"] = tr_cont

        resp = requests.get(api_url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        body = resp.json()

        rows = body.get("output2", [])
        if not rows:
            break
        all_rows.extend(rows if isinstance(rows, list) else [rows])

        tr_cont = resp.headers.get("tr_cont", "")
        if tr_cont not in ("M", "F"):
            break

    if not all_rows:
        raise ValueError("KIS REST API — output2 데이터 없음")

    df = pd.DataFrame(all_rows)
    col_map = {
        "bstp_nmix_oprc": "Open",
        "bstp_nmix_hgpr": "High",
        "bstp_nmix_lwpr": "Low",
        "bstp_nmix_prpr": "Close",
        "acml_vol":        "Volume",
        "stck_bsop_date":  "_date",
    }
    df = df.rename(columns=col_map)

    if "_date" not in df.columns:
        raise ValueError("KIS 응답에 날짜 컬럼(stck_bsop_date) 없음")

    df["Date"] = pd.to_datetime(df["_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

    start_dt = pd.to_datetime(start, format="%Y%m%d")
    df = df[df.index >= start_dt]

    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    df = df[keep].copy()
    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]
    df.index.name = "Date"
    return df


# ---------------------------------------------------------------------------
# yfinance 폴백 (Stage 3)
# ---------------------------------------------------------------------------

def _fetch_kospi_yfinance(start: str, end: str) -> pd.DataFrame:
    """
    yfinance 를 이용해 KOSPI(^KS11) 일별 데이터를 수집한다.

    필요 패키지: pip install yfinance --break-system-packages
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance 미설치. pip install yfinance --break-system-packages")

    start_dt = pd.to_datetime(start, format="%Y%m%d")
    end_dt   = pd.to_datetime(end,   format="%Y%m%d") + pd.Timedelta(days=1)

    ticker = yf.Ticker("^KS11")
    df     = ticker.history(
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        auto_adjust=True,
    )

    if df is None or df.empty:
        raise ValueError("yfinance ^KS11 데이터 없음")

    # timezone 제거
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"

    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    df = df[keep].copy()
    df = df[df["Close"] > 0]
    return df


# ---------------------------------------------------------------------------
# 기존 KIS open-trading-api 라이브러리 방식 (레거시, 사용 안 함)
# ---------------------------------------------------------------------------

def _fetch_kospi_pykrx(start: str, end: str) -> pd.DataFrame:
    """
    pykrx로 KOSPI 종합지수(1001) OHLCV를 수집한다.

    코드 시도 순서: 1001(KOSPI 종합) → 1028(폴백)
    실패 시 빈 DataFrame 반환.
    수집 성공 시 사용된 코드와 Close 가격 범위를 출력해 잘못된 지수 사용을 방지한다.
    """
    for code in (KOSPI_INDEX_CODE, _PYKRX_KOSPI_FALLBACK):
        try:
            raw = krx_stock.get_index_ohlcv_by_date(start, end, code)
            if raw is None or raw.empty:
                continue
            df = _normalize_ohlcv(raw)
            df = df[df["Close"] > 0]
            if not df.empty:
                c_min = df["Close"].min()
                c_max = df["Close"].max()
                print(
                    f"     [pykrx] 코드={code}  "
                    f"Close 범위: {c_min:,.0f} ~ {c_max:,.0f}  "
                    f"({len(df)}일)"
                )
                if _is_valid_kospi(df):
                    return df
                else:
                    print(
                        f"     ⚠ 코드={code} Close 범위({c_min:.0f}~{c_max:.0f}pt)가 "
                        f"KOSPI 1001 범위({_KOSPI_CLOSE_MIN:.0f}~{_KOSPI_CLOSE_MAX:.0f}pt)를 벗어남 "
                        f"— 업종/파생 지수 오수집 의심, 이 코드 건너뜀"
                    )
        except Exception:
            continue
    return pd.DataFrame()


def _fetch_kospi_from_vault(start: str, end: str) -> pd.DataFrame:
    """
    longlivevault의 data/raw/krx_kospi_*.parquet 파일에서 KOSPI 인덱스 데이터를 취합한다.

    각 raw 파일에는 당일 전체 KOSPI 종목 snapshot이 담겨 있다.
    여기서 KOSPI 종합지수(Close 가중평균 대신 시장 대표가 없으므로)를
    직접 제공하지 않지만, 별도 krx_index 파일이 없으면 빈 DataFrame 반환.

    실제로는 pykrx 방식이 KOSPI 지수를 더 정확히 제공하므로,
    vault raw 파일에 별도 코스피 지수 컬럼/파일이 있는 경우만 사용.
    """
    vault_raw = Path(VAULT_PATH) / "data" / "raw"
    if not vault_raw.exists():
        return pd.DataFrame()

    # krx_kospi_YYYYMMDD.parquet 파일 목록
    raw_files = sorted(vault_raw.glob("krx_kospi_*.parquet"))
    if not raw_files:
        return pd.DataFrame()

    start_dt = pd.to_datetime(start, format="%Y%m%d")
    end_dt   = pd.to_datetime(end,   format="%Y%m%d")

    rows = []
    for fp in raw_files:
        # 파일명에서 날짜 추출: krx_kospi_YYYYMMDD.parquet
        try:
            date_str = fp.stem.split("_")[-1]   # 'YYYYMMDD'
            file_dt  = pd.to_datetime(date_str, format="%Y%m%d")
        except Exception:
            continue

        if not (start_dt <= file_dt <= end_dt):
            continue

        try:
            fdf = pd.read_parquet(fp)
            # raw 파일에 KOSPI 지수 행이 있는지 확인 (ticker "1001" 또는 인덱스 행)
            # 일반적으로 raw 파일은 개별 종목 snapshot → 지수 없음
            # 지수 행 없으면 skip
            if "Ticker" not in fdf.columns and "ticker" not in fdf.columns:
                continue
        except Exception:
            continue

    # raw 파일에 지수가 없음 → 빈 DataFrame (pykrx 폴백으로 넘어감)
    return pd.DataFrame()


def _fetch_kospi_kis(end: str, start: str) -> pd.DataFrame:
    """
    KIS open-trading-api로 KOSPI 지수를 수집한다.

    경로: /Users/kaneyoun/open-trading-api/examples_llm/
    인증: ~/KIS/config/kis_devlp.yaml (open-trading-api 기본 설정)

    반환: Open/High/Low/Close/Volume, DatetimeIndex
    """
    # sys.path에 KIS 라이브러리 경로 추가
    for p in (_KIS_EXAMPLES_PATH, _KIS_INDEX_PATH):
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        import kis_auth as ka                                      # type: ignore
        from inquire_index_daily_price import (                    # type: ignore
            inquire_index_daily_price,
        )
    except ImportError as e:
        raise ImportError(f"KIS 라이브러리 로드 실패: {e}") from e

    # KIS API 인증 (토큰이 이미 있으면 재사용)
    try:
        ka.auth()
    except Exception as e:
        raise RuntimeError(f"KIS 인증 실패: {e}") from e

    # 18개월 데이터 = 최대 ~390거래일 → max_depth=5 (500행) 충분
    _, df2 = inquire_index_daily_price(
        fid_period_div_code="D",
        fid_cond_mrkt_div_code="U",
        fid_input_iscd="0001",    # KOSPI 종합지수
        fid_input_date_1=end,
        max_depth=5,
    )

    if df2 is None or df2.empty:
        raise ValueError("KIS API KOSPI 데이터 없음")

    # 컬럼 매핑 (KIS 한국어 약어 → 표준)
    col_map = {
        "bstp_nmix_oprc": "Open",
        "bstp_nmix_hgpr": "High",
        "bstp_nmix_lwpr": "Low",
        "bstp_nmix_prpr": "Close",
        "acml_vol":        "Volume",
        "stck_bsop_date":  "_date",
    }
    df2 = df2.rename(columns=col_map)

    # 날짜 인덱스 설정
    if "_date" not in df2.columns:
        raise ValueError("KIS 응답에 날짜 컬럼(stck_bsop_date) 없음")

    df2["Date"] = pd.to_datetime(df2["_date"], format="%Y%m%d", errors="coerce")
    df2 = df2.dropna(subset=["Date"]).set_index("Date").sort_index()

    # start 날짜 이후로 필터
    start_dt = pd.to_datetime(start, format="%Y%m%d")
    df2 = df2[df2.index >= start_dt]

    # OHLCV 숫자 변환 (KIS는 문자열 반환, 쉼표 포함 가능)
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df2.columns]
    df2 = df2[keep].copy()
    for col in df2.columns:
        df2[col] = pd.to_numeric(
            df2[col].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
    df2 = df2.dropna(subset=["Close"])
    df2 = df2[df2["Close"] > 0]
    df2.index.name = "Date"

    return df2


def fetch_kospi(
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    KOSPI 종합지수 OHLCV 데이터를 반환한다.

    수집 우선순위:
      1. pykrx          (코드 1028, 실패 시 1001 재시도)
      2. KIS REST API   (~/KIS/config/kis_devlp.yaml 또는 ~/open-trading-api/kis_devlp.yaml)
      3. yfinance       (^KS11, pip install yfinance 필요)
      4. 모두 실패 시   빈 DataFrame 반환 (알파 계산 생략됨)
    """
    cache_path = _cache_path("KOSPI")

    if use_cache and _is_cache_valid(cache_path):
        df = pd.read_parquet(cache_path)
        if _is_valid_kospi(df):
            print(f"  [cache] KOSPI: {len(df)} rows")
            return df
        else:
            close_min = df["Close"].min() if not df.empty else float("nan")
            close_max = df["Close"].max() if not df.empty else float("nan")
            print(
                f"  [cache] KOSPI 캐시 무효 — Close 범위 {close_min:.0f}~{close_max:.0f}pt "
                f"가 1001 종합지수 범위({_KOSPI_CLOSE_MIN:.0f}~{_KOSPI_CLOSE_MAX:.0f}pt) 벗어남 "
                f"→ 재수집 후 덮어씌움"
            )
            # unlink 대신 fall-through: 성공한 단계가 cache_path에 덮어씀

    print(f"  [fetch] KOSPI {start}~{end}")

    # --- Stage 0: vault raw krx_kospi 파일 ---
    print("    Stage 0: vault raw...", end=" ", flush=True)
    try:
        df = _fetch_kospi_from_vault(start, end)
        if not df.empty:
            df.to_parquet(cache_path)
            print(f"✓ {len(df)} rows")
            return df
        print("✗ (지수 데이터 없음 → pykrx)")
    except Exception as exc:
        print(f"✗ {exc}")

    # --- Stage 1: pykrx ---
    print("    Stage 1: pykrx...", end=" ", flush=True)
    try:
        df = _fetch_kospi_pykrx(start, end)
        if not df.empty:
            df.to_parquet(cache_path)
            print(f"✓ {len(df)} rows")
            return df
        print("✗ empty")
    except Exception as exc:
        print(f"✗ {exc}")

    # --- Stage 2: KIS REST API (직접 호출) ---
    print("    Stage 2: KIS REST API...", end=" ", flush=True)
    try:
        df = _fetch_kospi_kis_rest(start, end)
        if not df.empty:
            df.to_parquet(cache_path)
            print(f"✓ {len(df)} rows")
            return df
        print("✗ empty")
    except Exception as exc:
        print(f"✗ {exc}")

    # --- Stage 3: yfinance ---
    print("    Stage 3: yfinance (^KS11)...", end=" ", flush=True)
    try:
        df = _fetch_kospi_yfinance(start, end)
        if not df.empty:
            df.to_parquet(cache_path)
            print(f"✓ {len(df)} rows")
            return df
        print("✗ empty")
    except Exception as exc:
        print(f"✗ {exc}")

    print("  ⚠ KOSPI 데이터 수집 실패 (4단계 모두 실패) — 알파 계산 생략")
    return pd.DataFrame()


def collect_all(
    months: int = 18,
    use_cache: bool = True,
    ticker_list: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    지정 종목 + KOSPI 지수를 일괄 수집한다.

    Parameters
    ----------
    months      : 조회 기간 (기본 18개월)
    use_cache   : 캐시 사용 여부
    ticker_list : 수집할 종목 코드 리스트.
                  None이면 TICKERS(전체 59종목) 사용.

    Returns
    -------
    dict  {ticker_or_'KOSPI': DataFrame}
    실패한 종목은 결과에서 제외된다.
    """
    if ticker_list is None:
        ticker_list = list(TICKERS.keys())

    n_total = len(ticker_list)
    start, end = _date_range(months)
    print(f"\n{'='*60}")
    print(f"데이터 수집: {start} ~ {end}  ({months}개월)")
    print(f"{'='*60}")

    data: dict[str, pd.DataFrame] = {}

    # KOSPI 지수
    kospi = fetch_kospi(start, end, use_cache)
    if not kospi.empty:
        data["KOSPI"] = kospi
    else:
        print("  ⚠ KOSPI 데이터 없음 — 알파 계산 불가")

    # 개별 종목
    for ticker in ticker_list:
        df = fetch_ohlcv(ticker, start, end, use_cache)
        if not df.empty:
            data[ticker] = df

    n_stocks = len(data) - (1 if "KOSPI" in data else 0)
    print(f"\n수집 완료: {n_stocks}/{n_total} 종목, KOSPI {'포함' if 'KOSPI' in data else '제외'}")
    return data


def get_backtest_split(
    df: pd.DataFrame,
    warmup_months: int = 12,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    전체 인덱스를 워밍업 구간과 실거래 구간으로 분리한다.

    Returns
    -------
    (warmup_idx, trade_idx)
    warmup_idx : 지표 계산용 (처음 12개월)
    trade_idx  : 실제 매매 시뮬레이션 구간 (마지막 6개월)
    """
    cutoff = df.index[0] + pd.DateOffset(months=warmup_months)
    warmup_idx = df.index[df.index < cutoff]
    trade_idx = df.index[df.index >= cutoff]
    return warmup_idx, trade_idx
