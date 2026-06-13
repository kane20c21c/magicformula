"""
_vault.py
---------
longlivevault 진입점 통합 헬퍼.

이전에는 main.py / optimizer.py / collector.py / daily_signal.py 각각이
- _VAULT_PATH 후보 리스트
- sys.path 삽입
- CORE_TICKERS 임포트 + 폴백
- EXCLUDE_TICKERS 정의
- TICKER_NAMES 한글 dict
- SECTOR_MAP / SECTOR_ORDER
를 따로 들고 있었다. 본 모듈이 이 모두를 단일 진실 원천(SSOT)으로 제공한다.

데이터의 정본
-------------
- ticker 목록 : longlivevault.stolab_data.ohlcv_store.CORE_TICKERS
- 섹터 매핑   : longlivevault.stolab_data.core_tickers.TICKER_LIST (튜플 3번째)
- 종목명     : OHLCV parquet 의 Name 컬럼 (vault 가 자동 채움)
              → 빈 경우 본 모듈의 TICKER_NAMES_FALLBACK 사용

공개 API
--------
- VAULT_PATH, VAULT_AVAILABLE     : 모듈 임포트 시 결정되는 상태
- CORE_TICKERS                    : frozenset[str]
- TICKER_SECTORS                  : dict[str, str] (ticker → sector)
- SECTOR_ORDER                    : list[str] (vault 정의 순서)
- DEFAULT_EXCLUDE                 : frozenset[str] (분석 제외 종목)
- TICKER_NAMES_FALLBACK           : dict[str, str]
- get_universe(name) -> list[str] : universe 식별자 해석
- get_ticker_name(ticker, parquet_name=None) -> str
- get_sector(ticker) -> str

universe 식별자
---------------
- "core_all"        : vault CORE_TICKERS 전체 — 데일리용 (현재 69개)
- "core_excl_split" : core_all - DEFAULT_EXCLUDE — 백테스트용 (현재 67개)
- "core_59" / "core_57" : 위의 두 식별자에 대한 deprecated alias
                          (vault 코어가 60→69 로 확대돼 숫자가 안 맞지만,
                           기존 yaml/launchd/스크립트 호환을 위해 유지)
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# vault 경로 탐색 + sys.path 등록
# ---------------------------------------------------------------------------

_MAGIC_ROOT = Path(__file__).resolve().parent.parent  # Magic Formula/

_VAULT_CANDIDATES = [
    _MAGIC_ROOT.parent / "longlivevault",                       # 형제 폴더 (환경 독립)
    Path("/Users/kaneyoun/DriveForALL/StoLab/longlivevault"),   # 절대 경로 fallback
]

VAULT_PATH: Path = next(
    (p for p in _VAULT_CANDIDATES if p.exists()),
    _VAULT_CANDIDATES[0],
)

if str(VAULT_PATH) not in sys.path:
    sys.path.insert(0, str(VAULT_PATH))

# ---------------------------------------------------------------------------
# vault 모듈 import + 폴백 처리
# ---------------------------------------------------------------------------

try:
    from stolab_data.ohlcv_store import CORE_TICKERS as _VAULT_CORE       # type: ignore
    from stolab_data.core_tickers import TICKER_LIST as _VAULT_TICKER_LIST  # type: ignore
    VAULT_AVAILABLE: bool = True
except ImportError:
    _VAULT_CORE = frozenset()
    _VAULT_TICKER_LIST = []
    VAULT_AVAILABLE = False

# 확장(extend) 종목 — vault 가 더 넓은 모니터링 유니버스(시총 200)를 위해 제공.
# 미설치/구버전 vault 면 빈 집합으로 폴백 (core 만으로 동작).
try:
    from stolab_data.extend_tickers import (  # type: ignore
        EXTEND_TICKERS as _VAULT_EXTEND,
        EXTEND_LIST as _VAULT_EXTEND_LIST,
    )
except ImportError:
    _VAULT_EXTEND = frozenset()
    _VAULT_EXTEND_LIST = []

# ---------------------------------------------------------------------------
# 정본 데이터
# ---------------------------------------------------------------------------

# 분석 제외 종목 (사업 분할 / 합병 / 거래정지 등)
DEFAULT_EXCLUDE: frozenset[str] = frozenset({
    "207940",   # 삼성바이오로직스 — 사업 분할
    "0126Z0",   # 삼성에피스홀딩스 — 사업 분할
})

# 종목 목록 (vault 정본, vault 미설치 시 빈 집합)
CORE_TICKERS: frozenset[str] = frozenset(_VAULT_CORE)

# 확장 종목 집합 (vault extend.parquet 대상 131종목)
EXTEND_TICKERS: frozenset[str] = frozenset(_VAULT_EXTEND)

# 섹터 매핑 (vault TICKER_LIST + EXTEND_LIST 에서 빌드)
# core 와 extend 가 겹치지 않으므로 단순 병합. core 우선.
TICKER_SECTORS: dict[str, str] = {
    t: s for t, _name, s in _VAULT_EXTEND_LIST
}
TICKER_SECTORS.update({
    t: s for t, _name, s in _VAULT_TICKER_LIST
})

# 섹터 순서 (vault 정의 순서대로 등장 순) — core 먼저, 이어서 extend 신규 섹터
SECTOR_ORDER: list[str] = []
for _t, _n, _s in list(_VAULT_TICKER_LIST) + list(_VAULT_EXTEND_LIST):
    if _s not in SECTOR_ORDER:
        SECTOR_ORDER.append(_s)

# 종목명 fallback (vault Name 컬럼이 None / 빈 문자열일 때 사용)
# 출처: collector.TICKERS 와 daily_signal.TICKER_NAMES 통합본
# 차이가 있는 종목은 더 자주 사용되는 표기를 채택
TICKER_NAMES_FALLBACK: dict[str, str] = {
    # 반도체
    "000660": "SK하이닉스",       "005930": "삼성전자",          "042700": "한미반도체",
    "058470": "리노공업",         "240810": "원익IPS",           "039030": "이오테크닉스",
    "000990": "DB하이텍",         "403870": "HPSP",              "357780": "솔브레인",
    "005290": "동진쎄미켐",
    # 반도체 핵심장비
    "000150": "두산",             "007660": "이수페타시스",      "095340": "ISC",
    # 로봇
    "058610": "에스피지",         "277810": "레인보우로보틱스",  "108490": "로보티즈",
    "454910": "두산로보틱스",     "348340": "뉴로메카",          "056080": "유진로봇",
    # 에너지·전송
    "010120": "LS ELECTRIC",      "298040": "효성중공업",        "267260": "HD현대일렉트릭",
    "001440": "대한전선",
    # 에너지·생산
    "000720": "현대건설",         "047040": "대우건설",          "034020": "두산에너빌리티",
    "052690": "한전기술",         "032820": "우리기술",
    # 에너지·보관
    "373220": "LG에너지솔루션",   "006400": "삼성SDI",           "005490": "POSCO홀딩스",
    "247540": "에코프로비엠",     "051910": "LG화학",            "086520": "에코프로",
    # 바이오
    "068270": "셀트리온",         "196170": "알테오젠",
    "207940": "삼성바이오로직스", "0126Z0": "삼성에피스홀딩스",
    # 조선
    "329180": "HD현대중공업",     "042660": "한화오션",          "010140": "삼성중공업",
    # 방산
    "012450": "한화에어로스페이스","047810": "한국항공우주",      "064350": "현대로템",
    "272210": "한화시스템",       "079550": "LIG디펜스앤에어로스페이스",   # ← 사명변경(舊 LIG넥스원)
    # 은행
    "055550": "신한지주",         "086790": "하나금융지주",      "105560": "KB금융",
    "316140": "우리금융지주",
    # 증권
    "006800": "미래에셋증권",     "071050": "한국금융지주",      "016360": "삼성증권",
    "039490": "키움증권",
    # 인터넷통신
    "017670": "SK텔레콤",         "030200": "KT",                "032640": "LG유플러스",
    "035420": "NAVER",            "035720": "카카오",
    # 자동차
    "005380": "현대차",           "012330": "현대모비스",        "000270": "기아",
    "161390": "한국타이어",       "204320": "HL만도",
    # 자동차소부장
    "009150": "삼성전기",         "011070": "LG이노텍",          "066570": "LG전자",
    "307950": "현대오토에버",
    # ETF
    "102110": "TIGER 200",
}

# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

# universe 식별자 → ticker 집합 빌더
# 정식 이름: core_all (전체) / core_excl_split (분석용)
# core_59 / core_57 은 vault 가 60종목이던 시기의 deprecated alias —
#   숫자는 더 이상 일치하지 않지만 호환을 위해 보존 (각각 core_all / core_excl_split 와 동일).
_UNIVERSE_BUILDERS: dict[str, "callable"] = {
    "core_all":        lambda: sorted(CORE_TICKERS),
    "core_excl_split": lambda: sorted(CORE_TICKERS - DEFAULT_EXCLUDE),
    # ── 확장 유니버스 (core ∪ extend = 시총 200) ───────────────
    "extended_all":        lambda: sorted(CORE_TICKERS | EXTEND_TICKERS),
    "extended_excl_split": lambda: sorted((CORE_TICKERS | EXTEND_TICKERS) - DEFAULT_EXCLUDE),
    # ── deprecated aliases (호환용) ────────────────────────────
    "core_59":         lambda: sorted(CORE_TICKERS),                       # core_all alias
    "core_57":         lambda: sorted(CORE_TICKERS - DEFAULT_EXCLUDE),     # core_excl_split alias
}


def get_universe(name: str = "core_excl_split") -> list[str]:
    """
    universe 식별자를 ticker list 로 해석한다.

    Parameters
    ----------
    name : "core_all" / "core_excl_split"
           (백테스트/분석 기본값은 "core_excl_split", 데일리는 "core_all")
           "core_59" / "core_57" 도 동작하나 deprecated — 새 코드는 사용 금지.

    Returns
    -------
    정렬된 ticker 리스트.
    vault 미설치 시 빈 리스트.
    """
    if not VAULT_AVAILABLE:
        return []

    builder = _UNIVERSE_BUILDERS.get(name)
    if builder is None:
        raise ValueError(
            f"알 수 없는 universe={name!r}. "
            f"허용: {sorted(_UNIVERSE_BUILDERS)}"
        )
    return builder()


def get_sector(ticker: str) -> str:
    """
    종목의 섹터를 반환한다. vault 에 없으면 '기타'.
    """
    return TICKER_SECTORS.get(ticker, "기타")


_INVALID_NAME_TOKENS = frozenset({"None", "nan", "NaN", "<NA>", "NA", "null"})


def get_ticker_name(ticker: str, parquet_name=None) -> str:
    """
    종목명을 반환한다.

    우선순위:
    1. parquet_name (vault OHLCV 의 Name 컬럼) 이 의미있는 값이면 그걸 사용
    2. TICKER_NAMES_FALLBACK 의 값
    3. ticker 자체 (둘 다 없을 때)

    의미있다고 판정하지 않는 값들:
    - Python ``None``
    - ``pandas.NA`` / ``numpy.nan`` (pd.isna 가 True)
    - 빈 문자열 / 공백만 있는 문자열
    - 문자열 ``"None"`` / ``"nan"`` / ``"NaN"`` / ``"<NA>"`` / ``"NA"`` / ``"null"``
    """
    # 1) None
    if parquet_name is None:
        return TICKER_NAMES_FALLBACK.get(ticker, ticker)

    # 2) pandas/numpy NA-like (pd.NA, np.nan, NaT 등)
    try:
        import pandas as _pd
        if _pd.isna(parquet_name):
            return TICKER_NAMES_FALLBACK.get(ticker, ticker)
    except (ImportError, TypeError, ValueError):
        # pd.isna 가 적용 안 되는 타입은 그냥 문자열 검사로 진행
        pass

    # 3) 문자열 정규화 후 invalid 토큰 검사
    s = str(parquet_name).strip()
    if s and s not in _INVALID_NAME_TOKENS:
        return s

    return TICKER_NAMES_FALLBACK.get(ticker, ticker)


# ---------------------------------------------------------------------------
# 디버그용 CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"VAULT_PATH       : {VAULT_PATH}")
    print(f"VAULT_AVAILABLE  : {VAULT_AVAILABLE}")
    print(f"CORE_TICKERS     : {len(CORE_TICKERS)}개")
    print(f"DEFAULT_EXCLUDE  : {sorted(DEFAULT_EXCLUDE)}")
    print(f"SECTOR_ORDER     : {SECTOR_ORDER}")
    print(f"core_all         : {len(get_universe('core_all'))}개")
    print(f"core_excl_split  : {len(get_universe('core_excl_split'))}개")
    print(f"  (legacy alias) core_59 = {len(get_universe('core_59'))}, "
          f"core_57 = {len(get_universe('core_57'))}")
    print(f"079550 sector    : {get_sector('079550')!r}")
    print(f"079550 name      : {get_ticker_name('079550')!r}")
    print(f"079550 name(vault): {get_ticker_name('079550', 'LIG디펜스앤에어로스페이스')!r}")
