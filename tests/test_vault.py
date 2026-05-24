"""
tests/test_vault.py
-------------------
_vault 모듈의 종목명/유니버스/검증 동작.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from magic_formula._vault import (
    CORE_TICKERS,
    DEFAULT_EXCLUDE,
    TICKER_NAMES_FALLBACK,
    SECTOR_ORDER,
    get_sector,
    get_ticker_name,
    get_universe,
)


# ---------------------------------------------------------------------------
# get_ticker_name — 강건성
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("invalid_value", [
    None, "", "   ",
    "None", "nan", "NaN", "<NA>", "NA", "null",
    np.nan, pd.NA, pd.NaT,
])
def test_get_ticker_name_falls_back_for_invalid(invalid_value):
    """invalid 값(None / NaN / 빈 문자열 / NA 토큰)이면 fallback 사용."""
    assert get_ticker_name("000660", invalid_value) == "SK하이닉스"


def test_get_ticker_name_uses_vault_name_when_valid():
    """vault Name 컬럼이 의미있는 값이면 그대로 사용 (fallback 보다 우선)."""
    assert get_ticker_name("000660", "삼성전자") == "삼성전자"     # 의도적 mismatch — vault 우선
    assert get_ticker_name("000660", "  앞뒤공백  ") == "앞뒤공백"   # strip 적용


def test_get_ticker_name_falls_back_to_ticker_when_unknown():
    """fallback dict 에도 없는 ticker 면 ticker 자체 반환."""
    assert get_ticker_name("999999", None) == "999999"


# ---------------------------------------------------------------------------
# universe
# ---------------------------------------------------------------------------

def test_core_tickers_basic():
    """vault CORE_TICKERS 는 비어있지 않고, EXCLUDE 는 사업분할 2종목."""
    assert len(CORE_TICKERS) > 0
    assert DEFAULT_EXCLUDE == frozenset({"207940", "0126Z0"})
    # EXCLUDE 종목은 반드시 CORE_TICKERS 의 부분집합
    assert DEFAULT_EXCLUDE.issubset(CORE_TICKERS)


def test_universe_core_all():
    """core_all = CORE_TICKERS 전체."""
    universe = set(get_universe("core_all"))
    assert universe == set(CORE_TICKERS)
    assert len(get_universe("core_all")) == len(CORE_TICKERS)


def test_universe_core_excl_split():
    """core_excl_split = CORE_TICKERS - DEFAULT_EXCLUDE."""
    universe = set(get_universe("core_excl_split"))
    assert universe == set(CORE_TICKERS) - DEFAULT_EXCLUDE
    assert len(universe) == len(CORE_TICKERS) - len(DEFAULT_EXCLUDE)


def test_universe_legacy_aliases():
    """core_59 / core_57 deprecated alias 가 새 이름과 동일 결과."""
    assert get_universe("core_59") == get_universe("core_all")
    assert get_universe("core_57") == get_universe("core_excl_split")


def test_universe_unknown_raises():
    with pytest.raises(ValueError, match="알 수 없는 universe"):
        get_universe("unknown_universe")


# ---------------------------------------------------------------------------
# sector / ticker names
# ---------------------------------------------------------------------------

def test_known_sector():
    assert get_sector("000660") == "반도체"
    assert get_sector("005930") == "반도체"
    assert get_sector("079550") == "방산"


def test_unknown_sector_returns_default():
    assert get_sector("999999") == "기타"


def test_sector_order_complete():
    """모든 SECTOR_ORDER 항목이 ticker 매핑에 존재."""
    from magic_formula._vault import TICKER_SECTORS
    sectors_in_use = set(TICKER_SECTORS.values())
    assert sectors_in_use == set(SECTOR_ORDER)


def test_ticker_names_fallback_covers_all_core():
    """모든 vault CORE_TICKERS 가 TICKER_NAMES_FALLBACK 에 있어야 함."""
    missing = CORE_TICKERS - set(TICKER_NAMES_FALLBACK.keys())
    assert missing == set(), f"fallback 누락: {missing}"


def test_lig_ticker_name_uses_renamed():
    """079550 종목명은 사명변경 후 'LIG디펜스앤에어로스페이스' (옛 LIG넥스원이 아니어야 함)."""
    assert TICKER_NAMES_FALLBACK["079550"] == "LIG디펜스앤에어로스페이스"
