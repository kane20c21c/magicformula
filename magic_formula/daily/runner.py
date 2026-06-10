"""
magic_formula.daily.runner
==========================
매일 실행되는 데일리 시그널 파이프라인 (v2_combined 단일 경로).

1. 전 종목 full-column OHLCV 로드 (vault → collector — LLV Wyckoff 컬럼 포함)
2. Wyckoff_Label 게이트 시리즈 (LLV 가 채운 값 — hillstorm 직접 호출 없음)
3. make_regimes → breadth + quickregime (전 종목 횡단면)
4. compute_area_scores + combine_scores (robust 가중 + Markdown 게이트)
5. 신호: 종합점수 prev <= threshold < today
6. JSON + MD 저장 + 레짐 사이드카 (StockPortfolio 즉석 v2 가 소비)

설정 출처
---------
``configs/active_strategy.yaml`` (v2_combined) — ``magic_formula.config.load_strategy()``.
가중치 / 임계값 / 게이트 / universe 모두 yaml 에서.

출력 (스키마는 구버전과 동일 유지 — 투자포폴 등 소비자 호환)
----
- ``output/signals/daily_signal_YYYYMMDD.json``
- ``output/signals/daily_signal_YYYYMMDD.md``
- ``output/signals/daily_regimes_YYYYMMDD.json``

변경 이력
---------
2026-06-10 v2 단일화: v1 경로(run/fetch_and_score/scorer 가중평균) 삭제,
MD 렌더링은 daily/report.py 로 분리, 영역 점수 중복 계산 제거.
"""

from __future__ import annotations

import json
import warnings
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from magic_formula._vault import (
    SECTOR_ORDER,
    TICKER_SECTORS as SECTOR_MAP,
    get_ticker_name,
    get_universe,
)
from magic_formula.config import load_strategy
from magic_formula.daily.report import json_default, write_md
from magic_formula.signals.rules import check_breakout_signal

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------

_PKG_DIR    = Path(__file__).resolve().parent.parent   # magic_formula/
_PROJ_ROOT  = _PKG_DIR.parent                          # Magic Formula/
OUTPUT_DIR  = _PROJ_ROOT / "output" / "signals"

LOOKBACK_CALENDAR_DAYS = 500   # 조회 시작 = end_date − 500일 (워밍업 포함)
MIN_ROWS = 80                  # 이보다 짧으면 SKIP


# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------

def _load_ohlcv_full(ticker: str, end_date: str) -> pd.DataFrame | None:
    """collector.fetch_ohlcv 로 vault full-column OHLCV 로드 (Wyckoff 포함)."""
    from magic_formula.data.collector import fetch_ohlcv
    start = (datetime.strptime(end_date, "%Y%m%d")
             - timedelta(days=LOOKBACK_CALENDAR_DAYS)).strftime("%Y%m%d")
    df = fetch_ohlcv(ticker, start, end_date)
    if df is None or df.empty or len(df) < MIN_ROWS:
        return None
    return df


def _phase_series(df: pd.DataFrame) -> pd.Series:
    """LLV 가 채운 Wyckoff_Label 컬럼을 게이트 입력으로 반환.

    없으면 빈 시리즈 → 게이트 무력 (경고). Magic Formula 는 hillstorm 을
    import 하지 않는다 — LLV 책임.
    """
    if "Wyckoff_Label" in df.columns:
        return df["Wyckoff_Label"]
    warnings.warn(
        "[daily.runner] Wyckoff_Label 컬럼 없음 — LLV 백필이 안 됐을 가능성. "
        "게이트가 무력화됩니다.",
        stacklevel=2,
    )
    return pd.Series(index=df.index, dtype=object)


# ---------------------------------------------------------------------------
# 전일 OHLCV 정보
# ---------------------------------------------------------------------------

def get_prev_day_info(df: pd.DataFrame) -> dict:
    """전일(최신일) OHLCV + 등락률 + 거래량 변동률."""
    if len(df) < 2:
        return {}
    today = df.iloc[-1]
    prev = df.iloc[-2]
    close = float(today["Close"])
    pv = int(prev["Volume"]) if "Volume" in df else 0
    vol = int(today["Volume"]) if "Volume" in df else 0
    return {
        "open": float(today["Open"]), "high": float(today["High"]),
        "low": float(today["Low"]), "close": close, "volume": vol,
        "change_pct": round((close - prev["Close"]) / prev["Close"] * 100, 2) if prev["Close"] else 0.0,
        "vol_chg_pct": round((vol - pv) / pv * 100, 2) if pv else 0.0,
    }


# ---------------------------------------------------------------------------
# 메인 파이프라인
# ---------------------------------------------------------------------------

def run(
    target_date: str | None = None,
    config_path: str | None = None,
) -> dict:
    """
    데일리 파이프라인 실행 (v2_combined).

    Parameters
    ----------
    target_date : vault 조회 종료 일자 ('YYYYMMDD'). None 이면 오늘.
    config_path : active_strategy.yaml 경로. None 이면 기본 위치.

    Returns
    -------
    결과 dict (JSON 저장용). 'date' 는 **실제 데이터의 마지막 거래일**
    (전 종목 최빈값) 이며 파일명도 이 날짜를 쓴다. 'executed_at' 은 실행 시점.
    """
    from magic_formula.analysis import area_scores as A

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_strategy(config_path)
    print(f"[daily.runner] 전략 설정: {cfg.summary()}")

    executed_at = datetime.today().strftime("%Y%m%d")
    if target_date is None:
        target_date = executed_at

    tickers = get_universe(cfg.universe)
    if not tickers:   # vault 미설치 안전망
        tickers = list(SECTOR_MAP.keys())
    print(f"[daily.runner] 실행={executed_at} 조회종료={target_date} "
          f"universe={cfg.universe} ({len(tickers)}개) thr={cfg.threshold} "
          f"gate={'ON' if cfg.gate_enabled else 'OFF'}")

    # ── 1. 전 종목 로드 (레짐은 횡단면이라 전체 필요) ──
    stock_data: dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = _load_ohlcv_full(t, target_date)
        if df is not None:
            stock_data[t] = df
    print(f"[daily.runner] 로드 완료: {len(stock_data)}종목")

    if not stock_data:
        print("[daily.runner] ⚠ 처리된 종목 0개 — 종료")
        return {}

    # ── 2. Wyckoff 국면 (LLV 가 채운 컬럼) ──
    phases = {t: _phase_series(df) for t, df in stock_data.items()}

    # ── 3. 레짐 (횡단면) ──
    regime_b, regime_q = A.make_regimes(stock_data)

    # ── 4~5. 영역 점수(1회) + 결합 + 신호 ──
    all_results: list[dict] = []
    signals: list[dict] = []
    last_dates: list = []

    for t, df in stock_data.items():
        areas = A.compute_area_scores(df, regime_b, regime_q)
        comp = A.combine_scores(
            areas, cfg.weights, phases[t],
            gate=cfg.gate_enabled, exclude_phases=cfg.gate_exclude_phases,
        )
        if comp.dropna().empty or len(comp) < 2:
            continue
        last_dates.append(df.index[-1])

        cur = comp.iloc[-1]
        prv = comp.iloc[-2]
        cur_v = float(cur) if pd.notna(cur) else None     # 게이트 제외(NaN)면 None
        prv_v = float(prv) if pd.notna(prv) else None
        is_signal = check_breakout_signal(
            pd.DataFrame({"composite_score": comp}), cfg.threshold
        )

        phase_now = phases[t].iloc[-1] if len(phases[t]) else ""

        # Wyckoff 전환 신호 + 강도 (LLV 가 채운 컬럼 — 평소 None)
        wy_sig = (df["Wyckoff_Signal"].iloc[-1]
                  if "Wyckoff_Signal" in df.columns else None)
        wy_str = (df["Wyckoff_Signal_Strength"].iloc[-1]
                  if "Wyckoff_Signal_Strength" in df.columns else None)
        if pd.isna(wy_sig):
            wy_sig = None
        if pd.isna(wy_str):
            wy_str = None
        else:
            try:
                wy_str = int(wy_str)
            except (ValueError, TypeError):
                wy_str = None

        name = get_ticker_name(t, df.iloc[-1].get("Name") if "Name" in df.columns else None)
        sector = SECTOR_MAP.get(t, "기타")

        row = {
            "ticker": t, "name": name, "sector": sector,
            "composite_score": round(cur_v, 2) if cur_v is not None else None,
            "prev_score": round(prv_v, 2) if prv_v is not None else None,
            "area1_trend": round(float(areas["trend"].iloc[-1]), 2),
            "area2_momentum": round(float(areas["momentum"].iloc[-1]), 2),
            "area3_volume": round(float(areas["volume"].iloc[-1]), 2),
            "area4_volatility": round(float(areas["volatility"].iloc[-1]), 2),
            "area5_wyckoff": 0.0,                 # v2 는 Wyckoff 점수 미사용 (게이트로)
            "wyckoff_phase": str(phase_now) if phase_now and pd.notna(phase_now) else "",
            "wyckoff_signal": str(wy_sig) if wy_sig else "",
            "wyckoff_signal_strength": wy_str,
            "gated_out": bool(cur_v is None),     # Markdown 등으로 제외됨
            "is_signal": is_signal,
            **get_prev_day_info(df),
        }
        all_results.append(row)
        if is_signal:
            signals.append(row)

    # 섹터별 정렬
    def _sector_key(r):
        try:
            return SECTOR_ORDER.index(r["sector"])
        except ValueError:
            return 999
    signals.sort(key=lambda r: (_sector_key(r), r["sector"], r["ticker"]))
    all_results.sort(key=lambda r: (_sector_key(r), r["sector"], r["ticker"]))

    # 데이터 기준일 — 전 종목 마지막 거래일 최빈값 (휴장/정지 종목 견고성)
    data_date = (Counter(last_dates).most_common(1)[0][0].strftime("%Y%m%d")
                 if last_dates else executed_at)

    # ── 레짐 시계열 사이드카 저장 (StockPortfolio 즉석 v2 가 빌려 씀) ──
    _save_regimes(regime_b, regime_q, data_date)

    result = {
        "date": data_date, "executed_at": executed_at,
        "strategy_id": cfg.strategy_id, "last_updated": cfg.last_updated or "",
        "rule": "threshold_breakout",
        "threshold": cfg.threshold,
        "candidate_threshold": cfg.candidate_threshold,   # 후보 모니터링 기준
        "weights": dict(cfg.weights),
        "system_version": "v2_combined", "gate": cfg.gate_enabled,
        "total_tickers": len(all_results), "signal_count": len(signals),
        "signals": signals, "all_scores": all_results,
    }

    json_path = OUTPUT_DIR / f"daily_signal_{data_date}.json"
    md_path = OUTPUT_DIR / f"daily_signal_{data_date}.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8")
    print(f"[daily.runner] JSON 저장: {json_path}")
    write_md(result, md_path)
    print(f"[daily.runner] MD 저장:   {md_path}")
    return result


# 구버전 호환 alias (외부에서 run_combined 이름으로 호출했을 수 있음)
run_combined = run


# ---------------------------------------------------------------------------
# 레짐 사이드카
# ---------------------------------------------------------------------------

def _regime_to_dict(ser: pd.Series) -> dict:
    """레짐 Series(datetime index) → {YYYYMMDD: label} dict. NaN 제외."""
    out: dict[str, str] = {}
    for idx, val in ser.items():
        if pd.isna(val):
            continue
        try:
            key = idx.strftime("%Y%m%d")
        except AttributeError:
            key = str(idx)
        out[key] = str(val)
    return out


def _save_regimes(regime_b: pd.Series, regime_q: pd.Series, data_date: str) -> None:
    """횡단면 레짐 2종을 사이드카 JSON 으로 저장.

    파일: output/signals/daily_regimes_{data_date}.json
    소비자(StockPortfolio 즉석 v2)는 가장 최근 파일을 읽어 코어 외 종목 점수에
    그대로 빌려 쓴다. 레짐은 시장 전체 공통이라 종목 무관.
    """
    path = OUTPUT_DIR / f"daily_regimes_{data_date}.json"
    payload = {
        "date": data_date,
        "breadth": _regime_to_dict(regime_b),   # 추세 영역 (10/10/0.60)
        "quick": _regime_to_dict(regime_q),     # 거래량·변동성 영역 (3/5/0.52)
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"[daily.runner] 레짐 저장: {path}")
