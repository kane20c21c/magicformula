# -*- coding: utf-8 -*-
"""통일장 시장 층 재료 — LLV 에서 읽는다 (LLV 가 데이터 정본, 여기는 읽기만).

  코스피 200 시가·종가 : LLV data/ohlcv/tickers/KOSPI200.parquet   (그날 행 = 19:00·20:30 kis_update, 다음 날 08:01 KRX 가 덮어씀)
  KRX 중형 TMI 종가    : LLV data/ohlcv/tickers/KRX_TMI_MID.parquet (08:01 daily_update — 다음 날 확정)
  시장 브레드스 비율   : LLV data/market_breadth.parquet — KRX 확정치(다음 날) 이력 + 판정일만 한국투자 잠정치

판정일(as_of) 값이 아직 없을 때 (19:10 잠정 판정의 정상 상태)
  브레드스 : KRX 확정 없음 → 한국투자 값 (KIS_PHASES 순서대로 첫 번째 있는 시점) → 그것도 없으면 결측 = 전날 상태 유지
  TMI      : 결측 = 그날 200일선 상태를 전날 그대로 둔다 (judge.ma_state 가 결측일을 '유지' 로 처리)
⚠ **판정일이 아닌 날의 결측은 사고다** — 200일선 창 안에 하루라도 빠지면 그 뒤 199일 동안 이평이 비어
  상태가 조용히 얼어붙는다. load_inputs 가 찾아 InputGapError 로 멈춘다 (배치 실패로 드러나게).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

LLV_ROOT = Path(__file__).resolve().parents[2] / "LongLiveVault"
# 판정일 한국투자 잠정치를 고를 시점 순서 — reg 15:33 · sig 19:00 · fact 20:30.
# 2026-11-10 실측(하루치): 애프터마켓(16:00~20:00)은 이 값을 바꾸지 않았다 — 15:35·16:41·19:06 등락 수·지수·
#   누적 거래량이 모두 같았다. 그래도 정규장 확정 직후인 reg 를 맨 앞에 두어 애프터 영향 가능성을 원천 차단한다.
KIS_PHASES = ("reg", "sig", "fact")
GAP_LOOKBACK = 260            # 결측 검사 창 (거래일) — 200일선 + 여유


class InputGapError(RuntimeError):
    """판정일이 아닌 날에 재료가 빠졌다 — 판정을 멈춘다."""


@dataclass
class Inputs:
    as_of: pd.Timestamp
    k_open: pd.Series
    k_close: pd.Series
    tmi: pd.Series
    ratio: pd.Series
    source: dict = field(default_factory=dict)     # 판정일 재료의 출처·잠정 여부


def _llv(root: Optional[Path]) -> Path:
    root = Path(root) if root else LLV_ROOT
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _index(root: Path, name: str) -> pd.DataFrame:
    d = pd.read_parquet(root / "data" / "ohlcv" / "tickers" / f"{name}.parquet")
    d["Date"] = pd.to_datetime(d["Date"])
    return d.drop_duplicates("Date", keep="last").set_index("Date").sort_index()


def load_inputs(as_of=None, root: Optional[Path] = None, kis_phases: tuple = KIS_PHASES,
                breadth_path: Optional[Path] = None) -> Inputs:
    """as_of(기본 = 코스피 200 마지막 날)까지의 재료를 읽는다. 판정일 값의 출처는 Inputs.source 에 남긴다."""
    root = _llv(root)
    from stolab_data import market_breadth as mb

    K, T = _index(root, "KOSPI200"), _index(root, "KRX_TMI_MID")
    day = pd.Timestamp(as_of) if as_of is not None else K.index.max()
    if day not in K.index:
        raise InputGapError(f"코스피 200 {day:%Y-%m-%d} 행 없음 — kis_update(19:00) 전이거나 휴장일")
    K, T = K.loc[:day], T.loc[:day]
    src: dict = {"k200": "LLV KOSPI200"}

    # 판정일이 아닌 날의 결측 검사 (200일선이 얼어붙는 사고 방지)
    past = K.index[K.index < day][-GAP_LOOKBACK:]
    bad_k = past[K.loc[past, ["Open", "Close"]].isna().any(axis=1).to_numpy()]
    miss_t = past.difference(T.dropna(subset=["Close"]).index)
    if len(bad_k) or len(miss_t):
        raise InputGapError(f"재료 결측 — 코스피 200 {[d.strftime('%Y-%m-%d') for d in bad_k[:5]]} · "
                            f"중형 TMI {[d.strftime('%Y-%m-%d') for d in miss_t[:5]]} (총 {len(bad_k)}·{len(miss_t)}일). "
                            f"LLV backfill_krx_tmi_mid.py --resume 또는 daily_update {{날짜}} 로 메울 것")
    src["tmi"] = "KRX 확정" if day in T.index and pd.notna(T.loc[day, "Close"]) else "그날 없음 — 200일선 상태 전날 유지(잠정)"

    # 브레드스 — KRX 확정 이력 + 판정일만 한국투자
    bdf = mb.load_breadth(Path(breadth_path) if breadth_path else root / "data" / "market_breadth.parquet")
    krx = mb.ratio_series(bdf, mb.SOURCE_KRX, mb.KRX_PHASE).loc[:day]
    ratio, src["breadth"] = krx[krx.index < day], None
    if day in krx.index:
        ratio = krx; src["breadth"] = "KRX 확정"
    else:
        for ph in kis_phases:
            try:
                s = mb.ratio_series(bdf, mb.SOURCE_KIS, ph)
            except ValueError as e:                    # 한국투자 상승·상한 정의 미확정
                src["breadth"] = f"한국투자 사용 불가 ({e})"
                break
            if day in s.index:
                ratio = pd.concat([ratio, s.loc[[day]]]); src["breadth"] = f"한국투자 {ph} (잠정)"
                break
        if src["breadth"] is None or src["breadth"].startswith("한국투자 사용 불가"):
            src["breadth"] = (src["breadth"] or "그날 없음") + " — 브레드스 상태 전날 유지(잠정)"
    gaps = past[-250:].difference(krx.index)
    src["breadth_gaps_250d"] = [d.strftime("%Y-%m-%d") for d in gaps]
    src["provisional"] = not (src["tmi"] == "KRX 확정" and src["breadth"] == "KRX 확정")
    return Inputs(day, K["Open"].astype(float), K["Close"].astype(float), T["Close"].astype(float), ratio, src)
