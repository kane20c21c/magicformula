# -*- coding: utf-8 -*-
"""
features_v3.py — 데이포트 그림자 v3 = 2.0 후보 BG1MD (케인 2026-11-17, 볼트 MagicFormula 데이포트 2.0 피처 사다리)

구성 97개 = 정리 BASE 41 + 그 대섹터 안 순위(G1) 41 + 시장 10 + 요일 + 종목 드라이브 4
  정리 BASE 41 = 운영 BASE 48 − {같은 값 중복: vol_ratio·energy_index / 거의 같은 값: smart_pressure·retail_pressure·hope_vector
                  / 미래 정보: vol_idx·wy_label·wy_phase / 기준점 문제: close_idx / 시장 단: dow} + vol_idx_c(인과) + YZ_20·YZ_60(LLV)
  시장 10 = ⑦·⑤ 판정 · 브레드스 값 · C2 · 관문 창 (MF market_layer.judge) · 직전 18밤 유니버스 갭 변동성 · 코스피200 드라이브 4
  종목 드라이브 4 = 종가 기준 같은 부호 연속일의 길이·크기·속도·갭 몫
정의 정본 = 연구 스크립트 SP reports/dayport2/e1_feature_ladder.py (BG1MD) — 그것과 같은 계산을 운영 경로에 옮긴 것이다.

운영 build_features 산출물(features.parquet·cols.json)은 건드리지 않는다 — 신호 때 메모리에서 붙인다(run_daily --variant v3).
⚠ 시장 판정은 신호 시점(19:10)의 자료로 계산한다 — 그날 브레드스는 한국투자 잠정치다. 연구(e1)는 KRX 확정치로 쟀다.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

DROP_A = ["vol_ratio", "energy_index", "smart_pressure", "retail_pressure", "hope_vector",
          "vol_idx", "wy_label", "wy_phase", "close_idx", "dow"]
YZ = ["YZ_20", "YZ_60"]
MCOL = ["m_p7", "m_p5", "m_breadth", "m_c2", "m_gate", "m_rv18", "m_drv_len", "m_drv_size", "m_drv_speed", "m_drv_gap"]
SCOL = ["sdrv_len", "sdrv_size", "sdrv_speed", "sdrv_gap"]
_MF_ROOT = Path(__file__).resolve().parents[2]          # MagicFormula/


def base41(cols_all: dict) -> list:
    return [c for c in cols_all["BASE"] if c not in DROP_A] + ["vol_idx_c"] + YZ


def v3_cols(cols_all: dict) -> list:
    """BG1MD 97 — 순서 고정 (연구 e1 과 같다)."""
    b = base41(cols_all)
    return b + ["s1_" + c for c in b] + MCOL + ["dow"] + SCOL


def drive_cols(df: pd.DataFrame, key: str, o: str = "Open", c: str = "Close") -> pd.DataFrame:
    """종가 기준 같은 부호 연속일(드라이브)의 길이(부호 포함)·크기(누적 수익)·속도(오늘 |수익| ÷ 첫날 |수익|)·갭 몫(갭 로그합 ÷ 수익 로그합)."""
    g = df.groupby(key, sort=False)
    pc = g[c].shift(1); r = df[c] / pc - 1; s = np.sign(r).fillna(0)
    brk = (s != s.groupby(df[key]).shift(1)) | (df[key] != df[key].shift(1)) | (s == 0)
    rid = brk.cumsum(); run = (s.groupby(rid).cumcount() + 1).where(s != 0, 0)
    lr = np.log1p(r).where(s != 0); lg = np.log(df[o] / pc).where(s != 0)
    lsum = lr.groupby(rid).cumsum(); gsum = lg.groupby(rid).cumsum(); first = r.groupby(rid).transform("first")
    return pd.DataFrame({"len": run * s, "size": np.expm1(lsum).where(s != 0, 0.0),
                         "speed": (r.abs() / first.abs()).where(run >= 2),
                         "gap": (gsum / lsum).where((run >= 1) & (lsum.abs() > 1e-4)).clip(-3, 3)}, index=df.index)


def _llv_px(cols: list) -> pd.DataFrame:
    """LLV core+extend 뷰 순수 읽기 (build_panel 과 같은 원천)."""
    px = pd.concat([pd.read_parquet(p, columns=cols) for p in (cfg.CORE, cfg.EXTEND)], ignore_index=True)
    px["Date"] = pd.to_datetime(px.Date); px["Ticker"] = px.Ticker.astype(str).str.zfill(6)
    return px.drop_duplicates(["Date", "Ticker"], keep="last")


def market_frame(as_of) -> pd.DataFrame:
    """날짜 → 시장 10 (연구 e1.add_market_drive 와 같은 정의)."""
    if str(_MF_ROOT) not in sys.path:
        sys.path.insert(0, str(_MF_ROOT))
    from market_layer import judge
    from market_layer.inputs import load_inputs
    inp = load_inputs(str(pd.Timestamp(as_of).date()))
    J = judge(inp.k_open, inp.k_close, inp.tmi, inp.ratio)
    bv_start = J.breadth_value.dropna().index.min()
    mk = pd.DataFrame(index=J.index)
    mk["m_p7"] = J.p7.astype(float); mk["m_p5"] = J.p5.astype(float); mk["m_breadth"] = J.breadth_value
    mk["m_c2"] = J.c2.astype(float); mk["m_gate"] = J.gate_window.astype(float)
    if pd.notna(bv_start):
        mk.loc[mk.index < bv_start, ["m_p7", "m_p5", "m_breadth"]] = np.nan
    PP = _llv_px(["Date", "Ticker", "Name", "Open", "Close"])
    nm = PP.groupby("Ticker").Name.last().fillna("")
    ex = {"102110", "252650"} | set(nm[nm.astype(str).str.endswith(("우", "우B", "우C"))].index)
    PP = PP[~PP.Ticker.isin(ex)]
    O = PP.pivot(index="Date", columns="Ticker", values="Open"); C = PP.pivot(index="Date", columns="Ticker", values="Close")
    O, C = O.where(O > 0), C.where(C > 0)
    mkt = (O.shift(-1) / C - 1).mean(axis=1)
    mk["m_rv18"] = mkt.rolling(18, min_periods=18).std().shift(1).reindex(mk.index)
    k = pd.read_parquet(os.path.join(cfg.LLV_PATH, "data", "ohlcv", "tickers", "KOSPI200.parquet"), columns=["Date", "Open", "Close"])
    k["Date"] = pd.to_datetime(k.Date); k = k.sort_values("Date").reset_index(drop=True); k["key"] = "K"
    kd = drive_cols(k, "key"); kd.index = k.Date; kd.columns = ["m_drv_" + c for c in kd.columns]
    mk = mk.join(kd)
    print(f"[v3] 시장 판정 {inp.source} · 마지막 {J.index.max().date()}")
    return mk


def add_v3(d: pd.DataFrame, as_of=None, market: pd.DataFrame | None = None, yz: pd.DataFrame | None = None) -> pd.DataFrame:
    """features 프레임(d)에 v3 전용 컬럼을 붙여 돌려준다 (원본 순서 유지). market·yz 를 주면 그걸 쓴다(테스트용)."""
    d = d.copy(); d["Date"] = pd.to_datetime(d.Date); d["Ticker"] = d.Ticker.astype(str).str.zfill(6)
    order = d.index
    if yz is None:
        yz = _llv_px(["Date", "Ticker"] + YZ)
    n0 = len(d)
    d = d.drop(columns=[c for c in YZ if c in d.columns]).merge(yz[["Date", "Ticker"] + YZ], on=["Date", "Ticker"], how="left")
    assert len(d) == n0, "YZ 병합 후 행 수가 달라졌다"
    d.index = order
    d[YZ] = d[YZ].replace([np.inf, -np.inf], np.nan)
    n_sec = d.groupby(["Date", "sector_top"]).Ticker.transform("size")
    for c in YZ:
        d["s1_" + c] = d.groupby(["Date", "sector_top"])[c].rank(pct=True).where(n_sec >= cfg.MIN_SECTOR_N)
    srt = d.sort_values(["Ticker", "Date"])
    sd = drive_cols(srt[["Ticker", "Date", "Open", "Close"]], "Ticker")
    for c in sd.columns:
        d["sdrv_" + c] = sd[c].reindex(d.index)
    mk = market if market is not None else market_frame(as_of if as_of is not None else d.Date.max())
    m = mk.reindex(d.Date.values)
    for c in MCOL:
        d[c] = m[c].to_numpy() if c in m.columns else np.nan
    return d
