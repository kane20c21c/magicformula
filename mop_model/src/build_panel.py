# -*- coding: utf-8 -*-
"""
build_panel.py — LLV 원본 parquet → 정제 패널(panel.parquet)

입력: LLV core.parquet, extend.parquet, ticker_classification.json  (전부 LLV 정본)
출력: panel.parquet (196종목 × 거래일, 섹터 2계층, 결측 재구성, 타깃 포함)

핵심 처리:
  1) core+extend 병합, 티커 6자리 zfill
  2) 섹터 부착: classification 의 sector 만 사용
     ⚠ r2/slope/signal_rules/cell4 등은 2025-04 이후 수익률로 역산한 값이라
       look-ahead(미래참조) — 절대 피처로 쓰지 말 것.
     - sector_sub(34개), sector_top = sector_sub.split('.')[0] (20개)
  3) ETF/우선주 제외 (200 → 196)
  4) 시작일 컷 (2023-05-15)
  5) 데이터 품질: OHL<=0 거래정지 마스킹, Close>High 반올림보정
  6) ★ 0을 결측 마커로 취급 (ListShrs 90%/Amount 42%가 0)
     - Amount = Close×Volume 로 재구성 (검증: 오차중앙값 0%)
     - MarketCap = Close×ListShrs 로 재구성 (검증: 오차<1% 100%)
       ListShrs 복원: 원본 → MarketCap/Close 역산 → ffill → bfill
  7) 타깃 생성: Gap_T1(종가→익일시가), y_rel(그날 중앙값 초과), y_abs

⚠ 데이 포트는 오버나이트 매도를 하지 않지만(2% 손절 보유), 모델 타깃은
  Gap_T1 그대로 유지한다 — 학습된 예측력의 정의를 바꾸지 않기 위함.
  체결 규칙 변경은 소비자(StockPortfolio app/paper_day) 소관.
"""
import json, numpy as np, pandas as pd, os
import config as cfg


def build_panel():
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    # 1) 섹터 분류 (ticker/name/sector 만)
    j = json.load(open(cfg.CLASSJ, encoding="utf-8"))
    cls = pd.DataFrame([{"Ticker": v["ticker"], "Name": v["name"], "sector_sub": v["sector"]}
                        for v in j["classifications"].values()])
    cls["sector_top"] = cls.sector_sub.str.split(".").str[0]

    # 2) 주가 병합
    core = pd.read_parquet(cfg.CORE); core["src"] = "core"
    ext  = pd.read_parquet(cfg.EXTEND); ext["src"] = "extend"
    px = pd.concat([core, ext], ignore_index=True)
    px["Ticker"] = px.Ticker.astype(str).str.zfill(6)
    px = px.merge(cls, on="Ticker", how="left", suffixes=("", "_cls"))
    px["Name"] = px.Ticker.map(dict(zip(cls.Ticker, cls.Name)))   # 공식 명칭 사용
    assert px.sector_sub.isna().sum() == 0, "섹터 미부착 종목 존재"

    # 3) ETF/우선주 제외
    drop = px[px.sector_sub.isin(cfg.DROP_SUBSECTORS)].Ticker.unique()
    px = px[~px.Ticker.isin(drop)].copy()

    # 4) 시작일 컷
    px = px[px.Date >= cfg.START_DATE].sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # 5) OHLC 정제
    halt = (px.Open <= 0) | (px.High <= 0) | (px.Low <= 0)
    px["is_halt"] = halt
    px.loc[halt, ["Open", "High", "Low"]] = np.nan
    f = (px.Close > px.High) & px.High.notna(); px.loc[f, "High"] = px.loc[f, "Close"]
    f = (px.Close < px.Low) & px.Low.notna();  px.loc[f, "Low"]  = px.loc[f, "Close"]

    # 6) 0 → 결측, 재구성
    for c in ["ListShrs", "MarketCap", "Amount", "Vol_MA20", "Amount_Idx"]:
        if c in px.columns:
            px.loc[px[c] == 0, c] = np.nan

    ls = px.ListShrs.copy()
    est = np.where(px.MarketCap.notna() & (px.Close > 0), px.MarketCap / px.Close, np.nan)
    ls = ls.fillna(pd.Series(est, index=px.index))
    ls = px.assign(_ls=ls).groupby("Ticker")._ls.ffill()
    ls = px.assign(_ls=ls).groupby("Ticker")._ls.bfill()
    px["ListShrs"] = ls
    px["MarketCap"] = px.MarketCap.fillna(px.ListShrs * px.Close)
    px["Amount"] = px.Amount.fillna(px.Close * px.Volume)

    # 7) 타깃
    g = px.groupby("Ticker")
    px["Open_T1"]  = g.Open.shift(-1)
    px["Close_T1"] = g.Close.shift(-1)
    px["Gap_T1"]   = px.Open_T1 / px.Close - 1      # 종가 → 익일 시가 (모델 타깃 기준)
    px["C2C_T1"]   = px.Close_T1 / px.Close - 1
    px["Ret_1d"]   = g.Close.pct_change()
    med = px.groupby("Date").Gap_T1.transform("median")
    px["y_abs"] = (px.Gap_T1 > 0).astype(float)
    px["y_rel"] = (px.Gap_T1 > med).astype(float)   # ★ 상대타깃 (챔피언)
    px.loc[px.Gap_T1.isna(), ["y_abs", "y_rel"]] = np.nan
    px["mkt"]   = px.groupby("Date").Gap_T1.transform("mean")  # 동일가중 시장 (벤치마크)

    px.to_parquet(cfg.PANEL, index=False)
    print(f"[panel] {len(px):,}행 / {px.Ticker.nunique()}종목 / {px.Date.nunique()}거래일 "
          f"({px.Date.min().date()}~{px.Date.max().date()}) → {cfg.PANEL}")
    return px


if __name__ == "__main__":
    build_panel()
