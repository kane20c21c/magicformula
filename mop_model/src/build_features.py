# -*- coding: utf-8 -*-
"""
build_features.py — panel.parquet → 피처(features.parquet)

챔피언 피처 = BASE + XS + G1 + G5 = 148개:
  BASE 48  : 종목 자기값(비율 정규화). 원본 절대값은 종목간 비교 불가라 전부 비율화.
  XS   45  : BASE(범주형 dow/wy 제외 45개)의 '그날 전체 종목 중 백분위'
  G1   45  : 같은 45개의 '대섹터 내 백분위' (종목수>=5 섹터만; 계단형은 규제 역할)
  G5   10  : 핵심 10개의 '소섹터 내 백분위'

모든 횡단면(XS/G1/G5)은 날짜 t 안에서만 계산 → 미래참조 없음.

그림자 v2 (2026-10-29 Kane 승인 — WP9 (가)) = CHAMPION_V2 146개:
  CHAMPION 에서 wy_phase·wy_label 제거(WP9-1: Viterbi 전구간 라벨 = 미래 정보) +
  vol_idx·xs_vol_idx·s1_vol_idx 를 인과 버전 vol_idx_c·xs_vol_idx_c·s1_vol_idx_c 로 치환(WP9-2).
  ★ CHAMPION(운영 148)·BASE/XS/G1/G5 목록은 그대로다 — v2 컬럼은 따로 붙고 cols.json 의
    V2_EXTRA·CHAMPION_V2 키로만 노출된다. 운영 신호는 run_daily --variant v2 를 줄 때만 영향.

⚠ 이 횡단면 백분위가 LLV 로 이관될 수 없는 이유:
  종목이 하나 추가/제외되면 과거 전 구간의 값이 바뀐다 → LLV parquet 의
  종목단위 upsert 계약(_upsert_and_recompute)과 충돌. 판단 소유는 여기(MF).
"""
import json, sys, numpy as np, pandas as pd, os
import config as cfg

# 그림자 v2 (2026-10-29) — CHAMPION 에서 빼는 것 / 붙이는 것
V2_DROP  = ["wy_label", "wy_phase", "vol_idx", "xs_vol_idx", "s1_vol_idx"]
V2_EXTRA = ["vol_idx_c", "xs_vol_idx_c", "s1_vol_idx_c"]
VOL_IDX_C_MIN_PERIODS = 20      # LLV indicator_calculator.CAUSAL_IDX_MIN_PERIODS 와 같게


def _local_vol_idx_c(px):
    """LLV _causal_idx 와 같은 정의: Volume / 그날까지의 확장 중앙값(0 제외, 최소 20일) × 1000."""
    v = pd.to_numeric(px.Volume, errors="coerce")
    med = v.where(v > 0).groupby(px.Ticker).transform(
        lambda s: s.expanding(min_periods=VOL_IDX_C_MIN_PERIODS).median())
    return v / med.where(med > 0) * 1000


def _vol_idx_c(px):
    """LLV Vol_Idx_C 가 있으면 그대로 쓴다(정본). 없으면 패널에서 직접 계산 + 경고.
    ※ 패널은 START_DATE 컷 이후라 직접 계산은 LLV 값(전체 이력 기준)과 초반 구간이 다를 수 있다."""
    if "Vol_Idx_C" in px.columns and px.Vol_Idx_C.notna().any():
        return px.Vol_Idx_C.astype(float)
    print("[features] ⚠ LLV Vol_Idx_C 없음 — 패널에서 직접 계산 (LLV 백필 전?)", file=sys.stderr)
    return _local_vol_idx_c(px)


def _base(px):
    """BASE 48개 (종목 자기값, 비율 정규화)"""
    g = px.groupby("Ticker")
    F = pd.DataFrame(index=px.index)
    # 가격·추세
    F["px_ma5"]=px.Close/px.MA5-1; F["px_ma20"]=px.Close/px.MA20-1; F["px_ma60"]=px.Close/px.MA60-1
    F["ma5_ma20"]=px.MA5/px.MA20-1; F["ma20_ma60"]=px.MA20/px.MA60-1
    bw=(px.BB_upper-px.BB_lower); F["bb_pos"]=(px.Close-px.BB_lower)/bw.replace(0,np.nan); F["bb_width"]=bw/px.Close
    # 거래량
    F["rel_vol"]=px.Rel_Volume; F["vol_idx"]=px.Vol_Idx; F["vol_ratio"]=px.Volume/px.Vol_MA20
    # 모멘텀
    F["rsi"]=px.RSI; F["rsi_ma5"]=px.RSI_MA5; F["rsi_dev"]=px.RSI-px.RSI_MA5
    F["macd"]=px.MACD/px.Close; F["macd_sig"]=px.MACD_Signal/px.Close; F["macd_hist"]=px.MACD_Hist/px.Close
    # 자금흐름/수급
    amt20=g.Amount.transform(lambda s:s.rolling(20,min_periods=5).mean())
    F["ad_chg"]=px.AD_Line_Chg/px.Vol_MA20.replace(0,np.nan); F["chaikin"]=px.Chaikin_Osc/px.Vol_MA20.replace(0,np.nan)
    F["inst"]=px.Inst_Net/amt20; F["forgn"]=px.Foreign_Net/amt20; F["retail"]=px.Retail_Net/amt20
    # 자체지표 (원본 그대로)
    for c in ["Gap_Ratio","Confidence_Index","Energy_Index","Hope_Vector","Hope_Vector_Z",
              "Anxiety_Index","Anxiety_Index_Z","Smart_Pressure","Retail_Pressure","Supply_Score","Close_Idx"]:
        F[c.lower()]=px[c]
    # Weis
    F["weis_dir"]=px.Weis_Dir; F["weis_chg"]=px.Weis_Chg; F["weis_days"]=px.Weis_Days
    F["weis_vol"]=px.Weis_Vol/px.Vol_MA20.replace(0,np.nan)
    # 수익률/변동성
    for n in [1,5,20,60]: F[f"ret{n}"]=g.Close.transform(lambda s,n=n:s.pct_change(n))
    F["vol20"]=g.Ret_1d.transform(lambda s:s.rolling(20,min_periods=10).std())
    F["ret_vol"]=F.ret20/F.vol20.replace(0,np.nan)
    # 규모/기타
    F["log_mc"]=np.log(px.MarketCap.replace(0,np.nan)); F["turnover"]=px.Amount/px.MarketCap.replace(0,np.nan)
    F["log_amt"]=np.log(px.Amount.replace(0,np.nan)); F["dow"]=px.Date.dt.dayofweek
    F["wy_label"]=px.Wyckoff_Label.map({"Accumulation":0,"Markup":1,"Distribution":2,"Markdown":3})
    F["wy_phase"]=px.Wyckoff_Phase.map({"Downtrend":0,"Base":1,"Range":2,"Uptrend":3})
    return F.replace([np.inf,-np.inf],np.nan)


# G5 소섹터 백분위 대상 (핵심 10개)
G5_COLS = ["ret1","ret5","ret20","rsi","px_ma20","inst","forgn","turnover","vol20","bb_pos"]


def build_features(panel_path=None, panel_df=None, save=True):
    """panel_df 를 주면 인메모리 실행 (run_gauge 15:10, 2026-07-30). save=False 면
    features.parquet/cols.json 저장 생략 — 잠정 실행이 운영 산출물을 오염시키지 않게."""
    if panel_df is not None:
        px = panel_df.sort_values(["Ticker","Date"]).reset_index(drop=True)
    else:
        px = pd.read_parquet(panel_path or cfg.PANEL).sort_values(["Ticker","Date"]).reset_index(drop=True)
    F = _base(px)
    BASE = F.columns.tolist()
    d = pd.concat([px[["Date","Ticker","Name","sector_top","sector_sub","Open","High","Low","Close",
                       "Gap_T1","C2C_T1","Ret_1d","y_rel","y_abs","mkt","is_halt"]], F], axis=1)

    CORE = [c for c in BASE if c not in ["dow","wy_label","wy_phase"]]   # 순위 매길 45개
    d["n_sec"] = d.groupby(["Date","sector_top"]).Ticker.transform("size")
    d["n_sub"] = d.groupby(["Date","sector_sub"]).Ticker.transform("size")

    new_cols = {}
    XS=[]
    for c in CORE:
        new_cols["xs_"+c]=d.groupby("Date")[c].rank(pct=True); XS.append("xs_"+c)
    G1=[]
    for c in CORE:
        v=d.groupby(["Date","sector_top"])[c].rank(pct=True)
        new_cols["s1_"+c]=v.where(d.n_sec>=cfg.MIN_SECTOR_N); G1.append("s1_"+c)
    G5=[]
    for c in G5_COLS:
        v=d.groupby(["Date","sector_sub"])[c].rank(pct=True)
        new_cols["s5_"+c]=v.where(d.n_sub>=cfg.MIN_SECTOR_N); G5.append("s5_"+c)
    # 그림자 v2 인과 vol_idx 3종 — BASE/XS/G1 목록 밖에 따로 붙인다 (CHAMPION 불변)
    vic = pd.Series(_vol_idx_c(px).to_numpy(), index=d.index).replace([np.inf,-np.inf],np.nan)
    new_cols["vol_idx_c"] = vic
    new_cols["xs_vol_idx_c"] = vic.groupby(d.Date).rank(pct=True)
    new_cols["s1_vol_idx_c"] = vic.groupby([d.Date, d.sector_top]).rank(pct=True).where(d.n_sec>=cfg.MIN_SECTOR_N)
    d = pd.concat([d, pd.DataFrame(new_cols, index=d.index)], axis=1)

    d=d.replace([np.inf,-np.inf],np.nan)
    CHAMPION = BASE + XS + G1 + G5
    CHAMPION_V2 = [c for c in CHAMPION if c not in V2_DROP] + V2_EXTRA
    if save:
        d.to_parquet(cfg.FEATURES, index=False)
        json.dump({"BASE":BASE,"XS":XS,"G1":G1,"G5":G5,"CHAMPION":CHAMPION,
                   "V2_EXTRA":V2_EXTRA,"CHAMPION_V2":CHAMPION_V2},
                  open(cfg.COLS_JSON,"w"), ensure_ascii=False, indent=1)
    print(f"[features] {len(d):,}행 | BASE {len(BASE)} + XS {len(XS)} + G1 {len(G1)} + G5 {len(G5)} = {len(CHAMPION)}"
          f" | 그림자 v2 {len(CHAMPION_V2)}"
          + (f" → {cfg.FEATURES}" if save else " (인메모리)"))
    return d


if __name__ == "__main__":
    build_features()
