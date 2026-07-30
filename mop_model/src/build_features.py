# -*- coding: utf-8 -*-
"""
build_features.py — panel.parquet → 피처(features.parquet)

챔피언 피처 = BASE + XS + G1 + G5 = 148개:
  BASE 48  : 종목 자기값(비율 정규화). 원본 절대값은 종목간 비교 불가라 전부 비율화.
  XS   45  : BASE(범주형 dow/wy 제외 45개)의 '그날 전체 종목 중 백분위'
  G1   45  : 같은 45개의 '대섹터 내 백분위' (종목수>=5 섹터만; 계단형은 규제 역할)
  G5   10  : 핵심 10개의 '소섹터 내 백분위'

모든 횡단면(XS/G1/G5)은 날짜 t 안에서만 계산 → 미래참조 없음.

⚠ 이 횡단면 백분위가 LLV 로 이관될 수 없는 이유:
  종목이 하나 추가/제외되면 과거 전 구간의 값이 바뀐다 → LLV parquet 의
  종목단위 upsert 계약(_upsert_and_recompute)과 충돌. 판단 소유는 여기(MF).
"""
import json, numpy as np, pandas as pd, os
import config as cfg


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
    d = pd.concat([d, pd.DataFrame(new_cols, index=d.index)], axis=1)

    d=d.replace([np.inf,-np.inf],np.nan)
    CHAMPION = BASE + XS + G1 + G5
    if save:
        d.to_parquet(cfg.FEATURES, index=False)
        json.dump({"BASE":BASE,"XS":XS,"G1":G1,"G5":G5,"CHAMPION":CHAMPION},
                  open(cfg.COLS_JSON,"w"), ensure_ascii=False, indent=1)
    print(f"[features] {len(d):,}행 | BASE {len(BASE)} + XS {len(XS)} + G1 {len(G1)} + G5 {len(G5)} = {len(CHAMPION)}"
          + (f" → {cfg.FEATURES}" if save else " (인메모리)"))
    return d


if __name__ == "__main__":
    build_features()
