#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
holdtiming_validation.py
========================
홀드형 vs 타이밍형 종목에서 기술적 지표/와이코프 신호의 "의미"가
다르게(지속 vs 반전) 작동하는지 검증하는 재사용 하네스.

핵심 엔진 — "조건부 후행수익 분기 검정" (run_event_test)
-------------------------------------------------------
어떤 조건 C(예: RSI>70)가 참인 날을 종목별로 모아 N일 후행수익을 측정하고,
홀드형 그룹 vs 타이밍형 그룹으로 나눠 후행수익이 갈리는지(한쪽 +지속 / 한쪽 −반전)
검정한다. RSI·BB·MACD·갭·거래량·심리·와이코프 신호가 전부 "조건 C만 바꾸면 되는"
동일한 틀이다.

항목 정의는 ITEMS 레지스트리에 있고, 명세는 docs/holdtiming/검증항목명세.md.
한 항목씩 해결하기 위한 구조 — 한 항목 = ITEMS 의 한 엔트리.

실행
----
  PYTHONPATH=<pylibs> python3 scripts/holdtiming_validation.py --item V1
  PYTHONPATH=<pylibs> python3 scripts/holdtiming_validation.py --all
  PYTHONPATH=<pylibs> python3 scripts/holdtiming_validation.py --list

검증은 아직 실행 단계가 아니라 프레임워크/명세 확정 단계 — main 은 준비돼 있으나
실제 결과 해석/리포트는 항목별로 Kane 승인 후 진행한다.
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 경로 — 스크립트 위치 기준 상대 해석 (세션 마운트명 변경에 무관)
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve()
REPO = HERE.parent.parent                       # .../Magic Formula
STOLAB = REPO.parent                            # .../StoLab
VAULT = STOLAB / "longlivevault" / "data" / "ohlcv"
CORE = VAULT / "core.parquet"
EXTEND = VAULT / "extend.parquet"
KOSPI200 = VAULT / "tickers" / "KOSPI200.parquet"
GROUPS = REPO / "docs" / "holdtiming" / "stock_groups.csv"
OUT = REPO / "docs" / "holdtiming" / "results"
OUT.mkdir(parents=True, exist_ok=True)

HORIZONS_DEFAULT = [5, 10, 20]                   # 후행수익 구간(거래일)


# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------
def load_panel() -> dict[str, pd.DataFrame]:
    """core+extend → {ticker: df(Date 인덱스, 지표 포함)}."""
    panel = pd.concat([pd.read_parquet(CORE), pd.read_parquet(EXTEND)], ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"])
    panel["Ticker"] = panel["Ticker"].astype(str).str.zfill(6)
    out = {}
    for t, g in panel.groupby("Ticker"):
        g = g.sort_values("Date").copy()
        g.index = pd.DatetimeIndex(g["Date"])
        out[t] = g
    return out


def load_groups(scheme: str = "binary") -> dict[str, str]:
    """
    종목 → 그룹 라벨.
    scheme='binary' : 종목구분 컬럼(홀드/타이밍) — 혼합·저신호 포함 2분(중앙값 기준)
    scheme='type'   : Type 컬럼(홀드형/타이밍형/혼합형/저신호) 원본 4분류
    scheme='cell'   : 구획(섹터-종목 4셀)
    scheme='trend3' : 유형3(추세형-우상향/추세형-우하향/타이밍형) — 방향중립 재분류(2026-06-13)
    scheme='trend'  : 추세성(추세형/타이밍형) — R²≥0.7, 방향 무관
    scheme='dir'    : 방향(우상향/우하향) — 기울기 부호
    """
    df = pd.read_csv(GROUPS, dtype={"Ticker": str})
    df["Ticker"] = df["Ticker"].str.zfill(6)
    col = {"binary": "종목구분", "type": "Type", "cell": "구획",
           "trend3": "유형3", "trend": "추세성", "dir": "방향",
           "cell4": "셀4"}[scheme]
    df = df.dropna(subset=[col])   # 제외 종목(예: 우하향 추세형) 자동 제외
    return dict(zip(df["Ticker"], df[col]))


def fwd_return(df: pd.DataFrame, horizon: int, mode: str = "raw",
               kospi: pd.DataFrame | None = None) -> pd.Series:
    """Close(t+N)/Close(t)-1. mode='alpha' 면 KOSPI200 대비 초과수익."""
    r = df["Close"].shift(-horizon) / df["Close"] - 1.0
    if mode == "alpha" and kospi is not None:
        kr = kospi["Close"].reindex(df.index).shift(-horizon) / kospi["Close"].reindex(df.index) - 1.0
        r = r - kr
    return r


# ---------------------------------------------------------------------------
# 조건 함수 (항목별) — 각 함수는 df → bool Series (이벤트 발생일 = True)
# ---------------------------------------------------------------------------
def _cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))

def _cross_dn(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))

def c_rsi_overbought(df, th=70): return df["RSI"] > th
def c_rsi_oversold(df, th=30):  return df["RSI"] < th
def c_bb_break_up(df):          return df["Close"] > df["BB_upper"]
def c_bb_break_dn(df):          return df["Close"] < df["BB_lower"]
def c_macd_golden(df):          return _cross_up(df["MACD"], df["MACD_Signal"])
def c_macd_dead(df):            return _cross_dn(df["MACD"], df["MACD_Signal"])
def c_gap_up(df, th=0.03):      return df["Gap_Ratio"] > th   # Gap_Ratio 는 분수(0.03=3%)
def c_vol_breakout(df, rv=2.0): return (df["Rel_Volume"] > rv) & (df["Close"] > df["MA20"])
def c_anxiety_high(df, z=1.5):  return df["Anxiety_Index_Z"] > z
def c_hope_high(df, z=1.5):     return df["Hope_Vector_Z"] > z
def c_supply_strong(df, th=0.2):return df["Supply_Score"] > th
def c_wy_dist(df):              return df["Wyckoff_Signal"] == "DIST_CONFIRM"
def c_wy_acc(df):               return df["Wyckoff_Signal"].isin(["ACC_COMPLETE", "PANIC_ZONE"])
def c_phase_up2dn(df):          return (df["Wyckoff_Phase"] == "Downtrend") & (df["Wyckoff_Phase"].shift(1) == "Uptrend")

def _enter(series: pd.Series, target) -> pd.Series:
    """series 가 target(상태/집합)에 '진입'한 날 = True (전일은 아님)."""
    tgt = target if isinstance(target, (set, list, tuple)) else {target}
    cur = series.isin(tgt); prev = series.shift(1).isin(tgt)
    return (cur & ~prev).fillna(False)

# M1 — 국면 신호 속도×그룹 정합 (Phase 느림 vs Label 민감)
def c_phase_down(df):  return _enter(df["Wyckoff_Phase"], "Downtrend")              # Phase 하락 진입(느림)
def c_label_down(df):  return _enter(df["Wyckoff_Label"], "Markdown")              # Label 하락 진입(민감)
def c_label_down2(df): return _enter(df["Wyckoff_Label"], {"Markdown", "Distribution"})  # 변형: 분배 포함

# M2 — 국면 '상태' 매수 유효성 (전환 이벤트가 아니라 상태 보유)
def c_markup_state(df):  return df["Wyckoff_Label"] == "Markup"      # Label 민감 상승상태
def c_uptrend_state(df): return df["Wyckoff_Phase"] == "Uptrend"     # Phase 느린 상승상태


# ---------------------------------------------------------------------------
# 항목 레지스트리 — 명세와 1:1. expect = 가설(그룹별 예상 부호)
#   sig: '+' 후행수익 플러스(지속/상승) 예상, '-' 마이너스(반전/하락) 예상, '?' 미정
# ---------------------------------------------------------------------------
ITEMS = {
    # Phase 1 — 지표 의미 검증
    "V1":  dict(name="RSI 과매수(>70)",      cond=c_rsi_overbought, expect=dict(홀드="+", 타이밍="-"),
                note="홀드=과매수 지속(추세), 타이밍=되돌림"),
    "V2":  dict(name="RSI 과매도(<30)",      cond=c_rsi_oversold,   expect=dict(홀드="+", 타이밍="+"),
                note="타이밍이 반등 폭 더 클 것"),
    "V3":  dict(name="BB 상단 돌파",         cond=c_bb_break_up,    expect=dict(홀드="+", 타이밍="-"),
                note="홀드=추세강화 지속, 타이밍=fade"),
    "V4":  dict(name="BB 하단 이탈",         cond=c_bb_break_dn,    expect=dict(홀드="-", 타이밍="+"),
                note="타이밍=반등, 홀드=추세훼손 경고"),
    "V5":  dict(name="MACD 골든크로스",      cond=c_macd_golden,    expect=dict(홀드="+", 타이밍="?"),
                note="신뢰도 그룹차 확인"),
    "V6":  dict(name="상승갭(Gap_Ratio>3%)", cond=c_gap_up,         expect=dict(홀드="+", 타이밍="-"),
                note="홀드=지속, 타이밍=갭 소멸"),
    "V7":  dict(name="거래량 동반 돌파",      cond=c_vol_breakout,   expect=dict(홀드="+", 타이밍="?"),
                note="거래량 확인의 효과 그룹차"),
    "V8":  dict(name="과열 심리(Anxiety_Z>1.5)", cond=c_anxiety_high, expect=dict(홀드="?", 타이밍="-"),
                note="타이밍=고점 매도신호"),
    "V9":  dict(name="기대 과열(Hope_Z>1.5)",    cond=c_hope_high,    expect=dict(홀드="+", 타이밍="-"),
                note="홀드=무시 가능, 타이밍=분할익절"),
    "V10": dict(name="수급 강세(Supply>0.2)",    cond=c_supply_strong, expect=dict(홀드="+", 타이밍="+"),
                note="타이밍 진입 타이밍 핵심 여부 (core 한정—수급 컬럼)"),
    "V11": dict(name="와이코프 DIST_CONFIRM",    cond=c_wy_dist,      expect=dict(홀드="-", 타이밍="?"),
                note="홀드셀에서 청산 유효 vs 전체역효과 재검토(핵심)"),
    "V12": dict(name="와이코프 ACC/PANIC",       cond=c_wy_acc,       expect=dict(홀드="+", 타이밍="+"),
                note="매수신호 효과 그룹차"),
    "V13": dict(name="국면전환 Up→Down",         cond=c_phase_up2dn,  expect=dict(홀드="-", 타이밍="?"),
                note="추세종료 후행수익 그룹차"),
}

# Phase 2/3 은 별도 메서드 필요(엔진 외) — 명세에 기재, 코드는 후속:
#   C1.. RSI 등 임계 캘리브레이션(종목별 백분위 최적화)
#   E1.. 손절/보유기간/익절/사이징 미니 백테스트


# ---------------------------------------------------------------------------
# 검정 엔진
# ---------------------------------------------------------------------------
def run_event_test(panel, groups, cond_fn, horizon, mode="raw", kospi=None,
                   min_events=30) -> dict:
    """조건 C 이벤트 후 horizon일 후행수익을 그룹별로 비교."""
    rows = []
    for t, df in panel.items():
        grp = groups.get(t)
        if grp is None:
            continue
        try:
            mask = cond_fn(df).fillna(False)
        except KeyError:
            continue  # 해당 종목에 컬럼 없음(예: 수급=extend 결측)
        if mask.sum() == 0:
            continue
        fr = fwd_return(df, horizon, mode, kospi)
        ev = pd.DataFrame({"fwd": fr[mask]}).dropna()
        ev["group"] = grp
        rows.append(ev)
    if not rows:
        return {"error": "no events"}
    ev = pd.concat(rows, ignore_index=True)
    res = {"horizon": horizon, "mode": mode, "n_total": int(len(ev)), "by_group": {}}
    groups_present = [g for g in ev["group"].unique()]
    for g in groups_present:
        x = ev[ev.group == g]["fwd"]
        if len(x) < min_events:
            res["by_group"][g] = {"n": int(len(x)), "note": "events<min"}
            continue
        res["by_group"][g] = {
            "n": int(len(x)),
            "mean_%": float(x.mean() * 100),
            "median_%": float(x.median() * 100),
            "hit_rate": float((x > 0).mean()),
        }
    # 홀드 vs 타이밍 분기 검정 (binary scheme 가정)
    if {"홀드", "타이밍"}.issubset(set(groups_present)):
        a = ev[ev.group == "홀드"]["fwd"]; b = ev[ev.group == "타이밍"]["fwd"]
        if len(a) >= min_events and len(b) >= min_events:
            u = st.mannwhitneyu(a, b, alternative="two-sided")
            res["holdvstiming"] = {
                "median_diff_%": float((a.median() - b.median()) * 100),
                "mw_pvalue": float(u.pvalue),
                "diverge_sign": bool(np.sign(a.median()) != np.sign(b.median())),
            }
    return res


def run_m1(panel=None, groups=None, kospi=None, horizons=HORIZONS_DEFAULT, mode="raw") -> dict:
    """
    M1 — 국면 신호 속도×그룹 정합 검증.
    가설 H1(홀드↔Phase): 홀드형에선 Phase 하락신호가 Label보다 청산 품질↑(Label 휩쏘).
    가설 H2(타이밍↔Label): 타이밍형에선 Label 하락신호가 Phase보다 적시·정합↑.

    지표(그룹×신호별): n, 발화빈도(events/obs), 신호후 후행수익 중앙값,
                      휩쏘율 P(fwd>0|신호), 청산엣지=그룹baseline중앙값−신호중앙값(+면 약세예고).
    강세장 기저는 그룹 baseline 으로 보정 → '상대 정합'으로 해석.
    """
    panel = panel if panel is not None else load_panel()
    groups = groups if groups is not None else load_groups("binary")
    signals = {"Phase_down": c_phase_down, "Label_down": c_label_down}
    out = {"item": "M1", "name": "국면 신호 속도×그룹 정합 (Phase vs Label)", "by_horizon": {}}
    for h in horizons:
        # 그룹별 baseline (무조건 후행수익) + 총 관측수
        base, nobs, fwd_cache = {}, {}, {}
        for t, df in panel.items():
            g = groups.get(t)
            if g is None: continue
            fr = fwd_return(df, h, mode, kospi).dropna()
            fwd_cache[t] = fr
            base.setdefault(g, []).append(fr)
            nobs[g] = nobs.get(g, 0) + len(fr)
        base_med = {g: float(pd.concat(v).median() * 100) for g, v in base.items()}
        hres = {"baseline_median_%": base_med, "n_obs": nobs, "by_signal": {}}
        for signame, cond in signals.items():
            per_grp = {}
            for t, df in panel.items():
                g = groups.get(t)
                if g is None: continue
                try: mask = cond(df)
                except KeyError: continue
                if mask.sum() == 0: continue
                fr = fwd_cache[t].reindex(df.index[mask]).dropna()
                if len(fr): per_grp.setdefault(g, []).append(fr)
            sig = {}
            for g, lst in per_grp.items():
                x = pd.concat(lst)
                sig[g] = {
                    "n_events": int(len(x)),
                    "freq_%": float(len(x) / nobs[g] * 100) if nobs.get(g) else None,
                    "median_%": float(x.median() * 100),
                    "whipsaw_up_rate": float((x > 0).mean()),
                    "exit_edge_%": float(base_med[g] - x.median() * 100),
                }
            hres["by_signal"][signame] = sig
        out["by_horizon"][h] = hres
    return out


def run_item(item_id, panel=None, groups=None, kospi=None,
             horizons=HORIZONS_DEFAULT, scheme="binary", mode="raw") -> dict:
    if item_id not in ITEMS:
        raise KeyError(f"unknown item {item_id}")
    spec = ITEMS[item_id]
    panel = panel if panel is not None else load_panel()
    groups = groups if groups is not None else load_groups(scheme)
    out = {"item": item_id, "name": spec["name"], "expect": spec["expect"],
           "note": spec["note"], "by_horizon": {}}
    for h in horizons:
        out["by_horizon"][h] = run_event_test(panel, groups, spec["cond"], h, mode, kospi)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", help="단일 항목 ID (예: V1)")
    ap.add_argument("--all", action="store_true", help="Phase1 전체")
    ap.add_argument("--list", action="store_true", help="항목 목록")
    ap.add_argument("--scheme", default="binary",
                    choices=["binary", "type", "cell", "trend3", "trend", "dir", "cell4"])
    ap.add_argument("--mode", default="raw", choices=["raw", "alpha"])
    args = ap.parse_args()

    if args.list:
        for k, v in ITEMS.items():
            print(f"  {k:4} {v['name']:24} 가설 {v['expect']}")
        return
    panel = load_panel()
    groups = load_groups(args.scheme)
    kospi = pd.read_parquet(KOSPI200); kospi.index = pd.DatetimeIndex(pd.to_datetime(kospi["Date"]))
    if args.item == "M1":
        r = run_m1(panel, groups, kospi, mode=args.mode)
        print("\n===== M1 국면 신호 속도×그룹 정합 (Phase vs Label) =====")
        for h, hr in r["by_horizon"].items():
            print(f"  --- h{h} (baseline 중앙값: {hr['baseline_median_%']}) ---")
            for sn, sd in hr["by_signal"].items():
                for g, d in sd.items():
                    print(f"    {sn:11} {g:4} n{d['n_events']:5} freq{d['freq_%']:.1f}% "
                          f"med{d['median_%']:+.2f}% 휩쏘{d['whipsaw_up_rate']:.2f} 청산엣지{d['exit_edge_%']:+.2f}%")
        (OUT / "results_M1.json").write_text(json.dumps(r, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"[saved] {OUT/'results_M1.json'}"); return
    targets = list(ITEMS) if args.all else ([args.item] if args.item else [])
    if not targets:
        print("항목을 지정하세요: --item V1 / --item M1 / --all / --list"); return
    allres = {}
    for it in targets:
        r = run_item(it, panel, groups, kospi, mode=args.mode)
        allres[it] = r
        print(f"\n===== {it} {r['name']}  가설={r['expect']} =====")
        for h, hr in r["by_horizon"].items():
            bg = hr.get("by_group", {})
            hv = hr.get("holdvstiming", {})
            seg = " | ".join(f"{g}:med{d.get('median_%', float('nan')):+.2f}% hit{d.get('hit_rate', float('nan')):.2f}(n{d.get('n')})"
                             for g, d in bg.items())
            div = f"  Δmed={hv.get('median_diff_%', float('nan')):+.2f}% MWp={hv.get('mw_pvalue', float('nan')):.1e} 분기={hv.get('diverge_sign')}" if hv else ""
            print(f"  h{h:>2} {seg}{div}")
    (OUT / "results.json").write_text(json.dumps(allres, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n[saved] {OUT/'results.json'}")


if __name__ == "__main__":
    main()
