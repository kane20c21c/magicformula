#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_classification.py
==========================
4셀(섹터×종목) 분류 + 셀별 신호해석을 생성하는 정본 생성기.

소유: Magic Formula (판단 근거 프로젝트).
  - 로직·임계치는 configs/classification.yaml 에 정의된다.
  - 원자료(OHLCV)는 LLV 가 공급 (longlivevault core/extend parquet).
  - 결과 JSON 을 LLV data/ 에 써서 LLV 가 받아 서빙하게 한다 (LLV 는 판단 안 함).

방법 (configs/classification.yaml):
  - window_start(2025-04-30, 한국시장 레짐 변화 시작점)~현재 구간의
    log(종가) 선형회귀 R²·기울기.
  - 종목성격: R²≥0.70 & slope>0 → 추세형-우상향 / R²≥0.70 & slope≤0 → 추세형-우하향(셀 제외)
              / R²<0.70 → 타이밍형
  - 섹터성격: 섹터 내 종목 R² 중앙값 ≥0.70 → 추세섹터 / 미만 → 타이밍섹터
  - 셀4 = {섹터성격}-{종목성격}
  - 추세타이밍_정밀: R²<0.55 & 추세섹터 (스윙 전용·검증됨)

실행:
  PYTHONPATH=<pylibs> python3 scripts/generate_classification.py
  옵션: --check (기존 stock_groups.csv 와 대조만, 파일 안 씀)
"""
from __future__ import annotations
import argparse, json, sys, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve()
REPO = HERE.parent.parent                       # .../Magic Formula
STOLAB = REPO.parent                            # .../StoLab
VAULT = STOLAB / "longlivevault" / "data" / "ohlcv"
CORE = VAULT / "core.parquet"
EXTEND = VAULT / "extend.parquet"
CONFIG = REPO / "configs" / "classification.yaml"
# 결과는 LLV data/ 에 쓴다 (LLV 가 받아 서빙) + Magic Formula 사본
LLV_OUT = STOLAB / "longlivevault" / "data" / "ticker_classification.json"
MF_OUT = REPO / "output" / "classification" / "ticker_classification.json"
GROUPS_CSV = REPO / "docs" / "holdtiming" / "stock_groups.csv"   # 재현 대조용


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _sector_map() -> dict[str, str]:
    """LLV 의 정본 종목→섹터 (TICKER_LIST + EXTEND_LIST)."""
    sys.path.insert(0, str(STOLAB / "longlivevault"))
    from stolab_data import TICKER_LIST, EXTEND_LIST  # noqa: E402
    return {t[0]: t[2] for t in list(TICKER_LIST) + list(EXTEND_LIST)}


def _name_map() -> dict[str, str]:
    sys.path.insert(0, str(STOLAB / "longlivevault"))
    from stolab_data import TICKER_LIST, EXTEND_LIST  # noqa: E402
    return {t[0]: t[1] for t in list(TICKER_LIST) + list(EXTEND_LIST)}


def _fit_r2_slope(close: pd.Series) -> tuple[float, float]:
    """log(종가) 선형회귀 → (R², slope)."""
    y = np.log(close.values.astype(float))
    x = np.arange(len(y))
    if len(y) < 30:
        return (np.nan, np.nan)
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return (float(r2), float(b))


def compute(cfg: dict) -> dict:
    w0 = pd.Timestamp(cfg["window_start"])
    w1 = pd.Timestamp(cfg["window_end"]) if cfg.get("window_end") else None
    cut = cfg["cutoffs"]
    rules = cfg["cell_rules"]
    sectors = _sector_map()
    names = _name_map()

    panel = pd.concat([pd.read_parquet(CORE), pd.read_parquet(EXTEND)], ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"])
    panel["Ticker"] = panel["Ticker"].astype(str).str.zfill(6)

    # 1차: 종목별 R²/slope/BH
    recs: dict[str, dict] = {}
    for tkr, g in panel.groupby("Ticker"):
        g = g.sort_values("Date")
        g = g[g["Date"] >= w0]
        if w1 is not None:
            g = g[g["Date"] <= w1]
        if len(g) < 30:
            continue
        c = g["Close"].dropna()
        r2, slope = _fit_r2_slope(c)
        bh = float(c.iloc[-1] / c.iloc[0] - 1.0)
        # 이름: LLV ticker_list(빈 문자열 많음) → parquet Name 컬럼 폴백
        nm = names.get(tkr) or ""
        if not nm and "Name" in g.columns:
            nm_vals = g["Name"].dropna()
            nm = str(nm_vals.iloc[-1]) if len(nm_vals) else ""
        recs[tkr] = dict(
            ticker=tkr, name=nm, sector=sectors.get(tkr),
            r2=round(r2, 6), slope=round(slope, 8), bh=round(bh, 6), n=int(len(g)),
        )

    # 섹터 R² 중앙값
    sec_df = pd.DataFrame(recs).T
    sec_med = sec_df.groupby("sector")["r2"].median().to_dict()

    # 2차: 셀 배정
    for tkr, r in recs.items():
        r2 = r["r2"]; slope = r["slope"]
        smed = float(sec_med.get(r["sector"], np.nan))
        r["sector_med_r2"] = round(smed, 6)
        # 종목성격
        if r2 >= cut["stock_trend_r2"] and slope > 0:
            stock_type = "추세형-우상향"
        elif r2 >= cut["stock_trend_r2"] and slope <= 0:
            stock_type = "추세형-우하향"          # 셀 제외
        else:
            stock_type = "타이밍형"
        # 섹터성격
        sector_type = "추세" if smed >= cut["sector_trend_medr2"] else "타이밍"
        # 셀4 (우하향은 제외 → null)
        if stock_type == "추세형-우하향":
            cell4 = None
        else:
            stock_short = "추세" if stock_type == "추세형-우상향" else "타이밍"
            cell4 = f"{sector_type}-{stock_short}"
        precision = bool(r2 < cut["precision_timing_r2"] and sector_type == "추세")

        r["stock_type"] = stock_type
        r["sector_type"] = sector_type
        r["cell4"] = cell4
        r["precision_timing"] = precision
        r["signal_rules"] = rules.get(cell4) if cell4 else None

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generated_by": "Magic Formula scripts/generate_classification.py",
        "owner": "Magic Formula (판단 근거). LLV 는 받아 서빙만.",
        "method": cfg["method"],
        "window": {"start": cfg["window_start"], "end": cfg.get("window_end")},
        "cutoffs": cut,
        "caveat": cfg.get("caveat"),
        "classifications": recs,
    }


def save(result: dict) -> None:
    for path in (LLV_OUT, MF_OUT):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ saved {path}  ({len(result['classifications'])} tickers)")


def check_against_groups(result: dict) -> None:
    """기존 stock_groups.csv 와 R²·셀4·정밀 대조 (재현 검증)."""
    if not GROUPS_CSV.exists():
        print("⚠ stock_groups.csv 없음 — 대조 생략"); return
    sg = pd.read_csv(GROUPS_CSV, dtype={"Ticker": str})
    sg["Ticker"] = sg["Ticker"].str.zfill(6)
    cl = result["classifications"]
    n = r2d = celld = precd = 0
    for _, row in sg.iterrows():
        t = row["Ticker"]
        if t not in cl:
            print(f"  누락 {t} {row['Name']}"); continue
        n += 1
        c = cl[t]
        if abs(float(c["r2"]) - float(row["R2"])) > 1e-3:
            r2d += 1; print(f"  R²차이 {t} {row['Name']}: gen {c['r2']:.4f} vs sg {float(row['R2']):.4f}")
        sg_cell = row["셀4"] if pd.notna(row["셀4"]) else None
        if (c["cell4"] or None) != (sg_cell or None):
            celld += 1; print(f"  셀차이 {t} {row['Name']}: gen {c['cell4']} vs sg {sg_cell}")
        sg_prec = bool(row.get("추세타이밍_정밀", False)) if "추세타이밍_정밀" in sg.columns else None
        if sg_prec is not None and bool(c["precision_timing"]) != sg_prec:
            precd += 1; print(f"  정밀차이 {t} {row['Name']}: gen {c['precision_timing']} vs sg {sg_prec}")
    print(f"\n대조 {n}종목 | R²차이 {r2d} | 셀차이 {celld} | 정밀차이 {precd}")
    if r2d == celld == precd == 0:
        print("✅ 완전 재현 — 기존 검증 전부 유효")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="기존 stock_groups.csv 대조만 (파일 안 씀)")
    args = ap.parse_args()
    cfg = load_config()
    result = compute(cfg)
    check_against_groups(result)
    if not args.check:
        save(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
