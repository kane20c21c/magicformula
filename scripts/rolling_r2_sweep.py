#!/usr/bin/env python3
"""롤링 R² 창 길이 스윕 — 30 / 60 / 100 거래일 비교 (읽기 전용).

계획서: docs/holdtiming/롤링R2_검증계획_20260907.md

무엇을 하나
===========
현행 4셀 분류는 `window_start(2025-04-30) ~ 최신` 을 **한 번에** 회귀해 R² 하나를
낸다. 창이 계속 늘어나므로 "지금 추세 중인가" 가 아니라 "장기 방향이 우상향인가" 를
재게 된다. 실측(2026-09-07): 추세형 73종목 중 52개가 전체 R²≥0.70 인데 롤링
60일로는 ≥0.70 인 날이 40% 미만이었다 (KB금융 0.909 / 4%).

이 스크립트는 창 길이를 바꿔가며 **시점별 셀 배정**을 만들고 4기준으로 비교한다.
케인 결정(2026-09-07): 종목·섹터 **둘 다 롤링**, 창 시작점은 버리고 롤링만.

⚠ 읽기 전용이다. 어떤 정본도 고치지 않는다.

R² 를 어떻게 빠르게 구하나
=========================
단순선형회귀의 결정계수는 **설명변수와 반응변수의 피어슨 상관 제곱**과 정확히 같다.

    R² = corr(x, y)²      (x = 0,1,2,… 등간격, y = log 종가)

근사가 아니라 항등식이므로 pandas rolling.corr 로 벡터화할 수 있다. polyfit 을
날마다 도는 것보다 수백 배 빠르고 결과는 같다. 기울기 부호는 corr 부호와 같다.

4기준 (계획서 §1-1)
==================
  A 판별력   셀 간 이벤트 후행수익 스프레드 — 클수록 좋다
  B 안정성   셀이 바뀌는 빈도(월평균 종목 수) — 작을수록 좋다 (단 0 이면 무의미)
  C 선행성   셀 전환이 Wyckoff_Phase 전환보다 앞서나 — 앞설수록 좋다
  D 표본건전성 섹터 중앙값이 임계 ±0.05 에 머무는 비율 — 적을수록 좋다

⚠ A 와 B 는 상충한다. 창이 짧으면 민감해져 판별력이 오르지만 셀이 자주 뒤집혀
  매매가 흔들린다. 균형점을 고르는 것이 이 스윕의 목적이다.

⚠ point-in-time 이다 — 각 이벤트일의 셀은 **그날까지의 데이터로만** 정해진다.
  전체 창으로 분류해 과거 이벤트를 평가하면 룩어헤드가 되어 결과가 무효다.

실행:
    python3 -B scripts/rolling_r2_sweep.py [--windows 30,60,100] [--horizon 10]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve()
REPO = HERE.parent.parent
STOLAB = REPO.parent
sys.path.insert(0, str(REPO / "scripts"))

import holdtiming_validation as H  # noqa: E402  (load_panel / fwd_return / cond_fn)

CONFIG = REPO / "configs" / "classification.yaml"
OUT_DIR = REPO / "docs" / "holdtiming" / "results"

# 판별력 측정에 쓸 신호 — 2026-06-13 V1~V11 이 셀 차이를 크게 보인 것들
COND_FNS = {
    "RSI과매수(>70)": H.c_rsi_overbought,
    "RSI과매도(<30)": H.c_rsi_oversold,
    "BB상단돌파":     H.c_bb_break_up,
    "BB하단이탈":     H.c_bb_break_dn,
    "DIST_CONFIRM":  H.c_wy_dist,
    "ACC/PANIC":     H.c_wy_acc,
}


# ---------------------------------------------------------------------------
# 롤링 R² / 셀 배정
# ---------------------------------------------------------------------------
def rolling_r2_matrix(panel: dict[str, pd.DataFrame], win: int
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(날짜 × 종목) 롤링 R² 와 기울기부호 행렬.

    R² = corr(등간격 x, log 종가)². 항등식이라 근사가 아니다.
    """
    r2_cols, sg_cols = {}, {}
    for t, df in panel.items():
        y = np.log(df["Close"].astype(float))
        y = y.replace([np.inf, -np.inf], np.nan)
        if y.notna().sum() < win + 5:
            continue
        x = pd.Series(np.arange(len(y), dtype=float), index=y.index)
        c = x.rolling(win, min_periods=win).corr(y)
        r2_cols[t] = c ** 2
        sg_cols[t] = np.sign(c)
    return pd.DataFrame(r2_cols), pd.DataFrame(sg_cols)


def assign_cells(r2: pd.DataFrame, sign: pd.DataFrame, sector_cls: dict[str, str],
                 cut_stock: float, cut_sector: float
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """시점별 셀4 배정. 종목·섹터 **둘 다 롤링** (케인 2026-09-07).

    Returns: (셀4 DataFrame[날짜×종목], 섹터중앙값 DataFrame[날짜×분류섹터])
    """
    # 섹터별 중앙값 — 날짜마다 그 시점 R² 로 다시 계산한다
    groups: dict[str, list[str]] = {}
    for t in r2.columns:
        groups.setdefault(sector_cls.get(t, "기타"), []).append(t)
    sec_med = pd.DataFrame(
        {s: r2[cols].median(axis=1) for s, cols in groups.items()}
    )

    # 종목이 속한 섹터의 그날 중앙값을 종목 축으로 펼친다
    smed_by_ticker = pd.DataFrame(
        {t: sec_med[sector_cls.get(t, "기타")] for t in r2.columns}
    )

    is_trend_stock = (r2 >= cut_stock) & (sign > 0)
    is_down_trend = (r2 >= cut_stock) & (sign <= 0)      # 셀 제외 대상
    is_trend_sector = smed_by_ticker >= cut_sector

    cell = pd.DataFrame(
        np.where(is_trend_sector, "추세-", "타이밍-"), index=r2.index, columns=r2.columns
    ) + np.where(is_trend_stock, "추세", "타이밍")
    cell = pd.DataFrame(cell, index=r2.index, columns=r2.columns)
    cell = cell.mask(is_down_trend)          # 우하향 추세형 → 제외(NaN)
    cell = cell.mask(r2.isna())              # 워밍업 구간
    return cell, sec_med


# ---------------------------------------------------------------------------
# 4기준
# ---------------------------------------------------------------------------
def criterion_A(panel, cell, horizon, kospi) -> tuple[float, pd.DataFrame]:
    """판별력 — 신호별로 셀 간 후행수익 스프레드(최대−최소). 클수록 좋다."""
    rows = []
    for label, fn in COND_FNS.items():
        buf = []
        for t, df in panel.items():
            if t not in cell.columns:
                continue
            try:
                mask = fn(df).fillna(False)
            except KeyError:
                continue
            if not mask.any():
                continue
            fr = H.fwd_return(df, horizon, "alpha", kospi)
            g = cell[t].reindex(df.index)
            ev = pd.DataFrame({"fwd": fr[mask], "cell": g[mask]}).dropna()
            if len(ev):
                buf.append(ev)
        if not buf:
            continue
        ev = pd.concat(buf, ignore_index=True)
        agg = ev.groupby("cell")["fwd"].agg(["mean", "count"])
        agg = agg[agg["count"] >= 30]
        if len(agg) < 2:
            continue
        rows.append(dict(신호=label, 스프레드=(agg["mean"].max() - agg["mean"].min()) * 100,
                         셀수=len(agg), 이벤트=int(agg["count"].sum())))
    df = pd.DataFrame(rows)
    return (float(df["스프레드"].mean()) if len(df) else np.nan), df


def criterion_B(cell: pd.DataFrame) -> float:
    """안정성 — 셀이 바뀐 (종목·날짜) 건수를 월평균 종목 수로 환산. 작을수록 좋다."""
    changed = (cell != cell.shift()) & cell.notna() & cell.shift().notna()
    n_months = max(1, len(cell) / 21.0)
    return float(changed.sum().sum() / n_months)


def criterion_C(panel, cell) -> float:
    """선행성 — 셀 전환이 Wyckoff_Phase 전환보다 며칠 앞서나 (양수면 앞섬)."""
    leads = []
    for t, df in panel.items():
        if t not in cell.columns or "Wyckoff_Phase" not in df.columns:
            continue
        ph = df["Wyckoff_Phase"]
        ph_ch = ph.index[(ph != ph.shift()) & ph.notna() & ph.shift().notna()]
        c = cell[t].reindex(df.index)
        c_ch = c.index[(c != c.shift()) & c.notna() & c.shift().notna()]
        if len(ph_ch) == 0 or len(c_ch) == 0:
            continue
        for d in ph_ch:                       # 각 국면 전환에 가장 가까운 셀 전환
            diff = (d - c_ch).days
            prior = diff[diff >= 0]
            if len(prior):
                leads.append(prior.min())     # 셀 전환이 며칠 먼저였나
    return float(np.median(leads)) if leads else np.nan


def criterion_D(sec_med: pd.DataFrame, cut: float, band: float = 0.05) -> float:
    """표본 건전성 — 섹터 중앙값이 임계 ±band 에 머무는 비율(%). 적을수록 좋다."""
    near = (sec_med - cut).abs() < band
    return float(near.sum().sum() / near.notna().sum().sum() * 100)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="30,60,100")
    ap.add_argument("--horizon", type=int, default=10)
    args = ap.parse_args()
    wins = [int(w) for w in args.windows.split(",")]

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cut = cfg["cutoffs"]
    agg = {m: g for g, ms in (cfg.get("sector_aggregation") or {}).items() for m in ms}

    sys.path.insert(0, str(STOLAB / "longlivevault"))
    from stolab_data import TICKER_LIST, EXTEND_LIST  # noqa: E402
    raw_sec = {t[0]: t[2] for t in list(TICKER_LIST) + list(EXTEND_LIST)}
    sector_cls = {t: agg.get(s, s) for t, s in raw_sec.items()}

    print("패널 로드…")
    panel = H.load_panel()
    kospi = panel.get("102110")          # BM — alpha 계산용
    print(f"  {len(panel)}종목")

    summary, detail = [], {}
    for w in wins:
        print(f"\n── 창 {w}거래일")
        r2, sign = rolling_r2_matrix(panel, w)
        cell, sec_med = assign_cells(r2, sign, sector_cls,
                                     cut["stock_trend_r2"], cut["sector_trend_medr2"])
        A, A_df = criterion_A(panel, cell, args.horizon, kospi)
        B = criterion_B(cell)
        C = criterion_C(panel, cell)
        D = criterion_D(sec_med, cut["sector_trend_medr2"])
        cover = float(cell.notna().sum().sum() / cell.size * 100)
        summary.append(dict(창=w, A_판별력=A, B_월평균전환=B, C_선행일=C,
                            D_임계근접=D, 커버리지=cover))
        detail[f"A_{w}"] = A_df
        print(f"  A 판별력 {A:.3f}%p · B 월평균전환 {B:.1f}종목 · "
              f"C 선행 {C:.0f}일 · D 임계근접 {D:.1f}% · 커버 {cover:.1f}%")

    s = pd.DataFrame(summary)
    print("\n" + "=" * 62)
    print(s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("=" * 62)
    print("A 클수록 / B 작을수록 / C 클수록 / D 작을수록 좋다")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"rolling_window_sweep_{datetime.now():%Y%m%d}.xlsx"
    with pd.ExcelWriter(out) as xw:
        s.to_excel(xw, sheet_name="요약", index=False)
        for k, v in detail.items():
            if len(v):
                v.to_excel(xw, sheet_name=k[:31], index=False)
    print(f"\n✅ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
