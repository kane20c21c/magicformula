"""pattern_study.py — 눌림 패턴화 조건 × 반등 정의 격자 연구 (Kane 지시 2026-08-17).

목적 (Kane 문제 제기)
  · 패턴화 조건을 바꿔보는 것  = "눌림후반등을 얼마나 찾아내느냐" (재현율)
  · 반등 정의를 바꿔보는 것    = "그 눌림후반등이 진짜냐" (정의에 강건한가)

패턴 격자
  basis "high"     : 종가 ≤ ratio × N일 최고종가          (고점 낙폭 — 현행 계열)
                     N ∈ {20,40,60}, ratio ∈ {0.90,0.80,0.70}
  basis "low_prox" : 종가 ≤ (1+tol) × 직전 N일 최저종가    (저점 근접 — Kane 의도)
                     N ∈ {20,40,60}, tol ∈ {0.10,0.20,0.30}
  basis "low_break": 종가 ≤ ratio × 직전 N일 최저종가      (신저가 이탈 깊이 — 참고군)
                     ratio ∈ {0.90,0.80} (0.70 은 사실상 공집합)
  추세 필터        : 종가 > MA, MA ∈ {120,150,200,240}
  신호            : onset (조건이 전일 False → 오늘 True) + 유니버스 편입

반등 정의 (Kane 지정)
  a : 20거래일 내 종가 +10%
  b : 10거래일 내 종가 +10%
  c :  5거래일 내 종가  +5%

측정 (2015-01~2026-06, 신호 수준 — 슬롯/현금 교란 없음)
  정밀도 = P(반등 | onset)  — onset 다음날 시가 매수 가정, 이후 H일 내 종가가 시가 대비 +X%
  재현율 = 기회 에피소드(현행 눌림 −10%/60일 ∧ 반등정의) 중 onset 이 구간 내 발생한 비율
  전방수익 = onset 다음날 시가 → +5/10/20일 종가

실행: python3 pattern_study.py          (요약 출력 + out/pattern_grid.csv)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from backtest import UniverseParams, VolScaleParams, load_panel

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

BOUNCE_DEFS = {"a 20일+10%": (20, 0.10), "b 10일+10%": (10, 0.10), "c 5일+5%": (5, 0.05)}
WINDOWS = (20, 40, 60)
HIGH_RATIOS = (0.90, 0.80, 0.70)
LOW_TOLS = (0.10, 0.20, 0.30)
BREAK_RATIOS = (0.90, 0.80)
MAS = (120, 150, 200, 240)


def fwd_max_close(cl: pd.DataFrame, h: int) -> pd.DataFrame:
    """내일부터 h일간 최고 종가 (당일 제외)."""
    return cl.shift(-1).iloc[::-1].rolling(h, min_periods=1).max().iloc[::-1]


def episodes_np(mask: np.ndarray):
    out = []
    for j in range(mask.shape[1]):
        col = mask[:, j]
        if not col.any():
            continue
        d = np.diff(col.astype(np.int8))
        starts = list(np.flatnonzero(d == 1) + 1)
        ends = list(np.flatnonzero(d == -1))
        if col[0]:
            starts.insert(0, 0)
        if col[-1]:
            ends.append(len(col) - 1)
        out.extend((j, s, e) for s, e in zip(starts, ends))
    return out


if __name__ == "__main__":
    panel = load_panel(UniverseParams(), VolScaleParams())
    cl, op = panel.close, panel.open
    elig = panel.elig
    dates = panel.dates
    lo_i = int(np.searchsorted(dates, pd.Timestamp("2015-01-01")))
    hi_i = int(np.searchsorted(dates, pd.Timestamp("2026-06-30"), side="right")) - 1
    n_d = len(dates)
    print(f"패널 {len(panel.tickers)}종목 · {n_d:,}거래일\n")

    # ── 공용 재료 ──
    highs = {n: cl.rolling(n, min_periods=max(10, n // 2)).max() for n in WINDOWS}
    lows = {n: cl.shift(1).rolling(n, min_periods=max(10, n // 2)).min() for n in WINDOWS}
    mas = {n: cl.rolling(n, min_periods=int(n * 0.7)).mean() for n in MAS}
    fwd = {k: fwd_max_close(cl, h) for k, (h, x) in BOUNCE_DEFS.items()}
    op_np, cl_np = op.to_numpy(), cl.to_numpy()

    # 기회 집합 (정답, 패턴 무관 고정): 현행 눌림 −10%/60일 ∧ 반등정의 ∧ 유니버스
    dip_cur = (cl <= 0.90 * highs[60]) & elig
    opp_eps = {}
    for k, (h, x) in BOUNCE_DEFS.items():
        m = (dip_cur & (fwd[k] >= cl * (1 + x))).fillna(False).to_numpy()
        opp_eps[k] = [(j, s, e) for j, s, e in episodes_np(m) if lo_i <= s <= hi_i]
        print(f"기회 에피소드 [{k}] {len(opp_eps[k]):,}건")
    print()

    # ── 격자 ──
    combos = []
    for n in WINDOWS:
        for r in HIGH_RATIOS:
            combos.append(("고점낙폭", n, r, cl <= r * highs[n]))
        for t in LOW_TOLS:
            combos.append(("저점근접", n, t, cl <= (1 + t) * lows[n]))
        for r in BREAK_RATIOS:
            combos.append(("신저가이탈", n, r, cl <= r * lows[n]))

    rows = []
    for basis, n, param, cond0 in combos:
        for ma_n in MAS:
            cond = (cond0 & (cl > mas[ma_n]) & elig).fillna(False)
            onset = (cond & ~cond.shift(1).fillna(False)).to_numpy()
            onset[:lo_i] = False
            onset[hi_i + 1:] = False
            di, tj = np.nonzero(onset)
            ok = di + 1 < n_d
            di, tj = di[ok], tj[ok]
            entry = op_np[di + 1, tj]
            good = np.isfinite(entry) & (entry > 0)
            di, tj, entry = di[good], tj[good], entry[good]
            n_sig = len(di)
            if n_sig < 30:
                continue
            row = {"basis": basis, "window": n, "param": param, "ma": ma_n, "n_sig": n_sig}
            # 정밀도: 진입가(다음날 시가) 대비 H일 내 최고 종가 ≥ +X%
            for k, (h, x) in BOUNCE_DEFS.items():
                fm = fwd[k].to_numpy()[di + 1, tj]     # 매수일 기준 이후 h일
                row[f"정밀도_{k[0]}"] = np.nanmean(fm >= entry * (1 + x)) * 100
            # 전방수익
            for h in (5, 10, 20):
                k2 = np.minimum(di + 1 + h, n_d - 1)
                row[f"fwd{h}"] = np.nanmean(cl_np[k2, tj] / entry - 1) * 100
            # 재현율
            on_full = onset
            for k, eps in opp_eps.items():
                got = sum(1 for j, s, e in eps if on_full[s:min(e, hi_i) + 1, j].any())
                row[f"재현율_{k[0]}"] = got / len(eps) * 100 if eps else np.nan
            rows.append(row)

    df = pd.DataFrame(rows)
    df["정밀도_평균"] = df[[f"정밀도_{c}" for c in "abc"]].mean(axis=1)
    df["재현율_평균"] = df[[f"재현율_{c}" for c in "abc"]].mean(axis=1)
    df.to_csv(OUT / "pattern_grid.csv", index=False)

    pd.set_option("display.width", 250)
    base = df[(df.basis == "고점낙폭") & (df.window == 60) & (df.param == 0.90) & (df.ma == 120)]
    print("── 기준선 (고점낙폭 60/0.90 + MA120) ──")
    print(base.round(1).to_string(index=False), "\n")

    for c in ("a", "b", "c"):
        print(f"── 정밀도_{c} 상위 12 (n_sig ≥ 100) ──")
        top = df[df.n_sig >= 100].nlargest(12, f"정밀도_{c}")
        cols = ["basis", "window", "param", "ma", "n_sig",
                f"정밀도_{c}", f"재현율_{c}", "fwd10", "fwd20"]
        print(top[cols].round(1).to_string(index=False), "\n")

    print("── 종합: 정밀도_평균 상위 15 (n_sig ≥ 100) ──")
    top = df[df.n_sig >= 100].nlargest(15, "정밀도_평균")
    print(top[["basis", "window", "param", "ma", "n_sig", "정밀도_평균", "재현율_평균",
               "정밀도_a", "정밀도_b", "정밀도_c", "fwd10", "fwd20"]].round(1)
          .to_string(index=False), "\n")

    print("── 종합: 재현율_평균 상위 15 (n_sig ≥ 100) ──")
    top = df[df.n_sig >= 100].nlargest(15, "재현율_평균")
    print(top[["basis", "window", "param", "ma", "n_sig", "재현율_평균", "정밀도_평균",
               "fwd10", "fwd20"]].round(1).to_string(index=False))
    print(f"\n저장: {OUT/'pattern_grid.csv'}  (전체 {len(df)}조합)")
