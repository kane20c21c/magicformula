"""diag_recall.py — '기회는 많았는데 못 잡았다' 를 수치로 분해한다 (Kane 지시 2026-08-17).

문제 정의
  (1) 눌림 후 반등 기회는 많았는데 모델이 못 찾아냈다  ← 이 파일이 다루는 것 (재현율)
  (2) 모델이 부른 신호가 반등으로 안 이어졌다          ← diag_entry.py (정밀도)

기회(opportunity) 정의
  유니버스 편입 종목의 거래일 중
    · 눌림 상태 : close ≤ (1 − PULLBACK) × 60일 고점
    · 반등 성사 : 이후 HORIZON 거래일 안에 종가가 그날 종가 대비 +BOUNCE 이상
  연속된 기회일은 하나의 **에피소드**로 묶는다 (같은 반등을 중복 계산하지 않기 위해).

재현율 분해
  잡음   : 에피소드 기간 안에 신호가 뜨고 실제 매수까지 간 경우
  놓침A  : 신호는 떴는데 슬롯·현금이 없어 못 산 경우      → 사이징/슬롯 문제
  놓침B  : 신호 자체가 안 뜬 경우                          → 트리거 문제
  놓침B 는 다시 조건별로 귀속한다 (MA120 미달 / 눌림 미달 / onset 제한 / 유니버스 밖).

실행: python3 diag_recall.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from backtest import (EntryParams, ExitParams, PortfolioParams, UniverseParams,
                      VolScaleParams, compute_signals, load_panel, run_backtest)

PULLBACK = 0.10       # 눌림 정의 (전략과 동일)
BOUNCE = 0.10         # 반등 성사 기준
HORIZON = 20          # 반등 관측 지평 (거래일)


def opportunity_mask(panel, pullback=PULLBACK, bounce=BOUNCE, horizon=HORIZON):
    cl = panel.close
    high60 = cl.rolling(60, min_periods=30).max()
    in_dip = (cl <= (1 - pullback) * high60) & panel.elig
    # 이후 horizon 거래일 내 최고 종가 (당일 제외)
    fwd_max = cl.shift(-1).rolling(horizon, min_periods=1).max().shift(-(horizon - 1))
    bounced = fwd_max >= cl * (1 + bounce)
    return (in_dip & bounced).fillna(False), in_dip.fillna(False)


def episodes(mask: pd.DataFrame):
    """연속 True 구간을 (ticker, start_i, end_i) 리스트로."""
    m = mask.to_numpy()
    out = []
    for j in range(m.shape[1]):
        col = m[:, j]
        if not col.any():
            continue
        d = np.diff(col.astype(np.int8))
        starts = list(np.flatnonzero(d == 1) + 1)
        ends = list(np.flatnonzero(d == -1))
        if col[0]:
            starts.insert(0, 0)
        if col[-1]:
            ends.append(len(col) - 1)
        for s, e in zip(starts, ends):
            out.append((j, s, e))
    return out


if __name__ == "__main__":
    panel = load_panel(UniverseParams(), VolScaleParams())
    dates = panel.dates
    lo_i = int(np.searchsorted(dates, pd.Timestamp("2015-01-01")))
    hi_i = int(np.searchsorted(dates, pd.Timestamp("2026-06-30"), side="right")) - 1
    print(f"패널 종목 {len(panel.tickers)} · 구간 {dates[lo_i].date()}~{dates[hi_i].date()}\n")

    opp, in_dip = opportunity_mask(panel)
    eps = [(j, s, e) for j, s, e in episodes(opp) if lo_i <= s <= hi_i]
    print(f"기회 정의: 눌림 −{PULLBACK:.0%} 상태 & 이후 {HORIZON}거래일 내 +{BOUNCE:.0%} 반등")
    print(f"  기회 에피소드 {len(eps):,}건 (종목 {len({j for j,_,_ in eps})}개)\n")

    # ── 신호·매수 실적 ──
    ep0 = EntryParams()
    sig = compute_signals(panel, ep0)
    signal = sig["signal"].to_numpy()
    cond = sig["cond"].to_numpy()

    res = run_backtest(panel, ep0, ExitParams(), PortfolioParams())
    tr = res["trades"]
    buys = tr[tr.side == "BUY"]
    col = {t: j for j, t in enumerate(panel.tickers)}
    di = {d: i for i, d in enumerate(dates)}
    bought = np.zeros_like(signal)
    for _, r in buys.iterrows():
        bought[di[r["date"]], col[r["ticker"]]] = True

    cl = panel.close.to_numpy()
    ma120 = panel.close.rolling(120, min_periods=80).mean().to_numpy()
    high60 = panel.close.rolling(60, min_periods=30).max().to_numpy()
    elig = panel.elig.to_numpy()

    caught = miss_slot = miss_sig = 0
    reasons = {"MA120 아래": 0, "눌림 10% 미달": 0, "onset 제한(이미 눌림 진행중)": 0,
               "유니버스 밖": 0, "기타": 0}
    for j, s, e in eps:
        win = slice(s, min(e, hi_i) + 1)
        if signal[win, j].any():
            # 신호 다음날 매수까지 갔는가
            si = s + int(np.flatnonzero(signal[win, j])[0])
            if bought[si:min(si + 3, hi_i + 1), j].any():
                caught += 1
            else:
                miss_slot += 1
            continue
        miss_sig += 1
        # 왜 신호가 없었나 — 에피소드 첫날 기준으로 귀속
        i = s
        if not elig[i, j]:
            reasons["유니버스 밖"] += 1
        elif not np.isfinite(ma120[i, j]) or cl[i, j] <= ma120[i, j]:
            reasons["MA120 아래"] += 1
        elif cl[i, j] > (1 - PULLBACK) * high60[i, j]:
            reasons["눌림 10% 미달"] += 1
        elif cond[max(i - 1, 0), j]:
            reasons["onset 제한(이미 눌림 진행중)"] += 1
        else:
            reasons["기타"] += 1

    n = len(eps)
    print("── 재현율 분해 ──")
    print(f"  잡았다 (신호 + 매수)      {caught:5,d}  {caught/n*100:5.1f}%")
    print(f"  놓침A (신호O·슬롯/현금X)  {miss_slot:5,d}  {miss_slot/n*100:5.1f}%")
    print(f"  놓침B (신호 자체가 없음)  {miss_sig:5,d}  {miss_sig/n*100:5.1f}%")
    print("\n── 놓침B 사유 귀속 (에피소드 첫날 기준) ──")
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {k:<28} {v:5,d}  전체 대비 {v/n*100:5.1f}%")

    # ── onset 제한의 크기: 눌림 구간 길이 분포 ──
    dip_eps = [(j, s, e) for j, s, e in episodes(in_dip) if lo_i <= s <= hi_i]
    lens = np.array([e - s + 1 for _, s, e in dip_eps])
    print(f"\n── 눌림 구간(연속 −10% 이하) 길이 분포 — n={len(lens):,} ──")
    for q in (50, 75, 90, 95):
        print(f"  p{q}: {np.percentile(lens, q):.0f}거래일")
    print(f"  평균 {lens.mean():.1f}거래일 · 최대 {lens.max()}거래일")
    print("  ⚠ onset 은 이 구간의 '첫날 하루' 만 신호를 준다.")

    # ── 슬롯 사용 분포 ──
    eq = res["equity"]
    print(f"\n── 슬롯 사용 ── 평균 보유종목 {eq['n_pos'].mean():.1f}/20 "
          f"· 투자비중 평균 {(eq['mv']/eq['equity']).mean()*100:.1f}% "
          f"· 20슬롯 만재일 {(eq['n_pos']>=20).mean()*100:.1f}%")
    print(f"  보유 0종목인 날 {(eq['n_pos']==0).mean()*100:.1f}%")
