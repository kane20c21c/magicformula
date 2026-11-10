# -*- coding: utf-8 -*-
"""통일장 시장 층 판정 — 순수 함수 (입력 시계열 → 상태). True = 보유, False = 현금.

연구 코드(korean_mkt_study/regime_detect·regime_gate·market_layer_feasibility)와 **같은 식**을 자체 구현한다 —
연구 파일은 커밋 전이고 운영 모듈이 연구 스크립트에 기대면 안 되기 때문. 같음은 tests/test_market_layer.py 가
고정본(tests/fixtures/market_layer_research.csv)으로 지킨다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT, MarketLayerConfig


def hyst(x: pd.Series, hi: float, lo: float, start: bool = True) -> pd.Series:
    """히스테리시스 — x ≥ hi 면 True, x ≤ lo 면 False, 사이·결측은 직전 상태 유지 (시작 True)."""
    out, cur = [], start
    for v in x.to_numpy(dtype=float):
        if np.isfinite(v):
            if v >= hi:
                cur = True
            elif v <= lo:
                cur = False
        out.append(cur)
    return pd.Series(out, index=x.index, dtype=bool)


def ma_state(p: pd.Series, n: int) -> pd.Series:
    """종가가 n일 단순이동평균 위(True)/아래(False). 같으면·이평 결측이면 직전 상태 유지 (시작 True)."""
    m = p.rolling(n, min_periods=n).mean().to_numpy(dtype=float)
    out, cur = [], True
    for x, mm in zip(p.to_numpy(dtype=float), m):
        if np.isfinite(mm):
            if x > mm:
                cur = True
            elif x < mm:
                cur = False
        out.append(cur)
    return pd.Series(out, index=p.index, dtype=bool)


def breadth_value(ratio: pd.Series, idx: pd.Index, cfg: MarketLayerConfig = DEFAULT) -> pd.Series:
    """20일 지수평균 − 직전 250일 중앙값 (비율 자체의 날짜축에서 계산한 뒤 idx 로 맞춘다)."""
    r = ratio.dropna().sort_index()
    e = r.ewm(span=cfg.breadth_span, adjust=False).mean()
    med = r.rolling(cfg.breadth_median_win, min_periods=cfg.breadth_median_min).median()
    return (e - med).reindex(idx)


def breadth_state(ratio: pd.Series, idx: pd.Index, cfg: MarketLayerConfig = DEFAULT) -> pd.Series:
    return hyst(breadth_value(ratio, idx, cfg), cfg.breadth_band, -cfg.breadth_band)


def c2_state(k: pd.Series, mid: pd.Series, cfg: MarketLayerConfig = DEFAULT) -> pd.Series:
    """C2 괴리형 — 두 지수 모두 n일선 위(dv)인 상태가 꺾이는 날 퇴출, 퇴출 뒤 저점 +rb 반등 또는 dv 회복 시 복귀."""
    dv = (ma_state(k, cfg.c2_ma) & ma_state(mid.reindex(k.index), cfg.c2_ma)).to_numpy()
    kv = k.to_numpy(dtype=float)
    out, cur, low = [], True, np.inf
    for i in range(len(kv)):
        if cur:
            if i > 0 and dv[i - 1] and not dv[i]:
                cur, low = False, kv[i]
        else:
            low = min(low, kv[i])
            if kv[i] >= low * (1 + cfg.c2_rebound) or (dv[i] and not dv[i - 1]):
                cur = True
        out.append(cur)
    return pd.Series(out, index=k.index, dtype=bool)


def gap_shock(k_open: pd.Series, k_close: pd.Series, idx: pd.Index,
              cfg: MarketLayerConfig = DEFAULT) -> tuple[np.ndarray, pd.Series]:
    """(관문 창, 쇼크일). 쇼크일 = 하루 ≤ −3% 이고 그 하락에서 갭 몫(log 기준) ≥ 40%. 창 = 쇼크일 포함 10거래일."""
    op, cl = k_open.reindex(idx).astype(float), k_close.reindex(idx).astype(float)
    cp = cl.shift(1)
    r, g = cl / cp - 1, op / cp - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.log1p(g) / np.log1p(r)
    shock = (r <= cfg.gate_day_ret) & (share >= cfg.gate_gap_share)
    window = shock.astype(int).rolling(cfg.gate_window, min_periods=1).max() > 0
    return window.to_numpy(), shock


def gated(gate: np.ndarray, br: pd.Series, base: pd.Series) -> pd.Series:
    """관문 모드 — 관문 창 & 브레드스 하락이면 모드 진입(현금), 브레드스 상승이면 해제. 모드 밖은 base 를 따른다."""
    b, bs = br.to_numpy(bool), base.reindex(br.index).to_numpy(bool)
    out, mode = [], False
    for i in range(len(b)):
        if not mode and gate[i] and not b[i]:
            mode = True
        elif mode and b[i]:
            mode = False
        out.append(False if mode else bool(bs[i]))
    return pd.Series(out, index=br.index, dtype=bool)


def judge(k_open: pd.Series, k_close: pd.Series, tmi_close: pd.Series, ratio: pd.Series,
          cfg: MarketLayerConfig = DEFAULT) -> pd.DataFrame:
    """날짜별 판정표 — 날짜축 = 코스피 200 종가가 있는 날.

    열: breadth_value · breadth(상승 True) · c2 · gate_window · shock · mode(관문 모드) · p7(⑦) · p5(⑤)
    """
    idx = k_close.dropna().index
    k = k_close.reindex(idx).astype(float)
    bv = breadth_value(ratio, idx, cfg)
    br = hyst(bv, cfg.breadth_band, -cfg.breadth_band)
    c2 = c2_state(k, tmi_close.astype(float), cfg)
    win, shock = gap_shock(k_open, k, idx, cfg)
    p7 = gated(win, br, c2)
    p5 = gated(win, br, pd.Series(True, index=idx))
    return pd.DataFrame({"breadth_value": bv, "breadth": br, "c2": c2, "gate_window": win,
                         "shock": shock.to_numpy(), "mode": ~p5.to_numpy(), "p7": p7, "p5": p5}, index=idx)
