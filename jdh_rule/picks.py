"""jdh_rule/picks.py — 선행 3종목 선정 (유니버스 조정일 기준, 케인 2026-11-23).

규칙 (연구 k200_lead_walkforward.py 와 같은 계산, 선정 시점만 '매월 초' → '유니버스 적용일'):
  관측 = (신호일 t, 결과일 t+1) — 종목 10거래일 로그수익률 부호 vs KOSPI200 다음 날 부호.
  유니버스 적용일(effective_date) **이전** 결과만 → 직전 240관측을 40개씩 6구간.
  구간별 적중률(0 제외)·균형적중률(지수 상승일/하락일 적중률 평균). 6구간 모두 30관측 이상인 종목만.
  적중률 기울기(구간 번호 OLS) > 0 중 최근 4구간 균형적중률 평균 상위 3.
  후보 순서 = 유니버스 명단 순서 (동률 처리 재현용, 연구와 같음).
재료 = LLV panel.parquet(Close) · LLV tickers/KOSPI200.parquet · LLV groups/universe_*.json (읽기 전용).
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from jdh_rule.config import DEFAULT, JdhConfig, PANEL, KOSPI200_PARQUET, PICKS_DIR, RULE_VERSION, LLV_ROOT

KST = timezone(timedelta(hours=9))


def load_closes(tickers: list[str], panel: Path = PANEL) -> pd.DataFrame:
    p = pd.read_parquet(panel, columns=["Date", "Ticker", "Close"])
    p["Ticker"] = p["Ticker"].astype(str).str.zfill(6)
    p = p[p.Ticker.isin(tickers)]
    px = p.pivot(index="Date", columns="Ticker", values="Close").sort_index()
    px.index = pd.to_datetime(px.index)
    return px.reindex(columns=[t for t in tickers if t in px.columns])


def load_k200(path: Path = KOSPI200_PARQUET) -> pd.Series:
    k = pd.read_parquet(path).set_index("Date")["Close"].astype(float)
    k.index = pd.to_datetime(k.index)
    return k.sort_index()


def select_leads(px: pd.DataFrame, k200: pd.Series, cutoff, cfg: JdhConfig = DEFAULT) -> dict:
    """cutoff(유니버스 적용일) 이전 결과만으로 선행 종목을 고른다. px 열 순서 = 후보 순서."""
    cutoff = pd.Timestamp(cutoff)
    idx = px.index.intersection(k200.index)
    px, k = px.loc[idx], k200.loc[idx]
    sel = np.sign(np.log(px / px.shift(cfg.sel_ret_days))).values
    y = np.r_[np.sign(np.diff(np.log(k.values))), np.nan]
    nextd = pd.DatetimeIndex(list(idx[1:]) + [pd.NaT])
    need = cfg.n_blocks * cfg.block_len
    m = np.where(np.asarray(~nextd.isna() & (nextd < cutoff)))[0][-need:]
    if len(m) < need:
        raise ValueError(f"선정 자료 부족: {len(m)} < {need} 관측 (cutoff {cutoff.date()})")
    ss, yy = sel[m], y[m]
    nb, bl = cfg.n_blocks, cfg.block_len
    HR = np.full((nb, px.shape[1]), np.nan); BA = HR.copy()
    for i in range(nb):
        s_, y_ = ss[i * bl:(i + 1) * bl], yy[i * bl:(i + 1) * bl]
        ok = (s_ != 0) & ~np.isnan(s_) & (y_ != 0)[:, None]
        h = (s_ == y_[:, None]) & ok; n = ok.sum(0)
        up, dn = (y_ > 0)[:, None] & ok, (y_ < 0)[:, None] & ok
        with np.errstate(invalid="ignore", divide="ignore"):
            hr = h.sum(0) / n
            ba = 0.5 * ((h & up).sum(0) / up.sum(0) + (h & dn).sum(0) / dn.sum(0))
        hr[n < cfg.min_block_obs] = np.nan; ba[n < cfg.min_block_obs] = np.nan
        HR[i], BA[i] = hr, ba
    valid = ~np.isnan(HR).any(0)
    x = np.arange(nb) - (nb - 1) / 2
    with np.errstate(invalid="ignore"):
        mean_hr = HR.mean(0)
        slope = np.where(valid, (x[:, None] * (HR - mean_hr)).sum(0) / (x ** 2).sum(), np.nan)
        rec = BA[-cfg.recent_blocks:].mean(0)
    cand = np.where(valid & (slope > 0))[0]
    top = cand[np.argsort(-rec[cand], kind="stable")][:cfg.n_leads]
    tick = list(px.columns)
    return {
        "cutoff": str(cutoff.date()),
        "window": [str(idx[m[0]].date()), str(idx[m[-1]].date())],
        "leads": [{"ticker": tick[j], "slope": float(slope[j]), "bal_recent": float(rec[j]),
                   "hit_blocks": [round(float(v), 4) for v in HR[:, j]],
                   "bal_blocks": [round(float(v), 4) for v in BA[:, j]]} for j in top],
        "n_candidates": int(len(cand)), "n_valid": int(valid.sum()), "n_universe": len(tick),
    }


def _universe_doc(as_of: date) -> Optional[dict]:
    import sys
    if str(LLV_ROOT) not in sys.path:
        sys.path.insert(0, str(LLV_ROOT))
    from stolab_data.ticker_groups import get_universe
    return get_universe(as_of)


def _names(tickers: list[str], panel: Path = PANEL) -> dict:
    try:
        p = pd.read_parquet(panel, columns=["Date", "Ticker", "Name"])
        p["Ticker"] = p["Ticker"].astype(str).str.zfill(6)
        p = p[p.Ticker.isin(tickers)].sort_values("Date")
        return p.groupby("Ticker")["Name"].last().to_dict()
    except Exception:
        return {}


def picks_path(effective: str, picks_dir: Path = PICKS_DIR) -> Path:
    return picks_dir / f"picks_{effective}.json"


def load_picks(effective: str, picks_dir: Path = PICKS_DIR) -> Optional[dict]:
    p = picks_path(effective, picks_dir)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def ensure_picks(as_of: date, cfg: JdhConfig = DEFAULT, picks_dir: Path = PICKS_DIR,
                 universe_fn=None, panel: Path = PANEL, k200_path: Path = KOSPI200_PARQUET) -> dict:
    """as_of 에 적용되는 유니버스의 선행종목. 없으면 그 자리에서 만든다 (자가치유, 멱등)."""
    doc = (universe_fn or _universe_doc)(as_of)
    if not doc:
        raise RuntimeError(f"{as_of} 에 적용되는 유니버스 명단이 없음 (LLV groups/)")
    eff = str(doc.get("effective_date"))[:10]
    got = load_picks(eff, picks_dir)
    if got:
        return got
    members = [str(t).zfill(6) for t in doc["members"]]
    res = select_leads(load_closes(members, panel), load_k200(k200_path), eff, cfg)
    nm = _names([x["ticker"] for x in res["leads"]], panel)
    for x in res["leads"]:
        x["name"] = nm.get(x["ticker"], "")
    out = {"rule": "jdh", "rule_version": RULE_VERSION, "universe_effective": eff,
           "universe_check": str(doc.get("check_date"))[:10], "universe_file": doc.get("_file"),
           "created_at": datetime.now(KST).isoformat(timespec="seconds"), **res}
    picks_dir.mkdir(parents=True, exist_ok=True)
    tmp = picks_path(eff, picks_dir).with_suffix(".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(picks_path(eff, picks_dir))
    return out


__all__ = ["select_leads", "ensure_picks", "load_picks", "load_closes", "load_k200", "picks_path"]
