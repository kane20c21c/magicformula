# -*- coding: utf-8 -*-
"""통일장 국면 라벨 표 (regime history) — 다른 분석이 **날짜로 붙여 쓰는** 시장 국면 이력 (2026-11-10 케인).

판정기(judge.py)와 **같은 함수**로 전 구간을 다시 계산해 하루 한 줄로 저장한다. 식을 복사하지 않으므로
판정기 정의가 바뀌면 다음 갱신 때 표가 따라간다. 저장 = output/market_layer/regime_history.parquet (+ .meta.json).

열 (날짜축 = 코스피 200 종가가 있는 날)
  재료     k200_close · k200_ret · k200_gap(시가/전일종가−1) · tmi_close · breadth_ratio · breadth_value
  국면     breadth (상승 True / 하락 False) · c2 (보유 True / 퇴출 False) · regime ("상승·보유" 등 2×2)
  관문     shock (정의 B 갭쇼크일) · gate_window · days_since_shock · gate_mode
  판정     p7 (⑦ 보유 True) · p5 (⑤ 보유 True)
  구간     {breadth,c2,regime}_spell (구간 번호) · _day (구간 안 며칠째, 1부터) · _len (구간 전체 길이) ·
           _cens ("" / "시작 미상" / "진행 중" / "시작 미상·진행 중")

⚠ **유효 구간 밖은 결측(NA)이다** — 판정기는 시작값을 '보유'로 두고 결측일은 직전 상태를 유지하므로
  재료가 쌓이기 전 구간이 '상승·보유'로 보인다. 이 표는 그 구간을 지운다.
    브레드스 : 20일 지수평균 − 250일 중앙값이 처음으로 ±band 를 넘은 날부터 (그 전 상태는 시작값에 달려 있다)
    C2       : 두 지수의 200일선이 모두 생긴 뒤, 괴리형(둘 다 위)이 처음 성립한 날부터
    p5 = 브레드스 유효부터 · p7 = 둘 다 유효부터 · regime = 둘 다 유효부터
⚠ 구간 길이를 비교할 땐 _cens 가 빈 문자열인 구간만 쓸 것 — 첫 구간은 시작을, 마지막 구간은 끝을 모른다.
⚠ 라벨은 그날 **종가까지** 본 값이다. 그날 장중 매매 판단에 쓰려면 attach_regime(lag=1) 로 전날 라벨을 붙인다.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .config import DEFAULT, MarketLayerConfig
from .judge import judge, ma_state

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = PROJECT_ROOT / "output" / "market_layer" / "regime_history.parquet"
LABEL_COLS = ["breadth", "c2", "regime", "p7", "p5", "shock", "gate_window", "days_since_shock", "gate_mode"]
_REGIME = {(True, True): "상승·보유", (True, False): "상승·퇴출", (False, True): "하락·보유", (False, False): "하락·퇴출"}


def _first(mask: pd.Series) -> Optional[pd.Timestamp]:
    m = mask.fillna(False).to_numpy(bool)
    return mask.index[int(np.argmax(m))] if m.any() else None


def breadth_valid_from(bv: pd.Series, cfg: MarketLayerConfig = DEFAULT) -> Optional[pd.Timestamp]:
    """브레드스 상태가 시작값과 무관해지는 첫날 = 값이 처음 ±band 를 넘은 날."""
    return _first(bv.abs() >= cfg.breadth_band)


def c2_valid_from(k: pd.Series, mid: pd.Series, cfg: MarketLayerConfig = DEFAULT) -> Optional[pd.Timestamp]:
    """C2 가 시작값과 무관해지는 첫날 = 두 200일선이 모두 생긴 뒤 괴리형(둘 다 위)이 처음 성립한 날."""
    mid = mid.reindex(k.index)
    both = k.rolling(cfg.c2_ma, min_periods=cfg.c2_ma).mean().notna() & mid.rolling(cfg.c2_ma, min_periods=cfg.c2_ma).mean().notna()
    dv = ma_state(k, cfg.c2_ma) & ma_state(mid, cfg.c2_ma)
    return _first(both & dv)


def _mask(s: pd.Series, start: Optional[pd.Timestamp]) -> pd.Series:
    out = s.astype("boolean")
    if start is None:
        return pd.Series(pd.NA, index=s.index, dtype="boolean")
    out[s.index < start] = pd.NA
    return out


def spells(label: pd.Series) -> pd.DataFrame:
    """결측이 아닌 구간에서 값이 바뀔 때마다 구간 번호를 올린다 (결측 행은 결측)."""
    v = label.dropna()
    out = pd.DataFrame(index=label.index, columns=["spell", "day", "len", "cens"])
    out = out.astype({"spell": "Int64", "day": "Int64", "len": "Int64", "cens": "string"})
    if v.empty:
        return out
    sid = v.ne(v.shift()).fillna(True).astype(bool).cumsum().astype(int)   # 첫 행의 shift 는 NA → 새 구간
    day = sid.groupby(sid).cumcount() + 1
    ln = sid.map(sid.value_counts())
    first, last = sid.iloc[0], sid.iloc[-1]
    cens = pd.Series("", index=v.index, dtype="string")
    cens[sid == first] = "시작 미상"
    cens[sid == last] = np.where(sid[sid == last] == first, "시작 미상·진행 중", "진행 중")
    out.loc[v.index, "spell"] = sid.to_numpy()
    out.loc[v.index, "day"] = day.to_numpy()
    out.loc[v.index, "len"] = ln.to_numpy()
    out.loc[v.index, "cens"] = cens.to_numpy()
    return out


def build_history(k_open: pd.Series, k_close: pd.Series, tmi_close: pd.Series, ratio: pd.Series,
                  cfg: MarketLayerConfig = DEFAULT) -> tuple[pd.DataFrame, dict]:
    """(라벨 표, 유효 시작일 정보). 판정은 judge() 그대로 — 여기서는 결측 처리·구간·편의 열만 더한다."""
    d = judge(k_open, k_close, tmi_close, ratio, cfg)
    idx = d.index
    k = k_close.reindex(idx).astype(float)
    ko = k_open.reindex(idx).astype(float)
    kp = k.shift(1)
    b0 = breadth_valid_from(d.breadth_value, cfg)
    c0 = c2_valid_from(k, tmi_close.astype(float), cfg)
    both0 = None if b0 is None or c0 is None else max(b0, c0)

    br, c2 = _mask(d.breadth, b0), _mask(d.c2, c0)
    regime = pd.Series(pd.NA, index=idx, dtype="string")
    ok = br.notna() & c2.notna()
    regime[ok] = [_REGIME[(bool(a), bool(b))] for a, b in zip(br[ok], c2[ok])]

    shock = d.shock.fillna(False).astype(bool)
    pos = np.arange(len(idx))
    last = pd.Series(np.where(shock.to_numpy(), pos, np.nan), index=idx).ffill()
    since = (pd.Series(pos, index=idx) - last).astype("Int64")

    h = pd.DataFrame({
        "k200_close": k, "k200_ret": k / kp - 1, "k200_gap": ko / kp - 1,
        "tmi_close": tmi_close.reindex(idx).astype(float),
        "breadth_ratio": ratio.reindex(idx).astype(float), "breadth_value": d.breadth_value.astype(float),
        "breadth": br, "c2": c2, "regime": regime,
        "shock": shock, "gate_window": d.gate_window.astype(bool), "days_since_shock": since,
        "gate_mode": _mask(d["mode"], b0), "p7": _mask(d.p7, both0), "p5": _mask(d.p5, b0),
    }, index=idx)
    h.index.name = "date"
    for name in ("breadth", "c2", "regime"):
        sp = spells(h[name])
        for c in sp.columns:
            h[f"{name}_{c}"] = sp[c]
    info = {"breadth_valid_from": b0, "c2_valid_from": c0, "p7_valid_from": both0}
    return h, {k_: (None if v is None else v.strftime("%Y-%m-%d")) for k_, v in info.items()}


# ── 저장·읽기 ────────────────────────────────────────────────────────────────

def last_confirmed_day(root: Optional[Path] = None, breadth_path: Optional[Path] = None) -> pd.Timestamp:
    """코스피 200 · 중형 TMI · KRX 확정 브레드스가 **모두** 있는 마지막 날 — 표에는 확정 재료만 싣는다."""
    from .inputs import _index, _llv
    root = _llv(root)
    from stolab_data import market_breadth as mb
    K, T = _index(root, "KOSPI200"), _index(root, "KRX_TMI_MID")
    bdf = mb.load_breadth(Path(breadth_path) if breadth_path else root / "data" / "market_breadth.parquet")
    krx = mb.ratio_series(bdf, mb.SOURCE_KRX, mb.KRX_PHASE)
    return min(K.dropna(subset=["Close"]).index.max(), T.dropna(subset=["Close"]).index.max(), krx.index.max())


def update_history(out_path: Optional[Path] = None, root: Optional[Path] = None, dry_run: bool = False) -> dict:
    """LLV 운영 재료(확정분)로 표를 다시 만들어 원자적으로 저장한다. 반환 = 메타 dict."""
    from .inputs import load_inputs
    day = last_confirmed_day(root)
    inp = load_inputs(day, root=root)                    # 결측 사고는 InputGapError 로 멈춘다
    if inp.source.get("provisional"):
        raise RuntimeError(f"{day:%Y-%m-%d} 재료가 잠정이다 — {inp.source}")
    h, info = build_history(inp.k_open, inp.k_close, inp.tmi, inp.ratio)
    br = inp.ratio.dropna()
    meta = {"version": DEFAULT.version, "generated_at": datetime.now().isoformat(timespec="seconds"),
            "as_of": day.strftime("%Y-%m-%d"), "rows": int(len(h)),
            "first_date": h.index.min().strftime("%Y-%m-%d"), **info,
            "breadth_source": f"LLV market_breadth KRX close 전 종목 ({br.index.min():%Y-%m-%d}~{br.index.max():%Y-%m-%d}, {len(br)}일)",
            "tmi_source": "LLV KRX_TMI_MID", "k200_source": "LLV KOSPI200",
            "spells": {n: int(h[f"{n}_spell"].max()) if h[f"{n}_spell"].notna().any() else 0 for n in ("breadth", "c2", "regime")},
            "latest": {c: (None if pd.isna(h[c].iloc[-1]) else (h[c].iloc[-1] if isinstance(h[c].iloc[-1], str) else
                           (bool(h[c].iloc[-1]) if c not in ("days_since_shock",) else int(h[c].iloc[-1]))))
                       for c in ("breadth", "c2", "regime", "p7", "p5", "days_since_shock")}}
    if not dry_run:
        path = Path(out_path) if out_path else HISTORY_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        h.to_parquet(tmp); tmp.replace(path)
        mp = path.with_suffix(".meta.json")
        mt = mp.with_suffix(".json.tmp")
        mt.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"); mt.replace(mp)
        meta["path"] = str(path)
    return meta


def load_history(path: Optional[Path] = None) -> pd.DataFrame:
    h = pd.read_parquet(Path(path) if path else HISTORY_PATH)
    h.index = pd.to_datetime(h.index)
    return h


def attach_regime(df: pd.DataFrame, date_col: str = "date", cols: Optional[Iterable[str]] = None,
                  hist: Optional[pd.DataFrame] = None, lag: int = 0, prefix: str = "mkt_") -> pd.DataFrame:
    """df 의 date_col 날짜에 국면 라벨을 붙인다 (드라이브라면 date_col = D1).

    lag=0 — 그날 종가까지 본 라벨 (그날 종가 이후 판단·사후 분류용).
    lag=1 — 직전 거래일 라벨 (그날 장중 판단에 쓸 때 — 미래 정보 없음).
    표에 없는 날짜(휴장일·표 범위 밖)는 결측으로 남는다. 붙인 열 이름 = prefix + 원래 이름.
    """
    h = load_history() if hist is None else hist
    cols = list(cols) if cols is not None else ["breadth", "c2", "regime", "p7", "breadth_day", "c2_day", "regime_day"]
    lab = h[cols].shift(lag) if lag else h[cols]
    lab = lab.add_prefix(prefix)
    key = pd.to_datetime(df[date_col]).dt.normalize()
    out = df.copy()
    for c in lab.columns:
        out[c] = lab[c].reindex(key.to_numpy()).to_numpy()
        if str(lab[c].dtype) in ("boolean", "Int64", "string"):
            out[c] = out[c].astype(lab[c].dtype)
    return out
