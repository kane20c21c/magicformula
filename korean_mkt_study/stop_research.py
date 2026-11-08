# -*- coding: utf-8 -*-
"""stop_research.py — 손절 vs 버팀 연구 (케인 2026-11-05).

질문: 스윙 신호로 산 포지션을 손절하는 게 버티는 것보다 나은가? 어떤 폭·방식이면 나은가?

단계
  signal  신호 단위 — 신호마다 진입 후 경로를 따라가며 손절 규칙별 결과 vs 같은 기간 버팀
  mech    메커니즘 — 진입 후 −L% 에 처음 닿은 신호의 이후 수익 vs 비교 기준 (모멘텀 vs 평균회귀)
  (port 은 backtest.run_backtest 를 시간청산으로 확장해 별도 실행)

조건 (라이브 v1.2.3.8 에 맞춤)
  진입     backtest.compute_signals(EntryParams()) & ~strategy_reference.compute_entry_filter (W1 + R²·YZR 제외)
           신호 t 종가 → t+1 시가 매수. 기간 2015-01-01 ~ 2026-06-30 (data/, 유니버스 월별 재구성)
  손절     종가 ≤ 선 → 다음날 시가 × 99.5% 체결.  peak = 보유 중 종가 최고.
  비용     매수 0.015% / 매도 0.015% + 세금 0.2%  (모든 경로에 동일 적용)
  손절 후  현금, 수익 0 (무위험수익 미반영 — 60일 ≈ 0.6%p 손절 쪽에 불리)
사용: [STOP_DATA=data_1104] [STOP_END=2026-11-04] python3 stop_research.py signal | mech
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import backtest as bt
import strategy_reference as sr

import os
DATA_DIR = HERE / os.environ.get("STOP_DATA", "data_1104")        # 2026-11-05: 기본 = 11-04 연장본 (extend_data_1104.py)
END = os.environ.get("STOP_END", "2026-11-04")
bt.DATA = DATA_DIR
OUT = HERE / "out" / "stop_research" / (DATA_DIR.name + ("_day" if os.environ.get("STOP_UNIV", "swing") == "day" else "")); OUT.mkdir(parents=True, exist_ok=True)
START = "2015-01-01"
HORIZONS = (20, 40, 60, 120)
BUY_FEE, SELL_FEE, TAX, SLIP = 0.00015, 0.00015, 0.002, 0.005

UNIV = os.environ.get("STOP_UNIV", "swing")      # swing = 4조·외인30% 월별 유니버스 / day = 데이포트 유니버스(LLV 패널 217종목 상시 편입)

def load_day_panel(end):
    """데이포트 유니버스: LLV 패널 종목(ETF 제외) 전부 상시 편입. 가격은 data_1104/prices (2015~, 수정주가)."""
    sys.path.insert(0, str(HERE.parents[1] / "LongLiveVault"))
    from stolab_data.core_tickers import TICKER_LIST
    from stolab_data.extend_tickers import EXTEND_LIST
    tk = [t for t, _, s_ in list(TICKER_LIST) + list(EXTEND_LIST) if s_ != "ETF"]
    px = pd.read_parquet(bt.DATA / "prices.parquet")
    px = px[px.index.get_level_values("ticker").isin(tk)]
    px = px[(px.index.get_level_values("date") >= pd.Timestamp("2013-01-01")) & (px.index.get_level_values("date") <= pd.Timestamp(end))]
    wide = {c: px[c].astype("float64").unstack("ticker").sort_index() for c in ("open", "high", "low", "close", "volume")}
    for c in ("open", "high", "low", "volume"): wide[c] = wide[c].where(wide[c] > 0)
    dates, tickers = wide["close"].index, wide["close"].columns
    elig = wide["close"].notna()
    yz = bt._yang_zhang(wide["open"], wide["high"], wide["low"], wide["close"], 20)
    scale = pd.DataFrame(1.0, index=dates, columns=tickers)
    names = {t: n for t, n, _ in list(TICKER_LIST) + list(EXTEND_LIST)}
    return bt.Panel(wide["open"], wide["high"], wide["low"], wide["close"], wide["volume"], elig, yz, scale, dates, tickers, names)

def load():
    up, vp = bt.UniverseParams(), bt.VolScaleParams()
    p = load_day_panel(END) if UNIV == "day" else bt.load_panel(up, vp, end=END)
    sig = bt.compute_signals(p, bt.EntryParams())["signal"].fillna(False)
    flt = sr.compute_entry_filter(p.open, p.high, p.low, p.close).excluded.reindex_like(sig).fillna(False)
    s = sig & ~flt
    s = s.loc[START:END]
    # ATR14 (단순평균 TR) — 신호일 값
    pc = p.close.shift(1)
    tr = pd.concat([(p.high - p.low), (p.high - pc).abs(), (p.low - pc).abs()]).groupby(level=0).max()
    atr = tr.rolling(14, min_periods=14).mean()
    return p, s, sig.loc[START:END], flt.loc[START:END], atr

def signals_list(p, s):
    di = {d: i for i, d in enumerate(p.dates)}
    out = []
    for j, t in enumerate(p.tickers):
        for d in s.index[s[t].to_numpy()]:
            out.append((di[d], j, t))
    return out

# ── 규칙: 함수(i_entry 기준 k일째, entry, peak_close, close_k, atr0) → 선 ──
def rule_current(k, entry, peak, atr0):           # 현행 ③ (변동배율 끔): D+0~8 평단−20% / D+9~ peak>평단 → peak−5%, 아니면 평단−10%
    if k <= 8: return entry * 0.80
    return peak * 0.95 if peak > entry else entry * 0.90
def mk_fixed(x): return lambda k, e, pk, a: e * (1 - x)
def mk_trail(x): return lambda k, e, pk, a: pk * (1 - x)
def mk_atr(m): return lambda k, e, pk, a: e - m * a if np.isfinite(a) else -np.inf

RULES = {"현행③": rule_current,
         "고정−5%": mk_fixed(.05), "고정−10%": mk_fixed(.10), "고정−15%": mk_fixed(.15),
         "고정−20%": mk_fixed(.20), "고정−30%": mk_fixed(.30),
         "ATR×2": mk_atr(2), "ATR×3": mk_atr(3),
         "추적−10%": mk_trail(.10), "추적−20%": mk_trail(.20)}

def path_outcomes(p, atr, sl):
    op = p.open.to_numpy(); cl = p.close.to_numpy(); lo = p.low.to_numpy()
    at = atr.to_numpy(); n = len(p.dates); Hmax = max(HORIZONS)
    rows = []
    for (it, j, tk) in sl:
        e = it + 1
        if e >= n or not np.isfinite(op[e, j]) or op[e, j] <= 0: continue
        entry = op[e, j]; cost_in = entry * (1 + BUY_FEE); a0 = at[it, j]
        r = {"date": p.dates[it], "ticker": tk, "name": p.names.get(tk, ""), "entry": entry}
        last = min(n - 1, e + Hmax)
        c = cl[e:last + 1, j].copy()
        # 종가 결측(정지)은 직전값 이월
        for q in range(1, len(c)):
            if not np.isfinite(c[q]): c[q] = c[q - 1]
        for H in HORIZONS:
            if e + H > n - 1: r[f"hold{H}"] = np.nan; continue
            r[f"hold{H}"] = c[H] * (1 - SELL_FEE - TAX) / cost_in - 1
            seg = lo[e:e + H + 1, j]
            r[f"mae{H}"] = np.nanmin(seg) / entry - 1
        # 규칙별: 첫 트리거 k (종가 판정) → k+1 시가 체결
        for name, f in RULES.items():
            peak = c[0]; trig = None
            for k in range(0, len(c) - 1):
                peak = max(peak, c[k])
                if c[k] <= f(k, entry, peak, a0):
                    trig = k; break
            for H in HORIZONS:
                if e + H > n - 1: r[f"{name}|{H}"] = np.nan; continue
                if trig is not None and trig + 1 <= H:
                    fo = op[e + trig + 1, j]
                    fo = fo if np.isfinite(fo) and fo > 0 else c[trig + 1]
                    r[f"{name}|{H}"] = fo * (1 - SLIP) * (1 - SELL_FEE - TAX) / cost_in - 1
                    r[f"{name}|{H}|stopped"] = 1; r[f"{name}|{H}|day"] = trig + 1
                else:
                    r[f"{name}|{H}"] = r[f"hold{H}"]; r[f"{name}|{H}|stopped"] = 0; r[f"{name}|{H}|day"] = np.nan
        rows.append(r)
    return pd.DataFrame(rows)

def q(v, x): return np.nanquantile(v, x)

def report_signal(df):
    pd.set_option("display.width", 240)
    print(f"신호 {len(df)}건  ({df.date.min().date()}~{df.date.max().date()}), 종목 {df.ticker.nunique()}개")
    print("연도별:", df.date.dt.year.value_counts().sort_index().to_dict())
    for H in HORIZONS:
        h = df[f"hold{H}"].to_numpy(); ok = np.isfinite(h)
        print(f"\n== 보유 {H}거래일 끝 (n={ok.sum()}) — 평균 / 중앙 / 하위5% / 하위1% / 최악 / 플러스 / 손절률 / 손절이 버팀보다 나은 비율 / 평균−버팀 ==")
        def line(lbl, v, st=None):
            v = v[ok]
            s = (f"{lbl:<8} {np.mean(v)*100:+6.2f}% {np.median(v)*100:+6.2f}% {q(v,.05)*100:+7.2f}% {q(v,.01)*100:+7.2f}% "
                 f"{np.min(v)*100:+7.2f}% {np.mean(v>0)*100:5.1f}%")
            if st is not None:
                s += f"  손절 {np.mean(st[ok])*100:5.1f}%  나음 {np.mean(v > h[ok])*100:5.1f}%  차 {np.mean(v-h[ok])*100:+5.2f}%p"
            print(s)
        line("버팀", h)
        for name in RULES:
            line(name, df[f"{name}|{H}"].to_numpy(), df[f"{name}|{H}|stopped"].to_numpy())
    # 연도 구간별 (60일)
    H = 60; df["per"] = pd.cut(df.date.dt.year, [2014, 2017, 2019, 2021, 2023, 2026], labels=["15-17", "18-19", "20-21", "22-23", "24-26"])
    print(f"\n== 구간별 평균 (보유 {H}일) — 버팀 vs 규칙 ==")
    t = df.groupby("per", observed=True)[[f"hold{H}"] + [f"{n}|{H}" for n in RULES]].mean() * 100
    t.columns = ["버팀"] + list(RULES); t.insert(0, "n", df.groupby("per", observed=True).size())
    print(t.round(2).to_string())
    print(f"\n== 구간별 하위5% (보유 {H}일) ==")
    t = df.groupby("per", observed=True)[[f"hold{H}"] + [f"{n}|{H}" for n in RULES]].quantile(.05) * 100
    t.columns = ["버팀"] + list(RULES); print(t.round(1).to_string())

def mech(p, df_sig, sl):
    """진입 후 −L% (종가) 에 처음 닿은 날부터 이후 수익. 비교: 같은 기간 유니버스 전 종목·전 일자."""
    cl = p.close.to_numpy(); n = len(p.dates)
    LV = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30); FW = (20, 40, 60)
    # 같은 날 시작한 유니버스 동일가중 f일 수익 (시장 드리프트 제거용)
    EWF = {f: (p.close.shift(-f) / p.close - 1).where(p.elig).mean(axis=1).to_numpy() for f in FW}
    rows = []
    for (it, j, tk) in sl:
        e = it + 1
        if e >= n or not np.isfinite(p.open.iat[e, j]): continue
        entry = p.open.iat[e, j]
        c = cl[e:min(n, e + 61), j]
        for L in LV:
            hit = np.flatnonzero(c / entry - 1 <= -L)
            if not hit.size: continue
            k = e + hit[0]; r = {"L": L, "day": hit[0], "date": p.dates[it]}
            for f in FW:
                r[f"f{f}"] = cl[k + f, j] / cl[k, j] - 1 if k + f < n else np.nan
                r[f"x{f}"] = r[f"f{f}"] - EWF[f][k] if k + f < n else np.nan   # 유니버스 동일가중 대비 초과
                r[f"dd{f}"] = np.nanmin(cl[k + 1:k + f + 1, j]) / cl[k, j] - 1 if k + f < n else np.nan
            rows.append(r)
    m = pd.DataFrame(rows)
    # 기준: 유니버스 편입 종목·일자 전부 (같은 기간)
    el = p.elig.loc[START:END]; C = p.close
    base = {}
    for f in FW:
        fr = (C.shift(-f) / C - 1).loc[START:END].where(el)
        v = fr.to_numpy().ravel(); v = v[np.isfinite(v)]
        base[f] = (np.mean(v), np.median(v), np.mean(v > 0))
    # 기준2: 스윙 신호 진입일(손실 조건 없음)
    sbase = {}
    for f in FW:
        v = []
        for (it, j, tk) in sl:
            e = it + 1
            if e + f < n and np.isfinite(p.open.iat[e, j]): v.append(cl[e + f, j] / p.open.iat[e, j] - 1)
        v = np.array(v); sbase[f] = (np.mean(v), np.median(v), np.mean(v > 0))
    pd.set_option("display.width", 240)
    print("== 비교 기준 ==")
    for f in FW:
        print(f"  {f}일: 유니버스 전 종목·일자 평균 {base[f][0]*100:+.2f}% 중앙 {base[f][1]*100:+.2f}% 플러스 {base[f][2]*100:.1f}%"
              f"   |  스윙 진입 직후 평균 {sbase[f][0]*100:+.2f}% 중앙 {sbase[f][1]*100:+.2f}% 플러스 {sbase[f][2]*100:.1f}%")
    print("\n== 진입 후 −L% (종가, 60일 안) 에 처음 닿은 날부터의 수익 ==")
    g = m.groupby("L")
    t = pd.DataFrame({"건수": g.size(), "도달일_중앙": g.day.median()})
    for f in FW:
        t[f"{f}일_평균"] = g[f"f{f}"].mean() * 100; t[f"{f}일_중앙"] = g[f"f{f}"].median() * 100
        t[f"{f}일_플러스"] = g[f"f{f}"].apply(lambda v: (v.dropna() > 0).mean() * 100)
        t[f"{f}일_추가−10%↓"] = g[f"dd{f}"].apply(lambda v: (v.dropna() <= -0.10).mean() * 100)
    t.index = [f"−{int(L*100)}%" for L in t.index]
    print(t.round(2).to_string())
    print("\n== 시장 드리프트 제거: 같은 날 시작한 유니버스 동일가중 대비 초과수익 ==")
    x = pd.DataFrame({f"{f}일_초과평균": g[f"x{f}"].mean() * 100 for f in FW})
    for f in FW: x[f"{f}일_초과중앙"] = g[f"x{f}"].median() * 100; x[f"{f}일_초과플러스"] = g[f"x{f}"].apply(lambda v: (v.dropna() > 0).mean() * 100)
    x.index = t.index; print(x.round(2).to_string())
    # 비교: 스윙 진입일 기준 초과수익
    xs = {f: [] for f in FW}
    for (it, j, tk) in sl:
        e = it + 1
        for f in FW:
            if e + f < n and np.isfinite(p.open.iat[e, j]) and np.isfinite(cl[e + f, j]):
                xs[f].append(cl[e + f, j] / cl[e, j] - 1 - EWF[f][e])
    print("  (비교) 스윙 진입일부터 초과: " + "  ".join(f"{f}일 평균 {np.mean(v)*100:+.2f}% 중앙 {np.median(v)*100:+.2f}%" for f, v in xs.items()))
    m["per"] = pd.cut(m.date.dt.year, [2014, 2017, 2019, 2021, 2023, 2026], labels=["15-17", "18-19", "20-21", "22-23", "24-26"])
    print("\n== 구간별 — 진입 후 −10% / −20% 도달 뒤 20·60일 초과수익 평균 (건수) ==")
    for L in (0.10, 0.20):
        mm = m[m.L == L]; gg = mm.groupby("per", observed=True)
        print(f"  −{int(L*100)}%: " + " | ".join(f"{k}: 20일 {r.x20*100:+.2f}% / 60일 {r.x60*100:+.2f}% ({int(r.n)})"
              for k, r in gg.agg(x20=("x20", "mean"), x60=("x60", "mean"), n=("x20", "size")).iterrows()))
    m.to_csv(OUT / "mech_events.csv", index=False)
    return m

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "signal"
    p, s, sig_raw, flt, atr = load()
    sl = signals_list(p, s)
    print(f"[유니버스 {UNIV}] 종목 {len(p.tickers)}")
    print(f"[신호] W1 onset {int(sig_raw.to_numpy().sum())}건 → R²·YZR 제외 {int((sig_raw & flt).to_numpy().sum())}건 → 최종 {len(sl)}건")
    if mode == "signal":
        df = path_outcomes(p, atr, sl); df.to_csv(OUT / "signal_paths.csv", index=False); report_signal(df)
    elif mode == "mech":
        mech(p, None, sl)
