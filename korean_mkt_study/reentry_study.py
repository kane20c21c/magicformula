# -*- coding: utf-8 -*-
"""reentry_study.py — 손절 후 재진입 백테스트 (케인 2026-11-04 합의, 11-05 실행).

데이터  LLV core.parquet (ETF 2종 제외 84종목, 2023-01-02~). panel 아님 (케인 합의).
정의 (2026-11-04 노트 'B. 손절 후 재진입 백테스트' [합의]):
  스윙 저점   앞뒤 3거래일(K=3) 중 최저가. t+3 거래일 종가 이후에만 확정 (미래 참조 금지).
  손절 이벤트 종가가 직전(확정된) 스윙 저점 아래로 마감한 날. 손절선 = 그 스윙 저점.
              손절 체결가 = 이벤트 다음 날 시가 (K1: 종가 확인 + 익일 시가).
  탐색 창     이벤트 다음 날부터 60거래일 (W=60).
  기본 신호   종가 > 손절선 인 첫날 → 그날 종가 매수.
  조건 A      종가 > 손절선 + 거래량 ≥ 전일까지 20일 평균×1.5 + 당일 범위 ≥ 전일 ATR14
              + 종가가 당일 범위 상단 30% 안  인 첫날.
  조건 B      종가 > 손절선 + 현재 상승 파동 효율(|변화율|÷누적거래량) > 직전 하락 파동 효율 인 첫날.
              파동 = LLV Weis(종가 zigzag, ATR14×1.5 반전, 인과적).
클로이가 정한 것 (합의 밖, 결과에 표시):
  · 같은 종목의 이벤트는 겹치지 않는다 — 기본 신호 재매수 또는 창 종료 전엔 새 이벤트 없음.
  · 같은 스윙 저점으로는 이벤트 1회.
  · 거래비용 미반영.
사용: python3 reentry_study.py [core|panel]   (panel = 전체에서 ETF + 삼전·삼전우·하이닉스·SK스퀘어 제외)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
LLVROOT = HERE.parents[1] / "LongLiveVault"
OUT = HERE / "out" / "reentry"; OUT.mkdir(parents=True, exist_ok=True)
K, W = 3, 60
HOR = (5, 10, 20)
sys.path.insert(0, str(LLVROOT))

EXCLUDE = {"005930", "005935", "000660", "402340"}   # 삼성전자·삼성전자우·SK하이닉스·SK스퀘어 (케인 11-05, panel 실행 시)

def load(source="core"):
    from stolab_data.core_tickers import TICKER_LIST
    from stolab_data.extend_tickers import EXTEND_LIST
    etf = {t for t, _, s in list(TICKER_LIST) + list(EXTEND_LIST) if s == "ETF"}
    if source == "panel":
        etf = etf | EXCLUDE
    cols = ["Date", "Ticker", "Name", "Open", "High", "Low", "Close", "Volume", "ATR_14", "Weis_Dir", "Weis_Vol", "Weis_Chg"]
    d = pd.read_parquet(LLVROOT / "data" / "ohlcv" / f"{source}.parquet", columns=cols)
    d = d[~d.Ticker.isin(etf)].sort_values(["Ticker", "Date"]).reset_index(drop=True)
    g = d.groupby("Ticker")
    d["vma20_prev"] = g.Volume.transform(lambda s: s.astype(float).rolling(20, min_periods=20).mean().shift(1))
    d["atr_prev"] = g.ATR_14.shift(1)
    return d

def weis_eff(x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """각 날짜의 (현재 파동 효율, 직전 반대방향 파동의 최종 효율). 효율 = |Weis_Chg| / Weis_Vol"""
    dirs = x.Weis_Dir.to_numpy(); eff = (x.Weis_Chg.abs() / x.Weis_Vol.replace(0, np.nan)).to_numpy()
    prev_down = np.full(len(x), np.nan); last_down_final = np.nan
    for i in range(len(x)):
        if i > 0 and dirs[i - 1] == -1 and dirs[i] != -1:
            last_down_final = eff[i - 1]
        prev_down[i] = last_down_final
    return eff, prev_down

def run_ticker(x: pd.DataFrame):
    x = x.reset_index(drop=True)
    n = len(x); lo = x.Low.to_numpy(float); cl = x.Close.to_numpy(float); hi = x.High.to_numpy(float)
    op = x.Open.to_numpy(float); vol = x.Volume.to_numpy(float)
    vma = x.vma20_prev.to_numpy(float); atr = x.atr_prev.to_numpy(float); wdir = x.Weis_Dir.to_numpy(float)
    eff, prev_down = weis_eff(x)
    swing = np.zeros(n, bool)
    for t in range(K, n - K):
        if lo[t] == lo[t - K:t + K + 1].min() and vol[t] > 0:
            swing[t] = True
    rng = hi - lo
    pos = np.where(rng > 0, (cl - lo) / np.where(rng > 0, rng, 1), np.nan)
    cA_vol = vol >= 1.5 * vma
    cA_rng = rng >= atr
    cA_pos = pos >= 0.7
    cB = (wdir == 1) & (eff > prev_down)
    events = []; used = set(); blocked_until = -1; latest = -1
    for d in range(1, n):
        t_new = d - 1 - K                     # d-1 종가 기준으로 확정된 가장 최근 스윙
        if t_new >= 0 and swing[t_new]:
            latest = t_new
        if d <= blocked_until or latest < 0 or latest in used:
            continue
        lvl = lo[latest]
        if cl[d] < lvl and cl[d - 1] >= lvl and d + 1 < n:
            used.add(latest)
            ev = {"ticker": x.Ticker[0], "name": x.Name.iloc[-1], "swing_date": x.Date[latest], "event_date": x.Date[d],
                  "level": lvl, "event_close": cl[d], "stop_fill": op[d + 1]}
            win = range(d + 1, min(n, d + 1 + W))
            ev["window_full"] = d + W < n
            masks = {"base": cl > lvl, "A": (cl > lvl) & cA_vol & cA_rng & cA_pos, "B": (cl > lvl) & cB,
                     "A_no_vol": (cl > lvl) & cA_rng & cA_pos, "A_no_rng": (cl > lvl) & cA_vol & cA_pos,
                     "A_no_pos": (cl > lvl) & cA_vol & cA_rng}
            base_day = None
            for k, m in masks.items():
                hit = next((i for i in win if m[i]), None)
                ev[f"{k}_day"] = x.Date[hit] if hit is not None else pd.NaT
                ev[f"{k}_wait"] = (hit - d) if hit is not None else np.nan
                ev[f"{k}_px"] = cl[hit] if hit is not None else np.nan
                for h in HOR:
                    ev[f"{k}_r{h}"] = cl[hit + h] / cl[hit] - 1 if hit is not None and hit + h < n else np.nan
                ev[f"{k}_mae20"] = lo[hit + 1:hit + 21].min() / cl[hit] - 1 if hit is not None and hit + 20 < n else np.nan
                if k == "base": base_day = hit
            # 손절 안 하고 버텼다면: 이벤트일 종가 → +20
            ev["hold_r20"] = cl[d + 20] / cl[d] - 1 if d + 20 < n else np.nan
            # 경로 비교 (H=60): 버팀 = 이벤트일 종가→d+60 종가 / 손절·재매수 = 다음날 시가 매도 → 신호일 종가 재매수 → d+60 종가
            #   재매수 신호가 창 안에 없으면 현금(손절 체결가 그대로)
            ev["c_d60"] = cl[d + W] if d + W < n else np.nan
            events.append(ev)
            blocked_until = base_day if base_day is not None else d + W
    return events

def summarize(ev: pd.DataFrame, k: str, base_n: int) -> dict:
    s = ev[ev[f"{k}_day"].notna()]
    r20 = s[f"{k}_r20"].dropna()
    return {"신호": len(s), "발생률": len(s) / base_n,
            "재매수가_손절선대비_중앙": (s[f"{k}_px"] / s.level - 1).median(),
            "재진입프리미엄_중앙(손절체결가대비)": (s[f"{k}_px"] / s.stop_fill - 1).median(),
            "대기일_중앙": s[f"{k}_wait"].median(),
            "r5_중앙": s[f"{k}_r5"].median(), "r10_중앙": s[f"{k}_r10"].median(),
            "r20_평균": r20.mean(), "r20_중앙": r20.median(), "승률20": (r20 > 0).mean(), "r20_n": len(r20),
            "승률5": (s[f"{k}_r5"].dropna() > 0).mean(),
            "MAE20_중앙": s[f"{k}_mae20"].median()}

def main(source="core"):
    global OUT
    d = load(source)
    if source != "core":
        OUT = OUT / source; OUT.mkdir(parents=True, exist_ok=True)
    allev = []
    for _, x in d.groupby("Ticker", sort=False):
        allev += run_ticker(x)
    ev = pd.DataFrame(allev)
    ev.to_csv(OUT / "events.csv", index=False)
    N = len(ev)
    # 기준선: 전 종목·전 일자 20일 수익률
    d["r20"] = d.groupby("Ticker").Close.transform(lambda s: s.shift(-20) / s - 1)
    pd.set_option("display.width", 220); pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    print(f"== 데이터 == 종목 {d.Ticker.nunique()}  기간 {d.Date.min().date()}~{d.Date.max().date()}  봉 {len(d):,}")
    print(f"== 손절 이벤트 == {N}건  (종목당 중앙 {ev.groupby('ticker').size().median():.0f}건, 60일 창이 다 찬 이벤트 {ev.window_full.sum()}건)")
    print("연도별:", ev.event_date.dt.year.value_counts().sort_index().to_dict())
    full = ev[ev.window_full]
    print(f"60일 안 미회복 비율 (창이 찬 이벤트 기준): 기본 {full.base_day.isna().mean():.1%}  A {full.A_day.isna().mean():.1%}  B {full.B_day.isna().mean():.1%}")
    tab = pd.DataFrame({k: summarize(ev, k, N) for k in ["base", "A", "B"]}).T
    print("\n== 비교표 ==\n", tab.to_string())
    print(f"\n기준선 — 손절 안 하고 버텼다면(이벤트일 종가→20일): 평균 {ev.hold_r20.mean():.3f} 중앙 {ev.hold_r20.median():.3f} 승률 {(ev.hold_r20.dropna()>0).mean():.3f} (n={ev.hold_r20.notna().sum()})")
    print(f"기준선 — 아무 종목·아무 날 20일: 평균 {d.r20.mean():.3f} 중앙 {d.r20.median():.3f} 승률 {(d.r20.dropna()>0).mean():.3f}")
    print("\n== 경로 비교 — 이벤트 뒤 60거래일 끝에서 (이벤트일 종가 기준 수익률) ==")
    e = ev[ev.c_d60.notna()].copy()
    hold = e.c_d60 / e.event_close - 1
    rows = {"버팀(손절 안 함)": hold, "손절만(재매수 없음)": e.stop_fill / e.event_close - 1}
    for k in ["base", "A", "B"]:
        re_ = e[f"{k}_px"].notna()
        rows[f"손절→{k} 재매수"] = np.where(re_, e.stop_fill / e[f"{k}_px"] * e.c_d60 / e.event_close, e.stop_fill / e.event_close) - 1
    pt = pd.DataFrame({k: {"평균": np.mean(v), "중앙": np.median(v), "플러스비율": np.mean(np.asarray(v) > 0),
                           "버팀보다 나은 비율": np.mean(np.asarray(v) > hold.to_numpy())} for k, v in rows.items()}).T
    print(f"n={len(e)}"); print(pt.to_string())
    for y, g in e.groupby(e.event_date.dt.year):
        h = g.c_d60 / g.event_close - 1
        b = np.where(g.base_px.notna(), g.stop_fill / g.base_px * g.c_d60 / g.event_close, g.stop_fill / g.event_close) - 1
        print(f"  {y}: 버팀 중앙 {h.median():+.3f} / 손절→기본재매수 중앙 {np.median(b):+.3f}  (n={len(g)})")

    print("\n== 연도별 (신호일 기준 연도, r20 중앙 / 승률 / 건수) ==")
    for k in ["base", "A", "B"]:
        s = ev[ev[f"{k}_day"].notna()].copy(); s["y"] = s[f"{k}_day"].dt.year
        y = s.groupby("y")[f"{k}_r20"].agg(med="median", win=lambda v: (v.dropna() > 0).mean(), n="count")
        print(f"[{k}]", " | ".join(f"{i}: {r.med:+.3f} / {r.win:.0%} / {int(r.n)}" for i, r in y.iterrows()))
    print("\n== 종목 쏠림 (신호 상위 5종목 비중) ==")
    for k in ["base", "A", "B"]:
        s = ev[ev[f"{k}_day"].notna()]; c = s.ticker.value_counts()
        print(f"[{k}] 종목 {c.size}개, 상위5 비중 {c.head(5).sum()/c.sum():.1%}  ", dict(c.head(5)))
    print("\n== 조건 A 하위조건 기여도 (하나씩 뺐을 때) ==")
    sub = pd.DataFrame({k: summarize(ev, k, N) for k in ["A", "A_no_vol", "A_no_rng", "A_no_pos"]}).T
    print(sub[["신호", "대기일_중앙", "재매수가_손절선대비_중앙", "r20_중앙", "승률20", "MAE20_중앙"]].to_string())

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "core")   # python3 reentry_study.py [core|panel]
