# -*- coding: utf-8 -*-
"""swing_shock_gates.py — 스윙포트 진입 게이트(쇼크 차단) + 손절 유예 백테스트 (케인 2026-11-07 22:22 확정 정의).

엔진   stop_port.run_port (스윙 라이브 방식: W1 + R²·YZR 제외 신호, t+1 시가, 10슬롯, 현행③ 4분기 손절 종가 판정 → 익일 시가×99.5%)
       유니버스 4조·외인 30% 월별 (backtest.UniverseParams — 3번째의 '최소 30종목'은 미반영)
게이트 (신호일 t 포함 최근 10거래일 안에 그 종목 쇼크 태그가 있으면 신호 버림 — shock_tags.py, 인과)
  G1 단독 쇼크 전부 · G2 큰 단독(≤−10%) · G3 저거래량 단독(<1배, 케인 지정) · G4 고거래량 단독(≥2배, 비교)
  G5 섹터 동반 + ① 갭상승→붕괴 · G6 = G1 + G5          (민감도: 창 5·20거래일)
손절 유예 S1  판정일 코스피200 하루 수익이 −5% 초과 ~ −2% 이하(작은·중간 시장 쇼크)면 그날 종가 손절 판정을 건너뜀 (종목 조건 없음 — 케인)
평가   학습 2015-01-02~2026-05-29 / 시기 15–19·20–22·23–26.5 / 보류 2026-06-01~11-04 — 수익·CAGR·Sharpe·MDD·투자비중·청산수·손절비중
       차단된 신호 자체의 이후 성과: t+1 시가 → +10·+20일 종가 (왕복 0.23% 차감), 코스피200 대비 초과
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
os.environ.setdefault("STOP_UNIV", "swing")
import stop_research as sr, stop_port as sp
OUT = HERE / "out" / "regime"
PER = [("2015-01-01", "2026-05-29", "학습 전체"), ("2015-01-01", "2019-12-31", "15–19"), ("2020-01-01", "2022-12-31", "20–22"),
       ("2023-01-01", "2026-05-29", "23–26.5"), ("2026-06-01", "2026-11-04", "보류")]

def gate_masks(p, win=10):
    t = pd.read_parquet(OUT / "shock_tags.parquet"); t = t[t.ticker.isin(p.tickers)]
    conds = {"G1 단독": t.ctx == "단독", "G2 큰 단독": (t.ctx == "단독") & (t["size"] == "큰"),
             "G3 저거래량 단독": (t.ctx == "단독") & (t.volbin == "저"), "G4 고거래량 단독": (t.ctx == "단독") & (t.volbin == "고"),
             "G5 섹터동반 ①": (t.ctx == "섹터 동반") & (t["shape"] == "①")}
    out = {}
    for nm, c in conds.items():
        x = t[c]; w = pd.DataFrame(False, index=p.dates, columns=p.tickers)
        w = w.astype(float); w.values[w.index.get_indexer(x.date), w.columns.get_indexer(x.ticker)] = 1.0
        out[nm] = w.rolling(win, min_periods=1).max() > 0
    out["G6 단독 + 섹터동반 ①"] = out["G1 단독"] | out["G5 섹터동반 ①"]
    return out

def perf(eq, tr):
    rows = {}
    for a, b, lbl in PER:
        e = eq.equity.loc[a:b]
        if len(e) < 2: continue
        r = e.pct_change().dropna(); yrs = (e.index[-1] - e.index[0]).days / 365.25
        s = tr[(tr.side == "SELL") & (tr.date >= a) & (tr.date <= b)] if len(tr) else tr
        rows[lbl] = dict(수익=(e.iloc[-1] / e.iloc[0] - 1) * 100, CAGR=((e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1) * 100 if yrs > 0.5 else np.nan,
                         Sharpe=r.mean() / r.std() * np.sqrt(252), MDD=(e / e.cummax() - 1).min() * 100,
                         투자비중=(eq.mv.loc[a:b] / e).mean() * 100, 청산=len(s), 승률=(s.ret > 0).mean() * 100 if len(s) else np.nan,
                         손절비중=(s.reason == "stop").mean() * 100 if len(s) else np.nan)
    return rows

def sig_outcome(p, s, mask):
    """신호(t) → t+1 시가 진입 +10·+20일 종가 수익(비용 차감)·코스피200 초과."""
    k = pd.read_parquet(HERE / "data_1104" / "kospi200_daily.parquet").Close.astype(float); k.index = pd.to_datetime(k.index); k = k.reindex(p.dates)
    op, cl = p.open.to_numpy(), p.close.to_numpy(); kc = k.to_numpy(); n = len(p.dates); rows = []
    S = s.reindex(index=p.dates, columns=p.tickers).fillna(False).to_numpy(); M = mask.reindex(index=p.dates, columns=p.tickers).fillna(False).to_numpy()
    for i, j in zip(*np.nonzero(S)):
        if i + 21 >= n or not np.isfinite(op[i + 1, j]) or op[i + 1, j] <= 0: continue
        e = op[i + 1, j]; rows.append(dict(date=p.dates[i], blocked=bool(M[i, j]),
            a10=cl[i + 11, j] / e - 1 - 0.0023, a20=cl[i + 21, j] / e - 1 - 0.0023, k20=kc[i + 21] / kc[i + 1] - 1))
    d = pd.DataFrame(rows); d["x20"] = d.a20 - d.k20; return d

def main():
    p, s, _, _, atr = sr.load()
    k = pd.read_parquet(HERE / "data_1104" / "kospi200_daily.parquet").Close.astype(float); k.index = pd.to_datetime(k.index); kr = k.pct_change()
    skip = ((kr <= -0.02) & (kr > -0.05)).reindex(p.dates).fillna(False)
    print(f"신호 {int(s.to_numpy().sum())} · 종목 {len(p.tickers)} · 손절 유예 판정일 {int(skip.sum())}일 (학습 {int(skip.loc[:'2026-05-29'].sum())} · 보류 {int(skip.loc['2026-06-01':].sum())})")
    runs = [("기준 (현행)", s, None)]
    M10 = gate_masks(p, 10)
    for nm, m in M10.items(): runs.append((nm, s & ~m.reindex_like(s).fillna(False), None))
    runs.append(("S1 손절 유예", s, skip))
    runs.append(("G1 단독 + S1", s & ~M10["G1 단독"].reindex_like(s).fillna(False), skip))
    runs.append(("G3 저거래량 단독 + S1", s & ~M10["G3 저거래량 단독"].reindex_like(s).fillna(False), skip))
    for win in (5, 20):
        Mw = gate_masks(p, win)
        for nm in ("G1 단독", "G3 저거래량 단독", "G6 단독 + 섹터동반 ①"): runs.append((f"{nm} (창 {win})", s & ~Mw[nm].reindex_like(s).fillna(False), None))
    rows, blk = [], []
    for lbl, sig, sk in runs:
        eq, tr = sp.run_port(p, sig, atr, sr.rule_current, None, start="2015-01-01", end="2026-11-04", skip=sk)
        for per, m in perf(eq, tr).items(): rows.append({"변형": lbl, "구간": per, **m})
        nb = int((s & ~sig).to_numpy().sum()); rows[-1]["차단 신호"] = nb
        print(f"  {lbl}: 차단 {nb}", flush=True)
    T = pd.DataFrame(rows)
    for lbl, sig, sk in runs[1:7]:
        m = (s & ~sig); d = sig_outcome(p, s, m)
        for per_a, per_b, pl in (PER[0], PER[4]):
            x = d[(d.date >= per_a) & (d.date <= per_b)]
            for b, g in x.groupby("blocked"):
                blk.append({"게이트": lbl, "구간": pl, "신호": "차단됨" if b else "통과", "건수": len(g), "날짜": g.date.nunique(),
                            "10일 중앙%": g.a10.median() * 100, "20일 중앙%": g.a20.median() * 100, "20일 평균%": g.a20.mean() * 100,
                            "20일 승률%": (g.a20 > 0).mean() * 100, "20일 초과 중앙%": g.x20.median() * 100})
    B = pd.DataFrame(blk)
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300)
    piv = T.pivot_table(index="변형", columns="구간", values=["수익", "MDD", "Sharpe"], aggfunc="first", sort=False)
    print(piv.round(2).to_string())
    print(T[T.구간 == "학습 전체"][["변형", "CAGR", "Sharpe", "MDD", "투자비중", "청산", "승률", "손절비중", "차단 신호"]].round(2).to_string(index=False))
    print(B.round(1).to_string(index=False))
    T.to_csv(OUT / "swing_gates_perf.csv", index=False, encoding="utf-8-sig"); B.to_csv(OUT / "swing_gates_blocked.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
