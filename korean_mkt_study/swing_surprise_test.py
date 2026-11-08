# -*- coding: utf-8 -*-
"""swing_surprise_test.py — 급등 연구 6단계: 스윙포트 보유 규칙·진입 게이트 (케인 2026-11-08 20:06 '진행').
엔진  stop_port.run_port (라이브: W1 + R²·YZR 제외, t+1 시가, 10슬롯×1,000만, 현행③ 손절 종가 판정 → 익일 시가×99.5%)
태그  surprise_tags.py (인과, 주 정의 ≥ +6%)
보유 규칙 (보유 중 급등 τ: 진입일 ≤ τ, 판정일 − τ ≤ 20거래일)
  H1 단독·섹터 동반 급등 → 이후 종가 < 급등일 저가 면 익일 시가 정리 (매뉴얼 초안 — 시장 동반 제외)
  H2 모든 급등(시장 동반 포함) 저가 이탈 정리 (비교)
  H3 단독 큰 급등만 저가 이탈 정리
  H4 단독 급등 즉시 실현 (급등일 종가 판정 → 익일 시가 정리, 비교)
진입 게이트 (신호일 포함 최근 10거래일 안에 그 종목 급등 태그 → 신호 버림)
  E1 단독 급등 전부 · E2 단독 큰 · E3 단독 ⑤ 갭상승→밀림
평가  학습 2015-01-02~2026-04-30 · 시기 15–19 / 20–22 / 23–26.4 · 보류 2026-05-01~11-04"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
os.environ.setdefault("STOP_UNIV", "swing")
import stop_research as sr, stop_port as sp
from swing_shock_gates import sig_outcome
OUT = HERE / "out" / "regime"
PER = [("2015-01-01", "2026-04-30", "학습 전체"), ("2015-01-01", "2019-12-31", "15–19"), ("2020-01-01", "2022-12-31", "20–22"),
       ("2023-01-01", "2026-04-30", "23–26.4"), ("2026-05-01", "2026-11-04", "보류")]
def perf(eq, tr):
    rows = {}
    for a, b, lbl in PER:
        e = eq.equity.loc[a:b]
        if len(e) < 2: continue
        r = e.pct_change().dropna(); yrs = (e.index[-1] - e.index[0]).days / 365.25
        s = tr[(tr.side == "SELL") & (tr.date >= a) & (tr.date <= b)] if len(tr) else tr
        rows[lbl] = dict(수익=(e.iloc[-1] / e.iloc[0] - 1) * 100, CAGR=((e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1) * 100 if yrs > 0.5 else np.nan,
                         Sharpe=r.mean() / r.std() * np.sqrt(252), MDD=(e / e.cummax() - 1).min() * 100, 투자비중=(eq.mv.loc[a:b] / e).mean() * 100,
                         청산=len(s), 승률=(s.ret > 0).mean() * 100 if len(s) else np.nan, 손절비중=(s.reason == "stop").mean() * 100 if len(s) else np.nan,
                         급등청산=(s.reason == "hold_exit").sum() if len(s) else 0, 급등청산_평균=(s[s.reason == "hold_exit"].ret.mean() * 100) if len(s) else np.nan)
    return rows

def main():
    p, s, _, _, atr = sr.load()
    t = pd.read_parquet(OUT / "surprise_tags.parquet"); t = t[t.main & t.ticker.isin(p.tickers)]
    di = pd.Series(np.arange(len(p.dates)), index=p.dates); t = t[t.date.isin(p.dates)].copy(); t["i"] = t.date.map(di).astype(int)
    t["j"] = t.ticker.map({tk: j for j, tk in enumerate(p.tickers)}).astype(int)
    cl = p.close.reindex(index=p.dates, columns=p.tickers).to_numpy()
    print(f"신호 {int(s.to_numpy().sum())} · 종목 {len(p.tickers)} · 스윙 유니버스 급등(≥+6%) {len(t):,} (맥락 {t.ctx2.value_counts().to_dict()})", flush=True)
    def mk_hold(sel, immediate=False):
        ev = {}
        for r in t[sel].itertuples(): ev.setdefault(r.j, []).append((r.i, r.low))
        def f(j, first_i, ip):
            for (ti, lo) in ev.get(j, ()):
                if ti < first_i or ti > ip or ip - ti > 20: continue
                if immediate:
                    if ti == ip: return True
                elif ti < ip and np.isfinite(cl[ip, j]) and cl[ip, j] < lo: return True
            return False
        return f
    single = (t.ctx2 == "단독")
    holds = {"H1 단독·섹터 저가이탈 정리": mk_hold(single | (t.ctx2 == "섹터 동반")), "H2 모든 급등 저가이탈 정리": mk_hold(t.ctx2 != "미상"),
             "H3 단독 큰 저가이탈 정리": mk_hold(single & (t["size"] == "큰")), "H4 단독 즉시 실현": mk_hold(single, immediate=True)}
    def gmask(sel, win=10):
        w = pd.DataFrame(0.0, index=p.dates, columns=p.tickers); x = t[sel]
        w.values[x.i.to_numpy(), x.j.to_numpy()] = 1.0
        return (w.rolling(win, min_periods=1).max() > 0).reindex_like(s).fillna(False)
    gates = {"E1 단독 급등 진입 차단": gmask(single), "E2 단독 큰 진입 차단": gmask(single & (t["size"] == "큰")), "E3 단독 ⑤ 진입 차단": gmask(single & (t["shape"] == "⑤"))}
    runs = [("기준 (현행)", s, None)] + [(k, s, f) for k, f in holds.items()] + [(k, s & ~m, None) for k, m in gates.items()]
    rows = []
    for lbl, sig, he in runs:
        eq, tr = sp.run_port(p, sig, atr, sr.rule_current, None, start="2015-01-01", end="2026-11-04", hold_exit=he)
        for per, m in perf(eq, tr).items(): rows.append({"변형": lbl, "구간": per, **m, "차단 신호": int((s & ~sig).to_numpy().sum())})
        print(f"  {lbl} 완료", flush=True)
    T = pd.DataFrame(rows)
    blk = []
    for lbl, m in gates.items():
        d = sig_outcome(p, s, m)
        for a, b, pl in (PER[0], PER[4]):
            x = d[(d.date >= a) & (d.date <= b)]
            for bb, g in x.groupby("blocked"):
                blk.append({"게이트": lbl, "구간": pl, "신호": "차단됨" if bb else "통과", "건수": len(g), "20일 중앙%": g.a20.median() * 100,
                            "20일 평균%": g.a20.mean() * 100, "20일 승률%": (g.a20 > 0).mean() * 100, "20일 초과 중앙%": g.x20.median() * 100})
    B = pd.DataFrame(blk)
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300)
    print(T.pivot_table(index="변형", columns="구간", values=["수익", "Sharpe", "MDD"], aggfunc="first", sort=False).round(2).to_string())
    print(T[T.구간 == "학습 전체"][["변형", "CAGR", "Sharpe", "MDD", "투자비중", "청산", "승률", "손절비중", "급등청산", "급등청산_평균", "차단 신호"]].round(2).to_string(index=False))
    print(T[T.구간 == "보류"][["변형", "수익", "Sharpe", "MDD", "청산", "급등청산", "급등청산_평균"]].round(2).to_string(index=False))
    print(B.round(1).to_string(index=False))
    T.to_csv(OUT / "swing_surprise_perf.csv", index=False, encoding="utf-8-sig"); B.to_csv(OUT / "swing_surprise_blocked.csv", index=False, encoding="utf-8-sig")
    import surprise_xlsx
    piv = {k: T.pivot_table(index="변형", columns="구간", values=k, aggfunc="first", sort=False)[[x[2] for x in PER]] for k in ("수익", "Sharpe", "MDD")}
    ln = T[T.구간 == "학습 전체"].set_index("변형")[["CAGR", "Sharpe", "MDD", "투자비중", "청산", "승률", "손절비중", "급등청산", "급등청산_평균", "차단 신호"]]
    surprise_xlsx.write_generic(OUT / "급등연구_6단계_스윙포트.xlsx",
        [("급등 연구 6단계 — 스윙포트 보유 규칙·진입 게이트 (라이브 조건, 1억·10슬롯×1,000만)", True),
         ("H1 단독·섹터 급등 저가이탈 정리 (초안) · H2 모든 급등 · H3 단독 큰만 · H4 단독 즉시 실현 — 보유 중(진입일 이후) 급등, 20거래일 안 종가 < 급등일 저가 → 익일 시가×99.5%", False),
         ("E1~E3: 신호일 포함 최근 10거래일 안 단독 급등(전부 / 큰 / ⑤) 종목의 W1 신호 버림", False),
         ("학습 2015-01-02~2026-04-30 · 보류 2026-05-01~11-04 · 단위 %", False)],
        [("성과", [(f"구간별 {k} (%)" if k != "Sharpe" else "구간별 Sharpe", v, list(v.columns), False, None, ["변형"]) for k, v in piv.items()] +
                  [("학습 전체 상세", ln, ["CAGR", "MDD", "급등청산_평균"], False, None, ["변형"])]),
         ("차단 신호", [("차단된 신호 vs 통과 신호 (t+1 시가 → 20일)", B.set_index(["게이트", "구간", "신호"]), ["20일 중앙%", "20일 평균%", "20일 초과 중앙%"], False, None, ["게이트", "구간", "신호"])])])

if __name__ == "__main__":
    main()
