"""h3_inst_flow.py — 가설 3 검증 (2026-10-05, Kane): 기관 비중 증가가 수익률과 상관이 있다 → 쓸 수 있는 형태인가?

데이터  data/prices.parquet · investor_flow.parquet(금액, 2014~2026-06) · meta.parquet(월말 시총)
유니버스 전월 말 시총 상위 200 (우선주·스팩 제외). 보조로 스윙 유니버스(backtest elig)도 본다.
지표    기관강도_k = k일 기관 순매수 금액 합 ÷ k일 거래대금 합 (거래대금 ≈ 종가×거래량). 외국인도 같은 식(비교용).
검정    ① 시차 상관: 당일 기관강도 ↔ t+lag 일 수익률 (lag −10..+10), 날짜별 횡단면 순위상관의 평균·t
        ② 선행성: 기관강도_k(t) ↔ 이후 h일 초과수익, 겹치지 않는 날짜만. 과거 k일 수익률 통제(날짜별 회귀 계수 평균)
        ③ 지속성: 기관강도_5(t) ↔ 기관강도_5(t+5)
        ④ 5분위 상−하 이후 수익 (연도별·기간 A/B)
        ⑤ '확대 + 미반응'(10-05 보고서 후보): 기관강도_10 상위⅓ & 같은 10일 초과수익 하위⅓ → 이후 10일 초과수익
출력    out/h3/*.csv
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
DATA = HERE / "data"; OUT = HERE / "out" / "h3"; OUT.mkdir(parents=True, exist_ok=True)
SPLIT = pd.Timestamp("2022-01-01"); START = "2014-01-01"; TOPN = 200


def load():
    px = pd.read_parquet(DATA / "prices.parquet", columns=["close", "volume"])
    px = px[px.index.get_level_values("date") >= pd.Timestamp(START)]
    C = px["close"].astype("float64").unstack("ticker").sort_index(); V = px["volume"].astype("float64").unstack("ticker").reindex_like(C)
    C = C.where(C > 0); V = V.where(V > 0)                      # 거래정지 0값 → 결측
    fl = pd.read_parquet(DATA / "investor_flow.parquet")
    INS = fl["기관"].unstack("ticker").reindex_like(C); FOR = fl["외국인"].unstack("ticker").reindex_like(C)
    meta = pd.read_parquet(DATA / "meta.parquet")
    cap = meta["mktcap"].where(~meta["is_pref"].fillna(False).astype(bool) & ~meta["is_spac"].fillna(False).astype(bool))
    cap = cap.unstack("ticker").reindex(columns=C.columns)
    top = cap.rank(axis=1, ascending=False) <= TOPN
    top.index = top.index + pd.Timedelta(days=1)                 # 월말 값은 다음 달부터 사용
    U = top.reindex(C.index, method="ffill").fillna(False)
    return C, V, INS, FOR, U


def rowcorr(X: pd.DataFrame, Y: pd.DataFrame, min_n=30) -> pd.Series:
    """날짜별 횡단면 Spearman."""
    m = X.notna() & Y.notna()
    xr = X.where(m).rank(axis=1); yr = Y.where(m).rank(axis=1)
    xr = xr.sub(xr.mean(axis=1), axis=0); yr = yr.sub(yr.mean(axis=1), axis=0)
    r = (xr * yr).sum(axis=1) / np.sqrt((xr ** 2).sum(axis=1) * (yr ** 2).sum(axis=1))
    return r.where(m.sum(axis=1) >= min_n)


def tstat(s: pd.Series):
    s = s.dropna(); n = len(s)
    return (s.mean(), s.mean() / (s.std(ddof=1) / np.sqrt(n)) if n > 2 and s.std() > 0 else np.nan, n)


def strength(F, AMT, k):
    return F.rolling(k, min_periods=k).sum() / AMT.rolling(k, min_periods=k).sum()


def run(tag, C, V, INS, FOR, U):
    print(f"\n================ {tag} ================")
    R = C.pct_change(); AMT = C * V
    Cu = C.where(U); Ru = R.where(U); Rx = Ru.sub(Ru.mean(axis=1), axis=0)          # 유니버스 평균 대비 초과
    logC = np.log(C)
    print(f"기간 {C.index[0].date()} ~ {C.index[-1].date()} · 일평균 종목 수 {U.sum(axis=1).mean():.0f}")
    res = {}
    # ① 시차 상관 (당일 강도 ↔ t+lag 수익률)
    rows = []
    for nm, F in (("기관", INS), ("외국인", FOR)):
        S1 = (F / AMT).where(U)
        for lag in range(-10, 11):
            m, t, n = tstat(rowcorr(S1, Rx.shift(-lag)))
            rows.append(dict(주체=nm, lag=lag, r=m, t=t, n=n))
    d1 = pd.DataFrame(rows); d1.to_csv(OUT / f"{tag}_leadlag.csv", index=False, encoding="utf-8-sig")
    pv = d1.pivot(index="lag", columns="주체", values=["r", "t"]).round(3)
    print("\n① 시차 상관 (lag<0: 수익률이 먼저, lag>0: 수급이 먼저)"); print(pv.to_string())
    # ② 선행성 + 과거 수익률 통제, ④ 5분위
    rows = []; q_rows = []
    for nm, F in (("기관", INS), ("외국인", FOR)):
        for k in (5, 10, 20):
            S = strength(F, AMT, k).where(U); past = (logC - logC.shift(k)).where(U)
            pastx = past.sub(past.mean(axis=1), axis=0)
            same = tstat(rowcorr(S, pastx).iloc[::k])
            for h in (1, 5, 10, 20):
                fwd = (logC.shift(-h) - logC).where(U); fwdx = fwd.sub(fwd.mean(axis=1), axis=0)
                ic = rowcorr(S, fwdx).iloc[::h]
                # 날짜별 회귀: rank(fwd) ~ rank(S) + rank(past)
                coefs = []
                for dt in S.index[::h]:
                    x1 = S.loc[dt]; x2 = pastx.loc[dt]; y = fwdx.loc[dt]; mk = x1.notna() & x2.notna() & y.notna()
                    if mk.sum() < 30: continue
                    Z = np.c_[np.ones(mk.sum()), stats_rank(x1[mk]), stats_rank(x2[mk])]
                    b = np.linalg.lstsq(Z, stats_rank(y[mk]), rcond=None)[0]; coefs.append((dt, b[1], b[2]))
                cf = pd.DataFrame(coefs, columns=["date", "b_flow", "b_past"]).set_index("date")
                a = tstat(ic); A = tstat(ic[ic.index < SPLIT]); B = tstat(ic[ic.index >= SPLIT]); bf = tstat(cf.b_flow)
                rows.append(dict(주체=nm, k=k, h=h, 동시상관=same[0], IC=a[0], t=a[1], n=a[2], IC_A=A[0], t_A=A[1], IC_B=B[0], t_B=B[1],
                                 통제후=bf[0], 통제후_t=bf[1], 과거수익_계수=cf.b_past.mean()))
                if k == 10 and h in (5, 10, 20):
                    qs = S.rank(axis=1, pct=True)
                    sp = (fwdx.where(qs >= 0.8).mean(axis=1) - fwdx.where(qs <= 0.2).mean(axis=1)).iloc[::h]
                    a2 = tstat(sp); yr = sp.groupby(sp.index.year).mean()
                    q_rows.append(dict(주체=nm, h=h, 상하_스프레드=a2[0], t=a2[1], n=a2[2], A=sp[sp.index < SPLIT].mean(), B=sp[sp.index >= SPLIT].mean(),
                                       양수_해=int((yr > 0).sum()), 해=len(yr), **{str(y): v for y, v in yr.items()}))
    d2 = pd.DataFrame(rows); d2.to_csv(OUT / f"{tag}_predict.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print("\n② 선행성 (k일 강도 → 이후 h일 초과수익, 겹치지 않는 날짜)"); print(d2.round(3).to_string(index=False))
    d4 = pd.DataFrame(q_rows); d4.to_csv(OUT / f"{tag}_quintile.csv", index=False, encoding="utf-8-sig")
    print("\n④ 10일 강도 5분위 상−하 이후 수익"); print(d4.iloc[:, :9].round(4).to_string(index=False))
    return d1, d2, d4


def stats_rank(s):
    r = pd.Series(s).rank().to_numpy(dtype=float)
    return (r - r.mean()) / (r.std() + 1e-12)


def extra(tag, C, V, INS, FOR, U):
    AMT = C * V; logC = np.log(C); rows = []
    # ③ 지속성
    for nm, F in (("기관", INS), ("외국인", FOR)):
        for k in (5, 10):
            S = strength(F, AMT, k).where(U)
            ac = tstat(rowcorr(S, S.shift(-k)).iloc[::k])
            pos = S > 0; nxt = (S.shift(-k) > 0).where(S.shift(-k).notna())
            p_pp = nxt.where(pos).stack().mean(); p_np = nxt.where(~pos & S.notna()).stack().mean()
            rows.append(dict(주체=nm, k=k, 자기상관=ac[0], t=ac[1], 순매수후_순매수확률=p_pp, 순매도후_순매수확률=p_np))
    d3 = pd.DataFrame(rows); d3.to_csv(OUT / f"{tag}_persist.csv", index=False, encoding="utf-8-sig")
    print("\n③ 지속성 (k일 강도 ↔ 다음 k일 강도)"); print(d3.round(3).to_string(index=False))
    # ⑤ 확대 + 미반응
    k = h = 10; rows = []
    for nm, F in (("기관", INS), ("외국인", FOR)):
        S = strength(F, AMT, k).where(U); past = (logC - logC.shift(k)).where(U); pastx = past.sub(past.mean(axis=1), axis=0)
        fwd = (logC.shift(-h) - logC).where(U); fwdx = fwd.sub(fwd.mean(axis=1), axis=0)
        qs = S.rank(axis=1, pct=True); qr = pastx.rank(axis=1, pct=True)
        cells = {"확대+미반응": (qs >= 2 / 3) & (qr <= 1 / 3), "확대+반응": (qs >= 2 / 3) & (qr >= 2 / 3),
                 "축소+미반응": (qs <= 1 / 3) & (qr <= 1 / 3), "축소+반응": (qs <= 1 / 3) & (qr >= 2 / 3)}
        for cn, mk in cells.items():
            s = fwdx.where(mk).mean(axis=1).iloc[::h]; a = tstat(s); yr = s.groupby(s.index.year).mean()
            rows.append(dict(주체=nm, 셀=cn, 이후10일_초과=a[0], t=a[1], n=a[2], 일평균_종목=mk.sum(axis=1).mean(),
                             A=s[s.index < SPLIT].mean(), B=s[s.index >= SPLIT].mean(), 양수_해=int((yr > 0).sum()), 해=len(yr)))
    d5 = pd.DataFrame(rows); d5.to_csv(OUT / f"{tag}_cells.csv", index=False, encoding="utf-8-sig")
    print("\n⑤ 10일 강도 × 같은 10일 초과수익 3분위 셀 → 이후 10일 초과수익"); print(d5.round(4).to_string(index=False))


def main():
    C, V, INS, FOR, U = load()
    run("top200", C, V, INS, FOR, U); extra("top200", C, V, INS, FOR, U)
    try:                                               # 보조: 스윙 유니버스
        import backtest as bt
        p = bt.load_panel(bt.UniverseParams(), bt.VolScaleParams())
        E = p.elig.reindex(index=C.index, columns=C.columns).fillna(False).astype(bool)
        run("swing", C, V, INS, FOR, E); extra("swing", C, V, INS, FOR, E)
    except Exception as e:
        print("스윙 유니버스 생략:", e)


if __name__ == "__main__":
    main()
