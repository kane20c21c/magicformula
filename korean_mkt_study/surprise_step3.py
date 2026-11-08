# -*- coding: utf-8 -*-
"""surprise_step3.py — 급등 연구 3단계: 견고성 (학습 구간만). 주장 U1~U9 (케인 2026-11-08 13:20 확정, 결과 보기 전 고정).
판정 (급락 연구 5단계 + 케인 추가 1)
  ① 날짜 묶음 부트스트랩 95% 구간 0 밖 (600회)
  ② 시기 3개 (2015–19 / 2020–22 / 2023–26.4) 부호 일치 — 시기별 A·B 각 30건 이상
  ③ 몰림 상위 10일(학습 급등 건수) ±5거래일 제외 후 부호 유지
  ④ 2026-01~04 제외 후 부호 유지 (케인 2026-11-08)
  견고 = ①②③④ 모두 · 부분 = ① 통과, 나머지 일부 실패 · 약함 = ① 실패
  '우위 없음' 주장(U1)은 반대로 읽는다: 전체 구간이 0 을 포함하고 시기별로 유의한 양(+)이 없으면 '없음 확인'.
참고 = 참고 정의(+5/+7/+10%, 거래량 1/2배)에서의 부호."""
import numpy as np, pandas as pd
from stock_gapshock import D, OUT, LIMIT_CHANGE, universe
from stock_surprise import HOLD, COST
rng = np.random.default_rng(20261108); NB = 600

def prep(ref=False, hold=False):
    ev = pd.read_parquet(OUT / "stock_surprise_events_ctx.parquet"); ev = ev[(ev.date >= HOLD) if hold else (ev.date < HOLD)].copy()
    if ref:
        ev["size"] = pd.cut(ev.r, [0.05, 0.07, 0.10, 1], right=False, labels=["작은", "중간", "큰"])
        ev["vb"] = pd.cut(ev.vol_n, [0, 1, 2, np.inf], right=False, labels=["저", "중", "고"])
    else:
        ev = ev[ev.r >= 0.06].copy()
        ev["size"] = pd.cut(ev.r, [0.06, 0.09, 0.14, 1], right=False, labels=["작은", "중간", "큰"])
        ev["vb"] = pd.cut(ev.vol_n, [0, 1.5, 3, np.inf], right=False, labels=["저", "중", "고"])
    ev["shape"] = pd.cut(ev.share, [-np.inf, 0, 0.4, 0.6, 1, np.inf], right=False, labels=["①", "②", "③", "④", "⑤"])
    ev["ctx2"] = np.where(ev.ctx == "시장 동반", np.where(ev.pre_mkt, "시장·반등형", "시장·순수"), ev.ctx)
    ev = ev[ev.ctx != "맥락 미상"].copy(); ev["w20"] = ev.a20 > 0
    for m in ["g10", "g20", "half10", "b10", "w20"]: ev[m] = ev[m].astype(float)
    return ev.reset_index(drop=True)

# 기준선: 유니버스 아무 날 다음 날 시가 진입 a20 (날짜별 합·건수 → 부분 구간 평균)
p = pd.read_parquet(D / "prices.parquet").reset_index().sort_values(["ticker", "date"])
g = p.groupby("ticker", sort=False); o1 = g.open.shift(-1); pr = p.close / g.close.shift(1) - 1
p["a20"] = (g.close.shift(-20) / o1 - 1 - COST).clip(-0.9, 5); p["year"] = p.date.dt.year
p = p[(p.date < HOLD) & (o1 > 0) & (pr.abs() <= 0.3005) & p.a20.notna()]
p = p.merge(universe(p, pd.read_parquet(D / "meta.parquet")), on=["ticker", "year"])
BASE = p.groupby("date").a20.agg(["sum", "count"]); del p
def base_mean(dmask):  # dmask: 날짜 → bool 함수
    b = BASE[np.asarray(dmask(pd.DatetimeIndex(BASE.index)))]; return b["sum"].sum() / b["count"].sum()

EVM = prep(False); EVR = prep(True)
kdates = pd.read_parquet(D / "kospi200_daily.parquet").index; kdates = pd.DatetimeIndex(sorted(pd.to_datetime(kdates)))
top10 = EVM.groupby("date").size().nlargest(10).index
excl = set()
for d0 in top10:
    i = kdates.get_loc(d0); excl.update(kdates[max(0, i - 5): i + 6])
EXCL = pd.DatetimeIndex(sorted(excl))
SUB = {"전체": lambda d: np.ones(len(d), bool),
       "2015–19": lambda d: (d >= "2015-01-01") & (d < "2020-01-01"),
       "2020–22": lambda d: (d >= "2020-01-01") & (d < "2023-01-01"),
       "2023–26.4": lambda d: (d >= "2023-01-01"),
       "몰림날 제외": lambda d: ~pd.DatetimeIndex(d).isin(EXCL),
       "2026.1~4 제외": lambda d: d < "2026-01-01"}

# ── 주장 정의: f(ev) → (A, B, strata) 또는 ("level", A, const_kind)
def S(ev): return ev["size"].astype(str)
def C(ev): return ev.ctx2
def cellSS(ev): return ev["size"].astype(str) + "|" + ev["shape"].astype(str)
def cellSV(ev): return ev["size"].astype(str) + "|" + ev.vb.astype(str)
CLAIMS = [
 ("U1", "단독 급등은 추격 우위가 없다", "a20", 0, lambda e: ("level", (C(e) == "단독"), "base")),
 ("U1", "단독 급등은 추격 우위가 없다", "xb20", 0, lambda e: ("level", (C(e) == "단독"), 0.0)),
 ("U2", "시장 동반 − 단독 (수익)", "a20", +1, lambda e: ((e.ctx == "시장 동반"), (C(e) == "단독"), None)),
 ("U2", "시장 동반 − 단독 (베타 보정 → 차이 없음 기대)", "xb20", 0, lambda e: ((e.ctx == "시장 동반"), (C(e) == "단독"), None)),
 ("U3", "시장·반등형 − 시장·순수", "a20", +1, lambda e: ((C(e) == "시장·반등형"), (C(e) == "시장·순수"), None)),
 ("U3", "시장·반등형 − 시장·순수", "b10", +1, lambda e: ((C(e) == "시장·반등형"), (C(e) == "시장·순수"), None)),
 ("U4", "섹터 동반 − 단독", "a20", +1, lambda e: ((C(e) == "섹터 동반"), (C(e) == "단독"), None)),
 ("U4", "섹터 동반 − 단독", "b10", +1, lambda e: ((C(e) == "섹터 동반"), (C(e) == "단독"), None)),
 ("U4", "섹터 동반 − 단독", "g20", -1, lambda e: ((C(e) == "섹터 동반"), (C(e) == "단독"), None)),
 ("U5a", "단독: 작은 − 큰 (수익·승률)", "a20", +1, lambda e: ((C(e) == "단독") & (S(e) == "작은"), (C(e) == "단독") & (S(e) == "큰"), None)),
 ("U5a", "단독: 작은 − 큰 (수익·승률)", "w20", +1, lambda e: ((C(e) == "단독") & (S(e) == "작은"), (C(e) == "단독") & (S(e) == "큰"), None)),
 ("U5b", "단독: 큰 − 작은 (덜 반납)", "half10", -1, lambda e: ((C(e) == "단독") & (S(e) == "큰"), (C(e) == "단독") & (S(e) == "작은"), None)),
 ("U5b", "단독: 큰 − 작은 (더 깊은 추가 하락)", "dd10", -1, lambda e: ((C(e) == "단독") & (S(e) == "큰"), (C(e) == "단독") & (S(e) == "작은"), None)),
 ("U6", "단독: 저 − 고 거래량 (크기·형태 통제)", "g10", +1, lambda e: ((C(e) == "단독") & (e.vb == "저"), (C(e) == "단독") & (e.vb == "고"), cellSS(e))),
 ("U6", "단독: 저 − 고 거래량 (크기·형태 통제)", "b10", +1, lambda e: ((C(e) == "단독") & (e.vb == "저"), (C(e) == "단독") & (e.vb == "고"), cellSS(e))),
 ("U6", "단독: 저 − 고 거래량 (크기·형태 통제)", "dd10", -1, lambda e: ((C(e) == "단독") & (e.vb == "저"), (C(e) == "단독") & (e.vb == "고"), cellSS(e))),
 ("U7", "시장 동반: 저 − 고 거래량 (크기·형태 통제)", "b10", +1, lambda e: ((e.ctx == "시장 동반") & (e.vb == "저"), (e.ctx == "시장 동반") & (e.vb == "고"), cellSS(e))),
 ("U7", "시장 동반: 저 − 고 거래량 (크기·형태 통제)", "a10", +1, lambda e: ((e.ctx == "시장 동반") & (e.vb == "저"), (e.ctx == "시장 동반") & (e.vb == "고"), cellSS(e))),
 ("U8", "단독: ⑤갭상승→밀림 − ② (크기·거래량 통제)", "g10", +1, lambda e: ((C(e) == "단독") & (e["shape"] == "⑤"), (C(e) == "단독") & (e["shape"] == "②"), cellSV(e))),
 ("U8", "단독: ⑤갭상승→밀림 − ② (크기·거래량 통제)", "half10", +1, lambda e: ((C(e) == "단독") & (e["shape"] == "⑤"), (C(e) == "단독") & (e["shape"] == "②"), cellSV(e))),
 ("U8", "단독: ⑤갭상승→밀림 − ② (크기·거래량 통제)", "b10", -1, lambda e: ((C(e) == "단독") & (e["shape"] == "⑤"), (C(e) == "단독") & (e["shape"] == "②"), cellSV(e))),
 ("U9", "시장·반등형·작은·저거래량: 기준선 대비", "a20", +1, lambda e: ("level", (C(e) == "시장·반등형") & (S(e) == "작은") & (e.vb == "저"), "base")),
 ("U9", "시장·반등형·작은·저거래량: 베타 보정", "xb20", +1, lambda e: ("level", (C(e) == "시장·반등형") & (S(e) == "작은") & (e.vb == "저"), 0.0)),
 ("U9", "시장·반등형·작은·저거래량 − 나머지 시장·반등형", "a20", +1, lambda e: ((C(e) == "시장·반등형") & (S(e) == "작은") & (e.vb == "저"), (C(e) == "시장·반등형") & ~((S(e) == "작은") & (e.vb == "저")), None)),
]

def estimate(ev, m, spec, dmask, boot=False):
    """spec 를 ev[dmask] 에 적용. 반환 (값, 하한, 상한, nA, nB). 단위 = 원 지표 (비율·수익 소수)."""
    e = ev[np.asarray(dmask(pd.DatetimeIndex(ev.date)))].reset_index(drop=True)
    sp = spec(e); v = e[m].to_numpy(float)
    dates, di = np.unique(e.date.to_numpy(), return_inverse=True); nd = len(dates)
    W = np.stack([np.bincount(rng.integers(0, nd, nd), minlength=nd) for _ in range(NB)]).astype(float) if boot else None
    def sums(mask):
        ok = np.asarray(mask) & np.isfinite(v); return np.bincount(di[ok], v[ok], nd), np.bincount(di[ok], None, nd)
    if isinstance(sp[0], str):
        A = np.asarray(sp[1]); const = base_mean(dmask) if sp[2] == "base" else sp[2]
        sa, ca = sums(A); est = sa.sum() / ca.sum() - const
        if not boot: return est, np.nan, np.nan, int(A.sum()), 0
        with np.errstate(invalid="ignore", divide="ignore"): bt = (W @ sa) / (W @ ca) - const
        lo, hi = np.nanpercentile(bt, [2.5, 97.5]); return est, lo, hi, int(A.sum()), 0
    A, Bm, strata = np.asarray(sp[0]), np.asarray(sp[1]), sp[2]
    st = pd.Series("all", index=e.index) if strata is None else strata
    est, bt, ws = 0.0, np.zeros(NB), 0.0
    for s in st.dropna().unique():
        ms = (st == s).to_numpy(); a, b = A & ms, Bm & ms; na, nb = a.sum(), b.sum()
        if na < 20 or nb < 20: continue
        sa, ca = sums(a); sb, cb = sums(b); w = na + nb
        est += w * (sa.sum() / ca.sum() - sb.sum() / cb.sum()); ws += w
        if boot:
            with np.errstate(invalid="ignore", divide="ignore"): bt += w * ((W @ sa) / (W @ ca) - (W @ sb) / (W @ cb))
    if ws == 0: return np.nan, np.nan, np.nan, int(A.sum()), int(Bm.sum())
    if not boot: return est / ws, np.nan, np.nan, int(A.sum()), int(Bm.sum())
    lo, hi = np.nanpercentile(bt / ws, [2.5, 97.5]); return est / ws, lo, hi, int(A.sum()), int(Bm.sum())

def main():
    rows = []
    for cid, name, m, sign, spec in CLAIMS:
        r = {"주장": cid, "내용": name, "지표": m, "기대": {1: "+", -1: "−", 0: "0"}[sign]}
        v, lo, hi, na, nb = estimate(EVM, m, spec, SUB["전체"], boot=True)
        r.update({"전체": v * 100, "하한": lo * 100, "상한": hi * 100, "nA": na, "nB": nb})
        nmin = []
        for k in ["2015–19", "2020–22", "2023–26.4", "몰림날 제외", "2026.1~4 제외"]:
            v2, _, _, a2, b2 = estimate(EVM, m, spec, SUB[k]); r[k] = v2 * 100
            if k.startswith("20") and "제외" not in k: nmin.append(a2 if isinstance(spec(EVM)[0], str) else min(a2, b2))
        r["시기 최소 n"] = min(nmin)
        vr, *_ = estimate(EVR, m, spec, SUB["전체"]); r["참고 정의"] = vr * 100
        ci_out = (lo > 0) or (hi < 0)
        per = [r["2015–19"], r["2020–22"], r["2023–26.4"]]
        if sign == 0:   # '차이·우위 없음' 주장
            anypos = any(np.sign(x) > 0 for x in per) and ci_out and v > 0
            r["판정"] = "없음 확인" if not ci_out else ("반대: 유의한 양(+)" if v > 0 else "음(−)으로 유의")
        else:
            sg = lambda x: np.isfinite(x) and np.sign(x) == sign
            c1 = ci_out and sg(v); c2 = all(sg(x) for x in per) and r["시기 최소 n"] >= 30
            c3 = sg(r["몰림날 제외"]); c4 = sg(r["2026.1~4 제외"])
            r["①"], r["②"], r["③"], r["④"] = ("○" if c else "×" for c in (c1, c2, c3, c4))
            r["판정"] = "견고" if (c1 and c2 and c3 and c4) else ("부분" if c1 else "약함")
            r["참고 부호"] = "같음" if sg(r["참고 정의"]) else "다름"
        rows.append(r)
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
    print("몰림 상위 10일:", [d.strftime("%Y-%m-%d") for d in top10])
    print(f"기준선 a20: 전체 {base_mean(SUB['전체'])*100:+.2f}% · " + " · ".join(f"{k} {base_mean(SUB[k])*100:+.2f}%" for k in list(SUB)[1:]))
    cols = ["주장", "내용", "지표", "기대", "전체", "하한", "상한", "nA", "nB", "2015–19", "2020–22", "2023–26.4", "시기 최소 n", "몰림날 제외", "2026.1~4 제외", "①", "②", "③", "④", "참고 정의", "참고 부호", "판정"]
    print(T[cols].round(1).to_string(index=False))
    T[cols].to_csv(OUT / "surprise_step3_robust.csv", index=False, encoding="utf-8-sig")
    import surprise_xlsx; surprise_xlsx.write_step3(OUT / "급등연구_3단계_견고성.xlsx", T[cols], top10, {k: base_mean(SUB[k]) * 100 for k in SUB})

if __name__ == "__main__":
    main()
