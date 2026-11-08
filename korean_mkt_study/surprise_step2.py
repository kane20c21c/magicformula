# -*- coding: utf-8 -*-
"""surprise_step2.py — 급등 연구 2단계: 기본 대비 (학습 구간만). 순서 범위 → 크기 → 거래량 → 형태 (케인 2026-11-08).
확정 (케인 2026-11-08 08:34)
  크기(종목) 주 = 같은 드묾 경계 작은 [+6, +9) · 중간 [+9, +14) · 큰 ≥ +14%   (참고 = +5/+7/+10)
  거래량 주 = 저 < 1.5 · 중 1.5~3 · 고 ≥ 3 배   (참고 = 1 / 2)
  시장 동반을 반등형(직전 10거래일 코스피200 ≤ −2% 있음) / 순수 상승형으로 나눔
95% 구간 = 날짜 묶음 부트스트랩 600회 (같은 날 급등 = 한 덩어리). 칸 통제 대비 = 칸별 차이를 원표본 칸 크기(nA+nB)로 가중."""
import sys, numpy as np, pandas as pd
from stock_gapshock import D, OUT, LIMIT_CHANGE, universe
from stock_surprise import HOLD, COST
rng = np.random.default_rng(20261108); B = 600
ev = pd.read_parquet(OUT / "stock_surprise_events_ctx.parquet"); ev = ev[ev.date < HOLD].copy()
REF = "ref" in sys.argv[1:]
if REF:
    ev["size"] = pd.cut(ev.r, [0.05, 0.07, 0.10, 1], right=False, labels=["작은", "중간", "큰"])
    ev["vb"] = pd.cut(ev.vol_n, [0, 1, 2, np.inf], right=False, labels=["저", "중", "고"])
else:
    ev = ev[ev.r >= 0.06].copy()
    ev["size"] = pd.cut(ev.r, [0.06, 0.09, 0.14, 1], right=False, labels=["작은", "중간", "큰"])
    ev["vb"] = pd.cut(ev.vol_n, [0, 1.5, 3, np.inf], right=False, labels=["저", "중", "고"])
ev["shape"] = pd.cut(ev.share, [-np.inf, 0, 0.4, 0.6, 1, np.inf], right=False, labels=["①", "②", "③", "④", "⑤"])
ev["ctx2"] = np.where(ev.ctx == "시장 동반", np.where(ev.pre_mkt, "시장·반등형", "시장·순수"), ev.ctx)
ev = ev[ev.ctx != "맥락 미상"].copy()
ev["w20"] = ev.a20 > 0
M = ["g5", "g10", "g20", "half5", "half10", "b5", "b10", "du5", "du10", "dd5", "dd10", "a5", "a10", "a20", "w20", "xb20"]
PCT_RATE = {"g5", "g10", "g20", "half5", "half10", "b5", "b10", "w20"}
for m in M: ev[m] = ev[m].astype(float)
dates, di = np.unique(ev.date.to_numpy(), return_inverse=True); ND = len(dates)
W = np.stack([np.bincount(rng.integers(0, ND, ND), minlength=ND) for _ in range(B)]).astype(float)
def sums(mask, m):
    v = ev[m].to_numpy(); ok = mask & np.isfinite(v)
    return np.bincount(di[ok], v[ok], ND), np.bincount(di[ok], None, ND)
def contrast(m, A, Bm, strata=None):
    """A − B (평균). strata: 칸 라벨 Series → 칸 안 차이를 (nA+nB) 가중."""
    if strata is None: strata = pd.Series("all", index=ev.index)
    est, boot, wsum = 0.0, np.zeros(B), 0.0
    for s in strata.dropna().unique():
        ms = (strata == s).to_numpy(); a = A & ms; b = Bm & ms
        na, nb = a.sum(), b.sum()
        if na < 20 or nb < 20: continue
        sa, ca = sums(a, m); sb, cb = sums(b, m); w = na + nb
        est += w * (sa.sum() / ca.sum() - sb.sum() / cb.sum())
        with np.errstate(invalid="ignore", divide="ignore"): boot += w * ((W @ sa) / (W @ ca) - (W @ sb) / (W @ cb))
        wsum += w
    if wsum == 0: return np.nan, np.nan, np.nan
    lo, hi = np.nanpercentile(boot / wsum, [2.5, 97.5]); return est / wsum, lo, hi
def fmt(m, x):
    k = 100; return f"{x*k:+.1f}" if np.isfinite(x) else "—"
def desc(by):
    g = ev.groupby(by, observed=True)
    t = g[["g5", "g10", "g20", "half5", "half10", "b5", "b10", "w20"]].mean() * 100
    t2 = g[["du5", "du10", "dd5", "dd10"]].median() * 100
    t3 = g[["a5", "a10", "a20", "xb20"]].mean() * 100
    out = pd.concat([g.size().rename("n"), t, t2.add_prefix("med_"), t3], axis=1); return out.round(1)
def ctab(rows):
    """숫자 표: 지표마다 m (점추정, %p), m_lo, m_hi (95% 구간). 인쇄용 문자열은 show()."""
    out = []
    for label, A, Bm, strata in rows:
        r = {"대비": label, "nA": int(A.sum()), "nB": int(Bm.sum())}
        for m in M:
            e, lo, hi = contrast(m, A, Bm, strata); r[m] = e * 100; r[m + "_lo"] = lo * 100; r[m + "_hi"] = hi * 100
        out.append(r)
    return pd.DataFrame(out)
def show(t):
    d = t[["대비", "nA", "nB"]].copy()
    for m in M:
        sig = (t[m + "_lo"] > 0) | (t[m + "_hi"] < 0)
        d[m] = [("—" if not np.isfinite(v) else f"{v:+.1f}" + ("*" if s_ else "")) for v, s_ in zip(t[m], sig)]
    return d
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
tag = "ref" if REF else "main"
# 기준선: 유니버스 아무 날 다음 날 시가 진입 a5/a10/a20 (학습)
p = pd.read_parquet(D / "prices.parquet").reset_index().sort_values(["ticker", "date"])
g = p.groupby("ticker", sort=False); o1 = g.open.shift(-1)
base = {k: (g.close.shift(-k) / o1 - 1 - COST) for k in (5, 10, 20)}
p["year"] = p.date.dt.year; pr = p.close / g.close.shift(1) - 1
okb = (p.date < HOLD) & (o1 > 0) & (pr.abs() <= 0.3005)
u = universe(p, pd.read_parquet(D / "meta.parquet")); keyu = set(map(tuple, u.to_numpy()))
inu = pd.Series([ (t, y) in keyu for t, y in zip(p.ticker, p.year)], index=p.index)
bl = {k: v[okb & inu].clip(-0.9, 5).mean() * 100 for k, v in base.items()}
print(f"=== 2단계 ({tag}) 학습 n={len(ev):,} · 기준선(유니버스 아무 날, 다음 날 시가 진입) a5 {bl[5]:+.2f}% · a10 {bl[10]:+.2f}% · a20 {bl[20]:+.2f}%")
print("\n[A] 범위별 기술 (비율 %, du/dd 중앙 %, a·xb 평균 %)"); dA = desc("ctx2"); print(dA.to_string())
print("\n[A2] 전체"); print(desc(pd.Series("전체", index=ev.index)).to_string())
c = ev.ctx2
S = ev["size"]; V = ev.vb; SH = ev["shape"]
cell_sv = S.astype(str) + "|" + SH.astype(str)          # 크기·형태 칸
rowsB = [("시장 동반(전체) − 단독", (ev.ctx == "시장 동반").to_numpy(), (c == "단독").to_numpy(), None),
         ("섹터 동반 − 단독", (c == "섹터 동반").to_numpy(), (c == "단독").to_numpy(), None),
         ("시장·반등형 − 시장·순수", (c == "시장·반등형").to_numpy(), (c == "시장·순수").to_numpy(), None),
         ("시장·순수 − 단독", (c == "시장·순수").to_numpy(), (c == "단독").to_numpy(), None),
         ("시장 동반 − 단독 (크기 통제)", (ev.ctx == "시장 동반").to_numpy(), (c == "단독").to_numpy(), S.astype(str))]
print("\n[B] 범위 대비 (A − B, %p, * = 95% 구간 0 밖)"); tb = ctab(rowsB); print(show(tb).to_string(index=False))
print("\n[C] 크기 (범위 안)"); print(desc(["ctx2", "size"]).to_string())
rowsC = []
for x in ["단독", "섹터 동반", "시장·반등형", "시장·순수"]:
    mx = (c == x).to_numpy()
    rowsC.append((f"{x}: 큰 − 작은", mx & (S == "큰").to_numpy(), mx & (S == "작은").to_numpy(), None))
tc = ctab(rowsC); print(show(tc).to_string(index=False))
print("\n[D] 거래량 (범위 안)"); print(desc(["ctx2", "vb"]).to_string())
rowsD = []
for x in ["단독", "섹터 동반", "시장·반등형", "시장·순수"]:
    mx = (c == x).to_numpy()
    rowsD.append((f"{x}: 저 − 고 (크기·형태 통제)", mx & (V == "저").to_numpy(), mx & (V == "고").to_numpy(), cell_sv))
td = ctab(rowsD); print(show(td).to_string(index=False))
print("\n[E] 형태 (범위 안)"); print(desc(["ctx2", "shape"]).to_string())
rowsE = []
for x in ["단독", "섹터 동반", "시장·반등형", "시장·순수"]:
    mx = (c == x).to_numpy(); cs = S.astype(str) + "|" + V.astype(str)
    for sh in ["①", "③", "④", "⑤"]:
        rowsE.append((f"{x}: {sh} − ② (크기·거래량 통제)", mx & (SH == sh).to_numpy(), mx & (SH == "②").to_numpy(), cs))
te = ctab(rowsE); print(show(te).to_string(index=False))
# 설명력 R² (선형확률, 범주 더미)
def r2(y, cols):
    d = ev[[y] + cols].dropna(); X = pd.get_dummies(d[cols].astype(str), drop_first=True).astype(float); X.insert(0, "c", 1.0)
    b, *_ = np.linalg.lstsq(X.to_numpy(), d[y].to_numpy(), rcond=None); res = d[y].to_numpy() - X.to_numpy() @ b
    return 1 - res.var() / d[y].var()
print("\n[F] 설명력 R² (%) — 선형, 범주 더미")
rr = []
for y in ["g10", "half10", "b10", "a20", "xb20"]:
    rr.append({"지표": y, "범위": r2(y, ["ctx2"]), "크기": r2(y, ["size"]), "거래량": r2(y, ["vb"]), "형태": r2(y, ["shape"]),
               "범위+크기+형태": r2(y, ["ctx2", "size", "shape"]), "+거래량": r2(y, ["ctx2", "size", "shape", "vb"])})
r2t = (pd.DataFrame(rr).set_index("지표") * 100).round(2); print(r2t.to_string())
allt = pd.concat([tb.assign(sec="B"), tc.assign(sec="C"), td.assign(sec="D"), te.assign(sec="E")]); allt.to_csv(OUT / f"surprise_step2_contrasts_{tag}.csv", index=False, encoding="utf-8-sig")
pd.concat([dA, desc(["ctx2", "size"]), desc(["ctx2", "vb"]), desc(["ctx2", "shape"])]).to_csv(OUT / f"surprise_step2_desc_{tag}.csv", encoding="utf-8-sig")

# ── 엑셀 (케인 2026-11-08: 2단계도 엑셀로) ──
import surprise_xlsx
descs = {"범위": dA, "크기": desc(["ctx2", "size"]), "거래량": desc(["ctx2", "vb"]), "형태": desc(["ctx2", "shape"]), "전체": desc(pd.Series("전체", index=ev.index))}
surprise_xlsx.write_step2(OUT / f"급등연구_2단계_{tag}.xlsx", tag=tag, n=len(ev), baseline=bl, descs=descs,
                          contrasts={"범위": tb, "크기": tc, "거래량": td, "형태": te}, r2=r2t, metrics=M, ref=REF)
