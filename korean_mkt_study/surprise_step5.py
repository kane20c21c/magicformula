# -*- coding: utf-8 -*-
"""surprise_step5.py — 급등 연구 5단계: 보류 구간 2026-05-01 ~ (data_2015 끝 10-29, 20일 경로 있는 급등만) 한 번만 연다.
케인 2026-11-08 15:42 승인. 확인 목록·판정은 열기 전에 고정 (2026-11-08 4단계 보고):
  3단계 주장 U1~U9 전부 (사전 핵심 = 학습에서 견고했던 것) · 4단계 즉시/저가이탈/본전이탈 − 보유 (h=20)
  판정  확인 = 학습과 같은 부호 & 95% 구간 0 밖 · 같은 방향 = 같은 부호 · 반대 = 부호 반대
        '없음' 주장 = 보류 95% 구간이 0 을 포함하면 '없음 유지'
  국면  상승 2026-05-01~06-22 · 하락·회복 06-23~ (코스피200 06-22 고점)"""
import numpy as np, pandas as pd
import surprise_step3 as s3
from stock_gapshock import D, OUT, universe
from stock_surprise import HOLD, COST
SELL = 0.002; rng = np.random.default_rng(20261108); NB = 600
EVH = s3.prep(False, hold=True); EVHR = s3.prep(True, hold=True)
# 보류 기준선 (유니버스 아무 날 다음 날 시가 진입 a20)
p = pd.read_parquet(D / "prices.parquet").reset_index().sort_values(["ticker", "date"]).reset_index(drop=True)
g = p.groupby("ticker", sort=False); o1 = g.open.shift(-1); pr = p.close / g.close.shift(1) - 1
b = p.assign(a20=(g.close.shift(-20) / o1 - 1 - COST).clip(-0.9, 5), year=p.date.dt.year)
b = b[(b.date >= HOLD) & (o1 > 0) & (pr.abs() <= 0.3005) & b.a20.notna()]
b = b.merge(universe(p.assign(year=p.date.dt.year), pd.read_parquet(D / "meta.parquet")), on=["ticker", "year"])
s3.BASE = b.groupby("date").a20.agg(["sum", "count"])
PH = {"보류 전체": lambda d: np.ones(len(d), bool), "상승 05-01~06-22": lambda d: d <= "2026-06-22", "하락·회복 06-23~": lambda d: d > "2026-06-22"}
tr3 = pd.read_csv(OUT / "surprise_step3_robust.csv", encoding="utf-8-sig")
def verdict(sign_tr, v, lo, hi, nothing=False):
    if not np.isfinite(v): return "판정 불가"
    ci = (lo > 0) or (hi < 0)
    if nothing: return "없음 유지" if not ci else ("유의한 +" if v > 0 else "유의한 −")
    if np.sign(v) == sign_tr: return "확인" if ci else "같은 방향"
    return "반대"
rows = []
for i, (cid, name, m, sign, spec) in enumerate(s3.CLAIMS):
    t = tr3.iloc[i]; v, lo, hi, na, nb = s3.estimate(EVH, m, spec, PH["보류 전체"], boot=True)
    sign_tr = np.sign(t["전체"]) if sign == 0 else sign
    r = {"주장": cid, "내용": name, "지표": m, "학습 판정": t["판정"], "학습 값": t["전체"], "보류 값": v * 100, "보류 하한": lo * 100, "보류 상한": hi * 100,
         "보류 nA": na, "보류 nB": nb}
    for k in ["상승 05-01~06-22", "하락·회복 06-23~"]:
        v2, _, _, a2, b2 = s3.estimate(EVH, m, spec, PH[k]); r[k] = v2 * 100; r[k + " n"] = a2 if isinstance(spec(EVH)[0], str) else min(a2, b2)
    vr, *_ = s3.estimate(EVHR, m, spec, PH["보류 전체"]); r["보류 참고 정의"] = vr * 100
    r["보류 판정"] = verdict(sign_tr, v, lo, hi, nothing=(sign == 0))
    rows.append(r)
T3 = pd.DataFrame(rows)
# ── 4단계 보유 규칙 (보류)
ev = EVH
key = pd.Series(np.arange(len(p)), index=pd.MultiIndex.from_frame(p[["ticker", "date"]]))
pos = key.reindex(pd.MultiIndex.from_frame(ev[["ticker", "date"]])).to_numpy().astype(int)
TK = p.ticker.to_numpy(); Cc = p.close.to_numpy(float); Oo = p.open.to_numpy(float); n = len(p); J = np.arange(1, 22)
idx = np.minimum(pos[:, None] + J[None, :], n - 1); same = TK[idx] == TK[pos][:, None]
CL = np.where(same, Cc[idx], np.nan); OP = np.where(same, Oo[idx], np.nan); OP = np.where(np.isfinite(OP) & (OP > 0), OP, CL)
c0 = ev.close.to_numpy(); pc = ev.pc.to_numpy(); hi_ = ev.high.to_numpy(); lo_ = ev.low.to_numpy(); R = np.arange(len(ev))
def first(cond, k): c = cond[:, :k]; return np.where(c.any(1), c.argmax(1) + 1, 0)
dG = first(CL <= pc[:, None], 20); dL = first(CL < lo_[:, None], 20); dB = first(CL > hi_[:, None], 20)
hold = CL[:, 19] / c0 - 1 - SELL; ev = ev.assign(r_hold20=hold, r_now20=-SELL)
for nm, d in (("low", dL), ("pc", dG)):
    trig = d > 0; ev[f"r_{nm}20"] = np.where(trig, OP[R, np.clip(d, 1, 20)] / c0 - 1 - SELL, hold)
ev = ev[np.isfinite(ev.r_hold20)].copy()
def cells(e):
    s = (e.ctx2 == "단독").to_numpy()
    return {"단독": s, "섹터 동반": (e.ctx2 == "섹터 동반").to_numpy(), "시장·반등형": (e.ctx2 == "시장·반등형").to_numpy(),
            "단독·큰": s & (e["size"] == "큰").to_numpy(), "단독·거래량 고": s & (e.vb == "고").to_numpy(), "단독·거래량 저": s & (e.vb == "저").to_numpy()}
tr4 = pd.read_csv(OUT / "surprise_step4_robust.csv", encoding="utf-8-sig").set_index(["칸", "비교"])
def bdiff(e, a):
    e = e[np.isfinite(e[a]) & np.isfinite(e.r_hold20)]
    d = (e[a] - e.r_hold20).to_numpy(float); dates, di = np.unique(e.date.to_numpy(), return_inverse=True); nd = len(dates)
    s_ = np.bincount(di, d, nd); c_ = np.bincount(di, None, nd)
    W = np.stack([np.bincount(rng.integers(0, nd, nd), minlength=nd) for _ in range(NB)]).astype(float)
    return d.mean(), *np.percentile((W @ s_) / (W @ c_), [2.5, 97.5])
rows = []
for k, m in cells(ev).items():
    e = ev[m]
    for a, lab in (("r_now20", "(가) 즉시 − 보유"), ("r_low20", "(나) 저가이탈 − 보유"), ("r_pc20", "(라) 본전이탈 − 보유")):
        v, lo, hi = bdiff(e, a); t = tr4.loc[(k, lab)]
        r = {"칸": k, "비교": lab, "학습 판정": t["판정"], "학습 값": t["전체"], "보류 n": len(e), "보류 값": v * 100, "보류 하한": lo * 100, "보류 상한": hi * 100}
        for ph in ["상승 05-01~06-22", "하락·회복 06-23~"]:
            mm = np.asarray(PH[ph](pd.DatetimeIndex(e.date))); r[ph] = (e.loc[mm, a] - e.loc[mm, "r_hold20"]).mean() * 100; r[ph + " n"] = int(mm.sum())
        r["보류 판정"] = verdict(np.sign(t["전체"]), v, lo, hi)
        r["보류 보유 평균"] = e.r_hold20.mean() * 100; r["보류 보유 p10"] = e.r_hold20.quantile(.1) * 100; r["보류 규칙 p10"] = e[a].quantile(.1) * 100
        rows.append(r)
T4 = pd.DataFrame(rows)
# 보류 기술 (범위별 a20·g20·b10)
D_ = EVH.groupby("ctx2").agg(n=("a20", "size"), g20=("g20", "mean"), half10=("half10", "mean"), b10=("b10", "mean"), a20=("a20", "mean"), w20=("w20", "mean"), xb20=("xb20", "mean"))
for c in D_.columns[1:]: D_[c] *= 100
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
print(f"보류 급등 (주 정의, 20일 경로): {len(EVH):,} · 기간 {EVH.date.min():%Y-%m-%d} ~ {EVH.date.max():%Y-%m-%d} · 기준선 a20 {s3.base_mean(PH['보류 전체'])*100:+.2f}% "
      f"(상승 {s3.base_mean(PH['상승 05-01~06-22'])*100:+.2f} / 하락·회복 {s3.base_mean(PH['하락·회복 06-23~'])*100:+.2f})")
print("\n[보류 기술]"); print(D_.round(1).to_string())
print("\n[3단계 주장 — 보류]"); print(T3.round(1).to_string(index=False))
print("\n[4단계 보유 규칙 — 보류 h=20]"); print(T4.round(1).to_string(index=False))
T3.to_csv(OUT / "surprise_step5_holdout_claims.csv", index=False, encoding="utf-8-sig"); T4.to_csv(OUT / "surprise_step5_holdout_rules.csv", index=False, encoding="utf-8-sig")
import surprise_xlsx
nc3 = ["학습 값", "보류 값", "보류 하한", "보류 상한", "상승 05-01~06-22", "하락·회복 06-23~", "보류 참고 정의"]
nc4 = ["학습 값", "보류 값", "보류 하한", "보류 상한", "상승 05-01~06-22", "하락·회복 06-23~", "보류 보유 평균", "보류 보유 p10", "보류 규칙 p10"]
intro = [("급등(서프라이즈) 연구 — 5단계 보류 구간 (2026-05-01 ~, 20일 경로 있는 급등만, 한 번만 열었다 2026-11-08)", True),
         (f"보류 급등 {len(EVH):,}건 · 기준선 a20 {s3.base_mean(PH['보류 전체'])*100:+.2f}% · 국면: 상승 05-01~06-22 / 하락·회복 06-23~", False),
         ("판정: 확인 = 학습과 같은 부호 & 95% 구간 0 밖 · 같은 방향 = 같은 부호 · 반대 = 부호 반대 · '없음' 주장은 0 포함이면 '없음 유지'", False),
         ("단위 %p (U1·U9 수준 비교는 보류 기준선 대비). 빨강 = +, 파랑 = −", False)]
surprise_xlsx.write_generic(OUT / "급등연구_5단계_보류구간.xlsx", intro, [
    ("보류 기술", [("범위별 (비율·수익 %)", D_, ["a20", "xb20"], False, None, ["범위"])]),
    ("주장 U1~U9", [("3단계 주장의 보류 구간 결과", T3.set_index(["주장", "내용", "지표"]), nc3, True, None, ["주장", "내용", "지표"])]),
    ("보유 규칙", [("4단계 보유 규칙 − 보유 (h=20)", T4.set_index(["칸", "비교"]), nc4, True, None, ["칸", "비교"])])])
