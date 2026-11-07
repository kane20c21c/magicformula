# -*- coding: utf-8 -*-
"""stock_gapshock_volume.py — 쇼크 범위 × 형태 × 크기 + 거래량 조절변수 → 회복 (케인 2026-11-07 19:37).

가설 (케인, 결과표를 본 뒤 세운 사후 가설): 쇼크일 거래량이 적을수록 반등이 빠르고 단단하다.
거래량   쇼크일 거래량 ÷ 직전 20거래일 평균 (정지 등 이상 행 제외). 범위마다 분포가 달라 학습 구간 3분위 근처 고정값으로 3구간:
         종목 저 <1.0배 · 중 1.0~2.0배 · 고 ≥2.0배 (3분위 1.06 / 1.82)
         섹터 저 <0.8배 · 중 0.8~1.2배 · 고 ≥1.2배 (3분위 0.75 / 1.15 — 업종지수는 쇼크일에도 거래량이 거의 안 늘어 배수 수준이 낮다)
         지수 저 <1.0배 · 중 1.0~1.25배 · 고 ≥1.25배 (코스피200 거래량은 2015~ 만 → 111건)
지표     r10·r20 (10·20일 안 쇼크 전날 종가 회복) · hold10 (저가 10일 유지, 쿠션 편향 주의) · dd10 (10일 최대 추가 하락)
검증     저 − 고 차이를 '같은 칸(범위·크기·형태) 안' 비교로: 선형확률/선형 회귀에 칸 고정효과 + 거래량 구간 → 저−고 계수,
         날짜 묶음 부트스트랩 400회 · 시기 3개 부호 · 몰림 10일 제외 · (참고) 보류 구간 — 이미 연 구간이라 '눈 가린 확인'은 아님
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import stock_gapshock_robust as R
OUT = HERE / "out" / "regime"; RNG = np.random.default_rng(1937)
BINS = {"종목": [0, 1.0, 2.0, np.inf], "섹터": [0, 0.8, 1.2, np.inf], "지수": [0, 1.0, 1.25, np.inf]}
VL = ["저", "중", "고"]; MET = ["r10", "r20", "hold10", "dd10"]

def prep(holdout=False):
    st, se, ix, top = R.load(holdout)
    ix["ctx"] = "지수"; ix["S"] = pd.cut(ix.r, [-1, -0.05, -0.03, -0.02], labels=["큰", "중간", "작은"]).astype(str)
    for nm, d in (("종목", st), ("섹터", se), ("지수", ix)):
        d["V"] = pd.cut(d.vol_n, BINS[nm], labels=VL, right=False).astype(str); d.loc[d.vol_n.isna(), "V"] = np.nan
        d["cell"] = d.ctx + "|" + d.S + "|" + d["T"]
    return st, se, ix

def table(d, rows):
    g = d.dropna(subset=["V"]).groupby(rows + ["V"], observed=True)
    t = pd.DataFrame({"건수": g.size(), "날짜": g.date.nunique(), "10일 안 회복%": g.r10.mean() * 100, "20일 안 회복%": g.r20.mean() * 100,
                      "저가 10일 유지%": g.hold10.mean() * 100, "10일 최대 추가 하락%": g.dd10.median() * 100})
    return t

def fe_coef(d, y):
    """칸 고정효과 + 거래량 구간 더미 → (저 − 고) 계수. 중은 따로 더미."""
    d = d.dropna(subset=["V", y])
    X = pd.get_dummies(d[["cell"]], drop_first=True).astype(float)
    X["저"] = (d.V == "저").astype(float); X["중"] = (d.V == "중").astype(float); X.insert(0, "c", 1.0)
    b = np.linalg.lstsq(X.to_numpy(), d[y].to_numpy(float), rcond=None)[0]
    return b[list(X.columns).index("저")]

def fe_boot(d, y, B=400):
    d = d.dropna(subset=["V", y]).reset_index(drop=True)
    u, inv = np.unique(d.date.to_numpy(), return_inverse=True); order = np.argsort(inv, kind="stable"); cuts = np.r_[0, np.cumsum(np.bincount(inv))]
    groups = [order[cuts[i]:cuts[i + 1]] for i in range(len(u))]
    bs = [fe_coef(d.iloc[np.concatenate([groups[k] for k in RNG.integers(0, len(u), len(u))])], y) for _ in range(B)]
    return np.nanpercentile(bs, [2.5, 97.5])

def main():
    st, se, ix = prep(False); sth, seh, _ = prep(True)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    A = table(st, ["ctx", "S"]); print("== A. 종목: 맥락 × 크기 × 거래량 (형태 합침)\n", A.round(1).to_string())
    Bt = table(st, ["ctx", "T"]); print("\n== B. 종목: 맥락 × 형태 × 거래량 (크기 합침)\n", Bt.round(1).to_string())
    C = table(se, ["ctx", "S"]); print("\n== C. 섹터: 맥락 × 크기 × 거래량\n", C.round(1).to_string())
    Dd = table(ix, ["S"]); print("\n== D. 지수(2015~, 거래량 있는 111건): 크기 × 거래량\n", Dd.round(1).to_string())
    # 견고성: 칸 고정효과 저 − 고
    rows = []
    for scope, d, dh, ctxs in (("종목", st, sth, ["시장 동반", "섹터 동반", "단독", None]), ("섹터", se, seh, ["시장 동반", "섹터 단독", None])):
        for c in ctxs:
            sub = d if c is None else d[d.ctx == c]; subh = dh if c is None else dh[dh.ctx == c]
            for y in MET:
                pt = fe_coef(sub, y); lo, hi = fe_boot(sub, y, 300 if c is None else 400)
                er = {e: fe_coef(sub[sub.era == e], y) for e in ("15–19", "20–22", "23–26.5")}
                nc = fe_coef(sub[~sub.crowd], y); ho = fe_coef(subh, y) if len(subh.dropna(subset=["V"])) > 50 else np.nan
                sg = np.sign(pt); out0 = lo > 0 or hi < 0
                ok = all(np.sign(v) == sg for v in er.values()) and np.sign(nc) == sg
                rows.append({"범위": scope, "맥락": c or "전체", "지표": y, "저−고(칸 안)": pt * 100, "95% 하한": lo * 100, "95% 상한": hi * 100,
                             **{f"시기 {k}": v * 100 for k, v in er.items()}, "몰림날 제외": nc * 100, "보류(참고)": ho * 100,
                             "판정": "견고" if out0 and ok else ("부분" if out0 else "약함"),
                             "건수 저/중/고": "/".join(str(int((sub.V == v).sum())) for v in VL)})
    T = pd.DataFrame(rows); print("\n== E. 거래량 저 − 고 (같은 범위·크기·형태 칸 안 비교, %p; dd10 은 플러스 = 저거래량이 덜 빠짐)\n", T.round(1).to_string(index=False))
    # 설명력 증분 (종목)
    e = st.dropna(subset=["V"]).copy()
    def r2(y, cols):
        m = e[y].notna(); X = pd.get_dummies(e.loc[m, cols].astype(str), drop_first=True).astype(float); X.insert(0, "c", 1.0)
        yy = e.loc[m, y].to_numpy(float); b = np.linalg.lstsq(X.to_numpy(), yy, rcond=None)[0]; return (1 - (yy - X.to_numpy() @ b).var() / yy.var()) * 100
    e["dd10r"] = e.dd10.rank(pct=True); Rr = []
    for y, nm in (("r10", "10일 안 회복"), ("r20", "20일 안 회복"), ("hold10", "저가 10일 유지"), ("dd10r", "10일 최대 추가 하락")):
        Rr.append({"지표": nm, "범위+크기+형태": r2(y, ["ctx", "S", "T"]), "+ 거래량": r2(y, ["ctx", "S", "T", "V"]), "거래량 단독": r2(y, ["V"]), "범위 단독": r2(y, ["ctx"])})
    RR = pd.DataFrame(Rr); print("\n== F. 설명력 R² (%)\n", RR.round(2).to_string(index=False))
    with pd.ExcelWriter(OUT / "volume_moderator.xlsx") as w:
        for nm, t in (("A_종목_맥락크기", A), ("B_종목_맥락형태", Bt), ("C_섹터", C), ("D_지수", Dd)): t.round(1).reset_index().to_excel(w, nm, index=False)
        T.round(1).to_excel(w, "E_저-고_견고성", index=False); RR.round(2).to_excel(w, "F_설명력", index=False)

if __name__ == "__main__":
    main()
