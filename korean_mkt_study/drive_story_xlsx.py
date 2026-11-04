"""3일차 상승 드라이브 초상화 → 엑셀 (2026-11-04). 입력 out/crash/drive_story_k3.parquet (drive_story.py)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from crash_study import OUT
from xlsx_tables import write_tables

d = pd.read_parquet(OUT / "drive_story_k3.parquet"); N = len(d)
q = lambda c, n=5: pd.qcut(d[c].rank(method="first"), n, labels=[f"Q{i}" for i in range(1, n + 1)])
d["loud"] = (q("v_last") == "Q5").astype(int) + (q("rngx3") == "Q5").astype(int) + (q("clv3") == "Q5").astype(int) + (d.d3_above_n == 4).astype(int) + (d.overlap_c == "미끄럼").astype(int)
d["quiet"] = (q("v_last").isin(["Q1", "Q2"])).astype(int) + (q("rngx3").isin(["Q1", "Q2"])).astype(int) + (q("clv3").isin(["Q1", "Q2"])).astype(int) + (d.d3_above_n <= 2).astype(int) + (d.overlap_c == "계단").astype(int)
BASE = {"건수": "int", "비중": "pct", "지속률": "pct", "상승폭중앙": "spct", "D4수익중앙": "spct", "r10중앙": "spct"}

def rate(by, min_n=300):
    t = d.groupby(by, observed=True).agg(건수=("cont", "size"), 지속률=("cont", "mean"), 상승폭중앙=("tot", "median"), D4수익중앙=("next_r1", "median"), r10중앙=("r10", "median"))
    t.insert(1, "비중", t.건수 / N); return t[t.건수 >= min_n].reset_index()
def qrate(c, n=5):
    b = q(c, n); t = rate(b); med = d.groupby(b, observed=True)[c].median().values; t.insert(1, "구간중앙", med); return t

T = {}
# 1. 분포
cols = [("tot", "D0→D3 상승폭", "spct"), ("speed", "속도 A", "num3"), ("v_trend", "거래량 T", "num3"), ("overlap", "겹침 ō", "num3"), ("v_level", "드라이브 평균 거래량/직전20일", "num3"), ("v_last", "D3 거래량/직전20일", "num3"),
        ("gap1", "D1 시가 갭", "spct"), ("gap2", "D2 시가 갭", "spct"), ("gap3", "D3 시가 갭", "spct"), ("intra1", "D1 장중(시→종)", "spct"), ("intra2", "D2 장중", "spct"), ("intra3", "D3 장중", "spct"),
        ("gap_share", "상승폭 중 갭 비중", "snum"), ("clv1", "D1 종가 위치(0 저가~1 고가)", "num3"), ("clv2", "D2 종가 위치", "num3"), ("clv3", "D3 종가 위치", "num3"), ("ushadow3", "D3 윗꼬리 비율", "num3"),
        ("rngx1", "D1 봉 길이/직전20일 평균", "num3"), ("rngx3", "D3 봉 길이/직전20일 평균", "num3")]
rows = []
for c, lab, f in cols:
    s = d[c]; cont = d.cont
    rows.append({"변수": c, "설명": lab, "p10": s.quantile(.1), "p25": s.quantile(.25), "p50": s.median(), "p75": s.quantile(.75), "p90": s.quantile(.9), "_f": f,
                 "지속률(하위 Q1)": cont[q(c) == "Q1"].mean(), "지속률(상위 Q5)": cont[q(c) == "Q5"].mean()})
dist = pd.DataFrame(rows); fmts = dist.pop("_f")
T["1.분포"] = [("연속 변수 분포 (p10~p90) + 하위/상위 5분위의 지속률. 지속률 = D4 상승 비율, 전체 " + f"{d.cont.mean():.1%}", dist, {"p10": "num3", "p25": "num3", "p50": "num3", "p75": "num3", "p90": "num3", "지속률(하위 Q1)": "pct", "지속률(상위 Q5)": "pct"})]
# 1b 플래그
flags = [("hh2", "D2 고가 > D1 고가"), ("hh3", "D3 고가 > D2 고가"), ("hl2", "D2 저가 > D1 저가"), ("hl3", "D3 저가 > D2 저가"), ("body_up1", "D1 양봉"), ("body_up2", "D2 양봉"), ("body_up3", "D3 양봉"),
         ("d0_ma5", "D0 종가 > MA5"), ("d0_ma10", "D0 > MA10"), ("d0_ma20", "D0 > MA20"), ("d0_ma60", "D0 > MA60"), ("d3_ma5", "D3 > MA5"), ("d3_ma10", "D3 > MA10"), ("d3_ma20", "D3 > MA20"), ("d3_ma60", "D3 > MA60"),
         ("d0_ma_order", "D0 정배열(MA5>20>60)"), ("ma_order", "D3 정배열")]
fl = pd.DataFrame([{"플래그": c, "설명": lab, "해당 비율": d[c].mean(), "지속률(해당)": d.cont[d[c]].mean(), "지속률(비해당)": d.cont[~d[c]].mean(), "차이(pt)": (d.cont[d[c]].mean() - d.cont[~d[c]].mean()) * 100} for c, lab in flags])
T["1.분포"].append(("예/아니오 플래그 — 해당 비율과 지속률", fl, {"해당 비율": "pct", "지속률(해당)": "pct", "지속률(비해당)": "pct", "차이(pt)": "snum"}))
# 2. 봉 구조
T["2.봉구조"] = [("갭업 일수 (D1~D3 중 시가 > 전일 종가)", rate("gap_up_n"), BASE), ("고가 갱신 횟수 (D2, D3)", rate("hh_n"), BASE), ("저가 상승 횟수 (D2, D3)", rate("hl_n"), BASE), ("양봉 일수", rate("body_up_n"), BASE)]
for c, lab, f in [("gap3", "D3 시가 갭", "spct"), ("intra3", "D3 장중 수익", "spct"), ("clv3", "D3 종가 위치", "num3"), ("ushadow3", "D3 윗꼬리", "num3"), ("rngx3", "D3 봉 길이 배수", "num3"), ("gap_share", "상승폭 중 갭 비중", "snum"), ("v_last", "D3 거래량 배수", "num3"), ("tot", "상승폭", "spct")]:
    T["2.봉구조"].append((f"{lab} 5분위", qrate(c), {**BASE, "구간중앙": f}))
T["2.봉구조"].append(("D3 시가 갭 5분위 × D3 장중 5분위", rate([q("gap3").rename("gap3"), q("intra3").rename("intra3")], min_n=1000), BASE))
# 3. MA
T["3.이동평균"] = [("D0(시작 전) MA 배열 — 자리 = MA5·MA10·MA20·MA60 위(1)/아래(0)", rate("d0_stack"), BASE), ("D3 MA 배열", rate("d3_stack"), BASE),
                 ("D0 종가가 위에 있는 MA 개수", rate("d0_above_n"), BASE), ("D3 종가가 위에 있는 MA 개수", rate("d3_above_n"), BASE),
                 ("D0 MA 개수 × D3 MA 개수", rate(["d0_above_n", "d3_above_n"], min_n=500), BASE)]
# 4. 3변수
T["4.3변수"] = [("3변수 27조합 (속도 × 거래량 × 겹침)", rate(["speed_c", "v_trend_c", "overlap_c"], min_n=500), BASE),
              ("속도", rate("speed_c"), BASE), ("거래량", rate("v_trend_c"), BASE), ("겹침", rate("overlap_c"), BASE),
              ("겹침 × D3 위 MA 개수", rate(["overlap_c", "d3_above_n"]), BASE), ("겹침 × D0 MA 전부 아래(True)", rate([(d.d0_above_n == 0).rename("D0_MA전부아래"), "overlap_c"]), BASE),
              ("거래량 유형 × D3 거래량 배수 5분위", rate(["v_trend_c", q("v_last").rename("v_last")]), BASE)]
# 5. 조합
T["5.조합"] = [("시끄러움 점수 = D3 거래량 Q5 + D3 봉길이 Q5 + D3 종가 최상단 Q5 + MA 4개 위 + 미끄럼 (0~5)", rate("loud"), BASE),
             ("조용함 점수 = D3 거래량 Q1~2 + 봉길이 Q1~2 + 종가 하단 Q1~2 + MA 2개 이하 + 계단 (0~5)", rate("quiet"), BASE),
             ("D0 MA 전부 아래(True) × D3 거래량 5분위", rate([(d.d0_above_n == 0).rename("D0_MA전부아래"), q("v_last").rename("v_last")]), BASE),
             ("시총(전년말) × 겹침", rate(["cap_bucket", "overlap_c"]), BASE), ("시총 × 조용함 점수", rate(["cap_bucket", "quiet"], min_n=500), BASE),
             ("연도", rate("year"), BASE), ("연도 × 조용함 점수 ≥3 (True)", rate(["year", (d.quiet >= 3).rename("조용함3+")]), BASE)]
# 7. 조합 탐색 — 케인 조합 직접 + 의사결정나무(16~22 학습 → 23~26 검증) + 임계 민감도
from scipy.stats import ks_2samp, mannwhitneyu
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
d["rq"] = pd.qcut(d.rngx3.rank(method="first"), 5, labels=False) + 1; d["vq"] = pd.qcut(d.v_last.rank(method="first"), 5, labels=False) + 1; d["cq"] = pd.qcut(d.clv3.rank(method="first"), 5, labels=False) + 1
d["r_d1"] = (1 + d.gap1) * (1 + d.intra1) - 1
tr, te = d.year <= 2022, d.year >= 2023
def combo(mask, name):
    s = d[mask]
    return {"조건": name, "건수": len(s), "비중": len(s) / N, "지속률": s.cont.mean(), "D4수익중앙": s.next_r1.median(), "D4수익평균": s.next_r1.mean(), "r10중앙": s.r10.median(), "r10평균": s.r10.mean(),
            "지속률 16~22": s.cont[s.year <= 2022].mean(), "건수 16~22": int((s.year <= 2022).sum()), "지속률 23~26": s.cont[s.year >= 2023].mean(), "건수 23~26": int((s.year >= 2023).sum())}
CF = {"비중": "pct", "지속률": "pct", "D4수익중앙": "spct", "D4수익평균": "spct", "r10중앙": "spct", "r10평균": "spct", "지속률 16~22": "pct", "지속률 23~26": "pct", "건수 16~22": "int", "건수 23~26": "int"}
base = d.overlap_c.eq("계단") & d.speed_c.eq("감쇄") & d.v_trend_c.eq("감소"); small = d.rq <= 2; vnorm = d.vq.between(2, 3)
c1 = pd.DataFrame([combo(d.cont.notna(), "전체"), combo(base, "계단+감쇄+감소"), combo(base & small, " + D3 봉 작음(20일 평균의 0.84배↓, Q1~2)"), combo(base & small & vnorm, " + D3 거래량 평소(0.53~1.18배, Q2~3)"),
                   combo(base & small & (d.vq <= 3), " + D3 거래량 ≤Q3"), combo(base & small & (d.vq <= 3) & (d.d3_above_n <= 2), "   + D3 MA 2개 이하 위"), combo(base & small & (d.vq <= 3) & (d.d0_above_n == 0), "   + D0 MA 전부 아래"),
                   combo(base & small & (d.vq <= 3) & (d.cq <= 2), "   + D3 종가 하단(Q1~2)"), combo(d.overlap_c.eq("계단") & small & (d.vq <= 3), "계단(속도·거래량 무관) + 봉 작음 + 거래량 ≤Q3"),
                   combo(small & (d.vq <= 3) & (d.d0_above_n == 0) & (d.cq <= 2), "봉 작음 + 거래량 ≤Q3 + D0 MA 전부 아래 + 종가 하단 (겹침 무관)")])
sel = (d.d3_ma60_gap <= -0.10) & (d.intra1 > 0.045)
c2 = pd.DataFrame([combo(sel, "D3 종가 ≤ MA60 −10% AND D1 장중(시→종) > 4.5%  ← 나무가 찾은 잎"), combo(d.d3_ma60_gap <= -0.10, "D3 종가 ≤ MA60 −10% 만"), combo(d.intra1 > 0.045, "D1 장중 > 4.5% 만"),
                   combo(sel & d.overlap_c.eq("계단"), "조합 + 계단"), combo(sel & d.overlap_c.eq("미끄럼"), "조합 + 미끄럼"), combo(sel & (d.cap_bucket == "1천억~3천억"), "조합 + 시총 1천~3천억"), combo(sel & (d.cap_bucket == "5조+"), "조합 + 시총 5조+")])
yr = d[sel].groupby("year").agg(건수=("cont", "size"), 지속률=("cont", "mean"), r10중앙=("r10", "median"), r10평균=("r10", "mean")).reset_index()
grid = []
for gth in (-0.05, -0.08, -0.10, -0.12, -0.15, -0.20):
    row = {"MA60 갭 ≤": gth}
    for ith in (0.02, 0.03, 0.045, 0.06, 0.08):
        s = d[(d.d3_ma60_gap <= gth) & (d.intra1 > ith)]; row[f"D1장중>{ith:.1%} 지속률"] = s.cont.mean(); row[f"D1장중>{ith:.1%} 건수"] = len(s)
    grid.append(row)
grid = pd.DataFrame(grid)
feats = ["speed", "v_trend", "overlap", "v_level", "v_last", "gap1", "gap2", "gap3", "intra1", "intra2", "intra3", "clv1", "clv2", "clv3", "ushadow3", "rngx1", "rngx3", "gap_share", "hh_n", "hl_n", "body_up_n", "gap_up_n", "d0_above_n", "d3_above_n", "d3_ma5_gap", "d3_ma20_gap", "d3_ma60_gap", "tot"]
X = d[feats].replace([np.inf, -np.inf], np.nan); X = X.fillna(X.median()); y = d.cont.astype(int)
leaves = []
for depth, leaf in ((3, 3000), (4, 2000), (5, 1500)):
    clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=0).fit(X[tr], y[tr])
    lt = pd.DataFrame({"leaf": clf.apply(X[tr]), "y": y[tr].values}).groupby("leaf").y.agg(["mean", "size"]); le = pd.DataFrame({"leaf": clf.apply(X[te]), "y": y[te].values}).groupby("leaf").y.agg(["mean", "size"])
    t = lt.join(le, lsuffix="_tr", rsuffix="_te").sort_values("mean_tr", ascending=False)
    rho = t[["mean_tr", "mean_te"]].corr(method="spearman").iloc[0, 1]
    for i, (lf, r) in enumerate(t.iterrows()):
        if i < 4 or i >= len(t) - 2:
            leaves.append({"나무": f"depth{depth}/leaf≥{leaf}", "잎 순위(train)": i + 1, "잎 수": len(t), "학습 지속률": r.mean_tr, "학습 건수": int(r.size_tr), "검증 지속률": r.mean_te, "검증 건수": int(r.size_te), "학습↔검증 순위상관": rho})
leaves = pd.DataFrame(leaves)
T["6.조합탐색"] = [("① 케인 조합 — 계단+감쇄+감소에 봉·거래량·위치 조건을 더 얹기 (조건을 얹어도 50~52% 에서 포화)", c1, CF),
                ("② 나무가 찾은 잎 — 낮은 자리 + 강한 첫날 (임계 양방향 단조 → 잡음 아님)", c2, CF),
                ("②-1 연도별", yr, {"지속률": "pct", "r10중앙": "spct", "r10평균": "spct"}),
                ("②-2 임계 민감도 — 행 = D3 종가/MA60−1 상한, 열 = D1 장중 하한", grid, {**{c: "pct" for c in grid.columns if c.endswith("지속률")}, **{c: "int" for c in grid.columns if c.endswith("건수")}, "MA60 갭 ≤": "spct"}),
                ("③ 의사결정나무 잎 — 2016~22 학습, 2023~26 검증. 상위 4·하위 2 잎. 순위상관이 낮으면 잎 대부분이 검증에서 무너진 것", leaves, {"학습 지속률": "pct", "검증 지속률": "pct", "학습 건수": "int", "검증 건수": "int", "학습↔검증 순위상관": "num3"})]

# 8. 두 집단 비교 — (A) 이어짐 vs 끝남, (B) 6일↑ vs 3일끝
L = pd.read_parquet(OUT / "drive_len_up.parquet")[["ticker", "date", "n"]].rename(columns={"date": "date_d1"})
dn = d.merge(L, on=["ticker", "date_d1"], how="left")
contv = [("tot", "D0→D3 상승폭"), ("r_d1", "D1 수익(종가)"), ("speed", "속도 A"), ("v_trend", "거래량 T"), ("overlap", "겹침 ō"), ("v_level", "드라이브 평균 거래량 배수"), ("v_last", "D3 거래량 배수"), ("gap1", "D1 갭"), ("gap2", "D2 갭"), ("gap3", "D3 갭"),
         ("intra1", "D1 장중"), ("intra2", "D2 장중"), ("intra3", "D3 장중"), ("gap_share", "갭 비중"), ("clv1", "D1 종가 위치"), ("clv2", "D2 종가 위치"), ("clv3", "D3 종가 위치"), ("ushadow3", "D3 윗꼬리"), ("rngx1", "D1 봉 길이 배수"), ("rngx3", "D3 봉 길이 배수"),
         ("d3_ma5_gap", "D3 종가/MA5−1"), ("d3_ma20_gap", "D3 종가/MA20−1"), ("d3_ma60_gap", "D3 종가/MA60−1")]
flags = [("hh2", "D2 고가 갱신"), ("hh3", "D3 고가 갱신"), ("hl2", "D2 저가 상승"), ("hl3", "D3 저가 상승"), ("body_up1", "D1 양봉"), ("body_up2", "D2 양봉"), ("body_up3", "D3 양봉"), ("d0_ma5", "D0 > MA5"), ("d0_ma10", "D0 > MA10"), ("d0_ma20", "D0 > MA20"), ("d0_ma60", "D0 > MA60"),
         ("d3_ma5", "D3 > MA5"), ("d3_ma10", "D3 > MA10"), ("d3_ma20", "D3 > MA20"), ("d3_ma60", "D3 > MA60"), ("d0_ma_order", "D0 정배열"), ("ma_order", "D3 정배열")]
def compare(A, B, la, lb):
    rows = []
    for c, lab in contv:
        a = A[c].dropna(); b = B[c].dropna(); auc = mannwhitneyu(a, b).statistic / (len(a) * len(b)); ks = ks_2samp(a, b).statistic
        rows.append({"변수": c, "설명": lab, f"{la} p25": a.quantile(.25), f"{la} p50": a.median(), f"{la} p75": a.quantile(.75), f"{lb} p25": b.quantile(.25), f"{lb} p50": b.median(), f"{lb} p75": b.quantile(.75),
                     "중앙값차/IQR": (a.median() - b.median()) / (d[c].quantile(.75) - d[c].quantile(.25)), "AUC": auc, "KS": ks, "|AUC−0.5|": abs(auc - 0.5)})
    t1 = pd.DataFrame(rows).sort_values("|AUC−0.5|", ascending=False)
    f = pd.DataFrame([{"플래그": c, "설명": lab, f"{la} 비율": A[c].mean(), f"{lb} 비율": B[c].mean(), "차이(pt)": (A[c].mean() - B[c].mean()) * 100} for c, lab in flags]).sort_values("차이(pt)")
    cats = []
    for c in ["speed_c", "v_trend_c", "overlap_c", "d0_above_n", "d3_above_n", "hh_n", "hl_n", "gap_up_n", "cap_bucket"]:
        va = A[c].value_counts(normalize=True); vb = B[c].value_counts(normalize=True)
        for k in sorted(set(va.index) | set(vb.index), key=str): cats.append({"변수": c, "값": str(k), f"{la} 비율": va.get(k, 0), f"{lb} 비율": vb.get(k, 0), "차이(pt)": (va.get(k, 0) - vb.get(k, 0)) * 100})
    return t1, f, pd.DataFrame(cats)
F1 = {c: f for c, f in [(k, v) for k, v in zip(["중앙값차/IQR", "AUC", "KS", "|AUC−0.5|"], ["snum", "num3", "num3", "num3"])]}
def fmt_t1(la, lb): return {**F1, **{f"{x} p{p}": "num3" for x in (la, lb) for p in (25, 50, 75)}}
def fmt_f(la, lb): return {f"{la} 비율": "pct", f"{lb} 비율": "pct", "차이(pt)": "snum"}
def logit_auc(mask_sub, target):
    Xs = dn[[c for c, _ in contv] + ["d0_above_n", "d3_above_n", "hh_n", "hl_n", "gap_up_n"]].replace([np.inf, -np.inf], np.nan); Xs = Xs.clip(Xs.quantile(.01), Xs.quantile(.99), axis=1).fillna(Xs.median()); Xs = (Xs - Xs.mean()) / Xs.std()
    trm = mask_sub & (dn.year <= 2022); tem = mask_sub & (dn.year >= 2023)
    m = LogisticRegression(max_iter=1000).fit(Xs[trm], target[trm]); p = m.predict_proba(Xs[tem])[:, 1]; yt = target[tem]
    return {"AUC(16~22 학습 → 23~26 검증)": roc_auc_score(yt, p), "예측 상위 10% 실제 비율": yt[p >= np.quantile(p, .9)].mean(), "예측 하위 10% 실제 비율": yt[p <= np.quantile(p, .1)].mean(), "검증 기준 비율": yt.mean()}
A, B = dn[dn.cont], dn[~dn.cont]; t1, f, cats = compare(A, B, "이어짐", "끝남"); la = logit_auc(dn.cont.notna(), dn.cont)
T["7.두집단비교"] = [(f"(A) 3일차 시점 — 이어짐(D4 상승) {len(A):,} vs 끝남 {len(B):,} : 연속 변수 (AUC 0.5 = 구분 불가, KS 0 = 분포 동일)", t1, fmt_t1("이어짐", "끝남")),
                 ("(A) 플래그", f, fmt_f("이어짐", "끝남")), ("(A) 범주", cats, fmt_f("이어짐", "끝남")),
                 ("(A) 모든 변수를 합친 로지스틱", pd.DataFrame([la]), {k: "num3" if "AUC" in k else "pct" for k in la})]
ok = dn.n.notna(); A, B = dn[ok & (dn.n >= 6)], dn[ok & (dn.n == 3)]; t1, f, cats = compare(A, B, "6일↑", "3일끝"); lb = logit_auc(ok & ((dn.n == 3) | (dn.n >= 6)), (dn.n >= 6))
T["7.두집단비교"] += [(f"(B) 3일차 시점 — 결국 6일↑ 간 {len(A):,} vs 3일에 끝난 {len(B):,} (4~5일 {int((ok & dn.n.between(4, 5)).sum()):,} 제외)", t1, fmt_t1("6일↑", "3일끝")),
                  ("(B) 플래그", f, fmt_f("6일↑", "3일끝")), ("(B) 범주", cats, fmt_f("6일↑", "3일끝")),
                  ("(B) 모든 변수를 합친 로지스틱 (6일↑ vs 3일끝)", pd.DataFrame([lb]), {k: "num3" if "AUC" in k else "pct" for k in lb})]
ge6 = []
for c in ["d3_ma60_gap", "intra1", "r_d1", "clv3", "ushadow3", "v_last", "rngx3", "overlap", "tot"]:
    b = pd.qcut(dn[c].rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]); r = (dn[ok].groupby(b[ok], observed=True).n >= 6).mean() if False else dn[ok].assign(g=(dn.n >= 6)).groupby(b[ok], observed=True).g.mean()
    ge6.append({"변수": c, **{k: v for k, v in r.items()}})
ge6 = pd.DataFrame(ge6); ge6.insert(1, "기준", (dn.n[ok] >= 6).mean())
T["7.두집단비교"].append(("(B) 3일차 변수 5분위별 P(결국 6일↑)", ge6, {c: "pct" for c in ge6.columns if c != "변수"}))

memo = pd.DataFrame({"항목": ["표본", "시점", "지속", "3변수", "봉 구조", "MA 배열", "점수", "생성"], "내용": [
    f"상승 드라이브 3일차(k=3) 시점 행 {N:,}건 — 2016~2026, 동전주 제외, 다음날 수익이 있는 행만. 드라이브 단위가 아니라 '3일차에 서 있는' 시점이므로 4일 이상 간 드라이브도 모두 한 번씩 포함",
    "모든 묘사 변수는 D0(시작 전 거래일)~D3 정보만. 결과 = D4 종가 > D3 종가(지속) 여부, 보조로 D4 수익 중앙값·D3→D+10 수익 중앙값",
    f"전체 지속률 {d.cont.mean():.1%} — 3일차에서 하루 더 이어질 확률",
    "속도 A=|r_D3|/|r_D1| · 거래량 T=V_D3/V_D1 · 겹침 ō=(o_D2+o_D3)/2, 경계는 drive_types.THRESH (상승형)",
    "갭 = 시가/전일 종가−1, 장중 = 종가/시가−1, 종가 위치 = (C−L)/(H−L), 윗꼬리 = (H−max(O,C))/(H−L), 봉 길이 배수 = (H−L)/직전 20일 평균 (H−L). 갭 비중 = 3일 갭 복리 / 상승폭",
    "MA 는 단순이동평균(5·10·20·60), D0 는 드라이브 시작 전 거래일 종가 기준, D3 는 3일차 종가 기준. 배열 문자열 자리순 = MA5·MA10·MA20·MA60",
    "시끄러움·조용함 점수는 클로이가 탐색용으로 만든 합산 지표 — 규칙이 아니라 묘사. 6번 시트의 나무·로지스틱은 2016~22 학습 → 2023~26 검증(연도 홀드아웃)",
    "drive_story.py → out/crash/drive_story_k3.parquet → drive_story_xlsx.py (2026-11-04)"]})
T["8.메모"] = [("정의·주의사항", memo, {})]
out = OUT / "drive_story_k3_20261104.xlsx"; write_tables(T, out); print("saved", out)
