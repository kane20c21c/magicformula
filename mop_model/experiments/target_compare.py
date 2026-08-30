# -*- coding: utf-8 -*-
"""
target_compare.py — target_walkforward.py 산출물 비교 분석 (검증 전용)

읽는 것: build/tw_{y_rel,y_abs,y_relz}.parquet  (target_walkforward.py 가 만든다)
        + output/signals/signal_*.json (라이브 대조용)
        + LLV core/extend parquet (YZ_20 — 변동성 편향 측정용)

내는 것 (전부 화면 출력):
  【1】 p 십분위별 갭 — 랭크가 목표와 단조한가
  【2】 단조성·판별력 통계 (십분위 스피어만 · 일별 IC · 일별 AUC · p↔YZ_20 상관)
  【3】 rank 구간별 변동성 백분위 — 랭크가 변동성 순위표인가
  【4】 변동성 분위 내부 IC — 변동성을 통제해도 효과가 남는가
  【5】 rank ≤ k 상위 선별 수익 (갭 비용후 · 익일종가 · 장중)
  【6】 워크포워드 ↔ 라이브 신호 대조 (재현 검증)
  【7】 월별 추이 + 검정력 시뮬레이션 — 짧은 표본으로 판단하면 무엇을 놓치는가

⚠ 여기 수익은 전부 **t종가 매수 → t+1 시가 매도** 백테스트다. 실제 데이 포트는
  17:00 NXT 애프터마켓 매수 + 당일고점 −1% 트레일링이라 **직접 비교 금지**
  (walkforward.py 머리 주석과 같은 경고). 랭크 품질 판정용이다.

사용: python3 target_compare.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import config as cfg  # noqa: E402

pd.set_option("display.width", 250)
NAME = {"y_rel": "현행 y_rel (갭>중앙)", "y_abs": "y_abs (갭>0)", "y_relz": "변동성조정갭"}
BUY_FEE, SELL_FEE = 0.00015, 0.00215      # walkforward.py 와 동일 (참고용)


def _auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    if y.min() == y.max():
        return np.nan
    r = stats.rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def load_vol():
    cols = ["Date", "Ticker", "YZ_20"]
    v = pd.concat([pd.read_parquet(cfg.CORE, columns=cols),
                   pd.read_parquet(cfg.EXTEND, columns=cols)], ignore_index=True)
    v["D"] = v.Date.astype(str).str[:10]
    return v.drop_duplicates(subset=["Ticker", "D"])[["Ticker", "D", "YZ_20"]]


def load_live():
    """라이브 신호 JSON → (D, ticker, rank, p) — 워크포워드 재현 검증용."""
    rows = []
    for f in sorted(Path(cfg.SIGNAL_DIR).glob("signal_2*.json")):
        d = json.load(open(f))
        for r in d["ranking"]:
            rows.append(dict(D=d["as_of"], ticker=r["ticker"],
                             rank_live=r["rank"], p_live=r["p"]))
    return pd.DataFrame(rows)


def main():
    D = {}
    for t in ["y_rel", "y_abs", "y_relz"]:
        p = Path(cfg.OUT_DIR) / f"tw_{t}.parquet"
        if not p.exists():
            print(f"[{t}] 없음 — target_walkforward.py 를 먼저 돌릴 것: {p}")
            continue
        d = pd.read_parquet(p)
        d["D"] = d.Date.astype(str).str[:10]
        d["gap"] = d.Gap_T1
        d["day"] = d.close_T1 / d.Close - 1
        d["intra"] = d.close_T1 / d.open_T1 - 1
        d = d[d.gap.notna() & d.p.notna()].copy()
        d["dec"] = d.groupby("D").p.transform(
            lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1)
        d["vpct"] = d.groupby("D").YZ_20.transform(lambda s: s.rank(pct=True) * 100)
        d["vq"] = d.groupby("D").YZ_20.transform(
            lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
        d["rb"] = pd.cut(d["rank"], [0, 10, 30, 100, 201],
                         labels=["1-10", "11-30", "31-100", "101-201"])
        D[t] = d
        print(f"[{t}] {len(d)}행 · {d.D.nunique()}일 ({d.D.min()} ~ {d.D.max()})")
    if not D:
        return

    # 타깃별 진행이 다를 수 있으므로 공통 날짜로만 비교 (기간 차이가 결론을 오염시킨다)
    common = set.intersection(*[set(d.D.unique()) for d in D.values()])
    D = {t: d[d.D.isin(common)].copy() for t, d in D.items()}
    print(f"\n▶ 공통 비교 구간 {len(common)}일 ({min(common)} ~ {max(common)})")

    print("\n" + "=" * 120)
    print("【1】 p 십분위별 갭 중앙값 (%)   ※ 랭크가 목표를 담으면 1→10 단조 증가해야 한다")
    print("=" * 120)
    tb = pd.DataFrame({NAME[t]: d.groupby("dec").gap.median() * 100 for t, d in D.items()}).T
    print(tb.round(3).to_string())
    print("\n  갭 평균 (%)")
    print(pd.DataFrame({NAME[t]: d.groupby("dec").gap.mean() * 100 for t, d in D.items()}).T.round(3).to_string())

    print("\n" + "=" * 120)
    print("【2】 단조성·판별력")
    print("=" * 120)
    res = []
    for t, d in D.items():
        dm = d.groupby("dec").gap.median()
        ic = d.groupby("D").apply(lambda x: stats.spearmanr(x.p, x.gap).correlation,
                                  include_groups=False).dropna()
        y = (d.gap > d.groupby("D").gap.transform("median")).astype(int)
        auc = d.assign(_y=y).groupby("D").apply(lambda x: _auc(x._y, x.p),
                                                include_groups=False).dropna()
        icv = d.groupby("D").apply(lambda x: stats.spearmanr(x.p, x.YZ_20).correlation,
                                   include_groups=False).dropna()
        res.append(dict(타깃=NAME[t], 십분위스피어만=stats.spearmanr(dm.index, dm.values).correlation,
                        IC평균=ic.mean(), IC_p=stats.ttest_1samp(ic, 0).pvalue,
                        IC양수일=(ic > 0).mean() * 100, AUC평균=auc.mean(),
                        AUC_p=stats.ttest_1samp(auc - .5, 0).pvalue,
                        p_YZ상관=icv.mean()))
    print(pd.DataFrame(res).round(4).to_string(index=False))
    print("  p_YZ상관 = p ↔ YZ_20 일별 스피어만 평균 (높을수록 랭크가 변동성 순위표에 가깝다)")

    print("\n" + "=" * 120)
    print("【3】 rank 구간별 변동성 백분위 (그날 유니버스 내 0~100, 중앙값)")
    print("=" * 120)
    print(pd.DataFrame({NAME[t]: d.groupby("rb", observed=True).vpct.median()
                        for t, d in D.items()}).T.round(1).to_string())

    print("\n" + "=" * 120)
    print("【4】 변동성 분위 내부 IC (갭 기준) — 통제해도 효과가 남는가")
    print("=" * 120)
    out = {}
    for t, d in D.items():
        out[NAME[t]] = {int(q): d[d.vq == q].groupby("D").apply(
            lambda x: stats.spearmanr(x.p, x.gap).correlation if len(x) > 5 else np.nan,
            include_groups=False).dropna().mean() for q in sorted(d.vq.dropna().unique())}
    print(pd.DataFrame(out).T.round(4).to_string())

    print("\n" + "=" * 120)
    print("【5】 rank ≤ k 상위 선별 (%)   ※ 갭 비용후 = t종가매수→t+1시가매도, 나머지 비용전")
    print("=" * 120)
    rows = []
    for t, d in D.items():
        for k in [5, 10, 15, 30]:
            s = d[d["rank"] <= k]
            net = (1 + s.gap) * (1 - SELL_FEE) / (1 + BUY_FEE) - 1
            rows.append(dict(타깃=NAME[t], k=k, n=len(s), 갭평균=s.gap.mean() * 100,
                             갭비용후=net.mean() * 100, 익일종가=s.day.mean() * 100,
                             장중=s.intra.mean() * 100, 변동성백분위=s.vpct.median()))
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # ── 워크포워드 ↔ 라이브 대조 ─────────────────────────────
    w = D.get("y_rel")
    if w is None:
        return
    live = load_live()
    x = w.merge(live, left_on=["D", "Ticker"], right_on=["D", "ticker"], how="inner")
    if len(x):
        print("\n" + "=" * 120)
        print(f"【6】 워크포워드 ↔ 라이브 신호 대조 — 겹치는 {x.D.nunique()}일 · {len(x)}행")
        print("=" * 120)
        ic = x.groupby("D").apply(lambda g: stats.spearmanr(g.p, g.p_live).correlation,
                                  include_groups=False)
        ov = x.groupby("D").apply(lambda g: len(set(g.nsmallest(10, "rank").Ticker)
                                                & set(g.nsmallest(10, "rank_live").Ticker)),
                                  include_groups=False)
        print(f"  p 순위상관(일별) 평균 {ic.mean():+.3f} (최저 {ic.min():+.3f}) | "
              f"상위10 겹침 평균 {ov.mean():.1f}/10")
        print("  ※ 완전 일치하지 않는 것은 매일 재학습의 무작위성 — 같은 모델임을 확인하는 용도")

    print("\n" + "=" * 120)
    print("【7】 월별 추이 + 검정력 — 짧은 표본으로 판단하면 무엇을 놓치는가 (현행 y_rel)")
    print("=" * 120)
    w = w.assign(ym=w.D.str[:7])
    rows = []
    for ym, g in w.groupby("ym"):
        ic = g.groupby("D").apply(lambda d: stats.spearmanr(d.p, d.gap).correlation,
                                  include_groups=False).dropna()
        a = g[g["rank"] <= 10].groupby("D").gap.mean(); b = g.groupby("D").gap.mean()
        rows.append(dict(월=ym, 일수=g.D.nunique(), IC=ic.mean(), IC음수일=int((ic < 0).sum()),
                         초과갭=(a - b).mean() * 100,
                         t_p=stats.ttest_rel(a, b).pvalue if len(a) > 2 else np.nan))
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    ic = w.groupby("D").apply(lambda d: stats.spearmanr(d.p, d.gap).correlation,
                              include_groups=False).dropna()
    dif = (w[w["rank"] <= 10].groupby("D").gap.mean() - w.groupby("D").gap.mean()).dropna()
    print(f"\n  일별 IC: 평균 {ic.mean():+.3f} · 음수일 {int((ic < 0).sum())}일 "
          f"({(ic < 0).mean()*100:.1f}%) · 5~95% {ic.quantile(.05):+.2f}~{ic.quantile(.95):+.2f}")
    print(f"  rank≤10 초과갭: 평균 {dif.mean()*100:+.3f}%p · 음수일 {(dif < 0).mean()*100:.1f}%")
    rng = np.random.default_rng(7)
    for nm, pop in [("IC", ic.values), ("rank≤10 초과갭", dif.values)]:
        for n in [20, 24, 60]:
            hit = np.mean([stats.ttest_1samp(rng.choice(pop, n, replace=True), 0).pvalue < 0.05
                           for _ in range(2000)])
            print(f"    {nm:14s} {n:3d}일 표본에서 유의(p<0.05) 검출 확률 {hit*100:.0f}%")


if __name__ == "__main__":
    main()
