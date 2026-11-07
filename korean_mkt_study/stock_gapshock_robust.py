# -*- coding: utf-8 -*-
"""stock_gapshock_robust.py — 5단계 견고성 검증 (케인 2026-11-07 14:16: '5단계로 검증해서 robust 한 결과를 만들고, 그 뒤 투자 시사점').

입력  out/regime/stock_shock_events_ctx.parquet · sector_shock_events.parquet · index_shock_events.parquet (stock_gapshock.py 산출)
대비  2~3단계에서 나온 주장들을 고정 목록으로 (결과를 보고 고르지 않도록 여기 먼저 적는다)
검증  ① 날짜 묶음 부트스트랩 95% 구간 (같은 날 쇼크 = 한 덩어리, 600회)
      ② 시기 3개 부호 일치 — 종목·섹터 2015–19 / 2020–22 / 2023–26.5, 지수 2000–07 / 2008–14 / 2015–26.5
      ③ 쇼크가 가장 많이 몰린 10개 날짜(종목 기준) ±5거래일 제외 후 부호 유지
판정  견고 = ①이 0 밖 & ② 세 시기 같은 부호 & ③ 같은 부호 · 부분 = ①이 0 밖이지만 ②나 ③에서 한 번 뒤집힘(또는 시기 칸 30건 미만) · 약함 = ①이 0 포함
지표  a20 = 다음 날 시가 진입 → +20일 종가, 왕복 0.23% 차감 (종목만) · xb20 = 베타 보정 +20일 초과 (종목) · x20 = 코스피200 대비 +20일 초과 (섹터)
      f20 = +20일 수익 (쇼크일 종가 기준) · r10/r20 = 10/20거래일 안 쇼크 전날 종가 회복 (종가 기준)
      hold10 = 쇼크일 저가 10일 유지 (종가 기준, ⚠ 쿠션 편향) · dd10 = 10일 최대 추가 하락 (쇼크일 종가 → 이후 10일 최저 종가, 중앙값)
      (2026-11-07 케인: 결과 지표 묶음을 r10·r20·hold10·dd10 으로 — 수익 지표는 그대로 유지)
보류 구간(2026-06-01~)은 쓰지 않는다.
"""
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; OUT = HERE / "out" / "regime"; B = 600; RNG = np.random.default_rng(20261107)
T5 = lambda s: pd.cut(s, [-np.inf, 0, .4, .6, 1, np.inf], labels=list("①②③④⑤")).astype(str)

def load(holdout=False):
    st = pd.read_parquet(OUT / "stock_shock_events_ctx.parquet"); st = st[(st.hold == holdout) & (st.ctx != "맥락 미상")].copy()
    st["T"] = T5(st.share); st["S"] = pd.cut(st.r, [-1, -0.10, -0.07, -0.05], labels=["큰", "중간", "작은"]).astype(str)
    st["a20"] = (1 + st.f20) / (1 + st.next_gap) - 1 - 0.0023
    st["r20"] = (st.recover.fillna(61) <= 20).astype(float); st["hold10"] = st.hold10.astype(float); st["r10"] = (st.recover.fillna(61) <= 10).astype(float)
    st["era"] = pd.cut(st.date.dt.year, [2014, 2019, 2022, 2026], labels=["15–19", "20–22", "23–26.5"]).astype(str)
    se = pd.read_parquet(OUT / "sector_shock_events.parquet"); se = se[se.hold == holdout].copy()
    se["T"] = T5(se.share); se["ctx"] = np.where(se.k_r <= -0.02, "시장 동반", "섹터 단독")
    se["S"] = pd.cut(se.r, [-1, -0.06, -0.045, -0.03], labels=["큰", "중간", "작은"]).astype(str)
    se["r20"] = (se.recover.fillna(61) <= 20).astype(float); se["hold10"] = se.hold10.astype(float); se["r10"] = (se.recover.fillna(61) <= 10).astype(float)
    se["era"] = pd.cut(se.date.dt.year, [2014, 2019, 2022, 2026], labels=["15–19", "20–22", "23–26.5"]).astype(str)
    ix = pd.read_parquet(OUT / "index_shock_events.parquet"); ix = ix[ix.hold == holdout].copy()
    ix["T"] = T5(ix.share); ix["r20"] = (ix.recover.fillna(61) <= 20).astype(float); ix["hold10"] = ix.hold10.astype(float); ix["r10"] = (ix.recover.fillna(61) <= 10).astype(float)
    ix["era"] = pd.cut(ix.date.dt.year, [1999, 2007, 2014, 2026], labels=["00–07", "08–14", "15–26.5"]).astype(str)
    # 몰림 날짜: 종목 쇼크 수 상위 10일 ±5거래일
    cal = pd.Index(sorted(pd.read_parquet(HERE / "data_2015" / "kospi200_daily.parquet").index.map(pd.Timestamp)))
    top = st.date.value_counts().head(10).index; pos = cal.get_indexer(top)
    near = set(cal[np.clip(np.concatenate([pos + k for k in range(-5, 6)]), 0, len(cal) - 1)])
    for d in (st, se, ix): d["crowd"] = d.date.isin(near)
    return st, se, ix, sorted(top)

def stat(sub, a, b, col):
    """b 가 None 이면 a 집단의 값 자체(0 대비). 수익 = 중앙값, 비율 = 평균."""
    f = np.nanmedian if col in ("a20", "xb20", "x20", "f20", "dd10") else np.nanmean
    va = sub.loc[a(sub), col].to_numpy()
    if b is None: return f(va) if len(va) else np.nan
    vb = sub.loc[b(sub), col].to_numpy()
    return f(va) - f(vb) if len(va) and len(vb) else np.nan

def boot(sub, a, b, cols):
    d = sub.date.to_numpy(); u, inv = np.unique(d, return_inverse=True)
    order = np.argsort(inv, kind="stable"); cuts = np.r_[0, np.cumsum(np.bincount(inv))]
    groups = [order[cuts[i]:cuts[i + 1]] for i in range(len(u))]
    res = {c: [] for c in cols}
    for _ in range(B):
        ii = np.concatenate([groups[k] for k in RNG.integers(0, len(u), len(u))]); s = sub.iloc[ii]
        for c in cols: res[c].append(stat(s, a, b, c))
    return {c: np.nanpercentile(v, [2.5, 97.5]) for c, v in res.items()}

def run_contrast(name, df, a, b, cols, eras):
    ci = boot(df, a, b, cols); rows = []
    for c in cols:
        pt = stat(df, a, b, c); lo, hi = ci[c]; sgn = np.sign(pt)
        ev = {e: stat(df[df.era == e], a, b, c) for e in eras}
        en = {e: int(a(df[df.era == e]).sum()) for e in eras}
        nc = stat(df[~df.crowd], a, b, c)
        out0 = lo > 0 or hi < 0
        era_ok = all(np.sign(v) == sgn for v in ev.values() if np.isfinite(v)) and all(np.isfinite(v) for v in ev.values())
        thin = min(en.values()) < 30
        verdict = "견고" if out0 and era_ok and np.sign(nc) == sgn and not thin else ("부분" if out0 else "약함")
        sc = 100
        rows.append({"대비": name, "지표": c, "전체": round(pt * sc, 1), "95%구간": f"[{lo*sc:+.1f}, {hi*sc:+.1f}]",
                     **{f"시기 {e}": round(v * sc, 1) if np.isfinite(v) else np.nan for e, v in ev.items()},
                     "시기별 A건수": "/".join(str(en[e]) for e in eras), "몰림날 제외": round(nc * sc, 1), "판정": verdict})
    return rows

def main():
    st, se, ix, top = load(); E3 = ["15–19", "20–22", "23–26.5"]; EI = ["00–07", "08–14", "15–26.5"]
    print("몰림 상위 10일:", [d.date().isoformat() for d in top])
    ctx = lambda c: (lambda s: s.ctx == c); sh = lambda t, c=None: (lambda s: (s["T"] == t) & ((s.ctx == c) if c else True))
    notsh = lambda t, c=None: (lambda s: (s["T"] != t) & ((s.ctx == c) if c else True))
    rows = []
    R = lambda *a: rows.extend(run_contrast(*a))
    # ── 종목
    R("종목 | 시장 동반 − 단독", st, ctx("시장 동반"), ctx("단독"), ["a20", "xb20", "r10", "r20", "hold10", "dd10"], E3)
    R("종목 | 섹터 동반 − 단독", st, ctx("섹터 동반"), ctx("단독"), ["a20", "xb20", "r10", "r20", "hold10", "dd10"], E3)
    R("종목 | 단독 그 자체 (0 대비)", st, ctx("단독"), None, ["a20", "xb20"], E3)
    R("종목 | 시장 동반 그 자체 (0 대비)", st, ctx("시장 동반"), None, ["a20", "xb20"], E3)
    R("종목 | 시장 동반: 큰 − 작은", st, lambda s: (s.ctx == "시장 동반") & (s.S == "큰"), lambda s: (s.ctx == "시장 동반") & (s.S == "작은"), ["a20", "xb20", "r10", "r20", "hold10", "dd10"], E3)
    for c in ("시장 동반", "단독", "섹터 동반"):
        R(f"종목 | {c}: ① − ②", st, sh("①", c), sh("②", c), ["xb20", "r10", "r20", "hold10", "dd10"], E3)
        R(f"종목 | {c}: ⑤ − ②", st, sh("⑤", c), sh("②", c), ["xb20", "r10", "r20", "hold10", "dd10"], E3)
        R(f"종목 | {c}: ③ − 나머지", st, sh("③", c), notsh("③", c), ["xb20", "r10", "r20", "hold10", "dd10"], E3)
    # ── 섹터 (업종지수)
    R("섹터 | 시장 동반 − 섹터 단독", se, ctx("시장 동반"), ctx("섹터 단독"), ["x20", "f20", "r10", "r20", "hold10", "dd10"], E3)
    for c in ("시장 동반", "섹터 단독"):
        R(f"섹터 | {c}: ① − ②", se, sh("①", c), sh("②", c), ["x20", "r10", "r20", "hold10", "dd10"], E3)
        R(f"섹터 | {c}: ⑤ − ②", se, sh("⑤", c), sh("②", c), ["x20", "r10", "r20", "hold10", "dd10"], E3)
        R(f"섹터 | {c}: ③ − 나머지", se, sh("③", c), notsh("③", c), ["x20", "r10", "r20", "hold10", "dd10"], E3)
    # ── 지수
    ix["ctx"] = "지수"
    R("지수 | ① − ②", ix, sh("①"), sh("②"), ["f20", "r10", "r20", "hold10", "dd10"], EI)
    R("지수 | ⑤ − ②", ix, sh("⑤"), sh("②"), ["f20", "r10", "r20", "hold10", "dd10"], EI)
    R("지수 | ③ − 나머지", ix, sh("③"), notsh("③"), ["f20", "r10", "r20", "hold10", "dd10"], EI)
    R("지수 | 큰(≤−3%) − 작은", ix, lambda s: s.r <= -0.03, lambda s: s.r > -0.03, ["f20", "r10", "r20", "hold10", "dd10"], EI)
    t = pd.DataFrame(rows); t.to_csv(OUT / "robust_contrasts.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300); pd.set_option("display.max_colwidth", 40)
    print(t.drop(columns=[c for c in t.columns if c.startswith("시기 ")]).to_string(index=False))
    print(); print(t[["대비", "지표"] + [c for c in t.columns if c.startswith("시기 ")]].to_string(index=False))

if __name__ == "__main__":
    main()
