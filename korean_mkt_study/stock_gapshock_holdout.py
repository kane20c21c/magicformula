# -*- coding: utf-8 -*-
"""stock_gapshock_holdout.py — 6단계 보류 구간 확인 (케인 2026-11-07 19:25 "진행해줘").

보류 구간  2026-06-01 ~ (data_2015 끝 2026-10-29, 20거래일 경로가 있는 쇼크만 → 쇼크일 ~2026-09 말까지)
           학습 구간 결과를 보고 고친 것은 없다 — 아래 목록은 보류 구간을 열기 전(2026-11-07 19:30)에 고정.

판정 목록 (방향 = 학습 구간 부호)
  [견고]  K1 종목 단독 그 자체: a20 < 0 · xb20 < 0
          K2 종목 시장 동반 − 단독: a20 · xb20 · r10 · r20 · hold10 > 0
          K3 종목 섹터 동반 ① − ②: r10 · r20 · hold10 · dd10 < 0
          K4 종목 단독 ⑤ − ②: r10 · r20 < 0
  [부분]  P1 종목 시장 동반 ③ − 나머지: hold10 < 0      P2 섹터 시장 동반 ① − ②: r10 · r20 > 0
          P3 지수 ③ − 나머지: f20 · r20 < 0               P4 종목 시장 동반 ⑤ − ②: r20 > 0
          P5 종목 시장 동반 그 자체: a20 > 0 (학습에서 몰림날 의존)
  판정    확인 = 같은 부호 & 날짜 묶음 95% 구간이 0 밖 · 같은 방향 = 같은 부호지만 구간이 0 포함 · 반대 = 부호 반대 · 표본 없음
  게이트   문서 §4 (2026-11-06 국면 게이트 v2) — 하루 −3% 쇼크 종류별 게이트 · 시장 브레드스 ±0.05 · 평소 보유. 보류 구간 수익·MDD·보유 비중.
          확인용 한 줄 추가: 갭 몫 0.4~1 (⑤ 갭→반등 제외 — 2026-11-07 형태 분류에서 나온 구분)
산출  out/regime/holdout_contrasts.csv · holdout_gate.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import stock_gapshock_robust as R
OUT = HERE / "out" / "regime"; HOLD = pd.Timestamp("2026-06-01")

LIST = [  # (묶음, 이름, 범위, A, B, 지표들, 기대 부호)
    ("견고", "K1 종목 단독 그 자체", "st", ("ctx", "단독"), None, ["a20", "xb20"], -1),
    ("견고", "K2 종목 시장 동반 − 단독", "st", ("ctx", "시장 동반"), ("ctx", "단독"), ["a20", "xb20", "r10", "r20", "hold10"], +1),
    ("견고", "K3 종목 섹터 동반 ① − ②", "st", ("sh", "①", "섹터 동반"), ("sh", "②", "섹터 동반"), ["r10", "r20", "hold10", "dd10"], -1),
    ("견고", "K4 종목 단독 ⑤ − ②", "st", ("sh", "⑤", "단독"), ("sh", "②", "단독"), ["r10", "r20"], -1),
    ("부분", "P1 종목 시장 동반 ③ − 나머지", "st", ("sh", "③", "시장 동반"), ("not", "③", "시장 동반"), ["hold10"], -1),
    ("부분", "P2 섹터 시장 동반 ① − ②", "se", ("sh", "①", "시장 동반"), ("sh", "②", "시장 동반"), ["r10", "r20"], +1),
    ("부분", "P3 지수 ③ − 나머지", "ix", ("sh", "③", None), ("not", "③", None), ["f20", "r20"], -1),
    ("부분", "P4 종목 시장 동반 ⑤ − ②", "st", ("sh", "⑤", "시장 동반"), ("sh", "②", "시장 동반"), ["r20"], +1),
    ("부분", "P5 종목 시장 동반 그 자체", "st", ("ctx", "시장 동반"), None, ["a20"], +1),
]

def sel(spec):
    if spec is None: return None
    if spec[0] == "ctx": return lambda s: s.ctx == spec[1]
    t, c = spec[1], spec[2]
    if spec[0] == "sh": return lambda s: (s["T"] == t) & ((s.ctx == c) if c else True)
    return lambda s: (s["T"] != t) & ((s.ctx == c) if c else True)

def contrasts():
    tr = dict(zip(("st", "se", "ix"), R.load(False)[:3])); ho = dict(zip(("st", "se", "ix"), R.load(True)[:3]))
    for d in (tr["ix"], ho["ix"]): d["ctx"] = "지수"
    rows = []
    for grp, name, sc, a, b, cols, sgn in LIST:
        A, B = sel(a), sel(b); h = ho[sc]
        ci = R.boot(h, A, B, cols) if A(h).sum() and (B is None or B(h).sum()) else {}
        for c in cols:
            vt, vh = R.stat(tr[sc], A, B, c), R.stat(h, A, B, c)
            lo, hi = ci.get(c, (np.nan, np.nan))
            if not np.isfinite(vh): v = "표본 없음"
            elif np.sign(vh) != sgn: v = "반대"
            else: v = "확인" if (lo > 0 or hi < 0) else "같은 방향"
            rows.append({"묶음": grp, "대비": name, "지표": c, "기대 부호": "+" if sgn > 0 else "−", "학습": round(vt * 100, 1), "보류": round(vh * 100, 1) if np.isfinite(vh) else np.nan,
                         "보류 95% 하한": round(lo * 100, 1), "보류 95% 상한": round(hi * 100, 1), "보류 A 건수": int(A(h).sum()),
                         "보류 B 건수": int(B(h).sum()) if B else np.nan, "보류 A 날짜": int(h.loc[A(h)].date.nunique()), "판정": v})
    t = pd.DataFrame(rows); t.to_csv(OUT / "holdout_contrasts.csv", index=False, encoding="utf-8-sig")
    print("보류 구간 쇼크 수 — 종목", len(ho["st"]), "(날짜", ho["st"].date.nunique(), ") · 섹터", len(ho["se"]), "· 지수", len(ho["ix"]),
          "| 종목 맥락", ho["st"].ctx.value_counts().to_dict(), "| 마지막 쇼크일", ho["st"].date.max().date())
    pd.set_option("display.width", 260); print(t.to_string(index=False)); return t

def gate():
    import regime_gate as G
    lv = pd.read_parquet(OUT / "layer_levels.parquet"); S = pd.read_parquet(OUT / "detect_states.parquet"); k = lv["코스피 200"]
    stt = lambda c: S[c].reindex(k.index).ffill().fillna(True).astype(bool)
    hold = pd.Series(True, index=k.index); br = stt(G.BR["시장±0.05"])
    D = {"단순 보유": hold, "(참고) SMA120": stt("K200 120일선"), "(참고) C2 괴리형+10%복귀": stt("[조합 C2] 괴리형 퇴출 → 퇴출 뒤 저점 +10% 반등 복귀")}
    shocks = {}
    for kind in ("−3% 전부", "갭 주도+섞임 (갭 몫≥40%)", "장중 주도 (갭 몫≤40%)"):
        gt, sh = G.gate_series(k, kind); D[f"게이트 {kind}"] = G.gated2(gt, br, hold); shocks[kind] = sh
    d = pd.read_parquet(HERE / "data_1104" / "kospi200_daily.parquet")[["Open", "Close"]].astype(float); d.index = pd.to_datetime(d.index); d = d.reindex(k.index)
    cp = d.Close.shift(1); r = d.Close / cp - 1; share = np.log1p(d.Open / cp - 1) / np.log1p(r)
    sh5 = (r <= -0.03) & (share >= 0.4) & (share <= 1.0); gt5 = (sh5.astype(int).rolling(10, min_periods=1).max() > 0).to_numpy()
    D["게이트 갭 몫 0.4~1 (⑤ 제외, 확인용)"] = G.gated2(gt5, br, hold); shocks["갭 몫 0.4~1"] = sh5
    w = k.index >= HOLD; kr = k.pct_change().fillna(0); rows = []
    for nm, s in D.items():
        pos = s.shift(1).fillna(True).astype(bool)[w]; rr = kr[w].where(pos, 0.0); eq = (1 + rr).cumprod()
        sw = s[w]; rows.append({"판정기": nm, "보류 구간 수익%": (eq.iloc[-1] - 1) * 100, "MDD%": ((eq / eq.cummax()) - 1).min() * 100,
                                "보유 비중%": pos.mean() * 100, "전환 횟수": int((sw != sw.shift()).iloc[1:].sum())})
    T = pd.DataFrame(rows); T.to_csv(OUT / "holdout_gate.csv", index=False, encoding="utf-8-sig")
    print(f"\n게이트 보류 구간 {k.index[w].min().date()} ~ {k.index[w].max().date()} ({w.sum()}거래일)"); print(T.round(1).to_string(index=False))
    for kind, sh in shocks.items():
        x = sh[w & sh.fillna(False).to_numpy()]
        print(f"  쇼크 [{kind}] {len(x)}일:", ", ".join(f"{i.date()}({share[i]:.2f})" for i in x.index))
    # 상태 타임라인 (퇴출 구간)
    for nm in [n for n in D if n.startswith("게이트")]:
        s = D[nm][w]; out = s[~s]; print(f"  {nm} 현금 {len(out)}일:", (f"{out.index.min().date()}~{out.index.max().date()}" if len(out) else "없음"))

if __name__ == "__main__":
    contrasts(); gate()
