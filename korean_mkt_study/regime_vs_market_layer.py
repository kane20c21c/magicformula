# -*- coding: utf-8 -*-
"""regime_vs_market_layer.py — 실습 A '방향 × 소음' 국면 vs 통일장 시장 층 국면 비교 (성과지표 스터디, 케인 2026-11-18~19).

목적   실습 A 의 2×2 국면(방향 × 소음)과 통일장 regime_history 를 나란히 놓고, 특히 '소음(변동성) 축'이 통일장 국면
       안에서 무엇을 더 구분하는지 본다. **서술용** — 이 결과로 market_layer 를 고치지 않는다 (표기도 검증 뒤 결정, 케인 11-19).
내 국면 (regime.py · regime_filter.py 와 같은 정의 — 아래 재현 검산이 2015-01~2026-11-04 이전 결과와 같은지 확인한다)
  가격   data_idx/krx_indices.parquet 코스피 200 종가 + kospi200_2000_2014.parquet (준비 구간)
  방향   종가 vs 120일 SMA, 완충대 ±2% (사이는 직전 상태 유지)
  소음   r 20일 σ×√252 vs 확장창 중앙값(min 250, 그때까지의 과거만), 완충대 ±10%
  성과   오늘 국면 → 다음 날 수익률. 종료일 이후 자료는 계산에 넣지 않는다 (종료일의 다음 날 결과도 없음).
통일장  output/market_layer/regime_history.parquet 를 **읽기만** 한다 (market_layer/README '다른 프로젝트에서 쓰기').
정답지  ② 이상적 행동일 기준 (케인 11-19): 6/23~25 현금 확대 · 8/3~8/5 비중 확대 · 10/8 현금 확대, 목표 ±2거래일.
        가격 꼭짓점(6/22·7/30·9/23)은 어차피 생기고, 이상적 행동일이 나와야 확정된다는 판단.
        편차 = 그 축이 기대 상태로 들어간 날 − 행동 창(가까운 끝), 거래일. 음수 = 이름(이미 그 상태), 양수 = 늦음.
⚠ 정답지 세 점과 2026-06~ 은 통일장 보류 구간 — 이 비교는 이미 보류 구간을 본 서술이다. 판정 기준으로 쓰지 말 것.
⚠ 구간 길이는 앞·뒤 검열 구간을 뺀다 (통일장 = _cens == "").
⚠ 전환 짝짓기 창은 근거 있는 단일값이 없어 여러 길이(5~120)를 함께 본다. 짝은 같은 방향끼리 1:1, 가까운 것부터.
실행   python3 korean_mkt_study/regime_vs_market_layer.py [--end 2026-11-16] [--start 2016-04-22]
출력   표준출력 + out/regime/regime_vs_market_layer_<end>.txt
노트   옵시디언 30_NoteforStock 「2026-11-19 실습 A 국면 vs 통일장 시장 층 — 소음 축은 무엇을 더 구분하나」
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
IDX = HERE / "data_idx"
H_PATH = HERE.parent / "output" / "market_layer" / "regime_history.parquet"
OUT = HERE / "out" / "regime"
LBL = {(1, 0): "조용한 상승", (1, 1): "시끄러운 상승", (0, 0): "조용한 하락", (0, 1): "시끄러운 하락"}
ORDER = ["조용한 상승", "시끄러운 상승", "조용한 하락", "시끄러운 하락"]
RG = ["상승·보유", "상승·퇴출", "하락·보유", "하락·퇴출"]
LEARN_END = pd.Timestamp("2026-05-29")          # 통일장 학습 끝 — 2026-06~ 보류
WINDOWS = (5, 10, 20, 40, 60, 120)
# 정답지 ② 이상적 행동일 (케인 11-19): (이름, 창 시작, 창 끝, 기대 상태 True=상승/보유)
KEY = [("6/22 고점 → 현금 확대", "2026-06-23", "2026-06-25", False),
       ("7/30 저점 → 비중 확대", "2026-08-03", "2026-08-05", True),
       ("10월 고점 → 현금 확대", "2026-10-08", "2026-10-08", False)]
KEY_TOL = 2
# 재현 검산 기준값 — 실습 A 노트(2026-11-18) 4·6절, 2015-01-02~2026-11-04 완충대
PRIOR = dict(n=72, share=[39.2, 20.1, 23.5, 17.1], mean=[63.3, 34.4, 36.0, 27.6], sharpe=[0.94, 0.89, -0.64, 0.83])


class Tee:
    def __init__(self, path): self.f = open(path, "w", encoding="utf-8"); self.o = sys.stdout
    def write(self, s): self.o.write(s); self.f.write(s)
    def flush(self): self.o.flush(); self.f.flush()


# ── 재료 ──────────────────────────────────────────────────────────────────────
def load_price(end):
    old = pd.read_parquet(IDX / "kospi200_2000_2014.parquet")
    old = old.reset_index() if "Date" not in old.columns else old
    old["Date"] = pd.to_datetime(old["Date"]); old = old.set_index("Date").Close.astype(float)
    k = pd.read_parquet(IDX / "krx_indices.parquet")
    new = k[k.name == "코스피 200"].set_index("date").sort_index().close.astype(float); new.index = pd.to_datetime(new.index)
    px = pd.concat([old, new]).sort_index(); px = px[~px.index.duplicated()]
    return old, new, px.loc[:end]


def hyst(x, lo, hi, init):
    out, s = [], init
    for a, l, h in zip(x, lo, hi):
        if a > h: s = 1
        elif a < l: s = 0
        out.append(s)
    return pd.Series(out, index=x.index)


def my_regime(px):
    """완충대 방향·소음 (regime_filter.make('hyst') 와 같다). 반환: 방향(1/0), 소음(1/0), 4국면 라벨, 다음 날 수익률."""
    r = px.pct_change()
    ma = px.rolling(120).mean(); vol = r.rolling(20).std() * np.sqrt(252); med = vol.expanding(min_periods=250).median()
    ok = ma.notna() & vol.notna() & med.notna()
    P, M, V, T = px[ok], ma[ok], vol[ok], med[ok]
    raw_up, raw_nz = (P > M).astype(int), (V > T).astype(int)
    up = hyst(P, M * 0.98, M * 1.02, raw_up.iloc[0]); nz = hyst(V, T * 0.90, T * 1.10, raw_nz.iloc[0])
    st = pd.Series([LBL[(int(u), int(n))] for u, n in zip(up, nz)], index=up.index)
    return up, nz, st, r.shift(-1)


def spells(s):
    s = s.dropna(); g = (s != s.shift()).cumsum()
    return pd.DataFrame({"s": s.groupby(g).first(), "a": s.index.to_series().groupby(g).first(),
                         "b": s.index.to_series().groupby(g).last(), "len": s.groupby(g).size()})


def inner_spells(s):
    sp = spells(s); return sp.iloc[1:-1] if len(sp) > 2 else sp.iloc[0:0]      # 앞·뒤 검열 구간 제외


def perf(nxt, mask):
    y = nxt[mask[mask].index].dropna(); n = len(y); yrs = n / 252
    if n < 2: return dict(일수=n, 연수=yrs, 연수익=np.nan, 연σ=np.nan, Sharpe=np.nan, SE=np.nan)
    return dict(일수=n, 연수=yrs, 연수익=y.mean() * 252 * 100, 연σ=y.std() * np.sqrt(252) * 100,
                Sharpe=y.mean() / y.std() * np.sqrt(252), SE=1 / np.sqrt(yrs))


# ── 1. 데이터 확인 · 재현 검산 ────────────────────────────────────────────────────
def check(old, new, px, H, st, nxt, start, end):
    print(f"=== 1. 데이터 확인 (공통 구간 {start.date()} ~ {end.date()}) ===")
    print(f"접합: 2014 마지막 {old.index[-1].date()} {old.iloc[-1]:.2f} / 2015 첫날 {new.index[0].date()} {new.iloc[0]:.2f}")
    print(f"krx_indices 코스피 200: {new.index[0].date()} ~ {new.index[-1].date()} {len(new)}일, 결측 {int(new.isna().sum())}"
          f" (종료일 뒤 {int((new.index > end).sum())}일은 계산 제외)")
    print(f"regime_history: {H.index[0].date()} ~ {H.index[-1].date()} {len(H)}행")
    for c in ["breadth", "c2", "regime", "p7", "k200_close"]:
        v = H[c].dropna(); print(f"  {c:10s} 결측 {int(H[c].isna().sum()):4d}  유효 {v.index[0].date()} ~ {v.index[-1].date()}")
    w = H.loc[start:end]
    print(f"공통 구간 결측: regime {int(w.regime.isna().sum())} · p7 {int(w.p7.isna().sum())} · breadth {int(w.breadth.isna().sum())} · c2 {int(w.c2.isna().sum())}")
    a, b = set(px.loc[start:end].index), set(w.index)
    print(f"날짜축: 코스피 200 에만 {len(a - b)}일 / 통일장 에만 {len(b - a)}일")
    j = pd.concat([px.rename("krx"), H.k200_close.rename("h")], axis=1).dropna(); d = (j.krx - j.h).abs()
    print(f"k200_close vs krx 종가: 겹침 {len(j)}일, 최대 차 {d.max():.4f}, 0.005 초과 {int((d > 0.005).sum())}일")
    if end < pd.Timestamp("2026-11-04"): print("재현 검산 건너뜀 (종료일이 2026-11-04 앞)"); return
    s = st.loc["2015-01-02":"2026-11-04"]; nx = nxt.copy(); nx.loc["2026-11-04"] = np.nan; sp = spells(s)
    got = dict(n=len(sp), share=[float(round((s == x).mean() * 100, 1)) for x in ORDER],
               mean=[float(round(sp[sp.s == x].len.mean(), 1)) for x in ORDER],
               sharpe=[float(round(nx[s[s == x].index].dropna().mean() / nx[s[s == x].index].dropna().std() * np.sqrt(252), 2)) for x in ORDER])
    okk = got == PRIOR
    print(f"재현 검산 (2015-01-02~2026-11-04, 실습 A 노트 값): {'✓ 일치' if okk else '✗ 불일치'}  구간 {got['n']} · 비중 {got['share']} · "
          f"평균 {got['mean']} · Sharpe {got['sharpe']}")
    if not okk: print(f"  ⚠ 기대값 {PRIOR}")
    last = spells(st).iloc[-1]; print(f"{end.date()} 현재 내 국면: {last.s} {last.len}거래일째 ({last.a.date()}~)")


# ── 2. 방향끼리 ────────────────────────────────────────────────────────────────
def transitions(s, pos):
    ch = s.ne(s.shift()); ch.iloc[0] = False
    return pd.DataFrame({"pos": pos[ch[ch].index].values, "to": s[ch].values}, index=ch[ch].index)


def match(A, B, W):
    """같은 방향 전환끼리 |지연| ≤ W 안에서 1:1, 가까운 것부터. lag = B − A (거래일, 음수 = B 가 먼저)."""
    cand = [(abs(rb.pos - ra.pos), rb.pos - ra.pos, ia, ib, ra.to)
            for ia, ra in A.iterrows() for ib, rb in B.iterrows() if ra.to == rb.to and abs(rb.pos - ra.pos) <= W]
    cand.sort(key=lambda x: x[0]); ua, ub, out = set(), set(), []
    for _, lag, ia, ib, to in cand:
        if ia in ua or ib in ub: continue
        ua.add(ia); ub.add(ib); out.append((ia, ib, lag, to))
    return pd.DataFrame(out, columns=["A", "B", "lag", "to"])


def direction(D, pos):
    print(f"\n=== 2. 방향끼리 비교 ({len(D)}일) ===")
    pairs = [("내방향", "breadth"), ("내방향", "c2"), ("breadth", "c2")]
    f = lambda v: "상승/보유" if v else "하락/퇴출"
    for a, b in pairs:
        print(f"\n[{a} vs {b}] 일치율 {(D[a] == D[b]).mean() * 100:.1f}%")
        print(pd.crosstab(D[a].map(f), D[b].map(f)).to_string())
    TR = {c: transitions(D[c], pos) for c in ["내방향", "breadth", "c2", "p7", "내소음"]}
    print("\n전환 수:", {c: len(t) for c, t in TR.items()})
    print("\n전환 짝짓기 — lag = B − A 거래일 (음수 = B 가 먼저)")
    print("A        B        창   짝 A미짝 B미짝  중앙  평균  B먼저% | 상승전환: 중앙(n) 먼저/같은날/늦게 | 하락전환: 중앙(n) 먼저/같은날/늦게")
    for a, b in pairs:
        for W in WINDOWS:
            m = match(TR[a], TR[b], W)
            def side(x): return f"{x.median() if len(x) else np.nan:6.1f}({len(x):2d}) {int((x < 0).sum()):2d}/{int((x == 0).sum()):2d}/{int((x > 0).sum()):2d}"
            print(f"{a:8s} {b:8s} {W:3d} {len(m):4d} {len(TR[a]) - len(m):4d} {len(TR[b]) - len(m):5d} {m.lag.median():5.1f} {m.lag.mean():5.1f}"
                  f" {(m.lag < 0).mean() * 100:5.1f}% | {side(m[m.to == True].lag)} | {side(m[m.to == False].lag)}")
    for a, b in [("내방향", "breadth"), ("내방향", "c2")]:
        m = match(TR[a], TR[b], 60)
        print(f"\n[{a} → {b}, 창 60] 짝 목록 (lag + 면 {b} 가 늦음)")
        for _, x in m.sort_values("A").iterrows():
            print(f"  {x.A.date()} → {x.B.date()}  {'상승' if x.to else '하락'}  {int(x.lag):+d}{'  (보류 구간)' if x.A > LEARN_END else ''}")


# ── 3. 정답지 ② 이상적 행동일 ─────────────────────────────────────────────────────
NAME = {"내방향": ("상승", "하락"), "breadth": ("상승", "하락"), "c2": ("보유", "퇴출"), "p7": ("보유", "현금"), "내소음": ("시끄러움", "조용")}


def entry_dev(ser, e, ws, we, pos):
    """기대 상태 e 로 들어간 날과 행동 창 [ws, we] 의 편차 (거래일). 반환 (진입일, 편차) — 진입 없음이면 (None, None)."""
    if ser[ws] == e:
        prev = ser.loc[:ws]; g = (prev != prev.shift()).cumsum(); ent = prev[g == g.iloc[-1]].index[0]
        return ent, min(0, int(pos[ent] - pos[ws]))
    hit = ser.loc[ws:][ser.loc[ws:] == e]
    if hit.empty: return None, None
    ent = hit.index[0]; return ent, (0 if ent <= we else int(pos[ent] - pos[we]))


def answer_key(D, pos, px):
    print(f"\n=== 3. 정답지 ② 이상적 행동일 (목표 ±{KEY_TOL}거래일) — 세 점 모두 통일장 보류 구간 ===")
    print("편차: 음수 = 행동 창보다 일찍(이미 그 상태) · 0 = 창 안 · 양수 = 늦게.  괄호 = 그날 코스피 200 종가")
    for nm, a, b, e in KEY:
        ws = D.index[D.index >= pd.Timestamp(a)][0]; we = D.index[D.index <= pd.Timestamp(b)][-1]
        print(f"\n[{nm}] 창 {ws.date()}~{we.date()}  기대 = {'상승/보유' if e else '하락/퇴출/현금'}  (창 첫날 종가 {px[ws]:.2f})")
        for s in ["내방향", "breadth", "c2", "p7"]:
            ent, dev = entry_dev(D[s], e, ws, we, pos)
            if ent is None: print(f"  {s:8s} {D.index[-1].date()} 까지 기대 상태 진입 없음 (지금 {NAME[s][0] if D[s].iloc[-1] else NAME[s][1]})"); continue
            ok = "✓" if abs(dev) <= KEY_TOL else "✗"
            print(f"  {s:8s} {ent.date()} ({px[ent]:8.2f})  편차 {dev:+4d}  {ok}   창 첫날 대비 {(px[ent] / px[ws] - 1) * 100:+6.1f}%")
        z = D.내소음.loc[:we]; g = (z != z.shift()).cumsum(); since = z[g == g.iloc[-1]].index[0]
        print(f"  내소음   창 동안 {NAME['내소음'][0] if z.iloc[-1] else NAME['내소음'][1]} ({since.date()}~) — 방향 신호 아님")


# ── 4. 통일장 regime × 내 소음 ──────────────────────────────────────────────────────
def noise_cells(D, nxt):
    print(f"\n=== 4. 통일장 regime × 내 소음 — 다음 날 수익률, Rf 0 (학습 Sharpe = ~{LEARN_END.date()}) ===")
    sp = inner_spells(D["칸"]); out = {}
    def row(mask, cell=None):
        p = perf(nxt, mask); pl = perf(nxt, mask & (D.index <= LEARN_END))
        d = dict(비중=mask.mean() * 100, **p, 학습Sharpe=pl["Sharpe"], 보류일수=int((mask & (D.index > LEARN_END)).sum()))
        if cell:
            x = sp[sp.s == cell]; d.update(구간=len(x), 중앙일=x.len.median() if len(x) else np.nan, 최장=x.len.max() if len(x) else np.nan)
        return d
    for g in RG:
        out[f"{g} 전체"] = row(D.regime == g)
        for n in ["조용", "시끄러움"]: out[f"{g}|{n}"] = row(D["칸"] == f"{g}|{n}", f"{g}|{n}")
    out["전체"] = row(D.regime.notna())
    for n in ["조용", "시끄러움"]: out[f"소음 {n}"] = row(D.소음 == n)
    print(pd.DataFrame(out).T.round({"비중": 1, "연수": 2, "연수익": 1, "연σ": 1, "Sharpe": 2, "SE": 2, "학습Sharpe": 2, "중앙일": 0}).to_string())
    print("\n같은 regime 안: Sharpe 차(시끄러움 − 조용) · SE ≈ √(1/연수₁ + 1/연수₂) · 연σ 배율")
    for g in RG:
        a, b = out[f"{g}|시끄러움"], out[f"{g}|조용"]
        print(f"  {g}: {a['Sharpe'] - b['Sharpe']:+.2f} (SE ≈ {np.sqrt(1 / a['연수'] + 1 / b['연수']):.2f})  연σ {b['연σ']:.1f}% → {a['연σ']:.1f}% (×{a['연σ'] / b['연σ']:.1f})")
    print("\n시끄러운 칸: 상위 반등일 5일을 빼면 Sharpe")
    for g in RG:
        m = D["칸"] == f"{g}|시끄러움"; y = nxt[m[m].index].dropna(); y2 = y.drop(y.nlargest(5).index)
        print(f"  {g}|시끄러움: {y.mean() / y.std() * np.sqrt(252):.2f} → {y2.mean() / y2.std() * np.sqrt(252):.2f} (n {len(y)})")
    z = D.내소음; g_ = (z != z.shift()).cumsum(); since = z[g_ == g_.iloc[-1]].index[0]
    h = D.loc[D.index > LEARN_END]
    print(f"\n보류 구간 {len(h)}일 중 시끄러움 {int(h.내소음.sum())}일 · 지금 소음 상태 {'시끄러움' if z.iloc[-1] else '조용'} ({since.date()}~)")


# ── 5. p7 과 내 4국면 ──────────────────────────────────────────────────────────
def p7_cross(D, nxt):
    print("\n=== 5. p7(⑦) 과 내 4국면 ===")
    ct = pd.crosstab(D.내국면, D.p7.map({True: "p7 보유", False: "p7 현금"})).reindex(ORDER)
    print("일수:"); print(ct.to_string())
    print("\np7 현금 날이 내 국면 어디에 있나 (열 %):"); print((ct / ct.sum() * 100).round(1).to_string())
    print("\n내 국면별 p7 비율 (행 %):"); print((ct.div(ct.sum(1), axis=0) * 100).round(1).to_string())
    print("\n내 국면 × p7 다음 날 성과")
    for s in ORDER:
        for v, nm in [(True, "보유"), (False, "현금")]:
            p = perf(nxt, (D.내국면 == s) & (D.p7 == v))
            print(f"  {s:8s} p7 {nm}: 일수 {p['일수']:4d}  연수익 {p['연수익']:6.1f}%  연σ {p['연σ']:5.1f}%  Sharpe {p['Sharpe']:5.2f} (SE≈{p['SE']:.2f})")
    print("\n내 국면 × 통일장 regime (일수):"); print(pd.crosstab(D.내국면, D.regime).reindex(index=ORDER, columns=RG).to_string())


# ── 6. 지속기간 ────────────────────────────────────────────────────────────────
def durations(D, H, start, end):
    print("\n=== 6. 지속기간 (앞·뒤 검열 구간 제외) ===")
    hh = H.loc[start:end]
    def hist_sp(col):
        g = hh[[f"{col}_spell", f"{col}_len", f"{col}_cens", col]].dropna(subset=[f"{col}_spell"]).drop_duplicates(f"{col}_spell")
        g = g[g[f"{col}_cens"] == ""]; return g[f"{col}_len"].astype(int), g[col].astype(str)
    def summ(L): return dict(구간=len(L), 중앙=L.median(), 평균=L.mean(), 최장=L.max(), 이하5일=(L <= 5).mean() * 100, 이하20일=(L <= 20).mean() * 100)
    rows = {f"통일장 {c}": summ(hist_sp(c)[0]) for c in ["regime", "breadth", "c2"]}
    for nm, s in [("내 4국면", D.내국면), ("내 방향", D.내방향), ("내 소음", D.내소음), ("8칸(regime×소음)", D["칸"])]:
        rows[nm] = summ(inner_spells(s).len)
    print(pd.DataFrame(rows).T.round(1).to_string())
    L, lab = hist_sp("regime")
    print("\n통일장 regime 국면별:"); print(pd.DataFrame({"len": L.values, "s": lab.values}).groupby("s").len.agg(["count", "median", "mean", "max"]).round(1).to_string())
    print("\n내 4국면 국면별:"); print(inner_spells(D.내국면).groupby("s").len.agg(["count", "median", "mean", "max"]).reindex(ORDER).round(1).to_string())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--start", default="2016-04-22"); ap.add_argument("--end", default="2026-11-16")
    a = ap.parse_args(); start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    OUT.mkdir(parents=True, exist_ok=True); sys.stdout = Tee(OUT / f"regime_vs_market_layer_{end:%Y%m%d}.txt")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
    old, new, px = load_price(end)
    up, nz, st, nxt = my_regime(px)
    H = pd.read_parquet(H_PATH); H.index = pd.to_datetime(H.index); H = H.sort_index()
    if H.index[-1] < end: sys.exit(f"⚠ regime_history 마지막 {H.index[-1].date()} < 종료일 {end.date()} — 종료일을 당기거나 이력을 갱신할 것")
    H = H.loc[:end]
    check(old, new, px, H, st, nxt, start, end)
    idx = H.loc[start:end].index
    D = pd.DataFrame({"내방향": up.reindex(idx).astype(bool), "내소음": nz.reindex(idx).astype(bool),
                      "breadth": H.breadth.loc[idx].astype(bool), "c2": H.c2.loc[idx].astype(bool), "p7": H.p7.loc[idx].astype(bool),
                      "regime": H.regime.loc[idx].astype(str), "내국면": st.reindex(idx)}, index=idx)
    D["소음"] = D.내소음.map({True: "시끄러움", False: "조용"}); D["칸"] = D.regime + "|" + D.소음
    pos = pd.Series(np.arange(len(idx)), index=idx)
    direction(D, pos); answer_key(D, pos, px); noise_cells(D, nxt); p7_cross(D, nxt); durations(D, H, start, end)
    print(f"\n저장: {OUT / f'regime_vs_market_layer_{end:%Y%m%d}.txt'}")


if __name__ == "__main__":
    main()
