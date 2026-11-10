# -*- coding: utf-8 -*-
"""regime_detect.py — 작업계획 2-c: 실시간 국면 판정기 비교 (케인 2026-11-05 20:25).

판정기  매일 종가까지의 정보로 '상승(True)/하락(False)' 상태를 낸다 (미래 참조 없음, 체결은 다음 날)
정답    KOSPI200 zigzag 20% (사후) 의 국면과 전환점 — 학습 구간만 (평가 창 2016-07-01 ~ 2026-05-29, 브레드스 자료가 2016-01~)
지표    ① 일치율: 전체 / 정답 상승일 / 정답 하락일
        ② 연간 상태 전환 횟수 (헛신호의 대리)
        ③ 인지 지연: 정답 전환일 T 기준 [T−40, 다음 전환) 안에서 새 방향으로 바뀐 뒤 10거래일 이상 유지된 첫 전환일 − T (음수 = 먼저). 없으면 놓침
사용    python3 regime_detect.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from stop_regime import zigzag

OUT = HERE / "out" / "regime"; TRAIN_END = pd.Timestamp("2026-05-29"); EVAL0 = pd.Timestamp("2016-07-01")

def ma_line(p, n, kind="SMA"):
    if kind == "EMA": return p.ewm(span=n, adjust=False, min_periods=n).mean()
    if kind == "WMA":
        w = np.arange(1, n + 1, dtype=float); return p.rolling(n, min_periods=n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)
    return p.rolling(n, min_periods=n).mean()

def ma_state(p, n, band=0.0, kind="SMA"):
    m = ma_line(p, n, kind); s = pd.Series(np.nan, index=p.index); cur = True
    for i, (x, mm) in enumerate(zip(p.to_numpy(), m.to_numpy())):
        if np.isfinite(mm):
            if x > mm * (1 + band): cur = True
            elif x < mm * (1 - band): cur = False
        s.iloc[i] = cur
    return s.astype(bool)

def zz_state(p, x):
    s = []; cur = True; ext = p.iloc[0]
    for v in p.to_numpy():
        if cur:
            ext = max(ext, v)
            if v <= ext * (1 - x): cur = False; ext = v
        else:
            ext = min(ext, v)
            if v >= ext * (1 + x): cur = True; ext = v
        s.append(cur)
    return pd.Series(s, index=p.index)

def hyst(x, hi, lo):
    s = []; cur = True
    for v in x.to_numpy():
        if np.isfinite(v):
            if v >= hi: cur = True
            elif v <= lo: cur = False
        s.append(cur)
    return pd.Series(s, index=x.index)

def truth(k):
    pts = zigzag(k.loc["2014-06-01":], 0.20); st = pd.Series(np.nan, index=k.index)
    for (d, t) in pts: st.loc[d:] = (t == "L")   # 저점 이후 = 상승 국면
    turns = [(d, "고점" if t == "H" else "저점") for d, t in pts[1:-1] if pd.Timestamp("2015-01-02") <= d <= TRAIN_END]
    return st.astype(bool), turns

def detectors(lv, br):
    k = lv["코스피 200"]; D = {}
    for n in (200, 120, 60): D[f"K200 {n}일선"] = ma_state(k, n)
    D["K200 200일선 ±2% 밴드"] = ma_state(k, 200, 0.02)
    # 11-06 11:22 케인: SMA 대신 EMA·WMA 로 속도 올리기
    for kind in ("EMA", "WMA"):
        for n in (200, 120, 60): D[f"K200 {kind}{n}"] = ma_state(k, n, kind=kind)
        D[f"K200 {kind}120 ±2% 밴드"] = ma_state(k, 120, 0.02, kind=kind)
    D["K200 SMA120 ±2% 밴드"] = ma_state(k, 120, 0.02)
    D["K200 20일선>60일선"] = (k.rolling(20).mean() > k.rolling(60).mean())
    for x in (0.10, 0.15, 0.20): D[f"K200 실시간 zigzag {int(x*100)}%"] = zz_state(k, x)
    for c in ("메가갤럭시 동일가중", "유니버스 동일가중", "코스피200 동일가중(ETF 252650)"): D[f"{c.split('(')[0].strip()} 200일선"] = ma_state(lv[c].dropna(), 200).reindex(lv.index).ffill().fillna(True).astype(bool)
    e10 = br["ema10"].reindex(lv.index); e20 = br["ratio"].ewm(span=20, adjust=False).mean().reindex(lv.index)
    D["브레드스 EMA10 0.55/0.45"] = hyst(e10, 0.55, 0.45); D["브레드스 EMA20 0.52/0.48"] = hyst(e20, 0.52, 0.48)
    # 한국 시장은 하락 종목이 늘 많아 절대 임계가 한쪽으로 쏠림 → 직전 250일 중앙값 기준 상대 임계
    med = br["ratio"].rolling(250, min_periods=120).median().reindex(lv.index)
    for d in (0.03, 0.05): D[f"브레드스 EMA20 − 250일 중앙 ±{d}"] = hyst(e20 - med, d, -d)
    thr = br["thrust"].astype(bool).reindex(lv.index).fillna(False); ma = ma_state(k, 200); s = []; cur = True
    for t, m_ in zip(thr.to_numpy(), ma.to_numpy()):
        if t: cur = True
        elif not m_: cur = False
        s.append(cur)
    D["츠바이크 진입 + K200 200일선 이탈 퇴출"] = pd.Series(s, index=lv.index)
    for c in ("KRX 중형 TMI", "코스피 200 중소형주", "코스닥"): D[f"{c} 200일선"] = ma_state(lv[c], 200)
    D["괴리형: K200·중형TMI 둘 다 200일선 위"] = ma & ma_state(lv["KRX 중형 TMI"], 200)
    D["괴리형: K200·코스피200중소형 둘 다 200일선 위"] = ma & ma_state(lv["코스피 200 중소형주"], 200)
    rs = lv["KRX 중형 TMI"] / k; D["괴리형: K200 200일선 위 & 중형/K200 상대강도 60일선 위"] = ma & (rs > rs.rolling(60).mean())
    # ── 11-06 케인 요청 ① 브레드스 수정안 ──
    # 진단: 2020-06~2021-06 상승장에서 상승 종목 비율 EMA20 이 250일 중앙값 ±0.03 안에서만 움직여 +0.05 를 못 넘음
    #       → 한 번 −0.05 로 내려간 뒤 '하락'에 1년 갇힘 (개수 기반 브레드스는 크기를 못 봄 + 끈끈한 밴드)
    d20 = e20 - med
    D["[브레드스 수정 F1] EMA20 − 250일 중앙 0 교차"] = hyst(d20, 0.0, -1e-9)
    e40 = br["ratio"].ewm(span=40, adjust=False).mean().reindex(lv.index)
    D["[브레드스 수정 F1b] EMA40 − 250일 중앙 0 교차"] = hyst(e40 - med, 0.0, -1e-9)
    ewi = (1 + br["ew_ret"].fillna(0)).cumprod().reindex(lv.index).ffill()      # 시장 전체 동일가중 지수 (크기를 보는 브레드스)
    for n in (60, 120): D[f"[브레드스 수정 F2] 시장 동일가중 지수 {n}일선"] = ma_state(ewi.dropna(), n).reindex(lv.index).ffill().fillna(True).astype(bool)
    base = ma_state(k, 120); s = []; last_sig = -999; bst = True
    for i, v in enumerate(d20.to_numpy()):
        if np.isfinite(v) and v >= 0.05: bst, last_sig = True, i
        elif np.isfinite(v) and v <= -0.05: bst, last_sig = False, i
        s.append(bst if i - last_sig <= 20 else base.iloc[i])
    D["[브레드스 수정 F3] ±0.05 신호 뒤 20일만 브레드스, 그 밖엔 K200 120일선"] = pd.Series(s, index=lv.index)
    # ── 11-06 08:47 케인 요청: 브레드스를 유니버스(·메가갤럭시)로 ──
    for tag, f in (("유니버스", "breadth_universe_2015.csv"), ("메가갤럭시", "breadth_mega_2015.csv")):
        b = pd.read_csv(HERE / "out" / "breadth" / f, parse_dates=["date"]).set_index("date")
        be20 = b["ratio"].ewm(span=20, adjust=False).mean(); bmed = b["ratio"].rolling(250, min_periods=120).median()
        bd = (be20 - bmed).reindex(lv.index)
        for dd in (0.05, 0.03): D[f"[{tag} 브레드스] EMA20 − 250일 중앙 ±{dd}"] = hyst(bd, dd, -dd)
        be40 = b["ratio"].ewm(span=40, adjust=False).mean()
        D[f"[{tag} 브레드스] EMA40 − 250일 중앙 0 교차"] = hyst((be40 - bmed).reindex(lv.index), 0.0, -1e-9)
        s = []; last_sig = -999; bst = True
        for i, v in enumerate(bd.to_numpy()):
            if np.isfinite(v) and v >= 0.05: bst, last_sig = True, i
            elif np.isfinite(v) and v <= -0.05: bst, last_sig = False, i
            s.append(bst if i - last_sig <= 20 else base.iloc[i])
        D[f"[{tag} 브레드스] ±0.05 신호 뒤 20일만, 그 밖엔 K200 120일선"] = pd.Series(s, index=lv.index)
    # ── 11-06 케인 요청 ② 괴리형 + 실시간 zigzag 10% ──
    dv = ma & ma_state(lv["KRX 중형 TMI"], 200); z10 = zz_state(k, 0.10)
    D["[조합 C1] 괴리형 OR zigzag10% (둘 다 하락일 때만 하락)"] = dv | z10
    kv = k.to_numpy(float); dvv = dv.to_numpy()
    for rb in (0.07, 0.10, 0.15):
        s = []; cur = True; low = np.inf
        for i in range(len(kv)):
            if cur:
                if i > 0 and dvv[i - 1] and not dvv[i]: cur = False; low = kv[i]      # 괴리형이 '하락'으로 바뀌는 날 퇴출
            else:
                low = min(low, kv[i])
                if kv[i] >= low * (1 + rb) or (dvv[i] and not dvv[i - 1]): cur = True  # 퇴출 뒤 최저가에서 rb 반등 or 괴리형 복귀
            s.append(cur)
        D[f"[조합 C2] 괴리형 퇴출 → 퇴출 뒤 저점 +{int(rb*100)}% 반등 복귀"] = pd.Series(s, index=lv.index)
    print("츠바이크 신호 횟수(2016~):", int(br["thrust"].astype(str).str.lower().eq("true").sum()))
    return D

def lag(st, T, D_new, nxt, prv=None):
    # 창 시작 = max(T−40, 직전 정답 전환일) — 직전 국면이 시작되기도 전의 상태를 '먼저 알아챔'으로 세지 않기 위해 (11-06 수정)
    idx = st.index; i0 = idx.get_loc(T); lo = max(0, i0 - 40, idx.get_loc(prv) if prv is not None else 0); hi = idx.get_loc(nxt) if nxt is not None else min(len(idx) - 1, i0 + 250)
    v = st.to_numpy()
    for i in range(max(lo, 1), hi):
        if v[i] == D_new and v[i - 1] != D_new and (v[i:i + 10] == D_new).all(): return i - i0
    if v[lo] == D_new and (v[lo:i0 + 1] == D_new).all(): return np.inf   # 창 전부터 이미 새 방향 = 직전 국면을 아예 못 잡고 유지
    return np.nan

def evaluate(D, tr, turns):
    w = (tr.index >= EVAL0) & (tr.index <= TRAIN_END); t = tr[w]; yrs = w.sum() / 252; rows = []
    for nm, st in D.items():
        s = st.reindex(tr.index).ffill().astype(bool); sw = s[w]
        r = {"판정기": nm, "일치율%": (sw == t).mean() * 100, "상승일 일치%": (sw[t] == True).mean() * 100, "하락일 일치%": (sw[~t] == False).mean() * 100,
             "연간 전환": (sw != sw.shift()).iloc[1:].sum() / yrs, "상승 비중%": sw.mean() * 100}
        L = {"고점": [], "저점": []}
        for j, (T, kind) in enumerate(turns):
            if T < EVAL0: continue
            nxt = turns[j + 1][0] if j + 1 < len(turns) else None; prv = turns[j - 1][0] if j > 0 else None
            L[kind].append(lag(s, T, kind == "저점", nxt, prv))
        for kind in ("고점", "저점"):
            a = np.array(L[kind], float); f = a[np.isfinite(a)]
            r[f"{kind} 지연 중앙"] = np.median(f) if f.size else np.nan
            r[f"{kind} 놓침·유지"] = int((~np.isfinite(a)).sum())
            r[f"{kind} 지연 목록"] = " ".join("—" if np.isnan(x) else ("유지" if np.isinf(x) else f"{int(x):+d}") for x in a)
        rows.append(r)
    return pd.DataFrame(rows).set_index("판정기")

def main():
    lv = pd.read_parquet(OUT / "layer_levels.parquet"); br = pd.read_csv(HERE / "out" / "breadth" / "breadth_market.csv", parse_dates=["date"]).set_index("date")
    tr, turns = truth(lv["코스피 200"]); D = detectors(lv, br)
    E = evaluate(D, tr, turns); pd.set_option("display.width", 320); pd.set_option("display.max_colwidth", 40)
    print("평가 전환점:", ", ".join(f"{d.date()} {k}" for d, k in turns if d >= EVAL0))
    print(E.round(1).sort_values("일치율%", ascending=False).to_string()); E.to_csv(OUT / "detect_compare.csv")
    pd.DataFrame(D).to_parquet(OUT / "detect_states.parquet")

if __name__ == "__main__":
    main()
