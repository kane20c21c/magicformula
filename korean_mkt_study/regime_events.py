# -*- coding: utf-8 -*-
"""regime_events.py — 2-c 보완 5: 조정 사건 기준 판정기 평가 (케인 2026-11-06 18:07 "해줘").

조정 사건  코스피 200 이 ⓐ 직전 20거래일 최고가 대비 −8% 이하 OR ⓑ 하루 −3% 이하 — 10거래일 안에 다시 켜지면 같은 사건
           보조 정의: 직전 10거래일 최고가 대비 −8% (급락만)
           고점 P = 첫 발동일 앞 20거래일 안 최고 종가 · 저점 L = 첫 발동일부터 60거래일 안 최저 종가
판정기     매일 종가로 상승/하락. 사건마다 — 퇴출: P 이후 처음 '하락'이 된 날 (P 에 이미 하락이면 P 에서 퇴출로 봄)
           복귀: 퇴출 뒤 처음 '상승'이 된 날. 왕복 이득 = 퇴출가 ÷ 복귀가 − 1 (+ 면 그냥 들고 있던 것보다 이득)
헛 퇴출    어느 사건 창(P ~ 저점 뒤 회복)에도 걸리지 않은 '하락' 구간 — 비용 = 그 구간 동안 코스피 200 상승률
타이밍     '상승' 판정 다음 날부터 코스피 200 보유, '하락'이면 현금 — 단순 보유 대비 CAGR·MDD (사건 안 / 밖 기여 분리)
기간       2016-07-01 ~ 2026-05-29 (학습). 보류 구간 미사용
사용       python3 regime_events.py [20|10]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
OUT = HERE / "out" / "regime"; A, B = pd.Timestamp("2016-07-01"), pd.Timestamp("2026-05-29")
DET = {"시장 브레드스 ±0.05": "브레드스 EMA20 − 250일 중앙 ±0.05", "유니버스 브레드스 ±0.05": "[유니버스 브레드스] EMA20 − 250일 중앙 ±0.05",
       "유니버스 브레드스 ±0.03": "[유니버스 브레드스] EMA20 − 250일 중앙 ±0.03", "메가 브레드스 ±0.05": "[메가갤럭시 브레드스] EMA20 − 250일 중앙 ±0.05",
       "유니버스 브레드스 혼합": "[유니버스 브레드스] ±0.05 신호 뒤 20일만, 그 밖엔 K200 120일선", "메가 브레드스 혼합": "[메가갤럭시 브레드스] ±0.05 신호 뒤 20일만, 그 밖엔 K200 120일선",
       "SMA120": "K200 120일선", "SMA200 ±2%": "K200 200일선 ±2% 밴드", "WMA60": "K200 WMA60", "실시간 zigzag 10%": "K200 실시간 zigzag 10%",
       "C2 괴리형+10%복귀": "[조합 C2] 괴리형 퇴출 → 퇴출 뒤 저점 +10% 반등 복귀"}

def events(k, win, use_shock=True):
    r = k.pct_change(); c = (k <= k.rolling(win, min_periods=1).max() * 0.92)
    if use_shock: c = c | (r <= -0.03)
    ev = []; last = -999
    for i in np.flatnonzero(c.to_numpy()):
        if i - last > 10: ev.append(i)
        last = i
    out = []
    for i in ev:
        a = max(0, i - 20); p = a + int(np.argmax(k.to_numpy()[a:i + 1])); l = i + int(np.argmin(k.to_numpy()[i:i + 61]))
        rec = next((j for j in range(l, min(len(k), l + 250)) if k.iloc[j] >= k.iloc[p]), min(len(k) - 1, l + 250))
        out.append(dict(trig=i, peak=p, trough=l, rec=rec))
    return out

def per_event(k, s, e):
    kv, sv = k.to_numpy(float), s.to_numpy(bool); p, l, rec = e["peak"], e["trough"], e["rec"]
    if not sv[p]: x = p                                           # 고점에 이미 하락 판정 → 고점에서 현금으로 봄
    else:
        x = next((j for j in range(p + 1, l + 1) if not sv[j]), None)
        if x is None:                                             # 저점까지 퇴출 못 함
            x2 = next((j for j in range(l + 1, rec + 1) if not sv[j]), None)
            if x2 is None: return dict(대응="미대응", 퇴출낙폭=np.nan, 퇴출후추가하락=np.nan, 복귀_저점대비=np.nan, 왕복이득=0.0, x=None, y=None)
            x = x2
    y = next((j for j in range(x + 1, len(kv)) if sv[j]), len(kv) - 1)
    return dict(대응="퇴출" if x <= l else "저점 뒤 퇴출", 퇴출낙폭=(kv[x] / kv[p] - 1) * 100, 퇴출후추가하락=(kv[l] / kv[x] - 1) * 100 if x <= l else 0.0,
                복귀_저점대비=(kv[y] / kv[l] - 1) * 100, 왕복이득=(kv[x] / kv[y] - 1) * 100, x=x, y=y)

def timing(k, s, inside):
    r = k.pct_change().fillna(0).to_numpy(); pos = np.r_[True, s.to_numpy(bool)[:-1]]      # 판정 다음 날부터 보유
    w = (k.index >= A) & (k.index <= B); rr = np.where(pos, r, 0.0)
    eq = np.cumprod(1 + rr[w]); bh = np.cumprod(1 + r[w]); yrs = w.sum() / 252
    mdd = lambda e: (e / np.maximum.accumulate(e) - 1).min() * 100
    # 현금 기간에 피한(+)·놓친(−) 코스피 200 수익 — 사건 창 안 / 밖
    cash = (~pos) & w; av_in = -np.log1p(r[cash & inside]).sum() * 100; av_out = -np.log1p(r[cash & ~inside]).sum() * 100
    return {"CAGR%": (eq[-1] ** (1 / yrs) - 1) * 100, "단순보유 CAGR%": (bh[-1] ** (1 / yrs) - 1) * 100, "MDD%": mdd(eq), "단순보유 MDD%": mdd(bh),
            "보유 비중%": pos[w].mean() * 100, "현금기간 사건 안(로그%p)": av_in, "현금기간 사건 밖(로그%p)": av_out}

def main(win=20):
    lv = pd.read_parquet(OUT / "layer_levels.parquet"); S = pd.read_parquet(OUT / "detect_states.parquet")
    k = lv["코스피 200"]; pd.set_option("display.width", 320); pd.set_option("display.max_colwidth", 30)
    ib = int(np.searchsorted(k.index, B, side="right")) - 1
    EV = [e for e in events(k, win, use_shock=(win == 20)) if A <= k.index[e["trig"]] and e["trig"] + 60 <= ib]   # 저점 찾는 60일 창이 보류 구간에 닿는 사건 제외
    for e in EV: e["rec"] = min(e["rec"], ib)
    tag = f"{win}일 최고가 −8%" + (" 또는 하루 −3%" if win == 20 else "")
    print(f"조정 사건 [{tag}] {len(EV)}개 (2016-07 ~ 2026-05 발동)")
    print("  " + ", ".join(f"{k.index[e['trig']].date()}(고점대비 저점 {(k.iloc[e['trough']]/k.iloc[e['peak']]-1)*100:.0f}%)" for e in EV))
    inside = np.zeros(len(k), bool)
    for e in EV: inside[e["peak"]:e["rec"] + 1] = True
    rows = []; detail = {}
    for nm, c in DET.items():
        s = S[c].reindex(k.index).ffill().fillna(True).astype(bool)
        P = pd.DataFrame([per_event(k, s, e) for e in EV]); detail[nm] = P
        resp = P[P.대응 != "미대응"]
        r = {"판정기": nm, "대응 사건": f"{len(resp)}/{len(P)}", "퇴출 시 고점대비 중앙%": resp.퇴출낙폭.median(), "퇴출 뒤 추가하락 중앙%": resp.퇴출후추가하락.median(),
             "복귀 시 저점대비 중앙%": resp.복귀_저점대비.median(), "왕복이득 중앙%": P.왕복이득.median(), "왕복이득 +인 사건%": (P.왕복이득 > 0).mean() * 100,
             "왕복이득 합(%p)": P.왕복이득.sum()}
        r.update(timing(k, s, inside)); rows.append(r)
    T = pd.DataFrame(rows).set_index("판정기")
    print("\n== 사건 기준 ==\n", T.iloc[:, :7].round(1).to_string())
    print("\n== 타이밍 (판정 다음 날부터 코스피 200 보유/현금) ==\n", T.iloc[:, 7:].round(1).to_string())
    T.to_csv(OUT / f"events_eval_{win}.csv")
    # 케인 표시 6곳
    marks = ["2020-01-28", "2022-01-25", "2022-06-13", "2022-09-23", "2024-08-02", "2025-08-01", "2025-11-14"]
    tri = [k.index[e["trig"]] for e in EV]; sel = [min(range(len(EV)), key=lambda j: abs((tri[j] - pd.Timestamp(m)).days)) for m in marks]
    sel = list(dict.fromkeys(sel)); rows = []
    for j in sel:
        e = EV[j]; r = {"사건(발동일)": tri[j].date(), "고점→저점": f"{k.iloc[e['peak']]:.1f}→{k.iloc[e['trough']]:.1f} ({(k.iloc[e['trough']]/k.iloc[e['peak']]-1)*100:+.0f}%)"}
        for nm in ("시장 브레드스 ±0.05", "메가 브레드스 ±0.05", "유니버스 브레드스 ±0.05", "SMA120", "실시간 zigzag 10%"):
            d = detail[nm].iloc[j]
            r[nm] = "미대응" if pd.isna(d.x) else f"{k.index[int(d.x)].strftime('%m-%d')} {k.iloc[int(d.x)]:.0f} → {k.index[int(d.y)].strftime('%y-%m-%d')} {k.iloc[int(d.y)]:.0f} ({d.왕복이득:+.1f}%)"
        rows.append(r)
    print("\n== 케인 표시 구간 (퇴출일 가격 → 복귀일 가격, 왕복이득) ==\n", pd.DataFrame(rows).to_string(index=False))
    pd.DataFrame(rows).to_csv(OUT / f"events_marks_{win}.csv", index=False)

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
