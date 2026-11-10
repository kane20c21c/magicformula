# -*- coding: utf-8 -*-
"""regime_shock.py — 2-b 보강: 쇼크일 기점 회복 속도 비교 (케인 2026-11-05 20:01).

쇼크일  KOSPI200 종가 일간 수익률 ≤ −3% (기본; −2.5%·−4% 민감도). 앞선 쇼크일로부터 20거래일 안이면 같은 사건 → 첫날만 기점
        학습 구간(2026-05-29 까지 기점 + 60거래일 경로가 학습 구간 안)만 집계. 보류 구간 사건은 건수만 표시
그룹    ① 코스피 100 · 코스피 200 · 코스피  ② KRX 100 · KRX 300  ③ 메가갤럭시·유니버스 (시총가중·동일가중)
지표    기점 종가 대비 h일 뒤 수익률(h=1,3,5,10,20,40,60) · 이후 20일 최저(추가 하락) · 쇼크 전날 종가 회복까지 거래일(최대 120)
        코스피 200 과의 차이(같은 사건 짝) — 중앙값과 '더 빨랐던 사건 비율'
사용    python3 regime_shock.py [임계%=-3]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
OUT = HERE / "out" / "regime"; TRAIN_END = pd.Timestamp("2026-05-29")
GROUPS = {"① 코스피 계열": ["코스피 100", "코스피 200", "코스피"], "② KRX 계열": ["KRX 100", "KRX 300"],
          "③ 우리 종목군": ["메가갤럭시 시총가중", "메가갤럭시 동일가중", "유니버스 시총가중", "유니버스 동일가중"],
          "④ 중소형 (코스피·KRX)": ["코스피 200 중소형주", "코스피200 동일가중(ETF 252650)", "코스피 중형주", "코스피 소형주", "KRX 중형 TMI", "KRX 소형 TMI", "KRX 초소형 TMI"],
          "⑤ 코스닥": ["코스닥 150", "코스닥", "코스닥 중형주", "코스닥 소형주"]}   # ④⑤ 케인 20:12 추가
COLS = [c for g in GROUPS.values() for c in g]
H = (1, 3, 5, 10, 20, 40, 60)

def events(lv, th, anchor="first"):
    """쇼크 묶음: 직전 쇼크일과 20거래일 안이면 같은 묶음. anchor=first → 묶음 첫 쇼크일 / bottom → 묶음(마지막 쇼크일+5일까지) 중 KOSPI200 최저 종가일"""
    k = lv["코스피 200"]; r = k.pct_change(); cl = []; last = -999
    for i in np.flatnonzero((r <= th / 100).to_numpy()):
        if i - last > 20: cl.append([i, i])
        else: cl[-1][1] = i
        last = i
    global PRE; kv = k.to_numpy(float)
    out = [a for a, _ in cl] if anchor == "first" else [a + int(np.nanargmin(kv[a:min(b + 6, len(kv))])) for a, b in cl]
    PRE = {o: a - 1 for o, (a, _) in zip(out, cl)}   # 회복 기준 = 묶음 첫 쇼크 전날 종가
    return out

PRE = {}
def measure(lv, ev):
    a = lv[COLS].to_numpy(float); rows = []
    for i in ev:
        for j, c in enumerate(COLS):
            base, pre = a[i, j], a[PRE[i], j]; d = dict(date=lv.index[i], 지수=c, 쇼크일=(a[i, j] / a[i - 1, j] - 1) * 100, 쇼크전대비=(base / pre - 1) * 100)
            for h in H: d[f"+{h}일"] = (a[i + h, j] / base - 1) * 100 if i + h < len(a) else np.nan
            d["이후20일 최저"] = (np.nanmin(a[i + 1:i + 21, j]) / base - 1) * 100
            k = np.flatnonzero(a[i + 1:i + 121, j] >= pre); d["전날종가 회복일"] = int(k[0] + 1) if k.size else np.nan
            rows.append(d)
    return pd.DataFrame(rows)

def main(th=-3.0, anchor="first"):
    lv = pd.read_parquet(OUT / "layer_levels.parquet"); pd.set_option("display.width", 260)
    print(f"\n######## 기점 = {'묶음 첫 쇼크일' if anchor == 'first' else '묶음 안 KOSPI200 최저 종가일'} ########")
    ev_all = events(lv, th, anchor); idx = lv.index
    ev = [i for i in ev_all if i + 60 < len(idx) and idx[i + 60] <= TRAIN_END]
    hold = [idx[i].date() for i in ev_all if idx[i] > TRAIN_END - pd.Timedelta(days=90) and i not in ev]
    for t2 in (-2.5, -4.0): print(f"(민감도) 임계 {t2}% 사건 수: {len(events(lv, t2))}")
    ev_all = events(lv, th, anchor)   # PRE 를 이 임계·기점으로 다시 맞춤
    print(f"임계 {th}% · 학습 사건 {len(ev)}개 · 보류/경로 미완 사건(제외) {hold}")
    M = measure(lv, ev); M.to_csv(OUT / f"shock_events_{anchor}_{abs(th)}.csv", index=False)
    print("\n사건:", ", ".join(f"{idx[i].date()}({M[(M.date == idx[i]) & (M.지수 == '코스피 200')]['쇼크일'].iloc[0]:.1f}%)" for i in ev))
    met = [f"+{h}일" for h in H] + ["이후20일 최저", "전날종가 회복일"]
    M["전날종가 회복일"] = M["전날종가 회복일"].fillna(121)   # 120일 안 미회복 = 121 로 두고 중앙값 (NaN 을 빼면 회복한 사건만 남아 왜곡)
    med = M.groupby("지수")[met].median().reindex(COLS)
    rec = M.groupby("지수")["전날종가 회복일"].apply(lambda x: pd.Series({"20일내 회복%": (x <= 20).mean() * 100, "60일내 회복%": (x <= 60).mean() * 100})).unstack().reindex(COLS)
    print("\n== 사건 중앙값 (기점 = 쇼크일 종가, %) ==\n", pd.concat([med, rec], axis=1).round(1).to_string())
    k = M[M.지수 == "코스피 200"].set_index("date")
    rows = {}
    for c in COLS:
        x = M[M.지수 == c].set_index("date"); d = {}
        for h in ("+5일", "+10일", "+20일", "+40일", "+60일"):
            df = x[h] - k[h]; d[f"{h} 차이 중앙"] = df.median(); d[f"{h} 앞선 비율%"] = (df > 0).mean() * 100
        df = x["전날종가 회복일"] - k["전날종가 회복일"]
        d["회복일 차이 중앙(음수=빠름)"] = df.median(); d["회복 더 빠른%"] = (df < 0).mean() * 100; d["회복 더 늦은%"] = (df > 0).mean() * 100
        # 평소 대비: 학습 구간 아무 날에서 시작한 같은 h 의 (지수 − 코스피200) 중앙값 — 꾸준한 초과수익(드리프트)과 쇼크 뒤 회복을 구분
        lt = lv.loc[:TRAIN_END]
        for h in (20, 60):
            base = ((lt[c].shift(-h) / lt[c]) - (lt["코스피 200"].shift(-h) / lt["코스피 200"])).median() * 100
            d[f"평소 +{h}일 차이 중앙"] = base; d[f"+{h}일 쇼크후−평소"] = d[f"+{h}일 차이 중앙"] - base
        rows[c] = d
    D = pd.DataFrame(rows).T
    print("\n== 코스피 200 대비 (같은 사건 짝) ==\n", D.round(1).to_string())
    med.join(rec).to_csv(OUT / f"shock_median_{anchor}_{abs(th)}.csv"); D.to_csv(OUT / f"shock_vs_k200_{anchor}_{abs(th)}.csv")
    plot(lv, ev, th, anchor)

def plot(lv, ev, th, anchor):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "NanumSquare"; plt.rcParams["axes.unicode_minus"] = False
    a = lv[COLS]; win = range(-5, 61)
    paths = {c: np.nanmedian(np.array([[a[c].iloc[i + w] / a[c].iloc[i] * 100 for w in win] for i in ev]), axis=0) for c in COLS}
    fig, axs = plt.subplots(1, len(GROUPS), figsize=(5.4 * len(GROUPS), 5), sharey=True)
    for ax, (g, cs) in zip(axs, GROUPS.items()):
        for c in (cs if "코스피 200" in cs else ["코스피 200"] + cs):
            ax.plot(list(win), paths[c], lw=2 if c != "코스피 200" else 1.2, ls="-" if c != "코스피 200" else "--", label=c)
        ax.axhline(100, color="grey", lw=0.6); ax.axvline(0, color="grey", lw=0.6); ax.set_title(g); ax.legend(fontsize=8); ax.set_xlabel("쇼크일 기점 거래일")
    axs[0].set_ylabel("쇼크일 종가 = 100 (사건 중앙값)")
    fig.suptitle(f"KOSPI200 일간 ≤ {th}% 쇼크일 기점 회복 경로 — 학습 구간 {len(ev)}개 사건의 중앙값"); fig.tight_layout()
    fig.savefig(OUT / f"shock_recovery_{anchor}.png", dpi=130); plt.close(fig)

if __name__ == "__main__":
    th = float(sys.argv[1]) if len(sys.argv) > 1 else -3.0
    for an in (sys.argv[2:] or ["first", "bottom"]): main(th, an)
