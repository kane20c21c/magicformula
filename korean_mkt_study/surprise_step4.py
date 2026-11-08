# -*- coding: utf-8 -*-
"""surprise_step4.py — 급등 연구 4단계: 경로·보유 관점 (학습 구간만, 주 정의). 케인 2026-11-08 14:24 승인. 결과 보기 전 고정한 정의:
① 갈림  20거래일 안 첫 사건: B = 종가 > 급등일 고가 H · G = 종가 ≤ 급등 전날 종가 pc · Lx = 종가 < 급등일 저가 L
② 2차 상승 진입  10거래일 안 첫 B 날 다음 날 시가 진입 → +5/+10/+20일 (왕복 0.23%)
   저가 이탈 진입   10거래일 안 첫 Lx 날 다음 날 시가 진입 → 같은 측정 (급락 D2 의 거울상: 이탈 뒤에도 빠지나)
③ 보유 규칙 (급등일 종가 C 에 이미 보유, h = 10·20거래일, 모두 h 안에 현금화, 매도 비용 0.2%)
   (다) 보유      : C_{t+h}/C − 1 − 0.2%
   (가) 즉시 실현 : −0.2% (급등일 종가 매도 — 절반 실현 = (가)와 (다)의 평균)
   (나) 저가 이탈 정리 : h 안 첫 Lx 다음 날 시가 매도, 없으면 (다)
   (라) 본전 이탈 정리 : h 안 첫 G(종가 ≤ pc) 다음 날 시가 매도, 없으면 (다)
   후회율 = 정리한 뒤 h 안에 종가가 H 를 넘은 비율 (팔고 나서 다시 오름)
   견고성: 규칙 − 보유 (h=20) 를 칸마다 부트스트랩 95% + 시기 3개 + 몰림날 제외 + 2026.1~4 제외."""
import numpy as np, pandas as pd
from stock_gapshock import D, OUT
from stock_surprise import HOLD, COST
from surprise_step3 import prep, SUB, top10
rng = np.random.default_rng(20261108); NB = 600; SELL = 0.002; H_ = 20
ev = prep(False)
p = pd.read_parquet(D / "prices.parquet").reset_index().sort_values(["ticker", "date"]).reset_index(drop=True)
key = pd.Series(np.arange(len(p)), index=pd.MultiIndex.from_frame(p[["ticker", "date"]]))
pos = key.reindex(pd.MultiIndex.from_frame(ev[["ticker", "date"]])).to_numpy().astype(int)
TK = p.ticker.to_numpy(); Cc = p.close.to_numpy(float); Oo = p.open.to_numpy(float)
n = len(p); J = np.arange(1, 32)   # 열 0 = t+1 … 열 30 = t+31 (② 진입 뒤 20일까지)
idx = np.minimum(pos[:, None] + J[None, :], n - 1); same = TK[idx] == TK[pos][:, None]
CL = np.where(same, Cc[idx], np.nan); OP = np.where(same, Oo[idx], np.nan)   # 열 0 = t+1 … 열 20 = t+21
OP[:, :] = np.where(np.isfinite(OP) & (OP > 0), OP, CL)                       # 시가 결손 → 종가 대체
c0 = ev.close.to_numpy(); pc = ev.pc.to_numpy(); hi = ev.high.to_numpy(); lo = ev.low.to_numpy()
def first(cond, k):  # 열 0..k-1 에서 첫 True 의 일수(1..k), 없으면 0
    c = cond[:, :k]; f = c.argmax(1) + 1; return np.where(c.any(1), f, 0)
dB = first(CL > hi[:, None], 20); dG = first(CL <= pc[:, None], 20); dL = first(CL < lo[:, None], 20)
ev["dB"], ev["dG"], ev["dL"] = dB, dG, dL
def firstof(a, b):
    return np.select([(a > 0) & ((b == 0) | (a < b)), (b > 0) & ((a == 0) | (b < a)), (a > 0) & (a == b)], ["돌파 먼저", "반납 먼저", "같은 날"], "둘 다 없음")
ev["갈림"] = firstof(dB, dG)
# ② 진입
def entry(dcol, within=10):
    d = ev[dcol].to_numpy(); ok = (d > 0) & (d <= within)
    out = {}
    for k in (5, 10, 20):
        ent = np.where(ok, OP[np.arange(len(ev)), np.clip(d, 1, 30)], np.nan)          # 사건 다음 날 시가 = 열 d
        ex = np.where(ok, CL[np.arange(len(ev)), np.clip(d + k - 1, 0, 30)], np.nan)
        out[f"e{k}"] = ex / ent - 1 - COST
    return ok, out
okB, eB = entry("dB"); okL, eL = entry("dL")
for k, v in eB.items(): ev["B_" + k] = v
for k, v in eL.items(): ev["L_" + k] = v
ev["has_B10"], ev["has_L10"] = okB, okL
# ③ 보유 규칙
R = np.arange(len(ev))
for h in (10, 20):
    hold = CL[:, h - 1] / c0 - 1 - SELL
    ev[f"r_hold{h}"] = hold; ev[f"r_now{h}"] = -SELL
    for nm, d in (("low", dL), ("pc", dG)):
        trig = (d > 0) & (d <= h - 1) | (d == h)   # h 일째 이탈이면 h+1 시가 매도 (열 h)
        trig = (d > 0) & (d <= h)
        sell = OP[R, np.clip(d, 1, 20)] / c0 - 1 - SELL
        ev[f"r_{nm}{h}"] = np.where(trig, sell, hold); ev[f"sold_{nm}{h}"] = trig
        # 후회: 정리 뒤 h 안에 종가 > H
        later = np.array([np.nanmax(CL[i, d[i]:h]) if trig[i] and d[i] < h else -np.inf for i in R])
        ev[f"regret_{nm}{h}"] = np.where(trig, later > hi, np.nan)
ev["cell"] = ev.ctx2
def cells(e):
    c = {"전체": np.ones(len(e), bool)}
    for x in ["단독", "섹터 동반", "시장·반등형", "시장·순수"]: c[x] = (e.ctx2 == x).to_numpy()
    s = (e.ctx2 == "단독").to_numpy()
    for k in ["작은", "중간", "큰"]: c[f"단독·{k}"] = s & (e["size"] == k).to_numpy()
    for k in ["저", "중", "고"]: c[f"단독·거래량 {k}"] = s & (e.vb == k).to_numpy()
    c["단독·⑤"] = s & (e["shape"] == "⑤").to_numpy(); c["단독·①"] = s & (e["shape"] == "①").to_numpy()
    return c
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
C_ = cells(ev)
# ── ① 갈림
t1 = pd.DataFrame({k: ev.loc[m, "갈림"].value_counts(normalize=True) * 100 for k, m in C_.items()}).T.fillna(0)
t1 = t1.reindex(columns=["돌파 먼저", "반납 먼저", "둘 다 없음"], fill_value=0); t1.insert(0, "n", [int(m.sum()) for m in C_.values()])
for k, m in C_.items():
    e = ev[m]; t1.loc[k, "돌파일 중앙"] = e.dB[e.dB > 0].median(); t1.loc[k, "반납일 중앙"] = e.dG[e.dG > 0].median()
    t1.loc[k, "저가이탈 20일 %"] = (e.dL > 0).mean() * 100; t1.loc[k, "저가이탈일 중앙"] = e.dL[e.dL > 0].median()
    t1.loc[k, "1일 안 돌파 %"] = (e.dB == 1).mean() * 100; t1.loc[k, "3일 안 돌파 %"] = ((e.dB > 0) & (e.dB <= 3)).mean() * 100
print("[①] 20일 안 첫 사건 (돌파 = 종가 > 급등일 고가, 반납 = 종가 ≤ 급등 전날 종가)"); print(t1.round(1).to_string())
# ② 진입
base = 0.44
def etab(pref, okcol):
    rows = {}
    for k, m in C_.items():
        e = ev[m & ev[okcol].to_numpy()]
        rows[k] = {"n": len(e), "발생 %": m.sum() and len(e) / m.sum() * 100,
                   "5일": e[pref + "e5"].mean() * 100, "10일": e[pref + "e10"].mean() * 100, "20일": e[pref + "e20"].mean() * 100,
                   "20일 중앙": e[pref + "e20"].median() * 100, "20일 승률": (e[pref + "e20"].dropna() > 0).mean() * 100}
    return pd.DataFrame(rows).T
t2 = etab("B_", "has_B10"); t3 = etab("L_", "has_L10")
print("\n[②-1] 2차 상승: 10일 안 첫 고가 돌파 → 다음 날 시가 진입 (비용 차감, %)  · 기준선 20일 +0.44%"); print(t2.round(1).to_string())
print("\n[②-2] 저가 이탈: 10일 안 첫 종가 < 급등일 저가 → 다음 날 시가 진입"); print(t3.round(1).to_string())
# ③ 보유 규칙
def rtab(h):
    rows = {}
    for k, m in C_.items():
        e = ev[m]; r = {"n": int(m.sum())}
        for nm, lab in (("hold", "(다) 보유"), ("now", "(가) 즉시"), ("low", "(나) 저가이탈"), ("pc", "(라) 본전이탈")):
            v = e[f"r_{nm}{h}"]; r[lab] = v.mean() * 100
            if nm != "now": r[lab + " 승률"] = (v > 0).mean() * 100; r[lab + " p10"] = v.quantile(0.1) * 100
        r["(나) 정리 %"] = e[f"sold_low{h}"].mean() * 100; r["(나) 후회 %"] = e[f"regret_low{h}"].mean() * 100
        r["(라) 정리 %"] = e[f"sold_pc{h}"].mean() * 100; r["(라) 후회 %"] = e[f"regret_pc{h}"].mean() * 100
        rows[k] = r
    return pd.DataFrame(rows).T
t4 = {h: rtab(h) for h in (10, 20)}
for h in (10, 20): print(f"\n[③] 보유 규칙 h={h} (급등일 종가 기준 수익 %, 매도 비용 0.2%)"); print(t4[h].round(1).to_string())
# ── 견고성: 규칙 − 보유 (h=20), (가) 즉시 − 보유 포함
def boot_diff(e, a, b):
    d = (e[a] - e[b]).to_numpy(float); dates, di = np.unique(e.date.to_numpy(), return_inverse=True); nd = len(dates)
    s = np.bincount(di, d, nd); c = np.bincount(di, None, nd)
    W = np.stack([np.bincount(rng.integers(0, nd, nd), minlength=nd) for _ in range(NB)]).astype(float)
    bt = (W @ s) / (W @ c); return d.mean(), *np.percentile(bt, [2.5, 97.5])
rob = []
for k, m in C_.items():
    for a, lab in (("r_low20", "(나) 저가이탈 − 보유"), ("r_pc20", "(라) 본전이탈 − 보유"), ("r_now20", "(가) 즉시 − 보유")):
        e = ev[m]; v, lo_, hi_ = boot_diff(e, a, "r_hold20")
        r = {"칸": k, "비교": lab, "n": len(e), "전체": v * 100, "하한": lo_ * 100, "상한": hi_ * 100}
        for sk in ["2015–19", "2020–22", "2023–26.4", "몰림날 제외", "2026.1~4 제외"]:
            mm = np.asarray(SUB[sk](pd.DatetimeIndex(e.date))); r[sk] = (e.loc[mm, a] - e.loc[mm, "r_hold20"]).mean() * 100
        sg = np.sign(v); ok1 = (lo_ > 0) or (hi_ < 0)
        same = all(np.sign(r[x]) == sg for x in ["2015–19", "2020–22", "2023–26.4", "몰림날 제외", "2026.1~4 제외"])
        r["판정"] = ("견고" if ok1 and same else "부분" if ok1 else "약함") + (" (+ 규칙 유리)" if v > 0 else " (− 보유 유리)")
        rob.append(r)
t5 = pd.DataFrame(rob); print("\n[③-견고성] 규칙 − 보유 (h=20, %p)"); print(t5.round(1).to_string(index=False))
t5.to_csv(OUT / "surprise_step4_robust.csv", index=False, encoding="utf-8-sig")
import surprise_xlsx
sig5 = None
intro = [("급등(서프라이즈) 연구 — 4단계 경로·보유 관점 (학습 2015-01-02 ~ 2026-04-30, 주 정의)", True),
         ("돌파 = 종가 > 급등일 고가 · 반납 = 종가 ≤ 급등 전날 종가 · 저가 이탈 = 종가 < 급등일 저가", False),
         ("② 진입: 사건 다음 날 시가 매수 → +5/+10/+20일 종가, 왕복 0.23% 차감 · 기준선(유니버스 아무 날) 20일 +0.44%", False),
         ("③ 보유 규칙: 급등일 종가에 보유 중이라고 보고, h 거래일 뒤 모두 현금화. 매도 비용 0.2%. (가) 즉시 실현 · (나) 저가 이탈 다음 날 시가 정리 · (라) 본전(급등 전날 종가) 이탈 다음 날 시가 정리 · (다) 보유", False),
         ("후회 % = 정리한 뒤 h 안에 종가가 급등일 고가를 넘은 비율. p10 = 하위 10% 수익", False),
         ("견고성 = 부트스트랩 95% + 시기 3개 + 몰림날 제외 + 2026.1~4 제외 부호 일치. 빨강 = +, 파랑 = −", False)]
rcols = [c for c in t4[20].columns if c != "n" and "%" not in c]
t5i = t5.set_index(["칸", "비교"])
surprise_xlsx.write_generic(OUT / "급등연구_4단계_경로보유.xlsx", intro, [
    ("갈림", [("① 20일 안 첫 사건 (비율 %, 일수 중앙)", t1, [], False, None, ["칸"])]),
    ("2차상승·저가이탈", [("②-1 10일 안 첫 고가 돌파 → 다음 날 시가 진입 (%)", t2, ["5일", "10일", "20일", "20일 중앙"], False, None, ["칸"]),
                       ("②-2 10일 안 첫 저가 이탈 → 다음 날 시가 진입 (%)", t3, ["5일", "10일", "20일", "20일 중앙"], False, None, ["칸"])]),
    ("보유규칙", [(f"③ 보유 규칙 h=20 (급등일 종가 기준 %)", t4[20], rcols, False, None, ["칸"]),
               (f"③ 보유 규칙 h=10", t4[10], rcols, False, None, ["칸"])]),
    ("견고성", [("규칙 − 보유 (h=20, %p)", t5i, ["전체", "하한", "상한", "2015–19", "2020–22", "2023–26.4", "몰림날 제외", "2026.1~4 제외"], True, None, ["칸", "비교"])])])
