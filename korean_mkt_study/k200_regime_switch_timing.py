"""장단현포트 운영 시점 확인 (2026-11-23 케인 지시 — 구현 전 확인)
연구판(k200_regime_switch.py): t 종가 신호 → t+1 시가 체결, 틀 = t 종가 ⑦.
운영판: t 15:10 잠정 신호(≈ t 종가) → t 15:17 체결(≈ t 종가), 틀 = t-1 확정 ⑦ (08:40 확정분).
  근사: 신호·체결 모두 t 종가. ⑦ 는 매매일 전날 판정 → 연구판과 같은 배열(mode) 을 그대로 쓴다.
두 판을 ① 분기 리셋 ② 리셋 없는 연속 운용(2020-01-02 ~ 2026-11-20)으로 비교.
"""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "k200_regime_switch.py")).read().split("qs = pd.period_range")[0])
k5 = k10                                                    # SIG_N=5 로 실행됨
mvS_c = np.nan_to_num(k5).astype(int)                       # 매매일 i 의 신호 = i 자신(종가≈15:10)
mvL_c = np.zeros(N, int)
for i in range(1, N):
    if idx[i] >= pd.Timestamp(T0):
        tk = picks.get(idx[i].strftime("%Y-%m")); s = SG[i, [pos[t] for t in tk]]
        up, dn = int((s > 0).sum()), int((s < 0).sum())
        mvL_c[i] = 2 if up == 3 else -2 if dn == 3 else 1 if up == 2 else -1 if dn == 2 else 0
CLOSE = dict(LO=SYN["LC"], LC=SYN["LC"], SO=SYN["SC"], SC=SYN["SC"])

def run(i0, i1, mvHold, mvCash, P, log=False):
    m0 = p7.values[i0]; cur = "cash" if m0 == True else "short"
    uL = 10 * SLOT / P["LC"][i0]; uS = 10 * SLOT / P["SC"][i0] if cur == "short" else 0.0; cash = 10 * SLOT if cur == "cash" else 0.0
    vals = []
    for i in range(i0 + 1, i1 + 1):
        m = mode[i]; want = "cash" if m == 1 else "short" if m == 0 else cur
        if want != cur:
            if want == "short": uS += cash / (P["SO"][i] * (1 + FEE)); cash = 0.0
            else: cash += uS * P["SO"][i] * (1 - FEE); uS = 0.0
            cur = want
        mv = mvHold[i] if cur == "cash" else mvCash[i]
        for _ in range(abs(mv)):
            if cur == "cash":
                if mv > 0:
                    amt = min(cash, SLOT)
                    if amt <= 1e-6: break
                    uL += amt / (P["LO"][i] * (1 + FEE)); cash -= amt
                else:
                    q = min(uL, SLOT / P["LO"][i])
                    if q <= 1e-12: break
                    uL -= q; cash += q * P["LO"][i] * (1 - FEE)
            else:
                so, bo = (P["SO"][i], P["LO"][i]) if mv > 0 else (P["LO"][i], P["SO"][i])
                q = min(uS if mv > 0 else uL, SLOT / so)
                if q <= 1e-12: break
                qb = q * so * (1 - FEE) / (bo * (1 + FEE))
                if mv > 0: uS -= q; uL += qb
                else: uL -= q; uS += qb
        if log: vals.append(uL * P["LC"][i] + uS * P["SC"][i] + cash)
    fin = (uL * P["LC"][i1] + uS * P["SC"][i1]) * (1 - FEE) + cash
    return (fin / (20 * SLOT) - 1, np.array(vals)) if log else fin / (20 * SLOT) - 1

V = {"연구판(t+1 시가)": (mvS, mvL, SYN), "운영판(t 15:17≈종가)": (mvS_c, mvL_c, CLOSE)}
rows = []
for q in pd.period_range("2020Q1", "2026Q4", freq="Q"):
    ins = np.where((idx >= q.start_time) & (idx <= min(q.end_time, pd.Timestamp(T1))))[0]
    i0 = ins[0] - 1; ok = ins[~np.isnan(mode[ins])]; i1 = ok[-1]
    r = dict(분기=str(q), KOSPI200=KC[i1] / KC[i0] - 1)
    for k, (a, b, P) in V.items(): r[k] = run(i0, i1, a, b, P)
    r["롱10+현금10"] = 0.5 * ((1 + r["KOSPI200"]) * (1 - FEE) - 1); rows.append(r)
Q = pd.DataFrame(rows).set_index("분기"); F = Q.drop("2026Q4"); up = F.KOSPI200 > 0
pd.set_option("display.width", 220); print((Q * 100).round(1).to_string())
for c in Q.columns[1:]:
    print(f"{c}: 평균 {F[c].mean()*100:.2f} 중앙 {F[c].median()*100:.2f} 최악 {F[c].min()*100:.1f} 플러스 {(F[c]>0).sum()} 상승 {F[c][up].mean()*100:.1f} 하락 {F[c][~up].mean()*100:.1f}")
d = F["운영판(t 15:17≈종가)"] - F["연구판(t+1 시가)"]; print(f"운영판 − 연구판: 이긴 {(d>0).sum()}/27, 평균 {d.mean()*100:+.2f}%p, 중앙 {d.median()*100:+.2f}%p")
# 연속 운용 (리셋 없음)
i0 = int(np.where(idx >= pd.Timestamp(T0))[0][0]) - 1; i1 = int(np.where(~np.isnan(mode))[0][-1])
out = {}
for k, (a, b, P) in V.items():
    ret, v = run(i0, i1, a, b, P, log=True); s = pd.Series(v, index=idx[i0 + 1:i1 + 1]); dd = (s / s.cummax() - 1).min()
    yrs = (idx[i1] - idx[i0]).days / 365.25; out[k] = s
    print(f"[연속] {k}: 누적 {ret*100:+.1f}% · 연율 {((1+ret)**(1/yrs)-1)*100:+.1f}% · MDD {dd*100:.1f}%")
kk = pd.Series(KC[i0 + 1:i1 + 1], index=idx[i0 + 1:i1 + 1]); print(f"[연속] KOSPI200 {kk.iloc[-1]/KC[i0]*100-100:+.1f}% MDD {(kk/kk.cummax()-1).min()*100:.1f}%  · 롱10+현금10 {0.5*(kk.iloc[-1]/KC[i0]-1)*100:+.1f}%")
yr = pd.DataFrame({k: s.groupby(s.index.year).last() for k, s in out.items()}); yr.loc[2019] = 20 * SLOT; yr = yr.sort_index()
print("[연속] 연도별 수익률(%):"); print((yr.pct_change().dropna() * 100).round(1).to_string())
Q.to_csv(os.path.join(OUT2, "timing_quarterly.csv"), encoding="utf-8-sig"); pd.DataFrame(out).to_csv(os.path.join(OUT2, "timing_continuous_daily.csv"), encoding="utf-8-sig")
