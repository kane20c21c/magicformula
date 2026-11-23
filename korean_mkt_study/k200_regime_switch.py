"""통일장 ⑦ 로 틀을 바꾸는 전진 검증 (2026-11-23 케인 확정 설계)

  ⑦ 보유 → 롱/현금 틀 + KOSPI200 자기추세 5일 신호      (기본 ±1, 비교 ±2)
  ⑦ 현금 → 롱/숏  틀 + 선행종목 5일 신호               (기본 ② 2개 이상, 비교 ③)
판정 시점: t 종가 기준 ⑦(regime_history.p7) → t+1 시가에 반영 (신호와 같은 시점, 미래 정보 없음).
틀 전환: ⑦ 이 바뀐 다음 날 시가에 롱이 아닌 쪽 전부를 바꾼다 (현금 → 인버스 매수 / 인버스 → 매도 후 현금). 롱 슬롯은 유지.
          그다음 그날 신호대로 슬롯 이동. 분기 시작 틀 = 직전 분기 말 ⑦.
선행종목 선정·가격·수수료·분기 리셋은 k200_lead_walkforward.py 와 같다 (SIG_N=5, SEL_N=10 으로 실행해 재사용).
비교: 선행② 롱/숏 단독 · 자기±1 롱/현금 단독 · 롱10+현금10 · ⑦ 판정만(보유=롱10+현금10, 현금=롱10+숏10, 신호 매매 없음)
⑦ 이력 = MagicFormula/output/market_layer/regime_history.parquet (유효 2016-04-22~, 확정 재료만).
"""
import os, numpy as np, pandas as pd
os.environ["SIG_N"], os.environ["SEL_N"] = "5", "10"
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "k200_lead_walkforward.py")).read()
exec(src.split("N = len(idx); mv2")[0])            # picks, idx, K, SG, k10(=5일), SYN, REAL, sim, sim_cash, lead_move …
OUT2 = os.path.join(ROOT, "out", "k200_regime_switch"); os.makedirs(OUT2, exist_ok=True)
H = pd.read_parquet(os.path.join(ROOT, "..", "output", "market_layer", "regime_history.parquet"))
p7 = H["p7"].reindex(idx)
N = len(idx)
mvL = np.zeros(N, int); mvS = np.r_[0, np.nan_to_num(k10[:-1])].astype(int)
for i in range(1, N):
    if idx[i] >= pd.Timestamp(T0): mvL[i] = lead_move(i)[0]
mvL3 = np.where(np.abs(mvL) == 2, mvL, 0)
mode = np.r_[np.nan, p7.values[:-1].astype("float")]     # 매매일 i 에 쓰는 ⑦ = 전날(i-1) 종가 판정. 1=보유, 0=현금

def sim_switch(i0, i1, mvHold, mvCash, P, trade=True):
    """mvHold: ⑦보유일 신호(롱/현금), mvCash: ⑦현금일 신호(롱/숏). trade=False 면 틀만 바꾸고 신호 매매 없음."""
    m0 = p7.values[i0]; cur = "cash" if m0 == True else "short"
    uL = 10 * SLOT / P["LC"][i0]; uS = 10 * SLOT / P["SC"][i0] if cur == "short" else 0.0; cash = 10 * SLOT if cur == "cash" else 0.0
    fee = 0.0; sw = 0; hold_days = 0
    for i in range(i0 + 1, i1 + 1):
        m = mode[i]; want = "cash" if m == 1 else "short" if m == 0 else cur
        if want != cur:
            sw += 1
            if want == "short":
                uS += cash / (P["SO"][i] * (1 + FEE)); fee += cash * FEE / (1 + FEE); cash = 0.0
            else:
                pr = uS * P["SO"][i]; cash += pr * (1 - FEE); fee += pr * FEE; uS = 0.0
            cur = want
        hold_days += cur == "cash"
        mv = (mvHold[i] if cur == "cash" else mvCash[i]) if trade else 0
        for _ in range(abs(mv)):
            if cur == "cash":
                if mv > 0:
                    amt = min(cash, SLOT)
                    if amt <= 1e-6: break
                    uL += amt / (P["LO"][i] * (1 + FEE)); fee += amt * FEE / (1 + FEE); cash -= amt
                else:
                    q = min(uL, SLOT / P["LO"][i])
                    if q <= 1e-12: break
                    uL -= q; cash += q * P["LO"][i] * (1 - FEE); fee += q * P["LO"][i] * FEE
            else:
                so, bo = (P["SO"][i], P["LO"][i]) if mv > 0 else (P["LO"][i], P["SO"][i])
                q = min(uS if mv > 0 else uL, SLOT / so)
                if q <= 1e-12: break
                pr = q * so * (1 - FEE); qb = pr / (bo * (1 + FEE)); fee += q * so * FEE + qb * bo * FEE
                if mv > 0: uS -= q; uL += qb
                else: uL -= q; uS += qb
    val = uL * P["LC"][i1] + uS * P["SC"][i1]
    return (val * (1 - FEE) + cash) / (20 * SLOT) - 1, sw, hold_days

qs = pd.period_range("2020Q1", "2026Q4", freq="Q"); rows = []
for q in qs:
    ins = np.where((idx >= q.start_time) & (idx <= min(q.end_time, pd.Timestamp(T1))))[0]
    i0, i1 = ins[0] - 1, ins[-1]
    if pd.isna(p7.values[i0]) or np.isnan(mode[ins]).any():
        last = int(np.where(~np.isnan(mode[ins]))[0].max()) if (~np.isnan(mode[ins])).any() else -1
        i1 = ins[last]                          # ⑦ 이력이 끝난 날까지만 (2026-11-19 판정 → 11-20 매매까지 가능)
    r = dict(분기=str(q), 끝=idx[i1].date(), KOSPI200=KC[i1] / KC[i0] - 1)
    r["전환형②±1"], r["틀전환수"], hd = sim_switch(i0, i1, mvS, mvL, SYN)
    r["⑦보유일비율"] = hd / (i1 - i0)
    r["전환형③±2"] = sim_switch(i0, i1, mvS * 2, mvL3, SYN)[0]
    r["⑦판정만"] = sim_switch(i0, i1, mvS, mvL, SYN, trade=False)[0]
    r["선행②_롱숏"] = sim(i0, i1, mvL, SYN)[0]
    r["자기±1_롱현금"] = sim_cash(i0, i1, mvS, SYN)
    r["롱10+현금10"] = 0.5 * ((1 + r["KOSPI200"]) * (1 - FEE) - 1)
    if not np.isnan(REAL["SO"][i0 + 1:i1 + 1]).any() and not np.isnan(REAL["SC"][i0]):
        r["전환형②±1_실제ETF"] = sim_switch(i0, i1, mvS, mvL, REAL)[0]
    rows.append(r)
Q = pd.DataFrame(rows); Q.to_csv(f"{OUT2}/quarterly_results.csv", index=False, encoding="utf-8-sig")
C = ["전환형②±1", "전환형③±2", "⑦판정만", "선행②_롱숏", "자기±1_롱현금", "롱10+현금10"]
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
X = Q.set_index("분기"); print((X[["KOSPI200"] + C] * 100).round(1).join(X[["⑦보유일비율", "틀전환수"]].round(2)).to_string())
F = X.drop("2026Q4"); up = F.KOSPI200 > 0
S = pd.DataFrame({c: {"평균": F[c].mean(), "중앙": F[c].median(), "최악": F[c].min(), "플러스": (F[c] > 0).sum(),
                      "상승분기": F[c][up].mean(), "하락분기": F[c][~up].mean(), "1위": (F[C].idxmax(axis=1) == c).sum()} for c in C}).T
print((S[["평균", "중앙", "최악", "상승분기", "하락분기"]] * 100).round(2).join(S[["플러스", "1위"]]).to_string())
for b in C[2:]:
    d = F["전환형②±1"] - F[b]; print(f"전환형②±1 − {b}: 이긴 {(d > 0).sum()}/27, 평균 {d.mean() * 100:+.2f}%p, 중앙 {d.median() * 100:+.2f}%p")
re = X.dropna(subset=["전환형②±1_실제ETF"]); print("실제ETF 대조:", ((re[["전환형②±1", "전환형②±1_실제ETF"]]) * 100).round(1).to_dict("list"))
