# -*- coding: utf-8 -*-
"""
intraday_sim.py — "장중 오르는 종목 고르기" 2단계 규칙 시뮬 (검증 전용, 운영 아님)

입력: intraday_eval.py 1단계 결과 build/id_intra_plus.parquet (p·rank·open_T1·close_T1) — 재학습 없음.
      + features.parquet 에서 익일 High/Low(손절 근사)·회전율·변동성(top10 프로필)
      + LLV core.parquet 의 TIGER200(102110) 시가갭 (쇼크일 판정, 데이포트와 같은 척도 ≤ −2%)
      + (선택) build/id_reg_plus.parquet — 블렌드 변형 (p 순위평균)

전략 정의: 신호일 D 16:20 → 익일 T1 09:00 시가 단일가 매수 → T1 15:30 종가 동시호가 매도. 매일 전액 재투입, 슬롯 동일비중.
  · 양 끝이 단일가라 체결가 = 시가·종가 (슬리피지 0 가정). 왕복 비용 COST(기본 0.23%).
  · 손절 근사(보수적): T1 저가 ≤ 시가×(1−s) 이면 그날 수익률 = −s (체결이 정확히 손절가에 된다고 가정 — 실제로는 갭스루 없음(장중)이라 근사 정확).
    ⚠ 시가→저가 경로를 모르므로 "저가가 먼저 오고 반등" 도 손절로 센다 → 손절 변형은 실제보다 나쁘게 나온다(하한).

변형 (사전 등록 — base 에서 하나씩만 바꿈, 튜닝 아님):
  base      : k=10, 섹터캡 없음, 손절 없음, 비용 0.23, 신호 intra_plus
  k5 / k15  : 슬롯 수
  seccap3   : 같은 대섹터 최대 3 (초과분은 다음 순위로 충원)
  stop2 / stop3 : 손절 −2% / −3% (저가 근사)
  cost30 / cost35 : 왕복 0.30 / 0.35
  blend     : p = rank-avg(intra_plus, reg_plus) — 1차에서 reg 가 좌측 보호를 잃고 우측을 얻었으므로 둘을 섞으면 둘 다 갖는지
  noshock   : 쇼크일(TIGER200 시가갭 ≤ −2%) 미진입 — 09:00 시가갭은 매수 시점에 관측 가능 (데이포트 09:02 판정과 동일 정보)

사전 등록 판정 (base 기준):
  ① 연환산 순익(복리, 비용후) > 0   ② MDD < 20.3% (데이포트 실측 8/18 고점 대비 −20.3%)
  ③ 쇼크일 top10 평균 장중 > 같은 날 유니버스 평균   셋 다 → 3단계 그림자 1개월.
  블렌드 채택 = top10 비용후 일평균 ≥ 0.295 AND 좌측 꼬리 초과 ≤ −6.6 (1단계 GATE2 그대로), 아니면 intra_plus 단독.
  보조(판정 아님): 연도별 분할(2024-09~12 / 2025 / 2026), top10 회전율·변동성 백분위 프로필, 섹터 집중, 최악 5일.

사용 (맥 로컬):
  cd MagicFormula/mop_model/experiments
  python3 intraday_sim.py                      # 모든 변형 → 표 + build/id_sim_summary.csv + build/id_sim_daily_<변형>.csv
  python3 intraday_sim.py --only base,stop2    # 일부만
"""
import argparse, os, sys
from pathlib import Path
import numpy as np, pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import config as cfg  # noqa: E402

COST = 0.23; TAIL = 3.0; SHOCK_GAP = -0.02; DD_REF = 20.3
BASE = dict(k=10, seccap=None, stop=None, cost=COST, blend=False, noshock=False)
VARIANTS = {
    "base": {}, "k5": dict(k=5), "k15": dict(k=15), "seccap3": dict(seccap=3),
    "stop2": dict(stop=0.02), "stop3": dict(stop=0.03), "cost30": dict(cost=0.30), "cost35": dict(cost=0.35),
    "blend": dict(blend=True), "noshock": dict(noshock=True),
}


def load(build):
    R = pd.read_parquet(os.path.join(build, "id_intra_plus.parquet")); R["Date"] = pd.to_datetime(R.Date)
    R = R.dropna(subset=["open_T1", "close_T1", "intra_T1"])
    rp = os.path.join(build, "id_reg_plus.parquet")
    if os.path.exists(rp):
        Q = pd.read_parquet(rp)[["Date", "Ticker", "p"]].rename(columns={"p": "p_reg"}); Q["Date"] = pd.to_datetime(Q.Date)
        R = R.merge(Q, on=["Date", "Ticker"], how="left")
        R["p_blend"] = R.groupby("Date")[["p", "p_reg"]].rank(pct=True).mean(axis=1)
    else:
        print("⚠ id_reg_plus.parquet 없음 — blend 변형 생략"); R["p_blend"] = np.nan
    F = pd.read_parquet(cfg.FEATURES, columns=["Date", "Ticker", "High", "Low", "turnover", "vol20"]); F["Date"] = pd.to_datetime(F.Date)
    F = F.sort_values(["Ticker", "Date"])
    F["High_T1"] = F.groupby("Ticker").High.shift(-1); F["Low_T1"] = F.groupby("Ticker").Low.shift(-1)
    F["xs_to"] = F.groupby("Date").turnover.rank(pct=True); F["xs_vol"] = F.groupby("Date").vol20.rank(pct=True)
    R = R.merge(F[["Date", "Ticker", "High_T1", "Low_T1", "xs_to", "xs_vol"]], on=["Date", "Ticker"], how="left")
    # 쇼크일: TIGER200 T1 시가갭
    try:
        bm = pd.read_parquet(cfg.CORE, columns=["Date", "Ticker", "Open", "Close"]); bm = bm[bm.Ticker == "102110"].sort_values("Date")
        bm["Date"] = pd.to_datetime(bm.Date); bm["gap0"] = bm.Open / bm.Close.shift(1) - 1
        nxt = dict(zip(bm.Date.values[:-1], bm.gap0.values[1:]))       # 신호일 D → T1 시가갭
        R["bm_gap_T1"] = R.Date.map(nxt)
    except Exception as e:
        print(f"⚠ TIGER200 로드 실패 ({e}) — 쇼크일 판정 생략"); R["bm_gap_T1"] = np.nan
    R["ret"] = (R.close_T1 / R.open_T1 - 1) * 100
    R["low_ret"] = (R.Low_T1 / R.open_T1 - 1) * 100
    return R.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def select(g, k, seccap, score):
    g = g.sort_values(score, ascending=False)
    if seccap is None: return g.head(k)
    out, cnt = [], {}
    for _, r in g.iterrows():
        s = r.sector_top
        if cnt.get(s, 0) >= seccap: continue
        out.append(r); cnt[s] = cnt.get(s, 0) + 1
        if len(out) == k: break
    return pd.DataFrame(out)


def simulate(R, k=10, seccap=None, stop=None, cost=COST, blend=False, noshock=False):
    score = "p_blend" if blend else "p"
    if blend and R.p_blend.isna().all(): return None
    rows = []
    for D, g in R.groupby("Date"):
        g = g.dropna(subset=[score])
        if len(g) < 50: continue
        uni = g.ret.mean(); shock = bool(g.bm_gap_T1.iloc[0] <= SHOCK_GAP) if pd.notna(g.bm_gap_T1.iloc[0]) else False
        if noshock and shock:
            rows.append(dict(d=D, ret=0.0, gross=0.0, uni=uni, n=0, shock=shock, stopped=0, nsec=0, xs_to=np.nan, xs_vol=np.nan, left=np.nan, right=np.nan)); continue
        t = select(g, k, seccap, score)
        r = t.ret.copy()
        stopped = 0
        if stop is not None:
            hit = t.low_ret <= -stop * 100
            r[hit] = -stop * 100; stopped = int(hit.sum())
        net = r - cost
        rows.append(dict(d=D, ret=net.mean(), gross=t.ret.mean(), uni=uni, n=len(t), shock=shock, stopped=stopped,
                         nsec=t.sector_top.nunique(), xs_to=t.xs_to.mean(), xs_vol=t.xs_vol.mean(),
                         left=((t.ret <= -TAIL).mean() - (g.ret <= -TAIL).mean()) * 100, right=((t.ret >= TAIL).mean() - (g.ret >= TAIL).mean()) * 100))
    W = pd.DataFrame(rows).set_index("d"); W["equity"] = (1 + W.ret / 100).cumprod()
    return W


def summarize(W):
    eq = W.equity; dd = (eq / eq.cummax() - 1) * 100; n = len(W); yrs = n / 252
    return {"days": n, "일평균순익%": W.ret.mean(), "일표준편차": W.ret.std(), "Sharpe": W.ret.mean() / W.ret.std() * np.sqrt(252) if W.ret.std() > 0 else np.nan,
            "누적%": (eq.iloc[-1] - 1) * 100, "연환산%": ((eq.iloc[-1]) ** (1 / yrs) - 1) * 100, "MDD%": dd.min(), "양의날%": (W.ret > 0).mean() * 100,
            "최악일%": W.ret.min(), "top평균장중%": W.gross.mean(), "유니버스평균%": W.uni.mean(), "좌측꼬리": W.left.mean(), "우측꼬리": W.right.mean(),
            "손절/일": W.stopped.mean(), "섹터수": W.nsec.mean(), "회전율백분위": W.xs_to.mean(), "변동성백분위": W.xs_vol.mean(),
            "쇼크일수": int(W.shock.sum()), "쇼크일top%": W[W.shock].gross.mean() if W.shock.any() else np.nan, "쇼크일유니버스%": W[W.shock].uni.mean() if W.shock.any() else np.nan}


def main(only=None):
    build = cfg.OUT_DIR; R = load(build)
    print(f"입력 {len(R)}행 · {R.Date.nunique()}일 ({R.Date.min().date()}~{R.Date.max().date()}) · 쇼크일 {int(R.groupby('Date').bm_gap_T1.first().le(SHOCK_GAP).sum())}일")
    names = [v for v in VARIANTS if not only or v in only]
    res = {}; dailies = {}
    for v in names:
        W = simulate(R, **{**BASE, **VARIANTS[v]})
        if W is None: continue
        res[v] = summarize(W); dailies[v] = W
        W.to_csv(os.path.join(build, f"id_sim_daily_{v}.csv"))
    T = pd.DataFrame(res).T
    pd.set_option("display.width", 250); print("\n== 변형별 요약 (비용후, 복리) =="); print(T.round(3).to_string())
    T.to_csv(os.path.join(build, "id_sim_summary.csv"))
    # 연도별 (base)
    if "base" in dailies:
        W = dailies["base"]; W["yr"] = W.index.year
        Y = W.groupby("yr").apply(lambda x: pd.Series({"days": len(x), "일평균순익%": x.ret.mean(), "누적%": ((1 + x.ret / 100).prod() - 1) * 100,
                                                       "양의날%": (x.ret > 0).mean() * 100, "MDD%": ((x.equity / x.equity.cummax() - 1) * 100).min(), "top평균장중%": x.gross.mean(), "유니버스%": x.uni.mean()}))
        print("\n== base 연도별 =="); print(Y.round(3).to_string())
        print("\n== base 최악 5일 =="); print(W.nsmallest(5, "ret")[["ret", "gross", "uni", "shock", "nsec"]].round(2).to_string())
        # 판정
        b = res["base"]; c = [b["연환산%"] > 0, abs(b["MDD%"]) < DD_REF, (b["쇼크일top%"] > b["쇼크일유니버스%"]) if pd.notna(b["쇼크일top%"]) else None]
        print(f"\n== 사전 등록 판정 (base) — ①연환산>0 {'O' if c[0] else 'X'} ({b['연환산%']:+.1f}%) · ②MDD<{DD_REF} {'O' if c[1] else 'X'} ({b['MDD%']:.1f}%) · "
              f"③쇼크일 top>유니버스 {('O' if c[2] else 'X') if c[2] is not None else '판정불가'} ({b['쇼크일top%']:+.2f} vs {b['쇼크일유니버스%']:+.2f}, n={b['쇼크일수']}) → "
              f"{'3단계 그림자' if all(x for x in c if x is not None) and c[2] is not None else '보류'}")
    if "blend" in res:
        b = res["blend"]; ok = b["일평균순익%"] >= 0.295 and b["좌측꼬리"] <= -6.6
        print(f"== 블렌드 판정 — 비용후 {b['일평균순익%']:+.3f} (≥0.295 {'O' if b['일평균순익%'] >= 0.295 else 'X'}) · 좌측 {b['좌측꼬리']:+.2f} (≤−6.6 {'O' if b['좌측꼬리'] <= -6.6 else 'X'}) → {'블렌드 채택' if ok else 'intra_plus 단독 유지'}")
    print("\n⚠ 손절 변형은 저가 근사라 하한이다. 시가 단일가·종가 동시호가 체결 가정, 슬리피지 0. 유니버스는 현재 명단(생존편향).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=None, help="쉼표 구분 변형 이름"); a = ap.parse_args()
    main(only=a.only.split(",") if a.only else None)
