# -*- coding: utf-8 -*-
"""regime_exposure.py — 통일장 2-d: 시장 층 두 장치(줄이기·늘리기)를 스윙 포트에 붙여 본다
(케인 2026-11-09 20:35 "계획대로 진행, 결선 구분 없이 모든 후보를 다 돌려").

엔진   stop_port.run_port (스윙 라이브 방식: 1억 · 10슬롯×1,000만 고정 · 종목당 2슬롯 · 눌림 깊은 순 · t+1 시가 · 비용)
       판정은 그날 종가 → 다음 날 시가 집행. 청산 = 버팀(손절 없음) + 시간 청산 T40 / T60
무대   STOP_UNIV=swing (메가갤럭시: 4조·외인30% 월별) / day (유니버스: LLV 217종목 상시 — 50억 거래대금 필터 아님)
기간   학습 2016-07-01 ~ 2026-05-29. 보류 구간 미사용
단계 A — 줄이기: 판정기 7 × 방식 3 + 기준(장치 없음) + 사후 상한(zigzag 20% 상승 국면 — 실시간 아님) × 방식 3
  판정기 ① SMA120 ② SMA200 ±2% ③ 실시간 zigzag 10% ④ C2 괴리형+10%복귀 ⑤ 갭 쇼크 게이트·시장 브레드스·평소 보유
         ⑥ 같은 게이트·평소 SMA120 ⑦ C2 평소 + 갭 쇼크(10일 안)·시장 브레드스 하락이면 추가 퇴출 (regime_gate.gated2)
  방식   진입중단 = 하락 판정 동안 신규 매수 없음 / 슬롯절반 = 하락 동안 슬롯 상한 5 /
         현금화 = 하락 판정이면 보유 전량 시가×99.5% 매도 + 신규 매수 없음
단계 B — 늘리기 (장치 없음 위). 신호 = regime_expand 의 사건(쿨다운 20): (가)·(나1)·(열)·셋 중 하나
  창     사건 판정일 e → e+1 ~ e+20 거래일 집행
  수단   슬롯확대 = 창 동안 슬롯 상한 15 (슬롯 크기 그대로)
         ETF = 창 첫날 시가에 종목 매수 뒤 남은 현금 전부로 코스피 200(대용) 매수, 창 끝 다음 날 시가 매도.
               그 사이 종목 매수 현금이 모자라면 그만큼 지수를 팔아 바로 쓴다 [가정]. 수수료 0.015%, 거래세·슬리피지 없음
               대용 근거 [실측 11-09]: 102110 vs 코스피 200 일 수익 상관 0.985 (2023-01~2026-11), 연 누적 차 0.2~0.8%p,
               2026-01~05 121.5% vs 121.6%
  ⚠ 단계 B 를 단계 A 에서 고른 줄이기 위에 얹는 시험은 A 선택(케인과 함께) 뒤에 따로 돌린다
반복   후보마다 결정적 1회(눌림 깊은 순 = 라이브) + 후보 순서 무작위 30회 (seed 20261109+it, 후보끼리 같은 seed)
판정 기준 (미리 고정, 2026-11-09)
  ① CAGR ≥ 기준  ② MDD 가 기준보다 작음(얕음)  ③ 세 구간(2016.7~19 / 20~22 / 23~26.5) 중 둘 이상에서 구간 수익 > 기준
     — ①~③ 은 결정적 1회
  ④ 무작위 30회에서 같은 seed 기준보다 CAGR 이 높은 비율 ≥ 2/3
  단계 A 는 같은 방식의 SMA200 판정기보다 나을 때만 다른 판정기를 고른다.
  회수 비율 = (후보 CAGR − 기준) ÷ (같은 방식 사후 상한 CAGR − 기준)
사용   STOP_UNIV=swing python3 regime_exposure.py run  ·  STOP_UNIV=day python3 regime_exposure.py run  ·  python3 regime_exposure.py report
산출   out/regime/exposure_runs_{swing|day}.csv (후보별 결정적 1회 + 무작위 30회 요약) · exposure_mc_{swing|day}.csv (런별 CAGR·MDD)
       exposure_events_{swing|day}.csv (늘리기 사건별 20일 기여) · exposure_summary.csv · exposure_report.txt
"""
import os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
OUT = HERE / "out" / "regime"
A, B = "2016-07-01", "2026-05-29"
PER = [("2016-07-01", "2019-12-31", "16.7-19"), ("2020-01-01", "2022-12-31", "20-22"), ("2023-01-01", "2026-05-29", "23-26.5")]
NMC, SEED, TS, WIN = 30, 20261109, (40, 60), 20
MODES = ("진입중단", "슬롯절반", "현금화")


# ── 판정기·신호 (코스피 200 달력) ──────────────────────────────────────────
def judges(k):
    import regime_gate as RG
    S = pd.read_parquet(OUT / "detect_states.parquet"); G2 = pd.read_parquet(OUT / "gate2_states.parquet")
    st = lambda c: S[c].reindex(k.index).ffill().fillna(True).astype(bool)
    g2 = lambda c: G2[c].reindex(k.index).ffill().fillna(True).astype(bool)
    c2 = st("[조합 C2] 괴리형 퇴출 → 퇴출 뒤 저점 +10% 반등 복귀")
    gt, _ = RG.gate_series(k, "갭 주도+섞임 (갭 몫≥40%)")
    return {"① SMA120": st("K200 120일선"), "② SMA200 ±2%": st("K200 200일선 ±2% 밴드"), "③ zigzag 10%": st("K200 실시간 zigzag 10%"),
            "④ C2": c2, "⑤ 갭쇼크·브레드스·평소보유": g2("갭 주도+섞임 (갭 몫≥40%) · 시장±0.05 · 평소 보유"),
            "⑥ 갭쇼크·브레드스·평소SMA120": g2("갭 주도+섞임 (갭 몫≥40%) · 시장±0.05 · 평소 SMA120"),
            "⑦ C2+갭쇼크": RG.gated2(gt, st(RG.BR["시장±0.05"]), c2).astype(bool)}

def expand_events(k, o):
    import regime_expand as RX
    bs, be = RX.breadth_state(k), RX.breadth_ema10(k)
    H = pd.read_csv(OUT / "expand_heat.csv", index_col=0, parse_dates=True, encoding="utf-8-sig")["heat_recon"].reindex(k.index); h5 = H.rolling(5, min_periods=1).max()
    sig = {"(가) 눌림목": RX.sig_pullback(k, bs), "(나1) 투매 뒤 복귀": RX.sig_recover(k, RX.gap_shock(k, o)),
           "(열) 열량 꺾임": (h5 >= 80) & (H <= h5 - 5) & (be > be.shift(5))}
    allT = np.ones(len(k), bool)
    ev = {nm: RX.events_of(s.fillna(False).astype(bool), allT) for nm, s in sig.items()}
    ev["셋 중 하나"] = sorted(set().union(*ev.values()))
    return ev

def window(k, idx, n=WIN):
    w = np.zeros(len(k), bool)
    for e in idx: w[e:e + n] = True          # 판정일 e ~ e+19 → 집행일 e+1 ~ e+20
    return pd.Series(w, index=k.index)

def mode_kw(mode, st):
    if mode == "진입중단": return dict(gate_in=st)
    if mode == "슬롯절반": return dict(cap=pd.Series(np.where(st, 10, 5), index=st.index))
    if mode == "현금화": return dict(gate_in=st, liq=~st)
    raise ValueError(mode)

def configs(k, kd, ev):
    import stop_regime as SR
    up, _ = SR.regime(k.index)
    C = [("기준", "장치 없음", "", {})]
    for m in MODES: C.append(("A", "사후 상한(zigzag 20%)", m, mode_kw(m, up.astype(bool))))
    for jn, st in judges(k).items():
        for m in MODES: C.append(("A", jn, m, mode_kw(m, st)))
    for sn, idx in ev.items():
        w = window(k, idx)
        C.append(("B", sn, "슬롯확대", dict(cap=pd.Series(np.where(w, 15, 10), index=k.index))))
        C.append(("B", sn, "ETF", dict(etf=dict(open=kd.Open, close=kd.Close, on=w))))
    return C

# ── 실행 ──────────────────────────────────────────────────────────────────
def stats(eq, tr):
    import stop_port as sp
    trs = tr[tr.ticker != "ETF"] if len(tr) else tr
    m = sp.metrics(eq, trs); pr = sp.period_returns(eq, PER)
    tot = ((eq.mv + (eq.etf if "etf" in eq.columns else 0.0)) / eq.equity).mean()
    return {"최종(억)": m["final"], "CAGR": m["cagr"] * 100, "MDD": m["mdd"] * 100, "Sharpe": m["sharpe"],
            "종목 투자비중": m["invested"] * 100, "총 투자비중": tot * 100, "청산수": m["n_trade"], **{f"구간 {a}": v * 100 for a, v in pr.items()}}

def run():
    import stop_research as sr, stop_port as sp, regime_expand as RX
    univ = sr.UNIV; t0 = time.time()
    p, s, _, _, atr = sr.load(); k, o, _ = RX.load_base()
    kd = pd.read_parquet(HERE / "data_1104" / "kospi200_daily.parquet")[["Open", "Close"]].astype(float); kd.index = pd.to_datetime(kd.index); kd = kd.reindex(k.index)
    ev = expand_events(k, o); C = configs(k, kd, ev)
    inwin = lambda e: pd.Timestamp(A) <= k.index[e] and e + WIN + 1 < len(k) and k.index[e + WIN + 1] <= pd.Timestamp(B)
    print(f"[{univ}] 종목 {len(p.tickers)} · 후보 {len(C)} × T{TS} × (결정적 1 + 무작위 {NMC}) · 학습 구간 늘리기 사건 "
          + ", ".join(f"{n} {sum(inwin(e) for e in v)}" for n, v in ev.items()), flush=True)
    rows, mcrows, evrows = [], [], []
    for T in TS:
        base_eq = None
        for stage, nm, mode, kw in C:
            eq, tr = sp.run_port(p, s, atr, None, T, start=A, end=B, **kw)
            r = {"무대": univ, "T": T, "단계": stage, "장치": nm, "방식": mode, **stats(eq, tr)}
            if stage == "기준": base_eq = eq.equity
            if stage == "B":
                for e in ev[nm]:
                    if not inwin(e): continue
                    d0, d1 = k.index[e], k.index[e + WIN + 1]
                    v = eq.equity.reindex([d0, d1]).to_numpy(); b = base_eq.reindex([d0, d1]).to_numpy()
                    evrows.append({"무대": univ, "T": T, "신호": nm, "수단": mode, "사건일": d0.date(), "끝": d1.date(),
                                   "후보 20일%": (v[1] / v[0] - 1) * 100, "기준 20일%": (b[1] / b[0] - 1) * 100,
                                   "기여(%p)": ((v[1] / v[0]) - (b[1] / b[0])) * 100, "기여(백만원)": ((v[1] - v[0]) - (b[1] - b[0])) / 1e6})
            cg, md = [], []
            for it in range(NMC):
                e2, t2 = sp.run_port(p, s, atr, None, T, start=A, end=B, rng=np.random.default_rng(SEED + it), **kw)
                m2 = sp.metrics(e2, t2[t2.ticker != "ETF"] if len(t2) else t2); cg.append(m2["cagr"] * 100); md.append(m2["mdd"] * 100)
                mcrows.append({"무대": univ, "T": T, "단계": stage, "장치": nm, "방식": mode, "it": it, "CAGR": cg[-1], "MDD": md[-1]})
            r.update({"무작위 CAGR 중앙": np.median(cg), "무작위 CAGR 10%": np.quantile(cg, .1), "무작위 CAGR 90%": np.quantile(cg, .9), "무작위 MDD 중앙": np.median(md)})
            rows.append(r)
        print(f"  T{T} 끝 · {time.time() - t0:.0f}초", flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"exposure_runs_{univ}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(mcrows).to_csv(OUT / f"exposure_mc_{univ}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(evrows).to_csv(OUT / f"exposure_events_{univ}.csv", index=False, encoding="utf-8-sig")
    print(f"[{univ}] 저장 완료 · {time.time() - t0:.0f}초")

# ── 판정 ──────────────────────────────────────────────────────────────────
def judge_rows():
    R = pd.concat([pd.read_csv(OUT / f"exposure_runs_{u}.csv", encoding="utf-8-sig") for u in ("swing", "day")], ignore_index=True)
    M = pd.concat([pd.read_csv(OUT / f"exposure_mc_{u}.csv", encoding="utf-8-sig") for u in ("swing", "day")], ignore_index=True)
    R["방식"] = R["방식"].fillna(""); M["방식"] = M["방식"].fillna("")
    pc = [c for c in R.columns if c.startswith("구간 ")]; out = []
    for (u, T), g in R.groupby(["무대", "T"]):
        b = g[g.단계 == "기준"].iloc[0]; mb = M[(M.무대 == u) & (M["T"] == T) & (M.단계 == "기준")].set_index("it").CAGR
        orc = g[g.장치.str.startswith("사후 상한")].set_index("방식").CAGR
        sma = g[g.장치 == "② SMA200 ±2%"].set_index("방식").CAGR
        for _, r in g.iterrows():
            if r.단계 == "기준": continue
            mv = M[(M.무대 == u) & (M["T"] == T) & (M.장치 == r.장치) & (M.방식 == r.방식)].set_index("it").CAGR
            x = dict(r); x["ΔCAGR"] = r.CAGR - b.CAGR; x["ΔMDD"] = r.MDD - b.MDD
            x["구간 우위 수"] = int(sum(r[c] > b[c] for c in pc)); x["무작위 이긴 비율"] = float((mv > mb.reindex(mv.index)).mean() * 100)
            x["무작위 ΔCAGR 중앙"] = float((mv - mb.reindex(mv.index)).median())
            if r.단계 == "A" and r.방식 in orc.index and orc[r.방식] != b.CAGR: x["회수 비율"] = x["ΔCAGR"] / (orc[r.방식] - b.CAGR) * 100
            if r.단계 == "A" and r.방식 in sma.index: x["SMA200 대비 CAGR"] = r.CAGR - sma[r.방식]
            x["①"] = x["ΔCAGR"] >= 0; x["②"] = x["ΔMDD"] > 0; x["③"] = x["구간 우위 수"] >= 2; x["④"] = x["무작위 이긴 비율"] >= 200 / 3
            x["통과"] = x["①"] and x["②"] and x["③"] and x["④"]; x["기준 CAGR"] = b.CAGR; x["기준 MDD"] = b.MDD
            out.append(x)
    return pd.DataFrame(out)

def report():
    J = judge_rows(); J.to_csv(OUT / "exposure_summary.csv", index=False, encoding="utf-8-sig")
    lines = []; P = lambda *a: lines.append(" ".join(str(x) for x in a))
    pd.set_option("display.width", 330); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
    combos = [("swing", 40), ("swing", 60), ("day", 40), ("day", 60)]; cname = {"swing": "메가", "day": "유니"}
    R = pd.concat([pd.read_csv(OUT / f"exposure_runs_{u}.csv", encoding="utf-8-sig") for u in ("swing", "day")], ignore_index=True)
    P("== 기준(장치 없음)과 사후 상한 ==")
    P(R[R.단계.eq("기준") | R.장치.str.startswith("사후 상한")][["무대", "T", "장치", "방식", "최종(억)", "CAGR", "MDD", "Sharpe", "종목 투자비중", "구간 16.7-19", "구간 20-22", "구간 23-26.5", "무작위 CAGR 중앙"]].round(2).to_string(index=False))
    for stage in ("A", "B"):
        g = J[(J.단계 == stage) & ~J.장치.str.startswith("사후 상한")]
        rows = []
        for (nm, md), h in g.groupby(["장치", "방식"], sort=False):
            r = {"장치": nm, "방식": md, "통과 조합": f"{int(h.통과.sum())}/4"}
            for u, T in combos:
                z = h[(h.무대 == u) & (h["T"] == T)].iloc[0]
                r[f"{cname[u]}T{T}"] = f"{z.ΔCAGR:+.1f}/{z.ΔMDD:+.1f}/{z['무작위 이긴 비율']:.0f}/{z['구간 우위 수']}" + ("✓" if z.통과 else "")
            r["ΔCAGR 평균"] = h.ΔCAGR.mean(); r["ΔMDD 평균"] = h.ΔMDD.mean(); r["이긴 비율 평균"] = h["무작위 이긴 비율"].mean()
            if stage == "A": r["회수 평균"] = h["회수 비율"].mean(); r["SMA200 대비 평균"] = h["SMA200 대비 CAGR"].mean()
            rows.append(r)
        P(f"\n== 단계 {stage} — 칸 = ΔCAGR(%p)/ΔMDD(%p, +=얕아짐)/무작위 이긴 비율(%)/구간 우위 수, ✓ = 4기준 통과 ==")
        P(pd.DataFrame(rows).round(2).to_string(index=False))
    EV = pd.concat([pd.read_csv(OUT / f"exposure_events_{u}.csv", encoding="utf-8-sig") for u in ("swing", "day")], ignore_index=True)
    P("\n== 단계 B 사건별 20일 기여 (%p) — 4조합 평균 ==")
    P(EV.groupby(["신호", "수단", "사건일"]).agg(**{"기여 평균%p": ("기여(%p)", "mean"), "최소": ("기여(%p)", "min"), "최대": ("기여(%p)", "max"),
                                             "기준 20일% 평균": ("기준 20일%", "mean")}).round(2).to_string())
    txt = "\n".join(lines); print(txt); (OUT / "exposure_report.txt").write_text(txt, encoding="utf-8")
    try:
        import regime_xlsx; print("엑셀", regime_xlsx.export_exposure(pd.Timestamp.now().strftime("%Y%m%d")))
    except Exception as e: print("⚠ 엑셀 저장 실패:", e)

# ── 2차 (케인 2026-11-10 09:08) — 고른 줄이기 4개 위 늘리기 + 보류 구간 1회 ─────────────
# 고른 줄이기 (케인 11-10): ⑤ 현금화 · ⑦ 현금화 · ④ C2 진입중단 · ⑦ 진입중단
# 늘리기 얹기  각 줄이기 위에 신호 4 × 수단 2. 늘리기는 그 줄이기가 '상승'일 때만 켠다 —
#              슬롯확대 = 창 & 상승이면 상한 15 / ETF = 창 & 상승이면 보유(하락 판정이 되면 다음 날 시가 매도)
#              비교 기준 = 같은 줄이기 단독. 판정 기준 ①~④ 그대로 (② 포함 — 케인 11-10 "빼지 말고")
# 보류 구간 판정 — 실행 전 고정 (2026-11-10 09:25)
#              2016-07-01 ~ 2026-11-04 를 끊지 않고 돌린 뒤 2026-06-01 ~ 11-04 구간만 잰다 (장치 없음과 비교, 조합마다)
#              H1 구간 수익 ≥ 장치 없음  H2 구간 MDD(구간 시작 직전 자산부터 잰 낙폭)가 장치 없음보다 얕음
#              H3 무작위 30회 중 구간 수익이 같은 seed 장치 없음보다 높은 비율 ≥ 2/3
#              대상 = 고른 줄이기 4개만 (사후 상한은 참고). 늘리기는 이번 2차 결과로 고른 뒤 따로 1회
PICKS = [("⑤ 갭쇼크·브레드스·평소보유", "현금화"), ("⑦ C2+갭쇼크", "현금화"), ("④ C2", "진입중단"), ("⑦ C2+갭쇼크", "진입중단")]
HOLD_A, HOLD_B = "2026-06-01", "2026-11-04"

def configs2(k, kd, ev):
    J = judges(k); C = [("기준", "장치 없음", "", "", {})]
    for jn, m in PICKS:
        st = J[jn]; akw = mode_kw(m, st); grp = f"{jn} {m}"
        C.append(("A", grp, "", "", akw))
        for sn, idx in ev.items():
            w = window(k, idx) & st
            C.append(("B", grp, sn, "슬롯확대", {**akw, "cap": pd.Series(np.where(w, 15, 10), index=k.index)}))
            C.append(("B", grp, sn, "ETF", {**akw, "etf": dict(open=kd.Open, close=kd.Close, on=w)}))
    return C

def seg_stats(e, a=HOLD_A, b=HOLD_B):
    e0 = e.loc[:pd.Timestamp(a) - pd.Timedelta(days=1)].iloc[-1]; s = e.loc[a:b]
    path = pd.concat([pd.Series([e0]), s.reset_index(drop=True)])
    return (s.iloc[-1] / e0 - 1) * 100, float(((path / path.cummax()) - 1).min() * 100)

def run2():
    import stop_research as sr, stop_port as sp, regime_expand as RX, stop_regime as SR
    univ = sr.UNIV; t0 = time.time()
    p, s, _, _, atr = sr.load(); k, o, _ = RX.load_base()
    kd = pd.read_parquet(HERE / "data_1104" / "kospi200_daily.parquet")[["Open", "Close"]].astype(float); kd.index = pd.to_datetime(kd.index); kd = kd.reindex(k.index)
    ev = expand_events(k, o); C = configs2(k, kd, ev)
    print(f"[{univ}] 2차 — 후보 {len(C)} × T{TS} × (1 + {NMC}) · 이어서 보류 구간 1회", flush=True)
    rows, mcrows = [], []
    for T in TS:
        for stage, grp, sn, mean, kw in C:
            eq, tr = sp.run_port(p, s, atr, None, T, start=A, end=B, **kw)
            r = {"무대": univ, "T": T, "단계": stage, "묶음": grp, "신호": sn, "수단": mean, **stats(eq, tr)}; cg = []
            for it in range(NMC):
                e2, t2 = sp.run_port(p, s, atr, None, T, start=A, end=B, rng=np.random.default_rng(SEED + it), **kw)
                m2 = sp.metrics(e2, t2[t2.ticker != "ETF"] if len(t2) else t2); cg.append(m2["cagr"] * 100)
                mcrows.append({"무대": univ, "T": T, "단계": stage, "묶음": grp, "신호": sn, "수단": mean, "it": it, "CAGR": cg[-1], "MDD": m2["mdd"] * 100})
            r.update({"무작위 CAGR 중앙": np.median(cg)}); rows.append(r)
        print(f"  2차 T{T} 끝 · {time.time() - t0:.0f}초", flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"exposure2_runs_{univ}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(mcrows).to_csv(OUT / f"exposure2_mc_{univ}.csv", index=False, encoding="utf-8-sig")
    # 보류 구간 1회
    J = judges(k); up, _ = SR.regime(k.index)
    H = [("장치 없음", {})] + [(f"{jn} {m}", mode_kw(m, J[jn])) for jn, m in PICKS] + [(f"(참고) 사후 상한 {m}", mode_kw(m, up.astype(bool))) for m in ("현금화", "진입중단")]
    hrows, hmc = [], []
    for T in TS:
        for nm, kw in H:
            eq, tr = sp.run_port(p, s, atr, None, T, start=A, end=HOLD_B, **kw); ret, mdd = seg_stats(eq.equity)
            seg = eq.loc[HOLD_A:HOLD_B]; trs = tr[(tr.side == "SELL") & (pd.to_datetime(tr.date) >= pd.Timestamp(HOLD_A))] if len(tr) else tr
            hrows.append({"무대": univ, "T": T, "장치": nm, "보류 구간 수익": ret, "보류 구간 MDD": mdd, "보류 구간 종목 투자비중": (seg.mv / seg.equity).mean() * 100,
                          "보류 구간 청산수": len(trs), "보류 구간 거래일": len(seg), "현금 보유일 비중": (seg.n == 0).mean() * 100})
            for it in range(NMC):
                e2, _ = sp.run_port(p, s, atr, None, T, start=A, end=HOLD_B, rng=np.random.default_rng(SEED + it), **kw); r2, m2 = seg_stats(e2.equity)
                hmc.append({"무대": univ, "T": T, "장치": nm, "it": it, "보류 구간 수익": r2, "보류 구간 MDD": m2})
        print(f"  보류 T{T} 끝 · {time.time() - t0:.0f}초", flush=True)
    pd.DataFrame(hrows).to_csv(OUT / f"holdout_runs_{univ}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(hmc).to_csv(OUT / f"holdout_mc_{univ}.csv", index=False, encoding="utf-8-sig")
    print(f"[{univ}] 2차·보류 저장 완료 · {time.time() - t0:.0f}초")

def report2():
    rdc = lambda n: pd.concat([pd.read_csv(OUT / f"{n}_{u}.csv", encoding="utf-8-sig") for u in ("swing", "day")], ignore_index=True)
    R, M = rdc("exposure2_runs"), rdc("exposure2_mc")
    for c in ("신호", "수단"): R[c] = R[c].fillna(""); M[c] = M[c].fillna("")
    pc = [c for c in R.columns if c.startswith("구간 ")]; out = []
    for (u, T, grp), g in R[R.단계 != "기준"].groupby(["무대", "T", "묶음"]):
        a = g[g.단계 == "A"].iloc[0]; z = R[(R.무대 == u) & (R["T"] == T) & (R.단계 == "기준")].iloc[0]
        ma = M[(M.무대 == u) & (M["T"] == T) & (M.묶음 == grp) & (M.단계 == "A")].set_index("it").CAGR
        for _, r in g[g.단계 == "B"].iterrows():
            mv = M[(M.무대 == u) & (M["T"] == T) & (M.묶음 == grp) & (M.신호 == r.신호) & (M.수단 == r.수단)].set_index("it").CAGR
            x = dict(r); x.update({"ΔCAGR": r.CAGR - a.CAGR, "ΔMDD": r.MDD - a.MDD, "구간 우위 수": int(sum(r[c] > a[c] for c in pc)),
                                   "무작위 이긴 비율": float((mv > ma.reindex(mv.index)).mean() * 100), "줄이기 단독 CAGR": a.CAGR, "줄이기 단독 MDD": a.MDD,
                                   "장치 없음 대비 ΔCAGR": r.CAGR - z.CAGR, "장치 없음 대비 ΔMDD": r.MDD - z.MDD})
            x["①"] = x["ΔCAGR"] >= 0; x["②"] = x["ΔMDD"] > 0; x["③"] = x["구간 우위 수"] >= 2; x["④"] = x["무작위 이긴 비율"] >= 200 / 3
            x["통과"] = x["①"] and x["②"] and x["③"] and x["④"]; out.append(x)
    J = pd.DataFrame(out); J.to_csv(OUT / "exposure2_summary.csv", index=False, encoding="utf-8-sig")
    HR, HM = rdc("holdout_runs"), rdc("holdout_mc"); ho = []
    for (u, T), g in HR.groupby(["무대", "T"]):
        b = g[g.장치 == "장치 없음"].iloc[0]; mb = HM[(HM.무대 == u) & (HM["T"] == T) & (HM.장치 == "장치 없음")].set_index("it")["보류 구간 수익"]
        for _, r in g[g.장치 != "장치 없음"].iterrows():
            mv = HM[(HM.무대 == u) & (HM["T"] == T) & (HM.장치 == r.장치)].set_index("it")["보류 구간 수익"]
            x = dict(r); x.update({"장치 없음 수익": b["보류 구간 수익"], "장치 없음 MDD": b["보류 구간 MDD"], "Δ수익": r["보류 구간 수익"] - b["보류 구간 수익"],
                                   "ΔMDD": r["보류 구간 MDD"] - b["보류 구간 MDD"], "무작위 이긴 비율": float((mv > mb.reindex(mv.index)).mean() * 100),
                                   "무작위 구간 수익 중앙": mv.median(), "장치 없음 무작위 중앙": mb.median()})
            x["H1"] = x["Δ수익"] >= 0; x["H2"] = x["ΔMDD"] > 0; x["H3"] = x["무작위 이긴 비율"] >= 200 / 3; x["통과"] = x["H1"] and x["H2"] and x["H3"]; ho.append(x)
    HO = pd.DataFrame(ho); HO.to_csv(OUT / "holdout_summary.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 330); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)
    S = J.groupby(["묶음", "신호", "수단"], sort=False).agg(**{"통과 조합": ("통과", "sum"), "ΔCAGR 평균": ("ΔCAGR", "mean"), "ΔMDD 평균": ("ΔMDD", "mean"),
            "이긴 비율 평균": ("무작위 이긴 비율", "mean"), "구간 우위 평균": ("구간 우위 수", "mean"), "①": ("①", "sum"), "②": ("②", "sum"), "③": ("③", "sum"), "④": ("④", "sum")})
    txt = ["== 2차: 줄이기 위 늘리기 (기준 = 같은 줄이기 단독, 4조합) ==", S.round(2).to_string(),
           "\n== 보류 구간 2026-06-01~11-04 ==", HR.round(2).to_string(index=False),
           "\n== 보류 구간 판정 (장치 없음 대비) ==", HO[["무대", "T", "장치", "보류 구간 수익", "장치 없음 수익", "Δ수익", "보류 구간 MDD", "장치 없음 MDD", "ΔMDD", "무작위 이긴 비율", "H1", "H2", "H3", "통과"]].round(2).to_string(index=False)]
    txt = "\n".join(txt); print(txt); (OUT / "exposure2_report.txt").write_text(txt, encoding="utf-8")
    try:
        import regime_xlsx; print("엑셀", regime_xlsx.export_stage2(pd.Timestamp.now().strftime("%Y%m%d")))
    except Exception as e: print("⚠ 엑셀 저장 실패:", e)

if __name__ == "__main__":
    {"run": run, "report": report, "run2": run2, "report2": report2}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
