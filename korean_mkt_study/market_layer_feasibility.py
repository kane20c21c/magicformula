# -*- coding: utf-8 -*-
"""market_layer_feasibility.py — 통일장 그림자 추적 1단계: ⑦·⑤ 재료를 그날 저녁 원천으로 바꿔도 판정이 같은가
(케인 2026-11-10 09:56 "둘다 제안대로").

왜    연구 판정은 정리된 과거 자료를 읽는다. 매일 운영하려면 그날 19:10 전에 확정되는 원천이 필요하다.
      - 시장 브레드스: 연구 = 동전주 제외(전년 거래대금 10억·시총 1,000억) 전 종목, KRX 원본은 다음 날 08:01 도착
        그날 대안 = 한국투자 업종지수(코스피 0001·코스닥 1001)의 상승·하락 종목 수 — 필터 없는 전 종목 [필드 존재는 아직 미확인]
      - C2 의 중형 지수: 연구 = KRX 중형 TMI (pykrx 만 가능, 그날 저녁 제공 여부 미확인)
        그날 대안 = 코스피 중형주 (한국투자 업종 코드로 받을 수 있는 지수)
      - 갭 쇼크(코스피 200 시가·종가)는 LLV 20:30 배치가 그날 받는다 — 대안 불필요
무엇  대안 재료로 판정을 다시 만들어 원래 판정과 비교: 상태 일치율, 전환 날짜 차이, 그리고 포트 결과(현금화, T40·T60)
      필터 없는 브레드스는 두 가지 — 전 종목 / 우선주·스팩 제외 (업종지수 종목 수가 어느 쪽인지 몰라서)
기간  2016-07-01 ~ 2026-11-04 (학습 + 보류 — 원천 바꿈의 영향만 보는 것이라 구간을 나누지 않는다)
사용  STOP_UNIV=swing python3 market_layer_feasibility.py  (day 도 같은 방식)
산출  out/regime/feasibility_states.csv · feasibility_port_{swing|day}.csv
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
OUT = HERE / "out" / "regime"; A, B = "2016-07-01", "2026-11-04"

def breadth_unfiltered(exclude_pref_spac):
    import breadth_study as BS
    px = pd.read_parquet(BS.DATA / "prices.parquet", columns=["close", "volume"]).reset_index(); px["ticker"] = px.ticker.astype(str)
    raw = BS.load_raw()
    if exclude_pref_spac:
        meta = pd.read_parquet(BS.DATA / "meta.parquet", columns=["is_pref", "is_spac"]).reset_index()
        bad = set(meta.loc[meta.is_pref.fillna(False) | meta.is_spac.fillna(False), "ticker"].astype(str))
        px = px[~px.ticker.isin(bad)]
        raw = raw[~raw.ticker.isin(bad) & raw.ticker.str[-1].eq("0")]       # 우선주 코드는 끝자리가 0 이 아니다
    tail = raw[raw.date >= BS.PX_END]
    joined = pd.concat([px.loc[px.date <= BS.PX_END, ["date", "ticker", "close", "volume"]], tail[tail.date > BS.PX_END]])
    seam = tail[tail.date == BS.PX_END].set_index("ticker").close
    m = (joined.date == BS.PX_END) & joined.ticker.isin(seam.index)
    joined.loc[m, "close"] = joined.loc[m, "ticker"].map(seam).to_numpy()
    c, _ = BS.daily_counts(joined, None)
    c["ratio"] = c.adv / (c.adv + c.dec)
    return c

def breadth_state(ratio, idx):
    import regime_detect as RD
    e20 = ratio.ewm(span=20, adjust=False).mean().reindex(idx); med = ratio.rolling(250, min_periods=120).median().reindex(idx)
    return RD.hyst(e20 - med, 0.05, -0.05)

def c2_state(k, mid, rb=0.10):
    import regime_detect as RD
    dv = (RD.ma_state(k, 200) & RD.ma_state(mid, 200)).to_numpy(); kv = k.to_numpy(float); s = []; cur = True; low = np.inf
    for i in range(len(kv)):
        if cur:
            if i > 0 and dv[i - 1] and not dv[i]: cur = False; low = kv[i]
        else:
            low = min(low, kv[i])
            if kv[i] >= low * (1 + rb) or (dv[i] and not dv[i - 1]): cur = True
        s.append(cur)
    return pd.Series(s, index=k.index)

def compare(a, b):
    w = a.loc[A:B]; v = b.reindex(w.index)
    fa = w[w != w.shift()].iloc[1:]; fb = v[v != v.shift()].iloc[1:]
    return {"일치율%": (w == v).mean() * 100, "다른 날": int((w != v).sum()), "원래 전환 수": len(fa), "대안 전환 수": len(fb)}

def main():
    import regime_gate as RG, regime_exposure as RE, regime_expand as RX
    import stop_research as sr, stop_port as sp
    k, o, _ = RX.load_base(); lv = pd.read_parquet(OUT / "layer_levels.parquet"); J = RE.judges(k)
    S = pd.read_parquet(OUT / "detect_states.parquet")
    br0 = S["브레드스 EMA20 − 250일 중앙 ±0.05"].reindex(k.index).ffill().fillna(True).astype(bool)
    # 원래 판정 재현 확인 (C2 를 이 스크립트 함수로 다시 만들어 같은지)
    c2_0 = J["④ C2"]; c2_re = c2_state(k, lv["KRX 중형 TMI"])
    gt, _ = RG.gate_series(k, "갭 주도+섞임 (갭 몫≥40%)")
    alt = {}
    for nm, ex in (("전 종목", False), ("우선주·스팩 제외", True)):
        c = breadth_unfiltered(ex); alt[nm] = breadth_state(c["ratio"], k.index)
        print(f"[브레드스 {nm}] 하루 평균 종목 수 {c.loc['2016-07-01':, 'adv'].add(c.loc['2016-07-01':, 'dec']).mean():.0f}")
    c2_alt = c2_state(k, lv["코스피 중형주"])
    rows = [{"비교": "C2 재현 (같은 재료)", **compare(c2_0, c2_re)}, {"비교": "C2: 중형 TMI → 코스피 중형주", **compare(c2_0, c2_alt)}]
    for nm, st in alt.items(): rows.append({"비교": f"브레드스: 연구(필터) → {nm}", **compare(br0, st)})
    V = {"⑦ 원래": J["⑦ C2+갭쇼크"], "⑤ 원래": J["⑤ 갭쇼크·브레드스·평소보유"]}
    for nm, st in alt.items():
        V[f"⑤ 브레드스={nm}"] = RG.gated2(gt, st, pd.Series(True, index=k.index)).astype(bool)
        V[f"⑦ 브레드스={nm}"] = RG.gated2(gt, st, c2_0).astype(bool)
        V[f"⑦ 브레드스={nm} · C2=코스피 중형주"] = RG.gated2(gt, st, c2_alt).astype(bool)
    V["⑦ C2=코스피 중형주 (브레드스 원래)"] = RG.gated2(gt, br0, c2_alt).astype(bool)
    for nm, st in V.items():
        if nm.startswith("⑦") and nm != "⑦ 원래": rows.append({"비교": f"{nm} vs ⑦ 원래", **compare(V["⑦ 원래"], st)})
        if nm.startswith("⑤") and nm != "⑤ 원래": rows.append({"비교": f"{nm} vs ⑤ 원래", **compare(V["⑤ 원래"], st)})
    T = pd.DataFrame(rows); pd.set_option("display.width", 250); print(T.round(2).to_string(index=False))
    T.to_csv(OUT / "feasibility_states.csv", index=False, encoding="utf-8-sig")
    # 포트: 현금화, 2016-07 ~ 2026-11-04 연속, 학습·보류 구간 따로
    p, s, _, _, atr = sr.load(); prow = []
    for Tn in (40, 60):
        base_eq, _ = sp.run_port(p, s, atr, None, Tn, start=A, end=B)
        for nm, st in V.items():
            eq, tr = sp.run_port(p, s, atr, None, Tn, start=A, end=B, **RE.mode_kw("현금화", st))
            r = {"무대": sr.UNIV, "T": Tn, "판정": nm}
            for lab, a_, b_ in (("학습", A, "2026-05-29"), ("보류", "2026-06-01", B)):
                e = eq.equity.loc[a_:b_]; e0 = eq.equity.loc[:pd.Timestamp(a_) - pd.Timedelta(days=1)]
                e0 = e0.iloc[-1] if len(e0) else sp.CAP; path = pd.concat([pd.Series([e0]), e.reset_index(drop=True)])
                r[f"{lab} 수익%"] = (e.iloc[-1] / e0 - 1) * 100; r[f"{lab} MDD%"] = ((path / path.cummax()) - 1).min() * 100
            prow.append(r)
    P = pd.DataFrame(prow); print(P.round(2).to_string(index=False))
    P.to_csv(OUT / f"feasibility_port_{sr.UNIV}.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
