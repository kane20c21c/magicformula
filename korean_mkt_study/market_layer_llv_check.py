# -*- coding: utf-8 -*-
"""market_layer_llv_check.py — 통일장 그림자 ② 착수 전 점검: LLV 운영 재료로 ⑦·⑤ 를 만들면 연구 판정과 같은가
(케인 2026-11-10 11:06, LLV 미니 백필 → 에어 당겨오기 뒤).

재료 바꿈 (연구 → 운영)
  코스피 200 시가·종가 : data_1104/kospi200_daily      → LLV tickers/KOSPI200.parquet
  KRX 중형 TMI         : layer_levels (pykrx)          → LLV tickers/KRX_TMI_MID.parquet (KRX Open API)
  시장 브레드스 비율    : 동전주 제외 필터, 2016~        → LLV market_breadth KRX 확정치 (필터 없는 전 종목, 2025-05-02~)
비교 구간
  W1 2025-11-03~2026-11-04 : 운영 브레드스 중앙값이 최소 120일을 채운 뒤 (연구 설정과 같은 최소 창)
  W2 2026-06-01~2026-11-04 : 보류 구간 (250일 창이 거의 다 찬 뒤)
사용  python3 market_layer_llv_check.py   →  out/regime/llv_check_*.csv
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
LLV = HERE.parents[1] / "LongLiveVault"; sys.path.insert(0, str(LLV))
OUT = HERE / "out" / "regime"
W = {"W1 2025-11~": ("2025-11-03", "2026-11-04"), "W2 보류 2026-06~": ("2026-06-01", "2026-11-04")}


def llv_index(name):
    d = pd.read_parquet(LLV / "data" / "ohlcv" / "tickers" / f"{name}.parquet")
    d["Date"] = pd.to_datetime(d["Date"]); return d.set_index("Date").sort_index()


def gate_from(op, cl, idx):
    """regime_gate.gate_series('갭 주도+섞임 (갭 몫≥40%)') 와 같은 식 — 시가·종가만 바꿔 넣는다."""
    op, cl = op.reindex(idx), cl.reindex(idx); cp = cl.shift(1); r = cl / cp - 1; g = op / cp - 1
    share = np.log1p(g) / np.log1p(r); sh = (r <= -0.03) & (share >= 0.4)
    return (sh.astype(int).rolling(10, min_periods=1).max() > 0).to_numpy(), sh


def cmp(a, b, a_, b_):
    w = a.loc[a_:b_]; v = b.reindex(w.index)
    diff = w.index[(w != v).to_numpy()]
    fa = int((w != w.shift()).iloc[1:].sum()); fb = int((v != v.shift()).iloc[1:].sum())
    return {"일수": len(w), "일치%": round((w == v).mean() * 100, 1), "다른 날": len(diff),
            "전환 수 (연구/운영)": f"{fa}/{fb}", "다른 날 범위": (f"{diff.min():%m-%d}~{diff.max():%m-%d}" if len(diff) else "")}


def main():
    import regime_gate as RG, regime_exposure as RE, regime_expand as RX, market_layer_feasibility as F
    from stolab_data import market_breadth as mb
    k, o, _ = RX.load_base(); idx = k.index; J = RE.judges(k)
    lv = pd.read_parquet(OUT / "layer_levels.parquet")
    S = pd.read_parquet(OUT / "detect_states.parquet")
    br0 = S[RX.BR_STATE].reindex(idx).ffill().fillna(True).astype(bool)
    rows = []

    # 1) 재료 값 대조
    K = llv_index("KOSPI200"); T = llv_index("KRX_TMI_MID")
    d = pd.read_parquet(HERE / "data_1104" / "kospi200_daily.parquet")[["Open", "Close"]].astype(float); d.index = pd.to_datetime(d.index)
    for nm, a, b in (("K200 종가", d.Close, K.Close), ("K200 시가", d.Open, K.Open), ("중형 TMI 종가", lv["KRX 중형 TMI"], T.Close)):
        j = pd.concat([a.rename("r"), b.rename("l")], axis=1).dropna().loc["2015-01-01":"2026-11-04"]
        rows.append({"항목": nm, "겹친 날": len(j), "최대 절대차": round((j.r - j.l).abs().max(), 4),
                     "차 > 0.01 인 날": int(((j.r - j.l).abs() > 0.01).sum())})
    br = pd.read_csv(HERE / "out" / "breadth" / "breadth_market.csv", parse_dates=["date"]).set_index("date")
    s_llv = mb.ratio_series(mb.load_breadth(LLV / "data" / "market_breadth.parquet"), "KRX", "close")
    s_unf = F.breadth_unfiltered(False)["ratio"]
    for nm, a in (("브레드스 비율: 연구 필터 vs LLV 전 종목", br["ratio"]), ("브레드스 비율: 연구 전 종목 vs LLV 전 종목", s_unf)):
        j = pd.concat([a.rename("r"), s_llv.rename("l")], axis=1).dropna().loc[:"2026-11-04"]
        dd = (j.r - j.l)
        rows.append({"항목": nm, "겹친 날": len(j), "최대 절대차": round(dd.abs().max(), 4), "평균차(연구−LLV)": round(dd.mean(), 4),
                     "차 > 0.01 인 날": int((dd.abs() > 0.01).sum()), "상관": round(j.r.corr(j.l), 4)})
    A1 = pd.DataFrame(rows); print("\n[1] 재료 값 대조"); print(A1.to_string(index=False))


    # 2) 상태 대조 — 재료를 하나씩 바꿔 어디서 갈리는지 본다
    br_re = F.breadth_state(br["ratio"], idx)                                  # 재현 확인 (같은 재료)
    br_unf = F.breadth_state(s_unf, idx)                                       # 연구 전 종목, 2016~ 긴 이력
    splice = pd.concat([s_unf.loc[:"2025-05-01"], s_llv]).sort_index()
    br_spl = F.breadth_state(splice[~splice.index.duplicated(keep="last")], idx)  # 앞은 연구 전 종목 + 뒤는 LLV
    br_llv = F.breadth_state(s_llv, idx)                                       # 운영 그대로 (LLV 만, 2025-05~)
    c2_0 = J["④ C2"]; c2_llv = F.c2_state(K.Close.reindex(idx), T.Close.reindex(idx))
    gt0, _ = RG.gate_series(k, "갭 주도+섞임 (갭 몫≥40%)"); gtl, _ = gate_from(K.Open, K.Close, idx)
    hold = pd.Series(True, index=idx)
    P7, P5 = J["⑦ C2+갭쇼크"], J["⑤ 갭쇼크·브레드스·평소보유"]
    V = [("브레드스 재현 (연구 필터 재계산)", br0, br_re),
         ("브레드스: 연구 필터 → 연구 전 종목(긴 이력)", br0, br_unf),
         ("브레드스: 연구 필터 → 연구 앞 + LLV 뒤", br0, br_spl),
         ("브레드스: 연구 필터 → LLV 만 (운영)", br0, br_llv),
         ("C2: 연구 → LLV K200·TMI", c2_0, c2_llv),
         ("⑦ 관문만 LLV K200", P7, RG.gated2(gtl, br0, c2_0).astype(bool)),
         ("⑦ C2만 LLV", P7, RG.gated2(gt0, br0, c2_llv).astype(bool)),
         ("⑦ 브레드스만 LLV", P7, RG.gated2(gt0, br_llv, c2_0).astype(bool)),
         ("⑦ 전부 LLV (운영)", P7, RG.gated2(gtl, br_llv, c2_llv).astype(bool)),
         ("⑤ 전부 LLV (운영)", P5, RG.gated2(gtl, br_llv, hold).astype(bool))]
    rows = []
    for nm, a, b in V:
        for wn, (a_, b_) in W.items():
            rows.append({"비교": nm, "구간": wn, **cmp(a, b, a_, b_)})
    A2 = pd.DataFrame(rows); print("\n[2] 상태 대조 (True=보유)"); print(A2.to_string(index=False))
    a_, b_ = W["W1 2025-11~"]
    D = pd.DataFrame({"⑦ 연구": P7, "⑦ 운영": V[8][2], "⑤ 연구": P5, "⑤ 운영": V[9][2], "브레드스 연구": br0,
                      "브레드스 운영": br_llv, "C2 연구": c2_0, "C2 운영": c2_llv,
                      "비율 연구필터": br["ratio"].reindex(idx), "비율 LLV": s_llv.reindex(idx)}).loc[a_:b_]
    D = D[(D["⑦ 연구"] != D["⑦ 운영"]) | (D["⑤ 연구"] != D["⑤ 운영"]) | (D["브레드스 연구"] != D["브레드스 운영"])]
    print(f"\n[3] W1 에서 하나라도 다른 날 {len(D)}일"); print(D.head(40).to_string())
    with pd.ExcelWriter(OUT / "통일장_운영재료_대조_20261110.xlsx") as xw:
        A1.to_excel(xw, "재료 값", index=False); A2.to_excel(xw, "상태 대조", index=False); D.to_excel(xw, "다른 날")


if __name__ == "__main__":
    main()
