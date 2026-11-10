# -*- coding: utf-8 -*-
"""regime_gate.py — 2-c 보완 6: '조정이 진행 중일 때만 브레드스를 듣는' 게이트형 (케인 2026-11-06 18:28 "고고").

게이트  코스피 200 이 직전 20거래일 최고가 대비 −g 이하 (g = 5% / 8%) OR 최근 10거래일 안에 하루 −3% 이하가 있었음
퇴출    게이트가 켜져 있고 브레드스 판정이 '하락'인 날 → 현금
복귀    퇴출 뒤 브레드스 판정이 '상승'으로 바뀌면 복귀 (게이트와 무관)
평소    ⓐ 보유  ⓑ K200 SMA120 판정을 따름
브레드스 시장 전체 ±0.05 · 유니버스 ±0.03 · 메가갤럭시 ±0.05 (2-c 의 같은 판정기)
평가    regime_events 의 사건 기준·타이밍 + 기간 3개(2016-07~2019 / 2020~2022 / 2023~2026-05) 타이밍 CAGR·MDD
⚠      학습 구간 결과를 보고 만든 설계 — 기간 분할에서 한 구간에만 기대면 채택하지 않는다. 보류 구간 미사용
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import regime_events as RE
from regime_detect import truth

OUT = RE.OUT; A, B = RE.A, RE.B
BR = {"시장±0.05": "브레드스 EMA20 − 250일 중앙 ±0.05", "유니버스±0.03": "[유니버스 브레드스] EMA20 − 250일 중앙 ±0.03",
      "메가±0.05": "[메가갤럭시 브레드스] EMA20 − 250일 중앙 ±0.05"}
PER = [("2016-07-01", "2019-12-31", "16.7–19"), ("2020-01-01", "2022-12-31", "20–22"), ("2023-01-01", "2026-05-29", "23–26.5")]

def gated(k, br_state, g, base):
    r = k.pct_change(); gate = ((k <= k.rolling(20, min_periods=1).max() * (1 - g)) | ((r <= -0.03).astype(int).rolling(10, min_periods=1).max() > 0)).to_numpy()
    b = br_state.to_numpy(bool); bs = base.to_numpy(bool); out = []; mode = False      # mode = 브레드스로 퇴출해 있는 중
    for i in range(len(b)):
        if not mode and gate[i] and not b[i]: mode = True
        elif mode and b[i]: mode = False
        out.append(False if mode else bs[i])
    return pd.Series(out, index=k.index)

def period_timing(k, s):
    r = k.pct_change().fillna(0); pos = s.shift(1).fillna(True).astype(bool); rows = {}
    for a, b, lbl in PER:
        rr = r.loc[a:b].where(pos.loc[a:b], 0.0); eq = (1 + rr).cumprod(); bh = (1 + r.loc[a:b]).cumprod(); y = len(rr) / 252
        rows[lbl] = f"{(eq.iloc[-1] ** (1 / y) - 1) * 100:+.1f}% / {((eq / eq.cummax()) - 1).min() * 100:.0f}%"
        rows[f"{lbl} 보유"] = f"{(bh.iloc[-1] ** (1 / y) - 1) * 100:+.1f}% / {((bh / bh.cummax()) - 1).min() * 100:.0f}%"
    return rows

def main():
    lv = pd.read_parquet(OUT / "layer_levels.parquet"); S = pd.read_parquet(OUT / "detect_states.parquet"); k = lv["코스피 200"]
    st = lambda c: S[c].reindex(k.index).ffill().fillna(True).astype(bool)
    hold = pd.Series(True, index=k.index); sma = st("K200 120일선")
    D = {"(참고) SMA120": sma, "(참고) C2 괴리형+10%복귀": st("[조합 C2] 괴리형 퇴출 → 퇴출 뒤 저점 +10% 반등 복귀"), "(참고) 시장 브레드스 ±0.05 단독": st(BR["시장±0.05"])}
    for bn, bc in BR.items():
        for g in (0.05, 0.08):
            for basen, base in (("평소 보유", hold), ("평소 SMA120", sma)):
                D[f"게이트 {int(g*100)}% · {bn} · {basen}"] = gated(k, st(bc), g, base)
    ib = int(np.searchsorted(k.index, B, side="right")) - 1
    EV = [e for e in RE.events(k, 20, True) if A <= k.index[e["trig"]] and e["trig"] + 60 <= ib]
    for e in EV: e["rec"] = min(e["rec"], ib)
    inside = np.zeros(len(k), bool)
    for e in EV: inside[e["peak"]:e["rec"] + 1] = True
    tr, _ = truth(k); w = (k.index >= A) & (k.index <= B); yrs = w.sum() / 252
    rows, prow = [], []
    for nm, s in D.items():
        P = pd.DataFrame([RE.per_event(k, s, e) for e in EV]); resp = P[P.대응 != "미대응"]
        r = {"판정기": nm, "대응": f"{len(resp)}/{len(P)}", "퇴출 고점대비": resp.퇴출낙폭.median(), "복귀 저점대비": resp.복귀_저점대비.median(),
             "왕복이득 중앙": P.왕복이득.median(), "왕복 +%": (P.왕복이득 > 0).mean() * 100}
        r.update(RE.timing(k, s, inside)); sw = s[w]
        r["연간 전환"] = (sw != sw.shift()).iloc[1:].sum() / yrs; r["zz20 일치%"] = (sw == tr[w]).mean() * 100
        rows.append(r); prow.append({"판정기": nm, **period_timing(k, s)})
    T = pd.DataFrame(rows).set_index("판정기"); pd.set_option("display.width", 340); pd.set_option("display.max_colwidth", 40)
    cols = ["대응", "퇴출 고점대비", "복귀 저점대비", "왕복이득 중앙", "왕복 +%", "CAGR%", "MDD%", "보유 비중%", "현금기간 사건 안(로그%p)", "현금기간 사건 밖(로그%p)", "연간 전환", "zz20 일치%"]
    print(f"조정 사건 {len(EV)}개 · 코스피 200 단순 보유 CAGR {T['단순보유 CAGR%'].iloc[0]:.1f}% / MDD {T['단순보유 MDD%'].iloc[0]:.1f}%\n")
    print(T[cols].round(1).to_string())
    PT = pd.DataFrame(prow).set_index("판정기"); print("\n== 기간별 타이밍 CAGR / MDD ==\n", PT.to_string())
    T.to_csv(OUT / "gate_eval.csv"); PT.to_csv(OUT / "gate_periods.csv"); pd.DataFrame(D).to_parquet(OUT / "gate_states.parquet")

if __name__ == "__main__" and len(sys.argv) == 1:
    main()

# ── v2 (케인 2026-11-06 21:45): 게이트를 '하루 −3% 쇼크'만으로, 쇼크 종류별로 ──
def gate_series(k, kind):
    d = pd.read_parquet(HERE / "data_1104" / "kospi200_daily.parquet")[["Open", "Close"]].astype(float); d.index = pd.to_datetime(d.index)
    d = d.reindex(k.index); cp = d.Close.shift(1); r = d.Close / cp - 1; g = d.Open / cp - 1; i = d.Close / d.Open - 1
    share = np.log1p(g) / np.log1p(r)
    if kind == "−3% 전부": sh = r <= -0.03
    elif kind == "섞임 (갭≤−1% & 장중≤−1%)": sh = (r <= -0.03) & (g <= -0.01) & (i <= -0.01)
    elif kind == "갭 주도+섞임 (갭 몫≥40%)": sh = (r <= -0.03) & (share >= 0.4)
    elif kind == "장중 주도 (갭 몫≤40%)": sh = (r <= -0.03) & (share <= 0.4)
    return (sh.astype(int).rolling(10, min_periods=1).max() > 0).to_numpy(), sh

def gated2(gate, br_state, base):
    b = br_state.to_numpy(bool); bs = base.to_numpy(bool); out = []; mode = False
    for i in range(len(b)):
        if not mode and gate[i] and not b[i]: mode = True
        elif mode and b[i]: mode = False
        out.append(False if mode else bs[i])
    return pd.Series(out, index=br_state.index)

def main_v2():
    lv = pd.read_parquet(OUT / "layer_levels.parquet"); S = pd.read_parquet(OUT / "detect_states.parquet"); k = lv["코스피 200"]
    st = lambda c: S[c].reindex(k.index).ffill().fillna(True).astype(bool)
    hold = pd.Series(True, index=k.index); sma = st("K200 120일선")
    D = {"(참고) SMA120": sma, "(참고) C2 괴리형+10%복귀": st("[조합 C2] 괴리형 퇴출 → 퇴출 뒤 저점 +10% 반등 복귀"),
         "(참고) 게이트 8%(낙폭 or −3%) · 유니버스±0.03 · 평소 SMA120": gated(k, st(BR["유니버스±0.03"]), 0.08, sma)}
    for kind in ("−3% 전부", "섞임 (갭≤−1% & 장중≤−1%)", "갭 주도+섞임 (갭 몫≥40%)", "장중 주도 (갭 몫≤40%)"):
        gt, sh = gate_series(k, kind); w = (k.index >= A) & (k.index <= B)
        print(f"쇼크 [{kind}] 2016-07~2026-05: {int(sh[w].sum())}일 —", ", ".join(str(x.date()) for x in sh[w][sh[w]].index))
        for bn, bc in BR.items():
            for basen, base in (("평소 보유", hold), ("평소 SMA120", sma)):
                D[f"{kind} · {bn} · {basen}"] = gated2(gt, st(bc), base)
    ib = int(np.searchsorted(k.index, B, side="right")) - 1
    EV = [e for e in RE.events(k, 20, True) if A <= k.index[e["trig"]] and e["trig"] + 60 <= ib]
    for e in EV: e["rec"] = min(e["rec"], ib)
    inside = np.zeros(len(k), bool)
    for e in EV: inside[e["peak"]:e["rec"] + 1] = True
    w = (k.index >= A) & (k.index <= B); yrs = w.sum() / 252; rows, prow = [], []
    for nm, s in D.items():
        P = pd.DataFrame([RE.per_event(k, s, e) for e in EV])
        r = {"판정기": nm, "왕복이득 중앙": P.왕복이득.median(), "왕복 +%": (P.왕복이득 > 0).mean() * 100}
        r.update(RE.timing(k, s, inside)); sw = s[w]; r["연간 전환"] = (sw != sw.shift()).iloc[1:].sum() / yrs
        rows.append(r); prow.append({"판정기": nm, **period_timing(k, s)})
    T = pd.DataFrame(rows).set_index("판정기"); pd.set_option("display.width", 340); pd.set_option("display.max_colwidth", 60)
    cols = ["왕복이득 중앙", "왕복 +%", "CAGR%", "MDD%", "보유 비중%", "현금기간 사건 안(로그%p)", "현금기간 사건 밖(로그%p)", "연간 전환"]
    print(f"\n단순 보유 CAGR {T['단순보유 CAGR%'].iloc[0]:.1f}% / MDD {T['단순보유 MDD%'].iloc[0]:.1f}%\n", T[cols].round(1).to_string())
    PT = pd.DataFrame(prow).set_index("판정기"); print("\n== 기간별 CAGR / MDD ==\n", PT[[c for c in PT.columns if "보유" not in c]].to_string())
    T.to_csv(OUT / "gate2_eval.csv"); PT.to_csv(OUT / "gate2_periods.csv"); pd.DataFrame(D).to_parquet(OUT / "gate2_states.parquet")

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "v2":
    main_v2()
