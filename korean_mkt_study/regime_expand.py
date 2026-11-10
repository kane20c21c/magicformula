# -*- coding: utf-8 -*-
"""regime_expand.py — 통일장 ① 단계: 시장 층 '늘리기' 신호와 발열계 열량을 사건 기준으로 잰다
(케인 2026-11-09 20:04 "진행. 제안대로"). 포트 시험(2-d)이 아니라, 신호 자체가 쓸 만한지 거르는 단계.

기준선  코스피 200 (2-c 와 같음, out/regime/layer_levels.parquet). 보조: 유니버스 동일가중 (같은 파일).
기간    학습 2016-07-01 ~ 2026-05-29 (2-c 와 같음). 신호 뒤 20거래일이 학습 끝을 넘는 사건은 뺀다. 보류 구간 미사용.

신호 정의 — 결과를 보기 전에 고정 (2026-11-09 20:40)
  (가)  시장 눌림목: 코스피 200 종가 > 200일선 AND 최근 20거래일 안에 '직전 20일 최고 종가 대비 −5% 이하'가 있었음
        AND 시장 브레드스 판정(EMA20 − 250일 중앙 ±0.05)이 오늘 하락→상승으로 바뀜
  (나1) 투매 뒤 복귀: 갭 쇼크일(코스피 200 하루 −3% 이하 & 갭 몫 ≥ 40%, 2-c 게이트와 같음) 뒤 60거래일 안에
        쇼크일 이후 최저 종가 대비 종가가 +10% 이상인 첫날 (C2 의 복귀 조건)
  (나2) 반등 룰 동시 다발: 같은 날 반등 룰 신호(드라이브 낙폭 ≤ −20% & 멈춘 첫날 ≥ 1.65σ) 종목 수 ≥ N
        종목 = 반등 룰 연구 표본 (out/crash — 거래대금 10억·시총 1,000억 필터 전 종목)
        N = 학습 구간 거래일 중 그 수 이상인 날이 2% 이하가 되는 가장 작은 정수 (건수 분포만 보고 정함)
  (열)  발열계 열량 꺾임: 열량 = 발열계 heat_pct 정의(종목별 YZ20 의 자기 직전 249일 대비 백분위의 중앙값,
        종목 = 발열계 유니버스 panel_states.csv), 가격 data_1104 로 2015~ 재구성
        최근 5거래일 열량 최고 ≥ 80 AND 오늘 열량 ≤ 그 최고 − 5 AND 시장 브레드스 EMA10 > 5거래일 전 값
  사건  신호가 켜진 날. 직전 사건 첫날 뒤 20거래일 안에 다시 켜지면 같은 사건 (첫날만 씀)

평가 (2-c 사건 기준과 같은 결)
  진입 = 신호 다음 날 코스피 200 시가. 수익 h = 신호일 h거래일 뒤 종가 ÷ 진입가 − 1 (h = 5·10·20)
  늦음 = 진입가 ÷ 직전 60거래일 최저 종가 L − 1      헛 진입 = 진입 뒤 20거래일 안에 종가가 L 아래로 간 비율
  기준 = 학습 구간 모든 거래일에 같은 방식으로 들어갔을 때. 연도별로는 그해 모든 날의 평균
통과 기준 (미리 고정)
  ① 사건 ≥ 20  ② 20일 수익 평균 > 기준 평균 AND 사건 있는 연도의 2/3 이상에서 (사건 평균 − 그해 기준 평균) > 0
  ③ 헛 진입 비율 ≤ 기준 비율
⚠ 온도·냉각은 재지 않는다 — 에어에 학습 구간 안 온도 자료가 2024-01~ 뿐이고, 냉각(W60 − W100)의 W100 이력이 없다.
사용   python3 regime_expand.py
산출   out/regime/expand_events.csv · expand_summary.csv · expand_years.csv · expand_heat.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; STOLAB = HERE.parents[1]
OUT = HERE / "out" / "regime"; CRASH = HERE / "out" / "crash"
SPOT = STOLAB / "SpotGauge"; LLV = STOLAB / "longlivevault" / "data" / "ohlcv"
A, B = pd.Timestamp("2016-07-01"), pd.Timestamp("2026-05-29")
COOL, HZ, LOWWIN, FWD = 20, (5, 10, 20), 60, 20
BR_STATE = "브레드스 EMA20 − 250일 중앙 ±0.05"


# ── 기준선 ────────────────────────────────────────────────────────────────
def load_base():
    lv = pd.read_parquet(OUT / "layer_levels.parquet"); k = lv["코스피 200"].astype(float)
    d = pd.read_parquet(HERE / "data_1104" / "kospi200_daily.parquet")[["Open", "Close"]].astype(float)
    d.index = pd.to_datetime(d.index); d = d.reindex(k.index)
    gap = (d.Close - k).abs().max()
    print(f"[점검] 코스피 200 종가 — layer_levels 와 kospi200_daily 최대 차이 {gap:.4f}, 시가 결측 {int(d.Open.isna().sum())}일")
    return k, d.Open, lv["유니버스 동일가중"].astype(float)

def breadth_state(k):
    S = pd.read_parquet(OUT / "detect_states.parquet")
    return S[BR_STATE].reindex(k.index).ffill().fillna(True).astype(bool)

def breadth_ema10(k):
    b = pd.read_csv(HERE / "out" / "breadth" / "breadth_market.csv", parse_dates=["date"]).set_index("date")
    return b["ema10"].reindex(k.index)

# ── 신호 ──────────────────────────────────────────────────────────────────
def sig_pullback(k, bs):
    up = k > k.rolling(200).mean()
    pull = ((k / k.rolling(20, min_periods=1).max() - 1) <= -0.05).astype(int).rolling(20, min_periods=1).max() > 0
    flip = bs & ~bs.shift(1, fill_value=True)
    return up & pull & flip

def gap_shock(k, o):
    cp = k.shift(1); r = k / cp - 1; g = o / cp - 1
    share = np.log1p(g) / np.log1p(r)
    return ((r <= -0.03) & (share >= 0.4)).fillna(False)

def sig_recover(k, shock, win=60, up=0.10):
    kv, sh = k.to_numpy(float), shock.to_numpy(bool); out = np.zeros(len(kv), bool)
    active, last, L = False, -10 ** 9, np.inf
    for i in range(len(kv)):
        if sh[i]:
            if not active: L = kv[i]
            active, last = True, i
        if not active: continue
        if i - last > win: active, L = False, np.inf; continue
        L = min(L, kv[i])
        if kv[i] >= L * (1 + up): out[i] = True; active, L = False, np.inf
    return pd.Series(out, index=k.index)

def bounce_counts(k):
    sb = pd.read_parquet(CRASH / "sim_bounce.parquet", columns=["ticker", "date", "B_sig"])
    de = pd.read_parquet(CRASH / "drive_end.parquet", columns=["ticker", "date", "tot", "n"])
    m = sb.merge(de, on=["ticker", "date"], how="inner")
    print(f"[점검] 반등 룰 표본 {len(m):,}건 · 드라이브 길이 최소 {int(m.n.min())}일 · 기간 {m.date.min().date()}~{m.date.max().date()}")
    s = m[(m.tot <= -0.20) & (m.B_sig >= 1.65)]
    cal = k.index; pos = cal.searchsorted(pd.to_datetime(s.date), side="right")   # 드라이브 마지막 하락일 → 다음 거래일 = 신호일
    d1 = cal[pos[pos < len(cal)]]
    return pd.Series(d1).value_counts().reindex(cal, fill_value=0), len(s)

def pick_n(cnt, w, share=0.02):
    c = cnt[w]
    for n in range(1, int(c.max()) + 2):
        if (c >= n).mean() <= share: return n

# ── 열량 재구성 (발열계 heat_at 와 같은 정의) ──────────────────────────────
def yz20_ann(px, n=20):
    """LLV indicator_calculator 와 같은 Yang–Zhang (연율 %). px: ticker·date 정렬, open/high/low/close."""
    kk = 0.34 / (1.34 + (n + 1) / (n - 1)); g = px.groupby("ticker", sort=False)
    o = np.log(px.open / g.close.shift(1)); c = np.log(px.close / px.open)
    u = np.log(px.high / px.open); d = np.log(px.low / px.open); rs = u * (u - c) + d * (d - c)
    t = px.ticker
    vo = o.groupby(t).rolling(n, min_periods=n).var(ddof=1).reset_index(level=0, drop=True)
    vc = c.groupby(t).rolling(n, min_periods=n).var(ddof=1).reset_index(level=0, drop=True)
    vr = rs.groupby(t).rolling(n, min_periods=n).mean().reset_index(level=0, drop=True)
    return np.sqrt(vo + kk * vc + (1 - kk) * vr) * np.sqrt(252) * 100

def heat_series(Y, lookback=250, minobs=200):
    """Y: 날짜 × 종목 YZ20. 날짜마다 종목별 '오늘 값의 자기 직전 249일 대비 백분위' → 중앙값."""
    a = Y.to_numpy(float); out = np.full(len(a), np.nan)
    for t in range(len(a)):
        win = a[max(0, t - lookback + 1):t + 1]; cur, prev = win[-1], win[:-1]
        ok = np.isfinite(cur) & (np.isfinite(win).sum(0) >= minobs)
        if not ok.any(): continue
        below = (prev[:, ok] < cur[ok]).sum(0); fin = np.maximum(np.isfinite(prev[:, ok]).sum(0), 1)
        out[t] = np.median(below / fin * 100)
    return pd.Series(out, index=Y.index)

def build_heat(k):
    uni = sorted(pd.read_csv(SPOT / "data" / "panel_states.csv", dtype={"ticker": str}).ticker.unique())
    px = pd.read_parquet(HERE / "data_1104" / "prices.parquet").reset_index()
    px = px[px.ticker.isin(uni)].copy(); px["date"] = pd.to_datetime(px.date)
    px = px[(px.open > 0) & (px.high > 0) & (px.low > 0) & (px.close > 0)]          # LLV 처럼 0가격 행 제외
    px = px.sort_values(["ticker", "date"]).reset_index(drop=True); px["yz"] = yz20_ann(px)
    Y = px.pivot(index="date", columns="ticker", values="yz").reindex(k.index)
    H = heat_series(Y)
    # 검증: 같은 정의를 LLV 정본 YZ_20_Ann 으로 계산해 겹치는 구간에서 비교
    f = [pd.read_parquet(LLV / x, columns=["Date", "Ticker", "YZ_20_Ann"]) for x in ("core.parquet", "extend.parquet")]
    L = pd.concat(f).drop_duplicates(["Date", "Ticker"]); L = L[L.Ticker.isin(uni)]
    YL = L.pivot(index="Date", columns="Ticker", values="YZ_20_Ann"); YL.index = pd.to_datetime(YL.index)
    HL = heat_series(YL.reindex(k.index))
    both = pd.concat({"재구성": H, "LLV": HL}, axis=1).dropna()
    yzc = pd.concat({"r": Y.stack(), "l": YL.reindex(k.index).stack()}, axis=1).dropna()
    rel = (yzc.r / yzc.l - 1).abs()
    print(f"[점검] 열량 종목 {len(uni)} · 재구성 가격 종목 {px.ticker.nunique()} · 열량 유효 {H.first_valid_index().date()}~")
    print(f"[점검] YZ20 재구성 vs LLV: 겹침 {len(yzc):,}행, 상대오차 중앙 {rel.median()*100:.2f}% · 95% {rel.quantile(.95)*100:.2f}%")
    print(f"[점검] 열량 재구성 vs LLV: {both.index.min().date()}~{both.index.max().date()} {len(both)}일, "
          f"상관 {both.corr().iloc[0,1]:.3f}, 절대차 중앙 {(both.재구성-both.LLV).abs().median():.2f} · 최대 {(both.재구성-both.LLV).abs().max():.2f}")
    pd.concat({"heat_recon": H, "heat_llv": HL}, axis=1).to_csv(OUT / "expand_heat.csv", index_label="date", encoding="utf-8-sig")
    return H

# ── 사건 평가 ──────────────────────────────────────────────────────────────
def events_of(sig, elig):
    ev = []
    for i in np.flatnonzero(sig.to_numpy(bool)):
        if ev and i - ev[-1] <= COOL: continue
        ev.append(i)                                   # 쿨다운은 학습 창 밖 신호까지 포함해 센다
    return [i for i in ev if elig[i]]

def measure(i, kv, ov, uv):
    e = ov[i + 1]; L = kv[max(0, i - LOWWIN + 1):i + 1].min()
    r = {f"수익{h}": (kv[i + h] / e - 1) * 100 for h in HZ}
    r.update(늦음=(e / L - 1) * 100, 헛진입=bool((kv[i + 1:i + 1 + FWD] < L).any()), 유니버스20=(uv[i + FWD] / uv[i] - 1) * 100)
    return r

def main():
    k, o, u = load_base(); cal = k.index; kv, ov, uv = k.to_numpy(float), o.to_numpy(float), u.to_numpy(float)
    ib = int(np.searchsorted(cal, B, side="right")) - 1
    w = (cal >= A) & (cal <= B)
    elig = w & (np.arange(len(cal)) + FWD <= ib) & np.isfinite(np.r_[ov[1:], np.nan])
    bs = breadth_state(k); be = breadth_ema10(k)
    cnt, nsig = bounce_counts(k); N = pick_n(cnt, w)
    d = cnt[w]; print(f"[점검] 반등 룰 신호 {nsig:,}건 · 학습 구간 하루 신호 종목 수: 0인 날 {(d==0).mean()*100:.1f}% · "
                      f"1~4 {((d>=1)&(d<5)).mean()*100:.1f}% · 5~ {(d>=5).mean()*100:.1f}% · 최대 {int(d.max())} → N = {N} (그 이상인 날 {(d>=N).mean()*100:.2f}%)")
    H = build_heat(k); h5 = H.rolling(5, min_periods=1).max()
    print(f"[점검] 학습 구간 열량 ≥80 인 날 {(H[w] >= 80).mean()*100:.1f}% · 분위 10/50/90 = {H[w].quantile(.1):.0f}/{H[w].quantile(.5):.0f}/{H[w].quantile(.9):.0f}")
    shock = gap_shock(k, o)
    SIG = {"(가) 시장 눌림목": sig_pullback(k, bs),
           "(나1) 투매 뒤 +10% 복귀": sig_recover(k, shock),
           f"(나2) 반등 룰 동시 다발 ≥{N}": cnt >= N,
           "(열) 열량 꺾임": (h5 >= 80) & (H <= h5 - 5) & (be > be.shift(5))}
    base = pd.DataFrame([dict(i=i, 연도=cal[i].year, **measure(i, kv, ov, uv)) for i in np.flatnonzero(elig)])
    by = base.groupby("연도").수익20.mean()
    rows, ev_rows, yr_rows = [], [], []
    for nm, s in SIG.items():
        s = s.fillna(False).astype(bool); ev = events_of(s, elig)
        E = pd.DataFrame([dict(신호=nm, 날짜=cal[i].date(), i=i, 연도=cal[i].year, 코스피200=kv[i], **measure(i, kv, ov, uv)) for i in ev])
        if E.empty: rows.append({"신호": nm, "사건": 0}); continue
        ev_rows.append(E)
        yd = (E.groupby("연도").수익20.mean() - by.reindex(E.연도.unique())).rename("사건−그해기준")
        yr_rows.append(pd.DataFrame({"신호": nm, "사건수": E.groupby("연도").size(), "사건 20일": E.groupby("연도").수익20.mean(),
                                     "그해 기준 20일": by.reindex(E.연도.unique()), "차이": yd}))
        r = {"신호": nm, "사건": len(E), "신호일 수(학습)": int((s & pd.Series(w, index=cal)).sum())}
        for h in HZ: r[f"수익{h} 평균"] = E[f"수익{h}"].mean(); r[f"기준{h} 평균"] = base[f"수익{h}"].mean()
        r.update({"수익20 중앙": E.수익20.median(), "기준20 중앙": base.수익20.median(), "20일 + 비율": (E.수익20 > 0).mean() * 100,
                  "기준 + 비율": (base.수익20 > 0).mean() * 100, "늦음 중앙": E.늦음.median(), "헛진입%": E.헛진입.mean() * 100,
                  "기준 헛진입%": base.헛진입.mean() * 100, "유니버스20 평균": E.유니버스20.mean(), "기준 유니버스20": base.유니버스20.mean(),
                  "연도 + 비율": (yd > 0).mean() * 100, "연도 수": len(yd)})
        r["①사건≥20"] = len(E) >= 20; r["②수익"] = (r["수익20 평균"] > r["기준20 평균"]) and (yd > 0).mean() >= 2 / 3
        r["③헛진입"] = r["헛진입%"] <= r["기준 헛진입%"]; r["통과"] = r["①사건≥20"] and r["②수익"] and r["③헛진입"]
        rows.append(r)
    T = pd.DataFrame(rows).set_index("신호"); EV = pd.concat(ev_rows); YR = pd.concat(yr_rows)
    pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40)
    print(f"\n기준(학습 구간 모든 거래일 {len(base)}일 진입)  20일 평균 {base.수익20.mean():.2f}% · 헛진입 {base.헛진입.mean()*100:.1f}%")
    print("\n== 요약 ==\n", T.T.to_string(float_format=lambda x: f"{x:.2f}"))
    print("\n== 연도별 (20일 수익, %) ==\n", YR.round(2).to_string())
    print("\n== 사건 목록 ==\n", EV.drop(columns=["i"]).round(2).to_string(index=False))
    T.to_csv(OUT / "expand_summary.csv", encoding="utf-8-sig"); EV.to_csv(OUT / "expand_events.csv", index=False, encoding="utf-8-sig"); YR.to_csv(OUT / "expand_years.csv", encoding="utf-8-sig")
    try:
        import regime_xlsx; print("엑셀", regime_xlsx.export_expand(pd.Timestamp.now().strftime("%Y%m%d")))
    except Exception as e: print("⚠ 엑셀 저장 실패:", e)

if __name__ == "__main__":
    main()
