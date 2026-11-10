"""반등 룰 #5 × 시장 국면 (2026-11-10 케인 — 유형 규칙에 국면을 통제 변수로 넣는 첫 점검).

반등 룰 정의 = MagicFormula bounce_rule/config.py v1.1.1.1 그대로
  드라이브 = 종가 연속 하락 ≥ 3일, D+1 = 처음 r ≥ 0 인 날 · 강도 = r(D+1) ÷ σ60(드라이브 끝까지 60일, ddof=1)
  신호 = 드라이브 낙폭 ≤ −20% AND 강도 ≥ 1.65σ · 데드캣 = 강도 < 0.3σ
결과 (진입 = D+2 시가, 왕복 0.3% 차감)
  지킴10 = D+2..D+10 최저 종가 ≥ 드라이브 끝 종가 (DRIVE_STUDY hold10 과 같은 정의)
  r5 = D+6 종가 / D+2 시가 − 1 − 0.3%  (5거래일 보유) · r10 = D+11 종가 / D+2 시가 − 1 − 0.3%
  규칙 = 5거래일 보유, 그 사이 종가가 드라이브 끝 종가 아래면 다음 날 시가 정리
국면 = 신호일(D+1) 종가 기준 라벨 (그날 저녁 판단 → 다음 날 시가 진입이라 미래 정보 없음). lag 1 = 드라이브 끝 날 라벨.
유니버스 = 연구와 같음 (전년 거래대금 ≥10억 & 전년말 시총 ≥1,000억, 2016~) — 운영 룰은 LLV 패널(약 200종목)이라 더 좁다.
구간 = 학습: D+11 ≤ 2026-05-29 / 보류: D+1 ≥ 2026-06-01 (--holdout, 검증 때만). 가격 = drive_regime.ext_prices (09-11~ LLV 원본).
신뢰구간 = 신호일 단위 부트스트랩 (같은 날 신호는 같은 시장을 맞는다).
출력 out/regime/bounce_regime{_holdout}.csv · bounce_events{_holdout}.parquet
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from crash_study import eligibility                              # noqa: E402
from drive_types import add_daily_fields                         # noqa: E402
from drive_regime import ext_prices, LEARN_END, HOLD_START, ROUT  # noqa: E402
from bounce_rule.config import DEFAULT as BC                     # noqa: E402
from market_layer import attach_regime, load_history             # noqa: E402

COST = 0.003
NB = 1000
RNG = np.random.default_rng(20261110)


def build_events(px: pd.DataFrame) -> pd.DataFrame:
    px = add_daily_fields(px, -1)                                 # r1, run(연속 하락일), …
    g = px.groupby("ticker", sort=False, observed=True)
    px["sig60"] = g.r1.transform(lambda s: s.rolling(BC.sigma_window, min_periods=BC.sigma_window).std())
    nxt_r1 = g.r1.shift(-1)
    end = (px.run >= BC.min_drive_days) & (nxt_r1 >= 0)          # 드라이브 마지막 하락일(Dn), 다음 날이 D+1
    sh = lambda col, k: g[col].shift(-k)
    ev = px.loc[end, ["ticker", "date", "close", "run", "sig60"]].copy()
    ev = ev.rename(columns={"date": "date_dn", "close": "low_close", "run": "n"})
    ev["date_d1"] = sh("date", 1)[end].values
    # 누적 낙폭 = Dn 종가 / 드라이브 시작 전 종가 − 1 (run 일 전)
    c = px.close.to_numpy(float); idx = np.flatnonzero(end.to_numpy()); n = px.run.to_numpy()[idx].astype(int)
    ev["drop"] = c[idx] / c[idx - n] - 1
    ev["d1_ret"] = nxt_r1[end].values
    ev["strength"] = ev.d1_ret / ev.sig60
    ev["open_e"] = sh("open", 2)[end].values
    ev["c6"] = sh("close", 6)[end].values; ev["c11"] = sh("close", 11)[end].values
    ev["date_d11"] = sh("date", 11)[end].values
    mins = pd.concat([sh("close", k)[end] for k in range(2, 11)], axis=1)
    ev["hold10"] = (mins.min(axis=1, skipna=False) >= ev.low_close).where(mins.notna().all(axis=1))
    # 규칙 수익: D+2 시가 진입, D+2..D+6 중 종가 < 저점이면 다음 날 시가 정리, 아니면 D+6 종가
    cl = np.column_stack([sh("close", k)[end].values for k in range(2, 7)])
    op = np.column_stack([sh("open", k)[end].values for k in range(3, 8)])
    brk = cl < ev.low_close.to_numpy()[:, None]
    first = np.where(brk.any(axis=1), brk.argmax(axis=1), -1)
    exit_px = np.where(first >= 0, op[np.arange(len(first)), np.clip(first, 0, 4)], cl[:, 4])
    ev["r_rule"] = exit_px / ev.open_e - 1 - COST
    ev["stopped"] = first >= 0
    ev["r5"] = ev.c6 / ev.open_e - 1 - COST; ev["r10"] = ev.c11 / ev.open_e - 1 - COST
    bad = ~(ev.open_e > 0)                                          # 시가 0(거래정지 등) — 수익 계산 불가
    ev.loc[bad, ["r5", "r10", "r_rule"]] = np.nan; ev.loc[bad, "stopped"] = np.nan
    ev["signal"] = (ev["drop"] <= BC.min_drop) & (ev.strength >= BC.strong_sigma)
    ev["deadcat"] = ev.strength < BC.deadcat_sigma
    ev["strong"] = ev.strength >= BC.strong_sigma
    ev["year"] = pd.to_datetime(ev.date_d1).dt.year
    ev["ticker"] = ev.ticker.astype(str)
    el = eligibility(px[["ticker", "close", "volume"]].assign(ticker=px.ticker.astype(str), year=px.date.dt.year.astype("int16")))
    ev = ev.merge(el[["ticker", "year", "eligible", "cap_bucket"]], on=["ticker", "year"], how="left")
    ev = ev[ev.eligible.fillna(False) & (ev.year >= 2016) & ev.sig60.notna() & (ev.sig60 > 0) & ev.date_d1.notna()]
    return ev.drop(columns=["eligible"]).reset_index(drop=True)


METRICS = {"지킴10%": ("hold10", "rate"), "r5평균": ("r5", "mean"), "r10평균": ("r10", "mean"),
           "r5>0%": ("r5", "pos"), "규칙평균": ("r_rule", "mean"), "규칙>0%": ("r_rule", "pos"), "저점이탈정리%": ("stopped", "rate")}


def _nd(d: pd.DataFrame, col: str, kind: str):
    x = d[col].astype(float)
    if kind == "pos":
        return (x > 0).astype(float).where(x.notna(), 0.0), x.notna().astype(float)
    return x.fillna(0.0), x.notna().astype(float)


def stats(d: pd.DataFrame, cluster="date_d1"):
    cols = {}
    for m, (c, k) in METRICS.items():
        a, b = _nd(d, c, k); cols[m + "_n"], cols[m + "_d"] = a.to_numpy(), b.to_numpy()
    day = pd.DataFrame(cols).groupby(d[cluster].to_numpy()).sum()
    tot = day.sum()
    pt = {"건수": len(d), "신호일": day.shape[0], **{m: tot[m + "_n"] / tot[m + "_d"] for m in METRICS}}
    a = day.to_numpy(); cl = list(day.columns); idx = RNG.integers(0, len(a), size=(NB, len(a))); s = a[idx].sum(axis=1)
    bs = {m: s[:, cl.index(m + "_n")] / s[:, cl.index(m + "_d")] for m in METRICS}
    return pt, bs


AX = {"breadth": {True: "상승", False: "하락"}, "c2": {True: "보유", False: "퇴출"},
      "gate_window": {True: "갭쇼크창 안", False: "갭쇼크창 밖"}}
SETS = {"신호(낙폭≤−20% & 강도≥1.65σ)": lambda e: e[e.signal],
        "강한 반등 전체(강도≥1.65σ)": lambda e: e[e.strong],
        "깊은 낙폭 전체(≤−20%)": lambda e: e[e["drop"] <= BC.min_drop],
        "깊은 낙폭 & 데드캣(<0.3σ)": lambda e: e[(e["drop"] <= BC.min_drop) & e.deadcat]}


def table(ev: pd.DataFrame, periods: dict, min_n: int = 20) -> pd.DataFrame:
    out = []
    for lag in (0, 1):
        e = attach_regime(ev, "date_d1" if lag == 0 else "date_dn", cols=list(AX), hist=HIST, prefix="")
        for per, (a, b) in periods.items():
            ep = e[(e.date_d1 >= a) & (e.date_d1 <= b)]
            for sname, f in SETS.items():
                es = f(ep)
                for ax, labs in AX.items():
                    rows, bss = {}, {}
                    for v, lab in labs.items():
                        g = es[es[ax] == v]
                        if len(g) >= min_n:
                            rows[lab], bss[lab] = stats(g)
                    t = pd.DataFrame(rows).T
                    if len(bss) == 2:
                        x, y = list(bss)
                        t.loc[f"차({x}−{y})"] = [np.nan, np.nan] + [t.loc[x, m] - t.loc[y, m] for m in METRICS]
                        t.loc["CI하"] = [np.nan, np.nan] + [np.percentile(bss[x][m] - bss[y][m], 2.5) for m in METRICS]
                        t.loc["CI상"] = [np.nan, np.nan] + [np.percentile(bss[x][m] - bss[y][m], 97.5) for m in METRICS]
                    if len(es) >= min_n:
                        pt, _ = stats(es); t.loc["(묶음 전체)"] = [pt[k] for k in t.columns] if len(t.columns) else None
                    out.append(t.assign(lag=lag, 기간=per, 묶음=sname, 축=ax).reset_index(names="그룹"))
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    t0 = time.time()
    HIST = load_history()
    holdout = "--holdout" in sys.argv
    p = ROUT / "bounce_events_all.parquet"
    if "--rebuild" in sys.argv or not p.exists():
        ev = build_events(ext_prices()); ROUT.mkdir(parents=True, exist_ok=True); ev.to_parquet(p)
    ev = pd.read_parquet(p)
    ev["date_d1"] = pd.to_datetime(ev.date_d1); ev["date_dn"] = pd.to_datetime(ev.date_dn); ev["date_d11"] = pd.to_datetime(ev.date_d11)
    if holdout:
        print("⚠ 보류 구간(2026-06~) — 검증 때만"); ev = ev[(ev.date_d1 >= HOLD_START) & ev.date_d11.notna()]
        periods = {"보류": ("2026-06-01", "2026-12-31")}
    else:
        ev = ev[ev.date_d11.notna() & (ev.date_d11 <= LEARN_END)]
        periods = {"전체": ("2016-01-01", "2026-05-29"), "2016~2020": ("2016-01-01", "2020-12-31"), "2021~2026.05": ("2021-01-01", "2026-05-29")}
    print(f"드라이브 끝 {len(ev):,} · 신호 {int(ev.signal.sum()):,} · D+1 {ev.date_d1.min():%Y-%m-%d}~{ev.date_d1.max():%Y-%m-%d}")
    T = table(ev, periods)
    tag = "_holdout" if holdout else ""
    T.to_csv(ROUT / f"bounce_regime{tag}.csv", index=False)
    print(f"저장 {ROUT / f'bounce_regime{tag}.csv'} · {time.time() - t0:.0f}s")
