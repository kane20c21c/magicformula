"""드라이브 × 시장 국면 비교 (2026-11-10 케인 — 통일장 국면 라벨 표를 D1 에 붙여 비교).

입력
  드라이브  상승형 out/crash/drive_len_up.parquet (drive_length.build(+1))
           하락형 out/crash/drive_len_down.parquet (drive_length.build(-1) — 없으면 만든다, --rebuild 로 다시)
           정의·유니버스는 drive_types·crash_study 그대로 (전년 거래대금 ≥10억 & 전년말 시총 ≥1,000억, 2016~)
  국면     MagicFormula output/market_layer/regime_history.parquet (market_layer.attach_regime) — **D1 의 라벨**
           (lag=1 = D0 라벨, 확인용)

구간 (케인 11-10)
  학습  드라이브 끝 + 10거래일(사후 관측 창)이 2026-05-29 안에 들어오는 것만. 이 파일의 모든 결과는 학습 구간.
  보류  2026-06-01~ 는 검증용으로 남긴다 — 이 스크립트는 기본으로 보지 않는다 (--holdout 은 검증 때만).

비교 두 묶음
  모양  길이 P(k+1|k)·평균 길이 · 속도 A · 거래량 T · 거래량 수준 · 겹침 ō · 누적·D1 등락폭 · 가속/증가/계단 비중
        신뢰구간 = D1 날짜 단위 부트스트랩
  사후  D+1 (하락형 반등 / 상승형 연장) · D+2..D+10 저점 지킴(하락형) / 고점 재돌파(상승형) · D+10 수익률 r10
        신뢰구간 = 드라이브 끝 날짜 단위 부트스트랩 (같은 날 끝난 드라이브는 같은 시장 앞날을 맞는다)
        묶음 = 전체 · 큰 드라이브(누적 ±12% 이상) · 깊은 하락 & D+1 ≥ +3% (DRIVE_STUDY §3.2 조건)

규칙
  · 드라이브 '건수' 비교는 하지 않는다 — 브레드스가 곧 상승·하락 종목 수라 순환
  · 국면 라벨은 날짜마다 하나 → 국면 그룹이 날짜를 나누므로 그룹별로 따로 뽑는다
  · 길이 n·사후값은 설명 용도. 신호로 쓰려면 보류 구간 검증이 먼저다
출력: out/regime/drive_regime_shape.csv · drive_regime_outcome.csv · drive_regime_type.csv
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from crash_study import OUT                                     # noqa: E402
from market_layer import attach_regime, load_history            # noqa: E402

LEARN_END = pd.Timestamp("2026-05-29")      # 학습 끝 (보류 = 2026-06-01~, 케인 11-10)
HOLD_START = pd.Timestamp("2026-06-01")
PX_CUT = pd.Timestamp("2026-09-14")         # prices.parquet 종가가 이 날부터 정규장과 어긋남 (breadth_study 주석)
POST = 10
ROUT = HERE / "out" / "regime"
NB = 1000
RNG = np.random.default_rng(20261110)
DEEP = 0.12
AXES = {"breadth": {True: "상승", False: "하락"}, "c2": {True: "보유", False: "퇴출"},
        "regime": {x: x for x in ["상승·보유", "상승·퇴출", "하락·보유", "하락·퇴출"]},
        "gate_window": {True: "갭쇼크창 안", False: "갭쇼크창 밖"}}
PERIODS = {"전체": ("2016-01-01", "2026-05-29"), "2016~2020": ("2016-01-01", "2020-12-31"),
           "2021~2026.05": ("2021-01-01", "2026-05-29")}


SEAM = pd.Timestamp("2026-09-11")          # prices.parquet 을 여기까지 쓰고 이 날 행부터 LLV KRX 원본으로 (breadth_study PX_END 와 같음)
LLV_RAW = HERE.parents[1] / "LongLiveVault" / "data" / "raw"


def ext_prices() -> pd.DataFrame:
    """prices.parquet(~09-10) + LLV KRX 원본(09-11~, 정규장 종가·원주가) — 보류 구간 검증용. 이음매 대조를 출력한다."""
    import glob
    from drive_types import load_ohlcv
    from crash_study import DATA
    px = load_ohlcv(DATA / "prices.parquet"); px["ticker"] = px.ticker.astype(str)
    fs = sorted(f for m in ("kospi", "kosdaq") for f in glob.glob(str(LLV_RAW / f"krx_{m}_*.parquet"))
                if f[-16:-8] >= "20260801")
    r = pd.concat([pd.read_parquet(f, columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]) for f in fs])
    r.columns = ["date", "ticker", "open", "high", "low", "close", "volume"]
    r["date"] = pd.to_datetime(r.date); r["ticker"] = r.ticker.astype(str); r = r[r.close > 0]
    # 이음매 대조 — 겹치는 08월~09-10 에서 prices 와 원본이 같은지 (자본변동 종목 빼면 같아야 한다)
    j = px[(px.date >= "2026-08-01") & (px.date < SEAM)].merge(r, on=["ticker", "date"], suffixes=("", "_r"))
    print(f"이음매 대조 (08-01~09-10, {len(j):,}행): 종가 일치 {(j.close.round() == j.close_r).mean():.1%} · "
          f"거래량 일치 {(j.volume == j.volume_r).mean():.1%}")
    days = sorted(r.date.unique()); print(f"원본 {pd.Timestamp(days[0]):%Y-%m-%d}~{pd.Timestamp(days[-1]):%Y-%m-%d} {len(days)}일")
    tail = r[r.date >= SEAM]
    out = pd.concat([px[px.date < SEAM][["ticker", "date", "open", "high", "low", "close", "volume"]], tail], ignore_index=True)
    out["ticker"] = out.ticker.astype("category")
    for c in ("open", "high", "low", "close"):
        out[c] = out[c].astype("float32")
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_drives(hist: pd.DataFrame, rebuild: bool = False, holdout: bool = False) -> dict[str, pd.DataFrame]:
    if holdout:
        pu, pd_ = OUT / "drive_len_up_ext.parquet", OUT / "drive_len_down_ext.parquet"
        if rebuild or not (pu.exists() and pd_.exists()):
            import drive_length
            px = ext_prices()
            drive_length.build(+1, px).to_parquet(pu); drive_length.build(-1, px).to_parquet(pd_)
        up, dn = pd.read_parquet(pu), pd.read_parquet(pd_)
    else:
        up = pd.read_parquet(OUT / "drive_len_up.parquet")
        p = OUT / "drive_len_down.parquet"
        if rebuild or not p.exists():
            import drive_length
            drive_length.build(-1).to_parquet(p)
        dn = pd.read_parquet(p)
    days = hist.index
    out = {}
    for name, d in (("상승형", up), ("하락형", dn)):
        d = d.copy()
        d["date"] = pd.to_datetime(d.date); d["date_end"] = pd.to_datetime(d.date_end)
        pos = days.searchsorted(d.date_end.values)
        ok = pos + POST < len(days)
        post_end = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
        post_end[ok] = days[pos[ok] + POST]
        d["post_end"] = post_end
        if holdout:
            last = d.date_end.max()                       # 가격 데이터 끝 — 사후 창이 그 안에 들어와야 한다
            d = d[(d.date >= HOLD_START) & d.post_end.notna() & (d.post_end <= last)]
        else:
            d = d[d.post_end.notna() & (d.post_end <= LEARN_END)]
        out[name] = d
    return out


# ── 지표 정의: (이름, 함수(d, 방향) → (분자, 분모), 종류) ────────────────────────────────
def _rate(mask, ok=None):
    ok = pd.Series(True, index=mask.index) if ok is None else ok
    return (mask & ok).astype(float), ok.astype(float)


def _mean(x):
    return x.fillna(0.0), x.notna().astype(float)


def _geo(x):
    return _mean(np.log(x.where(x > 0)))


SHAPE = [("P4|3", lambda d, s: _rate(d.n >= 4), "rate"),
         ("P5|4", lambda d, s: _rate(d.n >= 5, d.n >= 4), "rate"),
         ("P6|5", lambda d, s: _rate(d.n >= 6, d.n >= 5), "rate"),
         ("평균길이", lambda d, s: _mean(d.n.astype(float)), "mean"),
         ("속도A", lambda d, s: _geo(d.speed), "geo"),
         ("거래량T", lambda d, s: _geo(d.v_trend), "geo"),
         ("거래량수준", lambda d, s: _geo(d.v_level), "geo"),
         ("겹침ō", lambda d, s: _mean(d.overlap), "mean"),
         ("누적|등락|", lambda d, s: _mean(d.tot.abs()), "pct"),
         ("가속%", lambda d, s: _rate(d.speed_c.astype(str) == "가속", d.speed_c.notna()), "rate"),
         ("증가%", lambda d, s: _rate(d.v_trend_c.astype(str) == "증가", d.v_trend_c.notna()), "rate"),
         ("계단%", lambda d, s: _rate(d.overlap_c.astype(str) == "계단", d.overlap_c.notna()), "rate")]
# D+1 은 끝난 드라이브만 모았으므로 방향이 늘 반대(또는 0) — 크기만 본다 (하락형 = 반등 폭, 상승형 = 되밀림 폭)
OUTCOME = [("D+1평균", lambda d, s: _mean(d.next_r1), "pct"),
           ("지킴/재돌파%", lambda d, s: _rate(d.rebreak.astype(bool)), "rate"),
           ("r10평균", lambda d, s: _mean(d.r10), "pct"),
           ("r10>0%", lambda d, s: _rate(d.r10 > 0, d.r10.notna()), "rate")]
# 지킴/재돌파 — drive_length 의 rebreak: 하락형 = D+2..10 최저 종가 > 끝 종가(저점 지킴) / 상승형 = 최고 종가 > 끝 종가(고점 재돌파)


def per_drive(d: pd.DataFrame, s: int, metrics) -> pd.DataFrame:
    cols = {}
    for nm, f, _ in metrics:
        a, b = f(d, s)
        cols[f"{nm}_n"], cols[f"{nm}_d"] = a.to_numpy(float), b.to_numpy(float)
    return pd.DataFrame(cols, index=d.index)


def finish(kind, num, den):
    v = num / den
    return np.exp(v) if kind == "geo" else v


def stats(d: pd.DataFrame, s: int, metrics, cluster: str) -> tuple[dict, dict]:
    day = per_drive(d, s, metrics).groupby(d[cluster].values).sum()
    tot = day.sum()
    point = {"드라이브": len(d), "군집": day.shape[0]}
    for nm, _, k in metrics:
        point[nm] = finish(k, tot[f"{nm}_n"], tot[f"{nm}_d"])
    a = day.to_numpy(); cols = list(day.columns)
    idx = RNG.integers(0, len(a), size=(NB, len(a)))
    sm = a[idx].sum(axis=1)
    bs = {nm: finish(k, sm[:, cols.index(f"{nm}_n")], sm[:, cols.index(f"{nm}_d")]) for nm, _, k in metrics}
    return point, bs


def compare(d: pd.DataFrame, s: int, axis: str, metrics, cluster: str, min_n: int = 30) -> pd.DataFrame:
    labels = AXES[axis]
    dd = d.dropna(subset=[axis])
    rows, bs = {}, {}
    for v, lab in labels.items():
        g = dd[dd[axis] == v]
        if len(g) < min_n:
            continue
        rows[lab], bs[lab] = stats(g, s, metrics, cluster)
    t = pd.DataFrame(rows).T
    labs = [x for x in labels.values() if x in bs]
    if len(labels) == 2 and len(labs) == 2:
        a, b = labs
        names = [m[0] for m in metrics]
        t.loc[f"차({a}−{b})"] = [np.nan, np.nan] + [t.loc[a, m] - t.loc[b, m] for m in names]
        t.loc["CI하"] = [np.nan, np.nan] + [np.percentile(bs[a][m] - bs[b][m], 2.5) for m in names]
        t.loc["CI상"] = [np.nan, np.nan] + [np.percentile(bs[a][m] - bs[b][m], 97.5) for m in names]
    return t


SUBSETS = {"전체": lambda d, s: d,
           "큰 드라이브(±12%↑)": lambda d, s: d[d.tot * s >= DEEP],
           "깊은 하락 & D+1≥3%": lambda d, s: d[(d.tot <= -DEEP) & (d.next_r1 >= 0.03)] if s < 0 else d.iloc[0:0]}


def run(drives: dict, hist: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shape, outc, typ = [], [], []
    for lag in (0, 1):
        for dname, d0 in drives.items():
            s = +1 if dname == "상승형" else -1
            d = attach_regime(d0, "date", cols=list(AXES), hist=hist, lag=lag, prefix="")
            for per, (a, b) in PERIODS.items():
                dp = d[(d.date >= a) & (d.date <= b)]
                for axis in AXES:
                    t = compare(dp, s, axis, SHAPE, "date")
                    shape.append(t.assign(기간=per, lag=lag, 드라이브형=dname, 축=axis).reset_index(names="그룹"))
                    for sub, f in SUBSETS.items():
                        ds = f(dp, s)
                        if ds.empty:
                            continue
                        t = compare(ds, s, axis, OUTCOME, "date_end")
                        outc.append(t.assign(기간=per, lag=lag, 드라이브형=dname, 축=axis, 묶음=sub).reset_index(names="그룹"))
    # 유형 × C2 — 깊은 하락 & D+1≥3% 안에서 (DRIVE_STUDY §3.2 표가 국면에 따라 달라지나)
    d = attach_regime(drives["하락형"], "date", cols=["c2", "breadth"], hist=hist, prefix="")
    ds = d[(d.tot <= -DEEP) & (d.next_r1 >= 0.03)].dropna(subset=["c2"])
    for (ov, vt), g in ds.groupby([ds.overlap_c.astype(str), ds.v_trend_c.astype(str)]):
        for c2v, gg in g.groupby("c2"):
            typ.append({"겹침": ov, "거래량": vt, "C2": "보유" if c2v else "퇴출", "건수": len(gg),
                        "저점지킴%": gg.rebreak.mean() * 100, "r10평균%": gg.r10.mean() * 100})
    return (pd.concat(shape, ignore_index=True), pd.concat(outc, ignore_index=True), pd.DataFrame(typ))


if __name__ == "__main__":
    t0 = time.time()
    hist = load_history()
    drives = load_drives(hist, "--rebuild" in sys.argv, holdout="--holdout" in sys.argv)
    if "--holdout" in sys.argv:
        print("⚠ 보류 구간(2026-06~) — 검증 때만 쓴다"); PERIODS.clear(); PERIODS["보류"] = ("2026-06-01", "2026-12-31")
    for k, d in drives.items():
        print(f"{k}: {len(d):,} 드라이브 · D1 {d.date.min():%Y-%m-%d}~{d.date.max():%Y-%m-%d} · 사후 창 끝 ≤ {d.post_end.max():%Y-%m-%d}")
    S, O, T = run(drives, hist)
    ROUT.mkdir(parents=True, exist_ok=True)
    tag = "_holdout" if "--holdout" in sys.argv else ""
    S.to_csv(ROUT / f"drive_regime_shape{tag}.csv", index=False)
    O.to_csv(ROUT / f"drive_regime_outcome{tag}.csv", index=False)
    T.to_csv(ROUT / f"drive_regime_type{tag}.csv", index=False)
    print(f"저장 {ROUT} · {time.time() - t0:.0f}s")
