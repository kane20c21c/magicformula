"""드라이브 × 시장 국면 비교 (2026-11-10 케인 — 통일장 국면 라벨 표를 D1 에 붙여 길이·속도·거래량·갭 비교).

입력
  드라이브  상승형 out/crash/drive_len_up.parquet (drive_length.build(+1))
           하락형 out/crash/drive_len_down.parquet (drive_length.build(-1) — 없으면 만든다, --rebuild 로 다시)
           정의·유니버스는 drive_types·crash_study 그대로 (전년 거래대금 ≥10억 & 전년말 시총 ≥1,000억, 2016~)
  국면     MagicFormula output/market_layer/regime_history.parquet (market_layer.attach_regime) — D1 의 라벨
           lag=0 본 결과, lag=1(D0 라벨) 확인용

규칙
  · 2026-09-14 이후에 끝난 드라이브 제외 — prices.parquet 종가가 애프터마켓 개장 뒤 정규장 종가와 어긋남 (breadth_study 주석)
  · 드라이브 '건수' 비교는 하지 않는다 — 브레드스가 곧 상승·하락 종목 수라 순환. 성질(길이·속도·거래량·갭)만 비교
  · 신뢰구간 = D1 날짜 단위 부트스트랩 (같은 날 시작한 드라이브는 독립이 아니다). 국면 라벨은 날짜마다 하나라
    국면 그룹이 날짜를 나누므로 그룹별로 따로 뽑는다
  · 길이 n 은 사후값 — 이 표는 '설명' 용도, 신호 아님
출력: out/regime/drive_regime_*.csv · 표준출력 요약
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from crash_study import OUT                                     # noqa: E402
from market_layer import attach_regime, load_history            # noqa: E402

CUT = pd.Timestamp("2026-09-14")
ROUT = HERE / "out" / "regime"
NB = 1000
RNG = np.random.default_rng(20261110)
AXES = {"breadth": {True: "상승", False: "하락"}, "c2": {True: "보유", False: "퇴출"}}
PERIODS = {"전체": ("2016-01-01", "2026-09-13"), "2016~2020": ("2016-01-01", "2020-12-31"), "2021~2026.09": ("2021-01-01", "2026-09-13")}


def load_drives(rebuild: bool = False) -> dict[str, pd.DataFrame]:
    up = pd.read_parquet(OUT / "drive_len_up.parquet")
    p = OUT / "drive_len_down.parquet"
    if rebuild or not p.exists():
        import drive_length
        dn = drive_length.build(-1); dn.to_parquet(p)
    dn = pd.read_parquet(p)
    out = {}
    for name, d in (("상승형", up), ("하락형", dn)):
        d = d[pd.to_datetime(d.date_end) < CUT].copy()
        d["date"] = pd.to_datetime(d.date)
        out[name] = d
    return out


# ── 지표: 드라이브마다 (분자, 분모) — 날짜별로 더해 두면 부트스트랩이 빠르다 ─────────────────
def per_drive(d: pd.DataFrame) -> pd.DataFrame:
    n = d.n.astype(int)
    m = pd.DataFrame(index=d.index)
    for k in (3, 4, 5, 6):
        m[f"P{k + 1}|{k}_n"] = (n >= k + 1).astype(float); m[f"P{k + 1}|{k}_d"] = (n >= k).astype(float)
    m["평균길이_n"] = n.astype(float); m["평균길이_d"] = 1.0
    for col, nm in (("speed", "속도A(기하)"), ("v_trend", "거래량T(기하)"), ("v_level", "거래량수준(기하)")):
        x = np.log(d[col].where(d[col] > 0))
        m[f"{nm}_n"] = x.fillna(0.0); m[f"{nm}_d"] = x.notna().astype(float)
    for col, nm in (("overlap", "겹침ō(평균)"),):
        m[f"{nm}_n"] = d[col].fillna(0.0); m[f"{nm}_d"] = d[col].notna().astype(float)
    m["누적|등락|_n"] = d.tot.abs().fillna(0.0); m["누적|등락|_d"] = d.tot.notna().astype(float)
    m["D1|등락|_n"] = d.r_d1.abs().fillna(0.0); m["D1|등락|_d"] = d.r_d1.notna().astype(float)
    for col, lab, nm in (("speed_c", "가속", "가속%"), ("v_trend_c", "증가", "증가%"), ("overlap_c", "계단", "계단%")):
        ok = d[col].notna()
        m[f"{nm}_n"] = (d[col].astype(str) == lab).astype(float) * ok; m[f"{nm}_d"] = ok.astype(float)
    return m


METRICS = ["P4|3", "P5|4", "P6|5", "P7|6", "평균길이", "속도A(기하)", "거래량T(기하)", "거래량수준(기하)", "겹침ō(평균)",
           "누적|등락|", "D1|등락|", "가속%", "증가%", "계단%"]
GEO = {"속도A(기하)", "거래량T(기하)", "거래량수준(기하)"}
PCT = {"P4|3", "P5|4", "P6|5", "P7|6", "가속%", "증가%", "계단%", "누적|등락|", "D1|등락|"}


def finish(name: str, num, den):
    v = num / den
    return np.exp(v) if name in GEO else v


def daily_sums(d: pd.DataFrame) -> pd.DataFrame:
    return per_drive(d).groupby(d.date.values).sum()


def boot(day: pd.DataFrame) -> dict[str, np.ndarray]:
    """날짜 재추출 NB 회 → 지표별 분포."""
    a = day.to_numpy(); n = len(a)
    idx = RNG.integers(0, n, size=(NB, n))
    s = a[idx].sum(axis=1)                                    # (NB, 열)
    cols = list(day.columns)
    return {mname: finish(mname, s[:, cols.index(f"{mname}_n")], s[:, cols.index(f"{mname}_d")]) for mname in METRICS}


def group_table(d: pd.DataFrame, key: str, labels: dict) -> tuple[pd.DataFrame, dict]:
    rows, bs = {}, {}
    for v, lab in labels.items():
        g = d[d[key] == v]
        if g.empty:
            continue
        day = daily_sums(g)
        tot = day.sum()
        r = {"드라이브": len(g), "D1 날짜": day.shape[0]}
        for mname in METRICS:
            r[mname] = finish(mname, tot[f"{mname}_n"], tot[f"{mname}_d"])
        rows[lab] = r; bs[lab] = boot(day)
    return pd.DataFrame(rows).T, bs


def diff_ci(bs: dict, a: str, b: str) -> pd.Series:
    out = {}
    for mname in METRICS:
        x = bs[a][mname] - bs[b][mname]
        lo, hi = np.percentile(x, [2.5, 97.5])
        out[mname] = (lo, hi)
    return pd.Series(out)


def fmt(t: pd.DataFrame) -> pd.DataFrame:
    f = t.copy().astype(object)
    for c in t.columns:
        if c in ("드라이브", "D1 날짜"):
            f[c] = t[c].map(lambda x: f"{int(x):,}")
        elif c in PCT:
            f[c] = t[c].map(lambda x: f"{x * 100:.1f}%")
        else:
            f[c] = t[c].map(lambda x: f"{x:.3f}")
    return f


def run(lag: int, drives: dict, hist: pd.DataFrame, period: str = "전체", verbose: bool = True):
    a, b = PERIODS[period]
    res = []
    for dname, d0 in drives.items():
        d = attach_regime(d0, "date", cols=["breadth", "c2", "regime", "gate_window"], hist=hist, lag=lag, prefix="")
        d = d[(d.date >= a) & (d.date <= b)]
        for axis, labels in list(AXES.items()) + [("regime", {x: x for x in ["상승·보유", "상승·퇴출", "하락·보유", "하락·퇴출"]}),
                                                   ("gate_window", {True: "갭쇼크창 안", False: "갭쇼크창 밖"})]:
            dd = d.dropna(subset=[axis])
            t, bs = group_table(dd, axis, labels)
            labs = list(labels.values())
            if len(labs) == 2 and all(x in bs for x in labs):
                ci = diff_ci(bs, labs[0], labs[1])
                diff = t.loc[labs[0], METRICS] - t.loc[labs[1], METRICS]
                t.loc[f"차({labs[0]}−{labs[1]})"] = [np.nan, np.nan] + list(diff.values)
                t.loc["95%CI 하"] = [np.nan, np.nan] + [ci[m][0] for m in METRICS]
                t.loc["95%CI 상"] = [np.nan, np.nan] + [ci[m][1] for m in METRICS]
            t.insert(0, "축", axis); t.insert(0, "드라이브형", dname); t.insert(0, "lag", lag); t.insert(0, "기간", period)
            res.append(t.reset_index(names="그룹"))
    return pd.concat(res, ignore_index=True)


if __name__ == "__main__":
    t0 = time.time()
    drives = load_drives("--rebuild" in sys.argv)
    hist = load_history()
    for k, d in drives.items():
        print(f"{k}: {len(d):,} 드라이브 · D1 {d.date.min():%Y-%m-%d}~{d.date.max():%Y-%m-%d}")
    ROUT.mkdir(parents=True, exist_ok=True)
    allres = []
    for lag in (0, 1):
        for period in PERIODS:
            allres.append(run(lag, drives, hist, period))
    R = pd.concat(allres, ignore_index=True)
    R.to_csv(ROUT / "drive_regime_all.csv", index=False)
    print(f"저장 {ROUT / 'drive_regime_all.csv'} · {time.time() - t0:.0f}s")
