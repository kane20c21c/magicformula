#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yz_persistence_probe.py — "YZ 는 종목의 속성인가, 그렇다면 무엇의 속성인가"
==========================================================================
배경 (2026-09-08 케인 지적)
  cell12 리포트의 주석 ⑩ 에서 "YZ 분할은 사실상 스타일(섹터) 분할" 이라고 썼는데,
  근거가 **극단 29종목의 이름을 눈으로 훑은 것**뿐이었다. 케인 지적 세 가지:
    ① 안 움직인 종목 명단만으로 나눌 수 없다 (표본 29개로 섹터 38개를 가를 수 없다)
    ② 업종만이 아니라 **시총·매출·영업이익·업력**을 함께 봐야 한다
    ③ 우리 유니버스는 **반도체·전자·전기에 쏠려** 있어 섹터와 규모가 교란돼 있다
  → 이 스크립트가 그 재검증이다.

설계 (케인 확정 2026-09-08)
  종속변수 : 206종목 **각각의 YZ高 비율**(0~1). 극단 29종목이 아니라 전 종목 연속값.
             (29종목은 이 분포의 양 끝 꼬리로 자연히 포함된다)
  검정     : 단독 설명력 → **교란 통제 후 증분 설명력** → 순열검정.
             섹터는 38개라 표본이 작아 **우연히도 설명력이 나온다** — 순열분포로 판정한다.

⚠ 이 스크립트는 인과를 주장하지 않는다. "무엇과 같이 움직이는가" 까지만 답한다.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve()
REPO = HERE.parent.parent
STOLAB = REPO.parent
BM = "102110"
WIN_R2, WIN_YZ = 60, 60
N_PERM = 10_000
SEED = 20260908

# 케인이 지목한 "쏠림" 축 — 반도체·전자·전기 계열로 볼 섹터.
#   ⚠ 이 목록은 **판단이지 정본이 아니다.** 무엇을 넣고 뺐는지 드러내려고 상수로 뽑았다.
TECH_SECTORS = {
    "반도체.완품", "반도체.전공", "반도체.후공", "반도체.핵심", "반도체.장비", "반도체.기타",
    "로봇", "자동차소부장", "기계장비", "IT서비스", "인터넷통신", "에너지.전송",
}


# ─────────────────────────── 입력 ───────────────────────────
def load_panel() -> pd.DataFrame:
    vault = STOLAB / "longlivevault" / "data" / "ohlcv"
    parts = [pd.read_parquet(vault / n) for n in ("core.parquet", "extend.parquet")
             if (vault / n).exists()]
    if not parts:
        raise SystemExit(f"❌ OHLCV parquet 없음: {vault}")
    df = pd.concat(parts, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.zfill(6)
    return df


def yz_hi_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """종목별 YZ高 비율 — cell12 와 **동일한 분할 규칙**(그날 전 종목 중앙값)."""
    piv = lambda c: df.pivot_table(index="Date", columns="Ticker", values=c, aggfunc="last")
    close, yz, rsi = piv("Close").astype(float), piv("YZ_60").astype(float), piv("RSI").astype(float)
    close, yz, rsi = (t.drop(columns=[BM], errors="ignore") for t in (close, yz, rsi))

    y = np.log(close).replace([np.inf, -np.inf], np.nan)
    x = pd.Series(np.arange(len(y), dtype=float), index=y.index)
    r2 = y.apply(lambda c: x.rolling(WIN_R2, min_periods=WIN_R2).corr(c)) ** 2

    valid = r2.notna() & yz.notna() & rsi.notna()      # cell12 의 유효 관측 정의
    hi = (yz.ge(yz.median(axis=1), axis=0) & valid)
    n = valid.sum()
    out = pd.DataFrame({"yz_hi_ratio": hi.sum() / n, "n_days": n})
    return out[out["n_days"] > 0]


def features(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """케인이 지목한 설명변수 조립 — 시총·매출·영업이익·업력 + 유동성·시장·섹터."""
    sys.path.insert(0, str(STOLAB / "longlivevault"))
    from stolab_data import TICKER_LIST, EXTEND_LIST  # noqa: E402
    meta = {t[0]: (t[1], t[2]) for t in list(TICKER_LIST) + list(EXTEND_LIST)}

    d = df[df["Ticker"].isin(tickers)]
    g = d.groupby("Ticker")
    f = pd.DataFrame(index=pd.Index(tickers, name="Ticker"))
    f["name"] = [meta.get(t, ("", ""))[0] or "" for t in tickers]
    f["sector"] = [meta.get(t, ("", "기타"))[1] for t in tickers]
    f["mcap"] = g["MarketCap"].median()                       # 원
    f["amount"] = g["Amount"].median()                        # 원, 일 거래대금
    f["price"] = g["Close"].median()
    # 시장 — Market 은 대부분 공란이라 종목별 최빈 비공란
    mk = d[d["Market"].astype(str).str.len() > 0].groupby("Ticker")["Market"]
    f["market"] = mk.agg(lambda s: s.mode().iat[0] if len(s.mode()) else np.nan)

    # ── DART 재무 (MagicFormula/korean_mkt_study/data/dart_cache) ──
    cd = REPO / "korean_mkt_study" / "data" / "dart_cache"
    rows = [pd.read_parquet(cd / f"{t}.parquet") for t in tickers if (cd / f"{t}.parquet").exists()]
    if rows:
        fin = pd.concat(rows, ignore_index=True)
        fin["ticker"] = fin["ticker"].astype(str).str.zfill(6)
        # 종목별 **가장 최근 유효 연도** 를 쓴다 (결산 시차로 종목마다 다르다)
        for col, out in (("revenue", "revenue"), ("op_income", "op_income")):
            s = fin.dropna(subset=[col]).sort_values("year").groupby("ticker")[col].last()
            f[out] = s.reindex(f.index)
            yr = fin.dropna(subset=[col]).sort_values("year").groupby("ticker")["year"].last()
            f[f"{out}_year"] = yr.reindex(f.index)

    # ── 업력 프록시: fundamentals.parquet 최초 등장 ──
    # ⚠ 2010-01 부터 시작하는 파일이라 그 이전 상장은 **전부 2010 으로 몰린다**(우측중도절단).
    #   그래서 연속 변수로 쓰지 않고 '신생(2015~) / 중간 / 오래됨(2010 이전 상장)' 로도 본다.
    fp = REPO / "korean_mkt_study" / "data" / "fundamentals.parquet"
    if fp.exists():
        fu = pd.read_parquet(fp).reset_index()
        fu["ticker"] = fu["ticker"].astype(str).str.zfill(6)
        first = fu[fu["ticker"].isin(tickers)].groupby("ticker")["date"].min()
        f["first_year"] = first.dt.year.reindex(f.index)
        f["age_yrs"] = 2026 - f["first_year"]
        f["age_censored"] = f["first_year"].le(2010)          # True = 실제 업력은 이보다 길다

    # 파생
    f["log_mcap"] = np.log10(f["mcap"].replace(0, np.nan))
    f["log_rev"] = np.log10(f["revenue"].where(f["revenue"] > 0))
    f["log_opi"] = np.log10(f["op_income"].where(f["op_income"] > 0))   # 적자는 NaN (별도 플래그)
    f["op_deficit"] = f["op_income"].le(0)
    f["op_margin"] = f["op_income"] / f["revenue"].replace(0, np.nan)
    f["turnover"] = f["amount"] / f["mcap"].replace(0, np.nan)          # 일 회전율
    f["log_turnover"] = np.log10(f["turnover"].where(f["turnover"] > 0))
    f["is_tech"] = f["sector"].isin(TECH_SECTORS)
    return f


# ─────────────────────────── 검정 ───────────────────────────
def eta2(y: np.ndarray, grp: np.ndarray) -> float:
    """집단이 설명하는 분산 비율 (일원분산분석 η²)."""
    m = y.mean()
    sst = ((y - m) ** 2).sum()
    if sst <= 0:
        return np.nan
    ssb = sum(len(y[grp == g]) * (y[grp == g].mean() - m) ** 2 for g in np.unique(grp))
    return float(ssb / sst)


def perm_p(stat: float, y: np.ndarray, grp: np.ndarray, rng, n=N_PERM) -> float:
    """집단 라벨을 섞어 만든 귀무분포에서의 p — 표본이 작을 때 η² 는 그냥도 커진다."""
    if not np.isfinite(stat):
        return np.nan
    g = grp.copy()
    cnt = 0
    for _ in range(n):
        rng.shuffle(g)
        if eta2(y, g) >= stat:
            cnt += 1
    return (cnt + 1) / (n + 1)


def spearman(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    m = a.notna() & b.notna()
    if m.sum() < 10:
        return (np.nan, int(m.sum()))
    return (float(a[m].rank().corr(b[m].rank())), int(m.sum()))


def residualize(y: pd.Series, x: pd.Series) -> pd.Series:
    """y 에서 x 의 선형 성분을 뺀 잔차 (둘 다 있는 종목만)."""
    m = y.notna() & x.notna()
    if m.sum() < 10:
        return pd.Series(np.nan, index=y.index)
    b, a = np.polyfit(x[m], y[m], 1)
    r = pd.Series(np.nan, index=y.index)
    r[m] = y[m] - (a + b * x[m])
    return r


def ols_r2(y: pd.Series, X: pd.DataFrame) -> tuple[float, int, pd.Series]:
    """설명변수 묶음의 결정계수 + 잔차 (결측 종목은 통째로 제외 — 공통표본)."""
    X = X.astype(float)
    m = y.notna() & X.notna().all(axis=1)
    Y = y[m].values.astype(float)
    A = np.column_stack([np.ones(int(m.sum())), X[m].values])
    b, *_ = np.linalg.lstsq(A, Y, rcond=None)
    pred = A @ b
    r2 = 1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
    return float(r2), int(m.sum()), pd.Series(Y - pred, index=y[m].index)


CONT = [("log_turnover", "회전율(로그)"), ("log_rev", "매출(로그)"),
        ("log_opi", "영업이익(로그·흑자만)"), ("log_mcap", "시총(로그)"),
        ("age_yrs", "업력(년·우측절단)"), ("op_margin", "영업이익률"),
        ("price", "주가 수준")]
BASE = ["log_turnover", "log_mcap", "log_rev", "age_yrs"]


def analyze(f: pd.DataFrame) -> dict:
    """단독 설명력 → 교란 통제 → 증분 순열검정. 표 3개와 요약 수치를 돌려준다."""
    rng = np.random.default_rng(SEED)
    y = f["yz_hi_ratio"].astype(float)

    # A. 단독 설명력 + 회전율/시총 통제 후
    res_t, res_m = residualize(y, f["log_turnover"]), residualize(y, f["log_mcap"])
    rows = []
    for v, lab in CONT:
        r, n = spearman(y, f[v])
        rows.append({"변수": lab, "ρ": r, "ρ²": r * r, "n": n,
                     "회전율 통제 후 ρ": spearman(res_t, f[v])[0],
                     "시총 통제 후 ρ": spearman(res_m, f[v])[0] if v != "log_mcap" else np.nan})
    for v, lab in [("sector", "섹터 (37개)"), ("market", "시장 (KOSPI/KOSDAQ)"),
                   ("is_tech", "테크 계열 여부"), ("op_deficit", "영업적자 여부"),
                   ("age_censored", "2010년 이전 상장")]:
        m = f[v].notna() & y.notna()
        g = f.loc[m, v].astype(str).values
        e = eta2(y[m].values, g)
        mt = res_t.notna() & f[v].notna()
        rows.append({"변수": lab, "η²": e, "순열 p": perm_p(e, y[m].values, g, rng, 3000), "n": int(m.sum()),
                     "회전율 통제 후 η²": eta2(res_t[mt].values, f.loc[mt, v].astype(str).values),
                     "시총 통제 후 η²": eta2(res_m[mt].values, f.loc[mt, v].astype(str).values)})
    single = pd.DataFrame(rows)

    # B. 누적 설명력
    combos = [("회전율", ["log_turnover"]), ("시총", ["log_mcap"]), ("매출", ["log_rev"]),
              ("시총+매출", ["log_mcap", "log_rev"]),
              ("시총+매출+업력", ["log_mcap", "log_rev", "age_yrs"]),
              ("회전율+시총", ["log_turnover", "log_mcap"]),
              ("회전율+시총+매출", ["log_turnover", "log_mcap", "log_rev"]),
              ("연속 4변수 전부", BASE)]
    cum = pd.DataFrame([{"설명변수 묶음": l, "R²": ols_r2(y, f[c])[0], "n": ols_r2(y, f[c])[1]}
                        for l, c in combos])

    # C. 섹터 증분 — 규모를 전부 통제한 뒤에도 섹터가 남는가 (케인 지적의 최종 판정)
    r2b, _, resb = ols_r2(y, f[BASE])
    idx = resb.index
    sec = pd.get_dummies(f["sector"], drop_first=True).astype(float)
    r2f, nf, _ = ols_r2(y, pd.concat([f[BASE], sec], axis=1))
    lab_arr = f.loc[idx, "sector"].values.copy()
    null = []
    for _ in range(1000):
        rng.shuffle(lab_arr)
        d = pd.get_dummies(pd.Series(lab_arr, index=idx), drop_first=True).astype(float)
        null.append(ols_r2(y.loc[idx], pd.concat([f.loc[idx, BASE], d], axis=1))[0] - r2b)
    null = np.array(null)
    inc = r2f - r2b

    # D. 섹터 내 편차만으로 본 규모 효과 (반대 방향 통제)
    g2 = f.assign(_y=y).dropna(subset=["log_mcap"])
    big = g2.groupby("sector").filter(lambda d: len(d) >= 4)
    dm = lambda s, by: s - s.groupby(by).transform("mean")
    within = float(dm(big["_y"], big["sector"]).corr(
        dm(big["log_mcap"], big["sector"]), method="spearman"))

    shares = f["sector"].value_counts() / len(f)
    return {
        "single": single, "cum": cum,
        "r2_base": r2b, "r2_full": r2f, "inc": inc, "n_full": nf,
        "inc_null_med": float(np.median(null)), "inc_null_p95": float(np.quantile(null, .95)),
        "inc_p": (int((null >= inc).sum()) + 1) / (len(null) + 1),
        "within_sector_mcap_rho": within,
        "rho_mcap_all": spearman(y, f["log_mcap"])[0],
        "hhi": float((shares ** 2).sum()), "eff_sectors": float(1 / (shares ** 2).sum()),
        "n_sectors": int(f["sector"].nunique()),
        "tech_share": float(f["is_tech"].mean()), "n_tech": int(f["is_tech"].sum()),
        "rho_tech": spearman(y[f.is_tech], f.loc[f.is_tech, "log_mcap"])[0],
        "rho_nontech": spearman(y[~f.is_tech], f.loc[~f.is_tech, "log_mcap"])[0],
        # 섹터가 규모를 얼마나 먹고 있나 — 교란의 크기 자체
        "conf": {v: eta2(f[v].dropna().values, f.loc[f[v].notna(), "sector"].values)
                 for v in ("log_mcap", "log_rev", "age_yrs", "log_turnover")},
        "n_tickers": len(f),
    }


def build(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    df = load_panel() if df is None else df
    yz = yz_hi_ratio(df)
    f = features(df, list(yz.index)).join(yz)
    return f, analyze(f)


def main() -> int:
    f, R = build()
    pd.set_option("display.width", 200)
    print(f"■ YZ 상주 재검증 — {R['n_tickers']}종목 (BM 제외)\n")
    print(f"유니버스 쏠림: 섹터 {R['n_sectors']}개, 유효섹터수 {R['eff_sectors']:.1f}개 "
          f"(HHI {R['hhi']:.4f}) · 테크 계열 {R['n_tech']}종목 {R['tech_share']:.1%}")
    print("섹터가 먹고 있는 분산 η²:", {k: round(v, 3) for k, v in R["conf"].items()})
    print("\n[A] 단독 설명력 + 통제 후\n", R["single"].round(3).to_string(index=False))
    print("\n[B] 누적 설명력\n", R["cum"].round(3).to_string(index=False))
    print(f"\n[C] 섹터 증분: 연속 4변수 R²={R['r2_base']:.3f} → +섹터 R²={R['r2_full']:.3f} "
          f"(증분 {R['inc']:+.3f}, 귀무 중앙 {R['inc_null_med']:+.3f} / 95% {R['inc_null_p95']:+.3f}, "
          f"p={R['inc_p']:.4f}, n={R['n_full']})")
    print(f"[D] 시총 ρ 원자료 {R['rho_mcap_all']:+.3f} → 섹터 내 편차만 {R['within_sector_mcap_rho']:+.3f}")
    print(f"[E] 시총 ρ 테크 {R['rho_tech']:+.3f} / 비테크 {R['rho_nontech']:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
