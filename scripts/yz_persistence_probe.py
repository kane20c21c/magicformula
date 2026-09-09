#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yz_persistence_probe.py — "YZ 高/低 는 종목의 특성인가?"
=========================================================
질문의 내력 (케인 지적 2회, 둘 다 옳았다)

  **초판** — cell12 주석 ⑩ 에서 '항상 高/低' 29종목의 **이름을 눈으로 훑고**
  "YZ 분할은 사실상 스타일(섹터) 분할" 이라고 단정했다.

  **1차 지적 (2026-09-08)** — ① 29종목으로 섹터 37개를 가를 수 없다 ② 업종만이
  아니라 시총·매출·영업이익·업력을 봐야 한다 ③ 유니버스가 테크에 쏠려 섹터와
  규모가 교란된다. → 206종목 횡단면 회귀로 다시 검증했다 (아래 Part 2).

  **2차 지적 (2026-09-09) — 이게 핵심이다.** 케인이 물은 것은
  **"YZ 수준이 이 기간 종목의 안정적 특성이냐"** 인데, 나는 그걸
  **"어떤 회사가 변동성이 높냐"** 라는 다른 질문으로 바꿔치기해 답했다.
  게다가 그 답(작고·어리고·매출 적은 회사가 더 변동적)은 **교과서적 사실**이라
  안 나오는 게 이상한 것이지 발견이 아니다.

  → **답은 외부 변수가 필요 없다.** 그 종목의 YZ **순위가 시간에 대해 안 움직이는지**
    를 재고, **같은 잣대로 R²·RSI 와 비교**하면 된다. R² 축은 206종목 전부가
    양쪽을 오간다는 것을 이미 알고 있으므로 자연스러운 대조군이 된다.

Part 1 (답)  — 안정성 5종: 분산분해 ICC · 순위 자기상관 · 5분위 전이 · 시대분할 · 편재
Part 2 (참고) — 횡단면 설명변수. **답이 아니고, 결과도 예상 범위다.** 우회의 기록으로 남긴다.

⚠ 인과를 주장하지 않는다. Part 1 은 '안정적인가', Part 2 는 '무엇과 같이 움직이는가' 까지다.
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
WIN_R2 = 60
LAGS = [5, 20, 60, 120, 250, 500]        # 거래일
TRANS_LAG = 250                          # 전이행렬 시차 ≈ 1년
N_QUANTILE = 5
SEED = 20260909
ERAS = {"23-24": ("2023-01-01", "2024-12-31"), "25-26": ("2025-01-01", "2026-12-31")}

# 케인이 지목한 "쏠림" 축 (Part 2 전용) — 판단이지 정본이 아니다.
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


def axes(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """세 축을 **그날 전 종목 백분위**(0~1)로. 12셀 분할과 같은 잣대다.

    백분위로 바꾸는 이유가 둘 있다 —
      ① 세 축의 단위(무단위 σ / 0~1 / 0~100)가 달라 그대로는 비교가 안 된다
      ② 시장 전체가 함께 움직인 몫(공통 시간 효과)이 빠져, 남는 것이 순수한
         **종목 간 상대 위치**다. '종목의 특성인가' 라는 질문에 맞는 좌표계다.
    """
    piv = lambda c: df.pivot_table(index="Date", columns="Ticker", values=c, aggfunc="last")
    close, yz, rsi = (piv(c).astype(float).drop(columns=[BM], errors="ignore")
                      for c in ("Close", "YZ_60", "RSI"))
    y = np.log(close).replace([np.inf, -np.inf], np.nan)
    x = pd.Series(np.arange(len(y), dtype=float), index=y.index)
    r2 = y.apply(lambda c: x.rolling(WIN_R2, min_periods=WIN_R2).corr(c)) ** 2

    valid = r2.notna() & yz.notna() & rsi.notna()      # cell12 의 유효 관측 정의와 동일
    pct = lambda m: m.where(valid).rank(axis=1, pct=True)
    return {"YZ_60": pct(yz), "R²(60)": pct(r2), "RSI": pct(rsi)}, valid


# ─────────────── Part 1 — 안정성 (케인 질문의 답) ───────────────
def icc(m: pd.DataFrame) -> tuple[float, float, float]:
    """순위 분산 중 **종목 간** 몫. 1.0 = 종목마다 고정 / 0.0 = 매일 새로 뽑는 것과 같다."""
    s = m.stack()
    g = s.groupby(level=1)
    between, within = g.mean().var(ddof=1), g.var(ddof=1).mean()
    return float(between / (between + within)), float(np.sqrt(between)), float(np.sqrt(within))


def autocorr(m: pd.DataFrame, lags=LAGS) -> dict[int, float]:
    """오늘 순위가 h거래일 뒤에도 유지되는가 (전 종목·전 날짜 풀링 피어슨)."""
    out = {}
    for h in lags:
        a, b = m.iloc[:-h], m.shift(-h).iloc[:-h]
        ok = (a.notna() & b.notna()).values
        out[h] = float(np.corrcoef(a.values[ok], b.values[ok])[0, 1]) if ok.sum() > 100 else np.nan
    return out


def transition(m: pd.DataFrame, lag=TRANS_LAG, q=N_QUANTILE) -> tuple[pd.DataFrame, float]:
    """q분위 → lag 뒤 q분위 전이확률(%). 대각 평균이 20% 면 무작위와 같다."""
    qq = np.ceil(m * q).clip(1, q)
    a, b = qq.iloc[:-lag], qq.shift(-lag).iloc[:-lag]
    ok = (a.notna() & b.notna()).values
    T = pd.crosstab(a.values[ok], b.values[ok], normalize="index") * 100
    T.index = [f"{int(i)}분위" for i in T.index]
    T.columns = [f"→{int(c)}분위" for c in T.columns]
    diag = float(np.mean([T.iat[i, i] for i in range(min(T.shape))]))
    return T, diag


def era_rho(m: pd.DataFrame) -> tuple[float, int]:
    """전반기 순위 vs 후반기 순위. 시대가 바뀌어도 서열이 유지되는가."""
    e = [m.loc[a:b].median() for a, b in ERAS.values()]
    ok = e[0].notna() & e[1].notna()
    return float(e[0][ok].rank().corr(e[1][ok].rank())), int(ok.sum())


def sidedness(m: pd.DataFrame, valid: pd.DataFrame) -> tuple[int, int, int, int]:
    """반대편(반대 절반)에 가 본 적이 있는 종목 수."""
    hi, lo = (m >= 0.5) & valid, (m < 0.5) & valid
    A = set(m.columns[valid.any()])
    H, L = set(m.columns[hi.any()]), set(m.columns[lo.any()])
    return len(A - L), len(A - H), len(H & L), len(A)


def null_contrast(m: pd.DataFrame, valid: pd.DataFrame, rng) -> tuple[float, float, float, float]:
    """'高 비율' 의 실측 분포 vs **매일 동전던지기** 귀무분포.

    귀무는 종목마다 관측일수가 다르므로 그 일수로 이항분포를 뽑는다.
    실측이 귀무보다 넓게 퍼져 있을수록 순위가 종목에 붙어 있다는 뜻이다.
    """
    hi = (m >= 0.5) & valid
    n = valid.sum()
    obs = hi.sum() / n
    sim = pd.Series(rng.binomial(n.values, 0.5) / n.values, index=n.index)
    ext = lambda s: float(((s < 0.05) | (s > 0.95)).mean())
    return float(obs.std()), ext(obs), float(sim.std()), ext(sim)


def stability(P: dict[str, pd.DataFrame], valid: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SEED)
    rows, trans = [], {}
    for k, m in P.items():
        i, sb, sw = icc(m)
        ac = autocorr(m)
        T, diag = transition(m)
        rho, n_e = era_rho(m)
        only_hi, only_lo, both, tot = sidedness(m, valid)
        o_sd, o_ex, n_sd, n_ex = null_contrast(m, valid, rng)
        trans[k] = T
        rows.append({
            "축": k, "ICC": i, "종목간 σ": sb, "종목내 σ": sw,
            **{f"ρ({h}일)": ac[h] for h in LAGS},
            "1년 전이 대각%": diag, "시대 ρ": rho,
            "高만": only_hi, "低만": only_lo, "양쪽": both,
            "高비율 σ(실측)": o_sd, "高비율 σ(귀무)": n_sd,
            "극단 비율(실측)": o_ex, "극단 비율(귀무)": n_ex,
        })
    return {"table": pd.DataFrame(rows), "trans": trans,
            "n_obs": int(valid.sum().sum()), "n_tickers": int(valid.any().sum()),
            "n_days": int(valid.any(axis=1).sum()), "lags": LAGS}


# ─────────────── Part 2 — 횡단면 (참고. 답이 아니다) ───────────────
def features(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    sys.path.insert(0, str(STOLAB / "longlivevault"))
    from stolab_data import TICKER_LIST, EXTEND_LIST  # noqa: E402
    meta = {t[0]: (t[1], t[2]) for t in list(TICKER_LIST) + list(EXTEND_LIST)}

    d = df[df["Ticker"].isin(tickers)]
    g = d.groupby("Ticker")
    f = pd.DataFrame(index=pd.Index(tickers, name="Ticker"))
    f["name"] = [meta.get(t, ("", ""))[0] or "" for t in tickers]
    f["sector"] = [meta.get(t, ("", "기타"))[1] for t in tickers]
    f["mcap"], f["amount"] = g["MarketCap"].median(), g["Amount"].median()
    f["price"] = g["Close"].median()

    cd = REPO / "korean_mkt_study" / "data" / "dart_cache"
    rows = [pd.read_parquet(cd / f"{t}.parquet") for t in tickers if (cd / f"{t}.parquet").exists()]
    if rows:
        fin = pd.concat(rows, ignore_index=True)
        fin["ticker"] = fin["ticker"].astype(str).str.zfill(6)
        for col in ("revenue", "op_income"):
            f[col] = (fin.dropna(subset=[col]).sort_values("year")
                      .groupby("ticker")[col].last().reindex(f.index))

    fp = REPO / "korean_mkt_study" / "data" / "fundamentals.parquet"
    if fp.exists():
        fu = pd.read_parquet(fp).reset_index()
        fu["ticker"] = fu["ticker"].astype(str).str.zfill(6)
        first = fu[fu["ticker"].isin(tickers)].groupby("ticker")["date"].min()
        f["age_yrs"] = 2026 - first.dt.year.reindex(f.index)

    f["log_mcap"] = np.log10(f["mcap"].replace(0, np.nan))
    f["log_rev"] = np.log10(f["revenue"].where(f["revenue"] > 0))
    f["log_opi"] = np.log10(f["op_income"].where(f["op_income"] > 0))
    f["op_margin"] = f["op_income"] / f["revenue"].replace(0, np.nan)
    f["log_turnover"] = np.log10((f["amount"] / f["mcap"].replace(0, np.nan)).where(lambda s: s > 0))
    f["is_tech"] = f["sector"].isin(TECH_SECTORS)
    return f


def eta2(y: np.ndarray, grp: np.ndarray) -> float:
    m = y.mean()
    sst = ((y - m) ** 2).sum()
    if sst <= 0:
        return np.nan
    return float(sum(len(y[grp == g]) * (y[grp == g].mean() - m) ** 2
                     for g in np.unique(grp)) / sst)


def spearman(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    m = a.notna() & b.notna()
    return ((float(a[m].rank().corr(b[m].rank())), int(m.sum()))
            if m.sum() >= 10 else (np.nan, int(m.sum())))


def ols_r2(y: pd.Series, X: pd.DataFrame) -> tuple[float, int, pd.Series]:
    X = X.astype(float)
    m = y.notna() & X.notna().all(axis=1)
    Y = y[m].values.astype(float)
    A = np.column_stack([np.ones(int(m.sum())), X[m].values])
    b, *_ = np.linalg.lstsq(A, Y, rcond=None)
    pred = A @ b
    return (float(1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()),
            int(m.sum()), pd.Series(Y - pred, index=y[m].index))


CONT = [("log_turnover", "회전율(로그)"), ("log_rev", "매출(로그)"),
        ("log_opi", "영업이익(로그·흑자만)"), ("log_mcap", "시총(로그)"),
        ("age_yrs", "업력(년·우측절단)"), ("op_margin", "영업이익률"), ("price", "주가 수준")]
BASE = ["log_turnover", "log_mcap", "log_rev", "age_yrs"]


def cross_section(f: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SEED)
    y = f["yz_hi_ratio"].astype(float)
    rows = []
    for v, lab in CONT:
        r, n = spearman(y, f[v])
        rows.append({"변수": lab, "ρ": r, "n": n})
    for v, lab in [("sector", "섹터 (37개)"), ("is_tech", "테크 계열 여부")]:
        m = f[v].notna() & y.notna()
        rows.append({"변수": lab, "η²": eta2(y[m].values, f.loc[m, v].astype(str).values),
                     "n": int(m.sum())})
    r2b, _, resb = ols_r2(y, f[BASE])
    sec = pd.get_dummies(f["sector"], drop_first=True).astype(float)
    r2f, nf, _ = ols_r2(y, pd.concat([f[BASE], sec], axis=1))
    idx, lab_arr, null = resb.index, f.loc[resb.index, "sector"].values.copy(), []
    for _ in range(1000):
        rng.shuffle(lab_arr)
        d = pd.get_dummies(pd.Series(lab_arr, index=idx), drop_first=True).astype(float)
        null.append(ols_r2(y.loc[idx], pd.concat([f.loc[idx, BASE], d], axis=1))[0] - r2b)
    null = np.array(null)
    return {"single": pd.DataFrame(rows), "r2_base": r2b, "r2_full": r2f,
            "inc": r2f - r2b, "n_full": nf,
            "inc_null_med": float(np.median(null)), "inc_null_p95": float(np.quantile(null, .95)),
            "inc_p": (int((null >= r2f - r2b).sum()) + 1) / (len(null) + 1),
            "conf": {v: eta2(f[v].dropna().values, f.loc[f[v].notna(), "sector"].values)
                     for v in ("log_mcap", "log_rev", "age_yrs", "log_turnover")}}


# ─────────────────────────── 조립 ───────────────────────────
def build(df: pd.DataFrame | None = None) -> dict:
    df = load_panel() if df is None else df
    P, valid = axes(df)
    R = stability(P, valid)
    yz_hi = ((P["YZ_60"] >= 0.5) & valid).sum() / valid.sum()
    f = features(df, list(yz_hi.index)).assign(yz_hi_ratio=yz_hi)
    R["cross"] = cross_section(f)
    R["names"] = {
        "hi": sorted(f.loc[yz_hi >= 1.0, "name"], key=str),
        "lo": sorted(f.loc[yz_hi <= 0.0, "name"], key=str),
    }
    return R


def main() -> int:
    R = build()
    pd.set_option("display.width", 220)
    t = R["table"]
    print(f"■ YZ 는 종목의 특성인가 — {R['n_tickers']}종목 / {R['n_days']}일 / "
          f"{R['n_obs']:,}관측 (BM 제외)\n")
    print("[① 분산분해] 순위 분산 중 종목 간 몫")
    print(t[["축", "ICC", "종목간 σ", "종목내 σ"]].round(3).to_string(index=False))
    print("\n[② 순위 자기상관]")
    print(t[["축"] + [f"ρ({h}일)" for h in LAGS]].round(3).to_string(index=False))
    print("\n[③ 1년 전이 대각 · 시대 ρ · 편재]")
    print(t[["축", "1년 전이 대각%", "시대 ρ", "高만", "低만", "양쪽"]].round(3).to_string(index=False))
    print("\n[④ 귀무 대조]")
    print(t[["축", "高비율 σ(실측)", "高비율 σ(귀무)", "극단 비율(실측)", "극단 비율(귀무)"]]
          .round(3).to_string(index=False))
    for k, T in R["trans"].items():
        print(f"\n[전이행렬 {k}]\n{T.round(1).to_string()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
