"""
scripts/redesign_from_pos52.py
==============================
52주 위치를 축으로 한 황금률 재설계 탐색 (Kane 지시 2026-08-25).

배경
----
`validate_area_redesign.py ic` 로 재보니, 08-24 리포트가 권고한 배합
(RSI 80 / 52주 12 / BB 8) 의 종합점수는 20일 상대수익 IC 가 +0.0067(p=0.25)
로 **유의하지 않다**. 네 재료 중 유일하게 유의한 축은 **52주 위치**
(IC +0.0218, p=0.0005). 그래서 이 축에서 다시 시작한다.

평가 기준
---------
- 1차 : 날짜별 횡단면 Spearman IC (20거래일 상대수익) + 연도별 부호
- 2차 : 상위 3% 선택 수익 · 슬롯 실운용 MDD
  (IC 는 전체 순위를, 상위선택은 꼭대기를 잰다 — 둘이 갈리면 둘 다 보고한다)

⚠ 한계 — `validate_area_redesign.py` 와 동일
- 유니버스 205종목 현재 명단 (생존편향)
- 2023-05~2026-08 은 초강세장 구간

실행
----
    python3 scripts/redesign_from_pos52.py axis     # 52주 위치 축 성질
    python3 scripts/redesign_from_pos52.py screen   # 후보 지표 IC 전수 스크리닝
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

VAULT = PROJECT_ROOT.parent / "LongLiveVault"
OHLCV_DIR = VAULT / "data" / "ohlcv"
CACHE = PROJECT_ROOT / "output" / "redesign_pos52_panel.parquet"

START = "2023-05-02"
FWD_H = 20


# ===========================================================================
# 패널
# ===========================================================================

def build() -> pd.DataFrame:
    raw = pd.concat([pd.read_parquet(OHLCV_DIR / f)
                     for f in ("core.parquet", "extend.parquet")],
                    ignore_index=True)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    out = []
    for tk, g in raw.groupby("Ticker", sort=False):
        d = g.set_index("Date").sort_index()
        if len(d) < 120:
            continue
        c = d["Close"]
        f = pd.DataFrame(index=d.index)
        f["Ticker"] = tk
        for col in d.columns:
            if col not in ("Ticker", "Name", "Market"):
                f[col] = d[col]

        # ── 52주 위치 계열 (창 길이별) ────────────────────────────
        for w in (63, 126, 252, 504):
            hi = c.rolling(w, min_periods=min(60, w)).max()
            lo = c.rolling(w, min_periods=min(60, w)).min()
            f[f"pos{w}"] = (c - lo) / (hi - lo).replace(0, np.nan)
        # 52주 고점 대비 낙폭 · 저점 대비 상승폭
        hi252 = c.rolling(252, min_periods=60).max()
        lo252 = c.rolling(252, min_periods=60).min()
        f["dd_from_hi"] = c / hi252 - 1.0
        f["up_from_lo"] = c / lo252 - 1.0
        # 신고가 근접일 수
        f["days_since_hi"] = (
            c.rolling(252, min_periods=60)
             .apply(lambda x: len(x) - 1 - int(np.argmax(x)), raw=True))

        # ── 파생 지표 ────────────────────────────────────────────
        for n in (5, 20, 60, 120):
            f[f"ret{n}"] = c.pct_change(n)
        f["ma20_gap"] = c / c.rolling(20).mean() - 1.0
        f["ma60_gap"] = c / c.rolling(60).mean() - 1.0
        f["ma200_gap"] = c / c.rolling(200, min_periods=100).mean() - 1.0
        f["vol_ratio"] = (d["Volume"].rolling(5).mean()
                          / d["Volume"].rolling(60).mean().replace(0, np.nan))
        f["amt_log"] = np.log1p(d["Amount"].rolling(20).mean())
        f["mcap_log"] = np.log1p(d["MarketCap"])
        f["fwd"] = c.shift(-FWD_H) / c - 1.0
        out.append(f.reset_index())

    panel = pd.concat(out, ignore_index=True)
    panel = panel[panel["Date"] >= START].reset_index(drop=True)
    panel["fwd_rel"] = panel["fwd"] - panel.groupby("Date")["fwd"].transform("mean")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(CACHE, index=False)
    print(f"저장 {CACHE} · {len(panel):,}행 · 컬럼 {len(panel.columns)}개")
    return panel


def get_panel() -> pd.DataFrame:
    return pd.read_parquet(CACHE) if CACHE.exists() else build()


# ===========================================================================
# IC
# ===========================================================================

def ic_series(panel: pd.DataFrame, x: pd.Series) -> pd.Series:
    d = pd.DataFrame({"Date": panel["Date"], "x": x, "y": panel["fwd_rel"]}).dropna()
    if len(d) == 0:
        return pd.Series(dtype=float)
    return d.groupby("Date").apply(
        lambda g: g["x"].corr(g["y"], method="spearman") if len(g) >= 20 else np.nan,
        include_groups=False).dropna()


def ic_stat(panel: pd.DataFrame, x: pd.Series, name: str = "") -> dict:
    ic = ic_series(panel, x)
    if len(ic) < 30:
        return {"지표": name, "평균IC": np.nan, "t": np.nan, "p": np.nan,
                "연도부호": "-", "관측일": len(ic)}
    t, p = stats.ttest_1samp(ic.values, 0.0)
    yr = ic.groupby(ic.index.year).mean()
    return {"지표": name, "평균IC": ic.mean(), "t": t, "p": p,
            "연도부호": f"{int((yr > 0).sum())}/{len(yr)}",
            "관측일": len(ic)}


# ===========================================================================
# 명령
# ===========================================================================

def cmd_axis(panel: pd.DataFrame) -> None:
    print("=== 52주 위치 계열 — 창 길이별 IC ===")
    rows = [ic_stat(panel, panel[c], n) for c, n in
            [("pos63", "3개월 위치"), ("pos126", "6개월 위치"),
             ("pos252", "52주 위치 (현행)"), ("pos504", "2년 위치"),
             ("dd_from_hi", "52주 고점 대비 낙폭"),
             ("up_from_lo", "52주 저점 대비 상승폭"),
             ("days_since_hi", "신고가 이후 경과일")]]
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\n=== 52주 위치 인코딩별 IC (순위상관이라 단조변환은 동일해야 정상) ===")
    p = panel["pos252"]
    rows = [
        ic_stat(panel, p, "원값"),
        ic_stat(panel, panel.groupby("Date")["pos252"].rank(pct=True), "날짜별 백분위"),
        ic_stat(panel, ((p - 0.5) * 20).clip(-10, 10), "선형 ±10"),
        ic_stat(panel, np.tanh((p - 0.5) / 0.2) * 10, "시그모이드"),
        ic_stat(panel, (p >= 0.8).astype(float), "0.8 이상 이진"),
        ic_stat(panel, (p >= 0.95).astype(float), "0.95 이상 이진"),
    ]
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\n=== 52주 위치 10분위별 20일 상대수익(%) — 단조인가 ===")
    d = panel[["Date", "pos252", "fwd_rel"]].dropna()
    q = d.groupby("Date")["pos252"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 10, labels=False))
    t = pd.DataFrame({"20일 상대수익%": d.groupby(q)["fwd_rel"].mean() * 100,
                      "표본": d.groupby(q).size(),
                      "위치 중앙값": d.groupby(q)["pos252"].median()})
    t.index = [f"{i+1}분위" for i in t.index]
    print(t.round(3).to_string())

    print("\n=== 상단 구간을 더 잘게 (상위 20%를 5등분) ===")
    top = d[d.groupby("Date")["pos252"].rank(pct=True) >= 0.8].copy()
    q2 = top.groupby("Date")["pos252"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False,
                          duplicates="drop"))
    t2 = pd.DataFrame({"20일 상대수익%": top.groupby(q2)["fwd_rel"].mean() * 100,
                       "표본": top.groupby(q2).size(),
                       "위치 중앙값": top.groupby(q2)["pos252"].median()})
    print(t2.round(3).to_string())

    print("\n=== 연도별 IC 안정성 ===")
    ic = ic_series(panel, panel["pos252"])
    yr = pd.DataFrame({"평균IC": ic.groupby(ic.index.year).mean(),
                       "양수일%": ic.groupby(ic.index.year).apply(lambda x: (x > 0).mean() * 100),
                       "관측일": ic.groupby(ic.index.year).size()})
    print(yr.round(4).to_string())


SKIP = {"Date", "Ticker", "fwd", "fwd_rel", "Open", "High", "Low", "Close",
        "Volume", "Amount", "Name", "Market", "Wyckoff_Label",
        "Wyckoff_Signal_Desc", "Wyckoff_Phase", "Wyckoff_Undecidable",
        "Wyckoff_Phase_Provisional", "Weis_Dir"}


def cmd_screen(panel: pd.DataFrame) -> None:
    cols = [c for c in panel.columns
            if c not in SKIP and pd.api.types.is_numeric_dtype(panel[c])]
    print(f"후보 {len(cols)}개 스크리닝 중...\n")
    rows = [ic_stat(panel, panel[c], c) for c in cols]
    df = pd.DataFrame(rows).dropna(subset=["평균IC"])
    df["|IC|"] = df["평균IC"].abs()
    df = df.sort_values("|IC|", ascending=False)

    sig = df[(df["p"] < 0.01) & df["연도부호"].isin(["4/4", "0/4"])]
    print("=== 유의(p<0.01) + 연도 부호 4/4 일관 ===")
    print(sig.drop(columns="|IC|").round(4).to_string(index=False))

    print("\n=== 전체 상위 25 (|IC| 순) ===")
    print(df.head(25).drop(columns="|IC|").round(4).to_string(index=False))

    if len(sig) > 1:
        print("\n=== 통과 지표끼리 횡단면 상관 (52주 위치와의 중복도) ===")
        keep = list(sig["지표"])[:12]
        m = pd.DataFrame(index=keep, columns=keep, dtype=float)
        for a in keep:
            for b in keep:
                if a == b:
                    m.loc[a, b] = 1.0
                    continue
                d = panel[["Date", a, b]].dropna()
                m.loc[a, b] = d.groupby("Date").apply(
                    lambda g, x=a, y=b: g[x].corr(g[y], method="spearman"),
                    include_groups=False).mean()
        print(m.round(2).to_string())


# ===========================================================================
# 결합 — 날짜별 백분위 가중평균 (스케일 무관)
# ===========================================================================

def rank_pct(panel: pd.DataFrame, col: str) -> pd.Series:
    return panel.groupby("Date")[col].rank(pct=True)


def blend(panel: pd.DataFrame, spec: dict[str, float]) -> pd.Series:
    """spec = {컬럼: 가중치}. 각 축을 날짜별 백분위로 바꿔 가중평균."""
    acc, wsum = None, 0.0
    for col, w in spec.items():
        r = rank_pct(panel, col) * w
        acc = r if acc is None else acc.add(r, fill_value=np.nan)
        wsum += w
    return acc / wsum


def top_q_return(panel: pd.DataFrame, s: pd.Series, q: float = 0.03) -> dict:
    d = pd.DataFrame({"Date": panel["Date"], "s": s,
                      "r": panel["fwd_rel"]}).dropna()
    thr = d.groupby("Date")["s"].transform(lambda x: x.quantile(1 - q))
    sel = d[d["s"] >= thr]
    yr = sel.groupby(sel["Date"].dt.year)["r"].mean() * 100
    return {"상위3%": sel["r"].mean() * 100,
            "상위3% 연도부호": f"{int((yr > 0).sum())}/{len(yr)}"}


def cmd_combine(panel: pd.DataFrame) -> None:
    cands = {
        "① 낙폭 단독": {"dd_from_hi": 1},
        "② 52주위치 단독 (현행 축)": {"pos252": 1},
        "③ 낙폭 + Chaikin": {"dd_from_hi": 1, "Chaikin_Osc": 1},
        "④ 낙폭 + Weis": {"dd_from_hi": 1, "Weis_Days": 1},
        "⑤ 낙폭 + Chaikin + Weis": {"dd_from_hi": 1, "Chaikin_Osc": 1, "Weis_Days": 1},
        "⑥ ⑤ + 2년위치": {"dd_from_hi": 1, "Chaikin_Osc": 1, "Weis_Days": 1, "pos504": 1},
        "⑦ ⑤ + RSI": {"dd_from_hi": 1, "Chaikin_Osc": 1, "Weis_Days": 1, "RSI": 1},
        "⑧ 낙폭2 + Chaikin1 + Weis1": {"dd_from_hi": 2, "Chaikin_Osc": 1, "Weis_Days": 1},
        "⑨ 낙폭3 + Chaikin2 + Weis1": {"dd_from_hi": 3, "Chaikin_Osc": 2, "Weis_Days": 1},
    }
    rows = []
    for name, spec in cands.items():
        s = blend(panel, spec)
        rows.append({**ic_stat(panel, s, name), **top_q_return(panel, s)})
    print("=== 결합 후보 — IC 와 상위선택 ===")
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\n=== 채택 후보 10분위별 20일 상대수익(%) ===")
    dec = {}
    for name in ("① 낙폭 단독", "⑤ 낙폭 + Chaikin + Weis", "② 52주위치 단독 (현행 축)"):
        s = blend(panel, cands[name])
        d = pd.DataFrame({"Date": panel["Date"], "s": s,
                          "y": panel["fwd_rel"]}).dropna()
        q = d.groupby("Date")["s"].transform(
            lambda x: pd.qcut(x.rank(method="first"), 10, labels=False))
        dec[name] = d.groupby(q)["y"].mean() * 100
    t = pd.DataFrame(dec)
    t.index = [f"{i+1}분위" for i in t.index]
    print(t.round(3).to_string())

    print("\n=== 연도별 IC ===")
    out = {}
    for name in ("① 낙폭 단독", "⑤ 낙폭 + Chaikin + Weis", "② 52주위치 단독 (현행 축)"):
        ic = ic_series(panel, blend(panel, cands[name]))
        out[name] = ic.groupby(ic.index.year).mean()
    print(pd.DataFrame(out).round(4).to_string())


def cmd_ops(panel: pd.DataFrame) -> None:
    """실운용 비교 — IC 우위 설계 vs 상위선택 우위 설계."""
    from scripts.validate_area_redesign import slot_sim

    rsq = rank_pct(panel, "RSI")
    p52q = rank_pct(panel, "pos252")
    specs = {
        "① 낙폭 단독": blend(panel, {"dd_from_hi": 1}),
        "⑤ 낙폭+Chaikin+Weis": blend(panel, {"dd_from_hi": 1, "Chaikin_Osc": 1, "Weis_Days": 1}),
        "⑨ 낙폭3+Chaikin2+Weis1": blend(panel, {"dd_from_hi": 3, "Chaikin_Osc": 2, "Weis_Days": 1}),
        "⑦ ⑤+RSI": blend(panel, {"dd_from_hi": 1, "Chaikin_Osc": 1, "Weis_Days": 1, "RSI": 1}),
        "(비교) RSI80+52주20": 0.8 * rsq + 0.2 * p52q,
        "(비교) 52주 단독": p52q,
    }
    rows = []
    for name, s in specs.items():
        st = ic_stat(panel, s, name)
        for q in (0.03, 0.10):
            pass
        r3 = top_q_return(panel, s, 0.03)
        sim = slot_sim(panel, s, s.quantile(0.97))
        rows.append({"설계": name, "평균IC": st["평균IC"], "IC 부호": st["연도부호"],
                     "상위3%": r3["상위3%"], "누적%": sim["누적%"],
                     "CAGR%": sim["CAGR%"], "MDD%": sim["MDD%"],
                     "Sharpe": sim["Sharpe"], "현금%": sim["현금비중%"]})
    print("=== 슬롯 10 · t+1 시가 · 왕복 0.43% · 20거래일 보유 ===")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n=== 슬롯을 넓게 쓰면 (IC 설계는 다수 보유에 맞는 성격) ===")
    rows = []
    for name in ("⑨ 낙폭3+Chaikin2+Weis1", "(비교) RSI80+52주20"):
        s = specs[name]
        for n_slot, q in ((10, 0.97), (20, 0.90), (30, 0.85)):
            r = slot_sim(panel, s, s.quantile(q), n_slot=n_slot)
            rows.append({"설계": name, "슬롯": n_slot, "임계분위": f"상위{(1-q)*100:.0f}%",
                         "누적%": r["누적%"], "MDD%": r["MDD%"],
                         "Sharpe": r["Sharpe"], "현금%": r["현금비중%"]})
    print(pd.DataFrame(rows).round(2).to_string(index=False))


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "axis"
    if cmd == "build":
        build()
        return
    panel = get_panel()
    {"axis": cmd_axis, "screen": cmd_screen, "combine": cmd_combine,
     "ops": cmd_ops}[cmd](panel)


if __name__ == "__main__":
    main()
