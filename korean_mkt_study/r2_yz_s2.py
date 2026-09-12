"""r2_yz_s2.py — S2: 창 스윕. 대형주(4조↑ 주축 / 운영 유니버스 확인) 눌림목 onset 에서
YZR(단기/장기 창)·R²(창)·시장정규화 YZR 의 상위−하위 3분위 성과 차이를 잰다.

평가 창 (2026-09-12 Kane): H=5·10 주축, 20 보조. 판정 = 기간 A(2014-21)/B(2022-26) 부호 일관 + 이웃 고원.
신호 정의는 MA200/HW40 고정 (S1 에서 MA150/250·HW60 과 차이 없음).

입력: out/r2yz/events_200_40.parquet (r2_yz_events.py) + data/prices.parquet (특징 재계산)
출력: out/r2yz/s2_sweep.xlsx, 표준출력 요약
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from backtest import DATA, _yang_zhang                      # noqa: E402
from r2_yz_events import rolling_r2_slope                    # noqa: E402

OUT = HERE / "out" / "r2yz"
SPLIT = pd.Timestamp("2022-01-01")
HS = (5, 10, 15, 20)
YZ_SHORT = (5, 10, 20)
YZ_LONG = (60, 120, 250)
R2_WIN = (60, 100, 120, 150, 200)
pd.set_option("display.width", 220, "display.max_columns", 60, "display.float_format", "{:.3f}".format)


def build_features(tickers_needed: pd.Index) -> pd.DataFrame:
    px = pd.read_parquet(DATA / "prices.parquet")
    d = px.index.get_level_values("date")
    px = px[(d >= pd.Timestamp("2012-01-01")) & (d <= pd.Timestamp("2026-06-30"))]
    wide = {c: px[c].astype("float64").unstack("ticker").sort_index() for c in ("open", "high", "low", "close")}
    del px
    for c in wide:
        wide[c] = wide[c].where(wide[c] > 0)
    op, hi, lo, cl = (wide[k] for k in ("open", "high", "low", "close"))

    # 시장 횡단면 정규화용: 전 종목 YZ 창 — 일별 중앙값
    yz_all = {n: _yang_zhang(op, hi, lo, cl, n) for n in sorted(set(YZ_SHORT) | set(YZ_LONG))}
    mkt_med = {}
    for s in YZ_SHORT:
        for l in YZ_LONG:
            mkt_med[f"YZR_{s}_{l}"] = (yz_all[s] / yz_all[l]).median(axis=1)   # 그날 시장 전체 YZR 중앙값

    cols = wide["close"].columns.intersection(tickers_needed)
    feat = {}
    for s in YZ_SHORT:
        for l in YZ_LONG:
            k = f"YZR_{s}_{l}"
            r = yz_all[s][cols] / yz_all[l][cols]
            feat[k] = r
            feat[k + "_rel"] = r.div(mkt_med[k], axis=0)     # 시장 대비 상대 YZR
    ly = np.log(cl[cols])
    for w in R2_WIN:
        r2, sl = rolling_r2_slope(ly, w)
        feat[f"R2_{w}"] = r2
    for n in (20, 60):
        feat[f"YZ_{n}"] = yz_all[n][cols]
    long = {k: v.stack(dropna=False) for k, v in feat.items()}
    df = pd.DataFrame(long)
    df.index.names = ["date", "ticker"]
    return df.reset_index()


def hl(g: pd.DataFrame, H: int) -> tuple[float, float]:
    hi, lo = g[g.q == "H"][f"RET_{H}"], g[g.q == "L"][f"RET_{H}"]
    return (hi > 0).mean() - (lo > 0).mean(), hi.mean() - lo.mean()


def sweep(ev: pd.DataFrame, univ: str, vars_: list[str]) -> pd.DataFrame:
    rows = []
    for col in vars_:
        q = ev.dropna(subset=[col]).copy()
        if len(q) < 300:
            continue
        q["q"] = pd.qcut(q[col], 3, labels=["L", "M", "H"])
        r = dict(univ=univ, var=col, n=len(q))
        for H in HS:
            w, m = hl(q, H); wA, mA = hl(q[q.A], H); wB, mB = hl(q[~q.A], H)
            r.update({f"win{H}": w, f"win{H}_A": wA, f"win{H}_B": wB,
                      f"mean{H}": m, f"mean{H}_A": mA, f"mean{H}_B": mB,
                      f"consist{H}": int(np.sign(wA) == np.sign(wB) == np.sign(w))})
        # 종합 점수: 주축(5·10) 승률 H−L 평균, 두 기간 모두 같은 부호일 때만 인정
        r["score"] = np.mean([r["win5"], r["win10"]]) if (r["consist5"] and r["consist10"]) else 0.0
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ev = pd.read_parquet(OUT / "events_200_40.parquet")
    ev = ev[ev.MKTCAP >= 4e12].copy()
    for H in (5, 10, 15, 20, 40):
        ev.loc[~np.isfinite(ev[f"RET_{H}"]), f"RET_{H}"] = np.nan
    ev["A"] = ev.date < SPLIT
    feat = build_features(pd.Index(ev.ticker.unique()))
    keep = [c for c in ev.columns if not (c.startswith("R2_") or c.startswith("YZ") or c.startswith("SLOPE"))]
    ev = ev[keep].merge(feat, on=["date", "ticker"], how="left")
    vars_ = ([f"YZR_{s}_{l}" for s in YZ_SHORT for l in YZ_LONG]
             + [f"YZR_{s}_{l}_rel" for s in YZ_SHORT for l in YZ_LONG]
             + [f"R2_{w}" for w in R2_WIN] + ["YZ_20", "YZ_60"])
    res = pd.concat([sweep(ev, "4조↑", vars_), sweep(ev[ev.ELIG == 1], "운영", vars_)], ignore_index=True)
    show = ["univ", "var", "n", "win5", "win5_A", "win5_B", "win10", "win10_A", "win10_B", "win20", "win20_A", "win20_B",
            "mean5", "mean10", "mean20", "consist5", "consist10", "consist20", "score"]
    for u in ("4조↑", "운영"):
        t = res[res.univ == u].set_index("var")[show[2:]]
        print(f"\n########## {u}  (n={int(res[res.univ==u].n.max()):,})  H−L 승률차·평균차, A=2014-21 B=2022-26 ##########")
        print(t.sort_values("score", ascending=False))
    with pd.ExcelWriter(OUT / "s2_sweep.xlsx") as xw:
        res.to_excel(xw, sheet_name="sweep", index=False)
        ev.to_parquet(OUT / "events_200_40_large_s2feat.parquet", index=False)
    print("\nsaved", OUT / "s2_sweep.xlsx")


if __name__ == "__main__":
    main()
