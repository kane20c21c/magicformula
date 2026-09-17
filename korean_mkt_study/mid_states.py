# -*- coding: utf-8 -*-
"""mid_states.py — 중기 모델 상황 사다리 (A층 2014-01~2026-06, 관문 mid_gate 재사용)

로직: 조건을 평균내지 않고 **상황(state)** 으로 이산화 → 상황마다 발생 건수·상단/하단 먼저 확률·평균 경로수익·
     기저 대비 lift 를 찍고 흔한 상황 → 드문 상황 순으로 사다리를 세운다. 그 다음 흔한 조건부터 하나씩 더해가며
     기대값이 튀는 경로를 찾는다(손으로 타는 결정 트리).

라벨: mid_gate 트리플 배리어, 주 판정 N=10 k=1.0 (N=5/20 도 저장).
원하는 상태(기본): p_up ≥ 0.45 · p_dn ≤ 0.20 · mean r_path ≥ +1.0%  (기저 all N=10: 0.30 / 0.26 / 0.00)

상황 축 (첫 회차 12개 + MA 기간 변형):
  시장   mkt_{60,120,200}: 시총가중 유니버스 지수가 MA_P 위/아래 × MA_P 20일 기울기 ↑/↓ → 4국면   ← 케인 변형
         mkt_ret20: 시장 20일 수익률 고정구간 (<−8 / −8~−3 / −3~3 / 3~8 / >8 %)
  추세   stk_{60,120,200}: 종목 종가가 SMA_P 위/아래 × SMA_P 20일 기울기 ↑/↓ → 4상한        ← 케인 변형
         align: SMA 정배열 단계 0~4,   pos52: 52주 고가 대비 (신고가권 ≥−5% / 중간 / 바닥권 ≤−30%)
  단기   gap5: 종가/SMA5−1 고정구간 (<−5 / −5~−2 / −2~2 / 2~5 / >5 %)
         dd20: 20일 고점 대비 낙폭 (0~−3 / −3~−7 / −7~−12 / −12~−20 / <−20 %)
  구조   boxpos: 20일 고저 범위 안 종가 위치 5구간
  거래량 volq: 당일/20일평균 (<0.5 / 0.5~1 / 1~2 / 2~3 / >3),  udv: 20일 상승일 거래량합/하락일 거래량합 (<0.7 / 0.7~1 / 1~1.4 / >1.4)
  변동성 volexp: vol20/vol60 (<0.8 / 0.8~1.2 / >1.2)
  수급   flow4: 외인 20일 누적 부호 × 기관 20일 누적 부호
  규모   tier: bigf / big_only / small (배타 3계층)

검증 규칙: 셀 최소 MIN_CELLS 건 & MIN_DAYS 일, 전반(2015~2020)에서 찾고 후반(2021~2026-06)에서 확인.

실행: python3 mid_states.py [--quick]
산출: out/mid/states.parquet (날짜·종목·라벨·상태 롱 테이블 — 케인이 직접 자를 수 있게)
      out/mid/ladder_single.csv · ladder_ma_variation.csv · ladder_pairs.csv · ladder_drill.csv · ladder_summary.md
"""
import argparse, time, itertools
from pathlib import Path
import numpy as np, pandas as pd
import mid_gate as g

OUT = g.OUT
MIN_CELLS, MIN_DAYS = 3000, 100
SPLIT_YEAR = 2021                      # 전반 < 2021 ≤ 후반
TARGET = dict(p_up=0.45, p_dn=0.20, mean_rpath=0.010)
MA_REF = [60, 120, 200]
AXES = ["tier", "mkt_200", "mkt_ret20", "stk_200", "align", "pos52", "gap5", "dd20",
        "boxpos", "volq", "udv", "volexp", "flow4"]           # 사다리·드릴 기본 축 (MA200 기준)


def cut(x, edges, labels):
    return pd.cut(x, [-np.inf] + list(edges) + [np.inf], labels=labels).astype(str)


def cutw(wide, edges, labels):
    """wide DataFrame → 같은 shape 의 문자열 ndarray (NaN → 'nan')."""
    a = wide.to_numpy().ravel()
    return cut(pd.Series(a), edges, labels).to_numpy().reshape(wide.shape)


def quad(price, ma, slope):
    """위/아래 × 기울기↑/↓ → 4상한 문자열 (NaN 은 'nan')."""
    up = price > ma
    rise = slope > 0
    q = np.where(up & rise, "위+상승", np.where(up & ~rise, "위+하락", np.where(~up & rise, "아래+상승", "아래+하락")))
    q = np.where(np.isnan(ma) | np.isnan(slope) | np.isnan(price), "nan", q)
    return q


# ───────────────────────── 상태 테이블 ─────────────────────────
def build_states(quick):
    t0 = time.time()
    W, mktcap, foreign, managed, static, sector, fw = g.load(quick)
    C, V, H, L = W["close"], W["volume"], W["high"], W["low"]
    dates = C.index
    masks_df = g.build_masks(W, mktcap, foreign, managed, static)
    base = masks_df["all"]
    common = C.rolling(200, min_periods=200).mean().notna() & fw["외국인"].rolling(60, min_periods=60).sum().notna()
    ev = pd.Series(dates >= ("2023-01-01" if quick else g.EVAL_START), index=dates)
    mask = (base & common & g.bcast_rows(ev, C)).to_numpy()
    g.log(f"wide {C.shape}")

    # 시장 지수: 시총가중 일별 수익률 (all 유니버스, 전월말 시총)
    r = g.pc(C)
    wgt = mktcap.where(base & r.notna())
    mkt_ret = (r * wgt).sum(1) / wgt.sum(1)
    idx = (1 + mkt_ret.fillna(0)).cumprod()
    S = {}                                                 # 축 → wide(ndarray of str) 또는 1-D(날짜) Series
    for P in MA_REF:
        m = idx.rolling(P, min_periods=P).mean()
        S[f"mkt_{P}"] = pd.Series(quad(idx.to_numpy(), m.to_numpy(), (m / m.shift(20) - 1).to_numpy()), index=dates)
    S["mkt_ret20"] = cut(idx / idx.shift(20) - 1, [-0.08, -0.03, 0.03, 0.08], ["급락<-8", "약세-8~-3", "보합-3~3", "강세3~8", "급등>8"])

    # 종목 추세
    sma = {p: g.ma("sma", C, V, p) for p in [5, 20, 60, 120, 200]}
    for P in MA_REF:
        S[f"stk_{P}"] = quad(C.to_numpy(), sma[P].to_numpy(), (sma[P] / sma[P].shift(20) - 1).to_numpy())
    al = None
    for a, b in zip([5, 20, 60, 120], [20, 60, 120, 200]):
        term = (sma[a] > sma[b]).astype(float).where(sma[a].notna() & sma[b].notna())
        al = term if al is None else al + term
    S["align"] = al.to_numpy()
    hi52 = C.rolling(252, min_periods=120).max()
    S["pos52"] = cutw(C / hi52 - 1, [-0.30, -0.05], ["바닥권≤-30", "중간", "신고가권≥-5"])
    # 단기
    S["gap5"] = cutw(C / sma[5] - 1, [-0.05, -0.02, 0.02, 0.05], ["<-5", "-5~-2", "-2~2", "2~5", ">5"])
    hi20, lo20 = H.rolling(20, min_periods=20).max(), L.rolling(20, min_periods=20).min()
    S["dd20"] = cutw(C / hi20 - 1, [-0.20, -0.12, -0.07, -0.03], ["<-20", "-20~-12", "-12~-7", "-7~-3", "-3~0"])
    S["boxpos"] = cutw((C - lo20) / (hi20 - lo20), [0.2, 0.4, 0.6, 0.8], ["0-.2", ".2-.4", ".4-.6", ".6-.8", ".8-1"])
    # 거래량
    v20 = V.rolling(20, min_periods=10).mean()
    S["volq"] = cutw(V / v20, [0.5, 1.0, 2.0, 3.0], ["<.5", ".5-1", "1-2", "2-3", ">3"])
    upv = V.where(r > 0, 0.0).rolling(20, min_periods=20).sum()
    dnv = V.where(r < 0, 0.0).rolling(20, min_periods=20).sum()
    S["udv"] = cutw(upv / dnv.replace(0, np.nan), [0.7, 1.0, 1.4], ["<.7", ".7-1", "1-1.4", ">1.4"])
    # 변동성
    vol20 = r.rolling(20, min_periods=10).std(); vol60 = r.rolling(60, min_periods=30).std()
    S["volexp"] = cutw(vol20 / vol60, [0.8, 1.2], ["수축<.8", "보합", "확장>1.2"])
    # 수급
    fc = np.sign(fw["외국인"].rolling(20, min_periods=20).sum()); ic = np.sign(fw["기관"].rolling(20, min_periods=20).sum())
    f4 = np.where((fc > 0) & (ic > 0), "외+기+", np.where((fc > 0) & (ic <= 0), "외+기-", np.where((fc <= 0) & (ic > 0), "외-기+", "외-기-")))
    S["flow4"] = np.where(fc.isna() | ic.isna(), "nan", f4)
    # 규모
    big, bigf = masks_df["big"].to_numpy(), masks_df["bigf"].to_numpy()
    S["tier"] = np.where(bigf, "bigf", np.where(big, "big_only", "small"))
    g.log(f"states 계산 ({time.time()-t0:.0f}s)")

    labels = {N: g.build_labels(W, N, 1.0) for N in [5, 10, 20]}
    valid = mask & ~np.isnan(labels[10]["lab"])
    ii, jj = np.where(valid)
    df = pd.DataFrame({"date": dates[ii], "ticker": C.columns[jj]})
    for N, lb in labels.items():
        df[f"lab{N}"] = lb["lab"][ii, jj]; df[f"rpath{N}"] = lb["r_path"][ii, jj]; df[f"rraw{N}"] = lb["r_raw"][ii, jj]
    for k, v in S.items():
        df[k] = v.to_numpy()[ii] if isinstance(v, pd.Series) else v[ii, jj]
    df["year"] = df.date.dt.year
    df["half"] = np.where(df.year < SPLIT_YEAR, "A", "B")
    g.log(f"long table {df.shape} ({time.time()-t0:.0f}s)")
    return df


# ───────────────────────── 통계 ─────────────────────────
def stats(sub, N=10):
    lab, rp = sub[f"lab{N}"], sub[f"rpath{N}"]
    p_up, p_dn = (lab == 1).mean(), (lab == -1).mean()
    return dict(n=len(sub), n_days=sub.date.nunique(), p_up=p_up, p_dn=p_dn, spread=p_up - p_dn,
                p_vert=(lab == 0).mean(), mean_rpath=rp.mean(), mean_rraw=sub[f"rraw{N}"].mean())


def cell_table(df, axes, base, N=10):
    """axes 조합의 모든 셀 — 전체·전반·후반 통계 + lift. 빈도순 정렬."""
    rows = []
    for key, sub in df.groupby(axes, observed=True, dropna=True):
        key = key if isinstance(key, tuple) else (key,)
        if any(str(k) == "nan" for k in key):
            continue
        d = dict(zip(axes, key)); d.update(stats(sub, N))
        for h in ["A", "B"]:
            s = sub[sub.half == h]
            st = stats(s, N) if len(s) else {}
            d.update({f"{k}_{h}": st.get(k, np.nan) for k in ["n", "p_up", "p_dn", "mean_rpath"]})
        d["share"] = d["n"] / len(df)
        d["lift_up"] = d["p_up"] / base["p_up"]; d["lift_dn"] = d["p_dn"] / base["p_dn"]
        d["spread"] = d["p_up"] - d["p_dn"]
        d["ok_size"] = (d["n"] >= MIN_CELLS) and (d["n_days"] >= MIN_DAYS)
        d["consistent"] = np.sign(d["mean_rpath_A"]) == np.sign(d["mean_rpath_B"]) if pd.notna(d.get("mean_rpath_A")) else False
        d["hit_target"] = (d["p_up"] >= TARGET["p_up"]) and (d["p_dn"] <= TARGET["p_dn"]) and (d["mean_rpath"] >= TARGET["mean_rpath"])
        rows.append(d)
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def drill(df, axes, base, depth=4, N=10, objective="mean_rpath"):
    """흔한 조건부터 하나씩 추가 — 전반(A)에서 objective 최대 셀을 고르고 후반(B)으로 확인. 크기·일수 조건은 A 안에서."""
    path, cur = [], df
    used = []
    for _ in range(depth):
        best = None
        for ax in [a for a in axes if a not in used]:
            for v, sub in cur.groupby(ax, observed=True, dropna=True):
                if str(v) == "nan":
                    continue
                A = sub[sub.half == "A"]
                if len(A) < MIN_CELLS or A.date.nunique() < MIN_DAYS:
                    continue
                sA = stats(A, N)
                if best is None or sA[objective] > best[2][objective]:
                    best = (ax, v, sA, sub)
        if best is None:
            break
        ax, v, sA, sub = best
        sB = stats(sub[sub.half == "B"], N) if (sub.half == "B").any() else {}
        sAll = stats(sub, N)
        path.append(dict(depth=len(path) + 1, axis=ax, state=v, n=sAll["n"], share=sAll["n"] / len(df),
                         p_up=sAll["p_up"], p_dn=sAll["p_dn"], mean_rpath=sAll["mean_rpath"],
                         p_up_A=sA["p_up"], p_dn_A=sA["p_dn"], mean_rpath_A=sA["mean_rpath"], n_A=sA["n"],
                         p_up_B=sB.get("p_up", np.nan), p_dn_B=sB.get("p_dn", np.nan), mean_rpath_B=sB.get("mean_rpath", np.nan), n_B=sB.get("n", 0)))
        used.append(ax); cur = sub
    return pd.DataFrame(path)


def _fmt(df, cols):
    f = {"share": "{:.1%}", "p_up": "{:.3f}", "p_dn": "{:.3f}", "p_vert": "{:.3f}", "mean_rpath": "{:+.4f}", "mean_rraw": "{:+.4f}",
         "lift_up": "{:.2f}", "lift_dn": "{:.2f}", "spread": "{:+.3f}", "n": "{:,}", "n_days": "{:,}",
         "p_up_A": "{:.3f}", "p_dn_A": "{:.3f}", "mean_rpath_A": "{:+.4f}", "p_up_B": "{:.3f}", "p_dn_B": "{:.3f}", "mean_rpath_B": "{:+.4f}",
         "n_A": "{:,}", "n_B": "{:,}"}
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join((f.get(c, "{}").format(r[c]) if pd.notna(r[c]) else "—") for c in cols) + " |")
    return "\n".join(lines)


# ───────────────────────── 메인 ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="states.parquet 있어도 다시 계산")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pq = OUT / ("states_quick.parquet" if a.quick else "states.parquet")
    if pq.exists() and not a.rebuild:
        df = pd.read_parquet(pq); g.log(f"load {pq.name} {df.shape}")
    else:
        df = build_states(a.quick); df.to_parquet(pq, index=False); g.log(f"저장 {pq.name}")

    base = stats(df)
    all_axes = ["tier"] + [f"mkt_{P}" for P in MA_REF] + ["mkt_ret20"] + [f"stk_{P}" for P in MA_REF] + \
               ["align", "pos52", "gap5", "dd20", "boxpos", "volq", "udv", "volexp", "flow4"]
    single = pd.concat([cell_table(df, [ax], base).assign(axis=ax, state=lambda d: d[ax].astype(str)).drop(columns=[ax])
                        for ax in all_axes], ignore_index=True)
    cols = ["axis", "state", "n", "share", "n_days", "p_up", "p_dn", "p_vert", "mean_rpath", "lift_up", "lift_dn", "spread",
            "p_up_A", "p_dn_A", "mean_rpath_A", "p_up_B", "p_dn_B", "mean_rpath_B", "ok_size", "consistent", "hit_target"]
    single = single[cols]
    single.to_csv(OUT / "ladder_single.csv", index=False)

    # MA 기간 변형 비교: 시장 4국면 × 종목 4상한, P ∈ {60,120,200}
    var_rows = []
    for P in MA_REF:
        t = cell_table(df, [f"mkt_{P}", f"stk_{P}"], base).rename(columns={f"mkt_{P}": "mkt", f"stk_{P}": "stk"})
        var_rows.append(t.assign(P=P))
    var = pd.concat(var_rows, ignore_index=True)
    var.to_csv(OUT / "ladder_ma_variation.csv", index=False)
    g.log(f"single·variation ({time.time()-t0:.0f}s)")

    # 2축 조합 전수 (기본 축)
    pairs = pd.concat([cell_table(df, list(c), base).rename(columns={c[0]: "s1", c[1]: "s2"}).assign(ax1=c[0], ax2=c[1])
                       for c in itertools.combinations(AXES, 2)], ignore_index=True)
    pairs = pairs[pairs.ok_size].sort_values("mean_rpath_A", ascending=False)
    pairs.to_csv(OUT / "ladder_pairs.csv", index=False)
    g.log(f"pairs {len(pairs)} cells ({time.time()-t0:.0f}s)")

    # 드릴다운
    drills = []
    for obj in ["mean_rpath", "spread"]:
        drills.append(drill(df, AXES, base, objective=obj).assign(scope="all", objective=obj))
        for tier in ["bigf", "big_only", "small"]:
            sub = df[df.tier == tier]
            drills.append(drill(sub, [x for x in AXES if x != "tier"], base, objective=obj).assign(scope=tier, objective=obj))
    dr = pd.concat(drills, ignore_index=True)
    dr.to_csv(OUT / "ladder_drill.csv", index=False)
    (OUT / "ladder_summary.md").write_text(summary_md(df, base, single, var, pairs, dr, a.quick), encoding="utf-8")
    g.log(f"완료 → {OUT}  ({time.time()-t0:.0f}s)")


def summary_md(df, base, single, var, pairs, dr, quick):
    o = [f"# 중기 모델 상황 사다리 — A층 {'(quick)' if quick else ''}",
         f"{df.date.min().date()}~{df.date.max().date()} · {len(df):,}셀 · N=10 k=1.0 · 전반 A = <{SPLIT_YEAR}, 후반 B = ≥{SPLIT_YEAR}",
         f"기저: p_up {base['p_up']:.3f} · p_dn {base['p_dn']:.3f} · 만기 {base['p_vert']:.3f} · r_path {base['mean_rpath']:+.4f}",
         f"원하는 상태: p_up ≥ {TARGET['p_up']} · p_dn ≤ {TARGET['p_dn']} · r_path ≥ {TARGET['mean_rpath']:+.3f}  (셀 최소 {MIN_CELLS:,}건·{MIN_DAYS}일)", ""]
    o.append("## 1. 단일 축 사다리 (축별, 빈도순)")
    for ax, t in single.groupby("axis", sort=False):
        o.append(f"### {ax}")
        o.append(_fmt(t, ["state", "n", "share", "p_up", "p_dn", "mean_rpath", "lift_up", "lift_dn", "p_up_A", "mean_rpath_A", "p_up_B", "mean_rpath_B", "consistent"]))
        o.append("")
    o.append("## 2. MA 기간 변형 — 시장 4국면 × 종목 4상한 (P = 60 / 120 / 200)")
    for P in MA_REF:
        t = var[var.P == P].sort_values("n", ascending=False)
        o.append(f"### P = {P}")
        o.append(_fmt(t, ["mkt", "stk", "n", "share", "p_up", "p_dn", "mean_rpath", "spread", "mean_rpath_A", "mean_rpath_B", "consistent"]))
        o.append("")
    o.append("## 3. 2축 조합 — 전반(A) r_path 상위 20 (크기 조건 충족 셀만) · 후반(B)으로 확인")
    o.append(_fmt(pairs.head(20), ["ax1", "s1", "ax2", "s2", "n", "share", "p_up", "p_dn", "mean_rpath", "mean_rpath_A", "mean_rpath_B", "p_up_B", "p_dn_B", "consistent", "hit_target"]))
    hit = single[single.hit_target & single.ok_size]
    hp = pairs[pairs.hit_target]
    o.append(f"\n원하는 상태 도달: 단일 축 {len(hit)}셀 · 2축 {len(hp)}셀")
    if len(hp):
        o.append(_fmt(hp.head(20), ["ax1", "s1", "ax2", "s2", "n", "p_up", "p_dn", "mean_rpath", "mean_rpath_A", "mean_rpath_B"]))
    o.append("\n## 4. 드릴다운 — 전반(A)에서 고르고 후반(B)으로 확인")
    for (scope, obj), t in dr.groupby(["scope", "objective"], sort=False):
        o.append(f"### scope={scope} · objective={obj}")
        o.append(_fmt(t, ["depth", "axis", "state", "n", "share", "p_up", "p_dn", "mean_rpath", "mean_rpath_A", "mean_rpath_B", "p_up_B", "p_dn_B"]))
        o.append("")
    return "\n".join(o)


if __name__ == "__main__":
    main()
