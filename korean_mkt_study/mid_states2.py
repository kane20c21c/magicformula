# -*- coding: utf-8 -*-
"""mid_states2.py — 상황 사다리 2회차: 순서(sequence) 축 + 시장 200×60 이중 국면 (1회차 states.parquet 위에 열 추가)

1회차에서 배운 것:
  · 시장 축은 MA200(대국면), 종목 축은 MA60(빠른 상태)이 맞다. 시장 "아래+상승"은 P 에 따라 조정 초입/회복 초입이 섞인다 → 순서 축으로 가른다.
  · MA200 4국면은 2026-07~08 같은 "고공 조정"(200일선 위에서 −10%)을 못 본다 → 시장 60 을 중간 국면으로 같이 쓴다.
  · 원하는 상태(급락 후, 회복 초입 스프링)는 드물고 시간에 몰린다 → "며칠째"가 진입 타이밍을 정한다.

추가 축:
  mkt_turn_days   시장 MA200 20일 기울기가 ≤0 → >0 으로 바뀐 뒤 며칠째: 전환전 / 1~5 / 6~20 / 21~60 / 60+
  crash_days      시장 20일 수익률 ≤ −8% 가 마지막으로 찍힌 뒤 며칠째: 당일 / 1~5 / 6~20 / 21~60 / 60+·없음
  stk_bounce_days 종목 종가가 20일 최저를 갱신한 뒤 며칠째: 당일 / 1~3 / 4~10 / 11+
  (기존) tier · mkt_200 · mkt_60 · mkt_ret20 · stk_60 · stk_200 · gap5 · dd20 · pos52 · boxpos · volq · udv
  제외: volexp(라벨 부산물), flow4(6/30 이후 결측·무정보), align(관문에서 무정보)

산출: out/mid/states2.parquet · ladder2_single.csv · ladder2_pairs.csv · ladder2_drill.csv · ladder2_focus.csv · ladder2_summary.md
실행: python3 mid_states2.py  (states.parquet 필요 — mid_states.py 먼저)
"""
import time, itertools
import numpy as np, pandas as pd
import mid_gate as g
import mid_states as s1

OUT = g.OUT
AXES2 = ["tier", "mkt_200", "mkt_60", "mkt_ret20", "mkt_turn_days", "crash_days",
         "stk_60", "stk_200", "gap5", "dd20", "stk_bounce_days", "pos52", "boxpos", "volq", "udv"]


def days_since(event, valid=None):
    """1-D bool → 마지막 True 이후 경과일(당일 0). valid 가 주어지면 False 구간은 NaN."""
    ar = np.arange(len(event), dtype=float)
    last = pd.Series(np.where(event, ar, np.nan)).ffill().to_numpy()
    d = ar - last
    if valid is not None:
        d = np.where(valid, d, np.nan)
    return d


def add_axes(df):
    t0 = time.time()
    W, mktcap, foreign, managed, static, sector, fw = g.load(False)
    C = W["close"]; dates = C.index
    base = g.build_masks(W, mktcap, foreign, managed, static)["all"]
    r = g.pc(C)
    wgt = mktcap.where(base & r.notna())
    idx = (1 + ((r * wgt).sum(1) / wgt.sum(1)).fillna(0)).cumprod()
    ma200 = idx.rolling(200, min_periods=200).mean()
    slope = (ma200 / ma200.shift(20) - 1)
    pos = (slope > 0).to_numpy()
    turn = pos & ~np.roll(pos, 1); turn[0] = False
    td = days_since(turn, valid=pos) + 1                       # 전환 당일 = 1일째
    mkt_turn = pd.Series(np.where(~pos, "전환전", s1.cut(pd.Series(td), [5, 20, 60], ["1~5", "6~20", "21~60", "60+"]).to_numpy()), index=dates)
    mkt_turn = mkt_turn.where(slope.notna(), "nan")
    ret20 = idx / idx.shift(20) - 1
    crash = (ret20 <= -0.08).to_numpy()
    cd = days_since(crash)                                     # 첫 급락 전 구간은 NaN
    crash_days = s1.cut(pd.Series(cd), [0, 5, 20, 60], ["당일", "1~5", "6~20", "21~60", "60+"]).to_numpy()
    crash_days = np.where(np.isnan(cd) & ret20.notna().to_numpy(), "60+", crash_days)   # 급락 이력 없음 = 60+ 로 합침
    crash_days = pd.Series(np.where(ret20.notna(), crash_days, "nan"), index=dates)
    g.log(f"시장 순서 축 ({time.time()-t0:.0f}s) — 전환 이벤트 {int(turn.sum())}회, 급락일 {int(crash.sum())}일")

    low20 = C.rolling(20, min_periods=20).min()
    is_low = (C <= low20) & low20.notna()
    ar = np.arange(len(dates), dtype=float)[:, None]
    last_low = pd.DataFrame(np.where(is_low.to_numpy(), ar, np.nan), index=dates, columns=C.columns).ffill().to_numpy()
    bounce = ar - last_low
    bounce_lab = s1.cutw(pd.DataFrame(bounce, index=dates, columns=C.columns), [0, 3, 10], ["당일", "1~3", "4~10", "11+"])

    di = pd.Series(np.arange(len(dates)), index=dates)
    ci = pd.Series(np.arange(len(C.columns)), index=C.columns)
    ii = di.reindex(df.date).to_numpy(); jj = ci.reindex(df.ticker).to_numpy()
    ok = ~np.isnan(ii.astype(float)) & ~np.isnan(jj.astype(float))
    ii = ii.astype(int); jj = jj.astype(int)
    df["mkt_turn_days"] = mkt_turn.to_numpy()[ii]
    df["crash_days"] = crash_days.to_numpy()[ii]
    df["stk_bounce_days"] = np.where(ok, bounce_lab[ii, jj], "nan")
    g.log(f"축 추가 완료 ({time.time()-t0:.0f}s)")
    return df


def focus_tables(df, base):
    """국면 안에서 '며칠째'를 가르는 조건부 표 — 1회차 후보 국면을 순서 축으로 쪼갬."""
    out = []
    def T(name, sub, axes):
        t = s1.cell_table(sub, axes, base)
        t["share"] = t["n"] / len(df)                      # 전체 대비 비율로 통일
        t.insert(0, "focus", name)
        for a in axes:
            t = t.rename(columns={a: f"s_{axes.index(a)+1}"})
        t["axes"] = "×".join(axes)
        out.append(t)
    T("회복초입: 시장200 아래+상승 → 전환 며칠째", df[df.mkt_200 == "아래+상승"], ["mkt_turn_days"])
    T("회복초입 × 종목60 아래 → 전환 며칠째", df[(df.mkt_200 == "아래+상승") & df.stk_60.str.startswith("아래")], ["mkt_turn_days"])
    T("급락 후 경과일 × 시장200 위치", df, ["crash_days", "mkt_200"])
    T("급락 당일: 시장200 × 시장60", df[df.mkt_ret20 == "급락<-8"], ["mkt_200", "mkt_60"])
    T("5일 급락 종목(gap5<-5) → 저점 갱신 며칠째", df[df.gap5 == "<-5"], ["stk_bounce_days"])
    T("5일 급락 × 회복초입 → 저점 갱신 며칠째", df[(df.gap5 == "<-5") & (df.mkt_200 == "아래+상승")], ["stk_bounce_days"])
    T("시장 이중 국면: 200 × 60", df, ["mkt_200", "mkt_60"])
    T("고공 조정: 시장200 위+상승 × 시장60 아래 → 종목60", df[(df.mkt_200 == "위+상승") & df.mkt_60.str.startswith("아래")], ["stk_60"])
    return pd.concat(out, ignore_index=True)


def main():
    t0 = time.time()
    df = pd.read_parquet(OUT / "states.parquet")
    g.log(f"states.parquet {df.shape}")
    df = add_axes(df)
    df.to_parquet(OUT / "states2.parquet", index=False)
    base = s1.stats(df)

    single = pd.concat([s1.cell_table(df, [ax], base).assign(axis=ax, state=lambda d, ax=ax: d[ax].astype(str)).drop(columns=[ax])
                        for ax in ["mkt_turn_days", "crash_days", "stk_bounce_days", "mkt_60", "mkt_200"]], ignore_index=True)
    cols = ["axis", "state", "n", "share", "n_days", "p_up", "p_dn", "p_vert", "mean_rpath", "lift_up", "lift_dn", "spread",
            "p_up_A", "p_dn_A", "mean_rpath_A", "p_up_B", "p_dn_B", "mean_rpath_B", "ok_size", "consistent", "hit_target"]
    single = single[cols]; single.to_csv(OUT / "ladder2_single.csv", index=False)

    focus = focus_tables(df, base); focus.to_csv(OUT / "ladder2_focus.csv", index=False)
    g.log(f"single·focus ({time.time()-t0:.0f}s)")

    pairs = pd.concat([s1.cell_table(df, list(c), base).rename(columns={c[0]: "s1", c[1]: "s2"}).assign(ax1=c[0], ax2=c[1])
                       for c in itertools.combinations(AXES2, 2)], ignore_index=True)
    pairs = pairs[pairs.ok_size].sort_values("mean_rpath_A", ascending=False)
    pairs.to_csv(OUT / "ladder2_pairs.csv", index=False)
    g.log(f"pairs {len(pairs)} ({time.time()-t0:.0f}s)")

    drills = []
    for obj in ["spread", "mean_rpath"]:
        drills.append(s1.drill(df, AXES2, base, depth=5, objective=obj).assign(scope="all", objective=obj))
        for tier in ["bigf", "big_only", "small"]:
            sub = df[df.tier == tier]
            drills.append(s1.drill(sub, [x for x in AXES2 if x != "tier"], base, depth=5, objective=obj).assign(scope=tier, objective=obj))
    dr = pd.concat(drills, ignore_index=True); dr.to_csv(OUT / "ladder2_drill.csv", index=False)
    (OUT / "ladder2_summary.md").write_text(summary_md(df, base, single, focus, pairs, dr), encoding="utf-8")
    g.log(f"완료 → {OUT}  ({time.time()-t0:.0f}s)")


def summary_md(df, base, single, focus, pairs, dr):
    F = s1._fmt
    o = [f"# 중기 모델 상황 사다리 — 2회차 (순서 축 + 시장 200×60)",
         f"{df.date.min().date()}~{df.date.max().date()} · {len(df):,}셀 · N=10 k=1.0 · 전반 A <{s1.SPLIT_YEAR} / 후반 B",
         f"기저: p_up {base['p_up']:.3f} · p_dn {base['p_dn']:.3f} · r_path {base['mean_rpath']:+.4f} · 원하는 상태 p_up≥{s1.TARGET['p_up']} p_dn≤{s1.TARGET['p_dn']} r_path≥{s1.TARGET['mean_rpath']:+.3f}", ""]
    o.append("## 1. 새 축 단일 사다리")
    for ax, t in single.groupby("axis", sort=False):
        o.append(f"### {ax}")
        o.append(F(t, ["state", "n", "share", "n_days", "p_up", "p_dn", "mean_rpath", "spread", "mean_rpath_A", "mean_rpath_B", "consistent"])); o.append("")
    o.append("## 2. 국면 안에서 '며칠째' — 조건부 표")
    for name, t in focus.groupby("focus", sort=False):
        axes = t["axes"].iloc[0].split("×")
        scols = [f"s_{i+1}" for i in range(len(axes))]
        o.append(f"### {name}  ({' × '.join(axes)})")
        o.append(F(t.sort_values(scols), scols + ["n", "share", "n_days", "p_up", "p_dn", "mean_rpath", "spread", "mean_rpath_A", "mean_rpath_B", "consistent", "hit_target"])); o.append("")
    o.append("## 3. 2축 조합 — 전반 r_path 상위 20 (크기 조건 충족) · 후반 확인")
    o.append(F(pairs.head(20), ["ax1", "s1", "ax2", "s2", "n", "share", "p_up", "p_dn", "mean_rpath", "mean_rpath_A", "mean_rpath_B", "p_up_B", "p_dn_B", "consistent", "hit_target"]))
    hp = pairs[pairs.hit_target]
    o.append(f"\n원하는 상태 도달 2축 셀 {len(hp)}개")
    if len(hp):
        o.append(F(hp.head(20), ["ax1", "s1", "ax2", "s2", "n", "n_days", "p_up", "p_dn", "mean_rpath", "mean_rpath_A", "mean_rpath_B", "consistent"]))
    o.append("\n## 4. 드릴다운 (전반에서 고르고 후반으로 확인, 깊이 5)")
    for (scope, obj), t in dr.groupby(["scope", "objective"], sort=False):
        o.append(f"### scope={scope} · objective={obj}")
        o.append(F(t, ["depth", "axis", "state", "n", "share", "p_up", "p_dn", "mean_rpath", "mean_rpath_A", "mean_rpath_B", "p_up_B", "p_dn_B"])); o.append("")
    return "\n".join(o)


if __name__ == "__main__":
    main()
