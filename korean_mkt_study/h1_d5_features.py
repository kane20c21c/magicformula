"""h1_d5_features.py — 조건부 전환 1단계 (2026-10-05, Kane): 진입 후 D+5 종가 시점의 지표가
그 뒤 성과를 가르는가? 엔진 규칙을 바꾸지 않는 거래 단위 분석.

모집단  현행(v1.2.3.4) 11년 백테스트의 신규 진입(불타기는 표시만) 중 D+5 종가까지 살아 있는 건.
        패널의 거래정지 0값(시가·고가·저가·거래량)은 결측 처리.
성과    D+5 종가 → +10일 / +20일 수익률 (유니버스 동일가중 평균 대비 초과),
        trail = D+5 종가에서 '좁히지 않고'(피크 −20%×배율) 최대 60일 들고 갔을 때 수익률.
지표    실행 전 고정 (FEATURES). 모두 D+5 종가까지의 정보만 사용. 신호일 = 진입 전 거래일.
판정    ① 전체 Spearman ② 평단 대비 수익률(ret5) 통제 후 부분상관 ③ 기간 A(~2021)/B(2022~) 부호 일치
        ④ 지표 수만큼 본페로니.
출력    out/h1/d5_features.parquet · d5_feature_stats.csv
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
warnings.simplefilter("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "HillStorm"))
import backtest as bt
from backtest import EntryParams, UniverseParams, VolScaleParams, _yang_zhang, compute_signals, metrics
from r2_yz_events import rolling_r2_slope
from r2_yz_s3_portsim import run_backtest2, SPLIT
from h1_stop_width_sweep import make_stop, D6, SIZINGS, OUT
from weis_wave import compute_weis_wave

K = 5
FEATURES = ["ret5", "vol_ratio", "updown_vol", "rsi", "rsi_chg", "rsi_cross50", "weis_up", "weis_flip_up", "weis_vol_ratio",
            "recov", "above_ma5", "above_ma20", "yz20_chg", "yzr_10_120", "yzr_chg",
            "for_net5", "for_chg", "ins_net5", "ins_chg", "chaikin", "chaikin_chg", "chaikin_pos"]


def load_fixed():
    p = bt.load_panel(UniverseParams(), VolScaleParams())
    for nm in ("open", "high", "low", "volume"):
        df = getattr(p, nm); df.mask(df <= 0, np.nan, inplace=True)
    return p


def rsi14(c):
    d = c.diff(); g = d.clip(lower=0); l = (-d).clip(lower=0)
    ag = g.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    al = l.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    return 100 - 100 / (1 + ag / al.replace(0, np.nan))


def build():
    p = load_fixed(); ep = EntryParams()
    O, H, L, C, V = p.open, p.high, p.low, p.close, p.volume
    base_sig = compute_signals(p, ep)["signal"]
    r2, _ = rolling_r2_slope(np.log(C), 120)
    yz10 = _yang_zhang(O, H, L, C, 10); yz120 = _yang_zhang(O, H, L, C, 120); yz20 = p.yz20
    yzr = yz10 / yz120
    sigA = base_sig & (base_sig.index < SPLIT).reshape(-1, 1)
    nw = ~((r2 >= r2.where(sigA).stack().dropna().quantile(2 / 3)) & (yzr < yzr.where(sigA).stack().dropna().quantile(1 / 3)))
    res = run_backtest2(p, ep, pp=SIZINGS["20x500"], signal_mask=nw, cond_exit=D6, stop_fn=make_stop())
    print("현행(0값 처리):", bt.fmt(metrics(res), "V0"))

    rsi = C.apply(rsi14)
    hl = H - L; mfm = (((C - L) - (H - C)) / hl).where(hl != 0, 0.0).fillna(0.0)
    ad = (mfm * V.fillna(0)).cumsum()
    co = ad.ewm(span=3, adjust=False).mean() - ad.ewm(span=10, adjust=False).mean()   # LLV Chaikin_Osc 와 같은 식
    v20 = V.rolling(20, min_periods=10).mean()
    co_n = co / v20                                    # ÷ 20일 평균 거래량 (종목 간 비교용)
    ma5 = C.rolling(5).mean(); ma20 = C.rolling(20).mean(); hi40 = H.rolling(40, min_periods=20).max()
    amt = (C * V).to_numpy()
    flow = pd.read_parquet(HERE / "data" / "investor_flow.parquet")
    FOR = flow["외국인"].unstack("ticker").reindex(index=C.index, columns=C.columns)
    INS = flow["기관"].unstack("ticker").reindex(index=C.index, columns=C.columns)
    wdir = pd.DataFrame(np.nan, index=C.index, columns=C.columns); wvol = wdir.copy()
    for tk in C.columns:
        g = pd.DataFrame(dict(Open=O[tk], High=H[tk], Low=L[tk], Close=C[tk], Volume=V[tk])).dropna()
        if len(g) < 60:
            continue
        try:
            gg = g.reset_index(); gg = gg.rename(columns={gg.columns[0]: "Date"})
            w = compute_weis_wave(gg)                  # atr×1.5 (LLV 정본 모드)
            wdir.loc[g.index, tk] = w["Weis_Dir"].to_numpy(); wvol.loc[g.index, tk] = w["Weis_Vol"].to_numpy()
        except Exception as e:
            print("weis 실패", tk, e)
    pool_idx = (1 + C.pct_change().where(p.elig).mean(axis=1).fillna(0.0)).cumprod().to_numpy()

    # ── 신규 진입 추출 ──
    tr = res["trades"]; di = {d: i for i, d in enumerate(p.dates)}; held = {}; rows = []
    for _, t in tr.iterrows():
        if t.side == "BUY":
            if t.ticker in held:
                held[t.ticker]["pyr"] = True; continue
            held[t.ticker] = dict(i0=di[t.date], px=t.price, pyr=False)
        else:
            e = held.pop(t.ticker, None)
            if e:
                rows.append(dict(ticker=t.ticker, i0=e["i0"], entry=e["px"], pyr=e["pyr"], i_exit=di[t.date],
                                 trade_ret=t.ret, why=t.why))
    n = len(p.dates); out = []
    for r in rows:
        tk, i0 = r["ticker"], r["i0"]; i5 = i0 + K; s0 = i0 - 1
        if i5 + 20 >= n or r["i_exit"] <= i5 or s0 < 130:
            continue                                   # D+5 종가까지 생존 + 20일 뒤 관측 가능
        j = C.columns.get_loc(tk)
        c = C[tk].to_numpy(); h = H[tk].to_numpy(); l = L[tk].to_numpy(); v = V[tk].to_numpy(); o = O[tk].to_numpy()
        c5 = c[i5]
        if not np.isfinite(c5):
            continue
        f = dict(r, date=p.dates[i0], year=p.dates[i0].year)
        f["fwd10x"] = c[i5 + 10] / c5 - pool_idx[i5 + 10] / pool_idx[i5]
        f["fwd20x"] = c[i5 + 20] / c5 - pool_idx[i5 + 20] / pool_idx[i5]
        sc = p.vol_scale[tk].iloc[s0]; sc = 1.0 if not np.isfinite(sc) else float(sc)
        pct = min(max(0.20 * sc, 0.10), 0.40); peak = np.nanmax(np.r_[r["entry"], h[i0:i5 + 1]])
        end = min(i5 + 60, n - 1); tr_ret = c[end] / c5 - 1
        for q in range(i5 + 1, end + 1):
            if not np.isfinite(l[q]):
                continue
            if l[q] <= peak * (1 - pct):
                tr_ret = min(o[q], peak * (1 - pct)) * 0.995 / c5 - 1; break
            peak = max(peak, h[q])
        f["trail"] = tr_ret
        # 지표 (D+5 종가까지)
        f["ret5"] = c5 / r["entry"] - 1
        f["vol_ratio"] = np.log(np.nanmean(v[i0 + 1:i5 + 1]) / np.nanmean(v[s0 - 19:s0 + 1]))
        d = np.diff(c[i0 - 1:i5 + 1]); vv = v[i0:i5 + 1]
        up = np.nansum(vv[d > 0]); dn = np.nansum(vv[d < 0])
        f["updown_vol"] = np.log((up + 1) / (dn + 1)) if (up > 0 and dn > 0) else np.nan
        R5, R0 = rsi[tk].iloc[i5], rsi[tk].iloc[s0]
        f["rsi"] = R5; f["rsi_chg"] = R5 - R0; f["rsi_cross50"] = float(R0 < 50 <= R5)
        wd5, wd0 = wdir[tk].iloc[i5], wdir[tk].iloc[s0]
        f["weis_up"] = float(wd5 > 0) if np.isfinite(wd5) else np.nan
        f["weis_flip_up"] = float(wd0 < 0 < wd5) if np.isfinite(wd5) and np.isfinite(wd0) else np.nan
        wr = np.nan
        if np.isfinite(wd5) and wd5 > 0:               # 현재 상승 파동 누적 거래량 ÷ 직전 하락 파동 거래량
            ws = wdir[tk].to_numpy(); wv = wvol[tk].to_numpy(); q = i5
            while q > 0 and ws[q] > 0:
                q -= 1
            if q > 0 and ws[q] < 0 and np.isfinite(wv[q]) and wv[q] > 0 and wv[i5] > 0:
                wr = np.log(wv[i5] / wv[q])
        f["weis_vol_ratio"] = wr
        lo = np.nanmin(l[i0 - 5:i5 + 1]); top = hi40[tk].iloc[s0]
        f["recov"] = (c5 - lo) / (top - lo) if np.isfinite(top) and top > lo else np.nan
        f["above_ma5"] = float(c5 > ma5[tk].iloc[i5]); f["above_ma20"] = float(c5 > ma20[tk].iloc[i5])
        f["yz20_chg"] = np.log(yz20[tk].iloc[i5] / yz20[tk].iloc[s0])
        f["yzr_10_120"] = yzr[tk].iloc[i5]; f["yzr_chg"] = yzr[tk].iloc[i5] - yzr[tk].iloc[s0]
        a5 = np.nansum(amt[i0 + 1:i5 + 1, j]); a0 = np.nansum(amt[s0 - 4:s0 + 1, j])
        for nm, F in (("for", FOR), ("ins", INS)):
            x = F[tk].to_numpy()
            s5 = np.nansum(x[i0 + 1:i5 + 1]) / a5 if a5 > 0 else np.nan
            s_prev = np.nansum(x[s0 - 4:s0 + 1]) / a0 if a0 > 0 else np.nan
            f[f"{nm}_net5"] = s5; f[f"{nm}_chg"] = s5 - s_prev
        f["chaikin"] = co_n[tk].iloc[i5]; f["chaikin_chg"] = co_n[tk].iloc[i5] - co_n[tk].iloc[s0]
        f["chaikin_pos"] = float(co[tk].iloc[i5] > 0)
        out.append(f)
    df = pd.DataFrame(out); df.to_parquet(OUT / "d5_features.parquet")
    print(f"\n신규 진입 {len(rows)}건 → D+5 생존·관측 {len(df)}건 (A {int((df.date < SPLIT).sum())} / B {int((df.date >= SPLIT).sum())})"
          f" · 불타기 있었던 건 {int(df.pyr.sum())}")
    return df


def sp(a, b):
    m = a.notna() & b.notna()
    return tuple(stats.spearmanr(a[m], b[m])) if m.sum() > 30 else (np.nan, np.nan)


def partial(sub, x, y, z="ret5"):
    """ret5 를 통제한 순위 부분상관."""
    d = sub[[x, y, z]].dropna()
    if len(d) < 40 or d[x].nunique() < 2:
        return np.nan, np.nan
    R = d.rank()
    rx = R[x] - np.polyval(np.polyfit(R[z], R[x], 1), R[z]); ry = R[y] - np.polyval(np.polyfit(R[z], R[y], 1), R[z])
    return tuple(stats.pearsonr(rx, ry))


def stat_table(df):
    A = df[df.date < SPLIT]; B = df[df.date >= SPLIT]; st = []
    for x in FEATURES:
        row = dict(feature=x, n=int(df[x].notna().sum()))
        for y in ("fwd10x", "fwd20x", "trail"):
            row[f"{y}_r"], row[f"{y}_p"] = sp(df[x], df[y])
            fn = (lambda s: sp(s[x], s[y])) if x == "ret5" else (lambda s: partial(s, x, y))
            row[f"{y}_pr"], row[f"{y}_pp"] = fn(df); row[f"{y}_prA"] = fn(A)[0]; row[f"{y}_prB"] = fn(B)[0]
        ys = [(sp(g[x], g["fwd20x"]) if x == "ret5" else partial(g, x, "fwd20x"))[0] for _, g in df.groupby("year")]
        ys = [v for v in ys if np.isfinite(v)]; row["yrs_pos"] = sum(v > 0 for v in ys); row["yrs"] = len(ys)
        st.append(row)
    return pd.DataFrame(st)


def analyze(df):
    print(f"D+5 평단 위 {(df.ret5 > 0).mean():.1%} · 성과 평균: fwd10x {df.fwd10x.mean():+.2%} · fwd20x {df.fwd20x.mean():+.2%} · trail {df.trail.mean():+.2%}")
    S = stat_table(df); S.to_csv(OUT / "d5_feature_stats.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print("\n[단순 Spearman] r · p")
    print(S[["feature", "n", "fwd10x_r", "fwd10x_p", "fwd20x_r", "fwd20x_p", "trail_r", "trail_p"]].round(3).to_string(index=False))
    print("\n[ret5 통제 부분상관] 전체 pr · p · 기간 A/B · 연도 부호(fwd20x)")
    print(S[["feature", "fwd10x_pr", "fwd10x_pp", "fwd20x_pr", "fwd20x_pp", "fwd20x_prA", "fwd20x_prB",
             "trail_pr", "trail_pp", "trail_prA", "trail_prB", "yrs_pos", "yrs"]].round(3).to_string(index=False))
    print(f"\n본페로니 기준 p < {0.05 / len(FEATURES):.4f}")
    for lbl, sub in (("D+5 평단 아래", df[df.ret5 <= 0]), ("D+5 평단 위", df[df.ret5 > 0])):
        print(f"\n[{lbl}] n={len(sub)} · fwd20x 평균 {sub.fwd20x.mean():+.2%} · trail 평균 {sub.trail.mean():+.2%}"
              f" · trail>0 비율 {(sub.trail > 0).mean():.1%} — 3분위 상위−하위(연속형) / 1−0(이진)")
        for x in FEATURES[1:]:
            d = sub[[x, "fwd20x", "trail"]].dropna()
            if d[x].nunique() <= 2:
                hi_ = d[d[x] > 0.5]; lo_ = d[d[x] <= 0.5]
            else:
                q1, q2 = d[x].quantile([1 / 3, 2 / 3]); hi_ = d[d[x] >= q2]; lo_ = d[d[x] <= q1]
            if len(hi_) < 15 or len(lo_) < 15:
                continue
            t1 = stats.ttest_ind(hi_.fwd20x, lo_.fwd20x, equal_var=False)
            t2 = stats.ttest_ind(hi_.trail, lo_.trail, equal_var=False)
            print(f"  {x:<15} n {len(hi_):>3}/{len(lo_):<3} fwd20x {hi_.fwd20x.mean() - lo_.fwd20x.mean():+.2%} (p {t1.pvalue:.3f})"
                  f"  trail {hi_.trail.mean() - lo_.trail.mean():+.2%} (p {t2.pvalue:.3f})")


if __name__ == "__main__":
    analyze(build())
