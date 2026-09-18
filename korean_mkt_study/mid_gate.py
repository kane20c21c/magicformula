# -*- coding: utf-8 -*-
"""mid_gate.py — 중기 모델 0단계 관문 (A층: 스윙 포트 긴 데이터 2014-01~2026-06)

질문: t일 종가 시점의 정보 중, t+1 시가 진입 후 N일(5/10/20) 안에 상단 배리어를 먼저 칠 종목을
     가려내는 것이 있는가. (트리플 배리어 라벨 · 매일 순위 재평가 전제 · 횡단면 IC)

라벨 (트리플 배리어, 진입가 = t+1 시가):
  폭 = k × vol20(일간 수익률 표준편차) × √N,  k ∈ {0.5, 1.0}
  +1 상단 먼저(고가 터치) / −1 하단 먼저(저가 터치; 같은 날 동시면 −1 보수) / 0 만기(t+N 종가)
  r_path = +폭 / −폭 / 만기수익률 (배리어 체결 가정, 슬리피지 없음),  r_raw = t+N 종가/진입가 − 1

피처 (전부 t일까지의 정보):
  MA 격자  {sma,ema,wma,dema,evwma} × {5,20,60,120,200} → gap(종가/MA−1), slope(MA 5일 변화율),
           kind별 align(5>20, 20>60, 60>120, 120>200 성립 개수 0~4)
  모멘텀   ret5/20/60/120, ret20_5(최근 5일 제외), pos52(52주 고가 대비), vol20, vol60, volr(vol20/vol60)
  수급     외국인·기관 순매수 5/20/60일 누적 ÷ 20일 평균 거래대금 (investor_flow)
  기타     turnover(amt20/시총), log_mc
  (섹터 상대강도는 A층 불가 — meta.sector 가 전부 '기타'. B층 LLV 에서 잰다)

유니버스 계층 (월말 meta → 다음 달 적용):
  all  = 우선주·스팩 제외, 관리종목 아닌 달, 종가 ≥ 1,000원, 20일 평균 거래대금 ≥ 10억
  big  = all & 시총 ≥ 4조
  bigf = big & 외인지분 ≥ 30%   (스윙 포트 운영 유니버스)
  공통 유효조건: MA200 워밍업·수급 60일 누적 가능한 셀만 (모든 피처를 같은 셀에서 평가)

평가: 일별 스피어만 IC → 평균, 겹침 보정 t(유효 일수 = 일수/N), 양수일 비율,
     모멘텀(ret20·ret60) 통제 부분 IC, 순위 지속성(전일 순위와 상관), align 셀 기대값,
     기저율(+1/−1/0 — 전체·연도·vol20 사분위), 횡단면 분산, 오라클 top10.

실행: python3 mid_gate.py            (전체 — 레이어당 피처 80개, 수 분~십수 분)
      python3 mid_gate.py --quick    (2022-06~ · 시총 상위 300 — 스모크)
산출: out/mid/{base_rates,base_by_year,base_by_volq,ic,cells_align}.csv + summary.md
"""
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data_ext"
FLOW = HERE / "data" / "investor_flow.parquet"
OUT = HERE / "out" / "mid"

START, END = "2014-01-01", "2026-09-11"       # 수급 시작 ~ 백필 끝 (backfill_prices_krx.py, KRX 단독 거래량으로 통일)
EVAL_START = "2015-01-02"
FLOW_END = "2026-06-30"                       # investor_flow 는 KRX 차단으로 여기서 멈춤 — 수급 피처는 그 이후 결측
MA_KINDS = ["sma", "ema", "wma", "dema", "evwma"]
MA_P = [5, 20, 60, 120, 200]
NS = [5, 10, 20]
KS = [0.5, 1.0]
LAYERS = ["all", "big", "bigf"]
MIN_N = 20                                    # 하루 횡단면 최소 종목 수 (bigf 29~44종목 고려)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pc(df, n=1):
    return df.pct_change(n, fill_method=None)


# ───────────────────────── 데이터 ─────────────────────────
def load(quick):
    px = pd.read_parquet(DATA / "prices.parquet")
    d = px.index.get_level_values("date")
    px = px[(d >= START) & (d <= END)]
    meta = pd.read_parquet(DATA / "meta.parquet")
    flow = pd.read_parquet(FLOW)[["외국인", "기관"]]
    if quick:
        top = meta.groupby(level="ticker").mktcap.mean().nlargest(300).index
        px = px[px.index.get_level_values("ticker").isin(top)]
        d = px.index.get_level_values("date")
        px = px[d >= "2022-06-01"]
    W = {c: px[c].unstack("ticker").astype("float64") for c in ["open", "high", "low", "close", "volume"]}
    for c in ["open", "high", "low", "close"]:            # 0/음수 = 거래정지 마커 → 결측 (build_panel 규칙)
        W[c] = W[c].where(W[c] > 0)
    W["high"] = W["high"].where(W["high"] >= W["close"], W["close"])
    W["low"] = W["low"].where(W["low"] <= W["close"], W["close"])
    dates, tickers = W["close"].index, W["close"].columns

    m = meta.reset_index()
    m["mon"] = m.date.dt.to_period("M") + 1               # 월말 값 → 다음 달 적용

    def monthly_wide(col):
        t = m.pivot_table(index="mon", columns="ticker", values=col, aggfunc="last")
        t = t.reindex(columns=tickers)
        out = t.reindex(dates.to_period("M"))
        out.index = dates
        return out

    mktcap = monthly_wide("mktcap")
    foreign = monthly_wide("foreign")                       # 0~1 비율
    managed = monthly_wide("is_managed").fillna(0.0)
    static = m.groupby("ticker")[["is_pref", "is_spac"]].max().reindex(tickers).fillna(False).astype(bool)
    sector = m.groupby("ticker").sector.last().reindex(tickers)
    fw = {c: flow[c].unstack("ticker").reindex(index=dates, columns=tickers).astype("float64")
          for c in ["외국인", "기관"]}
    return W, mktcap, foreign, managed, static, sector, fw


def bcast_cols(series_bool, like):
    return pd.DataFrame(np.broadcast_to(series_bool.reindex(like.columns).to_numpy()[None, :], like.shape),
                        index=like.index, columns=like.columns)


def bcast_rows(series_bool, like):
    return pd.DataFrame(np.broadcast_to(series_bool.reindex(like.index).to_numpy()[:, None], like.shape),
                        index=like.index, columns=like.columns)


def build_masks(W, mktcap, foreign, managed, static):
    C, V = W["close"], W["volume"]
    amt20 = (C * V).rolling(20, min_periods=10).mean()
    base = C.notna() & (C >= 1000) & (amt20 >= 1e9) & (managed == 0)
    base = base & bcast_cols(~(static.is_pref | static.is_spac), C)
    big = base & (mktcap >= 4e12)
    bigf = big & (foreign >= 0.30)
    return {"all": base, "big": big, "bigf": bigf}


# ───────────────────────── 이동평균 ─────────────────────────
def ma(kind, C, V, p):
    if kind == "sma":
        return C.rolling(p, min_periods=p).mean()
    if kind == "ema":
        return C.ewm(span=p, adjust=False, min_periods=p).mean()
    if kind == "dema":
        e = C.ewm(span=p, adjust=False, min_periods=p).mean()
        return 2 * e - e.ewm(span=p, adjust=False, min_periods=p).mean()
    if kind == "wma":
        w = np.arange(1, p + 1, dtype=float)
        acc = None
        for i in range(p):
            term = C.shift(p - 1 - i) * w[i]
            acc = term if acc is None else acc + term
        return acc / w.sum()
    if kind == "evwma":                        # E_t = (1−v_t/Σv) E_{t−1} + (v_t/Σv) C_t
        vs = V.rolling(p, min_periods=p).sum()
        A = (V / vs.replace(0, np.nan)).to_numpy()
        Cn = C.to_numpy()
        E = np.full_like(Cn, np.nan)
        prev = np.full(Cn.shape[1], np.nan)
        for t in range(Cn.shape[0]):
            a = A[t]
            cur = (1 - a) * prev + a * Cn[t]
            cur = np.where(np.isnan(prev), Cn[t], cur)
            cur = np.where(np.isnan(a) | np.isnan(Cn[t]), np.nan, cur)
            E[t] = cur
            prev = cur
        return pd.DataFrame(E, index=C.index, columns=C.columns)
    raise ValueError(kind)


# ───────────────────────── 라벨 ─────────────────────────
def build_labels(W, N, k):
    C, H, L, O = W["close"], W["high"], W["low"], W["open"]
    sig = pc(C).rolling(20, min_periods=10).std()
    entry = O.shift(-1)
    width = (k * sig * np.sqrt(N)).to_numpy()
    en = entry.to_numpy()
    upv, dnv = en * (1 + width), en * (1 - width)
    Hn, Ln = H.to_numpy(), L.to_numpy()
    big = N + 1
    t_up = np.full(C.shape, big, dtype=np.int16)
    t_dn = t_up.copy()
    for d in range(1, N + 1):
        hs = np.full_like(Hn, np.nan); hs[:-d] = Hn[d:]
        ls = np.full_like(Ln, np.nan); ls[:-d] = Ln[d:]
        t_up[(hs >= upv) & (t_up == big)] = d
        t_dn[(ls <= dnv) & (t_dn == big)] = d
    r_raw = C.shift(-N).to_numpy() / en - 1
    valid = ~np.isnan(r_raw) & ~np.isnan(width) & (width > 0)
    lab = np.where(t_dn <= t_up, -1.0, 1.0)
    lab = np.where((t_up == big) & (t_dn == big), 0.0, lab)
    r_path = np.where(lab == 1, width, np.where(lab == -1, -width, r_raw))
    nan = np.nan
    lab = np.where(valid, lab, nan); r_path = np.where(valid, r_path, nan); r_raw = np.where(valid, r_raw, nan)
    f32 = lambda a: a.astype(np.float32)
    return {"lab": f32(lab), "r_path": f32(r_path), "r_raw": f32(r_raw),
            "y_up": f32(np.where(valid, lab == 1, nan)), "y_dn": f32(np.where(valid, lab == -1, nan))}


# ───────────────────────── 피처 (제너레이터 — 메모리 절약) ─────────────────────────
def feature_iter(W, mktcap, sector, fw, base_mask):
    C, V = W["close"], W["volume"]
    ret = pc(C)
    amt20 = (C * V).rolling(20, min_periods=10).mean()
    for kind in MA_KINDS:
        mas = {p: ma(kind, C, V, p) for p in MA_P}
        for p in MA_P:
            yield f"{kind}{p}_gap", C / mas[p] - 1
            yield f"{kind}{p}_slope", mas[p] / mas[p].shift(5) - 1
        al = None
        for a, b in zip(MA_P[:-1], MA_P[1:]):
            term = (mas[a] > mas[b]).astype(float).where(mas[a].notna() & mas[b].notna())
            al = term if al is None else al + term
        yield f"{kind}_align", al
        del mas
    for n in [5, 20, 60, 120]:
        yield f"ret{n}", pc(C, n)
    yield "ret20_5", C.shift(5) / C.shift(20) - 1
    yield "pos52", C / C.rolling(252, min_periods=120).max()
    v20 = ret.rolling(20, min_periods=10).std()
    v60 = ret.rolling(60, min_periods=30).std()
    yield "vol20", v20
    yield "vol60", v60
    yield "volr", v20 / v60
    for nm, f in fw.items():
        tag = "for" if nm == "외국인" else "inst"
        for n in [5, 20, 60]:
            yield f"{tag}_cum{n}", f.rolling(n, min_periods=n).sum() / amt20
    # ⚠ 섹터 상대강도는 A층에서 불가 — data_ext/meta.parquet 의 sector 가 전부 '기타'(자리표시자).
    #   B층(LLV 206 · ticker_classification) 에서만 잰다.
    yield "turnover", amt20 / mktcap
    yield "log_mc", np.log(mktcap)


# ───────────────────────── 횡단면 통계 ─────────────────────────
def xs_rank(F, mask):
    return F.where(mask).rank(axis=1).to_numpy(dtype=np.float32)


def rowcorr(A, B):
    """행(날짜)별 피어슨 상관 — 두 배열 모두 유효한 셀만. 유효 수 < MIN_N 이면 NaN."""
    m = ~np.isnan(A) & ~np.isnan(B)
    n = m.sum(1)
    A0 = np.where(m, A, 0.0); B0 = np.where(m, B, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mA = A0.sum(1) / n; mB = B0.sum(1) / n
        dA = np.where(m, A0 - mA[:, None], 0.0); dB = np.where(m, B0 - mB[:, None], 0.0)
        r = (dA * dB).sum(1) / np.sqrt((dA ** 2).sum(1) * (dB ** 2).sum(1))
    r[n < MIN_N] = np.nan
    return r


def ic_stats(ic, N):
    ic = ic[~np.isnan(ic)]
    if len(ic) < 50:
        return np.nan, np.nan, np.nan, len(ic)
    neff = len(ic) / N                                     # 겹치는 라벨 → 유효 관측 = 일수/N
    return ic.mean(), ic.mean() / (ic.std(ddof=1) / np.sqrt(neff)), (ic > 0).mean(), len(ic)


def partial_corr(rFY, rFZ1, rFZ2, rYZ1, rYZ2, rZ1Z2):
    """F↔Y 의 Z1,Z2 통제 부분상관 (일별, 4×4 상관행렬의 정밀행렬로)."""
    T = len(rFY)
    R = np.zeros((T, 4, 4)); R[:, [0, 1, 2, 3], [0, 1, 2, 3]] = 1.0
    for (i, j), v in {(0, 1): rFY, (0, 2): rFZ1, (0, 3): rFZ2, (1, 2): rYZ1, (1, 3): rYZ2, (2, 3): rZ1Z2}.items():
        R[:, i, j] = v; R[:, j, i] = v
    ok = ~np.isnan(R).any(axis=(1, 2))
    out = np.full(T, np.nan)
    if ok.any():
        P = np.linalg.pinv(R[ok])
        with np.errstate(invalid="ignore", divide="ignore"):
            out[ok] = -P[:, 0, 1] / np.sqrt(P[:, 0, 0] * P[:, 1, 1])
    return out


def rank_np(arr, mask):
    return pd.DataFrame(np.where(mask, arr, np.nan)).rank(axis=1).to_numpy(dtype=np.float32)


# ───────────────────────── 기저율 ─────────────────────────
def base_rates(labels, masks, v20, years):
    import warnings
    rows, yrows, qrows = [], [], []
    for L, mk in masks.items():
        vq = np.ceil(pd.DataFrame(np.where(mk, v20, np.nan)).rank(axis=1, pct=True).to_numpy() * 4)
        for (N, k), lb in labels.items():
            lab, rp, rr = lb["lab"], lb["r_path"], lb["r_raw"]
            sel = mk & ~np.isnan(lab)

            def shares(s, **tag):
                n = int(s.sum())
                if n == 0:
                    return None
                l = lab[s]
                return dict(layer=L, N=N, k=k, **tag, n=n, p_up=float((l == 1).mean()), p_dn=float((l == -1).mean()),
                            p_vert=float((l == 0).mean()), mean_rpath=float(rp[s].mean()), mean_rraw=float(rr[s].mean()))

            d = shares(sel)
            if d is None:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rr_m = np.where(sel, rr, np.nan)
                d["xs_std_rraw"] = float(np.nanmean(np.nanstd(rr_m, axis=1)))
                cnt = sel.sum(1)
                rp_m = np.where(sel, rp, -np.inf)
                top = -np.partition(-rp_m, 9, axis=1)[:, :10]
                d["oracle_top10"] = float(top[cnt >= 10].mean())
                d["n_days"] = int((cnt >= MIN_N).sum())
            rows.append(d)
            for y in np.unique(years):
                r = shares(sel & (years[:, None] == y), year=int(y))
                if r: yrows.append(r)
            for q in range(1, 5):
                r = shares(sel & (vq == q), volq=q)
                if r: qrows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(yrows), pd.DataFrame(qrows)


# ───────────────────────── 메인 ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    W, mktcap, foreign, managed, static, sector, fw = load(a.quick)
    C = W["close"]; dates = C.index
    log(f"wide {C.shape}  {dates[0].date()}~{dates[-1].date()}  ({time.time()-t0:.0f}s)")

    masks_df = build_masks(W, mktcap, foreign, managed, static)
    common = C.rolling(200, min_periods=200).mean().notna()          # MA200 워밍업만 공통 조건 (수급은 6/30 이후 결측이라 제외)
    ev = pd.Series(dates >= ("2023-01-01" if a.quick else EVAL_START), index=dates)
    masks = {L: (m & common & bcast_rows(ev, C)).to_numpy() for L, m in masks_df.items()}
    for L, mk in masks.items():
        n = mk.sum(1); n = n[n > 0]
        log(f"layer {L}: 일평균 {n.mean():.0f}종목 (min {n.min()}, max {n.max()}), {len(n)}일")

    labels = {(N, k): build_labels(W, N, k) for N in NS for k in KS}
    log(f"labels {len(labels)}개 ({time.time()-t0:.0f}s)")
    v20 = pc(C).rolling(20, min_periods=10).std().to_numpy()
    years = dates.year.to_numpy()
    br, by, bq = base_rates(labels, masks, v20, years)
    br.to_csv(OUT / "base_rates.csv", index=False); by.to_csv(OUT / "base_by_year.csv", index=False)
    bq.to_csv(OUT / "base_by_volq.csv", index=False)
    log(f"base rates 저장 ({time.time()-t0:.0f}s)")

    ic_rows, cell_rows = [], []
    for L in LAYERS:
        mk = masks[L]
        mkdf = pd.DataFrame(mk, index=dates, columns=C.columns)
        TR = {}
        for (N, k), lb in labels.items():
            for tg in ["r_path", "lab", "y_up", "y_dn"]:      # lab = −1/0/+1 방향만 (변동성 혼입 제거)
                TR[(N, k, tg)] = rank_np(lb[tg], mk)
            if k == KS[-1]:
                TR[(N, np.nan, "r_raw")] = rank_np(lb["r_raw"], mk)
        Z1 = xs_rank(pc(C, 20), mkdf); Z2 = xs_rank(pc(C, 60), mkdf)
        rZZ = rowcorr(Z1, Z2)
        YZ = {key: (rowcorr(t, Z1), rowcorr(t, Z2)) for key, t in TR.items()}
        log(f"[{L}] 타깃 순위 {len(TR)}개 준비 ({time.time()-t0:.0f}s)")
        nf = 0
        for name, F in feature_iter(W, mktcap, sector, fw, masks_df["all"]):
            RF = xs_rank(F, mkdf)
            persist = float(np.nanmean(rowcorr(RF[1:], RF[:-1])))
            rFZ1, rFZ2 = rowcorr(RF, Z1), rowcorr(RF, Z2)
            for key, t in TR.items():
                N, k, tg = key
                ic = rowcorr(RF, t)
                m, tt, pos, n = ic_stats(ic, N)
                if name in ("ret20", "ret60"):
                    pm = pt = np.nan
                else:
                    pm, pt, _, _ = ic_stats(partial_corr(ic, rFZ1, rFZ2, YZ[key][0], YZ[key][1], rZZ), N)
                ic_rows.append(dict(feature=name, layer=L, N=N, k=k, target=tg, ic=m, t=tt, pos=pos,
                                    ic_partial=pm, t_partial=pt, persist=persist, n_days=n))
            if name.endswith("_align"):
                Fn = F.to_numpy()
                for (N, k), lb in labels.items():
                    for v in range(5):
                        sel = mk & (Fn == v) & ~np.isnan(lb["lab"])
                        n = int(sel.sum())
                        if n:
                            cell_rows.append(dict(kind=name[:-6], layer=L, N=N, k=k, align=v, n=n,
                                                  mean_rpath=float(lb["r_path"][sel].mean()),
                                                  p_up=float((lb["lab"][sel] == 1).mean()),
                                                  p_dn=float((lb["lab"][sel] == -1).mean())))
            nf += 1
            if nf % 10 == 0:
                log(f"[{L}] {nf} features ({time.time()-t0:.0f}s)")
        del TR, YZ
    ic = pd.DataFrame(ic_rows); cells = pd.DataFrame(cell_rows)
    ic.to_csv(OUT / "ic.csv", index=False); cells.to_csv(OUT / "cells_align.csv", index=False)
    (OUT / "summary.md").write_text(summary_md(br, ic, cells, masks, dates, a.quick), encoding="utf-8")
    log(f"완료 → {OUT}  ({time.time()-t0:.0f}s)")


# ───────────────────────── 요약 ─────────────────────────
def _tbl(df, cols, fmt):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.get(c, "{}").format(r[c]) if pd.notna(r[c]) else "—" for c in cols) + " |")
    return "\n".join(lines)


def summary_md(br, ic, cells, masks, dates, quick):
    F4 = {"ic": "{:+.4f}", "t": "{:+.1f}", "pos": "{:.2f}", "ic_partial": "{:+.4f}", "t_partial": "{:+.1f}",
          "persist": "{:.3f}", "p_up": "{:.3f}", "p_dn": "{:.3f}", "p_vert": "{:.3f}", "mean_rpath": "{:+.4f}",
          "mean_rraw": "{:+.4f}", "xs_std_rraw": "{:.4f}", "oracle_top10": "{:+.4f}", "n": "{:,}", "k": "{:.1f}"}
    out = [f"# 중기 모델 0단계 관문 — A층 {'(quick)' if quick else ''}",
           f"평가 {dates[0].date()}~{dates[-1].date()} · 라벨 트리플 배리어(k×vol20×√N, 진입 t+1 시가) · "
           f"IC = 일별 스피어만, t 는 겹침 보정(유효일수 = 일수/N)", ""]
    out.append("## 1. 기저율 (k=1.0)")
    b = br[br.k == 1.0].sort_values(["layer", "N"])
    out.append(_tbl(b, ["layer", "N", "n", "p_up", "p_dn", "p_vert", "mean_rpath", "mean_rraw", "xs_std_rraw", "oracle_top10"], F4))
    out.append("\n> p_up−p_dn 가 양수면 배리어 대칭 기준 우상향 기저. oracle_top10 = 완전예지 상한.\n")

    ic1 = ic[(ic.k == 1.0) & (ic.N == 10)]
    for L in LAYERS:
        out.append(f"## 2. 피처 IC — layer {L}, N=10, k=1.0")
        for tg, title in [("r_path", "경로 수익률 r_path (진입 랭킹 관점)"),
                          ("lab", "방향 lab = −1/0/+1 (변동성 혼입 제거 — 주 판정)"),
                          ("y_up", "상단 먼저 y_up"), ("y_dn", "하단 먼저 y_dn (양수 = 하단 위험↑)")]:
            d = ic1[(ic1.layer == L) & (ic1.target == tg)].copy()
            d["abs_t"] = d.t.abs()
            d = d.sort_values("abs_t", ascending=False).head(12)
            out.append(f"### {title}")
            out.append(_tbl(d, ["feature", "ic", "t", "pos", "ic_partial", "t_partial", "persist"], F4))
            out.append("")
    out.append("## 3. 이동평균 종류 비교 — 방향 lab, N=10, k=1.0 (셀 = IC / t)")
    for L in ["all", "bigf"]:
        d = ic1[(ic1.layer == L) & (ic1.target == "lab")].set_index("feature")
        out.append(f"### layer {L}")
        hdr = "| 피처 | " + " | ".join(MA_KINDS) + " |"
        out.append(hdr); out.append("|" + "---|" * (len(MA_KINDS) + 1))
        for p in MA_P:
            for typ in ["gap", "slope"]:
                cells_ = []
                for kd in MA_KINDS:
                    nm = f"{kd}{p}_{typ}"
                    cells_.append(f"{d.loc[nm,'ic']:+.3f} / {d.loc[nm,'t']:+.1f}" if nm in d.index else "—")
                out.append(f"| {p}_{typ} | " + " | ".join(cells_) + " |")
        cells_ = []
        for kd in MA_KINDS:
            nm = f"{kd}_align"
            cells_.append(f"{d.loc[nm,'ic']:+.3f} / {d.loc[nm,'t']:+.1f}" if nm in d.index else "—")
        out.append("| align | " + " | ".join(cells_) + " |")
        out.append("")
    out.append("## 4. 정배열 단계별 기대값 — N=10, k=1.0 (align 0=역배열 … 4=완전 정배열)")
    c = cells[(cells.N == 10) & (cells.k == 1.0) & (cells.kind.isin(["sma", "evwma"])) & (cells.layer.isin(["all", "bigf"]))]
    out.append(_tbl(c.sort_values(["layer", "kind", "align"]), ["layer", "kind", "align", "n", "p_up", "p_dn", "mean_rpath"], F4))
    out.append("\n## 5. N·k 전체 — 방향 lab 기준 |t| 상위 5 (레이어별)")
    for L in LAYERS:
        d = ic[(ic.layer == L) & (ic.target == "lab")].copy(); d["abs_t"] = d.t.abs()
        top = d.sort_values("abs_t", ascending=False).groupby(["N", "k"]).head(5).sort_values(["N", "k", "abs_t"], ascending=[True, True, False])
        out.append(f"### layer {L}")
        out.append(_tbl(top, ["N", "k", "feature", "ic", "t", "pos", "ic_partial"], F4))
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    main()
