#!/usr/bin/env python3
"""
daily_WW_wf.py — 발열률 일일 워크포워드 (Weis Wave walk-forward)
================================================================
매일 16:30 구동. 산출물 3종:
  1) 온도 그래프  — W60/M4 + W100/M6 + TIGER200, 최근 60거래일
  2) Vblk 상위 10 — D-2/D-1/D+0 비교표 (+ D+0 종가·등락률)
  3) 일지 2종     — 온도일지 / 점수일지 (append, 중복 날짜 skip)

설계 정본: fever_model/docs/일일워크포워드_작업지시_20260814.md
로직 출처: src/nowcast_grid.py (온도) · src/resilience_v2.py (Vblk)
           · src/resilience_score.py (과열 플래그) — 검증 끝난 로직 그대로 이식

입력: LLV data/ohlcv/{core,extend}.parquet (읽기 전용)
      — Weis_Dir/Chg/Days/Vol · YZ_20_Ann · YZ_60_Ann · Close
      LLV 16:00 KIS 종가 배치가 지표까지 재계산하므로 16:30 구동 가능
      (수급·Wyckoff 는 발열률 입력이 아니라 20:30 배치를 기다릴 필요 없음)

유니버스: data/panel_states.csv 의 191종목 고정 (스터디와 동일 — 우선주·
         2023년 이후 신규상장 제외분이 이미 반영돼 있다). 신규 편입 없음.

사용법:
    python3 daily_WW_wf.py                    # 당일 1회 (일지에 있으면 skip)
    python3 daily_WW_wf.py --backfill 60      # 최초 1회: 온도 60거래일 백필
    python3 daily_WW_wf.py --force            # 같은 날짜 재계산·덮어쓰기
    python3 daily_WW_wf.py --no-graph         # 일지만 갱신
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # launchd 대비 (같은 폴더)
from _stolab import find_stolab                                  # noqa: E402

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))          # src/
FM   = os.path.dirname(BASE)                               # 리포 루트
DATA = os.path.join(FM, "data")
OUT  = os.path.join(FM, "output")
# ⚠ 깊이 고정식(dirname×2) 금지 — SpotGauge 이관 시 한 단계 어긋난다. `_stolab.py` 참조.
STOLAB = find_stolab(FM)                                   # StoLab/ (형제 프로젝트로 판정)
LLV  = os.environ.get("LLV_DIR", os.path.join(STOLAB, "longlivevault"))
OHLCV = os.path.join(LLV, "data", "ohlcv")

TEMP_LOG  = os.path.join(OUT, "나우캐스트_온도일지.csv")
STATE_LOG = os.path.join(OUT, "나우캐스트_국면일지.csv")   # 종목별 국면 (재가열 판정용)
SCORE_LOG = os.path.join(OUT, "탄력점수_일지.csv")
GRAPH     = os.path.join(OUT, "발열률_온도그래프.png")
TOP_CSV   = os.path.join(OUT, "발열률_상위10.csv")      # 메일용
ALL_CSV   = os.path.join(OUT, "발열률_전체.csv")        # 8501 페이지용 (D+0 전 종목)

os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# 파라미터 (변경 시 fever_model/CLAUDE.md 도 함께)
# ---------------------------------------------------------------------------
W_ADOPT, MINW_ADOPT = 60, 4        # 채택안 (2026-08-13 Kane)
W_PREV,  MINW_PREV  = 100, 6       # 비교용 구기준
GATE = 4                           # 특징 계산 최소 파동 (MINW 는 사후 마스킹)

RES_W, RES_MINW, RES_RECENT = 100, 6, 8   # 탄력점수 자체 창 (완화 미적용)
GRAPH_DAYS   = 120                 # 그래프 표시 구간 (2026-08-14 Kane: 60 → 120)
REHEAT_DAYS  = 60                  # 재가열 룩백 (최근 60거래일 내 '장기' 이력)
HIST_MEDIAN  = 29.0                # 역사적 온도 중앙값 (점선, 스터디 §5-4)
TIGER = "102110"

FCOLS = ["waves_per_yr", "amp_cv", "amp_skew", "up_dn_amp", "up_dn_days",
         "maxwave_pos", "late_amp_ratio", "days_mean_log", "days_cv",
         "amp_vol_corr", "yz20_med", "yz60_med", "yz_ratio"]
SNAME = {0: "단기", 1: "장기", 2: "저변"}

COL_UP, COL_DOWN = "#ef5350", "#1976D2"


# ---------------------------------------------------------------------------
# 데이터 적재
# ---------------------------------------------------------------------------
def load_panel():
    """LLV parquet → (종목축 배열 dict, tickers, dates, name/sector map)"""
    pf = pd.read_csv(os.path.join(DATA, "panel_states.csv"), dtype={"ticker": str})
    universe = sorted(pf["ticker"].unique())

    cols = ["Date", "Ticker", "Name", "Weis_Dir", "Weis_Chg", "Weis_Days",
            "Weis_Vol", "YZ_20_Ann", "YZ_60_Ann", "Close"]
    frames = [pd.read_parquet(os.path.join(OHLCV, f), columns=cols)
              for f in ("core.parquet", "extend.parquet")]
    df = (pd.concat(frames).drop_duplicates(["Date", "Ticker"])
          .sort_values(["Ticker", "Date"]))

    tiger = (df[df["Ticker"] == TIGER].set_index("Date")["Close"]
             .sort_index())                      # 그래프 우축 (유니버스와 무관)
    df = df[df["Ticker"].isin(universe)]

    dates = np.array(sorted(df["Date"].unique()))
    d2i = {d: i for i, d in enumerate(dates)}
    tickers = sorted(df["Ticker"].unique())
    N, T = len(tickers), len(dates)

    arr = {c: np.full((N, T), np.nan) for c in
           ("dir", "chg", "wdays", "vol", "yz20", "yz60", "close")}
    src = {"dir": "Weis_Dir", "chg": "Weis_Chg", "wdays": "Weis_Days",
           "vol": "Weis_Vol", "yz20": "YZ_20_Ann", "yz60": "YZ_60_Ann",
           "close": "Close"}
    t2j = {t: j for j, t in enumerate(tickers)}
    for t, g in df.groupby("Ticker"):
        j = t2j[t]
        idx = g["Date"].map(d2i).values
        for k, c in src.items():
            arr[k][j, idx] = g[c].values

    names = df.groupby("Ticker")["Name"].last().to_dict()
    return arr, tickers, dates, names, tiger, pf


def load_sector_map():
    """섹터·구분 정본 = LLV core_tickers + extend_tickers (동일 어휘 체계).

    반환 (섹터 dict, core 티커 set). core/extend 구분은 페이지 섹터표에서 쓴다.
    """
    if LLV not in sys.path:
        sys.path.insert(0, LLV)
    try:
        from stolab_data.core_tickers import TICKER_LIST
        from stolab_data.extend_tickers import EXTEND_LIST
        sec = {t: s for t, _, s in list(TICKER_LIST) + list(EXTEND_LIST)}
        return sec, {t for t, _, _ in TICKER_LIST}
    except Exception as e:                                   # fail-safe
        print(f"  ⚠ 섹터 맵 로드 실패 ({e}) — 섹터 공란으로 진행")
        return {}, set()


# ---------------------------------------------------------------------------
# 파동 복원 (nowcast_grid.py / resilience_v2.py 공통 로직)
# ---------------------------------------------------------------------------
def build_waves(arr, j, lo, hi):
    """방향 전환으로 접은 파동 열 → [(dir, amp, days, vol, end_idx), ...]"""
    dirv = arr["dir"][j, lo:hi + 1]
    ok = ~np.isnan(dirv)
    if ok.sum() < 10:
        return []
    ii = np.where(ok)[0]
    dv = dirv[ii]
    cut = np.where(np.diff(dv) != 0)[0]
    starts = np.r_[0, cut + 1]
    ends = np.r_[cut, len(dv) - 1]
    out = []
    for s, e in zip(starts, ends):
        seg = ii[s:e + 1] + lo
        out.append((dv[s],
                    np.nanmax(np.abs(arr["chg"][j, seg])),
                    np.nanmax(arr["wdays"][j, seg]),
                    np.nanmax(arr["vol"][j, seg]),
                    seg[-1]))
    return out


def _spearman(a, b):
    """scipy 없이 순위 상관 (scipy.stats.spearmanr 와 동치)."""
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------------------
# 1단 — 나우캐스트 온도
# ---------------------------------------------------------------------------
def nowcast_features(arr, full_waves, j, ti, W):
    """gate=4 로 계산. 반환 (특징 13, 파동수) 또는 None. MINW 는 사후 마스킹."""
    lo = ti - W + 1
    if lo < 0:
        return None
    w = build_waves(arr, j, lo, ti)
    if not w or len(w) < GATE:
        return None
    dirs = np.array([x[0] for x in w]); amps = np.array([x[1] for x in w])
    days = np.array([x[2] for x in w]); vols = np.array([x[3] for x in w])
    endi = np.array([x[4] for x in w], dtype=float)

    hist = [x[3] for x in full_waves[j] if x[4] < ti]        # 선견 없음
    if len(hist) < 5:
        return None
    vm = float(np.median(hist))
    if not np.isfinite(vm) or vm <= 0:
        return None
    vn = vols / vm

    up, dn = dirs > 0, dirs < 0
    if not up.any() or not dn.any():
        return None
    span = max(endi[-1] - endi[0], 1)
    pos = (endi - endi[0]) / span
    late = pos > 0.75
    yz20 = np.nanmedian(arr["yz20"][j, lo:ti + 1])
    yz60 = np.nanmedian(arr["yz60"][j, lo:ti + 1])
    if not (np.isfinite(yz20) and np.isfinite(yz60)) or yz60 <= 0:
        return None

    amean = amps.mean()
    f = [len(w) / (W / 252),
         amps.std() / amean,
         float(pd.Series(amps).skew()),
         np.log(amps[up].mean() / amps[dn].mean()),
         np.log(days[up].mean() / days[dn].mean()),
         float(pos[np.argmax(amps)]),
         amps[late].mean() / amean if late.any() else np.nan,
         np.log(days.mean()),
         days.std() / days.mean(),
         _spearman(amps, vn),
         yz20, yz60, yz20 / yz60]
    if any(not np.isfinite(x) for x in f):
        return None
    return f, len(w)


def nowcast_at(arr, full_waves, tickers, ti, W, minw, scaler, cent):
    """판정일 ti 의 종목별 국면 → DataFrame[ticker, state, n_waves, conf]"""
    feats, meta = [], []
    for j in range(len(tickers)):
        r = nowcast_features(arr, full_waves, j, ti, W)
        if r is None:
            continue
        f, nw = r
        if nw < minw:
            continue
        feats.append(f)
        meta.append((tickers[j], nw))
    if not feats:
        return pd.DataFrame(columns=["ticker", "state", "n_waves", "conf"])
    Z = scaler.transform(np.array(feats))
    D = np.linalg.norm(Z[:, None, :] - cent[None, :, :], axis=2)
    o = np.argsort(D, axis=1)
    m = np.arange(len(D))
    out = pd.DataFrame(meta, columns=["ticker", "n_waves"])
    out["state"] = [SNAME[i] for i in o[:, 0]]
    out["conf"] = D[m, o[:, 1]] / D[m, o[:, 0]]
    return out


def temperature(states: pd.DataFrame, n_universe: int) -> dict:
    """요동(단기+장기) 비율 = 온도. 분모는 판정된 종목."""
    if states.empty:
        return dict(hot=np.nan, 장기=0, 저변=0, 단기=0, 유보=n_universe)
    vc = states["state"].value_counts()
    judged = int(vc.sum())
    return dict(
        hot=round(float(states["state"].isin(["단기", "장기"]).mean()) * 100, 1),
        장기=int(vc.get("장기", 0)), 저변=int(vc.get("저변", 0)),
        단기=int(vc.get("단기", 0)), 유보=n_universe - judged)


# ---------------------------------------------------------------------------
# 2단 — Vblk 탄력점수
# ---------------------------------------------------------------------------
def res_signals(arr, full_waves, j, ti):
    """resilience_v2.signals + resilience_score 의 과열(climax) 플래그."""
    lo = ti - RES_W + 1
    if lo < 0:
        return None
    w = build_waves(arr, j, lo, ti)
    if len(w) < RES_MINW:
        return None
    r = w[-RES_RECENT:]
    up = [x for x in r if x[0] > 0]
    dn = [x for x in r if x[0] < 0]
    if not up or not dn:
        return None
    hist_vols = [x[3] for x in full_waves[j] if x[4] < ti]
    if len(hist_vols) < 5:
        return None
    vm = float(np.median(hist_vols))
    if vm <= 0:
        return None

    ua = np.mean([x[1] for x in up]); da = np.mean([x[1] for x in dn])
    uv = np.mean([x[3] for x in up]) / vm; dv = np.mean([x[3] for x in dn]) / vm
    yz = np.nanmedian(arr["yz20"][j, lo:ti + 1])
    if not np.isfinite(yz):
        return None
    s1 = np.log(ua / da)
    if not np.isfinite(s1):
        return None

    # 점화부 = 창 후반 1/3 (거래일 위치 기준)
    cut = lo + (RES_W * 2) // 3
    lt = [x for x in w if x[4] >= cut]
    lu = [x for x in lt if x[0] > 0]
    ld = [x for x in lt if x[0] < 0]
    if lu and ld:
        s2n = np.mean([x[1] for x in ld]) / np.mean([x[1] for x in lu])
        luv = np.mean([x[3] for x in lu]) / vm
        ldv = np.mean([x[3] for x in ld]) / vm
        s3n = ldv / luv if luv > 0 else np.nan
    else:
        s2n, s3n = np.nan, np.nan

    # 과열: 마지막 상승파 진폭이 자기 이력(완결 상승파) 95백분위 이상
    hist_up = [x[1] for x in full_waves[j] if x[0] > 0 and x[4] < ti]
    last_up = next((x for x in reversed(r) if x[0] > 0), None)
    climax = int(len(hist_up) >= 10 and last_up is not None
                 and last_up[1] >= np.percentile(hist_up, 95))
    return s1, s2n, s3n, yz, climax


def vblk_at(arr, full_waves, tickers, ti):
    """판정일 ti 전 종목 점수표 (백분위는 그날 판정된 종목 내 횡단면)."""
    raw = {}
    for j in range(len(tickers)):
        sg = res_signals(arr, full_waves, j, ti)
        if sg:
            raw[tickers[j]] = sg
    if len(raw) < 30:
        return None
    d = pd.DataFrame(raw, index=["s1", "s2n", "s3n", "yz", "climax"]).T
    pct = lambda s: s.rank(pct=True) * 100
    d["p1"]  = pct(d["s1"])
    d["p2n"] = (100 - pct(d["s2n"])).fillna(50.0)      # 점화부 쌍 없으면 중립
    d["p3n"] = (100 - pct(d["s3n"])).fillna(50.0)
    d["p4"]  = pct(d["yz"])
    # Vblk — 블록 균등 33 : 17 : 17 : 33 (2026-08-13 Kane 채택)
    d["Vblk"] = (d["p1"] + (d["p2n"] + d["p3n"]) / 2 + d["p4"]) / 3
    d["rank"] = d["Vblk"].rank(ascending=False, method="min").astype(int)
    # 영역별 순위 (백분위 높을수록 1위) — 페이지 섹터표에서 쓴다
    for c in ("p1", "p2n", "p3n", "p4"):
        d[f"{c}_rank"] = d[c].rank(ascending=False, method="min").astype(int)
    d.index.name = "ticker"
    return d.reset_index()


# ---------------------------------------------------------------------------
# 일지 입출력
# ---------------------------------------------------------------------------
def read_log(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"ticker": str})


def append_log(path, new: pd.DataFrame, keys):
    old = read_log(path)
    if not old.empty:
        new = new[~new.set_index(keys).index.isin(old.set_index(keys).index)]
        if new.empty:
            return old, 0
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out = out.sort_values(list(keys)).reset_index(drop=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out, len(new)


def drop_dates(path, dates_str):
    """--force: 해당 날짜 행 제거 후 재기록."""
    old = read_log(path)
    if old.empty:
        return
    old = old[~old["date"].isin(dates_str)]
    old.to_csv(path, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# 산출물
# ---------------------------------------------------------------------------
def draw_graph(temp_log: pd.DataFrame, path=GRAPH, days=GRAPH_DAYS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = temp_log.tail(days).copy()
    d["date"] = pd.to_datetime(d["date"])
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(d["date"], d["hot100"], color="#9e9e9e", lw=2,
            label="W100/M6 (previous)")
    ax.plot(d["date"], d["hot60"], color="#d32f2f", lw=3,
            label="W60/M4 (adopted)")
    ax.axhline(HIST_MEDIAN, color="#1976D2", ls=":", lw=1.2)
    ax.set_ylabel("hot share (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(d["date"], d["tiger_close"], color="#000000", lw=1.2,
             label="TIGER 200 (102110, right)")
    ax2.set_ylabel("TIGER 200 close (KRW)")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    ax.set_title(f"Fever rule — nowcast temperature & TIGER 200 "
                 f"(last {len(d)} trading days)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def build_top_table(score_log, states_now, names, sectors, closes, dates_used):
    """D-2/D-1/D+0 상위 10 비교표. dates_used = [D-2, D-1, D+0] (문자열)"""
    d0 = dates_used[-1]
    cur = score_log[score_log["date"] == d0].copy()
    top = cur.nsmallest(10, "rank")["ticker"].tolist()

    piv_s = score_log.pivot_table(index="ticker", columns="date", values="Vblk")
    piv_r = score_log.pivot_table(index="ticker", columns="date", values="rank")
    prev_top = set()
    if len(dates_used) >= 2:
        p = score_log[score_log["date"] == dates_used[-2]]
        prev_top = set(p.nsmallest(10, "rank")["ticker"])

    rows = []
    for i, t in enumerate(top, 1):
        r = cur[cur["ticker"] == t].iloc[0]
        mark = "↑" if t not in prev_top else ""
        rows.append({
            "순위": i, "티커": t, "종목명": names.get(t, ""),
            "섹터": sectors.get(t, ""),
            "국면": states_now.get(t, "유보"),
            **{f"Vblk {d[5:]}": (round(piv_s.at[t, d], 1)
                                 if d in piv_s.columns and t in piv_s.index
                                 and pd.notna(piv_s.at[t, d]) else None)
               for d in dates_used},
            **{f"순위 {d[5:]}": (int(piv_r.at[t, d])
                                 if d in piv_r.columns and t in piv_r.index
                                 and pd.notna(piv_r.at[t, d]) else None)
               for d in dates_used},
            "p4단독": int(r["p4_rank"]),
            "종가": int(closes[t][0]) if t in closes else None,
            "등락률": closes[t][1] if t in closes else None,
            "과열": "●" if int(r["climax"]) else "",
            "재가열": "●" if int(r["reheat"]) else "",
            "신규": mark,
        })
    out = pd.DataFrame(rows)
    dropped = sorted(prev_top - set(top))
    return out, dropped


def build_all_table(score_log, names, sectors, core_set, closes, d0):
    """D+0 전 종목표 (8501 섹터별 화면용). 영역별 순위 포함."""
    cur = score_log[score_log["date"] == d0].copy()
    cur["name"] = cur["ticker"].map(names)
    cur["sector"] = cur["ticker"].map(sectors)
    cur["group"] = ["core" if t in core_set else "extend" for t in cur["ticker"]]
    cur["close"] = [closes[t][0] if t in closes else None for t in cur["ticker"]]
    cur["chg_pct"] = [closes[t][1] if t in closes else None for t in cur["ticker"]]
    ren = {"p1_rank": "rank_추세", "p4_rank": "rank_YZ",
           "p2n_rank": "rank_눌림진폭", "p3n_rank": "rank_눌림거래량"}
    cur = cur.rename(columns=ren)
    cols = ["ticker", "name", "sector", "group", "rank", "Vblk", "nowcast",
            "rank_추세", "rank_YZ", "rank_눌림진폭", "rank_눌림거래량",
            "climax", "reheat", "close", "chg_pct"]
    return cur[cols].sort_values("rank")


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", type=int, default=0,
                    help="온도 백필 거래일 수 (최초 1회 60 권장)")
    ap.add_argument("--force", action="store_true", help="같은 날짜 재계산")
    ap.add_argument("--no-graph", action="store_true")
    args = ap.parse_args()

    t0 = datetime.now()
    print(f"[발열률 워크포워드] {t0:%Y-%m-%d %H:%M:%S} KST")

    arr, tickers, dates, names, tiger, pf = load_panel()
    sectors, core_set = load_sector_map()
    N, T = len(tickers), len(dates)
    last = str(pd.Timestamp(dates[-1]).date())
    print(f"  유니버스 {N}종목 · 거래일 {T}개 · 최신 {last}")

    scaler = RobustScaler().fit(pf[FCOLS].values)
    Xp = scaler.transform(pf[FCOLS].values)
    cent = np.vstack([Xp[pf["state"].values == s].mean(axis=0) for s in (1, 2, 3)])

    full_waves = {j: build_waves(arr, j, 0, T - 1) for j in range(N)}

    # ---- 대상 날짜 ----
    n_temp = max(args.backfill, 1)
    temp_idx = list(range(max(T - n_temp, 0), T))
    score_idx = list(range(max(T - 3, 0), T))          # D-2, D-1, D+0
    if args.force:
        drop_dates(TEMP_LOG, [str(pd.Timestamp(dates[i]).date()) for i in temp_idx])
        drop_dates(SCORE_LOG, [str(pd.Timestamp(dates[i]).date()) for i in score_idx])

    done_t = set(read_log(TEMP_LOG).get("date", pd.Series(dtype=str)))
    done_s = set(read_log(SCORE_LOG).get("date", pd.Series(dtype=str)))

    # ---- 1) 온도 ----
    trows, srows_state = [], []
    todo = [i for i in temp_idx
            if str(pd.Timestamp(dates[i]).date()) not in done_t]
    if todo:
        print(f"  온도 판정 {len(todo)}일 …")
    for k, ti in enumerate(todo, 1):
        ds = str(pd.Timestamp(dates[ti]).date())
        s60 = nowcast_at(arr, full_waves, tickers, ti,
                         W_ADOPT, MINW_ADOPT, scaler, cent)
        s100 = nowcast_at(arr, full_waves, tickers, ti,
                          W_PREV, MINW_PREV, scaler, cent)
        a, b = temperature(s60, N), temperature(s100, N)
        s60 = s60.copy(); s60.insert(0, "date", ds)
        srows_state.append(s60[["date", "ticker", "state", "n_waves"]])
        tc = tiger.get(dates[ti], np.nan)
        trows.append(dict(date=ds, hot60=a["hot"], hot100=b["hot"],
                          장기60=a["장기"], 저변60=a["저변"], 단기60=a["단기"],
                          유보60=a["유보"], tiger_close=(int(tc) if pd.notna(tc) else None)))
        if k % 10 == 0 or k == len(todo):
            print(f"    {k}/{len(todo)}  {ds} 온도 {a['hot']}% (W100 {b['hot']}%)")

    temp_log, n_new_t = (append_log(TEMP_LOG, pd.DataFrame(trows), ["date"])
                         if trows else (read_log(TEMP_LOG), 0))
    if srows_state:
        append_log(STATE_LOG, pd.concat(srows_state, ignore_index=True),
                   ["date", "ticker"])
    state_log = read_log(STATE_LOG)
    st_hist = {d: dict(zip(g["ticker"], g["state"]))
               for d, g in state_log.groupby("date")} if not state_log.empty else {}

    # ---- 2) Vblk 점수 ----
    srows = []
    todo_s = [i for i in score_idx
              if str(pd.Timestamp(dates[i]).date()) not in done_s]
    for ti in todo_s:
        ds = str(pd.Timestamp(dates[ti]).date())
        sc = vblk_at(arr, full_waves, tickers, ti)
        if sc is None:
            print(f"    ⚠ {ds}: 판정 종목 30 미만 — skip")
            continue
        sc.insert(0, "date", ds)
        srows.append(sc)
        print(f"    {ds} 점수 {len(sc)}종목 (1위 {sc.nsmallest(1,'rank')['ticker'].iloc[0]})")

    if srows:
        sdf = pd.concat(srows, ignore_index=True)
        # 국면·재가열 주입 — 국면일지 기준 (최근 60거래일 내 '장기' 이력 &
        # 현재는 '장기'가 아님 = 식었다 다시 데워질 후보, resilience_score V1 정의)
        sdf["nowcast"] = [st_hist.get(d, {}).get(t, "유보")
                          for d, t in zip(sdf["date"], sdf["ticker"])]
        reheat = {}
        for ds in sdf["date"].unique():
            past = [d for d in sorted(st_hist) if d < ds][-REHEAT_DAYS:]
            ever_long = {t for d in past for t, s in st_hist[d].items()
                         if s == "장기"}
            reheat[ds] = ever_long
        sdf["reheat"] = [int(t in reheat.get(d, set()) and n != "장기")
                         for d, t, n in zip(sdf["date"], sdf["ticker"],
                                            sdf["nowcast"])]
        keep = ["date", "ticker", "Vblk", "rank", "p1", "p2n", "p3n", "p4",
                "p1_rank", "p2n_rank", "p3n_rank", "p4_rank",
                "nowcast", "climax", "reheat"]
        for c in ["Vblk", "p1", "p2n", "p3n", "p4"]:
            sdf[c] = sdf[c].round(1)
        sdf["climax"] = sdf["climax"].astype(int)
        score_log, n_new_s = append_log(SCORE_LOG, sdf[keep], ["date", "ticker"])
    else:
        score_log, n_new_s = read_log(SCORE_LOG), 0

    # ---- 3) 산출물 ----
    if not args.no_graph and not temp_log.empty:
        try:
            draw_graph(temp_log)
            print(f"  그래프 → {os.path.relpath(GRAPH, FM)}")
        except Exception as e:      # matplotlib 부재 등 — 일지·표는 계속 만든다
            print(f"  ⚠ 그래프 생성 실패 ({e}) — 일지·표는 정상 산출")

    dates_used = sorted(score_log["date"].unique())[-3:]
    closes = {}
    for j, t in enumerate(tickers):
        c0, c1 = arr["close"][j, T - 1], arr["close"][j, T - 2]
        if np.isfinite(c0) and np.isfinite(c1) and c1 > 0:
            closes[t] = (c0, round((c0 / c1 - 1) * 100, 1))
    st_now = st_hist.get(last, {})
    top, dropped = build_top_table(score_log, st_now, names, sectors,
                                   closes, dates_used)
    top.to_csv(TOP_CSV, index=False, encoding="utf-8-sig")

    allt = build_all_table(score_log, names, sectors, core_set, closes,
                           dates_used[-1])
    allt.to_csv(ALL_CSV, index=False, encoding="utf-8-sig")
    n_core = int((allt["group"] == "core").sum())
    print(f"  전체표 {len(allt)}종목 (core {n_core} / extend {len(allt)-n_core}) "
          f"→ {os.path.relpath(ALL_CSV, FM)}")

    # ---- 콘솔 요약 ----
    tl = temp_log.tail(2)
    if len(tl) >= 1:
        cur = tl.iloc[-1]
        prev = tl.iloc[0] if len(tl) == 2 else cur
        spread = cur["hot60"] - cur["hot100"]      # 음수 = 냉각 진행
        arrow = "▲" if cur["hot60"] > prev["hot60"] else ("▼" if cur["hot60"] < prev["hot60"] else "—")
        print()
        print(f"  온도 {cur['hot60']}% {arrow} (전일 {prev['hot60']}%) | "
              f"W100 {cur['hot100']}% | 스프레드 {spread:+.1f}%p")
        print(f"  구성: 장기 {int(cur['장기60'])} / 저변 {int(cur['저변60'])} / "
              f"단기 {int(cur['단기60'])} / 유보 {int(cur['유보60'])}")
    print(f"  일지 append: 온도 {n_new_t}행 · 점수 {n_new_s}행")
    newcomers = top[top["신규"] == "↑"]["종목명"].tolist()
    if newcomers:
        print(f"  신규 진입: {', '.join(newcomers)}")
    if dropped:
        print(f"  이탈: {', '.join(names.get(t, t) for t in dropped)}")
    print(f"  상위10 → {os.path.relpath(TOP_CSV, FM)}")
    print(f"  소요 {(datetime.now() - t0).total_seconds():.0f}초")


if __name__ == "__main__":
    main()
