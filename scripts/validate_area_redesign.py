"""
scripts/validate_area_redesign.py
=================================
황금률 영역 재설계(2026-08-24 리포트 §5·§14) 검증 하네스.

08-24 재검증 세션의 분석 스크립트가 남아 있지 않아, 리포트 수치를 재현하고
개선안을 검증하기 위해 다시 만든 것이다. 판정 기준은 리포트와 동일하다.

평가 기준
---------
- 매일 종합점수 상위 q% 매수 → 20거래일 상대수익
- 상대수익 = 종목 20일 수익률 − 그날 유니버스 전체 20일 수익률 평균
- 구간 = 2023-05-02 이후 (52주 위치 min_periods=60 워밍업)

⚠ 한계 (리포트 §15 와 동일)
- 유니버스가 현재 명단이라 생존편향이 있다. 여기서 나오는 수치는 확정값이 아니다.
- 2023-05~2026-08 은 BM CAGR +45% 의 초강세장이다.

실행
----
    python3 scripts/validate_area_redesign.py build     # 패널 캐시 생성
    python3 scripts/validate_area_redesign.py rsi       # RSI 구간별 수익 (§8 재현)
    python3 scripts/validate_area_redesign.py compare   # 설계 비교 (§5·§14 재현)
    python3 scripts/validate_area_redesign.py dist      # 종합점수 분포·신호 개수
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from magic_formula.indicators import _rsi, _clip                    # noqa: E402
from magic_formula.analysis import area_scores as AS                # noqa: E402
from magic_formula.analysis import volatility_variants as VLV       # noqa: E402

VAULT = PROJECT_ROOT.parent / "LongLiveVault"
OHLCV_DIR = VAULT / "data" / "ohlcv"
CACHE = PROJECT_ROOT / "output" / "area_redesign_panel.parquet"

START = "2023-05-02"     # 52주 위치 워밍업 이후
FWD_H = 20               # 평가 지평 (거래일)


# ===========================================================================
# 후보 점수 함수
# ===========================================================================

def vol_current(df: pd.DataFrame, rg: pd.Series) -> pd.Series:
    """현행 — BB×52주×레짐 결합 점수표."""
    return VLV.score_joint_regime(df, rg)


def vol_a2(df: pd.DataFrame, rg: pd.Series) -> pd.Series:
    """A2 — 52주 위치(0.6) + BB %B 꺾임 반영(0.4). ±10. 레짐 무관."""
    p52 = VLV._pos_52w(df)
    bb = VLV._bb_pctb(df)
    s52 = (p52 - 0.5) * 20.0
    sbb = pd.Series(
        np.select(
            [bb > 1.0, bb >= 0.8, bb >= 0.6, bb >= 0.2, bb >= 0.0],
            [3.0, 10.0, 5.0, 0.0, -3.0],
            default=2.0,
        ),
        index=df.index,
    )
    return _clip(0.6 * s52 + 0.4 * sbb).fillna(0.0)


def mom_current(df: pd.DataFrame) -> pd.Series:
    """현행 — RSI(14) 5구간 계단, ≥90 에 +10."""
    return AS.score_momentum(df)


def _mom_from_table(df: pd.DataFrame, table: list[tuple[float, float]]) -> pd.Series:
    """table = [(하한, 점수), ...] 내림차순. 하한 이상이면 그 점수."""
    if len(df) < 35:
        return pd.Series(0.0, index=df.index)
    rsi = _rsi(df["Close"])
    s = pd.Series(np.nan, index=df.index)
    v = rsi.notna()
    s.loc[v] = 0.0
    for lo, sc in table:
        s.loc[v & (rsi >= lo)] = sc
    return _clip(s).fillna(0.0)


# 3순위 후보들 — 리포트가 정확한 수치를 주지 않아 직접 재서 정한다.
def mom_v2a(df):
    """5구간 유지, 최고점만 70~90 으로 이동 (≥90 은 +5)."""
    return _mom_from_table(df, [(30, -5.0), (50, 0.0), (70, 10.0), (90, 5.0)])


def mom_v2b(df):
    """RSI 70 이진 문 — §8 이 말한 '이 영역의 실체'."""
    return _mom_from_table(df, [(70, 10.0)])


def mom_v2c(df):
    """v2a + 과매도 감점 제거 (10~30 실측 수익이 양수라)."""
    return _mom_from_table(df, [(50, 0.0), (70, 10.0), (90, 5.0)])


def mom_v2d(df):
    """연속 — RSI 를 ±10 으로 선형 매핑. (RSI−50)/5. 자유 파라미터 없음."""
    if len(df) < 35:
        return pd.Series(0.0, index=df.index)
    return _clip((_rsi(df["Close"]) - 50.0) / 5.0).fillna(0.0)


def mom_v2(df: pd.DataFrame) -> pd.Series:
    """3순위 채택안 — sweep 결과 연속 선형이 계단보다 우월 (k 3.5~6 평탄)."""
    return mom_v2d(df)


# ===========================================================================
# 패널 생성
# ===========================================================================

def load_panel() -> pd.DataFrame:
    frames = []
    for name in ("core.parquet", "extend.parquet"):
        p = OHLCV_DIR / name
        if not p.exists():
            raise FileNotFoundError(p)
        frames.append(pd.read_parquet(p))
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def build() -> pd.DataFrame:
    raw = load_panel()
    stock_data: dict[str, pd.DataFrame] = {}
    for tk, g in raw.groupby("Ticker", sort=False):
        d = g.set_index("Date").sort_index()
        if len(d) >= 120:
            stock_data[tk] = d
    print(f"종목 {len(stock_data)}개 · {raw['Date'].min().date()} ~ {raw['Date'].max().date()}")

    rg_breadth, rg_quick = AS.make_regimes(stock_data)
    print("레짐 산출 완료")

    rows = []
    for i, (tk, d) in enumerate(stock_data.items(), 1):
        out = pd.DataFrame(index=d.index)
        out["Ticker"] = tk
        out["Close"] = d["Close"]
        out["Open"] = d["Open"]
        out["trend"] = AS.score_trend(d, rg_breadth)
        out["volume"] = AS.score_volume(d, rg_quick)
        out["mom_cur"] = mom_current(d)
        out["mom_v2a"] = mom_v2a(d)
        out["mom_v2b"] = mom_v2b(d)
        out["mom_v2c"] = mom_v2c(d)
        out["mom_v2d"] = mom_v2d(d)
        out["vol_cur"] = vol_current(d, rg_quick)
        out["vol_a2"] = vol_a2(d, rg_quick)
        out["rsi"] = _rsi(d["Close"])
        out["yz20"] = d["YZ_20"] if "YZ_20" in d.columns else np.nan
        out["pos52"] = VLV._pos_52w(d)
        out["bb"] = VLV._bb_pctb(d)
        out["phase"] = d["Wyckoff_Phase"] if "Wyckoff_Phase" in d.columns else np.nan
        out["fwd"] = d["Close"].shift(-FWD_H) / d["Close"] - 1.0
        rows.append(out.reset_index())
        if i % 50 == 0:
            print(f"  {i}/{len(stock_data)}")

    panel = pd.concat(rows, ignore_index=True)
    panel = panel[panel["Date"] >= START].reset_index(drop=True)
    # 상대수익 = 종목 fwd − 그날 유니버스 평균 fwd
    panel["fwd_rel"] = panel["fwd"] - panel.groupby("Date")["fwd"].transform("mean")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(CACHE, index=False)
    print(f"저장 {CACHE} · {len(panel):,}행")
    return panel


def get_panel() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    return build()


# ===========================================================================
# 결합 · 평가
# ===========================================================================

def composite(panel: pd.DataFrame, w: dict[str, float],
              vol_col: str, mom_col: str, gate: bool = True) -> pd.Series:
    wsum = w["trend"] + w["momentum"] + w["volume"] + w["volatility"]
    acc = (w["trend"] * panel["trend"] + w["momentum"] * panel[mom_col]
           + w["volume"] * panel["volume"] + w["volatility"] * panel[vol_col])
    comp = (acc / wsum).clip(-10, 10)
    if gate:
        comp = comp.where(~panel["phase"].isin(["Markdown"]))
    return comp


def struct_2stage(panel: pd.DataFrame, rsi_gate: float = 70.0) -> pd.Series:
    """
    리포트 §9 4순위 — 3지표 2단 구조.

        1단 문   : RSI(14) >= 70
        2단 정렬 : min( 백분위(52주 위치), 백분위(YZ_20 배율) )

    ⚠ YZ 배율 = YZ_20 / YZ_20_BM 인데, BM 값은 그날 전 종목에 같은 상수라
      날짜별 백분위를 취하면 배율이든 원값이든 순위가 동일하다. 그래서
      YZ_20 원값의 날짜별 백분위를 쓴다 (BM 시계열 불필요).

    반환값은 ±10 이 아니라 0~1 백분위다 — 임계 6.0 이 적용되지 않는다.
    """
    p52q = panel.groupby("Date")["pos52"].rank(pct=True)
    yzq = panel.groupby("Date")["yz20"].rank(pct=True)
    s = np.minimum(p52q, yzq)
    return s.where(panel["rsi"] >= rsi_gate)


def top_q_return(panel: pd.DataFrame, comp: pd.Series, q: float) -> pd.Series:
    """매일 상위 q 분위 선택 → 선택된 행의 20일 상대수익."""
    d = pd.DataFrame({"Date": panel["Date"], "s": comp,
                      "r": panel["fwd_rel"]}).dropna()
    thr = d.groupby("Date")["s"].transform(lambda x: x.quantile(1 - q))
    return d.loc[d["s"] >= thr, ["Date", "r"]].set_index("Date")["r"]


def summarize(name: str, panel: pd.DataFrame, comp: pd.Series, q: float) -> dict:
    from scipy import stats
    r = top_q_return(panel, comp, q)
    yr = r.groupby(r.index.year).mean() * 100
    t, p = stats.ttest_1samp(r.values, 0.0) if len(r) > 2 else (np.nan, np.nan)
    return {"설계": name, "전 구간": r.mean() * 100,
            **{str(y): yr.get(y, np.nan) for y in (2023, 2024, 2025, 2026)},
            "부호": f"{int((yr > 0).sum())}/{len(yr)}", "n": len(r), "p": p}


W_CUR = {"trend": 0.2, "momentum": 0.2, "volume": 0.0, "volatility": 0.6}
W_NEW = {"trend": 0.0, "momentum": 0.8, "volume": 0.0, "volatility": 0.2}


def cmd_rsi(panel: pd.DataFrame) -> None:
    """§8 재현 — RSI(14) 구간별 20일 상대수익."""
    d = panel.dropna(subset=["rsi", "fwd_rel"])
    bins = [-np.inf, 10, 30, 50, 70, 90, np.inf]
    labels = ["≤10", "10~30", "30~50", "50~70", "70~90", "≥90"]
    g = d.groupby(pd.cut(d["rsi"], bins, labels=labels, right=False), observed=False)
    out = pd.DataFrame({"표본": g.size(),
                        "비중%": g.size() / len(d) * 100,
                        "20일 상대수익%": g["fwd_rel"].mean() * 100})
    print(out.round(2).to_string())


def cmd_compare(panel: pd.DataFrame) -> None:
    from scipy import stats  # noqa: F401
    cases = [
        ("현행 T20/M20/Vu0/Va60", W_CUR, "vol_cur", "mom_cur"),
        ("1순위 A2 (가중치 현행)", W_CUR, "vol_a2", "mom_cur"),
        ("1+2순위 A2 + M80/Va20", W_NEW, "vol_a2", "mom_cur"),
        ("1+2+3 v2a 최고점이동", W_NEW, "vol_a2", "mom_v2a"),
        ("1+2+3 v2b RSI70 이진문", W_NEW, "vol_a2", "mom_v2b"),
        ("1+2+3 v2c 과매도감점 제거", W_NEW, "vol_a2", "mom_v2c"),
        ("1+2+3 v2d 연속", W_NEW, "vol_a2", "mom_v2d"),
    ]
    for q in (0.03, 0.05, 0.10):
        print(f"\n=== 상위 {q:.0%} · 20거래일 상대수익(%) ===")
        rows = [summarize(n, panel, composite(panel, w, v, m), q)
                for n, w, v, m in cases]
        print(pd.DataFrame(rows).round(3).to_string(index=False))


def portfolio(panel: pd.DataFrame, comp: pd.Series,
              n_slot: int = 10, cost: float = 0.0043) -> dict:
    """20거래일마다 상위 n_slot 종목 균등매수. 리포트 §14 표와 같은 조건."""
    d = pd.DataFrame({"Date": panel["Date"], "Ticker": panel["Ticker"],
                      "s": comp, "r": panel["fwd"]}).dropna()
    # ⚠ 리밸런스 격자는 전략과 무관하게 전체 패널에서 잡는다.
    #   필터된 프레임에서 잡으면 전략마다 날짜가 달라져 비교가 깨진다.
    all_dates = np.sort(panel.loc[panel["fwd"].notna(), "Date"].unique())
    dates = all_dates[::FWD_H]
    rets, picks = [], {}
    for t in dates:
        g = d[d["Date"] == t].nlargest(n_slot, "s")
        picks[t] = list(g["Ticker"])
        if len(g) == 0:
            rets.append(0.0)
            continue
        rets.append(g["r"].mean() - cost)
    r = pd.Series(rets, index=pd.DatetimeIndex(dates))
    eq = (1 + r).cumprod()
    yrs = (dates[-1] - dates[0]) / np.timedelta64(365, "D")
    per_yr = 252 / FWD_H
    out = {"누적%": (eq.iloc[-1] - 1) * 100,
           "CAGR%": (eq.iloc[-1] ** (1 / yrs) - 1) * 100,
           "MDD%(리밸런스)": (eq / eq.cummax() - 1).min() * 100,
           "Sharpe": r.mean() / r.std() * np.sqrt(per_yr),
           "빈슬롯기": int((pd.Series(rets) == 0.0).sum())}
    out["MDD%(일별)"] = _daily_mdd(panel, picks, dates, cost) * 100
    return out


def _close_pivot(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.pivot_table(index="Date", columns="Ticker", values="Close")


def _daily_mdd(panel: pd.DataFrame, picks: dict, dates, cost: float) -> float:
    """보유 구간 안의 일별 자산 곡선으로 MDD 를 잰다 (리밸런스 시점만 보면 과소평가)."""
    px = _close_pivot(panel)
    idx = px.index
    curve, level = [], 1.0
    for t in dates:
        tk = picks.get(t, [])
        i0 = idx.get_loc(t)
        seg = idx[i0:i0 + FWD_H + 1]
        if not tk or len(seg) < 2:
            continue
        sub = px.loc[seg, tk]
        path = (sub / sub.iloc[0]).mean(axis=1) * (1 - cost)
        curve.append(path * level)
        level = float(path.iloc[-1] * level)
    if not curve:
        return np.nan
    eq = pd.concat(curve)
    eq = eq[~eq.index.duplicated(keep="last")].sort_index()
    return float((eq / eq.cummax() - 1).min())


def cmd_port(panel: pd.DataFrame) -> None:
    cases = [
        ("현행", composite(panel, W_CUR, "vol_cur", "mom_cur")),
        ("1순위 A2", composite(panel, W_CUR, "vol_a2", "mom_cur")),
        ("1+2순위", composite(panel, W_NEW, "vol_a2", "mom_cur")),
        ("1+2+3순위 (연속 모멘텀)", composite(panel, W_NEW, "vol_a2", "mom_v2d")),
        ("4순위 2단구조 (RSI70 문)", struct_2stage(panel)),
    ]
    print("=== 20거래일마다 상위 10종목 균등매수 · 왕복 0.43% ===")
    rows = [{"전략": n, **portfolio(panel, c)} for n, c in cases]
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print("\n=== 상위 3% · 20일 상대수익(%) ===")
    rows = [summarize(n, panel, c, 0.03) for n, c in cases]
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n=== 반기별 상위 3% (%) ===")
    tbl = {}
    for n, c in cases:
        r = top_q_return(panel, c, 0.03)
        half = r.index.year.astype(str) + "H" + ((r.index.month > 6).astype(int) + 1).astype(str)
        tbl[n] = r.groupby(half).mean() * 100
    print(pd.DataFrame(tbl).round(2).to_string())

    print("\n=== 4순위 RSI 문턱 민감도 (상위 3%) ===")
    rows = [summarize(f"RSI≥{g:.0f}", panel, struct_2stage(panel, g), 0.03)
            for g in (55, 60, 65, 70, 75, 80)]
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def slot_sim(panel: pd.DataFrame, score: pd.Series, thr: float,
             n_slot: int = 10, hold: int = FWD_H,
             cost: float = 0.0043) -> dict:
    """
    실제 슬롯 운용 시뮬레이션.

    - t일 종가 기준 점수가 thr 이상인 종목을 빈 슬롯만큼 상위부터 채운다
    - 체결은 t+1일 **시가** (라이브 조건 — CLAUDE.md 검증 함정 ①)
    - hold 거래일 뒤 시가 청산, 왕복 비용 cost
    - 이미 보유 중인 종목은 재진입하지 않는다
    """
    df = pd.DataFrame({"Date": panel["Date"], "Ticker": panel["Ticker"],
                       "s": score}).dropna()
    df = df[df["s"] >= thr]
    sig = {d: list(g.sort_values("s", ascending=False)["Ticker"])
           for d, g in df.groupby("Date")}

    op = panel.pivot_table(index="Date", columns="Ticker", values="Open")
    cl = panel.pivot_table(index="Date", columns="Ticker", values="Close")
    # 결측 종가는 직전값으로 — 안 하면 청산가가 NaN 이 되어 자산이 통째로 NaN 이 된다
    cl = cl.ffill()
    op = op.where(op.notna(), cl)
    idx = cl.index

    cash, held = 1.0, {}          # ticker -> (진입가, 청산예정 위치, 금액)
    eq, invested_days, n_trades = [], 0, 0

    for i, t in enumerate(idx):
        # 청산
        for tk in [k for k, v in held.items() if v[1] == i]:
            px_in, _, amt = held.pop(tk)
            px_out = op.at[t, tk]
            if pd.isna(px_out):
                px_out = px_in          # 가격 소실 — 본전 처리 (보수적)
            cash += amt * (px_out / px_in) * (1 - cost)
        # 진입 (전일 신호 → 오늘 시가)
        if i > 0:
            # ⚠ 슬롯 금액은 '현재 자산'의 1/n — 초기 자본 고정으로 두면 복리가 안 걸린다
            equity_now = cash + sum(
                a * ((op.at[t, k] / p0) if not pd.isna(op.at[t, k]) else 1.0)
                for k, (p0, _, a) in held.items())
            slot_amt = equity_now / n_slot
            for tk in sig.get(idx[i - 1], []):
                if len(held) >= n_slot:
                    break
                if tk in held:
                    continue
                px = op.at[t, tk] if tk in op.columns else np.nan
                if pd.isna(px) or px <= 0:
                    continue
                amt = min(slot_amt, cash)
                if amt <= 1e-9:
                    break
                cash -= amt
                held[tk] = (px, min(i + hold, len(idx) - 1), amt)
                n_trades += 1
        # 평가
        mv = sum(a * ((cl.at[t, k] / p) if not pd.isna(cl.at[t, k]) else 1.0)
                 for k, (p, _, a) in held.items())
        eq.append(cash + mv)
        invested_days += len(held)

    e = pd.Series(eq, index=idx)
    yrs = (idx[-1] - idx[0]).days / 365.25
    r = e.pct_change(fill_method=None).dropna()
    return {"임계": thr, "누적%": (e.iloc[-1] - 1) * 100,
            "CAGR%": (e.iloc[-1] ** (1 / yrs) - 1) * 100,
            "MDD%": (e / e.cummax() - 1).min() * 100,
            "Sharpe": r.mean() / r.std() * np.sqrt(252),
            "평균보유": invested_days / len(idx),
            "현금비중%": (1 - invested_days / len(idx) / n_slot) * 100,
            "매매횟수": n_trades}


def daily_ic(panel: pd.DataFrame, x: pd.Series) -> dict:
    """날짜별 횡단면 Spearman IC — 20일 상대수익 대비."""
    d = pd.DataFrame({"Date": panel["Date"], "x": x,
                      "y": panel["fwd_rel"]}).dropna()
    ic = d.groupby("Date").apply(
        lambda g: g["x"].corr(g["y"], method="spearman")
        if len(g) >= 20 else np.nan, include_groups=False).dropna()
    from scipy import stats
    t, p = stats.ttest_1samp(ic.values, 0.0)
    return {"평균 IC": ic.mean(), "IC 표준편차": ic.std(),
            "t값": t, "p": p, "양수일 비중%": (ic > 0).mean() * 100,
            "관측일": len(ic)}


def cmd_ic(panel: pd.DataFrame) -> None:
    """지표별 예측력 — 케인 질문: 이 재료들이 정말 20일 수익과 상관이 있나."""
    W_M100 = {"trend": 0.0, "momentum": 1.0, "volume": 0.0, "volatility": 0.0}
    items = [
        ("RSI(14) 원값", panel["rsi"]),
        ("52주 위치 원값", panel["pos52"]),
        ("BB %B 원값", panel["bb"]),
        ("YZ_20 원값", panel["yz20"]),
        ("─ 점수화 후 ─", None),
        ("모멘텀 점수 (연속)", panel["mom_v2d"]),
        ("모멘텀 점수 (현행 계단)", panel["mom_cur"]),
        ("변동성 A2 점수", panel["vol_a2"]),
        ("변동성 현행 점수", panel["vol_cur"]),
        ("추세 점수 (가중 0)", panel["trend"]),
        ("─ 결합 ─", None),
        ("종합 M100/Va0", composite(panel, W_M100, "vol_a2", "mom_v2d")),
        ("종합 M80/Va20 (채택안)", composite(panel, W_NEW, "vol_a2", "mom_v2d")),
        ("종합 현행", composite(panel, W_CUR, "vol_cur", "mom_cur")),
        ("2단구조", struct_2stage(panel)),
    ]
    rows = []
    for name, s in items:
        if s is None:
            rows.append({"지표": name})
            continue
        rows.append({"지표": name, **daily_ic(panel, s)})
    print("=== 날짜별 횡단면 Spearman IC · 20거래일 상대수익 ===")
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\n=== 재료끼리의 상관 (날짜별 횡단면 Spearman 평균) ===")
    cols = {"RSI": "rsi", "52주위치": "pos52", "BB%B": "bb", "YZ": "yz20"}
    m = pd.DataFrame(index=list(cols), columns=list(cols), dtype=float)
    for a, ca in cols.items():
        for b, cb in cols.items():
            if a == b:
                m.loc[a, b] = 1.0
                continue
            d = panel[["Date", ca, cb]].dropna()
            m.loc[a, b] = d.groupby("Date").apply(
                lambda g, x=ca, y=cb: g[x].corr(g[y], method="spearman"),
                include_groups=False).mean()
    print(m.round(3).to_string())

    print("\n=== 10분위별 20일 상대수익(%) — 단조인가, 꼭대기만인가 ===")
    dec = {}
    for name, s in [("현행", composite(panel, W_CUR, "vol_cur", "mom_cur")),
                    ("M80/Va20", composite(panel, W_NEW, "vol_a2", "mom_v2d")),
                    ("모멘텀 단독", panel["mom_v2d"]),
                    ("변동성 A2 단독", panel["vol_a2"])]:
        d = pd.DataFrame({"Date": panel["Date"], "s": s,
                          "y": panel["fwd_rel"]}).dropna()
        q = d.groupby("Date")["s"].transform(
            lambda x: pd.qcut(x.rank(method="first"), 10, labels=False))
        dec[name] = d.groupby(q)["y"].mean() * 100
    t = pd.DataFrame(dec)
    t.index = [f"{i+1}분위" + (" (최상)" if i == 9 else "") for i in t.index]
    print(t.round(2).to_string())

    print("\n=== 52주위치·BB 를 빼면 실제로 손해인가 (상위 3% · 실운용) ===")
    rows = []
    for name, w, v in [("RSI 단독 M100", W_M100, "vol_a2"),
                       ("M80 + A2 Va20 (채택안)", W_NEW, "vol_a2"),
                       ("M80 + 현행 변동성 Va20", W_NEW, "vol_cur")]:
        s = composite(panel, w, v, "mom_v2d")
        rows.append({"설계": name,
                     "상위3% 상대수익%": summarize("x", panel, s, 0.03)["전 구간"],
                     **slot_sim(panel, s, s.quantile(0.97))})
    print(pd.DataFrame(rows).round(2).to_string(index=False))


def cmd_thr(panel: pd.DataFrame) -> None:
    """임계별 실운용 성과 — 슬롯 채움까지 반영."""
    s2 = struct_2stage(panel)
    print("=== 2단구조 · 슬롯 10개 · 20거래일 보유 · t+1 시가 · 왕복 0.43% ===")
    rows = [slot_sim(panel, s2, t) for t in
            (0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97)]
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print("\n=== 슬롯 수 비교 (임계 0.90) ===")
    rows = [{**slot_sim(panel, s2, 0.90, n_slot=n), "슬롯": n}
            for n in (5, 8, 10, 12, 15, 20)]
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print("\n=== 같은 조건에서 다른 설계 (임계는 각자 상위 3% 수준) ===")
    cur = composite(panel, W_CUR, "vol_cur", "mom_cur")
    n12 = composite(panel, W_NEW, "vol_a2", "mom_cur")
    n123 = composite(panel, W_NEW, "vol_a2", "mom_v2d")
    rows = []
    for name, sc, thr in [("현행 (6.0)", cur, 6.0),
                          ("현행 (상위3%)", cur, cur.quantile(0.97)),
                          ("1+2순위 (상위3%)", n12, n12.quantile(0.97)),
                          ("1+2+3순위 (상위3%)", n123, n123.quantile(0.97)),
                          ("2단구조 (0.90)", s2, 0.90)]:
        rows.append({"설계": name, **slot_sim(panel, sc, thr)})
    print(pd.DataFrame(rows).round(2).to_string(index=False))


def cmd_halves(panel: pd.DataFrame) -> None:
    """반기별 — 리포트 §15 가 지목한 2026H2 취약 구간 확인."""
    cases = [
        ("현행", W_CUR, "vol_cur", "mom_cur"),
        ("1순위 A2", W_CUR, "vol_a2", "mom_cur"),
        ("1+2순위", W_NEW, "vol_a2", "mom_cur"),
        ("1+2+3순위", W_NEW, "vol_a2", "mom_v2d"),
    ]
    tbl = {}
    for n, w, v, m in cases:
        r = top_q_return(panel, composite(panel, w, v, m), 0.03)
        half = r.index.year.astype(str) + "H" + ((r.index.month > 6).astype(int) + 1).astype(str)
        tbl[n] = r.groupby(half).mean() * 100
    print("=== 상위 3% · 반기별 20일 상대수익(%) ===")
    print(pd.DataFrame(tbl).round(2).to_string())


def cmd_sweep(panel: pd.DataFrame) -> None:
    """연속 모멘텀의 기울기 민감도 — 과최적화 점검."""
    for q in (0.03, 0.10):
        print(f"\n=== 상위 {q:.0%} · 기울기별 (RSI−50)/k ===")
        rows = []
        for k in (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0):
            m = ((panel["rsi"] - 50.0) / k).clip(-10, 10).fillna(0.0)
            tmp = panel.copy()
            tmp["_m"] = m
            rows.append(summarize(f"k={k}", tmp,
                                  composite(tmp, W_NEW, "vol_a2", "_m"), q))
        rows.append(summarize("계단(현행)", panel,
                              composite(panel, W_NEW, "vol_a2", "mom_cur"), q))
        print(pd.DataFrame(rows).round(3).to_string(index=False))


def cmd_dist(panel: pd.DataFrame) -> None:
    cases = [
        ("현행", W_CUR, "vol_cur", "mom_cur"),
        ("1순위 A2", W_CUR, "vol_a2", "mom_cur"),
        ("1+2순위", W_NEW, "vol_a2", "mom_cur"),
        ("1+2+3순위", W_NEW, "vol_a2", "mom_v2d"),
    ]
    rows = []
    for n, w, v, m in cases:
        c = composite(panel, w, v, m)
        d = pd.DataFrame({"Date": panel["Date"], "s": c}).dropna()
        n_days = d["Date"].nunique()
        rows.append({
            "설계": n, "평균": c.mean(), "표준편차": c.std(),
            "p90": c.quantile(0.90), "p97": c.quantile(0.97), "p99": c.quantile(0.99),
            "최대": c.max(),
            "일평균 ≥6.0": (d["s"] >= 6.0).sum() / n_days,
            "일평균 ≥5.0": (d["s"] >= 5.0).sum() / n_days,
            "≥6.0 비중%": (d["s"] >= 6.0).mean() * 100,
        })
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n=== 임계별 일평균 신호 종목 수 ===")
    grid = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
    rows = []
    for n, w, v, m in cases:
        c = composite(panel, w, v, m)
        d = pd.DataFrame({"Date": panel["Date"], "s": c}).dropna()
        nd = d["Date"].nunique()
        rows.append({"설계": n, **{f"≥{t}": round((d["s"] >= t).sum() / nd, 1)
                                  for t in grid}})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== 현행과 같은 신호량(일 11.4종목)이 되는 임계 ===")
    base = composite(panel, W_CUR, "vol_cur", "mom_cur")
    bd = pd.DataFrame({"Date": panel["Date"], "s": base}).dropna()
    target = (bd["s"] >= 6.0).sum() / bd["Date"].nunique()
    tgt_c = (bd["s"] >= 5.0).sum() / bd["Date"].nunique()
    for n, w, v, m in cases:
        c = composite(panel, w, v, m)
        d = pd.DataFrame({"Date": panel["Date"], "s": c}).dropna()
        nd = d["Date"].nunique()
        thr = d["s"].quantile(1 - target * nd / len(d))
        thc = d["s"].quantile(1 - tgt_c * nd / len(d))
        print(f"  {n:12s} 진입 임계 {thr:5.2f} (현행 6.0 대응) · "
              f"후보 임계 {thc:5.2f} (현행 5.0 대응)")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "compare"
    if cmd == "build":
        build()
        return
    panel = get_panel()
    {"rsi": cmd_rsi, "compare": cmd_compare, "dist": cmd_dist,
     "sweep": cmd_sweep, "halves": cmd_halves, "port": cmd_port,
     "thr": cmd_thr, "ic": cmd_ic}[cmd](panel)


if __name__ == "__main__":
    main()
