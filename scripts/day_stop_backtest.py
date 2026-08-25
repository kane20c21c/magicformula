#!/usr/bin/env python3
"""
day_stop_backtest.py — 데이 포트 청산규칙 분봉 백테스트 (고정 1% vs 변동배율)

**엔진을 그대로 모사한다** (StockPortfolio/app/paper_day/engine.py):
  - 매수  : t일 17:00 NXT → 체결가는 **t일 종가**로 근사 (실거래 실측 괴리 ~0.3%)
  - 감시  : t+1일부터 09:07~15:17 **10분 간격** 샘플 (config.market_open~close 내)
  - day_peak = 그날 샘플된 최고가, **매일 장 시작 시 리셋** (첫 관측이 그날 시작 peak)
  - 갭관문: 그날 **첫 관측**에서만 평단 대비 ≤ −5% 이면 트리거
  - 손절  : 샘플가 ≤ day_peak × (1 − 손절폭) 이면 트리거
  - 체결  : 트리거 **+60초** 시점 분봉가로 전량 매도
  - 미트리거 시 오버나이트 보유 (강제청산 없음)

비교 규칙:
  R0 고정   : 손절폭 = base (현행 1.0%)
  R1 변동배율: 손절폭 = clip(base × 배율, lo, hi)
              배율 = YZ_20(신호일 t) ÷ D(t)
              D(t) = 직전 252거래일 · 유니버스 YZ_20 일별 중앙값의 중앙값 (절대척도)

⚠ 룩어헤드 없음 — 배율은 t일 종가로 확정되는 YZ_20(t) 만 쓴다 (16:00 배치 이후 확정,
  17:00 매수 시점에 알 수 있는 값). 손절은 t+1 이후에만 적용된다.
⚠ 이 백테스트는 **손절규칙의 효과만** 분리해 잰다. 종목 선택(모델 신호)은 개입하지
  않고 "매 거래일 종가 매수 → 손절까지 보유"를 반복한다. 전략 손익 추정치가 아니다.

사용:
    python3 scripts/day_stop_backtest.py --tickers 319660,068270
    python3 scripts/day_stop_backtest.py --tickers 319660 --base 1.0 --clip 0.4,2.5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# LLV 는 StoLab/ 아래 형제 저장소 — 머신(미니/에어) 무관 상대 경로.
# 빈 값은 미설정 취급(`or`), "~/..." 표기는 expanduser 로 편다.
STOLAB_ROOT = Path(__file__).resolve().parents[2]   # StoLab/
LLV = Path(os.getenv("LLV_PATH") or str(STOLAB_ROOT / "LongLiveVault")).expanduser()
STUDY = Path(__file__).resolve().parents[1] / "output" / "day_stop_study"

# ── 엔진 상수 (StockPortfolio/app/paper_day/config.py 와 동기) ──
SAMPLE_TIMES = [f"{h:02d}{m:02d}00" for h in range(9, 16) for m in (7, 17, 27, 37, 47, 57)]
SAMPLE_TIMES = [t for t in SAMPLE_TIMES if "090700" <= t <= "151700"]
SELL_DELAY_MIN = 1          # 60초 → 1분봉 뒤
GAP_STOP_PCT = 0.05         # 갭 관문 (이번 개편 대상 아님 — 양 규칙 공통)
BUY_FEE, SELL_FEE, SELL_TAX = 0.00015, 0.00015, 0.002
MAX_HOLD_TD = 10            # 안전장치 (실측 보유 1거래일)


def yz_k(n: int) -> float:
    return 0.34 / (1.34 + (n + 1) / (n - 1))


def load_scale(tickers: list[str], window: int = 252) -> pd.DataFrame:
    """LLV 정본 YZ_20 + 유니버스 절대척도 분모 D(t) → 배율."""
    o = LLV / "data" / "ohlcv"
    d = pd.concat([pd.read_parquet(o / "core.parquet"),
                   pd.read_parquet(o / "extend.parquet")], ignore_index=True)
    d["YZ_20"] = pd.to_numeric(d["YZ_20"], errors="coerce")
    d = d[d["Volume"] > 0]
    daily = d.groupby("Date")["YZ_20"].median()
    D = daily.rolling(window, min_periods=int(window * 0.8)).median()
    s = d[d["Ticker"].isin(tickers)][["Ticker", "Date", "Close", "YZ_20"]].copy()
    s["D"] = s["Date"].map(D)
    s["scale"] = s["YZ_20"] / s["D"]
    return s.dropna(subset=["scale"]).sort_values(["Ticker", "Date"])


def load_minutes(ticker: str) -> dict[str, dict]:
    out = {}
    for f in sorted(STUDY.glob(f"minute_{ticker}_*.json")):
        out[f.stem.split("_")[-1]] = json.loads(f.read_text(encoding="utf-8"))
    return out


def _price_at(bars: dict, hhmmss: str) -> float | None:
    """해당 분봉, 없으면 직전 가장 가까운 분봉 (엔진의 현재가 조회와 같은 성격)."""
    if hhmmss in bars:
        return float(bars[hhmmss]["p"])
    prior = [t for t in bars if t <= hhmmss]
    return float(bars[max(prior)]["p"]) if prior else None


def simulate(entry_close: float, day_keys: list[str], minutes: dict,
             stop_pct: float) -> dict | None:
    """t+1일부터 감시. 트리거 시 +1분봉가로 매도. 반환 None = 기간 내 미청산."""
    avg = entry_close
    last_px, last_dk, px, last_hold = None, None, None, 0
    for hold_i, dk in enumerate(day_keys[:MAX_HOLD_TD], start=1):
        bars = minutes.get(dk)
        if not bars:
            break          # ⚠ continue 금지 — 캐시 구멍을 건너뛰면 먼 미래 분봉에 팔게 된다
        day_peak = None          # 매일 리셋 — 그날 첫 관측이 시작 peak
        for st in SAMPLE_TIMES:
            px = _price_at(bars, st)
            if px is None:
                continue
            first_obs = day_peak is None          # 그날의 첫 유효 관측인가
            day_peak = px if first_obs else max(day_peak, px)

            trig = None
            # ① 갭 관문 — 그날 첫 관측에서만 1회 (엔진 first_obs_today 와 동일)
            if first_obs and GAP_STOP_PCT > 0 and (px / avg - 1) <= -GAP_STOP_PCT:
                trig = "gap"
            # ② 당일 고점 대비 (첫 관측은 px == day_peak 라 어차피 발동 안 함)
            if trig is None and (px - day_peak) / day_peak <= -stop_pct:
                trig = "peak"
            if trig:
                hh, mm = int(st[:2]), int(st[2:4])
                mm += SELL_DELAY_MIN
                if mm >= 60:
                    hh, mm = hh + 1, mm - 60
                sell = _price_at(bars, f"{hh:02d}{mm:02d}00") or px
                return _close(avg, sell, dk, st, trig, hold_i)
        last_px, last_dk, last_hold = px, dk, hold_i

    # 창 내 미트리거 — 마지막 관측가로 강제 마감 (쌍대비교에서 표본이 빠지지 않게)
    if last_px is not None:
        return _close(avg, last_px, last_dk, SAMPLE_TIMES[-1], "none", last_hold)
    return None


def _close(avg: float, sell: float, dk: str, st: str, kind: str, hold: int) -> dict:
    gross_in = avg * (1 + BUY_FEE)
    gross_out = sell * (1 - SELL_FEE - SELL_TAX)
    return {"exit_date": dk, "exit_time": st, "kind": kind, "hold_td": hold,
            "sell": sell, "ret_pct": (sell / avg - 1) * 100,
            "ret_net_pct": (gross_out / gross_in - 1) * 100}


def run(ticker: str, base: float, clip: tuple[float, float],
        scale_df: pd.DataFrame) -> pd.DataFrame:
    minutes = load_minutes(ticker)
    have = sorted(minutes.keys())
    idx = {d: i for i, d in enumerate(have)}
    s = scale_df[scale_df["Ticker"] == ticker].reset_index(drop=True)
    rows: list[dict] = []
    skipped: list[tuple] = []
    for i in range(len(s) - 1):
        r = s.iloc[i]
        entry_key = r["Date"].strftime("%Y%m%d")
        # ⚠ 진입일도 분봉 보유 구간 안이어야 한다 — 아니면 과거 종가에 사서
        #   먼 미래 분봉에 파는 유령 에피소드가 생긴다 (2026-08-17 버그 수정).
        if entry_key not in idx:
            continue
        # 전진일은 **LLV 거래일 연속**으로 잡는다 (분봉 캐시 목록으로 잡으면
        # 미수집 구간을 건너뛰어 먼 미래에 파는 유령 에피소드가 생긴다).
        fwd = [d.strftime("%Y%m%d")
               for d in s["Date"].iloc[i + 1: i + 1 + MAX_HOLD_TD]]
        if not fwd or fwd[0] not in idx:
            continue
        # ⚠ 진입가는 **분봉계**로 통일한다. LLV Close 는 수정주가, KIS 분봉은 원주가라
        #   자본변동(분할·증자) 종목에서 가격계가 어긋나 유령 수익률이 나온다.
        #   (17:00 NXT 매수의 근사로 그날 마지막 분봉가를 쓴다 — 실측 괴리 ~0.3%)
        eb = minutes.get(entry_key) or {}
        if not eb:
            continue
        entry_px = float(eb[max(eb)]["p"])
        adj = float(r["Close"])
        if adj > 0 and abs(entry_px / adj - 1) > 0.20:
            skipped.append((entry_key, entry_px, adj))   # 자본변동 의심 — 제외
            continue
        sc = float(r["scale"])
        for tag, pct in (("R0_고정", base / 100.0),
                         ("R1_배율", float(np.clip(base * sc, *clip)) / 100.0)):
            res = simulate(entry_px, fwd, minutes, pct)
            if res is None:
                continue
            rows.append({"ticker": ticker, "entry": entry_key, "rule": tag,
                         "scale": round(sc, 3), "stop_pct": round(pct * 100, 3), **res})
    if skipped:
        print(f"  ⚠ [{ticker}] 수정주가/원주가 괴리 >20%% 로 제외 {len(skipped)}일 "
              f"(예: {skipped[0][0]} 분봉 {skipped[0][1]:,.0f} vs 수정 {skipped[0][2]:,.0f})")
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", required=True)
    ap.add_argument("--base", type=float, default=1.0, help="기준 손절폭 %% (현행 1.0)")
    ap.add_argument("--clip", default="0.4,2.5", help="손절폭 하한,상한 %%")
    a = ap.parse_args()

    tickers = [t.strip() for t in a.tickers.split(",") if t.strip()]
    lo, hi = (float(x) for x in a.clip.split(","))
    sdf = load_scale(tickers)

    allr = []
    for tk in tickers:
        df = run(tk, a.base, (lo, hi), sdf)
        if df.empty:
            print(f"[{tk}] 에피소드 없음 — 분봉 캐시 확인")
            continue
        allr.append(df)
        nm = sdf[sdf.Ticker == tk]
        print(f"\n{'='*72}\n■ {tk}  배율 중앙 {nm.scale.median():.2f} "
              f"(최근 {nm.scale.iloc[-1]:.2f})  에피소드 {len(df)//2}쌍")
        g = df.groupby("rule").agg(
            n=("ret_net_pct", "size"), 손절폭=("stop_pct", "median"),
            평균=("ret_net_pct", "mean"), 중앙=("ret_net_pct", "median"),
            승률=("ret_net_pct", lambda s: (s > 0).mean() * 100),
            표준편차=("ret_net_pct", "std"), 최악=("ret_net_pct", "min"),
            최선=("ret_net_pct", "max"),
            보유일=("hold_td", "mean"), 청산시각=("exit_time", lambda s: s.mode().iloc[0]))
        print(g.round(3).to_string())
        # 쌍대 비교 (같은 진입일끼리)
        p = df.pivot_table(index="entry", columns="rule", values="ret_net_pct")
        p = p.dropna()
        if len(p) > 5 and p.shape[1] == 2:
            d = p["R1_배율"] - p["R0_고정"]
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std(ddof=1) > 0 else 0.0
            print(f"  쌍대차이(R1−R0) 평균 {d.mean():+.3f}%p  중앙 {d.median():+.3f}%p  "
                  f"n={len(d)}  t={t:+.2f}  R1우세 {100*(d>0).mean():.1f}%")
        print("  청산시각 분포(상위5):")
        print(df.groupby("rule").exit_time.apply(
            lambda s: ", ".join(f"{k[:2]}:{k[2:4]}({v})" for k, v in s.value_counts().head(5).items())
        ).to_string())

    if allr:
        out = pd.concat(allr, ignore_index=True)
        f = STUDY / "backtest_episodes.csv"
        out.to_csv(f, index=False, encoding="utf-8-sig")
        print(f"\n에피소드 원장 저장: {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
