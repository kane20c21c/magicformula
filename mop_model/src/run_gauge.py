# -*- coding: utf-8 -*-
"""
run_gauge.py — 15:10 잠정 섹터 오버나이트 게이지 (Kane 지시 2026-07-30)
========================================================================
장 마감 전(15:10) 현재가를 **임시 종가**로 주입해 MOp 챔피언 모델을 그대로
돌리고, Kane 지정 섹터 세트(gauge_config.SECTOR_SETS)별로 갭1(익일시가/당일종가)
상대 상승확률의 **시가총액 가중평균**을 계산해 15:15 메일+푸시로 보고한다.

파이프라인 (전부 인메모리 — 운영 산출물 무오염):
  1) LLV core+extend 로드, 당일 행 제거 (있다면)
  2) KIS 잠정 스냅샷 (data_service.fetch_today_ohlcv_snapshot, ~195종목 ≈ 80초)
  3) LLV 지표 37컬럼 재계산 (data_service.compute_indicators_frame, ≈ 10초)
  4) build_panel(px=..., save=False) → build_features(panel_df=..., save=False)
  5) 챔피언 학습(어제까지) → 당일 잠정 행 스코어 (run_daily 와 동일 로직, ≈ 70초)
  6) 세트별 시총 가중평균 p → output/gauge/gauge_YYYY-MM-DD.json + 메일/푸시

⚠ 잠정치 한계 (설계 전제 — Kane 인지):
  - 15:10 가격은 동시호가(15:20~15:30) 미반영. 거래량도 당일 누적 중간값.
  - 모델은 확정 종가로 학습됐으므로 이 확률은 잠정 게이지다.
  - **정본 신호는 16:20 run_daily** — 데이 포트 매수 판단은 그쪽이 소유.
  - p 는 절대확률이 아니라 유니버스 내 상대 백분위 (0.5 = 시장 중앙).

사용:
  python3 run_gauge.py                     # 운영 (launchd 15:10)
  python3 run_gauge.py --no-notify         # 계산만 (테스트)
  python3 run_gauge.py --date 2026-07-30   # 날짜 지정 (장후 검증용)
  python3 run_gauge.py --lgbm-only         # 속도 (CatBoost 생략)
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

import config as cfg
import gauge_config as gcfg
from build_panel import build_panel
from build_features import build_features
from model import fit_predict

KST = timezone(timedelta(hours=9))

# LLV 진입점 (data_service 단일 진입점 원칙)
sys.path.insert(0, cfg.LLV_PATH)
from stolab_data.data_service import (      # noqa: E402
    fetch_today_ohlcv_snapshot, compute_indicators_frame,
)
from stolab_data import is_trading_day      # noqa: E402

GAUGE_DIR = os.path.join(cfg.BASE_DIR, "output", "gauge")

# 잠정 스냅샷 입력 컬럼 (지표는 전량 재계산하므로 제외하고 넘긴다)
_PX_BASE_COLS = ["Date", "Ticker", "Name", "Open", "High", "Low", "Close",
                 "Volume", "Amount", "MarketCap", "ListShrs",
                 "Inst_Net", "Foreign_Net", "Retail_Net"]


def _num(x):
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def _load_universe():
    """core+extend 이력 + 유니버스 티커 목록 (ETF/우선주 제외) + 이름 정본."""
    j = json.load(open(cfg.CLASSJ, encoding="utf-8"))
    sec = {v["ticker"]: v["sector"] for v in j["classifications"].values()}
    names = {v["ticker"]: v["name"] for v in j["classifications"].values()}
    core = pd.read_parquet(cfg.CORE)
    ext = pd.read_parquet(cfg.EXTEND)
    px = pd.concat([core, ext], ignore_index=True)
    px["Ticker"] = px.Ticker.astype(str).str.zfill(6)
    tickers = sorted(t for t in px.Ticker.unique()
                     if sec.get(t) and sec[t] not in cfg.DROP_SUBSECTORS)
    return px, tickers, names


def gauge(today=None, use_ensemble=True, notify=True, save=True,
          sleep_sec=None):
    t0 = time.time()
    now = datetime.now(KST)
    today = pd.Timestamp(today) if today else pd.Timestamp(now.date())

    if not is_trading_day(today.date()):
        print(f"[gauge] {today.date()} 휴장일 — 종료")
        return None

    # 1) 이력 로드 + 당일 행 제거 (재실행/장후 테스트 멱등성)
    px, universe, names = _load_universe()
    px = px[px.Date != np.datetime64(today)].copy()
    print(f"[gauge] 이력 {len(px):,}행, 유니버스 {len(universe)}종목")

    # 2) 잠정 스냅샷
    price_time = datetime.now(KST).strftime("%H:%M")
    snap = fetch_today_ohlcv_snapshot(
        universe, sleep_sec=sleep_sec or gcfg.SNAPSHOT_SLEEP_SEC)
    snap = snap[snap.Date == np.datetime64(today)].copy()   # 휴장/지연 행 방어
    if len(snap) < len(universe) * 0.5:
        raise SystemExit(f"[gauge] 스냅샷 부족 ({len(snap)}/{len(universe)}) — 중단")
    snap["Ticker"] = snap.Ticker.astype(str).str.zfill(6)
    snap["Amount"] = snap.Close * snap.Volume
    missing = sorted(set(universe) - set(snap.Ticker))
    if missing:
        print(f"[gauge] 스냅샷 누락 {len(missing)}종목: {missing[:10]}")
    print(f"[gauge] 스냅샷 {len(snap)}종목 ({price_time} 기준, "
          f"{time.time()-t0:.0f}초 경과)")

    # 3) 병합 + 지표 재계산 (LLV 정본 로직 위임)
    for c in _PX_BASE_COLS:
        if c not in px.columns:
            px[c] = np.nan
        if c not in snap.columns:
            snap[c] = np.nan
    allpx = pd.concat([px[_PX_BASE_COLS], snap[_PX_BASE_COLS]],
                      ignore_index=True)
    # pandas 3.x: groupby.apply 가 그룹 컬럼(Ticker)을 제외하므로 백업 후 복원
    allpx["_tk"] = allpx["Ticker"]
    allpx = compute_indicators_frame(allpx)
    if "Ticker" not in allpx.columns:
        allpx = allpx.rename(columns={"_tk": "Ticker"})
    else:
        allpx = allpx.drop(columns=["_tk"])
    # Wyckoff 위임 검증 — hillstorm 실패 시 조용히 None 폴백되므로 여기서 확인
    wy_ratio = allpx.Wyckoff_Label.notna().mean()
    if wy_ratio < 0.5:
        print(f"[gauge] ⚠ Wyckoff 라벨 채움 {wy_ratio:.0%} — hillstorm 위임 실패 의심")
    print(f"[gauge] 지표 재계산 완료 (Wyckoff {wy_ratio:.0%}, "
          f"{time.time()-t0:.0f}초 경과)")

    # 4) 패널 → 피처 (인메모리)
    panel = build_panel(px=allpx, save=False)
    feat = build_features(panel_df=panel, save=False)
    cols = json.load(open(cfg.COLS_JSON))["CHAMPION"]
    lack = [c for c in cols if c not in feat.columns]
    if lack:
        raise SystemExit(f"[gauge] 피처 누락 {lack[:5]} — cols.json 불일치")

    # 5) 학습(어제까지) + 당일 잠정 스코어
    days = np.sort(feat.Date.unique())
    prior = days[days < np.datetime64(today)]
    if len(prior) == 0:
        raise SystemExit(f"[gauge] {today.date()} 이전 거래일 없음")
    train = feat[feat.Date <= prior[-1]]
    score = feat[feat.Date == np.datetime64(today)].copy()
    if score.empty:
        raise SystemExit(f"[gauge] {today.date()} 스코어 행 없음")
    scores, meta = fit_predict(train, score, cols, cfg.TARGET,
                               use_ensemble, return_meta=True)
    score["p"] = scores.values
    score = score.sort_values("p", ascending=False).reset_index(drop=True)
    score["rank"] = np.arange(1, len(score) + 1)
    meta["elapsed_sec"] = round(time.time() - t0, 1)

    # 시총 (패널 재구성값: ListShrs × 잠정 종가)
    mcap = panel[panel.Date == np.datetime64(today)].set_index("Ticker").MarketCap

    # 6) 세트별 시총 가중평균 (순수 로직 — gauge_core, 테스트 대상)
    from gauge_core import aggregate_sets
    sets_out = aggregate_sets(score.set_index("Ticker"), mcap,
                              gcfg.SECTOR_SETS, names)

    doc = {
        "schema_version": "gauge-1.0",
        "strategy_id": "mop_sector_gauge",
        "model_version": cfg.MODEL_VERSION,
        "as_of": str(today.date()),
        "generated_at": datetime.now(KST).isoformat(timespec="seconds"),
        "provisional": True,
        "price_time": price_time,
        "universe_count": int(len(score)),
        "snapshot_missing": missing,
        "train_meta": meta,
        "sets": sets_out,
    }

    if save:
        os.makedirs(GAUGE_DIR, exist_ok=True)
        out = os.path.join(GAUGE_DIR, f"gauge_{today.date()}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        with open(os.path.join(GAUGE_DIR, "gauge_latest.json"), "w",
                  encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        print(f"[gauge] → {out}")

    print(f"\n=== {today.date()} 섹터 오버나이트 게이지 ({price_time} 잠정) ===")
    for s in sets_out:
        wp = f"{s['weighted_p']*100:.1f}%" if s["weighted_p"] is not None else "—"
        print(f"{s['name']:<12} 가중 {wp:>7}  ({s['n_scored']}/{s['n_members']}종목)")
    print(f"\n소요 {meta['elapsed_sec']}초 / 학습 {meta.get('train_rows'):,}행 "
          f"/ 앙상블 {meta.get('ensemble')}")

    if notify:
        try:
            from gauge_notify import (build_gauge_email, build_gauge_push,
                                      send_email, send_push)
            subj, html = build_gauge_email(doc)
            send_email(subj, html)
            title, msg = build_gauge_push(doc)
            send_push(title, msg)
        except Exception as e:   # fail-safe — 통지 실패해도 JSON 은 남는다
            print(f"[gauge] 통지 실패 (JSON 저장은 완료): {e}")

    return doc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--lgbm-only", action="store_true")
    ap.add_argument("--no-notify", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--sleep", type=float, default=None,
                    help="스냅샷 종목 간 대기(초) — 기본 gauge_config.SNAPSHOT_SLEEP_SEC")
    a = ap.parse_args()
    gauge(today=a.date, use_ensemble=not a.lgbm_only,
          notify=not a.no_notify, save=not a.no_save, sleep_sec=a.sleep)
