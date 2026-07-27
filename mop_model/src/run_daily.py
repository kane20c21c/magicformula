# -*- coding: utf-8 -*-
"""
run_daily.py — 운영: 오늘의 매수 신호 생성 → 신호 JSON 산출

실행 시각: 거래일 16:20 (LLV 16:00 종가 배치 이후). launchd
           com.kane.magicformula-mop-signal 이 호출.

파이프라인:
  1) --rebuild 이면 build_panel → build_features 재실행 (LLV parquet 직결)
  2) '오늘까지 라벨 확정된 전 데이터'로 재학습 (워크포워드)
  3) 전 종목 점수 산출 → output/signals/signal_YYYY-MM-DD.json

라벨 타이밍(누출 방지): 오늘(t) 종가 시점에 라벨 확정된 마지막 feature-date =
직전 거래일. 따라서 train = Date <= 직전거래일, score = Date == 오늘.

★ 신호 JSON 에는 Top5 가 아니라 **전 종목 순위**를 싣는다.
  소비자(StockPortfolio app/paper_day)가 NXT 미거래 종목을 건너뛰고
  차순위로 충원해야 하기 때문 (Kane 확정 2026-07-27).

사용:
  python3 run_daily.py --rebuild          # 운영 (데이터 재생성 포함)
  python3 run_daily.py                    # 기존 features 로 신호만
  python3 run_daily.py --date 2026-07-24
  python3 run_daily.py --lgbm-only        # 속도(단독)
"""
import argparse, json, os, time
from datetime import datetime, timezone, timedelta

import numpy as np, pandas as pd
import config as cfg
from model import fit_predict

KST = timezone(timedelta(hours=9))


def _num(x):
    """JSON 직렬화 가능한 값으로 (NaN → None)."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def signal(today=None, use_ensemble=True, target=None, top_k=None,
           rebuild=False, save=True):
    target = target or cfg.TARGET
    top_k = top_k or cfg.TOP_K
    t0 = time.time()

    if rebuild:
        from build_panel import build_panel
        from build_features import build_features
        build_panel()
        build_features()

    d = pd.read_parquet(cfg.FEATURES).sort_values(["Date", "Ticker"]).reset_index(drop=True)
    cols = json.load(open(cfg.COLS_JSON))["CHAMPION"]
    days = np.sort(d.Date.unique())
    today = pd.Timestamp(today) if today else pd.Timestamp(days[-1])
    prior = days[days < np.datetime64(today)]
    if len(prior) == 0:
        raise SystemExit(f"{today.date()} 이전 거래일이 없음 — 학습 불가")
    prev = prior[-1]

    train = d[d.Date <= prev]
    score_df = d[d.Date == np.datetime64(today)].copy()
    if score_df.empty:
        raise SystemExit(f"{today.date()} 데이터 없음 — LLV 종가 배치 완료 여부 확인")

    scores, meta = fit_predict(train, score_df, cols, target, use_ensemble, return_meta=True)
    score_df["p"] = scores.values
    score_df = score_df.sort_values("p", ascending=False).reset_index(drop=True)
    score_df["rank"] = np.arange(1, len(score_df) + 1)
    meta["elapsed_sec"] = round(time.time() - t0, 1)

    ranking = [{
        "rank": int(r["rank"]),
        "ticker": str(r["Ticker"]),
        "name": r["Name"],
        "sector_top": r["sector_top"],
        "close": _num(r["Close"]),
        "p": round(float(r["p"]), 6),
        "is_halt": bool(r["is_halt"]),
    } for _, r in score_df.iterrows()]

    doc = {
        "schema_version": cfg.SCHEMA_VERSION,
        "strategy_id": cfg.STRATEGY_ID,
        "model_version": cfg.MODEL_VERSION,
        "as_of": str(today.date()),
        "generated_at": datetime.now(KST).isoformat(timespec="seconds"),
        "top_k": int(top_k),
        "universe_count": int(len(ranking)),
        "train_meta": meta,
        "ranking": ranking,
    }

    if save:
        os.makedirs(cfg.SIGNAL_DIR, exist_ok=True)
        out = os.path.join(cfg.SIGNAL_DIR, f"signal_{today.date()}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        latest = os.path.join(cfg.SIGNAL_DIR, "signal_latest.json")
        with open(latest, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        print(f"[signal] → {out}")

    print(f"\n=== {today.date()} 매수 신호 (Top{top_k}) — 데이 포트 ===")
    print(f"{'순위':>3} {'종목':<14} {'섹터':<10} {'종가':>10} {'점수':>7}")
    for r in ranking[:top_k]:
        cl = f"{r['close']:,.0f}" if r["close"] is not None else "-"
        print(f"{r['rank']:>3} {str(r['name'])[:14]:<14} {str(r['sector_top'])[:10]:<10} {cl:>10} {r['p']:>7.3f}")
    print(f"\n학습: {meta.get('train_first_date')}~{meta.get('train_last_date')} "
          f"{meta.get('train_rows'):,}행 / 피처 {meta.get('n_features')} / "
          f"앙상블 {meta.get('ensemble')} / {meta.get('elapsed_sec')}초")
    return doc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--lgbm-only", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="panel/features 재생성 후 신호")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    signal(today=a.date, use_ensemble=not a.lgbm_only,
           rebuild=a.rebuild, save=not a.no_save)
