# -*- coding: utf-8 -*-
"""
entry_hypotheses.py — 진입 신호 개선 가설 3종 워크포워드 (검증 전용, 운영 아님)

배경 (2026-09-12 데이포트 손실 진단, SP reports/데이포트_손실진단_20260912.html):
  · 8/18 이후 18일 라이브 IC −0.026 · 음수일 56% — 워크포워드 242일의 18일 롤링 창 225개
    중 그보다 나쁜 창 0개. 청산 룰(그림자 R1~R4)을 바꿔도 전부 손실 → 진입 신호 문제.
  · 148피처는 전부 종목 자기값 + 횡단면 백분위 → **시장 상태를 전혀 보지 못한다**
    (VKOSPI·지수·폭·OptGauge/SpotGauge 미포함, 케인 확인 2026-09-12).

⚠ 튜닝 금지 원칙(config.py) 준수 — 하이퍼파라미터는 손대지 않는다. 아래 3가설은
  "탐색"이 아니라 **사전 등록된 단일 구조 변경**이며 각각 한 번씩만 돌린다.
  base 도 같은 설정으로 다시 돌려 4개를 동일 조건에서 비교한다.

가설 (각각 base 대비 하나만 바뀜):
  base   : 현행 그대로 (y_rel, 148피처, 전 구간 학습)
  mkt    : + 시장 상태 피처 13개 (아래 MKT_COLS). 유니버스 집계 + TIGER200(102110).
           모두 날짜 t 마감 시점에 알 수 있는 값 → 룩어헤드 없음. 전 종목에 같은 값이
           들어가므로 상대타깃 y_rel 을 직접 맞추진 못하고, 종목 피처와의 **상호작용**
           (국면별로 선별 규칙이 달라지는가)만 학습할 수 있다. 그게 가설이다.
  recent : 학습 창을 **최근 250거래일**로 제한 (현행은 2023-05 이후 전부).
  secrel : 타깃을 **같은 대섹터(sector_top) 중앙값 대비**로 (종목수<5 섹터는 전체 중앙값).
           섹터 전체가 꺾이면 현행 타깃은 그 섹터 종목에 전부 0을 준다 → 섹터 방향을
           못 맞히면 종목 선별력까지 같이 죽는다는 가설.

사전 등록 판정 기준 (entry_eval.py 가 계산; 결과 보고 바꾸지 않는다):
  채택 후보 = 전 구간(2025-08-25~) 일별 IC 평균이 base − 0.010 이상
              AND rank≤10 초과갭이 base 이상
              AND 2026-08-18 이후 구간 IC 가 base 보다 높음
  셋 다 만족해야 "채택 후보". 하나라도 미달이면 기각. 후보가 둘 이상이면 케인 결정.

사용 (맥 로컬에서 — VM 아님):
  cd MagicFormula/mop_model/experiments
  python3 entry_hypotheses.py --hyp base   --start 2025-08-25 --end 2026-09-10
  python3 entry_hypotheses.py --hyp mkt    --start 2025-08-25 --end 2026-09-10
  python3 entry_hypotheses.py --hyp recent --start 2025-08-25 --end 2026-09-10
  python3 entry_hypotheses.py --hyp secrel --start 2025-08-25 --end 2026-09-10
  (스모크: --start 2026-08-01 --end 2026-08-10 --no-ensemble  ≈ 2~3분)
  결과: build/eh_<hyp>.parquet · 로그는 stdout → tee build/eh_<hyp>.log
  ⚠ features.parquet 이 --end 다음 거래일까지 있어야 한다 (Gap_T1 라벨).
     에어는 8/24 까지뿐 → 먼저 `cd src && python3 -c "from build_panel import build_panel;
     from build_features import build_features; build_panel(); build_features()"` (약 90초,
     run_daily 와 달리 신호 JSON 은 건드리지 않음).
"""
import argparse, os, sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import config as cfg                      # noqa: E402
from model import fit_predict             # noqa: E402

RETRAIN_EVERY = 5
RECENT_DAYS = 250

MKT_COLS = ["mkt_ret1", "mkt_ret5", "mkt_ret20", "mkt_breadth1", "mkt_gapdn3", "mkt_gap0_mean",
            "mkt_disp1", "mkt_vol20", "mkt_updays5", "bm_ret1", "bm_ret5", "bm_gap0", "bm_vol20"]


def add_market_state(d):
    """날짜 t 의 시장 상태 (전 종목 공통값). 모두 t 마감 후 확정되는 값만 사용."""
    d = d.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    g = d.groupby("Ticker")
    prev_close = g.Close.shift(1)
    d["_gap0"] = d.Open / prev_close - 1              # 오늘 시가 갭 (t−1 종가 → t 시가): 확정값
    d["_ret1"] = d.Close / prev_close - 1
    d["_ret5"] = g.Close.transform(lambda s: s.pct_change(5))
    d["_ret20"] = g.Close.transform(lambda s: s.pct_change(20))
    by = d.groupby("Date")
    M = pd.DataFrame({
        "mkt_ret1": by._ret1.mean(),
        "mkt_ret5": by._ret5.mean(),
        "mkt_ret20": by._ret20.mean(),
        "mkt_breadth1": by._ret1.apply(lambda s: (s > 0).mean()),
        "mkt_gapdn3": by._gap0.apply(lambda s: (s <= -0.03).mean()),   # N3 비율 (쇼크일 판정과 같은 척도)
        "mkt_gap0_mean": by._gap0.mean(),
        "mkt_disp1": by._ret1.std(),
        "mkt_vol20": by.vol20.mean() if "vol20" in d.columns else by._ret1.std(),
    })
    M["mkt_updays5"] = (M.mkt_ret1 > 0).astype(float).rolling(5, min_periods=1).sum()

    # 벤치마크 TIGER200 (LLV core 원본; 지수 parquet 은 에어에서 8/24 까지뿐이라 ETF 로 대체)
    bm = pd.read_parquet(cfg.CORE, columns=["Date", "Ticker", "Open", "Close"])
    bm = bm[bm.Ticker == "102110"].sort_values("Date").set_index("Date")
    bm.index = pd.to_datetime(bm.index)
    bmp = bm.Close.shift(1)
    B = pd.DataFrame({"bm_ret1": bm.Close / bmp - 1, "bm_ret5": bm.Close.pct_change(5),
                      "bm_gap0": bm.Open / bmp - 1})
    B["bm_vol20"] = B.bm_ret1.rolling(20, min_periods=10).std()
    M = M.join(B, how="left")

    d = d.drop(columns=["_gap0", "_ret1", "_ret5", "_ret20"])
    d = d.merge(M.reset_index().rename(columns={"index": "Date"}), on="Date", how="left")
    return d


def add_sector_target(d):
    """y_sec = Gap_T1 > 같은 (Date, sector_top) 중앙값. 섹터 종목수 < MIN_SECTOR_N 이면 전체 중앙값."""
    n = d.groupby(["Date", "sector_top"]).Gap_T1.transform("size")
    med_s = d.groupby(["Date", "sector_top"]).Gap_T1.transform("median")
    med_a = d.groupby("Date").Gap_T1.transform("median")
    med = med_s.where(n >= cfg.MIN_SECTOR_N, med_a)
    d["y_sec"] = np.where(d.Gap_T1.isna(), np.nan, (d.Gap_T1 > med).astype(float))
    return d


def run(hyp, start, end, out_path, retrain_every=RETRAIN_EVERY, use_ensemble=True):
    d = pd.read_parquet(cfg.FEATURES).sort_values(["Date", "Ticker"]).reset_index(drop=True)
    d["Date"] = pd.to_datetime(d.Date)
    cols = json.load(open(cfg.COLS_JSON))["CHAMPION"]
    target = "y_rel"
    if hyp == "mkt":
        d = add_market_state(d); cols = cols + MKT_COLS
        print(f"[mkt] 시장 피처 {len(MKT_COLS)}개 추가 · 결손율 "
              f"{d[MKT_COLS].isna().mean().round(3).to_dict()}", flush=True)
    elif hyp == "secrel":
        d = add_sector_target(d); target = "y_sec"
        print(f"[secrel] y_sec 양성률 {d.y_sec.mean():.3f} (y_rel {d.y_rel.mean():.3f})", flush=True)
    d = d.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    alldays = np.sort(d.Date.unique())
    nxt = {alldays[i]: alldays[i + 1] for i in range(len(alldays) - 1)}
    prv = {alldays[i]: alldays[i - 1] for i in range(1, len(alldays))}
    days = [x for x in alldays if pd.Timestamp(start) <= pd.Timestamp(x) <= pd.Timestamp(end) and x in nxt]
    print(f"[{hyp}] 스코어일 {len(days)}일 ({pd.Timestamp(days[0]).date()}~{pd.Timestamp(days[-1]).date()}) · "
          f"재학습 {int(np.ceil(len(days)/retrain_every))}회 · 피처 {len(cols)} · 타깃 {target} · "
          f"앙상블 {use_ensemble}", flush=True)

    out, model_s, t0 = [], None, time.time()
    for i, t in enumerate(days):
        if i % retrain_every == 0:
            train = d[d.Date <= prv[t]].dropna(subset=[target])
            if hyp == "recent":
                tdays = np.sort(train.Date.unique())[-RECENT_DAYS:]
                train = train[train.Date >= tdays[0]]
            score_block = d[d.Date.isin(days[i:i + retrain_every])]
            s, meta = fit_predict(train, score_block, cols, target=target,
                                  use_ensemble=use_ensemble, return_meta=True)
            model_s = pd.Series(s.values, index=score_block.index)
            print(f"  [{i+1:3d}/{len(days)}] 재학습 {pd.Timestamp(t).date()} train={meta['train_rows']} "
                  f"({meta['train_first_date']}~) lgbm_auc={meta.get('lgbm_valid_auc')} "
                  f"cat_auc={meta.get('cat_valid_auc')} 경과 {(time.time()-t0)/60:.1f}분", flush=True)
        cur = d[d.Date == t]
        r = cur[["Date", "Ticker", "sector_top", "Gap_T1", "y_rel", "Close"]].copy()
        r["p"] = model_s.reindex(cur.index).rank(pct=True).values
        r["rank"] = (-r.p).rank(method="first").astype(int)
        nx = d[d.Date == nxt[t]].set_index("Ticker")
        r["close_T1"] = r.Ticker.map(nx.Close); r["open_T1"] = r.Ticker.map(nx.Open)
        r["hyp"] = hyp
        out.append(r)
        if (i + 1) % 20 == 0:
            pd.concat(out).to_parquet(out_path, index=False)
    R = pd.concat(out); R.to_parquet(out_path, index=False)
    print(f"[{hyp}] 완료 {len(R)}행 → {out_path} · 총 {(time.time()-t0)/60:.1f}분", flush=True)
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", required=True, choices=["base", "mkt", "recent", "secrel"])
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--retrain-every", type=int, default=RETRAIN_EVERY)
    ap.add_argument("--no-ensemble", action="store_true", help="LGBM 단독 (스모크/속도용)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(cfg.OUT_DIR, f"eh_{a.hyp}.parquet")
    run(a.hyp, a.start, a.end, out, a.retrain_every, not a.no_ensemble)
