# -*- coding: utf-8 -*-
"""
target_walkforward.py — 타깃 정의 비교 워크포워드 (검증 전용, 운영 아님)

배경 (2026-08-30 라이브 24일 분석):
  · p ↔ YZ_20 일별 스피어만 **+0.518**, 24일 전부 양수 (p<0.0001)
  · 변동성 분위를 통제하면 p 의 갭 예측력이 사라짐 (분위별 IC −0.014 ~ +0.106, 부호 무작위)
  · rank 1~30 내부는 순열검정 p=0.382 — 순서 정보 없음
  · 원인 가설: 타깃 `y_rel = (Gap_T1 > 그날 중앙값)` 의 **`>` 비교 + 갭 동률**.
    갭이 정확히 0.00% 인 비율이 저변동군 13.5% vs 고변동군 8.2% 이고,
    그날 갭 중앙값이 정확히 0 인 날이 797일 중 119일(14.9%) →
    저변동 종목이 구조적으로 y_rel=0 을 받아 모델이 '변동성 예측기'로 수렴.
  · 타깃 정의만 바꿔 계산한 구조적 편향(변동성 5분위−1분위 발생률 차):
      y_rel 6.52%p / `>=` 로만 변경 4.93%p / y_abs 4.03%p /
      변동성조정갭 1.79%p / 변동성분위내 상대순위 0.04%p

목적: 타깃 3종을 **같은 피처·같은 구간·같은 하이퍼파라미터**로 재학습해
      ① 랭크-갭 단조성이 실제로 생기는지 ② 상위 선별 수익이 어떻게 되는지 비교.

⚠ 운영 파이프라인 무수정 — `src/` 는 읽기만 한다. 결과는 build/tw_*.parquet.
⚠ 하이퍼파라미터는 손대지 않는다 (config.py 의 튜닝 금지 이력: 36개 설정 탐색 →
  test-holdout 상관 −0.34). 바꾸는 것은 **타깃 정의 하나뿐**.
⚠ 재학습 주기 = 5거래일 1회 (운영은 매일). 하이퍼파라미터 고정이라 트리만 미세 차이이고
  타깃 간 상대 비교가 목적이므로 공정. 매일 재학습은 3타깃 9시간+ 라 채택 안 함.
⚠ 룩어헤드 방지: 학습은 `Date <= 재학습기준일의 직전 거래일` 까지만.
  y_relz 의 YZ_20 은 D 시점 값(과거)이고 Gap_T1 은 D+1 시가 — 라벨이므로 정상.

사용:
  python3 target_walkforward.py --target y_rel   --start 2025-08-25 --end 2026-08-21
  python3 target_walkforward.py --target y_relz  --start 2025-08-25 --end 2026-08-21
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import config as cfg                      # noqa: E402
from model import fit_predict             # noqa: E402
import json                               # noqa: E402

RETRAIN_EVERY = 5      # 거래일 (주 1회 근사)


def load_panel():
    """features + LLV YZ_20 결합 후 타깃 3종 생성."""
    d = pd.read_parquet(cfg.FEATURES)
    d = d.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    cols = ["Date", "Ticker", "YZ_20"]
    v = pd.concat([pd.read_parquet(cfg.CORE, columns=cols),
                   pd.read_parquet(cfg.EXTEND, columns=cols)], ignore_index=True)
    v = v.drop_duplicates(subset=["Ticker", "Date"])
    d = d.merge(v, on=["Ticker", "Date"], how="left")

    # ── 타깃 3종 ────────────────────────────────────────────────
    # ① y_rel  : 현행 (features 에 이미 있음)
    # ② y_abs  : 갭 > 0 (features 에 이미 있음) — 과거 기각안, 하네스 재현 확인용 대조군
    # ③ y_relz : 변동성조정갭 (Gap_T1 / YZ_20) > 그날 중앙값
    gz = d.Gap_T1 / d.YZ_20
    d["gz"] = gz.replace([np.inf, -np.inf], np.nan)
    med = d.groupby("Date").gz.transform("median")
    d["y_relz"] = np.where(d.gz.isna() | med.isna(), np.nan, (d.gz > med).astype(float))

    print(f"패널 {len(d)}행 · {d.Date.nunique()}거래일 · {d.Ticker.nunique()}종목 | "
          f"YZ_20 결손 {int(d.YZ_20.isna().sum())} · y_relz 결손 {int(d.y_relz.isna().sum())}",
          flush=True)
    return d


def run(target, start, end, out_path, retrain_every=RETRAIN_EVERY):
    d = load_panel()
    cols = json.load(open(cfg.COLS_JSON))["CHAMPION"]

    alldays = np.sort(d.Date.unique())
    nxt = {alldays[i]: alldays[i + 1] for i in range(len(alldays) - 1)}
    prv = {alldays[i]: alldays[i - 1] for i in range(1, len(alldays))}
    days = [x for x in alldays
            if pd.Timestamp(start) <= pd.Timestamp(x) <= pd.Timestamp(end) and x in nxt]
    print(f"[{target}] 스코어일 {len(days)}일 ({pd.Timestamp(days[0]).date()} ~ "
          f"{pd.Timestamp(days[-1]).date()}) · 재학습 {int(np.ceil(len(days)/retrain_every))}회",
          flush=True)

    out, model_s, t0 = [], None, time.time()
    for i, t in enumerate(days):
        if i % retrain_every == 0:                       # ── 재학습
            train = d[d.Date <= prv[t]].dropna(subset=[target])
            score_block = d[d.Date.isin(days[i:i + retrain_every])]
            s, meta = fit_predict(train, score_block, cols, target=target,
                                  use_ensemble=True, return_meta=True)
            model_s = pd.Series(s.values, index=score_block.index)
            el = time.time() - t0
            print(f"  [{i+1:3d}/{len(days)}] 재학습 {pd.Timestamp(t).date()} "
                  f"train={meta['train_rows']} lgbm_auc={meta.get('lgbm_valid_auc')} "
                  f"cat_auc={meta.get('cat_valid_auc')} 경과 {el/60:.1f}분", flush=True)

        cur = d[d.Date == t]
        p_raw = model_s.reindex(cur.index)
        r = cur[["Date", "Ticker", "Gap_T1", "y_rel", "y_abs", "y_relz", "YZ_20", "Close"]].copy()
        r["p"] = p_raw.rank(pct=True).values          # 그날 유니버스 백분위 (운영 신호와 동일 형태)
        r["rank"] = (-r.p).rank(method="first").astype(int)
        # D+1 종가 (익일 종가 수익 계산용)
        nx = d[d.Date == nxt[t]].set_index("Ticker")
        r["close_T1"] = r.Ticker.map(nx.Close)
        r["open_T1"] = r.Ticker.map(nx.Open) if "Open" in nx.columns else np.nan
        r["target_used"] = target
        out.append(r)

        if (i + 1) % 20 == 0:                          # 중간 저장 (중단 대비)
            pd.concat(out).to_parquet(out_path, index=False)

    R = pd.concat(out)
    R.to_parquet(out_path, index=False)
    print(f"[{target}] 완료 {len(R)}행 → {out_path} · 총 {(time.time()-t0)/60:.1f}분", flush=True)
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["y_rel", "y_abs", "y_relz"])
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--retrain-every", type=int, default=RETRAIN_EVERY)
    a = ap.parse_args()
    out = os.path.join(cfg.OUT_DIR, f"tw_{a.target}.parquet")
    run(a.target, a.start, a.end, out, a.retrain_every)
