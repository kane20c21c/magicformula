# -*- coding: utf-8 -*-
"""
model.py — 챔피언 모델 (LGBM + CatBoost 앙상블) 학습/예측

- 하이퍼파라미터는 config 에 고정 (튜닝 금지: 탐색 결과 test-holdout 상관 −0.34로 역효과)
- 앙상블 = 두 모델 예측확률을 '날짜별 백분위(rank)'로 바꿔 평균
  (rank 평균이 스케일 차이에 강건. 앙상블이 단일 대비 알파↑·셔플편향↓)
- use_ensemble=False 면 LGBM 단독 (속도용; 워크포워드 개념검증엔 충분)

워크포워드: 매일 '오늘까지 라벨 확정된 전 데이터'로 **처음부터 재학습**한다.
어제 모델을 이어 학습하는 warm-start 가 아니므로 **누적 저장할 모델 상태가 없다**.
남기는 것은 그날의 점수와 학습 메타(재현·감사용)뿐 — run_daily.py 가 JSON 으로 기록.
"""
import numpy as np, pandas as pd
import lightgbm as lgb
import config as cfg

try:
    from catboost import CatBoostClassifier
    _HAS_CAT = True
except Exception:
    _HAS_CAT = False


def _valid_split(train_df, frac=0.85):
    """조기종료용 valid = 학습구간의 최근 (1-frac) 날짜"""
    days = np.sort(train_df.Date.unique())
    cut = days[int(len(days) * frac)]
    return train_df[train_df.Date < cut], train_df[train_df.Date >= cut]


def fit_predict(train_df, score_df, cols, target="y_rel", use_ensemble=True,
                return_meta=False):
    """
    train_df : 학습 데이터(라벨 확정 구간만; 호출자가 보장)
    score_df : 예측 대상
    반환      : score_df 인덱스에 정렬된 '앙상블 점수'(높을수록 매수후보 상위)
                return_meta=True 이면 (점수, meta dict)
    """
    tr = train_df.dropna(subset=[target])
    TRf, VAf = _valid_split(tr)
    meta = {
        "train_rows": int(len(tr)),
        "train_first_date": str(pd.Timestamp(tr.Date.min()).date()) if len(tr) else None,
        "train_last_date": str(pd.Timestamp(tr.Date.max()).date()) if len(tr) else None,
        "n_features": int(len(cols)),
        "target": target,
        "ensemble": bool(use_ensemble and _HAS_CAT),
    }

    # LGBM
    m = lgb.LGBMClassifier(**cfg.LGBM_PARAMS)
    m.fit(TRf[cols], TRf[target], eval_set=[(VAf[cols], VAf[target])], eval_metric="auc",
          callbacks=[lgb.early_stopping(cfg.LGBM_EARLY_STOP, verbose=False)])
    p_lgb = m.predict_proba(score_df[cols])[:, 1]
    try:
        meta["lgbm_best_iter"] = int(m.best_iteration_)
        meta["lgbm_valid_auc"] = round(float(m.best_score_["valid_0"]["auc"]), 5)
    except Exception:
        pass

    if not (use_ensemble and _HAS_CAT):
        s = pd.Series(p_lgb, index=score_df.index)
        return (s, meta) if return_meta else s

    # CatBoost
    cb = CatBoostClassifier(early_stopping_rounds=cfg.CAT_EARLY_STOP, **cfg.CAT_PARAMS)
    cb.fit(TRf[cols].fillna(cfg.CAT_NAN), TRf[target],
           eval_set=(VAf[cols].fillna(cfg.CAT_NAN), VAf[target]), verbose=False)
    p_cat = cb.predict_proba(score_df[cols].fillna(cfg.CAT_NAN))[:, 1]
    try:
        meta["cat_best_iter"] = int(cb.get_best_iteration())
        meta["cat_valid_auc"] = round(float(cb.get_best_score()["validation"]["AUC"]), 5)
    except Exception:
        pass

    # 순위평균 앙상블
    r1 = pd.Series(p_lgb, index=score_df.index).rank(pct=True)
    r2 = pd.Series(p_cat, index=score_df.index).rank(pct=True)
    s = (r1 + r2) / 2.0
    return (s, meta) if return_meta else s
