# -*- coding: utf-8 -*-
"""
config.py — MOp 모델 경로/상수/하이퍼파라미터 (단일 진실 출처)

원본(Kane 실험3) 대비 변경점:
  - Data/ 복사본을 두지 않고 **LLV parquet 을 직접 읽는다** (데이터 단일소유 원칙).
    core/extend/ticker_classification.json 모두 longlivevault 가 정본.
  - derivatives_all.parquet 은 모델 미사용이라 제거 (레짐분석 전용이었음).
  - 거래/계좌 파라미터는 여기 없다 — StockPortfolio app/paper_mop/config.py 소유.
    (여기는 '신호를 만드는 규칙'만. 판단·체결 분리 원칙)

⚠ 하이퍼파라미터 튜닝 금지 — 36개 설정 탐색 결과 test-holdout 상관 −0.34.
  신호가 약해(AUC~0.55) 튜닝은 노이즈 적합 → 홀드아웃 악화. 보수적 기본값이 최적.
"""
import os

# ── 경로 ─────────────────────────────────────────────────────────────
BASE_DIR = os.environ.get("MOP_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR  = os.path.join(BASE_DIR, "build")            # panel/features 산출물
SIGNAL_DIR = os.path.join(BASE_DIR, "output", "signals")   # 일별 신호 JSON

# LLV 정본 직결 (복사본 없음)
LLV_PATH = os.environ.get(
    "LLV_PATH",
    os.path.expanduser("~/DriveForALL/StoLab/longlivevault"),
)
CORE   = os.path.join(LLV_PATH, "data", "ohlcv", "core.parquet")
EXTEND = os.path.join(LLV_PATH, "data", "ohlcv", "extend.parquet")
CLASSJ = os.path.join(LLV_PATH, "data", "ticker_classification.json")

PANEL    = os.path.join(OUT_DIR, "panel.parquet")
FEATURES = os.path.join(OUT_DIR, "features.parquet")
COLS_JSON= os.path.join(OUT_DIR, "cols.json")

# ── 유니버스/기간 규칙 ──────────────────────────────────────────────
START_DATE = "2023-05-15"        # 이전 구간은 종목수 부족(66)으로 횡단면 왜곡 → 컷
DROP_SUBSECTORS = ["ETF", "우선주"]   # 개별종목 아님 → 제외 (200 → 196)
MIN_SECTOR_N = 5                 # 섹터내 순위는 종목 5개 이상일 때만(계단형 노이즈 방지)

# ── 타깃 ────────────────────────────────────────────────────────────
# 상대타깃: 익일 갭(Gap_T1)이 "그날 유니버스 중앙값"을 초과하는가.
# (실험 결론: 절대타깃보다 상대타깃이 우수 — 시장 베타가 아니라 선택력을 학습)
TARGET = "y_rel"

# ── LightGBM (튜닝 안 함: 보수적 고정값) ────────────────────────────
LGBM_PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=100,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0, n_estimators=3000,
    verbose=-1, random_state=42, n_jobs=4,
)
LGBM_EARLY_STOP = 80

# ── CatBoost (앙상블 파트너) ────────────────────────────────────────
CAT_PARAMS = dict(
    loss_function="Logloss", eval_metric="AUC", learning_rate=0.03, depth=6,
    l2_leaf_reg=3.0, subsample=0.8, iterations=3000,
    random_seed=42, thread_count=4, allow_writing_files=False, verbose=False,
)
CAT_EARLY_STOP = 80
CAT_NAN = -999   # CatBoost 입력 결측 대체값

# ── 신호 ────────────────────────────────────────────────────────────
TOP_K = 5                 # 상위 K종목이 1순위 매수 후보
STRATEGY_ID = "mop_ml_top5"
MODEL_VERSION = "experiment3-ensemble-v1"
SCHEMA_VERSION = "1.0"

# ⚠ 소비자(StockPortfolio paper_mop)는 ranking 전체를 받아 NXT 미거래 종목을
#   건너뛰고 차순위로 충원한다 → 신호 JSON 에 196종목 전 순위를 싣는다.
