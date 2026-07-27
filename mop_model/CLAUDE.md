# mop_model — 데이 포트 신호 생성 (MOp ML Top5)

## 역할

매일 장 마감 후 **"내일 상대적으로 강할 5종목"** 을 골라 신호 JSON 을 만든다.
**판단(모델)은 여기가 소유**하고, 계좌·체결은 StockPortfolio `app/paper_day/` 가 소유한다.
4셀 분류 파이프라인과 같은 원칙 — *LLV 는 데이터 공급자지 판단자가 아니다*.

```
LLV (무수정)            → core/extend/ticker_classification.json 공급
   ↓ 읽기만
MagicFormula/mop_model  → panel · features · 매일 재학습 · 전 종목 순위  ★여기
   ↓ output/signals/signal_YYYY-MM-DD.json
StockPortfolio /day     → 데이 포트 가상계좌 (17:00 NXT 매수 + −2% 손절)
```

## 모델

- **LightGBM + CatBoost 앙상블** — 두 예측확률을 날짜별 백분위로 바꿔 평균
- **148피처** = BASE 48 + XS 45(전체 백분위) + G1 45(대섹터) + G5 10(소섹터)
- **상대타깃** `y_rel = (Gap_T1 > 그날 중앙값)`. `Gap_T1 = 익일시가/당일종가 − 1`
- **워크포워드** — 매일 "라벨 확정된 전 데이터"로 **처음부터 재학습**.
  warm-start 가 아니므로 **누적 저장할 모델 상태가 없다.** 남기는 건 그날의 순위와 학습 메타뿐.

### ⚠ 튜닝 금지
36개 설정 탐색 결과 test-holdout 상관 **−0.34**. 신호가 약해(AUC~0.56) 튜닝은
노이즈 적합 → 홀드아웃 악화. 하이퍼파라미터·피처는 고정, 트리만 갱신.

### ⚠ look-ahead 주의
`ticker_classification.json` 의 `r2/slope/cell4/signal_rules` 는 2025-04 이후
수익률로 역산한 값 — **절대 피처로 쓰지 말 것**. `sector` 필드만 사용한다.

### ⚠ 왜 features 를 LLV 로 못 옮기나
XS/G1/G5 100개 피처가 `groupby("Date").rank(pct=True)` — 그날 196종목 안의
횡단면 백분위다. 종목이 하나 추가/제외되면 **과거 전 구간 값이 바뀐다** →
LLV parquet 의 종목단위 upsert 계약(`_upsert_and_recompute`, `SUPPLY_COLS` 보존)과
정면충돌. 게다가 lightgbm/catboost 의존성이 LLV 를 import 하는 전 프로젝트로 번진다.

## 실행

```bash
pip3 install -r requirements.txt

cd src
python3 run_daily.py --rebuild        # 운영 (panel→features→학습→신호). 실측 약 90초
python3 run_daily.py                  # 기존 features 로 신호만
python3 run_daily.py --date 2026-07-24
python3 walkforward.py --start 2026-07-01 --end 2026-07-24   # 검증용 백테스트
```

경로가 다르면 환경변수 `MOP_DIR` / `LLV_PATH` 만 바꾸면 된다.

## 산출물

`output/signals/signal_YYYY-MM-DD.json` (+ `signal_latest.json`)

```json
{"schema_version":"1.0","strategy_id":"mop_ml_top5","as_of":"2026-07-24",
 "top_k":5,"universe_count":195,
 "train_meta":{"train_rows":148645,"train_last_date":"2026-07-23","n_features":148,
               "lgbm_best_iter":118,"lgbm_valid_auc":0.5615,"elapsed_sec":72.2},
 "ranking":[{"rank":1,"ticker":"086280","name":"현대글로비스","sector_top":"해외",
             "close":207000.0,"p":0.994872,"is_halt":false}, ...]}
```

★ **Top5 가 아니라 전 종목 순위**를 싣는다 — 소비자가 NXT 미거래 종목을 건너뛰고
차순위로 충원해야 하기 때문 (Kane 확정 2026-07-27).

## launchd

| Label | 시각 | 호출 |
|---|---|---|
| `com.kane.magicformula-mop-signal` | 매일 16:20 | `src/run_daily.py --rebuild` |

정본 plist 는 `configs/launchd/` — 운영본은 `~/Library/LaunchAgents/` 로 symlink.
LLV 16:00 kis_update 종가 배치 이후 20분 여유. 등록은 Kane 수동.

## ⚠ 백테스트 수치를 실현 기대치로 읽지 말 것

검증된 엣지(홀드아웃 89일 보정알파 +0.505%, 시뮬 +4.9%)는 **t일 종가 매수 →
t+1 시가 매도** 라는 실행 불가능한 이상체결 기준이다. 종가+1% 웃돈이면 −2.7% 로
부호가 뒤집힌다(거래당 엣지 +0.65%). 데이 포트는 실행 가능성을 우선해
**17:00 NXT 애프터마켓 매수 + −2% 손절 보유**로 운영하므로 성과가 다르다.
가상계좌의 목적은 그 괴리를 실측하는 것.
