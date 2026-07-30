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
StockPortfolio /day     → 데이 포트 가상계좌 (17:00 NXT 매수 + 당일고점 −1% 손절, 2026-07-30 −2%→−1%)
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

## 섹터 오버나이트 게이지 (15:10 잠정 — Kane 지시 2026-07-30)

장 마감 전 **15:10 현재가를 임시 종가로 주입**해 챔피언 모델을 인메모리로 돌리고,
Kane 지정 섹터 세트별 갭1 상대 상승확률의 **시가총액 가중평균**을 15:15경
메일+푸시(스윙 포트 방식)로 보고한다. **보고 전용 — 정본 신호는 16:20 run_daily.**

- **세트 정의**: `src/gauge_config.py` (Kane 편집 파일 — 8세트 44종목, 2026-07-30 확정)
- **러너**: `src/run_gauge.py` — LLV `data_service.fetch_today_ohlcv_snapshot`(잠정
  스냅샷, sleep 0.25초) + `compute_indicators_frame`(지표 37컬럼 정본 재계산) →
  `build_panel(px=…, save=False)` → `build_features(panel_df=…, save=False)` →
  챔피언 학습·스코어 → `gauge_core.aggregate_sets`(시총 가중평균, 순수 로직).
  **전부 인메모리 — panel/features/signals 운영 산출물 무오염.**
- **산출물**: `output/gauge/gauge_YYYY-MM-DD.json` (+ `gauge_latest.json`),
  통지는 `src/gauge_notify.py` (paper_day notify 계약 — GMAIL_*/PUSHOVER_* 재사용)
- **실측** (2026-07-30 드라이런): 총 109초 = 스냅샷 70 + 지표 10 + 피처/학습 29.
  지표 재계산은 parquet 정본과 완전 일치 (RSI/MA/MACD/Supply/Weis/Wyckoff diff 0).
  잠정 p vs 16:20 정본 p: 순위상관 0.90, |diff| 중앙값 0.05 / 최대 0.23
  (KIS raw vs UN 통합 거래량·Amount 근사·유니버스 1종목 차이 기인 — 게이지 용도 충분)
- **잠정치 한계 (전제)**: 동시호가(15:20~15:30) 미반영, 거래량 당일 누적 중간값,
  p 는 절대확률이 아니라 유니버스 내 상대 백분위 (0.5 = 시장 중앙)
- **pandas 3 주의**: venv pandas 3.x 는 `groupby.apply` 가 그룹 컬럼(Ticker)을
  제외한다 — run_gauge 가 `_tk` 백업/복원으로 방어 (LLV 배치는 시스템 파이썬이라 무관)
- 회귀 테스트: `MagicFormula/tests/test_mop_gauge.py` (집계·설정·통지·plist, 모킹)

## launchd

| Label | 시각 | 호출 |
|---|---|---|
| `com.kane.magicformula-mop-signal` | 매일 16:20 | `src/run_daily.py --rebuild` |
| `com.kane.magicformula-mop-gauge` | 매일 15:10 | `src/run_gauge.py` (휴장일 자체 판정 종료) |

정본 plist 는 `configs/launchd/` — 운영본은 `~/Library/LaunchAgents/` 로 symlink.
LLV 16:00 kis_update 종가 배치 이후 20분 여유. 등록은 Kane 수동.

## ⚠ 백테스트 수치를 실현 기대치로 읽지 말 것

검증된 엣지(홀드아웃 89일 보정알파 +0.505%, 시뮬 +4.9%)는 **t일 종가 매수 →
t+1 시가 매도** 라는 실행 불가능한 이상체결 기준이다. 종가+1% 웃돈이면 −2.7% 로
부호가 뒤집힌다(거래당 엣지 +0.65%). 데이 포트는 실행 가능성을 우선해
**17:00 NXT 애프터마켓 매수 + 당일고점 −1% 손절 보유**로 운영하므로 성과가 다르다
(청산 임계 정본 = StockPortfolio app/paper_day/config.py stop_drop_pct, 2026-07-30 −2%→−1%).
가상계좌의 목적은 그 괴리를 실측하는 것.
