# Magic Formula — 시스템 구조

작성: 2026-05-20 (P5b 트랙 분리) / 전면 개정: 2026-06-10 (v1 폐기, v2 단일 체제)

---

## 한 줄 요약

**4영역 레짐 적응 점수 + Wyckoff Markdown 게이트 → threshold_breakout 진입 →
매매 시뮬레이션 + 데일리 시그널 리포트.
분석 트랙이 도출한 최적 조합을 데일리 트랙이 yaml 한 줄로 받아 사용한다.**

---

## 두 트랙 분리

```
┌──────────────────────────────────┐    ┌──────────────────────────────────┐
│       분석 트랙 (월 1회)            │    │      데일리 트랙 (매일 평일)         │
│  무거움 · 오프라인 · 그리드 백테스트   │    │  가벼움 · 자동화 · launchd          │
└──────────────────────────────────┘    └──────────────────────────────────┘
              │                                          ▲
              │ scripts/update_strategy.py (v2 가드)      │ launchd
              ▼                                          │
       ┌─────────────────────────────────────────────────┘
       │  configs/active_strategy.yaml  (SSOT, system_version: v2_combined)
       │  ├ strategy_id, last_updated, source_analysis
       │  ├ scoring.weights {trend, momentum, volume, volatility}  (합 1.0)
       │  ├ scoring.threshold / candidate_threshold / universe
       │  ├ scoring.regimes (breadth / quickregime 파라미터)
       │  ├ scoring.gate (Markdown 매수 제외)
       │  └ trading.entry.position_size (10,000,000원) + trading.exit
       └─────────────────────────────────────────────────┐
                                                         │
              ┌──────────────────────────────────────────┘
              │ magic_formula.config.load_strategy()  → StrategyV2
              ▼
       공통 코어 호출 — analysis.area_scores / signals / simulator / metrics
```

이 yaml 한 파일이 두 트랙의 **유일한 통신 채널**이다.
`update_strategy.py` 는 대상 yaml 의 `system_version` 이 `v2_combined` 가 아니면
갱신을 거부한다 (구 스키마 덮어쓰기 사고 방지).

---

## 공통 코어 패키지

```
magic_formula/
├── _vault.py                 ← longlivevault 진입점 통합 (CORE_TICKERS / universe / 종목명)
├── config.py                 ← active_strategy.yaml (v2) 로더 / 검증 / 부분갱신
├── indicators.py             ← 기술지표 헬퍼 (RSI / Stoch / MACD / BB / ATR / OBV)
├── data/collector.py         ← OHLCV / KOSPI 수집 (vault 위임, full-column passthrough)
├── analysis/area_scores.py   ← ★ 운영 점수 정본: 4영역 + 레짐 + combine + 게이트
├── analysis/ic_framework.py  ← breadth 레짐 / IC 평가
├── signals/rules.py          ← threshold_breakout 진입 + C1/TIME 청산 판단
├── simulator/simulator.py    ← 매매 시뮬레이터 (비용/슬리피지, yaml trading 1:1)
├── metrics/metrics.py        ← 총수익률·알파·MDD·Sharpe·robust(상위5제외) …
├── optimizer/optimizer.py    ← 가중치 그리드 백테스트 + 랭킹/리포트
├── daily/runner.py           ← 데일리 트랙 본체 (파이프라인)
├── daily/report.py           ← 데일리 MD 렌더링 / JSON 직렬화 헬퍼
└── main.py                   ← 분석 트랙 CLI (scripts/run_analysis.py 가 위임)
```

핵심 설계: **영역 점수는 가중치와 무관** → `compute_area_scores()` 로 종목당 1회만
계산하고, `combine_scores()` 로 가중 결합만 반복한다 (데일리·그리드 공용).

---

## 데일리 트랙 흐름

```
launchd (평일)
    │
    ▼
scripts/daily_signal.py  ──(얇은 wrapper)
    │
    ▼
magic_formula.daily.runner.run(target_date, config_path)
    │
    ├─ config.load_strategy()              ← active_strategy.yaml (v2 검증 포함)
    ├─ _vault.get_universe(cfg.universe)
    ├─ 전 종목 full-column OHLCV 로드       ← LLV Wyckoff_Label/Signal 포함
    ├─ area_scores.make_regimes(전 종목)    ← breadth + quickregime (횡단면)
    │
    ├─ for ticker:
    │     ├─ compute_area_scores(df, 레짐)          # 종목당 1회
    │     ├─ combine_scores(areas, 가중치, 게이트)   # 종합점수 (게이트 NaN)
    │     └─ check_breakout_signal(prev ≤ thr < today)
    │
    ├─ data_date = 전 종목 마지막 거래일 최빈값
    └─ 저장:
       ├─ output/signals/daily_signal_{data_date}.json   (스키마 — 투자포폴 소비)
       ├─ output/signals/daily_signal_{data_date}.md
       └─ output/signals/daily_regimes_{data_date}.json  (레짐 사이드카)
```

**핵심**:
- 파일명은 **실행일이 아니라 데이터 마지막 거래일**
- vault 가 SSOT — Wyckoff 컬럼도 LLV parquet 정본 (hillstorm import 없음)
- yaml 의 가중치/임계값/게이트 그대로 사용 — 코드 안에 하드코딩 없음
- 레짐 사이드카는 StockPortfolio 즉석 v2 계산이 빌려 쓴다 (시장 공통)

---

## 분석 트랙 흐름

```
scripts/run_analysis.py
    │
    ▼
magic_formula.main.main()
    │
    ├─ [수집] collector.collect_all (universe = yaml scoring.universe)
    │
    ├─ [사전계산] optimizer.prepare_scoring_inputs
    │       레짐 2종 + 종목당 영역점수 4종 + ATR + Wyckoff phase   (1회)
    │
    ├─ [그리드] 가중치 조합 (step 0.2 → 56개) × 임계값 (5.0 / 6.0)
    │       combine_scores → run_simulation → compute_metrics
    │
    ▼
output/analysis/YYYY-MM-DD/
    ├─ weight_ranking.md          ← robust 평균수익 기준 정렬, update_strategy 가 파싱
    ├─ backtest_report.md
    ├─ summary.csv / trades.csv
    └─ equity_curves.png          (상위 5조합, matplotlib 설치 시)
```

랭킹 정렬 기준은 **robust 평균수익** (거래 수익 상위 5건 제외 평균) —
소수 대박 거래 의존 조합에 패널티를 준다 (M4 방식).

---

## 분석 결과 → 데일리 반영

```
output/analysis/2026-06-15/weight_ranking.md
        │
        │  "| 1 | T20/M20/Vu0/Va60 | 6.0 | +3.21% | ..."
        │
        ▼
scripts/update_strategy.py
    │
    ├─ load_strategy()                  ← v2 가드 (v1 yaml 이면 중단)
    ├─ parse_ranking_md(path, top=1)    → weights + threshold
    ├─ diff 출력 + 검증 (가중치 합 1.0 등)
    │
    ├─ (dry-run 이 아니면)
    │     ├─ 기존 yaml → configs/history/ 백업
    │     └─ scoring.weights / threshold / strategy_id 만 갱신
    │        (레짐/게이트/trading 구조 보존, ruamel 설치 시 주석도 보존)
    └─ 다음 데일리 실행부터 자동 적용
```

---

## 매매 규칙 (simulator = yaml trading 1:1)

| 단계 | 규칙 | 처리 |
|---|---|---|
| 진입 | composite prev ≤ threshold < today (게이트 NaN 은 신호 없음) | 다음날 시가 체결 |
| C1 | 종가 < 진입가 − ATR(14)×1 | 다음날 시가 전량 청산 (손절) |
| TIME | 보유 20거래일 이상 + 누적손익 ≤ 0% (이익 중이면 유지) | 다음날 시가 전량 청산 |
| END | 평가종료일 미청산 | 종가 강제 청산 |

---

## 거래 비용·자본 가정

| 항목 | 값 |
|---|---|
| 슬리피지 | 매수 +0.10%, 매도 -0.10% |
| 수수료 | 매수·매도 각 0.015% |
| 거래세 | 매도 0.20% |
| 종목당 자본 | **10,000,000 원** (yaml trading.entry.position_size 정본) |
| 수익률 정규화 기준 | 200,000,000 원 (metrics.INITIAL_CAPITAL) |

비용 산식 (2026-06-10 수정): 체결가에 슬리피지를 1회만 반영.
매수비용 = 체결가×수량×(1+수수료), 매도수령 = 체결가×수량×(1−수수료−거래세).

---

## 종목 universe

| 식별자 | 종목 수 | 사용처 |
|---|---|---|
| `core_all` | `len(CORE_TICKERS)` (현재 69) | 데일리 표시용 전체 |
| `core_excl_split` | `len(CORE_TICKERS - DEFAULT_EXCLUDE)` (현재 67) | 데일리 v2 / 분석 |
| `core_59` / `core_57` | (deprecated alias) | 기존 설정 호환용 — 새 코드 사용 금지 |

`_vault.DEFAULT_EXCLUDE = {"207940", "0126Z0"}` — 사업 분할로 시계열 신뢰성 낮은 종목 제외.

---

## 데이터 단일 진실 원천

```
longlivevault/data/ohlcv/
├── core.parquet                  ← 코어 종목 OHLCV + Name + 23지표 + Wyckoff 4컬럼
└── tickers/
    ├── KOSPI.parquet             ← 종합지수 (알파 벤치마크)
    └── {ticker}.parquet
```

Magic Formula 는 vault 의 `data_service.get_ohlcv` / `ohlcv_store.get_ohlcv` 로만 읽는다.
Wyckoff_Label/Signal/Signal_Strength 도 LLV 가 채운 컬럼을 그대로 받는다 —
**hillstorm 직접 호출 없음** (LLV 데이터 소유권 원칙).

---

## 변경 이력 (Phase 단위)

| Phase | 작업 | 주요 효과 |
|---|---|---|
| P1~P8 | 2026-05-20 패키지화 / 트랙 분리 / vault 단일화 / launchd | 기반 구조 |
| M4 | 2026-05-30 v2_combined 분석 (영역 spec + robust 그리드) | 운영 전략 도출 |
| v2 승격 | 2026-05-31 active_strategy.yaml v2 정본 | 데일리 v2 운영 |
| **v2 단일화** | **2026-06-10 v1 완전 폐기** | 분석 트랙 v2 이식(in-repo 재현), update_strategy v2 가드, 슬리피지 이중차감 수정, 자본 10M 통일, time_stop 구현, scorer→indicators, runner/report 분리, 테스트 64개 |
