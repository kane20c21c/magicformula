# Magic Formula — 시스템 구조

작성: 2026-05-20 (P5b 트랙 분리 완료 시점)

---

## 한 줄 요약

**5영역 황금률 점수 → R1/R2/R3/ADAPTIVE 진입 규칙 → 매매 시뮬레이션 + 데일리 시그널 리포트.
분석 트랙이 도출한 최적 조합을 데일리 트랙이 yaml 한 줄로 받아 사용한다.**

---

## 두 트랙 분리

```
┌──────────────────────────────────┐    ┌──────────────────────────────────┐
│       분석 트랙 (월 1회)            │    │      데일리 트랙 (매일 평일)         │
│  무거움 · 오프라인 · 1,530회 백테스트  │    │  가벼움 · 자동화 · launchd 20:40    │
└──────────────────────────────────┘    └──────────────────────────────────┘
              │                                          ▲
              │ scripts/update_strategy.py                │ launchd
              ▼                                          │
       ┌─────────────────────────────────────────────────┘
       │  configs/active_strategy.yaml  (SSOT)
       │  ├ strategy_id, last_updated, source_analysis
       │  ├ weights {trend, momentum, volume, volatility, wyckoff}
       │  ├ rule (R1 / R2 / R3 / ADAPTIVE)
       │  ├ area4_mode (trend / contrarian)
       │  ├ threshold
       │  └ universe (core_all / core_excl_split, legacy: core_59 / core_57)
       └─────────────────────────────────────────────────┐
                                                         │
              ┌──────────────────────────────────────────┘
              │ magic_formula.config.load_strategy()
              ▼
       공통 코어 호출 — magic_formula.scoring / signals / simulator / metrics
```

이 yaml 한 파일이 두 트랙의 **유일한 통신 채널**이다.
분석은 새 yaml 을 쓰고, 데일리는 그 yaml 을 읽기만 한다.
서로의 코드를 알 필요가 없고, 한쪽 변경이 다른쪽을 깨뜨리지 않는다.

---

## 공통 코어 패키지

`magic_formula/` 패키지가 두 트랙이 공유하는 코어 로직.

```
magic_formula/
├── _vault.py             ← longlivevault 진입점 통합 (CORE_TICKERS / sector / 종목명 fallback)
├── config.py             ← active_strategy.yaml 로더 / 덤퍼 / 검증
├── data/collector.py     ← OHLCV / KOSPI 수집 (vault 위임)
├── scoring/scorer.py     ← 5영역 점수 + 종합 점수
├── signals/rules.py      ← R1/R2/R3 진입 + C1/C_WY/C3 청산
├── signals/adaptive_rule_selector.py  ← 종목별 동적 규칙
├── simulator/simulator.py← 매매 시뮬레이터 (cost/slippage 포함)
├── metrics/metrics.py    ← 총수익률·KOSPI 알파·MDD·Sharpe …
├── optimizer/optimizer.py← 가중치 조합 백테스트 + 리포트
├── daily/runner.py       ← 데일리 트랙 본체
├── analysis/backtest.py  ← 분석 트랙 진입점 (main 위임)
└── main.py               ← 백테스트 CLI (--quick-test 등)
```

---

## 데일리 트랙 흐름

```
launchd (평일 20:40 KST)
    │
    ▼
scripts/daily_signal.py  ──(얇은 wrapper)
    │
    ▼
magic_formula.daily.runner.run(target_date, config_path)
    │
    ├─ config.load_strategy()          ← configs/active_strategy.yaml
    ├─ _vault.get_universe(cfg.universe)
    │
    ├─ for ticker in tickers:
    │     ├─ vault.data_service.get_ohlcv(ticker, start, end)
    │     ├─ scoring.compute_scores(df, cfg.weights, cfg.area4_mode)
    │     └─ check_r1_signal(scored, cfg.threshold)
    │
    ├─ data_date = Counter(last_dates).most_common(1)
    └─ 저장:
       ├─ output/signals/daily_signal_{data_date}.json
       └─ output/signals/daily_signal_{data_date}.md
```

**핵심**:
- 파일명은 **실행일이 아니라 데이터 마지막 거래일** — 토/공휴일 다음 평일도 올바른 기준일 유지
- vault 가 SSOT — pykrx/KIS REST/yfinance 폴백 모두 제거됨 (P4)
- yaml 의 가중치/규칙 그대로 사용 — 코드 안에 하드코딩 없음 (P2)

---

## 분석 트랙 흐름

```
scripts/run_analysis.py
    │
    ▼
magic_formula.analysis.backtest.main()  ──(magic_formula.main 위임)
    │
    ▼
[데이터 수집] data.collector.collect_all()
        │
        ▼
[51조합 × 3규칙 × 57종목] optimizer.run_all()
        │
        ├─ for combo in 51조합:
        │     for rule in [R1, R2, R3]:
        │         for ticker in 57종목:
        │             ├─ scoring.compute_scores(...)
        │             ├─ simulator.run_simulation(...)
        │             └─ metrics.compute_metrics(...)
        │
        ▼
[리포트] make_weight_ranking_md / make_backtest_report_md / equity_curves.png
        │
        ▼
output/analysis/YYYY-MM-DD/
    ├─ weight_ranking.md          ← update_strategy 도구가 읽음
    ├─ backtest_report.md
    ├─ trades.csv
    ├─ summary_*.csv
    └─ equity_curves.png
```

---

## 분석 결과 → 데일리 반영

```
output/analysis/2026-06-15/weight_ranking.md
        │
        │  "| 1 | HLLBH | 23/22/27/10/18 | R3 | +185% | ..."
        │
        ▼
scripts/update_strategy.py
    │
    ├─ parse_ranking_md(path, top=1)
    │     → weights={trend:0.23, ...}, rule="R3"
    ├─ ActiveStrategy.from_dict(...)
    ├─ .validate()                            ← 가중치 합 = 1.0 등
    │
    ├─ (dry-run 이 아니면)
    │     ├─ 기존 yaml → configs/history/YYYY-MM-DD_strategy_id.yaml 백업
    │     └─ configs/active_strategy.yaml 갱신
    │
    └─ 변경 사항 diff 출력
```

다음 데일리 실행부터 자동 적용된다. 코드 수정 없이 yaml 한 줄로 운영 전략 변경.

---

## 5영역 황금률 점수

각 거래일 `t` 에서 **`t` 시점까지의 데이터만** 사용 (look-ahead bias 방지).

| 영역 | 키 | 산출 항목 | 점수 범위 |
|---|---|---|---|
| 추세 | `trend` | MA 정배열, 골든/데드 크로스, MA60 기울기 | -10 ~ +10 |
| 모멘텀 | `momentum` | RSI(14), Stoch %K, MACD vs Signal, MACD Hist 방향 | -10 ~ +10 |
| 거래량 | `volume` | 상대거래량 vs MA20, OBV 5일 기울기 | -10 ~ +10 |
| 변동성·위치 | `volatility` | BB %B, 52주 위치 (mode 에 따라 추세추종/평균회귀) | -10 ~ +10 |
| 심리·Wyckoff | `wyckoff` | Wyckoff 국면, Hope Vector, Anxiety Index | -10 ~ +10 |

**종합 점수** = Σ (영역 점수 × 가중치). Wyckoff hillstorm 미설치 시 wyckoff 가중치는 나머지 4개 영역에 비례 재분배 (`_effective_weights`).

---

## 진입 규칙

| ID | 조건 |
|---|---|
| R1 | composite_score 가 `threshold` 를 상향 돌파한 첫날 (prev ≤ threshold < today) |
| R2 | composite_score 가 `threshold` 이상이면서 종목 미보유 (절대 수준) |
| R3 | composite_score 가 음수→양수 부호 전환 + ADX(14) > 20 |
| ADAPTIVE | 종목별 점수 패턴 분류로 R1/R3/SKIP 동적 선택 (rolling lookback 60일) |

---

## 청산 규칙 (v3)

| ID | 조건 | 처리 |
|---|---|---|
| C1 | 종가 < 진입가 − ATR × 1 | 다음날 시가 전량 청산 (손절) |
| C_WY | Wyckoff 추세 전환 신호 (3일 연속 음수 또는 진입 점수 대비 -4 하락) | 다음날 시가 전량 청산 (익절/추세종료) |
| C3 | composite_score ≤ -3 | 다음날 시가 전량 청산 (급락 안전망) |
| END | 백테스트 종료일 미청산 | 종가 강제 청산 |

---

## 거래 비용 가정

| 항목 | 값 |
|---|---|
| 슬리피지 | 매수 +0.10%, 매도 -0.10% |
| 수수료 | 매수·매도 각 0.015% |
| 거래세 | 매도 0.20% |
| 종목당 자본 | 20,000,000 원 |
| 초기 자본 | 200,000,000 원 (10종목 균등) |

---

## 종목 universe

| 식별자 | 종목 수 | 사용처 |
|---|---|---|
| `core_all` | `len(CORE_TICKERS)` (현재 69) | 데일리 (전 종목 표시, 분석 제외 종목도 점수만은 산출) |
| `core_excl_split` | `len(CORE_TICKERS - DEFAULT_EXCLUDE)` (현재 67) | 분석 (사업분할 등 EXCLUDE 빼고 1,530회 백테스트) |
| `core_59` / `core_57` | (deprecated alias) | 60종목 시기의 이름. 각각 `core_all` / `core_excl_split` 와 동일하게 작동. 새 코드는 사용 금지. |

`_vault.DEFAULT_EXCLUDE = {"207940": 삼성바이오로직스, "0126Z0": 삼성에피스홀딩스}` — 사업 분할로 시계열 신뢰성 낮은 종목 제외.

코어 종목 수가 vault 에서 늘어나도 (60→69 등) 위 식별자는 그대로 작동 — 숫자가 아닌 의미(전체 / 분할제외) 로 정의되기 때문.

---

## 데이터 단일 진실 원천

```
longlivevault/data/ohlcv/
├── core.parquet                  ← 코어 59종목 통합 (OHLCV + Name + 21지표)
└── tickers/
    ├── KOSPI.parquet             ← 종합지수 (Magic Formula 알파 벤치마크)
    ├── KOSPI200.parquet
    ├── VKOSPI.parquet
    └── {ticker}.parquet           ← 비코어 + 기타
```

Magic Formula 는 이 데이터를 vault 의 `data_service.get_ohlcv` / `ohlcv_store.get_ohlcv` 로만 읽는다. 별도 캐시 / pykrx / KIS REST / yfinance 폴백은 모두 제거됨 (P4).

종목명도 vault parquet 의 `Name` 컬럼을 정본으로 사용 (KRX `ISU_NM` / KIS `hts_kor_isnm`). vault Name 이 None/빈 값일 때만 `_vault.TICKER_NAMES_FALLBACK` 사용 (P3).

---

## 변경 이력 (Phase 단위)

| Phase | 작업 | 주요 효과 |
|---|---|---|
| P1 | .gitignore, output 아카이브, docs 중복 제거 | 디렉토리 정리 |
| P2 | configs/active_strategy.yaml + config 로더 | 데일리/분석 인터페이스 확립 |
| P3 | `_vault.py` 단일화 | vault path / CORE_TICKERS / 종목명 / 섹터 통합 |
| P4 | collector.py vault 위임 | 845→281줄, 외부 폴백 4개 제거 |
| P5a | src/ → magic_formula/ 패키지화 | Python 패키지 정식화 |
| P5b | daily/ + analysis/ 트랙 분리 | 책임 명확화 |
| P6 | update_strategy.py 도구 | 분석 → yaml 자동 반영 |
| P7 | 종목명 fix + launchd 자동화 | 운영 자동화 |
| P8 | tests / docs / requirements | 운영 가이드 정비 |
