# 황금률만들기 (Magic Formula)

종목 분석 페이지의 **종합 시그널 점수** 가중치를 백테스트로 검증하고,
도출된 최적 조합으로 **매일 진입 신호 리포트** 를 자동 생성하는 시스템.

**v2_combined 단일 체제** (2026-06-10 v1 완전 폐기):
4영역(추세·모멘텀·거래량·변동성) 레짐 적응 점수 + Wyckoff Markdown 게이트.

두 트랙으로 분리되어 있다:

- **데일리 트랙** — 평일 자동 실행, 당일 진입 신호 JSON/MD 리포트
- **분석 트랙** — 월 1회 오프라인 그리드 백테스트로 최적 가중치/임계값 도출

이 둘은 `configs/active_strategy.yaml` 단일 진실 원천(SSOT) 으로 연결된다.
분석에서 새 조합을 도출하면 yaml 한 줄 명령으로 반영되고, 데일리는 코드 수정 없이 다음 실행부터 적용한다.

---

## 디렉토리 구조

```
Magic Formula/
├── README.md                                 # 이 파일
├── requirements.txt                          # 의존성
├── .gitignore
│
├── magic_formula/                            # Python 패키지 v1.0 (v2 단일 체제)
│   ├── __init__.py                           # 공개 API (load_strategy, get_universe …)
│   ├── _vault.py                             # longlivevault 진입점 통합 (SSOT)
│   ├── config.py                             # active_strategy.yaml (v2) 로더/부분갱신
│   ├── indicators.py                         # 기술지표 헬퍼 (RSI/MACD/BB/ATR/OBV)
│   ├── main.py                               # 분석 트랙 본체 (그리드 백테스트 CLI)
│   ├── data/collector.py                     # OHLCV / KOSPI 수집 (vault 위임, full-column)
│   ├── analysis/area_scores.py               # ★ 4영역 점수 + 레짐 + 결합 (운영 점수 정본)
│   ├── analysis/ic_framework.py              # breadth 레짐 / IC 평가 프레임워크
│   ├── analysis/*_variants.py                # 영역별 점수 변형 (연구용)
│   ├── signals/rules.py                      # threshold_breakout 진입 + C1/TIME 청산
│   ├── simulator/simulator.py                # 매매 시뮬레이터 (yaml trading 스펙 1:1)
│   ├── metrics/metrics.py                    # 성과 지표 (robust 상위5제외 포함)
│   ├── optimizer/optimizer.py                # 가중치 그리드 백테스트 + 리포트
│   └── daily/
│       ├── runner.py                         # 데일리 트랙 본체 (파이프라인)
│       └── report.py                         # 데일리 MD 렌더링
│
├── scripts/                                  # 외부 진입점 (얇은 wrapper)
│   ├── daily_signal.py                       # 데일리 — launchd 가 호출
│   ├── run_analysis.py                       # 분석 — 월 1회 수동 실행
│   └── update_strategy.py                    # 분석 결과 → yaml 반영 (v2 가드 내장)
│
├── configs/                                  # 운영 설정 (git 추적)
│   ├── active_strategy.yaml                  # ★ 현재 운영 전략 (SSOT, v2_combined)
│   ├── active_strategy_v1.yaml               # 구 v1 백업 (참고용 — 코드 지원 종료)
│   ├── history/                              # yaml 변경 이력
│   └── launchd/                              # launchd 등록 plist + README
│
├── docs/
│   ├── architecture.md                       # 시스템 구조 / 두 트랙 / 데이터 흐름
│   ├── area_specs/                           # 영역별 신호 spec (M4 분석 확정)
│   └── ...
│
├── output/                                   # .gitignore (signals/ 만 추적)
│   ├── signals/                              # 데일리 결과 (JSON + MD + 레짐 사이드카)
│   ├── analysis/                             # 분석 산출물 (날짜별)
│   └── logs/                                 # launchd stdout / stderr
│
└── tests/                                    # pytest (config/signals/simulator/area_scores/vault)
```

---

## 빠른 시작

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

`longlivevault` 는 별도 프로젝트로 옆 폴더에 있다고 가정한다 (`../longlivevault`).
vault 의존성(pykrx 등) 은 vault 의 README 를 참고.

### 2. 데일리 시그널 (수동 실행)

```bash
python scripts/daily_signal.py
# → output/signals/daily_signal_YYYYMMDD.{json,md}
# → output/signals/daily_regimes_YYYYMMDD.json  (레짐 사이드카)
```

날짜는 **실행일이 아니라 데이터 마지막 거래일** 로 기록된다.

### 3. 데일리 시그널 (자동화)

`configs/launchd/README.md` 의 등록 방법을 따라 launchd 에 등록.

### 4. 분석 (월 1회)

```bash
python scripts/run_analysis.py                  # 그리드 56조합 × 2임계값
python scripts/run_analysis.py --quick-test     # 현 운영 전략 1조합 검증
python scripts/run_analysis.py --step 0.1       # 정밀 그리드 (286조합)
python scripts/run_analysis.py --no-gate        # Wyckoff 게이트 OFF 비교
```

산출물은 `output/analysis/YYYY-MM-DD/` 에 저장된다
(weight_ranking.md / backtest_report.md / summary.csv / trades.csv / equity_curves.png).

### 5. 분석 결과를 데일리에 반영

```bash
# 자동 — weight_ranking.md 의 1위 적용
python scripts/update_strategy.py \
    --from-ranking output/analysis/2026-06-15/weight_ranking.md

# 수동 — 가중치/임계값 직접 명시
python scripts/update_strategy.py \
    --weights "trend=0.2,momentum=0.2,volume=0.0,volatility=0.6" \
    --threshold 6.0

# 변경 사항만 미리 확인
python scripts/update_strategy.py --from-ranking ... --dry-run
```

기존 yaml 은 `configs/history/` 에 자동 백업된다.
대상 yaml 이 v2_combined 가 아니면 **갱신을 거부** 한다 (v1 덮어쓰기 사고 방지).

---

## 현재 운영 전략 (v2_combined)

`configs/active_strategy.yaml` 의 값이 운영 중인 전략이다.

| 필드 | 의미 |
|---|---|
| `strategy_id` | 전략 이름표 |
| `scoring.weights` | 4영역 가중치 (합 1.0): trend / momentum / volume / volatility |
| `scoring.threshold` | 진입 임계값 (현재 6.0) |
| `scoring.candidate_threshold` | 후보 모니터링 기준 (보통 5.0) |
| `scoring.universe` | 종목 집합: `core_all` / `core_excl_split` |
| `scoring.gate` | Wyckoff 국면 게이트 (Markdown 매수 제외) |
| `trading.entry.position_size` | 종목당 투입 자본 (10,000,000원) — simulator 와 동일 |
| `trading.exit` | C1(ATR 손절) / TIME(시간청산) / hold_if_profit / END |

---

## 점수 체계 (4영역 + 게이트)

각 거래일 `t` 에서 `t` 시점까지의 데이터만 사용 (look-ahead bias 방지).
영역별 ±10, 가중 결합 후 ±10 클립. 상세 spec: `docs/area_specs/*.md`.

| 영역 | 신호 | 레짐 |
|---|---|---|
| 추세 (`trend`) | Dv2(정30/크30/기40) + invert_dist_off_bull | breadth (10/10/0.60) |
| 모멘텀 (`momentum`) | RSI 10/90 극단 trend 단독 | 없음 (상시) |
| 거래량 (`volume`) | bear-only Q2+Q3+OBV_contra (강세장 0) | quickregime (3/5/0.52) |
| 변동성 (`volatility`) | BB×52주×레짐 결합 점수표 | quickregime |
| Wyckoff | 점수 아님 — **게이트** (Markdown 국면 매수 제외) | LLV parquet 컬럼 |

Wyckoff 컬럼(Label/Signal/Strength)은 LLV(longlivevault) 가 채운 parquet 정본을
읽기만 한다 — Magic Formula 는 hillstorm 을 import 하지 않는다.

---

## 매매 규칙 (시뮬레이터 = yaml 스펙 1:1)

| 단계 | 규칙 |
|---|---|
| 진입 | 종합점수 prev ≤ threshold < today (상향 돌파) → 다음날 시가 체결 |
| C1 | 종가 < 진입가 − ATR(14)×1 → 다음날 시가 전량 청산 (손절) |
| TIME | 보유 20거래일 + 누적손익 ≤ 0% → 다음날 시가 청산 (이익 중이면 유지) |
| END | 평가종료일 미청산 → 종가 강제 청산 |

비용: 슬리피지 ±0.10%, 수수료 0.015%×2, 거래세 0.20% (매도).

---

## 두 트랙 데이터 흐름

```
[분석 트랙]                              [데일리 트랙]
월 1회                                    매일 평일
                                                 ▲
output/analysis/                                 │ launchd
    └─ weight_ranking.md                         │
            │                                    │
            ▼                                    │
scripts/update_strategy.py  (v2 가드)            │
            │                                    │
            ▼                                    │
configs/active_strategy.yaml ────────── 읽기 ────┘
            ▲                                    │
            │                                    ▼
            └─ git history + configs/history/    output/signals/daily_signal_*.{json,md}
                                                 output/signals/daily_regimes_*.json
```

`configs/active_strategy.yaml` 이 두 트랙을 잇는 유일한 통신 채널.

---

## 변경 이력 (주요)

| 시점 | 작업 |
|---|---|
| 2026-05-20 | P1~P8 패키지화 / 두 트랙 분리 / vault 단일화 |
| 2026-05-31 | v2_combined 정본 승격 (M4 분석) |
| 2026-06-10 | **v1 완전 폐기 → v2 단일 체제.** 분석 트랙 v2 이식 (그리드 in-repo 재현), update_strategy v2 가드, 슬리피지 이중차감 수정, 자본 10M 통일, time_stop 구현, scorer→indicators 추출, runner/report 분리, 테스트 64개 |

---

## 작성

- Kane · 클로이 (Claude)
- 시작: 2026-05-16
