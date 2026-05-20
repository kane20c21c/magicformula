# 황금률만들기 (Magic Formula)

종목 분석 페이지의 **종합 시그널 점수** 가중치를 백테스트로 검증하고,
도출된 최적 조합으로 **매일 진입 신호 리포트** 를 자동 생성하는 시스템.

두 트랙으로 분리되어 있다:

- **데일리 트랙** — 평일 20:40 KST 자동 실행, 당일 진입 신호 JSON/MD 리포트
- **분석 트랙** — 월 1회 오프라인 백테스트로 최적 가중치/규칙 도출, 그 결과를 데일리에 반영

이 둘은 `configs/active_strategy.yaml` 단일 진실 원천(SSOT) 으로 연결된다.
분석에서 새 조합을 도출하면 yaml 한 줄 명령으로 반영되고, 데일리는 코드 수정 없이 다음 실행부터 적용한다.

---

## 디렉토리 구조

```
Magic Formula/
├── README.md                                # 이 파일
├── requirements.txt                          # 의존성
├── .gitignore
│
├── magic_formula/                            # Python 패키지 v0.5
│   ├── __init__.py                           # 공개 API (load_strategy, get_universe …)
│   ├── _vault.py                             # longlivevault 진입점 통합 (SSOT)
│   ├── config.py                             # active_strategy.yaml 로더 / 덤퍼
│   ├── main.py                               # 백테스트 CLI (analysis 가 위임)
│   ├── data/collector.py                     # OHLCV / KOSPI 수집 (vault 위임)
│   ├── scoring/scorer.py                     # 5영역 점수 + 종합 점수
│   ├── signals/rules.py                      # 진입 (R1/R2/R3) + 청산 규칙
│   ├── signals/adaptive_rule_selector.py     # 종목별 동적 규칙 선택
│   ├── simulator/simulator.py                # 매매 시뮬레이터
│   ├── metrics/metrics.py                    # 성과 지표 (총수익률·알파·MDD …)
│   ├── optimizer/optimizer.py                # 가중치 조합 백테스트
│   ├── daily/runner.py                       # 데일리 트랙 본체
│   └── analysis/backtest.py                  # 분석 트랙 진입점 (main 위임)
│
├── scripts/                                  # 외부 진입점 (얇은 wrapper)
│   ├── daily_signal.py                       # 데일리 — launchd 가 호출
│   ├── run_analysis.py                       # 분석 — 월 1회 수동 실행
│   └── update_strategy.py                    # 분석 결과 → yaml 반영 도구
│
├── configs/                                  # 운영 설정 (git 추적)
│   ├── active_strategy.yaml                  # ★ 현재 운영 전략 (SSOT)
│   ├── history/                              # yaml 변경 이력
│   └── launchd/                              # launchd 등록 plist + README
│
├── docs/
│   ├── architecture.md                       # 시스템 구조 / 두 트랙 / 데이터 흐름
│   ├── backtest_design_v2.md                 # 백테스트 설계
│   └── adaptive_rule_selector.md             # ADAPTIVE 규칙 설계
│
├── output/                                   # .gitignore (signals/ 만 추적)
│   ├── signals/                              # 데일리 결과 (JSON + MD)
│   ├── analysis/                             # 분석 산출물 (날짜별)
│   └── logs/                                 # launchd stdout / stderr
│
└── tests/                                    # pytest
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
```

날짜는 **실행일이 아니라 데이터 마지막 거래일** 로 기록된다.

### 3. 데일리 시그널 (자동화)

`configs/launchd/README.md` 의 등록 방법을 따라 launchd 에 등록하면 평일 20:40 KST 자동 실행.

```bash
ln -sf "$(pwd)/configs/launchd/com.stolab.magic-formula.daily-signal.plist" \
       ~/Library/LaunchAgents/com.stolab.magic-formula.daily-signal.plist
launchctl load ~/Library/LaunchAgents/com.stolab.magic-formula.daily-signal.plist
```

### 4. 분석 (월 1회)

```bash
python scripts/run_analysis.py                       # 전체 실행 (1,530회 백테스트)
python scripts/run_analysis.py --quick-test          # 빠른 검증
```

산출물은 `output/analysis/YYYY-MM-DD/` 에 저장된다.

### 5. 분석 결과를 데일리에 반영

```bash
# 자동 — weight_ranking.md 의 1위 적용
python scripts/update_strategy.py \
    --from-ranking output/analysis/2026-06-15/weight_ranking.md \
    --strategy-id CompR-202606

# 수동 — 가중치 직접 명시
python scripts/update_strategy.py \
    --weights "trend=0.23,momentum=0.22,volume=0.27,volatility=0.10,wyckoff=0.18" \
    --rule R3

# 변경 사항만 미리 확인
python scripts/update_strategy.py --from-ranking ... --dry-run
```

기존 yaml 은 `configs/history/` 에 자동 백업된다.

---

## 현재 운영 전략

`configs/active_strategy.yaml` 의 값이 운영 중인 전략이다.
변경하려면 `scripts/update_strategy.py` 를 사용. 직접 yaml 을 편집해도 되지만
검증 / history 백업 / 표기 일관성을 위해 도구 사용을 권장.

| 필드 | 의미 | 허용값 |
|---|---|---|
| `strategy_id` | 전략 이름표 (사람이 부르는 ID) | 자유 |
| `weights` | 5영역 가중치 (합 1.0) | trend / momentum / volume / volatility / wyckoff |
| `rule` | 진입 규칙 | R1 / R2 / R3 / ADAPTIVE |
| `area4_mode` | 변동성·위치 점수 산출 방식 | trend (추세추종) / contrarian (평균회귀) |
| `threshold` | R1·R2 임계값 | float, 보통 +1 ~ +6 |
| `universe` | 종목 집합 | core_57 (분석) / core_59 (데일리) / core_all |

---

## 두 트랙 데이터 흐름

```
[분석 트랙]                              [데일리 트랙]
월 1회                                    매일 평일 20:40 KST
                                                 ▲
output/analysis/                                 │ launchd
    └─ weight_ranking.md                         │
            │                                    │
            ▼                                    │
scripts/update_strategy.py                       │
            │                                    │
            ▼                                    │
configs/active_strategy.yaml ────────── 읽기 ────┘
            ▲                                    │
            │                                    ▼
            └─ git history + configs/history/    output/signals/daily_signal_YYYYMMDD.{json,md}
```

`configs/active_strategy.yaml` 이 두 트랙을 잇는 유일한 통신 채널.

---

## 5영역 가중치 시스템

각 거래일 `t` 에서 `t` 시점까지의 데이터만 사용해 5영역 점수를 산출한다 (look-ahead bias 방지).
각 영역은 -10 ~ +10 점, 가중평균으로 -10 ~ +10 종합 점수.

| 영역 | 산출 항목 |
|---|---|
| 추세 (`trend`) | MA 정배열, 골든/데드 크로스, MA60 기울기 |
| 모멘텀 (`momentum`) | RSI, Stoch %K, MACD vs Signal, MACD Histogram 방향 |
| 거래량 (`volume`) | 상대거래량 (vs MA20), OBV 방향 |
| 변동성·위치 (`volatility`) | BB %B, 52주 위치 |
| 심리·Wyckoff (`wyckoff`) | Wyckoff 국면, Hope Vector, Anxiety Index |

상세는 [docs/backtest_design_v2.md](docs/backtest_design_v2.md) 와 [docs/architecture.md](docs/architecture.md) 참고.

---

## 작성

- Kane · 클로이 (Claude)
- 시작: 2026-05-16
- 패키지 구조 정리: 2026-05-20 (P1~P7)
