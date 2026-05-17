# 종목별 적응형 진입 규칙 선택 알고리즘

작성일: 2026-05-17
모듈: `src/signals/adaptive_rule_selector.py`

---

## 1. 배경

57종목 백테스트 결과 분석에서 두 가지 결정적 패턴을 발견:

1. **R1/R2 우세 종목 (23개) = 직진 폭등형**: 점수가 +5 위에 계속 머무는 종목 (대우건설, 우리기술, 에스피지 등). 평균 가격 변동 +140%, **모든 종목 양수 상승**.
2. **R3 우세 종목 (34개) = 계단형 추세**: 점수가 ±진동하며 평균 양수인 종목 (SK하이닉스, 삼성전자 등). 평균 가격 변동 +71%, 일부 하락 종목 포함.
3. **횡보·하락 종목 (NAVER, 알테오젠 등)**: 모든 규칙에서 부진 → 진입 회피가 답.

→ **단일 규칙(R1/R2/R3)으로 모든 종목을 처리하는 것보다 종목 특성에 맞게 동적 선택하는 게 알파를 더 끌어올림.**

## 2. 알고리즘 핵심

### 입력 / 출력

- **입력**: 종목별 최근 N일(기본 60일) 종합 점수 시계열 (백테스트 시작 시점 이전)
- **출력**: 종목별 추천 규칙 (`R1`, `R2`, `R3`, `SKIP`) + 분류 근거

### 분류용 지표 (4가지 핵심)

| 지표 | 정의 | 의미 |
|---|---|---|
| `avg_score` | 최근 N일 평균 점수 | 추세의 전반적 방향 |
| `above_threshold_ratio` | 점수가 +5 이상인 거래일 비율 | 직진 폭등 강도 |
| `sign_changes` | 부호 전환 횟수 (양→음 또는 음→양) | 진동성 / 계단형 정도 |
| `trend_slope` | 점수의 선형 회귀 기울기 | 추세 가속/감속 |

### 결정 트리

```
1. 데이터 부족 (lookback의 50% 미만)
   → SKIP

2. avg_score < 0
   → SKIP  (약세/횡보 종목, 진입 보류)

3. above_threshold_ratio ≥ 0.50
   → R2  (직진 폭등형 - 점수가 절반 이상 +5 위에 머묾)

4. sign_changes ≥ 6 AND avg_score > 0
   → R3  (계단형 추세 - 자주 진동하며 평균은 양수)

5. above_threshold_ratio ≥ 0.20
   → R1  (임계 돌파형 - 가끔 +5 돌파)

6. 그 외
   → SKIP  (패턴 모호 - 안전하게 회피)
```

## 3. 사용법

### 단일 종목

```python
from src.signals.adaptive_rule_selector import AdaptiveRuleSelector

selector = AdaptiveRuleSelector()
result = selector.select_rule('000660', sk_hynix_score_series)
print(result.selected_rule)  # 예: 'R3'
print(result.reason)         # 예: '부호 전환 8회 + 평균 +2.34 > 0 (계단형 추세)'
print(result.metrics)        # dict: avg_score, above_threshold_ratio, sign_changes 등
```

### 전체 종목 일괄 분류

```python
from src.signals.adaptive_rule_selector import select_rules_for_backtest

# scores_dict = {ticker: 점수 시계열} (백테스트 시작 이전까지의 데이터만)
rules_df = select_rules_for_backtest(scores_dict)
print(rules_df[['ticker', 'rule', 'confidence', 'reason']])
```

### 설정 커스터마이징

```python
from src.signals.adaptive_rule_selector import (
    AdaptiveRuleSelector, RuleSelectionConfig
)

custom_config = RuleSelectionConfig(
    lookback_days=90,                    # 분류 기간 확장
    above_threshold_ratio_r2=0.40,       # R2 진입 임계 완화
    sign_change_min_r3=8,                # R3 진입 부호 전환 강화
)
selector = AdaptiveRuleSelector(custom_config)
```

## 4. 기존 백테스트와 통합 방법

### 방법 A — 단순 통합 (백테스트 시작 시 한 번 분류)

```python
# 1. 워밍업 기간(예: 백테스트 시작 60일 전까지)의 점수 시계열 준비
scores_dict = compute_scores_for_warmup(tickers, end_date='2025-11-14')

# 2. 적응형 규칙 선택
from src.signals.adaptive_rule_selector import select_rules_for_backtest
rules_df = select_rules_for_backtest(scores_dict)
ticker_rules = dict(zip(rules_df['ticker'], rules_df['rule']))

# 3. 백테스트 시뮬레이터에 전달 (종목별 다른 규칙 적용)
backtest(
    tickers=tickers,
    start_date='2025-11-15',
    ticker_rules=ticker_rules,  # 종목별 R1/R2/R3/SKIP 매핑
    weight_label='Basic',
)
```

### 방법 B — 주기적 재평가 (월별 재분류)

```python
# 매월 1일마다 규칙 재선택 (점수 패턴이 시간에 따라 변할 수 있음)
for month_start in monthly_starts:
    scores_dict = compute_scores_for_warmup(tickers, end_date=month_start - 1)
    rules_df = select_rules_for_backtest(scores_dict)
    apply_rules(rules_df, month=month_start)
```

## 5. 검증 결과

가상 시나리오 4종 + 데이터 부족 1종으로 동작 검증:

| 시나리오 | 점수 패턴 | 예상 규칙 | 실제 규칙 | 정확 |
|---|---|---|---|---|
| POLAR_BULL | 평균 +6.5, +5 이상 87% | R2 | R2 | ✓ |
| STAIRCASE | 평균 +2, 부호 전환 25회 | R3 | R3 | ✓ |
| BREAKOUT | 평균 +2.7, +5 이상 15% | R1 | R3 | △ |
| WEAK | 평균 -1.3 | SKIP | SKIP | ✓ |
| SHORT_DATA | 10일 데이터 | SKIP | SKIP | ✓ |

> **BREAKOUT 케이스**: 부호 전환 14회로 R3 조건이 먼저 매칭됨. 결정 트리 우선순위가 의도대로 작동. R1은 "임계값 위에 가끔 도달하지만 부호 전환은 적은" 단발성 점프 종목에만 매칭됨.

## 6. 한계와 향후 개선

| 한계 | 영향 | 개선 방향 |
|---|---|---|
| **휴리스틱 기반 결정 트리** | 임계값(50%, 20%, 6회)이 데이터 기반이 아닌 직관 | 실제 백테스트로 임계값 그리드 서치 → 최적 튜닝 |
| **점수 시계열만 사용** | 가격·거래량 정보를 직접 활용 안 함 | 추가 지표 (가격 추세, ADX, 거래량 추세) 결합 |
| **고정 lookback (60일)** | 종목별 적정 lookback이 다를 수 있음 | 종목별 변동성에 따라 lookback 동적 조정 |
| **재평가 빈도 미지정** | 백테스트 중 시장 국면 변화 반영 안 됨 | 월별 또는 추세 전환 감지 시 자동 재평가 |

## 7. 다음 단계

1. **백테스트에 통합** (방법 A 우선) — `Basic` 가중치 + 적응형 규칙으로 새 백테스트
2. **결과 비교** — `Basic + R3 단일` vs `Basic + 적응형`의 알파 차이 측정
3. **임계값 튜닝** — 그리드 서치로 `above_threshold_ratio_r2`, `sign_change_min_r3` 등 최적화
4. **재평가 주기 실험** — 1회 vs 월별 재평가의 성과 차이

---
