# Area 2 — 모멘텀 (Momentum) 영역 분석 결론

작성: 2026-05-29 · Kane & 클로이 (M4 분석)
출처: `scripts/test_momentum_*.py` 시리즈 (9개 분석)

---

## 한 줄 결론

**모멘텀 영역의 유일한 알파 신호는 RSI 극단(10/90) trend 단독 (+0.4pp).
Stoch %K / MACD 는 모멘텀 내부 결합에선 RSI 대비 도움 안 됨 (레짐 적응해도).**

---

## ① 점수 산출 spec — RSI 극단 trend 단독

| 항목 | 값 | 위치 |
|---|---|---|
| 인코딩 | 5-band 임계, 경계 10/90 (극단) | |
| 방향 | trend (과매수 = +) | |
| 점수 | RSI≥90 → **+10** / 80~90 → +5 / 30~70(중립) → 0 / 10~20 → −5 / ≤10 → **−10** | ±10 풀스케일 (추세와 통일) |
| coverage | ~0.15 (RSI 극단일 때만 베팅) | |

**핵심 발견**: RSI 90+ 극단 과매수 = 추세 폭발 (trend), 조정 아님 (meanrev).
경계가 어중간(70)하면 meanrev, 극단(90)이면 trend — **경계가 방향을 결정**.

```
RSI 5-band score (trend, 경계 10/90):
  RSI >= 90       → +10  (extreme 과매수 = 추세 가속)
  80 <= RSI < 90  → +5   (mild)
  30 <  RSI < 70  → 0    (중립, 베팅 안 함)  ※ mid band = (50±40/2)
  20 <  RSI <= 30 → -5
  RSI <= 10       → -10  (extreme 과매도)
```

> 점수 ±8/±4 (검증 시) → ±10/±5 (풀스케일) 변경은 단조 스케일이라
> hit_rate / coverage / IC 완전 불변 확인 (2026-05-29).

---

## ② 제외된 지표 — Stoch %K / MACD

| 지표 | 단독 best | 레짐 적응 후 | 결론 |
|---|---|---|---|
| Stoch %K | -2.5pp (meanrev) | -1.2pp | RSI 결합 도움 안 됨 |
| MACD | -4.0pp (trend) | -2.6pp | 동일 |

### 제외 근거 (모든 결합이 RSI 단독보다 나쁨)

| 변형 | realistic_hit | vs bench |
|---|---|---|
| **RSI 단독** | **0.549** | **+0.4pp** ← 최고 |
| RSI + Stoch 필터 | 0.540 | -0.1pp |
| RSI + Stoch 평균 | 0.526 | -1.6pp |
| RSI + Stoch_adaptive 결합 | 0.536 | -0.8pp |

### ⚠️ "제외"의 정확한 범위
- **모멘텀 영역 내부 결합에서만** RSI 대비 무가치.
- Stoch/MACD 는 **레짐 신호로서 0 은 아님** — 하락/전환장에서 강하게 양수
  (2026-05: Stoch +5.4, MACD +9.5 / 2025-08: Stoch +7.6).
- **5영역 결합 단계에서 레짐 다양성 제공 가능성** → 그때 재검토 (지금 폐기 아님).

---

## ③ 검증 성능 (2025-05-28 ~ 2026-05-28, h=3)

| 지표 | realistic_hit | vs bench | coverage |
|---|---|---|---|
| **RSI 극단 trend** | 0.549 | **+0.4pp** | 0.147 |
| Stoch (레짐적응) | 0.532 | -1.2pp | 0.421 |
| MACD (레짐적응) | 0.519 | -2.6pp | 0.705 |

전체 풀 bench (no-signal long-all, h=3): 0.544

### 월별 — RSI 시기 안정성

| 환경 | RSI 극단 vs bench |
|---|---|
| 강세장 (2025-06/10, 2026-04) | +0.5 ~ +1.7 |
| 조정장 | 0 ~ -1.9 |
| 하락 (2026-05) | +1.3 |

→ **거의 모든 월에서 0 또는 양수** (Stoch/MACD 와 정반대로 시기 안정).

---

## ④ 검증 horizon — 3일 (모멘텀 = 빠른 신호)

| horizon | bench | RSI 극단 best |
|---|---|---|
| h=3 | 0.544 | +0.4pp (10/90), realistic 0.549 |
| h=5 | 0.565 | -0.7pp |
| h=10 | 0.591 | -1.5pp |

→ 모멘텀은 단기(h=3)가 최적. horizon 길수록 bench 대비 악화.

---

## ⚠️ 한계

1. **+0.4pp 는 작음** — coverage 0.147 라 베팅 수 적어 신뢰구간 넓음. 백테스트 필요.
2. **RSI 극단 단조 개선 (0.512→0.573)** 은 진짜 신호 특징이나, 절대 우위는 미미.
3. **Stoch/MACD 레짐 적응이 추세만큼 안 살아남** — 추세는 turning point +15~33pp,
   모멘텀 Stoch 는 +5~9pp 수준 (강세장 손실이 상쇄).

---

## 분석 series

| 단계 | 스크립트 | 핵심 |
|---|---|---|
| 1 | `test_momentum_variants_horizons.py` | 88변형 → 모두 bench 미달 (-9pp, 추세보다 약) |
| 2 | `test_momentum_sigmoid_h3.py` | 임계+시그모이드 양방향 → h=3 무차이 |
| 3 | `test_momentum_per_indicator.py` | 지표별 단독 → MACD trend, RSI/Stoch meanrev (상쇄 구조 발견) |
| 4 | `test_momentum_kgrid.py` | k 그리드 → bench 못 넘음 |
| 5 | `test_momentum_threshold_grid.py` | 임계 경계 → **RSI 10/90 trend +0.4pp 발견** |
| 6 | `test_momentum_combination.py` | 결합 → RSI 단독이 best, Stoch 무가치 |
| 7 | `test_momentum_monthly.py` | 월별 → Stoch/MACD 가 레짐 신호임 발견 |
| 8 | `test_momentum_regime.py` | 레짐 적응 → Stoch/MACD 개선되나 RSI 단독 못 넘음 |

---

## 다음 단계

- 거래량 (Area 3) 분석 — RSI 와 독립 정보원 (OBV/상대거래량)
- 추후 5영역 결합 단계에서 Stoch/MACD 레짐 신호 재검토
