# Area 1 — 추세 (Trend) 영역 분석 결론

작성: 2026-05-29 · Kane & 클로이 (M4 분석)
출처: `scripts/test_trend_variants*.py` 시리즈 (5개 분석)

---

## 한 줄 결론

**현 추세 점수 (`Dv2_3_3_4`) 는 단독으론 약함 (vs bench −2.1pp). 다만 breadth 기반 레짐 적응 (`invert_dist_off_bull`) 으로 turning point 검출 시 큰 alpha (+15~33pp) 가능성 있음.**

---

## ① 점수 산출 spec

### 변형 ID: `Dv2_3_3_4`

| 항목 | 정의 | 위치 |
|---|---|---|
| 함수 | `score_D_v2(df, weights=(0.3, 0.3, 0.4))` | `magic_formula/analysis/trend_variants.py` |
| 가중치 | 정배열 0.30 / 크로스 0.30 / 기울기 0.40 | (sub 비중) |
| 합산 공식 | `(wa·a + wc·c + ws·s) / Σw`, clip ±10 | Σw = 1.0 |
| **이론상 max** | **±10.00** (풀 스케일) | |

### Sub-component (A 변형 — sub cap ±10)

| Sub | 함수 | 공식 | 범위 |
|---|---|---|---|
| 정배열 (align) | `_align_A` | MA5>MA20→+5, +MA20>MA60→+10 / 역배열→−10 | ±10 |
| 크로스 (cross) | `_cross_A` | 골든+10 / 데드−10, 5일 ffill | ±10 |
| 기울기 (slope) | `_slope_A` | MA60 5일 변화율 % × 50, clip | ±10 |

### 데이터 부족 처리
- `len(df) < 65` (= MA60 + 5일 기울기) → 전 구간 점수 = 0
- 구성 sub 중 NaN 있으면 그 행 점수 = 0

---

## ② 레짐 판별기 — breadth + breadth_trend

### 정의

```
시점 t의 breadth(t) =
    시점 [t-9, t] 거래일 × 67종목 = 670개 점 중
    각 점의 "10일 backward 수익률" 이 > 0 인 비율
```

| 파라미터 | 값 | 의미 |
|---|---|---|
| `breadth_lookback` | **10** 거래일 | 윈도우 크기 |
| `breadth_horizon` | **10** 거래일 | 각 점의 측정 간격 (신호 평가와 일치) |
| `breadth_trend_lookback` | **5** 거래일 | breadth 변화량 계산 lookback |
| `HIGH_THR` | **0.60** | 강세 임계 |
| `LOW_THR` | **0.40** | 하락 임계 |

함수: `magic_formula/analysis/ic_framework.py::compute_breadth_series`

### 4-mode 레짐 분류

```
def classify(breadth_now, breadth_trend):
    if breadth_now > 0.60:
        return "강세지속" if breadth_trend >= 0 else "강세약화"  # ← Distribution 의심
    if breadth_now < 0.40:
        return "하락"
    return "조정"
```

---

## ③ 신호 처리 — `invert_dist_off_bull` 모드

| 레짐 | 점수 처리 | 의미 |
|---|---|---|
| 강세지속 | **score = 0** (off) | 강세장에선 베팅 안 함 (단순 long-all 가 이김) |
| 강세약화 (Distribution) | **score × −1** (invert) | 추세 신호가 mean-reversion 으로 작동 |
| 조정 | **그대로** | 정상 신호 |
| 하락 | **그대로** | 정상 신호 (강한 양수 신호) |

---

## ④ 검증 성능 (2025-05-28 ~ 2026-05-28, 12개월)

### 전체 풀링

| 지표 | 값 |
|---|---|
| 평가 모집단 (n_directional_f) | ~15,000 obs |
| 전체 풀 bench (no-signal long-all) | 0.591 |
| realistic_hit | **0.570** |
| vs bench | **−2.1pp** |
| coverage | 70% (강세지속 71일 off) |

### 시기별 (월별 vs bench, pp)

| 월 | bench | 환경 | realistic_hit vs bench |
|---|---|---|---|
| 2025-06 | 0.705 | 강세 | −2.4 |
| 2025-09 | 0.642 | 강세 | −1.6 |
| 2025-10 | 0.675 | 강세 | −1.6 |
| **2025-11** | **0.493** | **전환** | **🔺 +15.7** |
| 2026-01 | 0.730 | 강세 | +1.1 |
| 2026-04 | 0.721 | 강세 | +0.9 |
| **2026-05** | **0.294** | **폭락** | **🔺 +33.1** |

→ **강세장 7개월: −1.6 ~ +1.1pp (안전)**, **전환점 2건: +15.7 / +33.1pp**

### 레짐 분포 (244 거래일)

| 레짐 | 일수 | 비중 |
|---|---|---|
| 강세지속 | 71일 | 29% |
| 강세약화 | 49일 | 20% |
| 조정 | 90일 | 37% |
| 하락 | 34일 | 14% |

---

## ⚠️ 한계와 주의사항

1. **평균은 long-all 못 이김** — vs bench −2.1pp. 12개월 단순 매수가 평균적으로 더 좋음.
2. **turning point 2건 의존** — alpha 폭발이 2025-11과 2026-05 두 번. **overfitting 의심**. 다른 시장 사이클에서 재현 여부 미검증.
3. **백테스트 미연계** — IC/hit_rate 기반 분석. 실제 거래 시 진입/청산/비용 영향 미검증.
4. **단기 (h=5) / 장기 (h=20) 일관성** — multi-horizon test (별도 스크립트) 에서도 베이스라인 격차 확대. h=10 에 최적화된 spec.

---

## 분석 series (이 결론에 이르기까지)

| 단계 | 스크립트 | 핵심 결과 |
|---|---|---|
| 1 | `test_trend_variants.py` (v1) | 51 변형 풀링 → 모두 hit_rate < 0.5 (KOSPI alpha) |
| 2 | `test_trend_variants.py` (v2) | raw return + warmup 6m + Dv2 그리드 36 → hit_rate ~0.58 (vs bench 0.65) |
| 3 | `test_trend_variants_horizons.py` | h=5/10/20 모두 vs bench 음수 (−3.2 ~ −6.4pp) |
| 4 | `test_trend_variants_sectors.py` | 15섹터 × 5 변형 = 75 case 중 양수 1개 (은행+Dv2, +0.4pp) |
| 5 | `test_trend_variants_monthly.py` | 강세장 음수, 하락장 양수 — 시장 레짐 의존성 발견 |
| 6 | `test_trend_regime_adaptive.py` | breadth (20/5) + simple 3-mode → off가 좋아 보였지만 모집단 축소 artifact |
| 7 | `test_trend_regime_v2.py` | realistic_hit + breadth (10/10) + 4-mode → **현재 결론** |

---

## 다음 단계 — Kane 결정 사항

- (A) 모멘텀 영역 동일 분석
- (B) 추세 영역 백테스트 (이 spec 으로 실제 거래 시뮬레이션)
- (C) 임계값/lookback 튜닝 (HIGH_THR, LOW_THR, breadth_trend_lookback 그리드 서치)

이 문서는 (A) → (B) 또는 (C) 진행 시 참조용.
