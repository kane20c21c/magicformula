# Area 3 — 거래량 (Volume) 영역 분석 결론

작성: 2026-05-30 · Kane & 클로이 (M4 분석)
출처: `scripts/test_volume_*.py` 시리즈 (5개 분석)

---

## 한 줄 결론

**거래량 영역은 "하락/조정장 방어 신호" 로 특화. quickregime(빠른 레짐)으로
강세장은 0(다른 영역 위임) / 하락·조정장은 Q2+Q3+OBV_contra (+8.2pp).
거래량 단독은 무의미 — 반드시 가격 방향과 결합해야 신호.**

---

## ① 점수 산출 spec — bear-only 레짐 적응 (±10 풀스케일)

| 항목 | 값 |
|---|---|
| 신호 구조 | quickregime 레짐 적응, bear-only (강세장 off) |
| 강세장 (강세지속/강세약화) | **0** (베팅 안 함 — 거래량 신호 무력, 다른 영역 위임) |
| 하락·조정장 | (Q2 + Q3 + OBV_contra) / 3, ±10 클립 |
| 가격 변화 윈도우 (PC) | 5 거래일 |
| 거래량 임계 | rel_vol HIGH=1.5 / LOW=0.7 |

### 하락·조정장 bear 신호 3 요소 (평균)

```
Q2 (가격↑ 거래량↓): ret_5>0 AND rel_vol<0.7 → -10   # 관심 식은 상승 = 하락 전조
Q3 (가격↓ 거래량↑): ret_5<0 AND rel_vol>1.5 → -10   # 투매
OBV_contra:         OBV 5일기울기 z-tanh × (-1) × 10  # OBV 식은 종목 반등 기대 → 하락 베팅 반대

bear_score = clip((Q2 + Q3 + OBV_contra) / 3, -10, +10)
```

---

## ② quickregime — 거래량 전용 레짐 판별기 (★ 추세/모멘텀과 다름)

거래량은 **추세 영역보다 빠른 레짐 감지**가 필요 (강세장 오분류 최소화).
별도 이름 `quickregime` 으로 구분.

| 파라미터 | quickregime (거래량) | 추세 영역 breadth |
|---|---|---|
| breadth lookback | **3** | 10 |
| breadth horizon | **5** | 10 |
| HIGH_THR | **0.52** | 0.60 |
| LOW_THR | 0.40 | 0.40 |
| trend_lookback | 5 | 5 |

```
quickregime(t):
  breadth(t) = 지난 3거래일 × 67종목의 5일 backward 수익률 양수 비율
  trend(t)   = breadth(t) - breadth(t-5)
  if breadth > 0.52:  강세지속(trend≥0) / 강세약화(trend<0)  → bear 신호 OFF
  elif breadth < 0.40: 하락                                  → bear 신호 ON
  else:                조정                                  → bear 신호 ON
```

**왜 빠른가**: lookback=3 으로 강세장 진입을 빨리 감지해 강세장 오분류
(→ bear 신호 새어듦) 최소화. lookback=5/10 은 강세장 손실 더 큼 (-3.3~-9.3).

---

## ③ 검증 성능 (2025-05-28 ~ 2026-05-28, h=5)

| 변형 | realistic_hit | vs bench | coverage |
|---|---|---|---|
| **bear-only (lookback=3, HIGH 0.52)** | 0.546 | **-1.8pp** | 0.36 |
| regime_full (bull 활성) | 0.497 | -6.7pp | 1.00 |
| Q1_static (강세 항상) | 0.568 | +0.3pp | 0.13 |
| vol_baseline (기존 모델) | 0.519 | -4.5pp | 1.00 |

전체 풀 bench (h=5): 0.565

### 시기별 — bear-only 의 진짜 가치

| 월 | bench | bear-only vs bench |
|---|---|---|
| 2026-04 (강세장) | 0.721 | -1.4 (거의 손실 없음) |
| **2026-05 (하락장)** | 0.408 | **+8.2** |
| 2026-03 (조정/하락) | 0.433 | +8~10 추정 |

→ 강세장 손실 거의 0, 하락/조정장 +8pp. "하락장 방어" 특화.

---

## ④ 핵심 발견 (분석 series)

1. **거래량 단독 무의미** — rel_vol/OBV/AD_Line/Chaikin 모두 IC ~0 (h=3 -3.6pp).
   상대거래량은 방향이 없음 (높은 거래량 = 매수일수도 매도일수도).

2. **거래량 + 가격 결합 시 신호** (Kane 로직 검증):
   "관심↑(거래량↑) + 가격↑ → 상승 지속" = Q1 (+0.3pp, h=5).

3. **레짐별 신호 방향 정반대** (월별 분석):
   - 강세장: Q1 (거래량 받친 상승) 약하게 유효
   - 하락장: Q2 (거래량 식은 상승 = 하락 전조) 강력 (hit 0.79, 2026-05)

4. **역명제 Q2 는 하락장 한정** — Kane "거래량↓→하락" 가설은 강세장에선
   틀리지만(상승 관성) 하락/조정장에선 맞음.

5. **OBV 도 레짐 의존** — 기존 모델의 OBV 상시 trend 사용이 강세장 -10~-20pp
   역효과였음. 하락장에선 contra (OBV↓→반등) 가 정답.

6. **강세장엔 거래량 알파 없음** — bench 0.72 라 어떤 거래량 선별도 손해.
   bull off 가 정답 (강세장은 추세/모멘텀에 위임).

---

## ⚠️ 한계

1. **전체 -1.8pp** — 강세장 기간(12개월 중 ~50%)에 0 베팅이 bench(상승) 대비
   중립~손해. 거래량의 가치는 순수 하락/조정장 한정.
2. **하락장 표본 적음** — 2026-05 (n 적음), 2026-03 정도. turning point 2~3건.
3. **quickregime breadth lag** — lookback=3 도 강세장 일부 오분류 (-1.4 잔여 손실).
4. **실거래 미연계** — entry/exit/cost 시뮬레이션 필요.

---

## 분석 series

| 스크립트 | 핵심 |
|---|---|
| `test_volume_per_indicator.py` | 거래량 단독 5지표 → 모두 IC~0, bench 미달 |
| `test_volume_price_combo.py` | 가격 결합 (Q1/Q2) → Kane 로직 검증, 선택적 베팅 |
| `test_volume_divergence.py` | 2×2 분면 → Q2 역명제 강세장 틀림 |
| `test_volume_monthly.py` | 월별 → Q1/Q2 레짐별 정반대 발견 |
| `test_volume_regime.py` | 레짐 적응 통합 → bull off 필요 발견 |
| `test_volume_breadth_lag*.py` | breadth lag 그리드 → quickregime(lookback=3) 확정 |

---

## 다음 단계

- 변동성·위치 (Area 4) 분석 — BB %B, 52주 위치
- 5영역 결합 단계에서 거래량 bear-only 를 하락장 방어로 활용
