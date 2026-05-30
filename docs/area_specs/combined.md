# 5영역 결합 시스템 — 종합 spec

작성: 2026-05-30 · Kane & 클로이 (M4 분석)
출처: 5영역 개별 분석 + scripts/backtest_combined.py

---

## 한 줄 결론

**Wyckoff 국면 게이트(Markdown 매수 제외) + 4영역 robust 가중 결합
(T20/M20/Vu0/Va60) + threshold 6.0. 베이스라인(구버전 +44%) 대비
거래당 평균수익 +85%, 안정성(상위5제외) +51%. in-sample 강세장 기준.**

---

## ① 종목 점수 산출 (단일 진입점)

`magic_formula/analysis/area_scores.py::compute_combined_score()`

```
1. 레짐 판별 (make_regimes):
   - regime_breadth (추세용): breadth lookback=10, horizon=10, HIGH=0.60
   - regime_quick (거래량/변동성용): breadth lookback=3, horizon=5, HIGH=0.52

2. 4영역 점수 (각 ±10):
   - 추세    score_trend(df, regime_breadth)   : Dv2 + invert_dist_off_bull
   - 모멘텀  score_momentum(df)                : RSI 10/90 극단 trend
   - 거래량  score_volume(df, regime_quick)    : bear-only Q2+Q3+OBV_contra
   - 변동성  score_volatility(df, regime_quick): BB×52주×레짐 결합표

3. 가중 결합 (COMBINED_WEIGHTS):
   종합점수 = (0.2·추세 + 0.2·모멘텀 + 0.0·거래량 + 0.6·변동성), ±10 클립

4. Wyckoff 게이트 (gate=True):
   Wyckoff_Label 이 Markdown 이면 종합점수 = NaN (매수 제외)
```

### 확정 파라미터
| 항목 | 값 | 비고 |
|---|---|---|
| 가중치 (T/M/Vu/Va) | **0.2 / 0.2 / 0.0 / 0.6** | robust 그리드 최적 |
| threshold | **6.0** | 확정 (5.0 후보) |
| 게이트 제외 국면 | Markdown | |

---

## ② 매매 규칙 (backtest_combined.py)

```
진입: 종합점수 threshold(6.0) 상향 돌파 (prev ≤ 6.0 < today)
      → 다음날 시가 매수, 종목당 1000만원 상한
      (보유 중이면 추가 신호 무시)

청산: (1) 종가 < 진입가 − ATR(14)×1 → 다음날 시가 손절
      (2) 보유 20거래일 + 누적손익 ≤ 0% → 다음날 시가 청산
      (3) 누적손익 > 0% → 보유 유지 (시간청산 안 함)
      (4) 평가종료일 강제청산
```

---

## ③ 백테스트 성능 (2025-06-01 ~ 2026-05-29, 67종목, 신호당 1000만원)

### 확정안 (robust T20/Va60, THR 6.0)
| 지표 | 값 |
|---|---|
| 거래수 | 73 |
| 거래당 평균수익 | **+85.2%** |
| 총 손익 | +6.22억 (투자 7.3억) |
| 상위5 제외 (robust) | +51.2% |
| hit_rate | 54.8% |
| 평균 보유 | 8일 |
| 최고/최저 | +832% / -9.2% |
| 청산 분포 | 강제청산 38 / 손절 19 / 시간손절 16 |

### 비교 (전략별)
| 전략 | 거래 | 평균수익 | 상위5제외 | hit |
|---|---|---|---|---|
| 베이스라인 (구버전 scorer) | 183 | +44.2% | +26.5% | 31.7% |
| 옵션1 (T70/M10/Vu10/Va10, THR5) | 178 | +55.4% | +35.3% | 35.4% |
| **확정 (robust T20/Va60, THR6)** | 73 | **+85.2%** | +51.2% | 54.8% |
| robust THR 5.0 (후보) | 122 | +76.5% | +55.2% | 48.4% |

---

## ④ 핵심 발견 (결합 단계)

1. **Wyckoff 는 점수 합산이 아닌 "국면 게이트"로 써야 가치** — 점수 합산(구버전)
   대비 게이트(Markdown 제외)가 +12%p. 단 구버전엔 Wyckoff 가 이미 점수에
   녹아있어 외부 게이트 추가는 무효 (이중 계산).
2. **영역들이 합의 안 함 (상관 -0.5 ~ +0.3)** — 추세↔변동성 음의 상관.
   같은 시점에 추세는 신고가주(+), 변동성은 저점주(+) 가리킴.
3. **robust 최적은 변동성 비중 高 (Va60)** — 추세 집중(T70)은 소수 대박 의존
   (상위5 제외 시 베이스라인 미달). 변동성이 진입 종목을 분산해 안정성 확보.
4. **거래량 Vu=0** — bear-only 라 강세장엔 거의 0. 결합 점수엔 실질 3영역만.
   하락장 길어지면 살아날 수 있음.
5. **전환신호 매매 액션은 역효과** — DIST/PANIC 강제매도가 대박 종목 조기 청산.
   Wyckoff 는 게이트로만.
6. **threshold 5.0 vs 6.0** — 6.0 평균수익/승률 높음, 5.0 총손익액/대박수 많음.
   자본 무제한 가정에선 5.0 총액 우위, 평균수익/효율은 6.0. → 6.0 확정.

---

## ⑤ 한계 / 미해결

1. **in-sample only** — 같은 1년(강세장)으로 가중치/threshold 최적화.
   out-of-sample 검증은 한국 시장 특수성(1년 3000→8000)으로 무의미 → 생략.
   대안: 종목 universe 확장으로 robust 성 보강.
2. **거래비용 미반영** — 슬리피지/수수료/세금 빼면 수익 감소 (특히 손절 거래).
3. **자본 제약 미반영** — 신호당 무한 1000만원. 동시보유 한도 넣으면 5.0/6.0 재평가.
4. **강세장 의존** — +85%가 시장(KOSPI 2배) 대비 알파인지 미분리.
5. **승률 55%, 대박 의존** — 추세추종 분포. 폭등주(+832%) 빠지면 수익 급감.

---

## ⑥ 코드 자산

| 파일 | 역할 |
|---|---|
| `magic_formula/analysis/area_scores.py` | 4영역 점수 + 레짐 + 종합점수 단일 진입점 |
| `magic_formula/analysis/{trend,momentum,volume,volatility}_variants.py` | 영역별 점수 함수 |
| `magic_formula/analysis/ic_framework.py` | breadth/평가 프레임 |
| `scripts/backtest_combined.py` | 결합 백테스트 엔진 |
| hillstorm `wyckoff_analysis` | 국면 분류 + 전환신호 (외부 의존) |

---

## 다음 단계

- Mac.Mini 데일리 트랙 포팅 (docs/HANDOFF_PORTING.md)
- 종목 universe 확장 (robust 성 보강)
