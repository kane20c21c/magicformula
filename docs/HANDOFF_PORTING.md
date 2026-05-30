# Mac.Mini 데일리 트랙 포팅 계획 — 5영역 결합 시스템 (v2)

작성: 2026-05-30 · Kane & 클로이 (M4 분석)
대상: Mac.Mini (Intel i5, 데일리 운영 트랙)
목적: M4에서 확정한 신 계산방법(v2_combined)을 데일리 시그널 생성에 반영

---

## 0. 왜 단순 yaml 업데이트로 안 되나

기존 계획은 `active_strategy.yaml` 의 가중치 한 줄 바꿔 git 으로 전달이었음.
**그러나 v2 는 가중치뿐 아니라 계산방법 전체가 교체됨:**

| | 구버전 (현 데일리 운영) | v2 (신 시스템) |
|---|---|---|
| 점수 함수 | `scorer.compute_scores` (5영역 가중평균) | `area_scores.compute_combined_score` |
| 영역 점수 | 고정 공식 (±5.33 등) | Dv2 / RSI극단 / bear-only / BB×52주 결합 |
| 레짐 | 없음 | 영역별 2종 (breadth / quickregime) |
| Wyckoff | 점수 20% 합산 | 국면 게이트 (Markdown 제외) |
| 진입 | R1 (composite threshold 돌파) | 종합점수 threshold 6.0 돌파 |
| 의존성 | OHLCV parquet | + hillstorm, full-column parquet |

→ **runner.py 의 점수/신호 로직을 통째로 분기 또는 교체해야 함.**

---

## 1. 포팅 전제 — Mac.Mini 환경 점검

### 1-1. 새 코드가 요구하는 의존성 (Mac.Mini에 없을 수 있음)

| 의존성 | 용도 | Mac.Mini 상태 확인 |
|---|---|---|
| **hillstorm** (`kane20c21c/hillstorm`) | Wyckoff 국면 분류 + 전환신호 | `~/DriveForALL/StoLab/hillstorm` 존재? |
| **plotly** | hillstorm wyckoff_analysis import 체인 | `pip show plotly` |
| **scipy** | (분석용, 데일리엔 불필요할 수도) | 확인 |
| full-column parquet | 거래량/BB/심리 21지표 | vault core.parquet 가 이미 포함 (collector 만 자름) |

> ⚠️ **Intel arm64 차이 없음** — 순수 파이썬/pandas/numpy. Apple Silicon 전용
> 코드 없음. plotly/hillstorm 도 Intel 에서 동작. 단 **설치만** 필요.

### 1-2. 데이터 — collector 가 컬럼을 자르는 문제

`magic_formula/data/collector.py` 가 OHLCV 5컬럼만 남김 (M4 분석 중 발견).
v2 는 AD_Line/Chaikin/Rel_Volume/Hope_Vector 등 필요.
→ 데일리도 **vault data_service.get_ohlcv 직접 호출** (full-column) 로 바꿔야 함.
   (분석 트랙은 이미 우회했음. 데일리도 동일 처리.)

---

## 2. 포팅 범위 — 코드 변경 지점

### 2-1. 그대로 가져갈 것 (git pull 로 동기화됨, Intel 무관)

- `magic_formula/analysis/area_scores.py` (단일 진입점)
- `magic_formula/analysis/{trend,momentum,volume,volatility}_variants.py`
- `magic_formula/analysis/ic_framework.py` (compute_breadth_series 만 필요)
- `magic_formula/scoring/scorer.py` (구 함수도 유지 — 베이스라인 폴백용)

### 2-2. 새로 손봐야 할 것 — daily/runner.py

현재 runner 흐름 (구버전):
```
get_ohlcv(ticker) → compute_scores(df, weights) → composite_score
                  → check_r1_signal(scored, threshold)
```

v2 흐름으로 교체 (★ 핵심 작업):
```
1. 전 종목 full-column OHLCV 로드 (data_service 직접)
2. hillstorm 국면 분류 (종목별 Wyckoff_Label)   ← 신규
3. make_regimes(stock_data) → breadth + quickregime  ← 신규
4. 각 종목: compute_combined_score(df, rg_b, rg_q, phase, weights, gate)
5. 신호: 종합점수 prev <= 6.0 < today (threshold 돌파)
6. 저장: daily_signal_{date}.{json,md}  (포맷 유지)
```

**구현 방식 — system_version 분기 권장:**
```python
cfg = load_strategy()  # active_strategy.yaml 또는 _v2.yaml
if cfg.system_version == "v2_combined":
    scores = run_combined(...)   # 신 경로
else:
    scores = run_legacy(...)     # 구 경로 (현 R1)
```
→ 한 파일에 두 경로 공존. yaml 한 줄(`system_version`)로 전환. 롤백 안전.

### 2-3. 레짐 계산 비용 — 데일리 성능

`make_regimes` 와 hillstorm 분류는 **전 종목을 한 번에** 봐야 함 (breadth =
횡단면). 데일리는 매일 1회라 부담 적음. 단:
- hillstorm 분류가 종목당 ~0.1초 → 67종목 ~7초. Intel 은 2~3배 → ~20초. OK.
- breadth 는 전 종목 시계열 필요 → 데일리도 전 종목 로드 (이미 함).

---

## 3. 포팅 절차 (단계별)

### STEP 1 — Mac.Mini 의존성 설치
```bash
# Mac.Mini
cd ~/DriveForALL/StoLab
# hillstorm 없으면
gh repo clone kane20c21c/hillstorm   # 또는 git clone
cd "Magic Formula" && source .venv/bin/activate   # (데일리용 venv)
pip install plotly                                 # hillstorm import 체인
# 검증
python -c "import sys; sys.path.insert(0,'../hillstorm'); from wyckoff_analysis import classify_wyckoff; print('OK')"
```

### STEP 2 — 코드 동기화
```bash
cd ~/DriveForALL/StoLab/"Magic Formula"
git pull   # area_scores.py + variants + runner 변경 받기
```
> ⚠️ Mac.Mini 는 **구 레포(kane20c21c/magicformula)** 를 쓰고 있을 수 있음.
>    M4 는 신 레포(kane20c21c/mf.in.air). **레포 정리 먼저** (4번 참조).

### STEP 3 — runner.py v2 분기 구현 (M4에서 작성 → push → Mac.Mini pull)
- `daily/runner.py` 에 `run_combined()` 추가 + system_version 분기
- 데이터 로딩을 data_service 직접 호출로 (full-column)

### STEP 4 — Mac.Mini 스모크 테스트
```bash
python scripts/daily_signal.py   # v2 yaml 로
# → output/signals/daily_signal_*.md 에 새 점수/신호 뜨는지 확인
# → 에러 없이 67종목 처리, 종합점수 + 게이트 반영 확인
```

### STEP 5 — yaml 전환 + launchd 유지
```bash
# active_strategy.yaml 을 v2 로 교체 (또는 system_version 만 추가)
cp configs/active_strategy_v2.yaml configs/active_strategy.yaml
git add configs/ && git commit -m "데일리 v2 전환" && git push
# launchd 는 그대로 (평일 20:40)
```

---

## 4. ★ 선결 과제 — 레포 일원화

현재 상태:
- M4 분석: `kane20c21c/mf.in.air` (신 레포, 모든 v2 코드)
- Mac.Mini 데일리: `kane20c21c/magicformula` (구 레포)일 가능성

**두 머신이 다른 레포면 git pull 로 동기화 불가.** 선택:

| 옵션 | 방법 | 트레이드오프 |
|---|---|---|
| A | Mac.Mini 도 mf.in.air 로 전환 | 깔끔. 단 Mac.Mini 작업이력 정리 필요 |
| B | mf.in.air → magicformula 로 머지 | 구 레포 유지. 히스토리 복잡 |
| C | v2 코드만 magicformula 에 cherry-pick | 수동, 누락 위험 |

> 클로이 추천: **A (Mac.Mini 를 mf.in.air 로)**. M4 가 이미 모든 코드의
> single source. Mac.Mini 는 데일리 실행만 하므로 레포 교체 부담 적음.
> 단 Mac.Mini 의 launchd plist 경로(폴더명 "Magic Formula" 공백 포함) 와
> M4 폴더명("MagicFormula" 공백 없음) 차이 주의 — 심볼릭 링크 또는 경로 통일.

---

## 5. 검증 체크리스트 (포팅 완료 기준)

- [ ] Mac.Mini 에 hillstorm + plotly 설치, import OK
- [ ] git pull 로 v2 코드 동기화 (레포 일원화 완료)
- [ ] runner.py v2 분기 동작 (system_version 으로 전환)
- [ ] daily_signal.py 실행 → 에러 없이 67종목, 종합점수 + Markdown 게이트 반영
- [ ] 신호 종목이 M4 backtest_combined 와 같은 날 같은 종목인지 spot-check
- [ ] launchd 평일 20:40 자동 실행 유지
- [ ] 첫 데일리 시그널 md 헤더에 strategy_id=COMBINED-v2 확인

---

## 6. 롤백 계획

문제 발생 시:
```bash
# active_strategy.yaml 의 system_version 을 legacy 로 (또는 구 yaml 복원)
git checkout configs/active_strategy.yaml   # 구버전 복원
python scripts/daily_signal.py              # R1 로 즉시 복귀
```
→ runner.py 가 분기 구조면 yaml 한 줄로 즉시 롤백. 구 함수(scorer.compute_scores)
   는 그대로 유지하므로 안전.

---

## 7. 미해결 / 추후

- 거래비용(슬리피지/수수료/세금) 데일리 반영 여부 (백테스트엔 미반영)
- 동시보유 한도 — 데일리 신호는 후보 나열, 실제 매수는 Kane 판단
- 종목 universe 확장 (robust 성 보강) — 별도 작업
- v2 의 실거래 검증 (현재 in-sample 백테스트만)
