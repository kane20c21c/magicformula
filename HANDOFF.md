# HANDOFF — 황금률만들기 (MagicFormula)

> 작성 2026-08-25 (KST) · 작성자 클로이
> 리포 위치 `~/DriveForALL/StoLab/MagicFormula`

**근거 태그** — 모든 항목에 출처를 붙였다.

| 태그 | 뜻 |
|---|---|
| `[문서]` | 리포 안 확정 문서·설정 파일에 적혀 있음 |
| `[실측]` | 2026-08-25 파일·git 상태를 직접 확인함 |
| `[세션]` | 과거 Cowork 대화 기록에서 확인 (문서화 안 됐을 수 있음) |
| `[의견]` | 클로이 판단 — 근거 자료가 아님 |

⚠ **이 문서는 정본이 아니다.** 각 룰 폴더의 스펙 파일이 정본이고, 여기는
"지금 어디까지 왔고 무엇이 남았나"의 스냅샷이다. 값이 상충하면 정본이 이긴다.

---

## 1. 프로젝트 목적

MagicFormula(황금률만들기)는 Kane의 한국 주식 투자 시스템에서 **매매 룰의 판단
로직을 소유하는 리포**다. "언제 무엇을 살 것인가"를 계산해 신호를 뱉는 일만 하고,
데이터 수집은 longlivevault가, 계좌·체결·서빙은 StockPortfolio가, 화면은
homalone(8501)이 맡는다 — 이 분업은 바꾸지 않는다는 것이 전제다. 현재 성격이 서로
다른 룰 4개(황금률·데이 포트·스윙 포트·발열률)를 한 지붕 아래 두되, **룰끼리
파라미터와 어휘를 섞지 않는 것**을 원칙으로 운영한다. 각 룰은 매일 정해진 시각에
launchd 잡으로 자동 산출되고, 결과는 메일·푸시·웹 화면으로 나간다. `[문서]`

---

## 2. 현재 상태

### 2.1 룰 4개 요약 `[문서]`

| # | 룰 | 버전 · 확정일 | 스펙 정본 | 성격 |
|---|---|---|---|---|
| 1 | **황금률** | `COMBINED-v2-2026-05` (2026-05-31 승격) | `configs/active_strategy.yaml` | 진입 신호 (종목 점수) |
| 2 | **데이 포트** | **v1.1.2.2** (2026-08-24~) | `~/DriveForALL/StoLab/StockPortfolio/app/paper_day/config.py` | 1일 스윙 (ML) |
| 3 | **스윙 포트** | **v1.2.2.3** (2026-08-17) | `korean_mkt_study/STRATEGY.md` + `strategy_spec.json` | 눌림목 진입 |
| 4 | **발열률** | 2026-08-14 구축 | `fever_model/CLAUDE.md` | 시장온도 + 관측 명단 |

버전 체계 = **모델.유니버스.진입.청산** (각 자리 최초 1, 조정마다 +1).
Kane 도입 2026-08-23, 스윙·데이 공통. `[문서]`

**1. 황금률** — 4영역 robust 가중 결합 **추세 20 / 모멘텀 20 / 거래량 0 / 변동성 60**
+ Wyckoff 게이트(Markdown 국면 매수 제외). 진입 = 종합점수 **6.0 상향돌파**,
체결 다음날 시가. 청산 = 진입가 −ATR(14)×1 손절 / 20거래일 + 누적손익 ≤0% 시간청산.
종목당 상한 10,000,000원, 후보 임계 5.0. `[문서]`
⚠ **점수 자체의 예측력이 2026-08-24 검증에서 부정됐다** — §3.6 · §5.1 참조.

**2. 데이 포트** — 16:20 신호 → **17:00 NXT 애프터마켓 10슬롯 × 2,000,000원** 매수
→ 청산 = 당일고점 −1% 트레일링 + 첫 관측 갭 관문 −5%. 충원 컷은 `p ≥ 0.925`
**및** `rank ≤ 15`, 못 채우면 비워둔다. `[문서]` `[실측]`
10슬롯 전환은 **`SP/data/paper_day/config.json` 오버라이드로만** 적용됐고 코드
기본값은 v1.1.1.x 원형(5슬롯 × 4,000,000원)을 유지한다. `[실측]` 확인 완료

**3. 스윙 포트** — 유니버스 월말 시총 **4조↑ & 외국인 30%↑**(AND),
진입 `close > SMA200` **및** `close ≤ 0.90 × ROLLING_MAX(close, 40)` onset만,
t일 종가 신호 → t+1일 시가 매수. 슬롯 20 × 5,000,000원(백테스트) /
2,000,000원(가상계좌). 청산은 **시간연동 4분기 × 종목 변동배율**. `[문서]`

```
배율 = 종목 YZ_20(신호일) ÷ 직전 252거래일 스윙풀 YZ_20 중앙값의 중앙값
       (LLV data/vol_scale.json, 클립 0.2~5.0, 미가용 시 1.0)
D+0~5           참조가 × (1 − clip(0.20 × 배율, 0.10, 0.40))
D+6~ 피크>평단   피크   × (1 − clip(0.05 × 배율, 0.03, 0.15))
D+6~ 피크≤평단   평단   × (1 − clip(0.10 × 배율, 0.05, 0.25))
```

**4. 발열률** — 1단 시장온도 나우캐스트 → 2단 Vblk 탄력점수
(추세 33 : 눌림진폭 17 : 눌림거래량 17 : YZ 33). 관측 지평 20~60거래일. `[문서]`
⚠ **매수 신호가 아니라 관측 명단**이다 — 진입·청산 규칙이 없다.

⚠ **1단과 2단은 창이 다르다** `[실측]` (2026-08-25 코드 추적으로 확정). 실효값은
`fever_model/src/daily_WW_wf.py` L65~69 상수가 전부다 — 이 파일은 `nowcast_grid.py`·
`resilience_v2.py` 를 **import 하지 않고 로직만 이식**했다.

| 단 | 상수 | 값 |
|---|---|---|
| 1단 온도 (채택안) | `W_ADOPT` / `MINW_ADOPT` | **60 / 4** |
| 1단 온도 (비교용 구기준) | `W_PREV` / `MINW_PREV` | 100 / 6 |
| 1단 특징 계산 게이트 | `GATE` | 4 (MINW 는 사후 마스킹) |
| **2단 탄력점수(Vblk)** | `RES_W` / `RES_MINW` / `RES_RECENT` | **100 / 6 / 8** (완화 미적용) |

`resilience_*.py` 의 `MIN_WAVES = 6` 은 **2단 로직 출처**라 모순이 아니고,
`nowcast_grid.py` 의 6 은 1단 구기준이다. 정본에도 명문화해 뒀다.

### 2.2 동작 중인 자동 잡 (MagicFormula 소유 7개) `[실측]`

| 시각 | 반복 | 레이블 | 내용 |
|---|---|---|---|
| 15:10 | 매일 | `com.kane.magicformula-mop-gauge` | 섹터 오버나이트 게이지 (`run_gauge.py`) |
| 16:20 | 매일 | `com.kane.magicformula-mop-signal` | 데이 포트 익일 신호 (`run_daily.py --rebuild`) |
| 16:30 / 20:40 | 평일 | `com.stolab.magic-formula.daily-signal` | 황금률 점수 (1차 / 수급 반영 2차) |
| 16:35 / 20:45 | 평일 | `com.stolab.magic-formula.daily-extended-signal` | 황금률 확장 시그널 |
| 16:30 | 평일 | `com.kane.fever-rule-daily` | 발열률 워크포워드 |
| 17:00 | 평일 | `com.kane.fever-rule-mail` | 발열률 메일 |
| 20:35 | 평일 | `com.kane.kms-shadow-track` | 스윙 포트 그림자 추적 (`shadow_track.py`) |

plist 정본은 `configs/launchd/`, 운영본은 `~/Library/LaunchAgents/` 심링크.
**단 발열률 2개는 심링크가 아니라 복사본** — 수정하면 재복사해야 한다.
`[실측]` 2026-08-25 확인: 7개 전부 launchctl 등재 + 최근 종료코드 0, 발열률 2개만
복사본(`-rw-`)이고 나머지 5개는 심링크가 맞다.

✅ `[실측]` 종전 "`~/DriveForALL/StoLab/StockPortfolio/SCHEDULED_TASKS.md` 에
`kms-shadow-track` 누락" 은 **해소**. 같은 점검에서 `longlivevault-{foreign-0811,
volscale-2050, derivatives-merge}` 3건도 빠져 있어 함께 채웠고, 등재 레이블을
`launchctl list` 와 대조해 **38개**로 맞췄다 — 현재 누락 0.

### 2.3 최근 산출 확인 `[실측]` (2026-08-25 기준)

| 산출물 | 최신 | 상태 |
|---|---|---|
| `mop_model/output/signals/signal_latest.json` | 2026-08-24 | 유니버스 **201종목**, `train_last_date=2026-08-21` |
| `mop_model/output/gauge/gauge_latest.json` | 2026-08-24 | 10세트, `snapshot_missing []` — 결손 0 |
| `output/signals/daily_signal_20260824.*` | 08-24 20:40 | 정상 |
| `output/signals/daily_extended_signal_20260824.*` | 08-24 20:45 | 정상 |
| `korean_mkt_study/data/shadow_v1112_state.json` | `last_processed=2026-08-24` | 보유 16종목 |
| `SP/data/paper_day/shadow/{R1,R2,R3}.json` | `last_processed=20260824` | R1 v1.1.1.2 / R2 v1.1.3.2 / R3 v1.1.2.1 |

### 2.4 관찰이 돌아가고 있는 것 `[문서]` `[세션]`

- **데이 포트 그림자 3규칙** — 본계정 v1.1.2.2 전환(08-24)의 비교군. 평일 20:55
  `SP/scripts/day_shadow_track.py`
- **스윙 포트 그림자 v1.1.1.2** — 직전 운영 모델(4조/25% · 60일/MA120)을 다음 달
  모델 평가까지 추적. 평일 20:35
- **발열률 8/7 빈티지 채점** — 20거래일(9월 4일경) · 60거래일(11월 초) 예정

---

## 3. 주요 결정사항과 그 이유

### 3.1 스윙 청산 — 변동배율의 분모는 '직전 1년', '그날 시장'이 아니다 (2026-08-16) `[문서]`

배율 = 종목 YZ_20 ÷ **직전 252거래일** 스윙풀 YZ_20 중앙값의 중앙값.

그날 시장 중앙값(상대 척도)으로 나누면 시장 전체가 동시에 요동칠 때 분모도 같이
커져 배율이 1.0으로 눌린다 → 평소와 같은 손절폭으로 폭락을 맞아 **포트 전체가
바닥에서 동시에 털린다.** 백테스트 2015-01~2026-06 실측 — 2020~2021 코로나 구간에서
상대 척도 **+0.2%** vs 절대 척도 **+21.2%**. 전 구간 Sharpe 0.65 → 0.79.

⚠ 이 문단을 지우지 말 것. 재설계 제안이 들어올 때마다 되돌아오는 지점이다.

### 3.2 스윙 진입 — 트리거를 풀면 재앙, 조정만 유효 (2026-08-17) `[문서]`

눌림 후 반등 기회의 60%가 'MA120 아래'에서 발생하지만, 추세필터를 제거·완화하면
MDD −68~−76%로 무너진다. **놓친 기회의 대부분은 사후적 기회일 뿐**이라는 것이 결론.
견고했던 개선은 두 가지뿐 — 추세선 연장(MA120 → **MA200**)과 창 단축(60일 → **40일**).
40일 창은 저변동 박스권에서 한물간 고점 기준의 저질 신호를 걸러 평시 손실을 1/3로
줄였다(−1,097만 → −380만원).
⚠ MA 곡선·창 곡선 모두 **비단조** — 40·200이라는 정확한 값에 과신 금물.

### 3.3 스윙 유니버스 — 외인 필터는 수익 필터가 아니라 위기 방어 필터 (2026-08-17) `[문서]`

시총·외인 어느 쪽이든 완화하면 네 방향 모두 단조 악화. 외인 완화 시 코로나 구간
손익이 +2,774만 → +201만원으로 붕괴한다. 5조/30%가 방어 최우수(Sharpe 0.73 /
MDD −16.8%)였으나 Kane이 **자본 활용률**(투자비중 26% → 30%)을 고려해 **4조/30%**
채택.

### 3.4 데이 포트 — 변동배율 이식 기각 (2026-08-17) `[문서]`

스윙의 손절폭 변동배율을 데이에 옮기려다 폐기했다. 배율의 물리적 근거는 확인됐지만
(YZ_20 ↔ 장중 하락폭 Spearman |0.956|, 위험 정규화 2.87 → 1.17) **수익이 따라오지
않는다.** `손절폭 = base × 배율^지수` 의 **최적 지수가 ~0**.

원인은 구조적이다 — 데이 손절폭 1%는 **장중 노이즈의 1/4**(고점→저가 중앙 −4.4%)이라
저변동 종목조차 도달률 95.7%다. 배율로 0.7~2.5%를 흔들어도 여전히 노이즈 안이라
결과가 안 바뀐다. 스윙은 변동성이 손절 도달확률을 지배하지만, **데이는 추세가
지배하고 변동성은 부차적**이다.

### 3.5 데이 포트 — 10슬롯 × 2,000,000원 전환 (2026-08-23) `[문서]` `[세션]`

첫 4주 검증에서 rank 6~15 구간의 우위 + 분산 효과가 확인돼 5×400만 → **10×200만**.
청산 −1%는 **유지** — 손절 임계 1.5%·2.0% 모두 비유의였고 스윕이 비단조였다
(08-17 변동배율 기각과 정합). 프리미엄 게이트·발열 게이트는 게이트 임계 비단조 /
발열의 익일 예측력 0으로 **보류**.
⚠ 오버라이드로만 적용 — 코드 기본값을 고치지 않은 것은 원형 보존이 목적. `[문서]`

### 3.6 황금률 — 종합점수의 예측력이 부정됨 (2026-08-24) `[세션]`

4개 연도 **모두 음수**. 원인은 과최적화가 아니라 **변동성 점수표의 52주 방향이
역전된 평가 기준 오류**(in-sample에서도 틀림). 부수 확인 —

- Wyckoff 게이트는 무력
- 현행 `time_stop`(20거래일 + 손익≤0)이 검증한 청산안 중 **최악**
  (CAGR 94.3%, MDD −67.5%)
- **"손절을 넣을 것인가"의 답은 아니오** — 손절·트레일링·조건부 시간청산·
  스윙 v1.2.0 이식 넷 다 단순 20거래일 청산보다 열등
- 제안된 대안 구조(미채택): `RSI(14) ≥ 70` 문 → `min(52주 위치, YZ 배율) ≥ 0.95`,
  청산 20거래일 단순, 슬롯 10~12

⚠ **CAGR 165%는 신뢰 보류** — 생존편향 · 강세장 · 무제한 복리가 겹쳐 있다.

### 3.7 섹터 게이지 — 고정가중 전환, 정규화 안 함 (2026-08-15) `[문서]`

당일 시가총액 자동가중을 폐지하고 **Kane 고정가중**으로. 세트 합을 **정규화하지
않는 것도 결정 사항** — 엑셀 반올림 탓에 합이 0.99~1.01인 세트가 있고 그 배율이
`weighted_p`에 그대로 실린다(세트 간 비교 시 최대 2% 편차). 실제 합은 JSON
`weight_sum`으로 확인한다. 세트 간 종목 중복도 허용(세트별 독립 집계).

### 3.8 알림 — stop·system 두 앱만 이중화 (2026-08-18) `[세션]`

Pushover + 텔레그램 이중화를 **`stop`(손절·데이청산·섹터게이지)** 과
**`system`(감시자·에러)** 두 앱에만 적용. signal/brief/watch는 제외.
미러 대상은 `.env`의 `TELEGRAM_MIRROR_APPS` 한 줄로 바꾼다.

### 3.9 그림자 추적을 '병행'으로 한 이유 (2026-08-17 · 08-23) `[문서]` `[세션]`

규칙을 바꿀 때 구모델을 즉시 버리지 않고 별도 상태파일로 계속 굴린다. 스윙은
v1.1.1.2, 데이는 R1/R2/R3 세 규칙. **다음 달 모델 평가 때 같은 기간·같은 시장에서
직접 비교하기 위한 것** — 백테스트로는 잡히지 않는 체결 괴리까지 포함해서 본다.

### 3.10 데이룰 실계좌 재알림 — 기준을 손절선이 아니라 직전 알림가로 (2026-08-18) `[세션]`

또쓰 계좌 재알림을 **직전 알림가 대비 −0.3%**로 잡고 날짜가 바뀌면 알림가·회차를
리셋. 손절선 기준으로 두면 갭하락일에 첫 알림이 차단되는 문제가 있었다.
임계는 `SP/data/realstop/config.json`의 `realert_worse_pct`로 코드 수정 없이 조정.

---

## 4. 폴더 구조

```
~/DriveForALL/StoLab/MagicFormula/
├── CLAUDE.md                    ★ 룰 레지스트리 — 작업 시작 전 먼저 읽는 파일
├── HANDOFF.md                   이 문서
├── README.md                    외부 문서 (API·디렉토리 상세)
│
├── configs/                     ── 황금률 설정
│   ├── active_strategy.yaml     ★★ 황금률 운영 정본 (가중치·임계·게이트·매매규칙)
│   ├── active_strategy_v{1,2}.yaml  구버전 백업 (v1은 2026-06-10 완전 폐기)
│   ├── classification.yaml      4셀 분류 설정
│   ├── area_specs/              영역별 신호 spec — trend·momentum·volatility·volume·wyckoff.yaml
│   └── launchd/                 ★ plist 정본 7개 + README(등록 절차)
│
├── magic_formula/               ── 황금률 엔진 (패키지)
│   ├── _vault.py                ★ LLV 진입점 통합 헬퍼 (SSOT — 경로·CORE_TICKERS·섹터맵)
│   ├── config.py · main.py
│   ├── analysis/area_scores.py  ★ 종합점수 계산 정본 (compute_combined_score)
│   ├── analysis/backtest.py · ic_framework.py · *_variants.py   영역별 실험 코드
│   ├── daily/runner.py · report.py    일일 신호 산출·리포트
│   ├── signals/rules.py         진입·청산 규칙
│   ├── simulator/ · optimizer/ · metrics/ · data/collector.py
│   └── indicators.py
│
├── mop_model/                   ── 데이 포트 (ML)
│   ├── CLAUDE.md                ★ 스펙 정본 (모델·피처·기각목록·튜닝 금지)
│   └── src/
│       ├── run_daily.py         ★ 16:20 운영 — panel→features→재학습→신호
│       ├── run_gauge.py         ★ 15:10 섹터 오버나이트 게이지 (인메모리, 산출물 무오염)
│       ├── gauge_config.py      ★ Kane 편집 파일 — 10세트 고정가중 정의
│       ├── gauge_core.py        집계 로직(순수) · gauge_notify.py 통지
│       ├── build_panel.py · build_features.py · model.py
│       └── walkforward.py       검증용 백테스트
│
├── korean_mkt_study/            ── 스윙 포트
│   ├── STRATEGY.md              ★★ 사람용 정본 (§4 청산근거 · §11 기각목록)
│   ├── strategy_spec.json       ★ 기계용 정본 (버전·유니버스·진입·청산 요약)
│   ├── strategy_reference.py    진입 계산 로직 (백테스트와 전량 일치 검증됨)
│   ├── backtest.py              ★ 규칙 실험은 반드시 이 파일로
│   ├── shadow_track.py          평일 20:35 그림자 추적
│   ├── pattern_study.py · sweep_*.py · diag_*.py · verify_entry.py   실험 도구
│   └── data/                    백테스트 원자료 711MB (git 미추적 — 삭제 금물)
│       ├── universe_2026-08_4jo_30pct.json   ★ 현행 유니버스 41종목
│       └── shadow_v1112_state.json           그림자 상태
│
├── fever_model/                 ── 발열률
│   ├── CLAUDE.md                ★ 스펙 정본
│   ├── src/daily_WW_wf.py       ★ 평일 16:30 운영 정본
│   ├── src/send_fever_mail.py   17:00 메일 (계산 없음)
│   ├── src/{nowcast_grid,resilience_v2,resilience_score}.py   로직 출처(실험 원본)
│   ├── data/panel_states.csv    고정 모델 (스케일러+중심점), 191종목
│   └── output/                  온도일지·국면일지·탄력점수일지 + 메일/8501용 산출물
│
├── scripts/                     ── 배치·검증
│   ├── daily_signal.py          ★ 황금률 16:30/20:40
│   ├── daily_extended_signal.py ★ 황금률 확장 16:35/20:45
│   ├── generate_classification.py   4셀 분류 산출
│   ├── day_stop_{backtest,sweep,2x2,minute_fetch}.py   데이 손절 검증 재현(08-17)
│   └── validate_*.py · holdtiming_validation.py        일회성 검증
│
├── output/                      산출물 — signals/ · classification/ · analysis/ · logs/
├── docs/                        architecture.md · backtest_design_v2.md · HANDOFF_{M4,PORTING}.md
└── tests/                       pytest — test_{area_scores,signals,simulator,vault,config,mop_gauge}.py
```

`[의견]` 처음 들어오는 사람이 읽을 순서: `CLAUDE.md`(전체 지도) → 손댈 룰의 스펙
정본 하나 → 그 룰의 기각목록. 코드부터 열면 왜 그 값인지를 모른 채 고치게 된다.

---

## 5. 미완료 작업 / 다음 단계

### 5.1 A — 황금률 재설계 판단 보류 (최우선) `[세션]`

§3.6 결과 이후 Kane이 "이번 세션은 여기까지"로 멈춘 상태. 남은 순서:

1. **point-in-time 유니버스 재검증** ← 1순위. 이게 끝나기 전에는 어떤 수치도 확정 금지
   (`magic_formula/_vault.py`가 참조하는 `core_tickers.py`가 현재 명단이라 생존편향)
2. 2020~2022 약세장 구간 백필 후 재검정
3. 소액 실운영으로 체결 괴리 측정
4. **"손절 불필요" 결론은 약세장 검증 전 확정 금지.** 계좌 전체 방어선은 별도 권고

근거 리포트: `~/DriveForALL/StoLab/StockPortfolio/reports/황금률_전구간_재검증_20260824.html`

### 5.2 B — 기한이 있는 관찰 `[문서]` `[세션]`

| 대상 | 기한 | 상태 |
|---|---|---|
| 데이 포트 그림자 3규칙 비교 | ~2026-09-24 (전환 1개월) | 상태파일 정상. **알림 없음** — Kane 개인 일정에 등록, "시간 되면 내가 이야기할게" |
| ⚠ day-shadow 배치 **수리 후 자동 실행 검증** | 오늘(08-25) 20:55 | `[실측]` 08-24 20:55 자동 실행은 `FileNotFoundError` 로 **실패(exit=1)** 했고 수리 커밋은 그 뒤 **23:37**(`3f9cdf3`). 데이터는 수동 캐치업으로 채워졌지만 **수리 후 자동 실행은 아직 0회** — 첫 성공을 확인해야 한다 |
| 스윙 포트 그림자 v1.1.1.2 | 다음 달 모델 평가 시 | 상태파일 정상 (보유 16종목) |
| 발열률 8/7 빈티지 채점 | **9월 4일경**(20거래일) · 11월 초(60거래일) | 예정만 잡혀 있음. `탄력점수_V2_채점_20260813.xlsx` 방식 재채점 |

### 5.3 C — 잔손질 `[실측]` `[세션]`

**남은 것**

| 항목 | 내용 |
|---|---|
| 섹터게이지 plist `PATH`에 `/usr/sbin` 누락 | `[실측]` **미조치 확정** — `com.kane.magicformula-mop-{gauge,signal}.plist` 의 PATH 가 `/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin`. joblib 코어수 경고만 나고 동작은 무해 |
| 데이룰 손절 재알림 라이브 검증 | 또쓰 보유 0이라 시뮬레이션 대체. 다음 매수 시 실제 재알림 확인 필요 |
| SP 테스트에 텔레그램 토큰 가드 회귀 테스트 | MorningBrief엔 있고 SP엔 없음 |
| `korean_mkt_study/data/` `.gitignore` 미등록 | untracked 최상위 25개 + `out/`, 711MB. 백테스트 원자료라 삭제 금물 |

**2026-08-25 이 세션에서 처리 완료** `[실측]`

| 항목 | 처리 |
|---|---|
| `mop_model/CLAUDE.md` "유니버스 밖 6종목" | 편입 완료로 교체 (`snapshot_missing []` 확인) |
| `mop_model/CLAUDE.md` 유니버스 수 195 · "5슬롯" 표기 | 201 · 슬롯 수는 소비자 소관(현행 10)으로 정정 |
| 발열률 `MIN_WAVES` 실효값 | 코드 추적으로 확정 → `fever_model/CLAUDE.md` 에 "파라미터 정본" 절 신설 (§2.1) |
| `SCHEDULED_TASKS.md` 누락 | `kms-shadow-track` + LLV 3건 추가, 집계 38개로 재산정 |
| MagicFormula 미커밋 문서 3건 + `day_stop_*.py` 4개 | 커밋 |
| LLV 일회성 스크립트 3개 | `scripts/archive/` 이동 후 커밋 |
| StockPortfolio 미커밋 3건 | **다른 세션이 08-25 오전에 이미 처리** — `day-shadow.plist` 정본 추가 포함 |

---

## 6. 주의사항

### 6.1 하지 말아야 할 것 — 기각 목록 `[문서]`

제안 전에 원문을 읽어야 하는 문서: `korean_mkt_study/STRATEGY.md` **§11**(기각 18건)
+ **§4**(청산 근거), `mop_model/CLAUDE.md`의 "기각목록"·"튜닝 금지" 절.

| 제안 | 결과 | 판정 |
|---|---|---|
| 스윙 추세필터 제거·완화 | 재현율↑이나 MDD −68~−76% | **제외** — 놓친 기회 60%는 사후적 기회일 뿐 |
| 스윙 유니버스 완화 (4조→2조 / 30%→15%) | 네 방향 단조 악화, 코로나 손익 붕괴 | **제외** |
| onset 재신호 (같은 눌림 N일 간격) | MDD −29% → −40%, 눌림 진행 중 물타기化 | **제외** |
| 하락 멈춤 확인 후 진입 | 20조합 전부 열등 | **제외** |
| 눌림 임계 변동성 정규화 | 12조합 전부 열등, 고정 10% 최적 재확인 | **제외** |
| 20일 고점 창 | 신호 정밀도 최고이나 포트폴리오 최악(1.41억) | **제외** |
| 데이 포트 손절폭 변동배율 | 최적 지수 ~0 | **기각** (§3.4) |
| 데이 포트 손절 임계 1.5·2.0% | 비유의 + 스윕 비단조 | **1% 유지** |
| MOp 하이퍼파라미터 튜닝 | 36개 설정 탐색 test-holdout 상관 **−0.34** | **금지** |
| 발열률 과열·재가열 감점 | 20일 지평에서 과열 종목이 부진하지 않음 | 플래그 표시만 |
| 발열률 계단 구조(higher lows) | 추세 우위에 이미 흡수 | 제외 확정 |

**룰 4개의 어휘·파라미터를 섞지 말 것** — 데이·스윙 포트는 황금률의 입력이 아니다.
Wyckoff는 황금률의 5번째 점수 요소가 아니라 **게이트**다. `[문서]`

### 6.2 검증할 때의 함정 `[문서]`

1. **규칙 비교는 라이브 조건으로** (t+1 시가 · 왕복 0.23%). 백테스트 관례(종가 ·
   0.6%)로 재면 고회전 규칙이 부당하게 불리해진다 — 실측 1.33억 → 2.00억
2. **지표는 신호일(t−1) 값인지 확인** — 변동배율 검증에서 실제로 룩어헤드 버그가
   났다(연 0.6%p 과대평가)
3. **`Wyckoff_Phase` 룩어헤드** — v2 소급 재계산분으로 백테스트하면 부풀려진다.
   실시간 기록과 일치율 63.9%, 부호 반전
4. **`ticker_classification.json`의 `r2/slope/cell4/signal_rules`는 룩어헤드** —
   2025-04 이후 수익률로 역산한 값. `sector` 필드만 쓸 것
5. **`p`는 확률이 아니라 그날 유니버스 백분위** — "익일 갭상승 확률 85%" 식 해석 불가
6. **발열률 온도 절대 레벨은 캘리브레이션 의존** — 방향·상대 변화로만 읽고,
   5일 단위 변화에 의미 부여 금지(20거래일 MA로 읽을 것)
7. **`MarketCap == Close × ListShrs`는 성립하지 않는다** — Close는 수정주가,
   나머지는 그 시점 실제값. 검산식으로 쓰지 말 것

### 6.3 알려진 버그·사고 이력 `[세션]` `[문서]`

1. **텔레그램은 `<font>` 태그를 못 받는다** — 색상 규약(빨강/파랑) 메시지를 그대로
   보내면 **알림이 아예 안 온다**. 미러 경로에 색 태그 금지
2. **`day_shadow_track.py` `append_csv` 부모 폴더 미생성** — 08-24 첫 실행
   FileNotFoundError의 원인(mkdir이 `save_state()`에만 있었음). 같은 패턴 재발 주의.
   `[실측]` 수리(`3f9cdf3`)는 코드에 반영됐으나 **자동 실행으로는 아직 미검증** — §5.2
3. **슬롯 사이징은 수수료 포함 총지출 기준** — `slot_krw // price`는 버그
4. **SP는 `reload=False`** — 파이썬을 고치면 반드시 서버 재시작. 안 하면 화면은 새
   코드를, API는 옛 코드를 부르는 '유령 버그'
5. **가상계좌 config를 바꾸면 실계좌 손절이 같이 바뀐다** — `SP/app/realstop`이
   가상계좌 config를 직접 import
6. **Cowork 샌드박스에서 마운트 폴더에 git을 쓰면 `.git/*.lock`이 남아** 다음 커밋을
   막는다. 커밋은 Desktop Commander로
7. **pandas 3.x는 `groupby.apply`가 그룹 컬럼을 제외한다** — `run_gauge.py`가
   `_tk` 백업/복원으로 방어 중

---

## 7. 외부 의존성

### 7.1 다른 프로젝트 `[문서]`

| 상대 | 위치 | 관계 |
|---|---|---|
| **longlivevault (LLV)** | `~/DriveForALL/StoLab/longlivevault` | **읽기 전용.** OHLCV·지표 45컬럼·Wyckoff·수급·`ticker_classification.json`·`data/vol_scale.json` 공급. 진입점은 `magic_formula/_vault.py` |
| **StockPortfolio (SP)** | `~/DriveForALL/StoLab/StockPortfolio` | 신호 소비자 + **청산 규칙 실행 정본**. 8000 포트 |
| **hillstorm** | `~/DriveForALL/StoLab/hillstorm` | Wyckoff 분류 엔진 (LLV가 위임 호출) |
| **homalone (아웃퍼포머)** | 8501 포트 | 화면 — Quickview(황금률) · Temp.View(발열률) |
| **MorningBrief** | `~/DriveForALL/StoLab/MorningBrief` | 메일·Pushover 발송 위임 |

⚠ **형제 폴더 배치가 곧 설정이다.** SP의 `app/core/config.py`가 `StoLab/` 아래
형제 폴더 기준으로 경로를 계산하므로, 폴더 이름·위치를 바꾸면 서버가 안 뜬다. `[문서]`
⚠ **MagicFormula는 LLV에 쓰지 않는다.** 판단은 여기, 데이터는 LLV — 이 분업을 깨면
`_upsert_and_recompute` 계약과 충돌한다.

### 7.2 데이터·API `[문서]`

| 원천 | 경유 | 용도 |
|---|---|---|
| KIS (한국투자증권) | LLV `kis_fetcher` | 일봉·분봉·현재가·계좌·체결. NXT 소급 분봉(`FHKST03010230`) |
| KRX Open API | LLV `krx_fetcher` | 전종목 일별·옵션/선물·지수 |
| pykrx (KRX 스크래핑) | LLV `investor_flow` · `foreign_holding` | 수급 백필 · 외국인 지분율 |
| 토스 / KB | LLV `toss_fetcher` · `kb_fetcher` | 시세 백업 · 계좌 |
| Gmail SMTP · Pushover · Telegram | MorningBrief `push_sender` | 신호·게이지·발열률 통지 |

MagicFormula 자체는 외부 API를 직접 부르지 않는다 — **전부 LLV·MorningBrief 경유**다.
예외는 `mop_model`이 쓰는 LLV `data_service.fetch_today_ohlcv_snapshot`(15:10 잠정
스냅샷)뿐이고 이것도 LLV 진입점이다. `[문서]`

### 7.3 런타임 `[문서]` `[실측]`

- 파이썬 의존성: `requirements.txt`(리포 공통) + `mop_model/requirements.txt`
  (lightgbm · catboost — **이 둘은 mop_model 전용**, LLV로 번지면 안 된다)
- 실행 기계: 맥미니(상시). launchd plist의 파이썬 인터프리터가 잡마다 다르므로
  **기계를 옮기면 여기서 먼저 깨진다** — `configs/launchd/README.md` 참조
- 경로 오버라이드: `MOP_DIR` / `LLV_PATH` / `HILLSTORM_PATH` 환경변수

---

*이 문서는 스냅샷이다. 룰이 바뀌면 정본 파일을 먼저 고치고, 여기는 나중에 맞춘다.*
