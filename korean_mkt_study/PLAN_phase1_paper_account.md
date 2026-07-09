# 구현 계획서 — Phase 1: 가상 계좌 (페이퍼 트레이딩) — v2

- **작성**: 2026-07-06 (클로이) / **v2 수정**: Kane 피드백 반영 (실전 체결 모델·비용·페이지명) / **컨펌 대기**
- **전략 정본**: `kr_pullback_largecap_foreign` v1.0.0 (STRATEGY.md / strategy_spec.json / strategy_reference.py)
- **전체 로드맵**: ①가상 계좌 페이지(이 문서) → ②20:30 진입신호 생성+이메일 통지 → ③토스 실계좌 청산 트리거 이메일+푸시

---

## 1. Kane 확정 사항

| 항목 | 확정값 | 비고 |
|---|---|---|
| 가상 계좌 자본 | **20슬롯 × 200만원 = 4,000만원** | config로 변경 가능 |
| 시작 방식 | **빈 계좌, 구축 완료 후 첫 신호부터** | 과거 소급 없음 |
| 유니버스 | **6/30 기준 시총 5조↑ & 외인 25%↑** — **Kane이 직접 universe_202607.json 생성** | 스키마 §6. 유니버스 확대는 문제 없음(Kane). 월말 자동 갱신은 다음 단계 |
| **매수 체결** | 신호 발생일(t) **다음 거래일 09:10 실시간가** | 토스 API 가격. 시뮬레이션의 't+1 종가'를 실전 모델로 대체 |
| **매도 체결** | 보유종목 **15분 1회 모니터링** → 청산 트리거 발생 시 **5분 후 실시간가**로 매도 | 시뮬레이션의 '당일 종가 손절'을 실전 모델로 대체 |
| **거래 비용** | 매수 **0.015%** / 매도 **0.015% + 세금 0.2% = 0.215%** | 왕복 약 0.23% (백테스트 0.6%보다 현실화) |
| 고가주 처리 | 슬롯(200만)으로 1주도 못 사는 종목은 **슬롯 초과라도 1주 확보** | |
| 페이지 이름 | 사이드바 **"한국포트"** | 라우트 /paper |
| 실계좌 비교 | 토스 실계좌와의 비교 기능 불필요 | |

**⚠ 백테스트 규약과의 의도된 괴리**: 체결가(09:10가/트리거+5분가)와 비용이 검증된 백테스트(종가 체결·0.6%)와 다름. 가상 계좌의 목적이 '백테스트 재현'이 아니라 '실전 예행'이므로 의도된 변경. 성과를 백테스트 수치와 직접 비교하지 말 것.

---

## 2. 아키텍처

```
Magic Formula/korean_mkt_study/     ← 전략 정본 (읽기 전용, import만)
    strategy_reference.py            신호(onset·눌림깊이) 계산 정본 — 복제 금지
    data/universe_202607.json        7월 유니버스 (Kane 생성, 스키마 §6)

longlivevault/                       ← 데이터 (기존 기능 사용, 수정 없음)
    get_ohlcv()                      일별 close 패널 (신호 계산용, 비코어 자동 수집)
    toss_fetcher.fetch_current_price 실시간가 (배치 200종목, 09:10 매수가·15분 모니터링용)
    trading_calendar                 거래일 판정 (쿨다운·다음 거래일 계산)

StockPortfolio/                      ← 엔진 + 원장 + API + 페이지 (신규 구현 전부 여기)
    app/paper/                       engine.py, storage.py, routers.py 등
    templates/paper.html             "한국포트" 페이지
    data/paper/                      state.json, trades.csv, equity.csv
    scripts/paper_monitor.py         모니터링 스크립트 (신호/매수는 curl plist)
```

실시간가는 Kane 지시대로 **토스 API** 사용 — 단 직접 호출하지 않고 LLV `toss_fetcher` 경유(토큰 캐시 공유·자가복구 로직 재사용, 데이터 접근 단일화).

---

## 3. 구현 컴포넌트

### 3-1. 유니버스
- **Kane이 universe_202607.json 직접 생성** (스키마 §6). 종목명은 비어 있으면 LLV 명부(`get_name_map`)로 자동 보충.
- 로드 시 검증: 티커 형식(6자리)·KRX 명부 등재(`is_valid_ticker`) — 미등재 티커는 경고 후 제외.

### 3-2. 신호 생성 (매 거래일 저녁, LLV 20:30 배치 후)
- `strategy_reference.compute_indicators()`로 당일 종가 기준 **onset** 계산 (120일선 위 + 60일고점 대비 10% 눌림, 어제 False→오늘 True).
- 유니버스 필터 AND, 쿨다운(1거래일) 통과분만.
- 신규(미보유) + 불타기(보유중 & 슬롯<2) 후보를 **눌림 깊은 순**으로 `pending_buys`에 등록 → 다음 거래일 09:10 체결 대기.
- 필요 close 히스토리(120거래일 워밍업)는 LLV `get_ohlcv`로 확보.

### 3-3. 매수 집행 (다음 거래일 09:10)
- `pending_buys`를 눌림 깊은 순으로 순회: 토스 실시간가 조회 → 정수주 `floor(200만/가격)` 매수, **가격>200만이면 1주**.
- 제약: 빈 슬롯(전체<20, 종목<2), 가용현금(T+1 미현금화 매도대금 제외). 매수 수수료 0.015%.
- 미체결 후보는 이월하지 않고 폐기 (다음 onset을 기다림 — 스펙의 onset-only 원칙 유지).

### 3-4. 청산 모니터링 (장중 15분 1회, 09:00~15:30)
- 보유종목 토스 실시간가 일괄 조회 → **peak 갱신** → `가격 ≤ peak×0.8` 판정.
- **peak 정의 (Kane 확정 2026-07-06)**: 포트폴리오 편입 후 **관측된 모든 값**의 최댓값 — 매수체결가, 장중 15분 샘플, 일별 확정 OHLC의 High. 보관 데이터는 데일리 OHLC(LLV 소유) + 당일 15분 샘플 1일치뿐이고 peak 자체는 포지션당 스칼라 1개로 유지.
- 트리거 발생 시 **5분 대기 → 재조회한 가격으로 전량 매도** (포지션 단위, 트랜치 구분 없음). 매도 비용 0.215%.
- 매도대금 T+1 현금화, 매도 당일 재매수 금지(쿨다운).

### 3-5. 원장 저장소 (`data/paper/`)
- `state.json`: 현금(가용/T+1대기), pending_buys, 포지션 {ticker: 슬롯수, 주식수, 평단, peak, 진입일}, 종목별 최근 매도일
- `trades.csv`: 전 거래 로그 (일시, 종목, 매수/매도, 수량, 체결가, 수수료·세금, 사유 = 신규/불타기/손절)
- `equity.csv`: 일별 마감 평가액 (현금 + 보유평가, 종가 기준)
- ※ parquet → CSV 로 변경 (구현 시): 거래량이 작고 append 단순, pyarrow 없이 검사 가능
- 모든 쓰기는 원자적(temp→rename), 일 1회 `backups/` 스냅샷

### 3-6. 스케줄 (launchd, plist 정본은 SP `configs/launchd/` + symlink)
| Label | 시각 | 역할 |
|---|---|---|
| `…paper-signal` | 매일 20:45 | §3-2 신호 생성 (LLV 20:30 배치 완료 확인 후, 미완료 시 보류+로그) |
| `…paper-buy` | 평일 09:10 | §3-3 매수 집행 (거래일 아니면 즉시 종료) |
| `…paper-monitor` | 15분 간격 (StartInterval 900) | §3-4 청산 모니터링 (장중 아니면 즉시 종료) |

각 스크립트는 SP 서버의 `POST /api/paper/*` 를 호출하는 thin wrapper — 엔진 로직은 서버 한 곳에만.

### 3-7. API
- `GET /api/paper/summary` · `positions` · `candidates`(pending) · `trades` · `equity`
- `POST /api/paper/run-signal` / `run-buy` / `run-monitor` — launchd·수동 공용, **멱등** (같은 슬롯·같은 날 중복 실행 방지)

### 3-8. 페이지 (`/paper`, 사이드바 **"한국포트"**)
- 요약 카드: 총자산 / 현금(가용·T+1대기) / 평가손익 / 슬롯 (n/20)
- 보유 테이블: 종목, 슬롯, 평단, 현재가, 수익률, peak, **손절선(peak×0.8)까지 거리%** — 5% 이내 경고색
- 매수 대기(내일 09:10 체결 예정) 테이블, 거래 로그, 일별 equity 차트
- 표기: 상승 `#ef5350`·하락 `#1976D2`, 우측 정렬, 원화 반올림, % 소수 1자리

### 3-9. 검증 (pytest)
- 신호: 엔진 onset이 `strategy_reference`와 일치
- 단위: 정수주(+1주 예외), 비용(0.015%/0.215%), T+1, 쿨다운, 불타기 한도, 멱등성
- 시나리오: 모의 가격 시퀀스로 신호→09:10 매수→peak 갱신→트리거→5분 후 매도 전 과정

---

## 4. 작업 순서 (2026-07-06 구현 완료)

| # | 작업 | 상태 |
|---|---|---|
| 1 | Kane: universe JSON 생성 (universe_2026-07_4jo_25pct.json — 시총 4조·외인 25%, 60종목) | ✅ |
| 2 | 유니버스 종목 close 확보 | ✅ 불필요 — 60종목 전부 LLV 기보유 (core 33 + extend 27) |
| 3 | 엔진(신호·매수·모니터링·청산) + 저장소 + pytest 12건 | ✅ 전부 통과 |
| 4 | API 10개 + /paper 페이지 + 사이드바 "한국포트" | ✅ |
| 5 | launchd plist 3종 (signal 20:45 / buy 09:10 / monitor 900초) | ✅ 정본 작성 — **등록은 Kane 수동** (§7) |
| 6 | 문서 갱신 | ✅ 이 문서 |

## 6-2. Phase 2 — 이메일 통지 (2026-07-06 구현 완료)

Kane 확정: 신호 메일 **2회**(20:45 생성 직후 + 다음 거래일 **08:45** 개장 15분 전 리마인더),
매수 체결 메일(09:10 집행 결과), 청산 체결 메일(트리거+5분 매도 직후). 신호 메일은
신호 0건이어도 매일 발송(배치 생존 확인용).

- `app/paper/notify.py` — Gmail SMTP(SP .env GMAIL_USER/GMAIL_APP_PW/ALERT_EMAIL,
  기존 인프라 재사용) + HTML 빌더 3종. 발송 실패는 로그만 — 원장 처리 불차단.
- `engine.run_reminder` + `POST /api/paper/notify-reminder` (멱등, 거래일 가드)
- launchd 추가: `com.kane.stockportfolio-paper-remind.plist` (평일 08:45)
- 테스트: tests/test_paper_notify.py 6건 (빌더·fail-safe·멱등)

## 7. launchd 등록 (Kane 실행)

```bash
cd ~/DriveForALL/StoLab/StockPortfolio/configs/launchd
for f in com.kane.stockportfolio-paper-*.plist; do
  ln -sf "$PWD/$f" ~/Library/LaunchAgents/"$f"
  launchctl load ~/Library/LaunchAgents/"$f"
done
```

리허설(서버 기동 상태에서): `/paper` 페이지의 수동 실행 버튼 또는
`curl -X POST http://localhost:8000/api/paper/run-signal` — 멱등이라 안전.

---

## 5. 리스크 / 메모

- **토스 API 장애 시**: 매수 집행·모니터링은 LLV 경유라 KIS 폴백 가능(`get_current_price` 1차 KIS). 단 Kane 지시가 '토스 API 사용'이므로 기본 토스, 실패 시 KIS 폴백 후 로그.
- **15분 간격의 한계**: 급락 시 트리거 인지 최대 15분 + 5분 지연 → 체결가가 손절선보다 낮을 수 있음(실전과 동일한 현실).
- **유니버스 월말 갱신**: 다음 단계로 이월 (8월 유니버스는 7/31 기준 필요).

---

## 6. universe_202607.json 스키마 (Kane 작성용)

```json
{
  "strategy_id": "kr_pullback_largecap_foreign",
  "base_date": "2026-06-30",
  "apply_month": "2026-07",
  "filters": { "mktcap_min_krw": 5000000000000, "foreign_ratio_min_pct": 25.0 },
  "created_by": "Kane",
  "stocks": [
    { "ticker": "005930", "name": "삼성전자", "mktcap_krw": 500000000000000, "foreign_ratio_pct": 50.1 },
    { "ticker": "000660", "name": "SK하이닉스", "mktcap_krw": 200000000000000, "foreign_ratio_pct": 55.3 }
  ]
}
```

| 필드 | 필수 | 설명 |
|---|---|---|
| `stocks[].ticker` | **필수** | 6자리 문자열, **앞자리 0 유지** ("005930", 숫자 아님) |
| `stocks[].name` | 선택 | 비우면 LLV 명부로 자동 보충 |
| `stocks[].mktcap_krw` | 선택 | 원 단위, 기록·페이지 표시용 (판정엔 미사용) |
| `stocks[].foreign_ratio_pct` | 선택 | %, 0~100, 기록·페이지 표시용 (판정엔 미사용) |
| `base_date` / `apply_month` | **필수** | 유니버스 기준일 / 적용 월 — 월 전환 검사에 사용 |
| `filters` | 권장 | 어떤 기준으로 뽑았는지 기록 (재현성) |

엔진이 매매 판정에 실제로 쓰는 건 **ticker 목록 + apply_month** 뿐이야. 나머지는 기록과 화면 표시용.
