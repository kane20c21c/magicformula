# 통일장 시장 층 판정기 (`market_layer`) — v1.0, 그림자 단계 (2026-11-10)

무대·시장·종목 3층 위험 구조 중 **시장 층**의 노출 판정. 매일 "시장 전체를 들고 있을 때인가, 현금으로 빠질 때인가"를 낸다.
True = 보유, False = 현금. 종목 고르기(스윙·데이 포트)와는 독립이다 — 그림자 추적(③)이 두 포트 위에 얹어 비교한다.

| 판정기 | 지위 | 정의 |
|---|---|---|
| ⑦ C2+갭쇼크 | **운영 후보** (케인 11-10) | 관문 모드 위에 평소 = C2 |
| ⑤ 갭쇼크·브레드스·평소보유 | 그림자 (기록만) | 관문 모드 위에 평소 = 보유 |

- **관문 모드** — 갭쇼크 창(코스피 200 하루 ≤ −3% & 하락 중 갭 몫 ≥ 40% 인 날로부터 10거래일) 안에서 브레드스가 하락이면 현금, 브레드스가 상승으로 돌아서면 해제.
- **브레드스** — (KOSPI+KOSDAQ 상승 종목) ÷ (상승+하락) 의 20일 지수평균 − 직전 250일 중앙값(최소 120일)이 +0.05 이상 상승 / −0.05 이하 하락 / 사이 유지. **전 종목**(케인 11-10 (가)).
- **C2** — 코스피 200 과 KRX 중형 TMI 가 둘 다 200일선 위(괴리형). 꺾이는 날 퇴출, 퇴출 뒤 저점 +10% 반등 또는 괴리형 회복 시 복귀.

파라미터 정본 = `config.py`, 판정 = `judge.py`(순수 함수), 재료 = `inputs.py`(LLV 읽기 전용), 실행 = `scripts/market_layer_signal.py` → `output/market_layer/YYYYMMDD.json`.

## 잠정과 확정

| | 19:10 저녁 (잠정) | 다음 날 아침 (확정) |
|---|---|---|
| 코스피 200 | LLV 19:00 kis_update | LLV 08:01 KRX |
| 브레드스 | 한국투자 업종지수 등락 수 (`inputs.KIS_PHASES` 순) | LLV KRX 원본 종가 대비 |
| 중형 TMI | 그날 없음 → 200일선 상태 **전날 유지** | LLV 08:01 KRX |

같은 날 파일에 `provisional` · `confirmed` 블록을 따로 두고, 둘 다 있으면 `provisional_matches_confirmed` 를 기록한다.

실행: SP 저녁 체인 **12단계**(19:00대, 잠정) · **08:40** `com.stolab.magicformula-market-layer-confirm`(`--confirm` — 확정 재료가 없으면 쓰지 않고 exit 1).
소비자: SP **유스윙포트**(`app/uswing`, 케인 11-10) — 직전 거래일 판정(확정 우선)으로 09:10 매수/전량 매도/보류를 정한다. 알림도 SP 가 보낸다.

## 함정

- ⚠ **판정일이 아닌 날의 재료 결측은 사고다.** 200일선 창에 하루라도 빠지면 그 뒤 199일 이평이 비어 상태가 조용히 얼어붙는다 —
  `load_inputs` 가 찾아 `InputGapError` 로 멈춘다(exit 1). TMI 결측은 LLV `backfill_krx_tmi_mid.py --resume` 로 메운다.
- 한국투자 '상승'에 '상한'이 들어 있는지는 LLV `market_breadth.KIS_ADV_INCLUDES_LIMIT` 가 정본 — **들어 있다**(`True`, 2026-11-10 실측).
  `None` 으로 되돌리면 판정일 브레드스를 쓰지 않고 전날 상태를 유지한다.
- 애프터마켓(16:00~20:00)은 등락 수·지수를 **바꾸지 않았다** (2026-11-10 하루치 — 15:35·16:41·19:06 같은 값).
  `KIS_PHASES` 는 reg(15:33) → sig(19:00) → fact(20:30) 그대로 둔다.
- 연구 재현: 연구 입력(필터 브레드스·pykrx TMI)으로 2015-01-02~2026-11-04 2,905일 ⑦·⑤·브레드스·C2 **하루도 다르지 않음**
  (`tests/test_market_layer.py`, 고정본 `tests/fixtures/market_layer_research.csv` — 생성 `korean_mkt_study/make_market_layer_fixture.py`).
  LLV 운영 재료로는 보류 구간 106일 100% 일치, 2025-12~2026-04 는 짧은 이력·정의 차이로 62일 갈림(운영 = 현금 쪽).

## 국면 라벨 표 — `output/market_layer/regime_history.parquet` (2026-11-10 케인)

판정기와 **같은 함수**(`judge()`)로 전 구간을 다시 계산한 하루 한 줄 이력. 드라이브·손절·쇼크 같은 **다른 분석이 날짜로 붙여 쓰는 공용 라벨**이다.
정의 정본 = `market_layer/history.py` 머리말. 메타(유효 시작일·재료 범위·최신 라벨) = 같은 이름 `.meta.json`.

| 열 | 뜻 |
|---|---|
| `breadth` | 브레드스 상승 True / 하락 False (nullable boolean) |
| `c2` | C2 보유 True / 퇴출 False |
| `regime` | 2×2 — `상승·보유` · `상승·퇴출` · `하락·보유` · `하락·퇴출` |
| `p7` · `p5` | ⑦ · ⑤ 보유 True / 현금 False |
| `shock` · `gate_window` · `days_since_shock` · `gate_mode` | 정의 B 갭쇼크일 · 10일 관문 창 · 마지막 쇼크 뒤 경과 거래일 · 관문 모드 |
| `{breadth,c2,regime}_spell/_day/_len/_cens` | 구간 번호 · 구간 안 며칠째(1부터) · 구간 길이 · 검열 표시 |
| `k200_close/_ret/_gap` · `tmi_close` · `breadth_ratio` · `breadth_value` | 재료 (확인용) |

```python
from market_layer import attach_regime, load_history
drives = attach_regime(drives, date_col="D1")          # D1 의 국면 → mkt_regime · mkt_breadth · mkt_c2 …
drives = attach_regime(drives, date_col="D1", lag=1)   # 그날 장중 판단용 — 전날 라벨 (미래 정보 없음)
```

- 갱신: 08:40 확정 판정(`--confirm`)이 끝에 표를 통째로 다시 만든다 (실패해도 판정은 막지 않고 ⚠ 만 찍는다).
  LLV 재료를 과거로 늘린 뒤에는 `python3 scripts/market_layer_history.py` 로 바로 다시 만들 수 있다. **확정 재료만** 싣는다
  (코스피 200·중형 TMI·KRX 확정 브레드스가 모두 있는 마지막 날까지 — 19:10 잠정은 안 들어간다).
- ⚠ **유효 구간 밖은 결측이다** — 판정기는 시작값을 '보유·상승'으로 두므로 재료가 쌓이기 전이 그렇게 보인다. 표는 그 구간을 지운다.
  브레드스 = 값이 처음 ±0.05 를 넘은 날부터, C2 = 두 200일선이 생긴 뒤 괴리형이 처음 성립한 날부터, ⑦·regime = 둘 다.
- 유효 시작 (2026-11-10 LLV 브레드스 2015 백필 뒤): **브레드스 2015-06-29~** (49구간) · **C2·⑦·regime 2016-04-22~** (C2 87구간, regime 129구간).
  연구 쪽 전 종목 비율(prices.parquet)로 메우지 말 것 — 겹치는 834일에서 LLV 와 상태가 **91.2%** 만 일치한다 (정의가 다르다).
- ⚠ 구간 길이 비교엔 `_cens == ""` 인 구간만 — 첫 구간은 시작을, 마지막 구간은 끝을 모른다.
- ⚠ 라벨은 그날 **종가까지** 본 값이다. 붙인 날 장중에 쓸 규칙이면 `lag=1`.

## 다른 프로젝트에서 쓰기 — 사용 방법 (2026-11-10 케인, 정본)

시장 국면을 쓰는 분석·규칙은 **새로 계산하지 말고 아래 둘 중 하나를 읽는다.** 판정은 MagicFormula `market_layer` 한 곳에서만 한다.

### 어디서 읽나

| 용도 | 읽는 곳 | 갱신 | 예 |
|---|---|---|---|
| **지난 기간 분석·연구** (이벤트·신호 표에 날짜로 붙이기) | `output/market_layer/regime_history.parquet` | 미니 08:40 확정 판정 끝에 통째로 재생성 · 에어는 02:00 `pull_from_mini` | `korean_mkt_study/drive_regime.py` · `bounce_regime.py` · `wyckoff_regime.py` |
| **매일 운영** (오늘·어제 판정으로 표기·게이트) | `output/market_layer/YYYYMMDD.json` — `confirmed` 우선, 없으면 `provisional` | 19:10 잠정 · 다음 날 08:40 확정 (같은 파일) | SP `app/adapters/market_layer.py` (유스윙포트 게이트) · `bounce_rule/regime.py` (반등 룰 메일 표기) |

### 코드

```python
# 연구 — MagicFormula 를 import 할 수 있을 때
import sys; sys.path.insert(0, "<StoLab>/MagicFormula")
from market_layer import attach_regime, load_history
ev = attach_regime(ev, date_col="date")                 # 그날 종가까지 본 라벨 → mkt_breadth · mkt_c2 · mkt_regime · mkt_p7 · *_day
ev = attach_regime(ev, date_col="date", lag=1)          # 그날 장중 판단 규칙 — 전날 라벨 (미래 정보 없음)
ev = attach_regime(ev, "date", cols=["breadth"], prefix="")   # 필요한 열만, 접두사 없이

# 연구 — import 가 어려운 저장소: parquet 을 직접 읽어 날짜로 조인
h = pd.read_parquet("<StoLab>/MagicFormula/output/market_layer/regime_history.parquet"); h.index = pd.to_datetime(h.index)
ev = ev.join(h[["breadth", "c2"]], on="date")

# 운영 — 그날 판정 (bounce_rule/regime.py 의 market_regime 이 그대로 쓰는 순서: 확정 → 잠정 → 이력 표 → None)
from bounce_rule.regime import market_regime
reg = market_regime("2026-11-10")   # {'breadth': '하락', 'c2': '보유', 'stage': '잠정', 'breadth_value': -0.06} 또는 None
```

### 원칙 — 2026-11-10 세 점검(드라이브 · 반등 룰 · 와이코프)에서 확인한 것

1. **국면 축은 브레드스를 쓴다.** 세 점검 모두에서 브레드스만 학습 두 기간·보류 구간에 걸쳐 같은 방향이었다.
   C2·갭쇼크 창은 기간마다 방향이 뒤집혔다 (근거: `korean_mkt_study/DRIVE_STUDY.md` §8.7·§9 · `WYCKOFF_REGIME.md`).
   C2·⑦ 는 '노출(보유/현금) 판정'용이다 — 종목·유형 규칙의 통제 변수로 쓰려면 따로 검정한다.
2. **라벨 날짜 = 판단 시점.** 드라이브는 D1, 신호는 신호일 종가 (그날 저녁 판단 → 다음 날 시가 진입이면 lag 0 이 미래 정보 없음).
   그날 장중에 판단하는 규칙이면 `lag=1`. 사후 정보(끝난 날·구간 다수결)로 붙인 라벨은 서술용으로만.
3. **수익은 코스피 200 초과로 비교한다.** 국면은 시장 방향을 같이 바꾼다 — 반등 룰에서 원수익 차(+6.1 vs +0.7%)의 대부분이 시장 몫이었다 (`DRIVE_STUDY.md` §9.4).
4. **건수 비교는 하지 않는다.** 브레드스가 곧 상승·하락 종목 수라 '하락 국면에 하락 드라이브가 많다' 는 순환이다. 성질·성적만 비교한다.
5. **학습 = 2026-05 까지, 2026-06~ = 보류(검증용).** 보류 구간은 판정 기준을 먼저 문서에 적고 커밋한 뒤 **한 번만** 연다. 판정은
   재현(같은 방향) · 강한 재현(95% 구간이 0 을 안 품음) · 불통과(반대) · 판정 불가(한쪽 20건 미만).
6. **같은 날 묶인 표본은 독립이 아니다.** 신뢰구간은 날짜 단위(드라이브 = D1, 사후 = 끝 날, 신호 = 신호일) 부트스트랩.
7. **결측은 빼고 쓴다.** 유효 구간 밖은 NA(판정기 시작값을 지운 것) — 상승·보유로 채우지 말 것. 구간 길이·전환 분석은 `_cens == ""` 구간만.
8. **규칙에 반영할 땐 '표기' 부터.** 반등 룰은 신호 조건을 그대로 두고 메일·JSON 에 국면만 적었다 (케인 11-10 (가)) — 표본이 쌓이면 다시 판정한다.

### 새로 쓰는 쪽 체크리스트

- [ ] 연구면 `attach_regime`(또는 parquet 조인), 운영이면 그날 JSON — 직접 계산하지 않았나
- [ ] 라벨 날짜가 판단 시점인가 (`lag` 확인)
- [ ] 초과수익(코스피 200 차감)으로 비교했나 · 날짜 단위 신뢰구간인가
- [ ] 학습/보류를 나눴나 · 보류 기준을 먼저 커밋했나
- [ ] 결과 문서에 이 절을 링크했나 (StoLab-meta `CLAUDE.md` '공용 라벨' 절이 여기를 가리킨다)
