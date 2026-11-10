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
- ⚠ 2026-11-10 에어 실측: 브레드스 재료가 LLV 에 **2023-06-01~** 만 있어 브레드스·⑦·regime 은 **2023-11-24~** 만 유효
  (브레드스 18구간, C2 2016-04-22~ 87구간). 2015년까지 늘리려면 LLV `backfill_market_breadth.py --start 20150102 --end 20230531 --fetch-missing`
  (미니, 평일 08~18시) 뒤 표를 다시 만든다. 연구 쪽 전 종목 비율(prices.parquet)로 앞을 메우지 말 것 — 겹치는 834일에서 LLV 와 상태가 **91.2%** 만 일치한다.
- ⚠ 구간 길이 비교엔 `_cens == ""` 인 구간만 — 첫 구간은 시작을, 마지막 구간은 끝을 모른다.
- ⚠ 라벨은 그날 **종가까지** 본 값이다. 붙인 날 장중에 쓸 규칙이면 `lag=1`.
