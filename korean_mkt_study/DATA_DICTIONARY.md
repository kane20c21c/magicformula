# 데이터 정의 사전 — 운영(LLV) vs 연구(korean_mkt_study)

2026-11-15 케인 — **한 방향 원칙**: 운영(LLV, 맥미니)과 연구(이 폴더, 맥에어)는 데이터를 따로 갖는다.
연구는 KRX 에서 직접 받고(`kms_store.py`), 운영 데이터를 원천으로 쓰지 않는다. 같은 이름이라도 정의가 다를 수 있어
여기에 나란히 적는다. 세트 목록·상태는 `kms_data.py` (`SETS`), 기간·행 수는 각 폴더 `MANIFEST.json`.

| 항목 | 운영 LLV (`panel.parquet` 등) | 연구 정본 (`data_2015/`) | 비고 |
|---|---|---|---|
| 기간·종목 | 2023-01-02~ · 와이드버스 285 | 2015-01-02~ · 시장 전체 ~2,700 (우선주·스팩 포함, 코넥스 제외) | 연구 동결본 `data/` 는 2010~2026-06-30 |
| 가격 관례 | **수정주가** (KIS `FID_ORG_ADJ_PRC=0`, pykrx 기본) | **수정주가** (pykrx 기본) | 수정 기준일 = 수집 시점. 이후 분할이 나면 과거는 다시 받아야 맞는다 |
| OHLC 시장 | KRX 정규장 (9-14 이후 `regular_ohlc` 교정) | KRX 정규장 — 2026-09-14 이후 고가·저가·종가는 **KRX Open API × 시가 비율** 로 교정(2026-11-15) | pykrx 수정주가가 9-14(애프터마켓 개장) 이후 종가에 애프터 가격을 섞는다 — 양쪽 모두 같은 사고를 고쳤다 |
| 거래량 | **KRX+NXT 통합(UN)** | **KRX(J)** 단독 | 2026-11-15 실측: 연구 거래량 = KRX 원본 ÷ 시가 비율 (99.9%). NXT 물량이 큰 종목은 연구 쪽이 작다 |
| 거래대금 | `Amount` (UN) | 일봉엔 없음 · `meta.value` = 월 마지막 거래일 거래대금 | |
| 시가총액·주식수 | `MarketCap`·`ListShrs` 일별 (KRX) | `meta.mktcap`·`meta.shares` **월말** | 연구 월말 라벨 = **달력 월말**(값은 그 달 마지막 거래일) |
| 외국인 지분 | `Foreign_Ratio` **%** 일별 — 보유/상장 직접 계산, KRX 라벨을 하루 당겨 실제 잔고일로 | `foreign.foreign_ratio` **%** 월말 · `meta.foreign` = **비율(÷100)** | 연구 옛 행은 pykrx float16 잔재(48.28125 류). 새 행은 보유/상장 정확값 (차 ≤0.01%p) |
| 투자자 수급 | `Inst_Net`·`Foreign_Net`·`Retail_Net` 원 — pykrx 백필 + KIS 일별(백만원 절사, 1백만원 이하 차 동일 취급) | `investor_flow` 개인·기관·외국인·기타법인 — ~2026-06-30 pykrx **원**, 07-01~10-29 KIS **백만원 반올림** | 둘 다 KRX 정규장 수급(NXT 제외). 연구 KIS 구간은 소형주 작은 순매수가 0 |
| 관리종목 | `issue_status` 일별 (한국투자 종목 마스터) — **2026-11-15 부터** · 코스피·코스닥 · 관리·정지·정리매매·환기·스팩·우선주 따로 | `meta.is_managed` — **코스닥만** KRX 소속부(월 마지막 거래일) 2015~ · `managed_known` = 코스닥 True / 코스피 False | KRX 소속부는 한 칸 라벨이라 하한값 (관리+환기 → '환기'로 보이는 경우). 코스피 과거 관리 이력은 원천이 없다 |
| 우선주·스팩 | `issue_status.Pref`·`Spac` | `meta.is_pref`(이름 끝 우/우B/우C)·`is_spac`(이름에 스팩) | 2026-11-14 코스피 우선주: 마스터 109 = 이름 규칙 109 |
| 섹터 | `core/extend_tickers` 정본 섹터 | `meta.sector='기타'` 자리표시자 | 연구 섹터는 `data_idx/krx_sector_members.parquet` (KRX 업종) |
| 종목군 | 와이드버스 ⊃ 유니버스 ⊃ 메가갤럭시 (LLV `groups/`) | 연구는 자기 데이터로 다시 계산 | 운영 명단을 연구 입력으로 쓰지 않는다 |
| 지수 | `tickers/KOSPI·KOSPI200·VKOSPI·KOSDAQ150·KRX_TMI_MID` (2015~) | `index_kospi`·`benchmark` 월말 · `kospi200_daily` (FDR KS200 형식) | |
| 파생 | `data/options/` 일별 opt/fut (2015~) + `derivatives_all.parquet` | `derivatives_all.parquet` | |

## 연구 정본을 늘릴 때

필요할 때만 (정기화하지 않는다 — 케인 2026-11-15). 시간대 제한 없음.
`kms_store.py` 가 KRX 에서 직접 받는다 — 가격은 KRX Open API 원주가 + 기준가로 수정계수, 외인·펀더멘털·수급은 pykrx,
pykrx 가 막히면 수급만 한국투자로. **LLV raw·패널을 원천으로 쓰지 않는다.**

## 백업

연구 데이터는 **에어에만** 있고 git 밖이다. 미니의 R2 일일 백업은 미니 StoLab 만 올린다.
바꾼 뒤(연장·교정) 케인이 `StoLab-meta/scripts/backup_research_to_r2.sh` 를 돌린다 → R2 `<버킷>/air-research/current/`
(바뀐·지운 파일은 `air-research/_generations/<날짜>/` 에 90일). 처음 한 번 `brew install rclone` + `setup_r2.sh`.

## 세트 상태

| 세트 | 폴더 | 상태 | 쓰임 |
|---|---|---|---|
| current | `data_2015/` | 정본 | 새 연구 |
| frozen_20260630 | `data/` | 동결 | 2026-11 이전 백테스트 재현 (고치지 않는다) |
| retired_ext_0911 | `data_ext/` | 은퇴 | LLV 패널로 연장한 사본 — 끝난 연구 재현용 |
| retired_1104 | `data_1104/` | 은퇴 | LLV 패널·raw 로 연장한 사본 — 끝난 연구 재현용 |
| indices | `data_idx/` | 보조 | 지수·업종 |
