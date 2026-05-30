# Magic Formula — M4 분석 워크벤치 핸드오프

> 작성: 2026-05-28 · Kane & 클로이  
> 다음 작업자 (= Mac.Air에서 작업할 다음 클로이) 가 컨텍스트를 잃지 않도록 정리한 문서.

---

## 0. 시나리오 결정 — A. 분석 워크벤치 분리

| 머신 | 역할 | 가동 트랙 |
|---|---|---|
| Mac Mini (Intel i5, 8GB) | **데일리 운영** 유지 | `scripts/daily_signal.py` · launchd 평일 20:40 |
| Mac (Apple Silicon M4, 16GB) | **분석 워크벤치** (신규) | `scripts/run_analysis.py` 부정기 |

두 머신은 `configs/active_strategy.yaml` 한 파일을 **git** 으로만 주고받는다.

```
[새 M4]                                    [옛 Intel]
분석 트랙 (Kane 이 생각날 때)               데일리 트랙 (평일 20:40 KST)
       │                                          ▲
       │ output/analysis/YYYY-MM-DD/              │
       ▼                                          │
update_strategy.py                                │
       │                                          │
       ▼                                          │
active_strategy.yaml ── git push ─→ GitHub ─→ git pull
                                                  │
                                                  ▼
                                           ★ 다음 평일 20:40 부터
                                             새 전략으로 데일리 생성
```

### 작업 단계

- **STEP A** — Mac.Air에 분석 환경 셋업 + 코어 데이터 카피 + 검증 (한 번, 필수)
- **STEP B** — **★ 이번 작업의 본 목표 ★** : M4 환경에서 백테스트 가속화 (코드 수정)
- **STEP C** — 분석 사이클 운영 (부정기, Kane 이 생각날 때 아무 때나)

---

## 1. 기존 작업 컨텍스트 (= 지금까지 한 일)

### 1-1. 시스템이 뭐 하는 거였더라

Magic Formula 는 종목 분석 페이지의 **종합 시그널 점수** (5영역 가중평균 -10 ~ +10 점) 가중치를 백테스트로 검증하고, 그 결과를 데일리 진입 신호 리포트에 반영하는 시스템. 두 트랙이 `configs/active_strategy.yaml` 단일 SSOT 로 연결돼 있다.

데이터 수집은 **longlivevault** (SSOT) 가 책임지고, Magic Formula 는 vault 의 parquet 만 읽는다. 시나리오 A 에서는 **Mac.Air의 vault batch 를 가동하지 않으므로**, core.parquet 시점은 Mac.Mini에서 카피한 시점에 고정된다 (= 다음 동기화까지의 "데이터 스냅샷").

### 1-2. 현재 운영 전략 (이전 시점 상태)

`configs/active_strategy.yaml` :

```yaml
strategy_id: CompR15
last_updated: '2026-05-18'
source_analysis: output/analysis/2026-05-18/weight_ranking.md
weights:
  trend: 0.30
  momentum: 0.15
  volume: 0.10
  volatility: 0.25
  wyckoff: 0.20
rule: R1
area4_mode: trend
threshold: 5.0
universe: core_all
```

### 1-3. 분석 트랙의 "처음 만든 코드" 규모

- Mac.Mini에서 돌리던 버전 : **51 가중치 조합 × 3 규칙 = 153회 백테스트** · 종목 universe ~59. **이 시점이 M4 가속화의 진짜 베이스라인** 
- 현재 (commit `e62d5c4`) : `_COMBO_SPECS` 가 **8 조합** 으로 축소되고, ADAPTIVE 규칙이 추가돼 **8 × 3 = 24회**. universe 는 `core_excl_split` (사업분할 제외, 약 69 종목).
- ※ `optimizer.py` 의 docstring/리포트 텍스트에는 아직 "51조합 × 3규칙 = 153회" 잔재가 남아있음 — 정리 대상.

### 1-4. 가장 최근 분석 산출물 — `output/analysis/2026-05-18/`

153회 백테스트 결과. **Top 1 ~ 13 위가 모두 R3 규칙** 이었음 (HLLBH 가 알파 +102.7% 로 1위). 그런데 현재 운영은 R1 — 분석 결과 ↔ 운영 전략의 차이는 의도된 것. 분석 과정에서 결과는 R1 = R3임이 밝혀짐.

---

## 2. Mac.Air (M4) 으로 가져갈 자산

### 2-1. Git 에서 그대로 받아오면 되는 것

| 저장소 | URL | 마지막 commit |
|---|---|---|
| Magic Formula | `https://github.com/kane20c21c/magicformula.git` | `e62d5c4` (main, clean) |
| longlivevault | `https://github.com/kane20c21c/longlivevault.git` | `24a8c20` (main, clean) |

두 저장소 모두 **uncommitted 변경 없음** — `git clone` 만으로 코드 100% 동기화 가능.

### 2-2. 직접 카피해야 하는 것 — 딱 두 파일

코드를 확인한 결과 Magic Formula 분석 트랙이 vault 에서 호출하는 건 단 두 가지:

| vault API | 읽는 파일 |
|---|---|
| `data_service.get_ohlcv(ticker, ...)` (코어) | `data/ohlcv/core.parquet` |
| `ohlcv_store.get_ohlcv("KOSPI", ...)` (알파용) | `data/ohlcv/tickers/KOSPI.parquet` |

그리고 `data_service.get_ohlcv()` 는 코어 종목이면 KIS API 안 부르고 parquet 직접 반환 — `if ticker in CORE_TICKERS: return ohlcv_store.get_ohlcv(...)`.

**따라서 카피 대상은 정확히 이 두 파일:**

```
longlivevault/data/ohlcv/core.parquet              (11 MB)
longlivevault/data/ohlcv/tickers/KOSPI.parquet     (수십 KB)
```

KOSPI.parquet 이 빠지면 백테스트는 돌긴 하지만 알파 메트릭이 `N/A` 로 나오므로, 두 개 모두 권장.

### 2-3. 카피 안 해도 되는 것 (체크리스트)

| 파일 | 카피 불필요 이유 |
|---|---|
| `data/ohlcv/tickers/VKOSPI.parquet` | Magic Formula 코드 어디서도 호출 없음 (`grep` 확인) |
| `data/ohlcv/tickers/KOSPI200.parquet` | 동일 |
| `data/krx_trading_calendar.json` | Magic Formula 는 OHLCV 의 Date 인덱스만 사용 |
| `data/us_market.json` | Morning Brief 전용, Magic Formula 의존 없음 |
| `data/cache/kis_token.json` | vault 가 KIS 호출할 때만 필요, 코어 종목은 KIS 미호출 |
| `data/raw/krx_*.parquet` | vault 일일 배치용 |
| `longlivevault/.env` | vault 가 KIS/KRX 호출 안 함 + import 시점 .env 로드 코드 없음 (`__init__.py` / `data_service.py` / `ohlcv_store.py` 확인) |
| `data/ohlcv/core.parquet.bak*` | Mac.Mini의 백업본들 |

`Magic Formula/output/analysis/2026-05-18/` 은 git 추적 안 되지만, 작업 컨텍스트 유지를 위해 가져가는 게 좋음 (선택).

### 2-4. Mac.Air에서 설치할 것

| 항목 | 필수도 | 비고 |
|---|---|---|
| Python 3.10+ (arm64 네이티브) | ★★★ | `python3 --version` + `python3 -c "import platform; print(platform.machine())"` 가 `arm64` 떠야 정상 |
| git | ★★★ | 보통 이미 있음. `git --version` 확인. 없으면 자동 prompt 또는 Xcode CLI |
| Magic Formula `requirements.txt` | ★★★ | pandas, numpy, pyyaml, pyarrow |
| longlivevault `requirements.txt` | ★★★ | pandas, pyarrow, requests, python-dotenv, pykrx, exchange-calendars, yfinance, pytest |
| Xcode CLI | ★ 조건부 | git 이 없거나 Homebrew/pyenv 를 새로 깔 때만. VS Code 가 이미 있고 git 이 동작하면 스킵 |
| Homebrew | ★ | pyenv 쓸 때만 |

> **Xcode CLI 의 진짜 역할** : VS Code 같은 에디터 UI 와 별개로, macOS 시스템 컴파일러/git/Homebrew 의존성. 클로이 코딩 자체에는 직접 필요 없음. `git --version` 과 `python3 --version` 이 둘 다 동작하면 명시적 설치 단계는 건너뛰어도 됨.

### 2-5. 경로 의존성 — 사용자명/폴더명 유지

코드에 **하드코딩된 절대경로** 가 있음. Mac.Air에서도 동일하게 유지하면 코드 수정 0:

```
~/DriveForALL/StoLab/Magic Formula/
~/DriveForALL/StoLab/longlivevault/
```

확인된 하드코딩 위치:

- `scripts/run_analysis.py:23` — `_VAULT_PATH = Path("/Users/kaneyoun/DriveForALL/StoLab/longlivevault")`

사용자명이 바뀐다면 위 경로 1곳을 치환해야 함. (launchd plist 의 경로들은 시나리오 A 에서는 Mac.Air에 안 등록하므로 무시.)

---

## 3. STEP A — Mac.Air 초기 셋업 (한 번만, 필수)

### 3-1. 시스템 준비

```bash
# 미리 확인
git --version
python3 --version
python3 -c "import platform; print(platform.machine())"   # arm64 떠야 함

# (선택) 위가 없거나 Intel python 이면 pyenv 로 정리
# brew install pyenv
# pyenv install 3.10.14 && pyenv global 3.10.14
```

### 3-2. 저장소 클론

```bash
mkdir -p ~/DriveForALL/StoLab
cd ~/DriveForALL/StoLab

git clone https://github.com/kane20c21c/magicformula.git "Magic Formula"
git clone https://github.com/kane20c21c/longlivevault.git
```

### 3-3. 데이터 카피 (parquet 2개)

Mac.Mini에서 미리 묶기:

```bash
# Mac.Mini (Intel)
cd ~/DriveForALL/StoLab/longlivevault
mkdir -p ~/Desktop/llv_for_m4/ohlcv/tickers
cp data/ohlcv/core.parquet           ~/Desktop/llv_for_m4/ohlcv/
cp data/ohlcv/tickers/KOSPI.parquet  ~/Desktop/llv_for_m4/ohlcv/tickers/
# 선택: 분석 컨텍스트
cp -r "../Magic Formula/output/analysis/2026-05-18" ~/Desktop/llv_for_m4/analysis_snapshot
```

AirDrop / 외장하드 / iCloud 로 Mac.Air에 옮기고:

```bash
# Mac.Air (M4)
cd ~/DriveForALL/StoLab/longlivevault
mkdir -p data/ohlcv/tickers
cp ~/Downloads/llv_for_m4/ohlcv/core.parquet           data/ohlcv/
cp ~/Downloads/llv_for_m4/ohlcv/tickers/KOSPI.parquet  data/ohlcv/tickers/

# 분석 컨텍스트 (선택)
mkdir -p "../Magic Formula/output/analysis"
cp -r ~/Downloads/llv_for_m4/analysis_snapshot "../Magic Formula/output/analysis/2026-05-18"
```

### 3-4. 가상환경 + 의존성

```bash
cd ~/DriveForALL/StoLab/longlivevault
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

cd "../Magic Formula"
ln -s ../longlivevault/.venv .venv   # vault 와 같은 venv 공유
source .venv/bin/activate
pip install -r requirements.txt
```

### 3-5. 검증 — 분석 트랙 스모크 테스트

```bash
# (1) vault import & 코어 종목 OHLCV 읽기 — KIS 호출 없이 parquet 만으로
python3 -c "from stolab_data.data_service import get_ohlcv; \
            df = get_ohlcv('000660', '20240101', '20240131'); \
            print(df.tail()); print('rows:', len(df))"
# → SK하이닉스 1월 OHLCV 가 떠야 정상

# (2) KOSPI 읽기 (알파용)
python3 -c "from stolab_data.ohlcv_store import get_ohlcv; \
            print(get_ohlcv('KOSPI', '20240101', '20240131').tail())"

# (3) Magic Formula 분석 트랙 quick-test (5~10분)
cd ~/DriveForALL/StoLab/"Magic Formula"
python scripts/run_analysis.py --quick-test
ls -la output/analysis/
# → 오늘 날짜 폴더 생성되면 정상
```

### 3-6. launchd 등록은 **하지 않음**

시나리오 A 에서는 Mac.Air에 launchd 안 등록한다. Mac.Mini이 데일리를 계속 책임진다. (단, Mac.Mini의 launchd 는 그대로 두는 것이 핵심.)

---

## 4. STEP B — ★ 이번 작업의 본 목표 ★ : M4 가속화

### 4-1. 미션

Mac.Mini에서 51조합 × 3규칙 = 153회 백테스트가 너무 느렸음. M4 의 멀티코어 + 16GB 메모리를 활용해서 **더 큰 조합 수 / 더 자주 백테스트** 가 가능한 환경을 만든다. 이것이 Mac.Air 위에서 작업하는 진짜 이유.

### 4-2. 작업 순서 — 단계별로 측정하면서 진행

#### (1) 베이스라인 측정

```bash
cd ~/DriveForALL/StoLab/"Magic Formula"
time python scripts/run_analysis.py
```

표 한 줄로 기록:

| 환경 | 실행 시간 | 메모리 피크 | 비고 |
|---|---|---|---|
| 옛 Intel i5 (8GB) | (약 5분 소요) | — | 153회 시절 |
| M4 (16GB) 코드 수정 없이 | ? 분 | ? GB | 24회 (현행 코드) |

> 클로이 추정: 코드 수정 없이도 i5 대비 3~5배 (싱글코어 향상 + 메모리 헤드룸 + Accelerate). 실측 필수.

#### (2) Accelerate BLAS 백엔드 적용 — 리스크 낮음

```bash
# conda-forge 사용 시 (권장)
conda install "libblas=*=*accelerate*" numpy

# 또는 venv 안에서 numpy 만 재빌드 (시간 걸림)
pip uninstall numpy
pip install numpy --no-binary numpy
```

다시 측정. 예상 가속 × 1.2~1.5.

#### (3) ★ optimizer 외층 루프 병렬화 (ROI 가장 클 것으로 예상. 코드는 참조용) ★

`magic_formula/optimizer/optimizer.py::run_all()` 의 두 중첩 루프:

```python
for combo in combos:                 # 8개
    scored_data = {ticker: compute_scores(...) for ticker in ...}
    for rule in RULES:                # 3개
        trades, equity = run_simulation(scored_data, rule, ...)
```

→ 24개 독립 단위. M4 코어 (보통 10개) 에 `multiprocessing.Pool` 또는 `joblib.Parallel` 로 분산.

**핵심 설계**:

- 조합 단위로 워커 할당 (8 워커). 각 워커가 자기 조합의 `scored_data` 한 번 계산 + 그 안에서 3 규칙은 순차로 도는 게 가장 단순. 이러면 M4 의 10개 코어 중 8개를 쓰는 셈.
- `scored_data` 는 pickle 가능해야 함 (multiprocessing 의 IPC 통과). pandas DataFrame 은 OK.
- 진행률 출력은 워커마다 print 가 섞이므로 `tqdm.contrib.concurrent.process_map` 이 깔끔.

```python
# 의사 코드
from multiprocessing import Pool

def _run_one_combo(combo_dict, raw_data, kospi_df, trade_start, trade_end):
    label, weights = combo_dict["label"], combo_dict["weights"]
    scored_data = {t: compute_scores(raw_data[t], weights) for t in raw_data if t != "KOSPI"}
    rows = []
    trades_rows = []
    for rule in RULES:
        trades_list, equity_df = run_simulation(scored_data, rule, trade_start, trade_end)
        tdf = trades_to_df(trades_list)
        metrics = compute_metrics(tdf, equity_df, kospi_df, trade_start, trade_end)
        # ... summary row 만들기 + tdf 에 label/rule 컬럼 추가
        rows.append(summary_row)
        if not tdf.empty:
            trades_rows.append(tdf)
    return rows, trades_rows

# run_all 안에서
with Pool(processes=8) as pool:
    results = pool.starmap(_run_one_combo,
                           [(c, raw_data, kospi_df, ts, te) for c in combos])
```

예상 가속 × 5~8. M4 코어 8개 활용 가능.

#### (4) run_simulation 내층 루프 병렬화 — 선택, 주의

```python
for ticker, df in scored_data.items():    # ~69개 종목, 독립
    trades, eq = simulate_ticker(...)
```

외층(3) 을 이미 했다면 추가 효과 작음. Mac.Mini (코어 4개) 또는 조합 수가 8개 미만으로 줄어들 때만 의미 있음. M4 + 8조합 환경에선 (3) 만으로 충분.

#### (5) Polars 마이그레이션 — 큰 작업, 신중

pandas → polars 로 전환하면 단일 코어에서도 추가 × 2~3 가속. 다만 `indicator_calculator.py` (vault) / `scoring/scorer.py` / `simulator/simulator.py` 전부 손봐야 함. (3) 까지로 만족스러우면 굳이 안 해도 됨.

### 4-3. 회귀 방지 — 결과 동등성 검증 (필수)

병렬화는 트레이드 순서를 흩뜨릴 수 있으니, 매 단계마다 베이스라인과 신 버전의 `summary_df` 가 정렬 후 완전히 동일한지 확인:

```python
import pandas as pd
pd.testing.assert_frame_equal(
    baseline_summary.sort_values(["weight_label","rule"]).reset_index(drop=True),
    new_summary.sort_values(["weight_label","rule"]).reset_index(drop=True),
    check_exact=False, atol=1e-6,
)
```

`all_trades_df` 도 동일하게. 가속화 PR 마다 이 검증을 통과해야 merge.

### 4-4. 부수적 정리 (가속화 작업 중에 같이)

- `optimizer.py` docstring/리포트 텍스트의 "51조합 × 3규칙 = 153회" 혹은 8조합 × 3규칙 = 24회은 기존 방식으로 새로운 조합으로 테스트 해야 함.
- `run_simulation` docstring 의 "10개 종목" 도 현행 universe 크기로 갱신.

### 4-5. 가속화 결과를 Mac.Mini에도 적용해야 하나?

Mac.Mini은 **데일리 트랙** 만 도므로 가속화 PR 의 영향이 거의 없음. 다만 새로 생성되는 코드는 별로 관리하여 Mac,Mini용 코드와 충돌하지 않도록 하고, 분석 과정과 주요 결과값 그리고, strategy.yaml을 저장해서 Mac.Mini에서 참조할 수 있도록 함.

```bash
# Mac.Mini에서, git pull 후
python scripts/daily_signal.py
# → 에러 없이 output/signals/daily_signal_*.md 생성되면 OK
```

---

## 5. STEP C — 분석 사이클 운영 (부정기, Kane 이 생각날 때)

월 1회로 못 박지 않고, Kane 이 "이번 달은 분석 한 번 돌려볼까" 싶을 때 아래 순서대로.

### 5-1. 데이터 동기화 — Mac.Mini → Mac.Air

```bash
# Mac.Mini (최신 vault batch 이후, 예: 평일 20:31 ~ 다음 날 새벽 사이)
cd ~/DriveForALL/StoLab/longlivevault
rsync -av data/ohlcv/core.parquet           m4-host:~/DriveForALL/StoLab/longlivevault/data/ohlcv/
rsync -av data/ohlcv/tickers/KOSPI.parquet  m4-host:~/DriveForALL/StoLab/longlivevault/data/ohlcv/tickers/
# (rsync 가 어려우면 AirDrop / 클라우드 드라이브로 대체)
```

> ⚠️ vault 가 Mac.Mini에서만 도니까, 분석 시점 = Mac.Mini core.parquet 을 분석 시점으로 갱신이 되었는지 확인하고 분석 시점과 데이터의 마지막 일자가 차이가 있을 경우, Kane에게 확인하고 진행 해야 함. 

### 5-2. 분석 실행 (Mac.Air)

```bash
cd ~/DriveForALL/StoLab/"Magic Formula"
git pull                                           # yaml/코드 최신화
source .venv/bin/activate
python scripts/run_analysis.py                     # 풀 백테스트 (STEP B 가속화 적용된 상태)
# 결과: output/analysis/YYYY-MM-DD/
#   - weight_ranking.md
#   - summary_*.csv
#   - trades_*.csv
#   - backtest_report.md
#   - equity_curves.png (matplotlib 있을 때)
```

### 5-3. 결과 검토 → yaml 업데이트

```bash
# 자동 — weight_ranking.md 1위 적용
python scripts/update_strategy.py \
    --from-ranking output/analysis/YYYY-MM-DD/weight_ranking.md \
    --strategy-id CompR-YYYYMM

# 변경 사항 미리 확인
python scripts/update_strategy.py \
    --from-ranking output/analysis/YYYY-MM-DD/weight_ranking.md \
    --strategy-id CompR-YYYYMM --dry-run

# 또는 수동 가중치
python scripts/update_strategy.py \
    --weights "trend=0.23,momentum=0.22,volume=0.27,volatility=0.10,wyckoff=0.18" \
    --rule R3
```

기존 yaml 은 `configs/history/YYYY-MM-DD_<strategy_id>.yaml` 로 자동 백업.

### 5-4. Mac.Mini sync — git 으로

```bash
# Mac.Air
cd ~/DriveForALL/StoLab/"Magic Formula"
git add configs/active_strategy.yaml configs/history/
git add -f output/analysis/YYYY-MM-DD/   # .gitignore 되어 있어 -f 필요
git commit -m "analysis(YYYY-MM): <strategy_id> 적용 — alpha +XX.X%"
git push

# Mac.Mini
cd ~/DriveForALL/StoLab/"Magic Formula"
git pull
# → 다음 평일 20:40 launchd 가 새 yaml 로 데일리 시그널 생성
```

> ⚠️ **반드시 확인**: Mac.Mini에서 git pull 후, 다음 데일리 시그널의 `output/signals/daily_signal_*.md` 헤더에 새 strategy_id / 가중치가 반영됐는지 눈으로 검증.

### 5-5. 운영 흐름 요약 — 한 사이클

```
[1] Mac.Mini → Mac.Air     : rsync core.parquet + KOSPI.parquet
[2] Mac.Air              : git pull → run_analysis.py → 결과 검토
[3] Mac.Air              : update_strategy.py → git commit + push
[4] Mac.Mini              : git pull
[5] 다음 평일 20:40      : Mac.Mini launchd 가 새 yaml 로 데일리 생성 (자동 검증)
```

---

## 6. 다음 클로이에게 한 마디

> Kane 은 호칭으로 본인을 **Kane**, 클로이는 **클로이** 로 부르길 원해. 가족처럼 따뜻한 톤 유지. 작업 요구 받으면 **순서를 먼저 확인** 하고 진행. 자료 못 찾거나 불확실하면 **"클로이 의견/추측"** 이라고 명시. 시간은 **KST (GMT+9)** 기준. 숫자 색상/포맷 규칙은 부모 CLAUDE.md (글로벌) 참고.

Mac.Air은 **분석 워크벤치** 야. 데일리 운영은 Mac.Mini 책임. Mac.Air에 launchd 등록 금지 (두 머신이 같이 데일리 만들면 데이터 중복).

**우선 순위는 명확해**:

1. **STEP A** — 셋업. 검증 3-5 의 (1)(2)(3) 이 통과되면 OK.
2. **STEP B** — 이번 작업의 본 목표. 베이스라인 측정 → (2) Accelerate → (3) 외층 병렬화 순서로. 각 단계마다 회귀 검증 (4-3) 통과해야 다음 단계로.
3. **STEP C** — Kane 이 부르면. 미리 자동화하지 말 것. 분석을 정기로 돌리고 싶다는 신호가 오기 전까지는 launchd / cron 등록 금지.

STEP B 의 (3) 외층 병렬화가 가장 큰 한 방. 거기에 에너지 집중하면 돼. (4)(5) 는 보너스.

— 2026-05-28, 클로이 (Intel 마지막 날 새벽)
