# Magic Formula — launchd 자동화

vault 와 동일한 launchd 방식으로 Magic Formula 데일리 시그널을 자동 실행한다.

## 파일

- `com.stolab.magic-formula.daily-signal.plist` — 평일 **16:30 + 20:40 KST 하루 2회** 데일리 시그널 자동 실행 (코어 69)
  - **16:30 1차판** — LLV 1차 KIS 종가 배치(16:00) 직후. 16:00 배치가 당일 종가 기준 Wyckoff(국면/시그널)까지 재계산해 parquet 에 적재하므로, 16:30 판에도 당일 종가 기준 Wyckoff 국면이 들어간다. 장 마감 직후 빠른 1차 신호.
  - **20:40 확정판** — LLV 2차 KIS 종가 배치(20:30) 직후. Wyckoff 계산 로직은 16:00 과 동일하나, KIS 종가가 20:30 에 확정 종가로 보정되면 그 입력 차이만큼 라벨이 달라질 수 있어 최종본. 같은 파일을 덮어쓴다.

- `com.stolab.magic-formula.daily-extended-signal.plist` — 평일 **16:35 + 20:45 KST 하루 2회** 확장(extend) 시그널 자동 실행 (시총 200 = core ∪ extend)
  - 코어 잡(16:30/20:40) **5분 뒤** 실행. universe="extended_all" 로 `daily_extended_signal_*.{json,md}` + `daily_extended_regimes_*.json` 산출. 코어 산출물은 덮어쓰지 않음.
  - 전제: LLV kis_update(16:00/20:30) 가 extend.parquet 까지 적재한 뒤 실행되어야 정확한 Wyckoff 라벨.

### 확장 잡 등록 (최초 1회)

```bash
ln -sf "/Users/kaneyoun/DriveForALL/StoLab/Magic Formula/configs/launchd/com.stolab.magic-formula.daily-extended-signal.plist" \
       ~/Library/LaunchAgents/com.stolab.magic-formula.daily-extended-signal.plist
launchctl load ~/Library/LaunchAgents/com.stolab.magic-formula.daily-extended-signal.plist
launchctl list | grep magic-formula        # daily-signal + daily-extended-signal 둘 다 보여야 정상
```
수동 테스트: `launchctl start com.stolab.magic-formula.daily-extended-signal`
또는 `python "/Users/kaneyoun/DriveForALL/StoLab/Magic Formula/scripts/daily_extended_signal.py"`. 로그: `output/logs/daily_extended_signal.{out,err}`.

## 등록 (최초 1회)

```bash
# 1) ~/Library/LaunchAgents/ 로 심볼릭 링크
ln -sf "/Users/kaneyoun/DriveForALL/StoLab/Magic Formula/configs/launchd/com.stolab.magic-formula.daily-signal.plist" \
       ~/Library/LaunchAgents/com.stolab.magic-formula.daily-signal.plist

# 2) launchd 에 로드
launchctl load ~/Library/LaunchAgents/com.stolab.magic-formula.daily-signal.plist

# 3) 등록 확인
launchctl list | grep magic-formula
```

심볼릭 링크를 사용하면 plist 수정 시 별도 작업 없이 즉시 반영된다 (단, unload→load 한 번 필요).

## 제거

```bash
launchctl unload ~/Library/LaunchAgents/com.stolab.magic-formula.daily-signal.plist
rm ~/Library/LaunchAgents/com.stolab.magic-formula.daily-signal.plist
```

## 수동 실행 (테스트)

등록된 작업을 즉시 한 번 실행하려면:

```bash
launchctl start com.stolab.magic-formula.daily-signal
```

또는 그냥 스크립트 직접:

```bash
python "/Users/kaneyoun/DriveForALL/StoLab/Magic Formula/scripts/daily_signal.py"
```

## 로그 확인

launchd 가 캡처한 stdout / stderr 는 다음 위치에 누적된다:

```
output/logs/daily_signal.out
output/logs/daily_signal.err
```

스크립트 자체가 또한 `output/signals/daily_signal_YYYYMMDD.json` / `.md` 에 결과를 저장한다.

## 트러블슈팅

### "python3 not found" 또는 ImportError

plist 의 `EnvironmentVariables.PATH` 가 사용자의 실제 python3 위치를 포함하는지 확인.
주로 확인할 곳:

- Homebrew (Apple Silicon): `/opt/homebrew/bin/python3`
- Homebrew (Intel): `/usr/local/bin/python3`
- pyenv: `~/.pyenv/shims/python3`
- 시스템 기본: `/usr/bin/python3`

`which python3` 결과를 확인하고, 그 경로의 부모 디렉토리를 PATH 에 추가하면 된다.

### 실행 시각이 어긋남

`Hour` / `Minute` 는 시스템 로컬 타임존 기준. macOS 가 KST(Asia/Seoul) 로 설정돼 있으면 20:40 이 곧 KST 20:40.

```bash
# 타임존 확인
sudo systemsetup -gettimezone
```

### 평일에 안 돌아감

```bash
# 다음 실행 시각 확인
launchctl list com.stolab.magic-formula.daily-signal
```

`PID` 가 `-` 이고 `Status` 가 `0` 이면 등록만 되고 아직 실행 안 됨 — 정상. 평일 20:40 이 되면 실행.

## 변경 사항 반영

plist 를 수정한 후에는:

```bash
launchctl unload ~/Library/LaunchAgents/com.stolab.magic-formula.daily-signal.plist
launchctl load   ~/Library/LaunchAgents/com.stolab.magic-formula.daily-signal.plist
```

`launchctl reload` 명령은 macOS Sequoia 이후 지원되지만, unload→load 가 가장 호환성 좋음.
