# 이관 체크리스트 — `MagicFormula/fever_model` → `SpotGauge`

> **작업자: 케인.** StoLab 절대 규칙 7상 어시스턴트는 수정·스모크테스트·`git commit`
> 까지만 하고, `git push` · 미니 `git pull` · `launchctl` · `~/DriveForALL/StoLab` 쓰기는
> 실행하지 않는다. 아래 명령은 **제시**이며 실행은 케인이 한다.
> 이관 완료 후 이 파일과 `README.md` 상단의 이관 경고 블록을 지운다.

**결정**: 2026-08-26 Kane — 별도 리포 + **git 이력 보존**, 이름 = **현물게이지 / SpotGauge**.
분리 근거는 `README.md` §왜 별도 리포인가.

---

## ⚠ 먼저 읽을 것 — 조용히 깨지는 지점 2개

### ① `output/` 은 git 밖에 있다 → subtree split 이 안 가져간다

MagicFormula 루트 `.gitignore` 의 `output/` 이 `fever_model/output/` 까지 덮는다.
2026-08-26 실측 — 추적 파일은 9개뿐이고 일지 3종은 **전부 미추적**이다.

| 파일 | 줄 수 | git |
|---|---:|---|
| 나우캐스트_국면일지.csv | 22,979 | ✗ |
| 탄력점수_일지.csv | 1,672 | ✗ |
| 나우캐스트_온도일지.csv | 127 | ✗ |
| 발열률_전체.csv · 상위10.csv · 그래프.png | — | ✗ (매일 재생성) |
| data/panel_states.csv | — | ✅ 추적 |

**국면일지가 특히 위험하다** — 재가열 플래그가 60거래일 국면 이력을 요구해서
(`daily_WW_wf.py` L532·L534), 유실되면 예외 없이 **플래그만 빈 값으로 나온다**.
그래서 4단계에 `output/` 수동 복사가 들어간다. (유실 시 `--backfill 120` 으로
재구축은 가능하다 — 백필 루프가 국면일지도 같이 쓴다. 단 LLV 재조회가 필요하고
그동안 산출이 빈다. **미실행 — 재구축 동등성은 검증하지 않았다.**)

### ② `STOLAB` 경로식이 폴더 깊이에 묶여 있었다 → **선제 수정 완료**

```
이관 전:  StoLab/MagicFormula/fever_model/src  → dirname×2 = StoLab            ✅
이관 후:  StoLab/SpotGauge/src                 → dirname×2 = StoLab 의 상위     ❌
```

어긋나면 LLV parquet 과 SMTP `.env` 를 못 찾는데, **둘 다 예외가 아니라 무산출·무발송**
으로 끝나 알아채기 어렵다. 2026-08-26 `src/_stolab.py:find_stolab()` 으로 교체했다 —
깊이가 아니라 **형제 프로젝트 존재 여부**로 StoLab 을 판정하므로 이관 전·후 반환값이
같다(실측 확인). 필요하면 `STOLAB_DIR` 환경변수로 덮어쓸 수 있다.

---

## 0. 사전 상태 맞추기

```bash
export STOLAB=<이 기계의 StoLab 루트>       # 미니: ~/DriveForALL/StoLab

cd $STOLAB/MagicFormula
git status                                  # 깨끗해야 한다
git log --oneline -1                        # 미니·에어가 같은 커밋인지 확인
```

⚠ **이관은 운영본(맥미니)에서 한다.** 에어에서 먼저 하면 미니의 launchd 가
사라진 경로를 계속 가리킨다. 미니·에어 둘 다 쓰는 경우 미니 완료 후 에어에서
`git pull` + 폴더 정리만 반복한다.

⚠ **장 시간대를 피한다.** 16:30 산출 / 17:00 메일 사이에 폴더가 사라지면
그날 발열률이 통째로 빈다. **17:30 이후 또는 주말**에 진행할 것.

## 1. subtree split — 커밋 4개 보존

```bash
cd $STOLAB/MagicFormula
git subtree split -P fever_model -b spotgauge-import

# 검증: 커밋이 남아 있고, 트리 루트가 CLAUDE.md/src/... 여야 한다
git log --oneline spotgauge-import
git ls-tree --name-only spotgauge-import
```

기대 커밋 (2026-08-26 기준 4개 + 이번 준비 커밋):
`c31c5d4 발열률 신설` · `0f41ca6 HANDOFF 신설` · `33d8c96 machine-agnostic paths` ·
`fc479dd 작업 경계 규칙 7` · (금일 준비 커밋)

## 2. 새 리포 생성

```bash
mkdir -p $STOLAB/SpotGauge && cd $STOLAB/SpotGauge
git init -b main
git remote add _src $STOLAB/MagicFormula
git fetch _src spotgauge-import
git reset --hard FETCH_HEAD
git remote remove _src

git log --oneline          # 이력이 그대로 넘어왔는지
ls                          # CLAUDE.md README.md MIGRATION.md configs data docs src
```

## 3. 산출물 수동 복사 (git 이 안 가져온 것)

```bash
cp -Rp $STOLAB/MagicFormula/fever_model/output $STOLAB/SpotGauge/

# 검증 — 줄 수가 위 표와 맞아야 한다
wc -l $STOLAB/SpotGauge/output/*.csv
```

이 시점부터 `.gitignore` 의 `!output/*일지.csv` 예외가 발효한다 (리포 루트가 됐으므로).
일지 3종을 실제로 추적할지는 케인 판단 — 추적하면 기계 이동에 안전하지만
매일 diff 가 뜬다. 추적하려면:

```bash
cd $STOLAB/SpotGauge
git status --short          # output/*일지.csv 가 ?? 로 보이는지 확인
git add output/*일지.csv && git commit -m "chore: 일지 3종 추적 시작"
```

## 4. launchd 재등록

```bash
cd $STOLAB/SpotGauge
for L in fever-rule-daily fever-rule-mail; do
  launchctl bootout gui/$(id -u)/com.kane.$L 2>/dev/null
  rm -f ~/Library/LaunchAgents/com.kane.$L.plist
  cp configs/launchd/com.kane.$L.plist ~/Library/LaunchAgents/
  launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.kane.$L.plist
done
launchctl list | grep fever-rule
launchctl print gui/$(id -u)/com.kane.fever-rule-daily | grep -E "state|program|path"
```

⚠ **심링크가 아니라 복사본이다** — plist 를 고칠 때마다 이 블록을 다시 돌려야 한다.
⚠ `Bootstrap failed: 5: Input/output error` 는 대개 라벨이 이미 올라가 있다는 뜻이다.
`launchctl list` 에 보이고 `launchctl print` 가 state/program 을 뱉으면 정상.

## 5. 즉시 검증 (폴더 지우기 **전에**)

```bash
cd $STOLAB/SpotGauge

# ① 경로 해석이 새 위치에서 맞는지 — 여기가 이번 이관의 급소다
python3 -c "
import sys; sys.path.insert(0,'src')
from _stolab import find_stolab
import os
s = find_stolab(os.path.abspath('.'))
print('STOLAB      =', s)
print('LLV parquet =', os.path.isdir(os.path.join(s,'longlivevault','data','ohlcv')))
print('SP .env     =', os.path.exists(os.path.join(s,'StockPortfolio','.env')))
"
#   셋 다 True/정상 경로여야 한다. False 면 STOLAB_DIR 로 덮어쓰고 원인 확인.

# ② 산출 재현 — 같은 날짜 재계산이 기존 값과 일치하는지
/usr/local/bin/python3 src/daily_WW_wf.py --force

# ③ 메일 미리보기 (발송 없음)
/usr/local/bin/python3 src/send_fever_mail.py --dry-run
open output/발열률_메일.html

# ④ launchd 경유 실행
launchctl kickstart -k gui/$(id -u)/com.kane.fever-rule-daily
tail -30 fever-daily.log
```

**합격 기준**: ①이 전부 정상 경로 · ②가 에러 없이 일지 갱신 · ③ HTML 이 종전과
같은 모양 · ④ 로그에 예외 없음. 하나라도 어긋나면 6단계로 넘어가지 말 것.

## 6. MagicFormula 쪽 정리

```bash
cd $STOLAB/MagicFormula
git rm -r fever_model                       # 추적분 제거
rm -rf fever_model                          # output/ 등 미추적 잔여분
rm -f configs/launchd/com.kane.fever-rule-{daily,mail}.plist
git add -A && git commit -m "chore: 발열률(fever_model) SpotGauge 로 분리 — 매매 룰 3개만 남김"
git branch -D spotgauge-import              # 임시 브랜치 정리
```

문서 정리(루트 `CLAUDE.md` · `HANDOFF.md` · `configs/launchd/README.md`)는
**2026-08-26 커밋에서 이미 반영**돼 있다 — 이 단계에서 다시 손댈 필요 없다.

⚠ `StoLab/CLAUDE.md` 의 프로젝트 목록에 현물게이지 추가는 케인 몫
(어시스턴트 쓰기 금지 구역).

## 7. 다음 영업일 확인

- 16:30 이후 `SpotGauge/fever-daily.log` 에 정상 종료
- 17:00 메일 1통 수신 — **오지 않으면 `.env` 탐색 실패를 먼저 의심**
- 아웃퍼포머 8501 **Temp.View** 페이지가 렌더되는지
  ⚠ homalone 이 `발열률_전체.csv` 를 **어느 경로로 읽는지 확인 필요** —
  homalone 리포는 이 세션에서 확인하지 못했다. 하드코딩된
  `MagicFormula/fever_model/output/` 이 있으면 SpotGauge 로 고쳐야 한다.
  **이관 전에 미리 grep 해 둘 것**: `grep -rn "fever_model" $STOLAB/homalone`
- `$STOLAB/_status/fever-rule-{daily,mail}.json` 종료코드 0

## 8. 롤백

5단계 검증 전에는 원본이 그대로 있으므로 되돌리기가 쉽다.

```bash
# launchd 를 옛 경로로 복구
cd $STOLAB/MagicFormula
for L in fever-rule-daily fever-rule-mail; do
  launchctl bootout gui/$(id -u)/com.kane.$L 2>/dev/null
  rm -f ~/Library/LaunchAgents/com.kane.$L.plist
  cp configs/launchd/com.kane.$L.plist ~/Library/LaunchAgents/
  launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.kane.$L.plist
done
rm -rf $STOLAB/SpotGauge
```

6단계까지 간 뒤라면 `git revert` 로 되돌리고 `output/` 을 SpotGauge 에서 복사해 온다.

---

## 완료 후 남는 것

| | 이관 전 | 이관 후 |
|---|---|---|
| MagicFormula | 룰 4개 | **매매 룰 3개** (황금률·데이·스윙) |
| SpotGauge | — | 시장 관측기 1개 |
| launchd 잡 | MagicFormula 7개 | MagicFormula 5개 + SpotGauge 2개 |
