# swing/ — 스윙 포트 운영 규칙 (운영 정본)

2026-11-15 케인 — **한 방향 원칙**: 운영은 연구 폴더(`korean_mkt_study/`)를 직접 읽지 않는다.
연구가 확정한 규칙은 이 폴더의 코드로 반영(커밋)되고, 운영은 여기만 읽는다. 연구는 이 코드를 기준선으로 import 해 비교 실험을 한다.

| 파일 | 역할 | 소비자 |
|---|---|---|
| `strategy_reference.py` | 진입 신호·유니버스 판정·R²·YZR 제외 필터 (순수 함수) | SP `app/paper/strategy.py`(가상계좌·실계좌 손절 워치), `shadow_track.py`, 연구 `stop_research.py` 등 |
| `shadow_track.py` | 직전 모델 그림자 추적 — 팩트 체인 5단계 `kms-shadow-track` (평일 20:30 체인) | SP `scripts/infra/fact_store_chain.sh` |
| `state/` | 그림자 상태·자산 로그 (**git 밖, 미니 런타임**) | `shadow_track.py` |

- 청산 규칙 정본은 여기 아니라 SP `app/paper/config.py` 다 (`strategy_reference.py` 머리 주석 참조).
- 유니버스(메가갤럭시) 명단 = LLV `groups/megagalaxy_*.json` (`stolab_data.ticker_groups`). 연구 폴더의 `universe_*.json` 은 읽지 않는다.
- `state/` 는 첫 실행 때 옛 위치 `korean_mkt_study/data/shadow_*` 에서 자동 복사된다(옛 파일은 남는다).
  미니에서 새 위치로 옮겨진 것을 확인한 뒤 `shadow_track.LEGACY_STATE` 와 `_migrate_legacy` 는 지워도 된다.

## 연구 확정 → 운영 반영 절차

1. 연구(`korean_mkt_study/`)에서 이 폴더 코드를 기준선으로 import 해 실험·검증한다. 연구 쪽 사본을 만들지 않는다.
2. 케인 확정 후 **이 폴더 코드를 수정**하고 `STRATEGY_VERSION`·머리 주석 버전 이력을 올린다.
3. 연구 결과의 특정 날짜 신호가 그대로 재현되는지 **회귀 테스트**를 `tests/` 에 추가한다 (`market_layer`·`bounce_rule` 과 같은 방식).
4. 직전 모델은 `shadow_track.MODELS` 에 그림자로 남긴다 (`--init --model <id>`).
5. 커밋 → 케인 push → 미니 pull (+ SP 서버 재시작).
