# Magic Formula — Tests

pytest 기반 단위 테스트.

## 실행

```bash
# 전체
pytest tests/

# 특정 파일
pytest tests/test_vault.py

# verbose + 첫 실패에서 중단
pytest tests/ -v -x
```

## 커버리지

| 파일 | 대상 모듈 | 검사 항목 |
|---|---|---|
| `test_vault.py` | `magic_formula._vault` | 종목명 강건성 (None/NaN/공백/`<NA>`), universe 크기, sector 매핑, LIG 사명변경 반영 |
| `test_config.py` | `magic_formula.config` | yaml load/dump round-trip, 검증 실패 케이스 (합≠1.0, 잘못된 rule 등), history 백업 |
| `test_signals.py` | `magic_formula.daily.runner.check_r1_signal` | R1 상향 돌파 / 이미 위 / 하락 / 경계값 / 데이터 부족 |

## 향후 추가 권장

- `test_scoring.py` — 5영역 점수 산출 (look-ahead bias 방지 검증 포함)
- `test_simulator.py` — 1종목 e2e (synthetic OHLCV → 가짜 진입 → 청산 → metrics)
- `test_update_strategy.py` — parse_ranking_md 의 다양한 행 포맷

## 환경 의존 테스트

`test_config.test_real_active_strategy_loads` 는 실제 `configs/active_strategy.yaml` 을 읽는다. 해당 파일이 없으면 자동 skip.

vault 데이터에 의존하는 테스트는 의도적으로 작성하지 않았다 (vault 미설치 환경에서도 패스해야).
