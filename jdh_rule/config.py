"""jdh_rule/config.py — 장단현포트 판정 상수 (정본). 근거 = 2026-11-23 연구
(연구 스크립트 k200_lead_walkforward.py · k200_regime_switch.py · k200_regime_switch_timing.py — 운영 코드는 연구 폴더를 읽지 않는다).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RULE_VERSION = "v1.1.1.1"          # 모델.유니버스.진입.청산 — 첫 판 (2026-11-23)

PROJECT_ROOT = Path(__file__).resolve().parents[1]           # MagicFormula/
STOLAB_ROOT = PROJECT_ROOT.parent
LLV_ROOT = STOLAB_ROOT / "LongLiveVault"
PANEL = LLV_ROOT / "data" / "ohlcv" / "panel.parquet"
KOSPI200_PARQUET = LLV_ROOT / "data" / "ohlcv" / "tickers" / "KOSPI200.parquet"
REGIME_HISTORY = PROJECT_ROOT / "output" / "market_layer" / "regime_history.parquet"
PICKS_DIR = PROJECT_ROOT / "output" / "jdh"                   # picks_{유니버스 적용일}.json
SIGNAL_DIR = PROJECT_ROOT / "output" / "signals_jdh"          # YYYYMMDD.json (15:10 잠정 판정)


@dataclass(frozen=True)
class JdhConfig:
    # ── 선행종목 선정 (유니버스 조정일마다, 케인 2026-11-23) ──
    sel_ret_days: int = 10         # 선정에 쓰는 종목 수익률 기간 (연구: 선정까지 5일로 바꾸면 나빠짐)
    n_blocks: int = 6              # 직전 240거래일 = 40일 × 6구간 (≈12개월)
    block_len: int = 40
    min_block_obs: int = 30        # 구간당 최소 관측 (0 제외 후)
    recent_blocks: int = 4         # 순위 = 최근 4구간 균형적중률 평균
    n_leads: int = 3
    # ── 매매 신호 (15:10 잠정 = 완전 4일 + 잠정 1일) ──
    sig_ret_days: int = 5
    lead_move_3: int = 2           # 3종목 모두 같은 방향
    lead_move_2: int = 1           # 정확히 2종목
    k200_move: int = 1             # KOSPI200 자기추세 ±1
    # ── ⑦ ──
    p7_max_stale_td: int = 1       # ⑦ 판정일이 직전 거래일보다 낡은 거래일 수 < 이 값이면 사용 (1 = 직전 거래일 판정만). 아니면 틀 유지


DEFAULT = JdhConfig()

__all__ = ["JdhConfig", "DEFAULT", "RULE_VERSION", "PROJECT_ROOT", "LLV_ROOT", "PANEL", "KOSPI200_PARQUET",
           "REGIME_HISTORY", "PICKS_DIR", "SIGNAL_DIR"]
