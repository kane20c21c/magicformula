"""jdh_rule — 장단현포트(룰 #6) 판정 로직 (2026-11-23 케인 확정).

통일장 ⑦ 로 틀을 고르고, 틀마다 다른 방향 신호로 KODEX200 / KODEX 인버스 / 현금 슬롯을 옮긴다.
  ⑦ 보유 → 롱/현금 틀 + KOSPI200 자기 5일 수익률 부호 (±1 슬롯)
  ⑦ 현금 → 롱/숏  틀 + 선행 3종목 5일 수익률 합의 (3개 ±2 · 정확히 2개 ±1)
판정만 여기서 한다 — 가상계좌·체결은 StockPortfolio app/jdh. 정본 = README.md.
"""
from jdh_rule.config import JdhConfig, DEFAULT, RULE_VERSION
from jdh_rule.picks import select_leads, ensure_picks, load_picks
from jdh_rule.signal import judge, lead_move, k200_move, latest_p7

__all__ = ["JdhConfig", "DEFAULT", "RULE_VERSION", "select_leads", "ensure_picks", "load_picks",
           "judge", "lead_move", "k200_move", "latest_p7"]
