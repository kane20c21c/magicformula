# -*- coding: utf-8 -*-
"""통일장 시장 층 판정기 파라미터 — 연구 정본 korean_mkt_study regime_* (2026-11-04~11-10) 에서 확정.

⑦ 운영 후보 (케인 2026-11-10 "⑦ 현금화로 운영하고 ⑤는 그림자로 기록만")
  ⑦ = 관문 모드(갭쇼크 & 브레드스 하락 → 현금, 브레드스 상승 → 해제) 위에 평소 = C2
  ⑤ = 같은 관문 모드 위에 평소 = 보유
재료
  시장 브레드스 비율 = (KOSPI+KOSDAQ 상승 종목) ÷ (상승+하락)  — 전 종목, 필터 없음 (케인 11-10 (가))
     상태 = 20일 지수평균 − 직전 250일 중앙값(최소 120일)이 +0.05 이상이면 상승, −0.05 이하면 하락, 사이는 유지
  C2   = 코스피 200 과 KRX 중형 TMI 가 둘 다 200일선 위(괴리형) — 둘 중 하나라도 꺾이는 날 퇴출,
         퇴출 뒤 저점 대비 +10% 반등 또는 괴리형 회복 시 복귀
  갭쇼크 관문 = 코스피 200 하루 −3% 이하 & 그 하락 중 갭(시가/전일종가) 몫 ≥ 40% 인 날로부터 10거래일 창
근거  학습 2016-07~2026-05 · 보류 2026-06~11-04 (K200 −33.8%), 2-d 포트 검정 — 통일장 문서 3-2·3-3장.
⚠ 연구는 동전주 제외 필터 브레드스였다. 운영은 전 종목 (같은 날 값과 다음 날 확정치 정의를 같게) —
  LLV 재료로 보류 구간 106일 ⑦·⑤ 100% 일치, 2025-12~2026-04 는 정의 차이로 일부 갈림 (market_layer_llv_check.py).
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class MarketLayerConfig:
    version: str = "v1.0"
    breadth_span: int = 20            # 비율 지수평균 (ewm span, adjust=False)
    breadth_median_win: int = 250     # 직전 중앙값 창 (거래일)
    breadth_median_min: int = 120     # 중앙값 최소 표본
    breadth_band: float = 0.05        # 히스테리시스 ±
    c2_ma: int = 200                  # 괴리형 이동평균 (단순)
    c2_rebound: float = 0.10          # 퇴출 뒤 저점 대비 복귀 반등
    gate_day_ret: float = -0.03       # 쇼크일 하루 수익률 상한
    gate_gap_share: float = 0.40      # 하락 중 갭 몫 하한 (로그 기준)
    gate_window: int = 10             # 쇼크일 포함 창 (거래일)


DEFAULT = MarketLayerConfig()
