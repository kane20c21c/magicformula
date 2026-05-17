"""
adaptive_rule_selector.py
=========================

종목별 적응형 진입 규칙 선택 알고리즘.

분석 결과로 도출된 4가지 종목 유형을 점수 시계열 패턴으로 자동 분류하여
각 종목에 가장 적합한 진입 규칙(R1/R2/R3) 또는 SKIP을 추천한다.

종목 유형별 적합 규칙
---------------------
- 직진 폭등형 (점수가 +5 위에 계속 머묾)        → R2 (절대 수준)
- 계단형 추세 (점수가 ±진동하며 평균 양수)        → R3 (부호 전환)
- 임계 돌파형 (점수가 가끔 +5 돌파)              → R1 (임계 돌파)
- 횡보/약세  (평균 점수 0 또는 음수)             → SKIP

핵심 원칙
---------
- look-ahead bias 방지: 백테스트 시작 시점 이전 데이터만 사용
- 재평가 옵션: 주기적(기본 월별)으로 분류 재수행 가능
- 모든 결정은 근거(reason)와 함께 반환

사용 예
-------
>>> from src.signals.adaptive_rule_selector import AdaptiveRuleSelector
>>> selector = AdaptiveRuleSelector()
>>> result = selector.select_rule('000660', score_series_for_sk_hynix)
>>> print(result.selected_rule, result.reason)
R3 부호 전환 8회 + 평균 +2.34 > 0 (계단형 추세)

작성: 클로이 / Kane
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config & Result
# ---------------------------------------------------------------------------


@dataclass
class RuleSelectionConfig:
    """규칙 선택 알고리즘 설정값."""

    lookback_days: int = 60
    """분류에 사용할 최근 거래일 수. 기본 60일."""

    min_data_ratio: float = 0.5
    """lookback 기간 중 최소 데이터 보유 비율. 미달 시 SKIP."""

    score_threshold: float = 5.0
    """진입 점수 임계값 (백테스트의 +5와 동일하게)."""

    avg_score_min_skip: float = 0.0
    """이 값 미만의 평균 점수는 SKIP (약세 종목 회피)."""

    above_threshold_ratio_r2: float = 0.50
    """이 비율 이상으로 점수가 임계값 위에 머무르면 R2."""

    above_threshold_ratio_r1: float = 0.20
    """이 비율 이상으로 점수가 임계값 위에 머무르면 R1."""

    sign_change_min_r3: int = 6
    """R3 분류를 위한 부호 전환 최소 횟수."""


@dataclass
class RuleSelectionResult:
    """규칙 선택 결과."""

    ticker: str
    selected_rule: str  # 'R1', 'R2', 'R3', 'SKIP'
    confidence: float
    metrics: dict = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core Algorithm
# ---------------------------------------------------------------------------


class AdaptiveRuleSelector:
    """종목별 진입 규칙 적응형 선택기."""

    def __init__(self, config: Optional[RuleSelectionConfig] = None):
        self.config = config or RuleSelectionConfig()

    # ---- 지표 계산 ----------------------------------------------------------

    def compute_metrics(self, scores: pd.Series) -> Optional[dict]:
        """점수 시계열에서 분류용 지표 산출.

        Parameters
        ----------
        scores : pd.Series
            거래일 인덱스의 종합 점수 시계열 (-10 ~ +10 범위 가정).
            NaN은 제거 후 계산.

        Returns
        -------
        dict or None
            데이터 부족 시 None 반환.
        """
        recent = scores.tail(self.config.lookback_days).dropna()

        # 데이터 부족 → 분류 불가
        min_required = int(self.config.lookback_days * self.config.min_data_ratio)
        if len(recent) < min_required:
            return None

        threshold = self.config.score_threshold
        values = recent.values

        avg_score = float(np.mean(values))
        std_score = float(np.std(values, ddof=0))
        above_threshold_ratio = float(np.mean(values >= threshold))
        below_neg_threshold_ratio = float(np.mean(values <= -threshold))

        # 부호 전환: 인접 두 값의 곱이 0보다 작은 경우
        if len(values) >= 2:
            sign_changes = int(np.sum(values[:-1] * values[1:] < 0))
        else:
            sign_changes = 0

        # 추세 기울기: 선형 회귀 (단위는 점수/일)
        if len(values) >= 2:
            x = np.arange(len(values))
            trend_slope = float(np.polyfit(x, values, 1)[0])
        else:
            trend_slope = 0.0

        # 점수가 임계값 위에 머무른 연속 구간의 평균 길이
        above_mask = values >= threshold
        runs = self._consecutive_run_lengths(above_mask)
        avg_run_above = float(np.mean(runs)) if runs else 0.0
        max_run_above = int(max(runs)) if runs else 0

        return {
            "n_days": int(len(values)),
            "avg_score": round(avg_score, 3),
            "std_score": round(std_score, 3),
            "above_threshold_ratio": round(above_threshold_ratio, 3),
            "below_neg_threshold_ratio": round(below_neg_threshold_ratio, 3),
            "sign_changes": sign_changes,
            "trend_slope": round(trend_slope, 4),
            "avg_run_above": round(avg_run_above, 2),
            "max_run_above": max_run_above,
        }

    @staticmethod
    def _consecutive_run_lengths(mask: np.ndarray) -> list:
        """True가 연속되는 구간의 길이 리스트 반환."""
        runs = []
        run = 0
        for v in mask:
            if v:
                run += 1
            else:
                if run > 0:
                    runs.append(run)
                run = 0
        if run > 0:
            runs.append(run)
        return runs

    # ---- 규칙 선택 ----------------------------------------------------------

    def select_rule(self, ticker: str, scores: pd.Series) -> RuleSelectionResult:
        """단일 종목 규칙 선택.

        결정 트리는 위에서 아래로 평가되며, 처음 매칭되는 규칙이 선택됨.

        Decision Tree
        -------------
        1. 데이터 부족              → SKIP
        2. 평균 점수 < 0            → SKIP (약세 회피)
        3. 임계값 위 비율 ≥ 50%      → R2 (직진 폭등형)
        4. 부호 전환 ≥ 6 + 평균>0    → R3 (계단형 추세)
        5. 임계값 위 비율 ≥ 20%      → R1 (임계 돌파형)
        6. 그 외                    → SKIP (패턴 모호)
        """
        cfg = self.config
        metrics = self.compute_metrics(scores)

        if metrics is None:
            return RuleSelectionResult(
                ticker=ticker,
                selected_rule="SKIP",
                confidence=0.0,
                metrics={},
                reason=f"데이터 부족 (lookback {cfg.lookback_days}일의 "
                       f"{int(cfg.min_data_ratio*100)}% 미만)",
            )

        avg = metrics["avg_score"]
        above_ratio = metrics["above_threshold_ratio"]
        sign_chg = metrics["sign_changes"]

        # Rule 2: 약세 회피
        if avg < cfg.avg_score_min_skip:
            return RuleSelectionResult(
                ticker=ticker,
                selected_rule="SKIP",
                confidence=0.85,
                metrics=metrics,
                reason=f"평균 점수 {avg:+.2f} < 0 (약세/횡보 종목, 진입 보류)",
            )

        # Rule 3: R2 (직진 폭등형)
        if above_ratio >= cfg.above_threshold_ratio_r2:
            return RuleSelectionResult(
                ticker=ticker,
                selected_rule="R2",
                confidence=0.90,
                metrics=metrics,
                reason=f"+{cfg.score_threshold:.0f} 이상 머문 비율 {above_ratio:.0%} "
                       f"≥ {cfg.above_threshold_ratio_r2:.0%} (직진 폭등형)",
            )

        # Rule 4: R3 (계단형 추세)
        if sign_chg >= cfg.sign_change_min_r3 and avg > 0:
            return RuleSelectionResult(
                ticker=ticker,
                selected_rule="R3",
                confidence=0.80,
                metrics=metrics,
                reason=f"부호 전환 {sign_chg}회 + 평균 {avg:+.2f} > 0 (계단형 추세)",
            )

        # Rule 5: R1 (임계 돌파형)
        if above_ratio >= cfg.above_threshold_ratio_r1:
            return RuleSelectionResult(
                ticker=ticker,
                selected_rule="R1",
                confidence=0.70,
                metrics=metrics,
                reason=f"+{cfg.score_threshold:.0f} 이상 머문 비율 {above_ratio:.0%} "
                       f"≥ {cfg.above_threshold_ratio_r1:.0%} (임계 돌파형)",
            )

        # Rule 6: 모호 → SKIP
        return RuleSelectionResult(
            ticker=ticker,
            selected_rule="SKIP",
            confidence=0.50,
            metrics=metrics,
            reason=f"명확한 패턴 없음 (평균 {avg:+.2f}, "
                   f"+{cfg.score_threshold:.0f}비율 {above_ratio:.0%}, "
                   f"부호전환 {sign_chg}회)",
        )

    # ---- 일괄 처리 ----------------------------------------------------------

    def select_for_universe(self, scores_dict: dict) -> pd.DataFrame:
        """다수 종목 일괄 규칙 선택.

        Parameters
        ----------
        scores_dict : dict[str, pd.Series]
            {ticker: 점수 시계열} 형태의 딕셔너리.

        Returns
        -------
        pd.DataFrame
            종목별 결과를 한 행으로 정리한 DataFrame.
            columns: ticker, rule, confidence, reason, + metrics
        """
        rows = []
        for ticker, scores in scores_dict.items():
            r = self.select_rule(ticker, scores)
            rows.append({
                "ticker": ticker,
                "rule": r.selected_rule,
                "confidence": r.confidence,
                "reason": r.reason,
                **r.metrics,
            })
        return pd.DataFrame(rows)

    def summary(self, results_df: pd.DataFrame) -> pd.Series:
        """일괄 결과의 규칙 분포 요약."""
        return results_df["rule"].value_counts()


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def select_rules_for_backtest(
    scores_dict: dict,
    config: Optional[RuleSelectionConfig] = None,
) -> pd.DataFrame:
    """편의 함수: 백테스트 직전 한 번에 모든 종목 분류.

    Parameters
    ----------
    scores_dict : dict
        {ticker: 점수 시계열} (백테스트 시작 시점 직전까지의 데이터만 포함)
    config : RuleSelectionConfig, optional
        선택 설정. 미지정 시 기본값 사용.

    Returns
    -------
    pd.DataFrame
    """
    selector = AdaptiveRuleSelector(config)
    return selector.select_for_universe(scores_dict)
