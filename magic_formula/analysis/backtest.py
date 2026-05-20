"""
magic_formula.analysis.backtest
================================
분석 트랙 진입점 — magic_formula.main 의 CLI 를 그대로 위임한다.

실제 본체 로직(파이프라인, optimizer 호출, 리포트 작성 등) 은
``magic_formula.main.main()`` 에 있으며, 본 모듈은 분석 트랙의
명시적 entrypoint 역할만 한다.

실행
----
    # 패키지 단축 호출
    python -m magic_formula.analysis.backtest

    # CLI 진입점
    python scripts/run_analysis.py

    # 옛 호환 진입점
    python magic_formula/main.py
"""

from __future__ import annotations

from magic_formula.main import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
