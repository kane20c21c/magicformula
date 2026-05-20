"""
tests/conftest.py
-----------------
pytest 공통 설정: Magic Formula 루트를 sys.path 에 추가해서 magic_formula 패키지를 import 가능하게.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
