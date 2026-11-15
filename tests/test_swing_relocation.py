"""swing/ 이전 (2026-11-15 케인 — 한 방향 원칙) 회귀.

① 그림자 상태는 swing/state/ 에 두고, 없으면 옛 위치(korean_mkt_study/data/)에서 한 번 복사한다 (옛 파일 유지).
② 유니버스는 LLV 메가갤럭시 명단만 읽는다 — 비면 예외 (연구 폴더 universe_*.json 폴백 없음).
③ v1112 고정 유니버스 = LLV megagalaxy_2026-07-01.json (MF universe_2026-07_4jo_25pct.json 이관분).
④ 운영 코드(swing/·SP paper)가 korean_mkt_study 를 import 경로로 쓰지 않는다.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "swing"))
import shadow_track as st  # noqa: E402


def test_state_migrates_from_legacy_once(tmp_path, monkeypatch):
    legacy, new = tmp_path / "legacy", tmp_path / "state"
    legacy.mkdir()
    (legacy / "shadow_v1237_state.json").write_text('{"x": 1}', encoding="utf-8")
    (legacy / "shadow_v1237_equity.csv").write_text("date,total\n2026-11-13,1\n", encoding="utf-8")
    monkeypatch.setattr(st, "STATE", new)
    monkeypatch.setattr(st, "LEGACY_STATE", legacy)
    sf, ef = st.files("v1237")
    assert sf == new / "shadow_v1237_state.json" and sf.read_text(encoding="utf-8") == '{"x": 1}'
    assert ef.exists() and (legacy / "shadow_v1237_state.json").exists()      # 옛 파일은 지우지 않는다
    sf.write_text('{"x": 2}', encoding="utf-8")                               # 새 위치가 있으면 다시 덮지 않는다
    st.files("v1237")
    assert sf.read_text(encoding="utf-8") == '{"x": 2}'


def test_state_without_legacy_starts_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "STATE", tmp_path / "state")
    monkeypatch.setattr(st, "LEGACY_STATE", tmp_path / "none")
    sf, ef = st.files("v1238")
    assert not sf.exists() and not ef.exists() and (tmp_path / "state").is_dir()


def _fake_groups(monkeypatch, docs):
    import types
    fake = types.SimpleNamespace(list_docs=lambda kind: [dict(d) for d in docs])
    pkg = types.ModuleType("stolab_data"); pkg.ticker_groups = fake
    monkeypatch.setitem(sys.modules, "stolab_data", pkg)
    monkeypatch.setitem(sys.modules, "stolab_data.ticker_groups", fake)


def test_universe_empty_llv_raises(monkeypatch):
    _fake_groups(monkeypatch, [])
    with pytest.raises(RuntimeError):
        st._universe_docs()


def test_fixed_v1112_points_to_llv_file(monkeypatch):
    assert st.MODELS["v1112"]["universe"] == "fixed:megagalaxy_2026-07-01.json"
    _fake_groups(monkeypatch, [
        {"_file": "megagalaxy_2026-07-01.json", "effective_date": "2026-07-01", "members": ["005930"]},
        {"_file": "megagalaxy_2026-11-16.json", "effective_date": "2026-11-16",
         "stocks": [{"ticker": "000660", "name": "SK하이닉스"}]},
    ])
    import pandas as pd
    doc = st.universe_for("v1112", pd.Timestamp("2026-11-20"))
    assert doc["_file"] == "megagalaxy_2026-07-01.json"
    assert doc["stocks"] == [{"ticker": "005930", "name": "005930"}]       # members 만 있는 문서도 stocks 로


def test_ops_code_does_not_import_research_folder():
    """경로 성분 문자열 "korean_mkt_study" 를 쓰는 코드가 없어야 한다 (예외: 1회용 LEGACY_STATE 복사 경로)."""
    import io, tokenize
    stolab = ROOT.parent
    targets = [ROOT / "swing" / "shadow_track.py", ROOT / "swing" / "strategy_reference.py"]
    sp = stolab / "StockPortfolio" / "app" / "paper"
    if sp.exists():
        targets += sorted(sp.glob("*.py"))
    bad = []
    for f in targets:
        src = f.read_text(encoding="utf-8")
        lines = src.splitlines()
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.STRING and tok.string.strip("\"'rbuf") == "korean_mkt_study":
                line = lines[tok.start[0] - 1]
                if "LEGACY_STATE" not in line:
                    bad.append(f"{f.name}:{tok.start[0]}: {line.strip()}")
    assert not bad, bad
