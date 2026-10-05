"""h1_shadow_v1234_check.py — 그림자 v1234(직전 운영 v1.2.3.4) 엔진 검증 (2026-10-05)

가상계좌 백업 상태(기본 9/13 = v1.2.3.4 적용 직전)에서 분기해 그림자 엔진으로 하루씩 전진시키고,
같은 기간 실제 가상계좌(data/paper/trades.csv · equity.csv)와 매수·매도·총자산을 대조한다.
운영 파일(data/shadow_*)은 건드리지 않는다 — 산출은 out/h1/shadow_v1234_check/.
"""
from __future__ import annotations
import contextlib, copy, io, json, sys
from pathlib import Path
import pandas as pd
import shadow_track as T

OUT = T.KMS / "out" / "h1" / "shadow_v1234_check"; OUT.mkdir(parents=True, exist_ok=True)
FORK = sys.argv[1] if len(sys.argv) > 1 else "20260913"
END = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else "2026-10-02")
MODEL = sys.argv[3] if len(sys.argv) > 3 else "v1234"

_orig_load = T.load_prices
CUT = {"d": None}
def load_cut(tickers, start="2025-06-01"):
    px = _orig_load(tickers, start)
    return {k: v[v.index <= CUT["d"]] for k, v in px.items()} if CUT["d"] is not None else px
T.load_prices = load_cut
T.files = lambda model: (OUT / f"{model}_state.json", OUT / f"{model}_equity.csv")
T.SP_PAPER_STATE = None


def fork_state():
    src = T.SP_PAPER / "backups" / f"state_{FORK}.json"
    real = T.SP_PAPER / "state.json"; keep = real.read_text(encoding="utf-8")
    # init_state 는 SP_PAPER/state.json 을 읽으므로, 읽는 경로만 임시로 바꾼다 (운영 파일은 쓰지 않는다)
    sp_dir = OUT / "_sp"; sp_dir.mkdir(exist_ok=True); (sp_dir / "state.json").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    old = T.SP_PAPER; T.SP_PAPER = sp_dir
    for f in OUT.glob(f"{MODEL}_*"): f.unlink()
    with contextlib.redirect_stdout(io.StringIO()):
        st = T.init_state(MODEL)
    T.SP_PAPER = old
    assert real.read_text(encoding="utf-8") == keep
    return st


def step(st, d):
    CUT["d"] = d
    with contextlib.redirect_stdout(io.StringIO()):
        return T.advance(MODEL, st)


def main():
    st = fork_state(); last = pd.Timestamp(st["last_processed"])
    CUT["d"] = None; days = [d for d in _orig_load(["005930"])["close"].index if last < d <= END]
    # 분기일 신호 큐 복원: 분기일 하루를 버리는 복사본으로 다시 돌려 pending_buys 만 가져온다
    prev = [d for d in _orig_load(["005930"])["close"].index if d < last][-1]
    tmp = copy.deepcopy(st); tmp["last_processed"] = prev.strftime("%Y-%m-%d")
    st["pending_buys"] = step(tmp, last)["pending_buys"]
    print(f"분기 {FORK} (기준일 {last.date()}) · 보유 {len(st['positions'])}종목 · 큐 {[q['ticker'] for q in st['pending_buys']]} · {len(days)}거래일 전진")
    names = {}; ev = []
    for d in days:
        before = {k: dict(v) for k, v in st["positions"].items()}
        st = step(st, d); after = st["positions"]; ds = d.strftime("%Y-%m-%d")
        for tk in before:
            names[tk] = before[tk].get("name", "")
            if tk not in after: ev.append((ds, tk, "SELL"))
            elif after[tk]["shares"] > before[tk]["shares"]: ev.append((ds, tk, "BUY"))
        for tk in after:
            names[tk] = after[tk].get("name", "")
            if tk not in before: ev.append((ds, tk, "BUY"))
    sh = pd.DataFrame(ev, columns=["date", "ticker", "side"])
    real = pd.read_csv(T.SP_PAPER / "trades.csv", dtype={"ticker": str})
    real = real[(real.date > last.strftime("%Y-%m-%d")) & (real.date <= END.strftime("%Y-%m-%d"))][["date", "ticker", "name", "side", "reason", "price"]]
    for _, r in real.iterrows(): names[r.ticker] = r["name"]
    sh["k"] = sh.ticker + sh.side; real["k"] = real.ticker + real.side
    m = sh.merge(real, on=["ticker", "side"], how="outer", suffixes=("_그림자", "_실제"))
    m["name"] = m.ticker.map(names); m["같은날"] = m.date_그림자 == m.date_실제
    m = m.sort_values(["ticker", "date_실제", "date_그림자"])[["name", "side", "date_그림자", "date_실제", "reason", "같은날"]]
    pd.set_option("display.width", 200); print(m.to_string(index=False))
    both = m.dropna(subset=["date_그림자", "date_실제"])
    print(f"\n그림자 {len(sh)}건 · 실제 {len(real)}건 · 짝 {len(both)}건 중 같은 날 {int(both.같은날.sum())}건")
    eq = pd.read_csv(OUT / f"{MODEL}_equity.csv")[["date", "total", "n_pos"]].merge(
        pd.read_csv(T.SP_PAPER / "equity.csv").iloc[:, [0, 4, 5]].set_axis(["date", "실제", "실제_n"], axis=1), on="date", how="left")
    eq["차이%"] = (eq.total / eq.실제 - 1) * 100
    print(eq.round(2).to_string(index=False)); m.to_csv(OUT / "trades_compare.csv", index=False, encoding="utf-8-sig")
    print("그림자 보유:", {v.get("name", k): v["entry_date"] for k, v in st["positions"].items()})


if __name__ == "__main__":
    main()
