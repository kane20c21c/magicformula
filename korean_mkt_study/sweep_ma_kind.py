"""sweep_ma_kind.py — 스윙 포트 추세선: **SMA vs EVWMA** 비교 (Kane 요청 2026-09-04).

질문
----
현행 진입 규칙의 추세 필터는 `close > SMA(close, 200)` 이다 (v1.2.2.3, W1).
이걸 **EVWMA200** 으로 바꾸면 결과가 어떻게 달라지나?

EVWMA 는 유동물량 V(=n일 거래량 합) 대비 그날 거래된 만큼만 평단이 교체되는
이동평균이다 — 거래가 실린 가격대에 선이 머무르므로, 거래 없이 흘러내린 구간에서는
SMA 보다 선이 늦게 따라온다. "거래가 실린 곳이 진짜 추세선" 이라는 가설.

설계
----
- **기간 스윕 필수** — 200 이라는 좌표는 SMA 곡선에서 찾은 값이고, STRATEGY.md §3 이
  "MA 곡선이 비단조라 200 에 과신 금물" 이라 적어 뒀다. EVWMA 에도 200 이 최적일
  이유가 없으므로 120/150/200/250 을 함께 잰다. SMA 도 같은 격자로 재서 대조군을 만든다.
- **워밍업 2종** — EVWMA 의 유동물량 V 는 min_periods 가 필요하다.
    · 엄격(=n)  : LLV 운영 컬럼 EVWMA_200 과 같은 정의. 배선 시 재계산이 필요 없다.
    · 완화(=140): 현행 SMA200 의 trend_min_periods 와 조건을 맞춘 공정 비교판.
  둘을 다 재야 "차이가 MA 종류에서 왔는지 워밍업 정책에서 왔는지" 가 갈린다.
- **워밍업은 매매 구간 밖으로 뺀다** — 패널을 2010-01 부터 싣고 매매는 2015-01 부터.
  EVWMA 는 첫 유효일 값이 *그날 종가 자체* 라 워밍업 직후엔 `close > EVWMA` 가 거의
  항상 참이 된다. 감쇠 반감기가 약 n·ln2 ≈ 139거래일이므로 5년(≈1,230거래일) 앞두면
  초기값 잔향이 사실상 0 이다.
  ⚠ SMA 결과는 패널 시작일에 **완전히 불변**임을 실측 확인했다 (2010/2011/2013 시작이
    소수점 8자리까지 동일) — 대조군이 이 처리로 유불리를 얻지 않는다.

검증 조건은 backtest.py 머리의 라이브 조건(t+1 시가·왕복 0.23%)을 그대로 쓴다.

실행: python3 sweep_ma_kind.py [--quick]
출력: out/sweep_ma_kind.csv  +  콘솔 표
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

import backtest as B

OUT = Path(__file__).resolve().parent / "out"

# 워밍업을 매매 구간(2015-01~) 밖으로 빼기 위한 패널 시작일. §설계 참조.
PANEL_START = "2010-01-04"

# ⚠ 40~300 까지 넓게 재는 이유 — 120/150/200/250 은 SMA 곡선을 탐색하며 굳은 격자다.
#   그 안에서만 재고 "EVWMA 가 졌다" 고 적으면 상대에게 유리한 창을 안 준 셈이 된다.
#   실제로 EVWMA 는 기간이 짧을수록 나아지는 경향이라 짧은 쪽 확인이 특히 필요했다.
PERIODS = (40, 60, 90, 120, 150, 200, 250, 300)

# 완화 워밍업은 **현행 비율을 기간에 맞춰 늘린 값**이다 — 현행 정본이 200/140 이므로 0.7.
# 고정 140 으로 두면 n=120 에서 min_periods > window 라 pandas 가 거부하고, n=250 에서는
# 완화 정도가 달라져 기간 축과 워밍업 축이 섞인다.
RELAXED_RATIO = 0.7


def _relaxed(n: int) -> int:
    return max(2, round(n * RELAXED_RATIO))


def _cases(quick: bool) -> list[tuple[str, dict]]:
    """(라벨, EntryParams kwargs) 목록. 현행 정본이 맨 앞."""
    out: list[tuple[str, dict]] = []
    periods = (200,) if quick else PERIODS

    for kind in ("sma", "evwma"):
        for warm in ("relaxed", "strict"):
            for n in periods:
                mp = _relaxed(n) if warm == "relaxed" else n
                star = " ★현행" if (kind == "sma" and warm == "relaxed" and n == 200) else ""
                tag = f"워밍업{mp}" + ("엄격" if warm == "strict" else "")
                out.append((f"{kind.upper()}{n} ({tag}){star}",
                            dict(trend_ma_kind=kind, trend_ma=n, trend_min_periods=mp)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="기간 200 만 (연결 확인용)")
    args = ap.parse_args()

    print(f"패널 적재 {PANEL_START}~ (워밍업 흡수용) …", flush=True)
    p = B.load_panel(B.UniverseParams(), B.VolScaleParams(), start=PANEL_START)
    print(f"  종목 {len(p.tickers)} · 거래일 {len(p.dates)}", flush=True)

    rows = []
    for label, kw in _cases(args.quick):
        ep = B.EntryParams(**kw)
        res = B.run_backtest(p, ep)
        m = B.metrics(res)
        rows.append(dict(
            케이스=label, 종류=kw["trend_ma_kind"].upper(), 기간=kw["trend_ma"],
            워밍업=kw["trend_min_periods"],
            최종자산=m["final"], CAGR=m["cagr"], Sharpe=m["sharpe"], MDD=m["mdd"],
            투자비중=m["invested"], 매수=m["n_buy"], 매도=m["n_sell"],
            승률=m["win_rate"], 평균수익=m["avg_ret"], 평균보유=m["avg_hold"],
        ))
        print(f"  {label:26s} 자산 {m['final']:>13,.0f}  CAGR {m['cagr']*100:5.2f}%  "
              f"Sharpe {m['sharpe']:5.2f}  MDD {m['mdd']*100:7.2f}%  "
              f"매수 {m['n_buy']:4d}  승률 {m['win_rate']*100:4.1f}%", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "sweep_ma_kind.csv", index=False)
    print(f"\n저장: {OUT / 'sweep_ma_kind.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
