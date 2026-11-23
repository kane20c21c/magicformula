"""기준선: 선행 종목 대신 KOSPI200 자기 10일 수익률 부호로 같은 슬롯 규칙을 돌린다 (2026-11-23).
  ±2 판 = 매일 '3개 일치'처럼 2슬롯 이동 / ±1 판 = 매일 1슬롯 이동.  사용: python3 k200_lead_sim_bench.py START END"""
import os, sys, numpy as np, pandas as pd
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2], "218410,218410,218410", "bench"]   # 틀만 빌림
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "k200_lead_sim.py")).read().split("sig = [signal")[0]
exec(src)
k10 = np.sign(np.log(k / k.shift(10)))
base = np.array([k10[days[i - 1]] for i in range(1, len(days))]).astype(int)
rng = np.random.default_rng(20261123)
for nm_, mv in [("KOSPI200 자기 10일 ±2", base * 2), ("KOSPI200 자기 10일 ±1", base)]:
    fin, fees, lg = run(mv); ret = fin / (20 * SLOT) - 1
    null = np.array([run(rng.permutation(mv))[0] for _ in range(1000)]) / (20 * SLOT) - 1
    print(f"  {START}~{END} {nm_}: {fin:,.0f}원 ({ret:+.1%}) 수수료 {fees:,.0f}원 | 섞기 중앙 {np.median(null):+.1%}, 실제 이상 {(null >= ret).mean():.1%}")
