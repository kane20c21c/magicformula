# -*- coding: utf-8 -*-
"""make_market_layer_fixture.py — market_layer 회귀 고정본 생성 (2026-11-10).

연구 판정(⑦·⑤·브레드스·C2)과 그 입력을 한 CSV 로 묶어 tests/fixtures/market_layer_research.csv 에 쓴다.
운영 모듈 market_layer 가 연구 스크립트 없이도 '연구와 같은 식' 임을 테스트로 지키기 위해서다.
  입력  k_open·k_close (data_1104/kospi200_daily) · tmi (layer_levels KRX 중형 TMI, pykrx)
        adv_res·dec_res (연구 필터 브레드스 out/breadth/breadth_market.csv)
        adv_llv·dec_llv (LLV market_breadth KRX 확정치, 전 종목 — 2025-05-02~)
  정답  br·c2·p7·p5 (연구 detect_states·gate2_states·regime_exposure.judges)
사용  python3 make_market_layer_fixture.py
"""
import sys, warnings
from pathlib import Path
import pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; LLV = HERE.parents[1] / "LongLiveVault"
for p in (HERE, ROOT, LLV):
    sys.path.insert(0, str(p))
OUT = ROOT / "tests" / "fixtures" / "market_layer_research.csv"


def main():
    import regime_expand as RX, regime_exposure as RE
    from stolab_data import market_breadth as mb
    k, o, _ = RX.load_base(); J = RE.judges(k)
    lv = pd.read_parquet(HERE / "out" / "regime" / "layer_levels.parquet")
    S = pd.read_parquet(HERE / "out" / "regime" / "detect_states.parquet")
    br = pd.read_csv(HERE / "out" / "breadth" / "breadth_market.csv", parse_dates=["date"]).set_index("date")
    L = mb.adv_dec(mb.load_breadth(LLV / "data" / "market_breadth.parquet").query("Source == 'KRX'"))
    L = L.groupby("Date")[["AdvAll", "DecAll"]].sum()
    idx = br.index.union(k.index).union(L.index)
    df = pd.DataFrame({"k_open": o.reindex(idx), "k_close": k.reindex(idx), "tmi": lv["KRX 중형 TMI"].reindex(idx),
                       "adv_res": br["adv"].reindex(idx), "dec_res": br["dec"].reindex(idx),
                       "adv_llv": L["AdvAll"].reindex(idx), "dec_llv": L["DecAll"].reindex(idx),
                       "br": S[RX.BR_STATE].reindex(k.index).ffill().fillna(True).astype(bool).reindex(idx),
                       "c2": J["④ C2"].reindex(idx), "p7": J["⑦ C2+갭쇼크"].reindex(idx),
                       "p5": J["⑤ 갭쇼크·브레드스·평소보유"].reindex(idx)}, index=idx)
    df = df.loc[:"2026-11-04"]
    df.index.name = "date"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, float_format="%.10g")
    print(f"저장 {OUT} — {len(df)}행 ({df.index.min():%Y-%m-%d} ~ {df.index.max():%Y-%m-%d}), {OUT.stat().st_size/1024:.0f}KB")


if __name__ == "__main__":
    main()
