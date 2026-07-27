# -*- coding: utf-8 -*-
"""
walkforward.py — 러닝(매일 재학습) 모델 백테스트 (검증 전용, 운영 아님)

개념: train/valid/test 고정분할은 '검증용'이었음. 배포에서는 매일
      '오늘까지 라벨 확정된 전 데이터'로 재학습하여 다음을 예측한다.
      하이퍼파라미터·피처는 고정, 모델(트리)만 매일 갱신.

라벨 타이밍: 종목 s행의 라벨 y_rel은 Gap_T1(=익일 시가/당일 종가)이므로
   '익일 시가 시점'에 확정. 따라서 매수일 t의 종가 시점에 학습 가능한
   가장 최근 라벨 확정 feature-date = t의 직전 거래일.
   → train = (Date <= 직전거래일), score = (Date == t)

⚠ 여기서 계산하는 수익은 **백테스트 이상체결**(t일 종가 매수 → t+1 시가 매도)이다.
  실제 데이 포트는 17:00 NXT 애프터마켓 매수 + 2% 손절 보유이므로
  이 수치와 **직접 비교 금지**. 모델 예측력의 상한 참고치로만 볼 것.

사용:
   python3 walkforward.py --start 2026-07-01 --end 2026-07-24
   python3 walkforward.py --start 2026-07-01 --end 2026-07-24 --static
   python3 walkforward.py --start 2026-07-01 --end 2026-07-24 --lgbm-only
"""
import argparse, json, numpy as np, pandas as pd
import config as cfg
from model import fit_predict

BUY_FEE = 0.00015    # 참고용(백테스트 재현) — 실제 비용 정본은 SP app/paper_day/config.py
SELL_FEE = 0.00215


def run(start, end, static=False, use_ensemble=True, target=None, top_k=None):
    target = target or cfg.TARGET
    top_k = top_k or cfg.TOP_K
    d = pd.read_parquet(cfg.FEATURES)
    cols = json.load(open(cfg.COLS_JSON))["CHAMPION"]
    d = d.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    alldays = np.sort(d.Date.unique())
    nxt = {alldays[i]: alldays[i + 1] for i in range(len(alldays) - 1)}
    prv = {alldays[i]: alldays[i - 1] for i in range(1, len(alldays))}
    buy_days = [x for x in alldays if (pd.Timestamp(x) >= pd.Timestamp(start)
                and pd.Timestamp(x) <= pd.Timestamp(end) and x in nxt)]

    static_model_scores = None
    if static:
        first_prev = prv[buy_days[0]]
        train = d[d.Date <= first_prev]
        static_model_scores = fit_predict(train, d[d.Date.isin(buy_days)], cols, target, use_ensemble)

    rows = []
    for t in buy_days:
        sday = nxt[t]
        score_df = d[d.Date == t].copy()
        if static:
            score_df["p"] = static_model_scores.loc[score_df.index].values
        else:
            train = d[d.Date <= prv[t]]               # ← 매일 확장 재학습
            score_df["p"] = fit_predict(train, score_df, cols, target, use_ensemble).values

        picks = score_df.sort_values("p", ascending=False).head(top_k).Ticker.tolist()
        b = d[(d.Date == t) & d.Ticker.isin(picks)].set_index("Ticker")
        s = d[(d.Date == sday) & d.Ticker.isin(picks)].set_index("Ticker")
        tk = [x for x in picks if x in b.index and x in s.index]
        rr = [ (s.loc[k, "Open"] * (1 - SELL_FEE)) / (b.loc[k, "Close"] * (1 + BUY_FEE)) - 1
               for k in tk if b.loc[k, "Close"] > 0 and not np.isnan(s.loc[k, "Open"]) ]
        rows.append(dict(buy=str(pd.Timestamp(t).date()), sell=str(pd.Timestamp(sday).date()),
                         n=len(rr), ret_net=float(np.mean(rr)) if rr else np.nan,
                         picks=";".join(b.loc[tk, "Name"].tolist())))
        print(f"  {rows[-1]['buy']} n={rows[-1]['n']} 오버나이트(비용후) {rows[-1]['ret_net']*100:+.2f}%")

    R = pd.DataFrame(rows)
    tag = "static" if static else "walkforward"
    out = f"{cfg.OUT_DIR}/backtest_{tag}.csv"; R.to_csv(out, index=False)
    print(f"\n[{tag}] 거래일 {len(R)} | 거래당 비용후 {R.ret_net.mean()*100:+.3f}% | 누적(단리합) {R.ret_net.sum()*100:+.1f}% → {out}")
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--static", action="store_true", help="정적(1회학습) 비교")
    ap.add_argument("--lgbm-only", action="store_true", help="LGBM 단독(속도)")
    a = ap.parse_args()
    run(a.start, a.end, static=a.static, use_ensemble=not a.lgbm_only)
