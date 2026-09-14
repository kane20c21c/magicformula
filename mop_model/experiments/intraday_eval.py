# -*- coding: utf-8 -*-
"""
intraday_eval.py — "장중 오르는 종목 고르기" 새 전략의 1단계 예측가능성 검정 (검증 전용, 운영 아님)

⚠ 데이포트 수정이 아니다. 보유 시간대가 다른 **별도 전략** 후보의 go/no-go 검정이다.
   데이포트 = 당일 종가(17:00 NXT) 매수 → 익일 갭.  이 전략 = 익일 시가 매수 → 익일 종가 매도.
   9/13 결론("새 정보원 없이 148피처 재조합 금지")은 데이포트 개선에 대한 것이고, 여기서 148피처를
   다시 쓰는 것은 "같은 피처가 다른 시간대의 방향을 가르는가"를 처음 재는 것이다.

배경 (2026-09-14, YZ 변동성 갭/일중 분해 — SP reports/YZ20_3성분_방향_20260911.html):
  · 유니버스 2026 장중(시가→종가) 중앙값 −0.29%/일, 양의 날 44% — 기저율은 역풍.
  · 우리금융 8/4 이후 장중 71% 양·중앙 +0.87 / 갭 32% 양·중앙 −0.36 (p=0.04) — 금융 섹터 공통.
  · 데이포트는 회전율·변동성 피처로 이런 종목을 정확히 걸러낸다(타깃상 옳음). 낮의 기회는 데이포트가 재지 않는다.
  0단계 (2024-09-02~2026-09-10, 491일, 거래정지 제외):
  · 장중 횡단면 분산 3.15%p (갭의 2배), 오라클 top10 +8.22%/일, +1% 이상 종목 하루 62개 → 고를 대상은 넓다.
  · 기준선 IC(→익일 장중): 회전율 −0.068 · vol20 −0.060 · 20일 갭드리프트 −0.058 · 전일 장중 −0.046 (전부 t≥5.7)
    양(+) 쪽은 20일 장중 양의날 비율 +0.024 (t 3.7) 뿐. 밤(갭)과 낮(장중)은 같은 피처에서 부호가 거울상.
  · 단순 top10 규칙은 비용(왕복 0.23%) 전 +0.12~+0.16%/일, 비용 후 전부 음수.
  · 필요 IC 역산: top10 기대치 ≈ IC × 6.5%p → 기저 −0.26 + 비용 0.23 을 넘겨 +0.3 남기려면 IC ≈ 0.12.

가설 (각각 한 번씩만, 튜닝 금지 원칙 준수 — 하이퍼파라미터 불변):
  intra_base : 148피처 그대로, 타깃만 y_intra = (익일 시가→종가 > 그날 중앙값).
  intra_plus : + 장중 전용 5피처 (모두 t 마감 후 확정값, 룩어헤드 없음)
               id20  = 최근 20일 장중 수익률 평균        id_up20 = 최근 20일 장중 양의날 비율
               id1   = 당일 장중 수익률                  gd20    = 최근 20일 갭 평균
               gshare20 = 20일 갭 분산 / (갭 분산 + 장중 분산)   (YZ 분해의 갭 비중과 같은 뜻)
  (시드 잡음 바닥은 --seed 로 intra_plus 를 2~3회 더 돌려 잰다.)

사전 등록 판정 기준 (결과 보고 바꾸지 않는다):
  ① 전 구간 일별 IC(스피어만 p vs 익일 장중) 평균 ≥ 0.08
  ② rank≤10 의 익일 장중 평균 − 왕복비용 0.23% > 0   (양 끝이 단일가라 슬리피지는 0 으로 둔다)
  ③ ② 가 단순 규칙 "id_up20 상위 10" 의 비용후 순익보다 커야 한다 (모델일 이유)
  ④ 전 구간 IC 음수일 비율 ≤ 45%
  넷 다 만족 → 2단계(규칙 시뮬). 하나라도 미달 → "현재 피처·16:20 정보로는 장중 방향을 못 고른다" 로 종결,
  다음 수는 새 정보원(08:50 신호: 미국 마감·야간선물·NXT 프리마켓)뿐.
  보조 지표(판정 아님): 좌측 꼬리(top10 중 −3% 이하 비율 − 유니버스), 우측 꼬리(+3% 이상), top10 섹터 집중, 8/4 이후 구간.

사용 (맥 로컬에서 — VM 아님):
  cd MagicFormula/mop_model/experiments
  python3 intraday_eval.py run --hyp intra_base --start 2024-09-02 --end 2026-09-10 --threads 8 | tee ../build/id_intra_base.log
  python3 intraday_eval.py run --hyp intra_plus --start 2024-09-02 --end 2026-09-10 --threads 8 | tee ../build/id_intra_plus.log
  python3 intraday_eval.py run --hyp intra_plus --start 2024-09-02 --end 2026-09-10 --threads 8 --seed 7 --out ../build/id_intra_plus_s7.parquet
  python3 intraday_eval.py eval                                   # build/id_*.parquet 전부 비교 + 판정
  (스모크: run --hyp intra_plus --start 2026-08-01 --end 2026-08-05 --no-ensemble  ≈ 1~2분)
  ⚠ features.parquet 이 --end 다음 거래일까지 있어야 한다 (C2C_T1·Gap_T1 라벨). 결과: build/id_<hyp>[_sN].parquet
"""
import argparse, glob, os, sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import config as cfg                      # noqa: E402
from model import fit_predict             # noqa: E402

RETRAIN_EVERY = 5
COST_RT = 0.23          # 왕복 비용 % (라이브 조건: 수수료+세금)
TAIL = 3.0              # 꼬리 정의 |장중| ≥ 3%
INTRA_COLS = ["id20", "id_up20", "id1", "gd20", "gshare20"]
GATE = dict(ic=0.08, neg_pct=45.0)


def add_intra_target(d):
    """y_intra = 익일 시가→종가 수익률이 그날 유니버스 중앙값보다 큰가.
    intra_T1 = (1+C2C_T1)/(1+Gap_T1) − 1  (features.parquet 의 두 라벨로 정확히 복원)."""
    d["intra_T1"] = (1.0 + d.C2C_T1) / (1.0 + d.Gap_T1) - 1.0
    med = d.groupby("Date").intra_T1.transform("median")
    d["y_intra"] = np.where(d.intra_T1.isna() | med.isna(), np.nan, (d.intra_T1 > med).astype(float))
    return d


def add_intra_feats(d):
    """장중 전용 5피처 — 전부 날짜 t 의 O/C 까지만 사용 (t 15:30 확정)."""
    d = d.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    g = d.groupby("Ticker")
    d["id1"] = d.Close / d.Open - 1.0                                  # 당일 장중
    d["_gap0"] = d.Open / g.Close.shift(1) - 1.0                        # 당일 갭 (t−1 종가 → t 시가)
    d["id20"] = d.groupby("Ticker").id1.transform(lambda s: s.rolling(20, min_periods=15).mean())
    d["id_up20"] = d.groupby("Ticker").id1.transform(lambda s: (s > 0).astype(float).rolling(20, min_periods=15).mean())
    d["gd20"] = d.groupby("Ticker")._gap0.transform(lambda s: s.rolling(20, min_periods=15).mean())
    vg = d.groupby("Ticker")._gap0.transform(lambda s: s.rolling(20, min_periods=15).var())
    vi = d.groupby("Ticker").id1.transform(lambda s: s.rolling(20, min_periods=15).var())
    d["gshare20"] = vg / (vg + vi)
    d = d.drop(columns=["_gap0"]).replace([np.inf, -np.inf], np.nan)
    print(f"[intra_plus] 결손율 {d[INTRA_COLS].isna().mean().round(4).to_dict()}", flush=True)
    return d


def run(hyp, start, end, out_path, retrain_every=RETRAIN_EVERY, use_ensemble=True, seed=None, threads=None):
    if seed is not None:
        cfg.LGBM_PARAMS["random_state"] = int(seed); cfg.CAT_PARAMS["random_seed"] = int(seed)
    if threads:
        cfg.LGBM_PARAMS["n_jobs"] = int(threads); cfg.CAT_PARAMS["thread_count"] = int(threads)
        print(f"[threads] LGBM n_jobs={threads} · CatBoost thread_count={threads}", flush=True)
    d = pd.read_parquet(cfg.FEATURES)
    d["Date"] = pd.to_datetime(d.Date)
    cols = json.load(open(cfg.COLS_JSON))["CHAMPION"]
    d = add_intra_target(d)
    if hyp == "intra_plus":
        d = add_intra_feats(d); cols = cols + INTRA_COLS
    d = d.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    target = "y_intra"
    print(f"[{hyp}] y_intra 양성률 {d.y_intra.mean():.3f} · 장중 라벨 결손율 {d.intra_T1.isna().mean():.4f}", flush=True)

    alldays = np.sort(d.Date.unique())
    nxt = {alldays[i]: alldays[i + 1] for i in range(len(alldays) - 1)}
    prv = {alldays[i]: alldays[i - 1] for i in range(1, len(alldays))}
    days = [x for x in alldays if pd.Timestamp(start) <= pd.Timestamp(x) <= pd.Timestamp(end) and x in nxt]
    print(f"[{hyp}] 스코어일 {len(days)}일 ({pd.Timestamp(days[0]).date()}~{pd.Timestamp(days[-1]).date()}) · "
          f"재학습 {int(np.ceil(len(days)/retrain_every))}회 · 피처 {len(cols)} · 타깃 {target} · 앙상블 {use_ensemble}", flush=True)

    out, model_s, t0 = [], None, time.time()
    for i, t in enumerate(days):
        if i % retrain_every == 0:
            train = d[d.Date <= prv[t]].dropna(subset=[target])
            score_block = d[d.Date.isin(days[i:i + retrain_every])]
            s, meta = fit_predict(train, score_block, cols, target=target, use_ensemble=use_ensemble, return_meta=True)
            model_s = pd.Series(s.values, index=score_block.index)
            print(f"  [{i+1:3d}/{len(days)}] 재학습 {pd.Timestamp(t).date()} train={meta['train_rows']} "
                  f"({meta['train_first_date']}~) lgbm_auc={meta.get('lgbm_valid_auc')} cat_auc={meta.get('cat_valid_auc')} "
                  f"경과 {(time.time()-t0)/60:.1f}분", flush=True)
        cur = d[d.Date == t]
        keep = ["Date", "Ticker", "sector_top", "Gap_T1", "intra_T1", "y_intra", "Close"] + (["id_up20"] if hyp == "intra_plus" else [])
        r = cur[keep].copy()
        r["p"] = model_s.reindex(cur.index).rank(pct=True).values
        r["rank"] = (-r.p).rank(method="first").astype(int)
        nx = d[d.Date == nxt[t]].set_index("Ticker")
        r["close_T1"] = r.Ticker.map(nx.Close); r["open_T1"] = r.Ticker.map(nx.Open)
        r["hyp"] = hyp
        out.append(r)
        if (i + 1) % 20 == 0:
            pd.concat(out).to_parquet(out_path, index=False)
    R = pd.concat(out); R.to_parquet(out_path, index=False)
    print(f"[{hyp}] 완료 {len(R)}행 → {out_path} · 총 {(time.time()-t0)/60:.1f}분", flush=True)
    return R


# ───────────────────────────── eval ─────────────────────────────
def _daily(R, score_col="p"):
    from scipy.stats import spearmanr
    rows = []
    for D, g in R.dropna(subset=["intra_T1", score_col]).groupby("Date"):
        if len(g) < 50: continue
        rk = (-g[score_col]).rank(method="first")
        t10 = g[rk <= 10]; ui = g.intra_T1 * 100; ti = t10.intra_T1 * 100
        rows.append({"d": D, "ic": spearmanr(g[score_col], g.intra_T1).correlation,
                     "top10": ti.mean(), "top10_net": ti.mean() - COST_RT, "uni": ui.mean(), "uni_med": ui.median(),
                     "up10": (ti > 0).mean() * 100,
                     "left_ex": ((ti <= -TAIL).mean() - (ui <= -TAIL).mean()) * 100,
                     "right_ex": ((ti >= TAIL).mean() - (ui >= TAIL).mean()) * 100,
                     "sec_max10": t10.sector_top.value_counts().iloc[0] if len(t10) else np.nan})
    return pd.DataFrame(rows)


def _summ(W):
    return {"days": len(W), "ic": W.ic.mean(), "ic_se": W.ic.std() / np.sqrt(max(len(W), 1)), "ic_neg%": (W.ic < 0).mean() * 100,
            "top10": W.top10.mean(), "top10_net": W.top10_net.mean(), "net>0일%": (W.top10_net > 0).mean() * 100,
            "up10%": W.up10.mean(), "uni_med": W.uni_med.mean(), "left_ex": W.left_ex.mean(), "right_ex": W.right_ex.mean(),
            "sec_max10": W.sec_max10.mean()}


def evaluate(split="2026-08-04"):
    files = sorted(glob.glob(os.path.join(cfg.OUT_DIR, "id_*.parquet")))
    if not files: sys.exit("build/id_*.parquet 없음")
    res = {}
    rule = None
    for f in files:
        R = pd.read_parquet(f); R["Date"] = pd.to_datetime(R.Date)
        h = os.path.basename(f)[3:-8]
        W = _daily(R); W["d"] = pd.to_datetime(W.d)
        res[h] = {"full": _summ(W), "pre": _summ(W[W.d < split]), "post": _summ(W[W.d >= split])}
        if rule is None and "id_up20" in R.columns:          # 단순 규칙 기준선: id_up20 상위 10 (모델 없음)
            Wr = _daily(R.dropna(subset=["id_up20"]), score_col="id_up20"); Wr["d"] = pd.to_datetime(Wr.d)
            rule = {"full": _summ(Wr), "pre": _summ(Wr[Wr.d < split]), "post": _summ(Wr[Wr.d >= split])}
    if rule is not None:
        res["rule:id_up20"] = rule
    for per in ["full", "pre", "post"]:
        print(f"\n== {per} (split {split}) ==")
        print(pd.DataFrame({h: v[per] for h, v in res.items()}).T.round(3).to_string())
    print(f"\n== 사전 등록 판정 — ①IC≥{GATE['ic']} ②top10 비용후(−{COST_RT}%)>0 ③②>단순규칙 ④IC 음수일≤{GATE['neg_pct']}% ==")
    rule_net = rule["full"]["top10_net"] if rule else -np.inf
    for h, v in res.items():
        if h.startswith("rule:"): continue
        F = v["full"]
        c = [F["ic"] >= GATE["ic"], F["top10_net"] > 0, F["top10_net"] > rule_net, F["ic_neg%"] <= GATE["neg_pct"]]
        print(f"  {h:18s} " + " · ".join(f"{n}{'O' if x else 'X'}" for n, x in zip("①②③④", c))
              + f" → {'2단계 진행' if all(c) else '기각'}  | IC {F['ic']:+.4f}±{F['ic_se']:.4f} · top10 비용후 {F['top10_net']:+.3f}%/일 · 규칙 {rule_net:+.3f}")
    print("  참고: 시드 잡음 = 같은 hyp 의 _sN 결과 간 IC·top10_net 스프레드. 그 안에 든 차이는 차이가 아니다.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--hyp", required=True, choices=["intra_base", "intra_plus"])
    r.add_argument("--start", required=True); r.add_argument("--end", required=True)
    r.add_argument("--retrain-every", type=int, default=RETRAIN_EVERY)
    r.add_argument("--no-ensemble", action="store_true"); r.add_argument("--out", default=None)
    r.add_argument("--seed", type=int, default=None); r.add_argument("--threads", type=int, default=None)
    e = sub.add_parser("eval"); e.add_argument("--split", default="2026-08-04")
    a = ap.parse_args()
    if a.cmd == "run":
        out = a.out or os.path.join(cfg.OUT_DIR, f"id_{a.hyp}.parquet")
        run(a.hyp, a.start, a.end, out, a.retrain_every, not a.no_ensemble, a.seed, a.threads)
    else:
        evaluate(a.split)
