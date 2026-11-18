# -*- coding: utf-8 -*-
"""scorecard.py — 성과지표 스코어카드 (성과지표 스터디 6-2, 케인 2026-11-18).

수익률 시계열 하나(+ 선택 재료)를 넣으면 한 장짜리 성과표를 채운다. 읽는 순서 = ①조건 ②신뢰성 ③효율 ④위험의 모양 ⑤국면.
정의는 스터디 노트 「2026-11-18 성과지표 스터디 실습 A」 와 같다 (옵시디언 30_NoteforStock).

입력
  r        일간 **단순** 수익률 Series (DatetimeIndex)            — 필수
  rf       일간 Rf Series (없으면 0). 연구 기본 = 콜 1일물: load_rf("CALL1D")
  bm       일간 BM 수익률 Series (있으면 IR·베타·젠센 알파·Appraisal)
  trades   매매 건별 수익률(소수) 또는 R-multiple Series (trades_are_r=True)
  n_trials 이 결과를 고르기까지 시도한 변형 수, trial_sr_var 그 변형들 일간 Sharpe 의 분산 (둘 다 있으면 DSR)
  regime   국면 라벨 Series (그날 종가까지 본 라벨) → 다음 날 수익률로 국면별 효율
출력  dict {영역: {항목: 값}} + "경고" 리스트.  to_frame(card) = 표, compare({이름: card}) = 나란히.

규칙
  - 연율화는 252 관례 (실제 연 거래일은 '조건'에 따로 표시). CAGR 은 달력 연수(365.25일).
  - 하방편차·Sortino 의 MAR = Rf, 분모는 전체 n. Sharpe 표준오차 = Mertens(왜도·첨도 반영).
  - MDD·수중·UI 는 r 로 만든 누적곡선 기준. 최장 수중은 마지막에 미회복이면 '진행 중' 표시.
  - 값이 없으면 None — 짐작으로 채우지 않는다.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

HERE = Path(__file__).resolve().parent
ANN = 252
EULER = 0.5772156649


def load_rf(name: str = "CALL1D", index: pd.Index | None = None) -> pd.Series:
    """data_idx/ecos_rates.parquet 의 연% 금리 → 일간 Rf. index 를 주면 그 날짜로 맞춘다(직전값 유지)."""
    t = pd.read_parquet(HERE / "data_idx" / "ecos_rates.parquet")
    s = t[t.name == name].set_index("date").rate.sort_index()
    if index is not None:
        s = s.reindex(s.index.union(index)).ffill().reindex(index)
    return (1 + s / 100) ** (1 / ANN) - 1


def load_k200() -> pd.Series:
    """data_idx/krx_indices.parquet 의 코스피 200 종가."""
    k = pd.read_parquet(HERE / "data_idx" / "krx_indices.parquet")
    return k[k.name == "코스피 200"].set_index("date").close.astype(float).sort_index()


def _dd(r: pd.Series):
    eq = pd.concat([pd.Series([1.0], [r.index[0] - pd.Timedelta(days=1)]), (1 + r).cumprod()])
    return eq, eq / eq.cummax() - 1


def _spells(mask: pd.Series):
    g = (~mask).cumsum()
    return mask[mask].groupby(g[mask]).agg(lambda s: (s.index[0], s.index[-1], len(s)))


def _cagr(r: pd.Series) -> float:
    yrs = ((r.index[-1] - r.index[0]).days + 1) / 365.25          # 첫 수익률 전날부터
    return (1 + r).prod() ** (1 / yrs) - 1 if yrs > 0 else np.nan

def scorecard(r: pd.Series, rf: pd.Series | None = None, bm: pd.Series | None = None,
              trades: pd.Series | None = None, trades_are_r: bool = False,
              n_trials: int = 1, trial_sr_var: float | None = None,
              regime: pd.Series | None = None, costs_included: bool | None = None) -> dict:
    r = r.dropna().astype(float).sort_index()
    rf = pd.Series(0.0, index=r.index) if rf is None else rf.reindex(r.index).ffill().fillna(0.0)
    ex = r - rf
    n = len(r); yrs = ((r.index[-1] - r.index[0]).days + 1) / 365.25
    card, warn = {}, []

    # ① 조건
    card["① 조건"] = {"시작": r.index[0].date(), "끝": r.index[-1].date(), "연수": round(yrs, 2), "일수": n,
                     "실제 연 거래일": round(n / yrs, 1) if yrs else None,
                     "Rf 금리 평균%": round((((1 + rf) ** ANN - 1).mean()) * 100, 2),
                     "BM": bm is not None, "비용 반영": costs_included,
                     "매매 수": None if trades is None else int(trades.dropna().size)}

    # ③ 효율에 쓰는 기본량
    srd = ex.mean() / ex.std(); sr = srd * np.sqrt(ANN)
    g3, g4x = float(skew(r)), float(kurtosis(r))                 # 초과첨도
    se = np.sqrt((1 - g3 * srd + (g4x / 4 + 0.5) * srd ** 2) / n) * np.sqrt(ANN)   # Mertens
    t = srd * np.sqrt(n)
    eq, dd = _dd(r)
    mdd = dd.min(); trough = dd.idxmin(); peak = eq.loc[:trough].idxmax()
    rec = eq.loc[trough:][eq.loc[trough:] >= eq.loc[peak]]
    uw = _spells(dd < 0); longest = max(uw.tolist(), key=lambda x: x[2]) if len(uw) else None
    ui = float(np.sqrt(((dd * 100) ** 2).mean()))
    cagr = _cagr(r); rf_y = (1 + rf).prod() ** (1 / yrs) - 1
    dsd = np.sqrt((np.minimum(ex, 0) ** 2).mean())
    last = r.loc[r.index[-1] - pd.DateOffset(months=36):]
    _, dd3 = _dd(last)

    # ② 신뢰성
    psr0 = norm.cdf(srd * np.sqrt(n - 1) / np.sqrt(1 - g3 * srd + (g4x + 2) / 4 * srd ** 2))
    dsr = None
    if n_trials > 1 and trial_sr_var:
        sr_star = np.sqrt(trial_sr_var) * ((1 - EULER) * norm.ppf(1 - 1 / n_trials) + EULER * norm.ppf(1 - 1 / (n_trials * np.e)))
        dsr = float(norm.cdf((srd - sr_star) * np.sqrt(n - 1) / np.sqrt(1 - g3 * srd + (g4x + 2) / 4 * srd ** 2)))
    card["② 신뢰성"] = {"Sharpe t": round(t, 2), "Sharpe 표준오차": round(se, 2),
                      "Sharpe 95% 하한": round(sr - 1.96 * se, 2), "Sharpe 95% 상한": round(sr + 1.96 * se, 2),
                      "PSR(SR>0)": round(float(psr0), 3), "시도 횟수": n_trials,
                      "DSR": None if dsr is None else round(dsr, 3)}

    # ③ 효율
    om = lambda x: np.clip(x, 0, None).mean() / np.clip(-x, 0, None).mean()
    card["③ 효율"] = {"누적%": round(((1 + r).prod() - 1) * 100, 1), "CAGR%": round(cagr * 100, 1),
                     "산술×252%": round(r.mean() * ANN * 100, 1), "Sharpe": round(sr, 2),
                     "Sharpe(Rf 0)": round(r.mean() / r.std() * np.sqrt(ANN), 2),
                     "Sortino": round(ex.mean() / dsd * np.sqrt(ANN), 2),
                     "Calmar": round(cagr / abs(mdd), 2) if mdd < 0 else None,
                     "Calmar 36개월": round(_cagr(last) / abs(dd3.min()), 2) if dd3.min() < 0 else None,
                     "Martin": round((cagr - rf_y) * 100 / ui, 2) if ui else None,
                     "Omega(L=0)": round(om(r), 3), "Omega(L=Rf)": round(om(ex), 3)}

    # ④ 위험의 모양
    q95, q99 = -np.quantile(r, 0.05), -np.quantile(r, 0.01)
    card["④ 위험의 모양"] = {"σ%": round(r.std() * np.sqrt(ANN) * 100, 1), "하방편차%": round(dsd * np.sqrt(ANN) * 100, 1),
                          "변동성손실 σ²/2 %p": round((r.std() * np.sqrt(ANN)) ** 2 / 2 * 100, 1),
                          "MDD%": round(mdd * 100, 1), "MDD 고점": peak.date(), "MDD 저점": trough.date(),
                          "MDD 회복": rec.index[0].date() if len(rec) else "미회복",
                          "최장 수중일": None if longest is None else longest[2],
                          "최장 수중 구간": None if longest is None else f"{longest[0].date()}~{longest[1].date()}" + (" (진행 중)" if longest[1] == dd.index[-1] else ""),
                          "수중 비율%": round((dd < 0).mean() * 100, 1), "−10% 이하 비율%": round((dd <= -0.10).mean() * 100, 1),
                          "Ulcer Index": round(ui, 1),
                          "VaR95%": round(q95 * 100, 2), "CVaR95%": round(-r[r <= -q95].mean() * 100, 2),
                          "VaR99%": round(q99 * 100, 2), "CVaR99%": round(-r[r <= -q99].mean() * 100, 2),
                          "왜도": round(g3, 2), "초과첨도": round(g4x, 2)}

    # BM 대비 (IR · 베타 · 젠센 알파 · Appraisal)
    if bm is not None:
        b = bm.reindex(r.index); ok = b.notna()
        act = (r - b)[ok]; y = ex[ok]; x = (b - rf)[ok]
        X = np.column_stack([np.ones(len(x)), x.values]); coef, *_ = np.linalg.lstsq(X, y.values, rcond=None)
        res = y.values - X @ coef; s2 = res @ res / (len(y) - 2); cov = s2 * np.linalg.inv(X.T @ X)
        a_t = coef[0] / np.sqrt(cov[0, 0])
        card["BM 대비"] = {"초과수익 연%": round(act.mean() * ANN * 100, 1), "TE%": round(act.std() * np.sqrt(ANN) * 100, 1),
                         "IR": round(act.mean() / act.std() * np.sqrt(ANN), 2), "베타": round(coef[1], 2),
                         "젠센 알파 연%": round(coef[0] * ANN * 100, 1), "알파 t": round(a_t, 2),
                         "Appraisal": round(coef[0] / np.sqrt(s2) * np.sqrt(ANN), 2),
                         "단순알파−젠센 %p": round((act.mean() - coef[0]) * ANN * 100, 1)}
        if abs(card["BM 대비"]["단순알파−젠센 %p"]) > 2:
            warn.append("베타 섞임 — 단순 알파(R−BM)와 젠센 알파 차이가 연 2%p 초과")

    # 매매 단위
    if trades is not None and trades.dropna().size:
        x = trades.dropna().astype(float); w, l = x[x > 0], x[x < 0]
        p = len(w) / len(x); b_ = w.mean() / -l.mean() if len(l) and len(w) else None
        card["매매 단위"] = {"매매 수": len(x), "승률%": round(p * 100, 1),
                           "승률 표준오차%p": round(np.sqrt(p * (1 - p) / len(x)) * 100, 1),
                           "손익비": None if b_ is None else round(b_, 2),
                           "본전 승률%": None if b_ is None else round(100 / (1 + b_), 1),
                           "PF": round(w.sum() / -l.sum(), 2) if len(l) else None,
                           "기대값": round(x.mean(), 3 if trades_are_r else 4),
                           "기대값 t(SQN)": round(x.mean() / x.std() * np.sqrt(len(x)), 2) if len(x) > 2 else None,
                           "단위": "R" if trades_are_r else "수익률(소수)"}
        if trades_are_r:
            card["매매 단위"]["−1R 초과 손실%"] = round((x < -1.05).mean() * 100, 1)
        if p > 0.65 and b_ is not None and b_ < 0.5:
            warn.append("높은 승률·낮은 손익비 — 음의 왜도(B형) 매매 의심")
        if len(x) < 30:
            warn.append("매매 30건 미만 — 승률·PF 신뢰도 낮음")

    # ⑤ 국면 (그날 라벨 → 다음 날 수익률)
    if regime is not None:
        lab = regime.reindex(r.index); y = r.shift(-1)
        reg = {}
        for k in [k for k in pd.unique(lab.dropna())]:
            z = y[lab == k].dropna()
            reg[str(k)] = {"비중%": round((lab == k).mean() * 100, 1), "일수": len(z),
                           "다음날 연수익%": round(z.mean() * ANN * 100, 1),
                           "Sharpe(Rf 0)": round(z.mean() / z.std() * np.sqrt(ANN), 2) if len(z) > 2 else None}
        card["⑤ 국면"] = reg
        pos = [k for k, v in reg.items() if v["Sharpe(Rf 0)"] and v["Sharpe(Rf 0)"] > 0]
        if len(pos) <= 1:
            warn.append(f"국면 의존 — Sharpe 플러스 국면이 {len(pos)}개뿐")

    # 경고
    if yrs < 2: warn.append(f"표본 {yrs:.1f}년 — 2년 미만, 효율 숫자 신뢰도 낮음")
    if t < 2: warn.append(f"Sharpe 유의하지 않음 (t {t:.2f} < 2)")
    if sr > 2: warn.append(f"Sharpe {sr:.2f} > 2 — 과최적화·비용 누락 의심")
    so = card["③ 효율"]["Sortino"]
    if sr > 1 and (g3 < 0 or (so and so / sr < 1.2)): warn.append("숨은 꼬리 — Sharpe 높은데 왜도<0 또는 Sortino÷Sharpe<1.2")
    if g4x > 3: warn.append(f"두꺼운 꼬리 (초과첨도 {g4x:.1f}) — 정규 VaR 대신 역사적 CVaR")
    cal = card["③ 효율"]["Calmar"]
    if cal is not None and sr > 0 and cal / sr < 0.6: warn.append(f"손실 몰림 — Calmar {cal} 가 Sharpe {sr:.2f} 에 비해 낮음")
    if costs_included is None: warn.append("비용 반영 여부 미기재")
    elif costs_included is False: warn.append("비용 미반영 — 실제보다 좋게 나온다")
    if n_trials > 1 and dsr is None: warn.append(f"시도 {n_trials}회인데 DSR 미계산 (trial_sr_var 필요)")
    if dsr is not None and dsr < 0.95: warn.append(f"DSR {dsr:.2f} < 0.95 — 여러 번 시도한 걸 감안하면 운일 수 있음")
    card["경고"] = warn
    return card


def to_frame(card: dict) -> pd.DataFrame:
    rows = []
    for sec, items in card.items():
        if sec == "경고":
            rows += [("경고", f"{i+1}", w) for i, w in enumerate(items)]
        elif sec == "⑤ 국면":
            for k, v in items.items():
                rows += [(sec, f"{k} · {kk}", vv) for kk, vv in v.items()]
        else:
            rows += [(sec, k, v) for k, v in items.items()]
    return pd.DataFrame(rows, columns=["영역", "항목", "값"]).set_index(["영역", "항목"])


def compare(cards: dict[str, dict]) -> pd.DataFrame:
    """{이름: card} → 나란히 비교 표 (경고 제외)."""
    return pd.concat({k: to_frame({s: v for s, v in c.items() if s != "경고"})["값"] for k, c in cards.items()}, axis=1)


if __name__ == "__main__":
    import sys
    end = sys.argv[1] if len(sys.argv) > 1 else None
    px = load_k200().loc["2015-01-02":end]
    r = px.pct_change().dropna()
    card = scorecard(r, rf=load_rf("CALL1D", r.index))
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
    print(f"KOSPI200 기준선 스코어카드 ({r.index[0].date()} ~ {r.index[-1].date()}, Rf = 콜 1일물)")
    print(to_frame(card).to_string())
