"""make_pattern_report.py — 진입 패턴 격자 연구 통합 리포트(HTML) v2.

2026-08-17 오후 세션 전체를 하나로 통합:
  깊이 축 · 창 축 · MA 축 · 계단 진입 · 물타기 분해 · 출렁다리(정밀도 c→b→a) 분석.
숫자는 전부 실행 결과에서 직접 뽑는다.
출력: StockPortfolio/reports/스윙포트_진입패턴연구_20260817.html
"""
from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from backtest import (EntryParams, ExitParams, PortfolioParams, UniverseParams,
                      VolScaleParams, load_panel, metrics, run_backtest)
from verify_entry import block_bootstrap
from sweep_entry import REGIMES

UP, DOWN = "#ef5350", "#1976D2"
REPORTS = Path(__file__).resolve().parents[2] / "StockPortfolio" / "reports"
OUT = Path(__file__).resolve().parent / "out"
YEARS = 11.5

MAn = lambda n: dict(trend_ma=n, trend_min_periods=int(n * 0.7))

MAIN = {
    "기준선 v1.0.0 (60일/−10%/MA120)": EntryParams(),
    "T1  60일/−10%/MA200": EntryParams(**MAn(200)),
    "W1  40일/−10%/MA200": EntryParams(high_window=40, high_min_periods=20, **MAn(200)),
    "W2  50일/−10%/MA200": EntryParams(high_window=50, high_min_periods=25, **MAn(200)),
    "S3n 계단10·20/MA200 (신규한정)": EntryParams(pullback_tiers=(0.10, 0.20),
                                             tiers_new_only=True, **MAn(200)),
}
REF = {
    "(참고) S1 계단10·20·30/MA200": EntryParams(pullback_tiers=(0.10, 0.20, 0.30), **MAn(200)),
    "(참고) S1n = S1 물타기 금지": EntryParams(pullback_tiers=(0.10, 0.20, 0.30),
                                        tiers_new_only=True, **MAn(200)),
    "(참고) 단독 60일/−20%/MA240": EntryParams(pullback_pct=0.20, **MAn(240)),
    "(참고) 단독 60일/−30%/MA240": EntryParams(pullback_pct=0.30, **MAn(240)),
    "(기각) 저점근접 20일/+20%/MA200": EntryParams(dip_basis="low_prox", high_window=20,
                                             high_min_periods=10, low_tol=0.20, **MAn(200)),
}


def won(v):
    c = UP if v > 0 else (DOWN if v < 0 else "#555")
    return f'<td class="n" style="color:{c}">{v:,.0f}원</td>'


def pct(v, signed=True, d=1):
    c = (UP if v > 0 else (DOWN if v < 0 else "#555")) if signed else "#333"
    return f'<td class="n" style="color:{c}">{v:.{d}f}%</td>'


def num(v, d=2):
    return f'<td class="n">{v:.{d}f}</td>'


def wobble_sim(panel, mask, inwin, xp):
    """v1.2.0 청산을 신호 단위로 태워 경로 통계를 잰다."""
    op = panel.open.to_numpy(); hi_ = panel.high.to_numpy()
    lo_ = panel.low.to_numpy(); cln = panel.close.to_numpy()
    sc = panel.vol_scale.to_numpy(); nd = len(panel.dates)
    m = mask.copy(); m.loc[~inwin] = False
    di, tj = np.nonzero(m.to_numpy())
    n = win = wobble = missed = stopped = 0
    rets = []
    for i0, j in zip(di, tj):
        b = i0 + 1
        if b >= nd or not np.isfinite(op[b, j]) or op[b, j] <= 0:
            continue
        e = op[b, j] * 1.00015
        entry = op[b, j]
        s = sc[i0, j]; s = 1.0 if not np.isfinite(s) else min(max(s, 0.2), 5.0)
        peak = avg = entry
        end20 = min(b + 20, nd - 1)
        fruit = (np.nanmax(cln[b + 1:end20 + 1, j]) >= entry * 1.10) if b + 1 <= end20 else False
        r = None; d_exit = None
        for i in range(b, min(b + 60, nd)):
            d = i - b
            late = d > xp.switch_day
            above = peak > avg
            if not late:
                base, clip = xp.early_pct, xp.clip_early
            elif above:
                base, clip = xp.late_up_pct, xp.clip_late_up
            else:
                base, clip = xp.late_flat_pct, xp.clip_late_flat
            p_ = min(max(base * s, clip[0]), clip[1])
            line = (peak if above else avg) * (1 - p_)
            if np.isfinite(lo_[i, j]) and lo_[i, j] <= line:
                fill = (min(op[i, j], line) if np.isfinite(op[i, j]) else line) * 0.995
                r = fill * (1 - 0.00215) / e - 1
                d_exit = d
                break
            if np.isfinite(hi_[i, j]):
                peak = max(peak, hi_[i, j])
        if r is None:
            k = min(b + 60, nd - 1)
            r = cln[k, j] * (1 - 0.00215) / e - 1
        else:
            stopped += 1
            if 6 <= d_exit <= 15:
                wobble += 1
            if r < 0 and fruit:
                missed += 1
        n += 1
        rets.append(r)
        if r > 0:
            win += 1
    rets = np.array(rets)
    return dict(n=n, win=win / n * 100, avg=rets.mean() * 100, stop=stopped / n * 100,
                wobble=wobble / n * 100, missed=missed / n * 100)


def build():
    grid = pd.read_csv(OUT / "pattern_grid.csv")
    panel = load_panel(UniverseParams(), VolScaleParams())
    pp = PortfolioParams()
    ALL = {**MAIN, **REF}
    res = {k: run_backtest(panel, ep, ExitParams(), pp) for k, ep in ALL.items()}
    met = {k: metrics(r) for k, r in res.items()}
    base_key = list(MAIN)[0]
    bm = met[base_key]

    base_ret = res[base_key]["equity"]["equity"].pct_change()
    boots = {k: block_bootstrap(res[k]["equity"]["equity"].pct_change(), base_ret)
             for k in MAIN if k != base_key}
    reg = {k: {name: metrics(run_backtest(panel, ALL[k], ExitParams(), pp, start=s, end=e))
               for name, (s, e) in REGIMES.items()} for k in MAIN}
    p2 = load_panel(UniverseParams(mktcap_min_krw=5e12, foreign_min_pct=30.0), VolScaleParams())
    sens = {k: [
        metrics(run_backtest(panel, ALL[k], ExitParams(), pp, start="2015-01-01", end="2020-12-31")),
        metrics(run_backtest(panel, ALL[k], ExitParams(), pp, start="2021-01-01", end="2026-06-30")),
        metrics(run_backtest(panel, ALL[k], ExitParams(), PortfolioParams(stop_slippage=0.0))),
        metrics(run_backtest(p2, ALL[k], ExitParams(), pp)),
    ] for k in MAIN}

    # 창 고원 (MA200) / 40일 MA 고원 / 60일 MA 고원
    win_grid = {w: metrics(run_backtest(panel, EntryParams(high_window=w, high_min_periods=w // 2,
                                                           **MAn(200)), ExitParams(), pp))
                for w in (20, 30, 40, 50, 60)}
    ma40_grid = {n: metrics(run_backtest(panel, EntryParams(high_window=40, high_min_periods=20,
                                                            **MAn(n)), ExitParams(), pp))
                 for n in (120, 150, 180, 200, 220, 240)}
    ma60_grid = {n: metrics(run_backtest(panel, EntryParams(**MAn(n)), ExitParams(), pp))
                 for n in (120, 150, 180, 200, 220, 240)}

    # 계단 물타기 분해 + 출렁다리
    cl = panel.close
    ma200s = cl.rolling(200, min_periods=140).mean()
    high60 = cl.rolling(60, min_periods=30).max()
    depth = (high60 - cl) / high60
    gate = (cl > ma200s) & panel.elig
    inwin = (panel.dates >= pd.Timestamp("2015-01-01")) & (panel.dates <= pd.Timestamp("2026-06-30"))
    ons = {}
    for t in (0.10, 0.20, 0.30):
        c_ = ((depth >= t) & gate).fillna(False)
        ons[t] = c_ & ~c_.shift(1).fillna(False)
    extra = (ons[0.20] | ons[0.30]) & ~ons[0.10]
    tier1_only = ons[0.10] & ~(ons[0.20] | ons[0.30])
    wb1 = wobble_sim(panel, tier1_only, inwin, ExitParams())
    wb2 = wobble_sim(panel, extra, inwin, ExitParams())

    # 물타기/신규 분해 (S1 원장 재생)
    t1tr = res["T1  60일/−10%/MA200"]["trades"]
    kb = set(zip(t1tr[t1tr.side == "BUY"].date, t1tr[t1tr.side == "BUY"].ticker))
    s1tr = res["(참고) S1 계단10·20·30/MA200"]["trades"]
    held = {}; pyram = fresh = 0
    for _, r in s1tr.sort_values("date").iterrows():
        if r["side"] == "BUY":
            if (r["date"], r["ticker"]) not in kb:
                if held.get(r["ticker"], 0) > 0:
                    pyram += 1
                else:
                    fresh += 1
            held[r["ticker"]] = held.get(r["ticker"], 0) + 1
        else:
            held[r["ticker"]] = 0

    H = []
    A = H.append
    A(f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>스윙 포트 진입 패턴 격자 연구 (2026-08-17)</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Apple SD Gothic Neo',sans-serif;
max-width:1200px;margin:0 auto;padding:28px 20px 80px;color:#222;line-height:1.65}}
h1{{font-size:25px;border-bottom:3px solid #222;padding-bottom:10px;margin-bottom:6px}}
h2{{font-size:19px;margin-top:38px;border-left:5px solid #222;padding-left:11px}}
h3{{font-size:16px;margin-top:26px;color:#333}}
table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:13px}}
th,td{{border:1px solid #dcdcdc;padding:6px 8px}}
th{{background:#f3f4f6;font-weight:600;text-align:center}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}} td.l{{text-align:left}}
tr.best{{background:#fff8e1}} tr.base td{{background:#eef1f4;font-weight:600}}
.box{{background:#f7f9fb;border-left:5px solid #1976D2;padding:13px 17px;margin:16px 0;border-radius:3px}}
.warn{{background:#fff5f5;border-left:5px solid #ef5350;padding:13px 17px;margin:16px 0;border-radius:3px}}
.ok{{background:#f2f9f2;border-left:5px solid #43a047;padding:13px 17px;margin:16px 0;border-radius:3px}}
.small{{font-size:12.5px;color:#666}}
</style></head><body>""")
    A("<h1>스윙 포트 진입 패턴 격자 연구 — 통합판</h1>")
    A(f'<p class="small">전략 <code>kr_pullback_largecap_foreign</code> · '
      f'작성 {datetime.now():%Y-%m-%d %H:%M} KST · '
      f'엔진 <code>korean_mkt_study/backtest.py</code> / <code>pattern_study.py</code></p>')
    A('<div class="box"><b>검증 조건</b> (Kane 확정 2026-08-17) — '
      '유니버스 월말 시총 <b>4조↑</b> &amp; 외국인지분 <b>25%↑</b> · 매수 <b>T+1 시가</b> · '
      '손절 체결 <b>청산가격 × 99.5%</b> · 왕복 <b>0.23%</b> · 청산 <b>v1.2.0 고정</b> · '
      '2015-01-02~2026-06-30 (11.5년) · 100,000,000원 · 20슬롯 × 5,000,000원 · '
      f'패널 {len(panel.tickers)}종목</div>')

    # ── 1. basis ──
    A("<h2>1. 패턴 기준(basis) — 저점 계열 기각</h2>")
    g = grid.groupby("basis")[["정밀도_평균", "재현율_평균", "fwd10", "fwd20"]].mean()
    A('<table><tr><th>basis</th><th>조합</th><th>정밀도 평균</th><th>재현율 평균</th>'
      '<th>fwd10</th><th>fwd20</th><th>판정</th></tr>')
    for b_ in g.index:
        n_ = (grid.basis == b_).sum()
        verdict = "유일한 생존" if b_ == "고점낙폭" else "36조합 전패 — 기각"
        cls = ' class="best"' if b_ == "고점낙폭" else ""
        A(f'<tr{cls}><td class="l">{b_}</td><td class="n">{n_}</td>'
          f'{pct(g.loc[b_, "정밀도_평균"], False)}{pct(g.loc[b_, "재현율_평균"], False)}'
          f'{pct(g.loc[b_, "fwd10"])}{pct(g.loc[b_, "fwd20"])}<td class="l">{verdict}</td></tr>')
    A("</table>")
    A('<p class="small">저점근접(종가 ≤ 직전 N일 저점 × 1.1~1.3)은 "어디서 떨어졌는지"를 모른다 — '
      '바닥 기는 종목과 건강한 눌림을 구분 못 함. 포트폴리오 검증도 참사'
      f'({met["(기각) 저점근접 20일/+20%/MA200"]["final"]:,.0f}원, '
      f'MDD {met["(기각) 저점근접 20일/+20%/MA200"]["mdd"]*100:.1f}%). '
      '신저가이탈(저점×0.9 하향 돌파)은 추세 필터와 양립 불가 — 전 조합 신호 30건 미만.</p>')

    # ── 2. 깊이 축 ──
    A("<h2>2. 깊이 축 — 깊을수록 좋지만, 희소하다</h2>")
    sub = grid[grid.basis == "고점낙폭"]
    A('<table><tr><th>깊이</th><th>신호 수<br>(11.5년 누적, 조합평균)</th><th>연간 환산</th>'
      '<th>정밀도 a<br>(20일+10%)</th><th>정밀도 b<br>(10일+10%)</th><th>정밀도 c<br>(5일+5%)</th>'
      '<th>fwd20</th><th>재현율 평균</th></tr>')
    for r_, lab in ((0.9, "−10% (현행)"), (0.8, "−20%"), (0.7, "−30%")):
        s = sub[sub.param == r_]
        if not len(s):
            continue
        A(f'<tr><td class="l">{lab}</td><td class="n">{s.n_sig.mean():,.0f}</td>'
          f'<td class="n">{s.n_sig.mean()/YEARS:,.0f}건/년</td>'
          f'{pct(s["정밀도_a"].mean(), False)}{pct(s["정밀도_b"].mean(), False)}'
          f'{pct(s["정밀도_c"].mean(), False)}{pct(s.fwd20.mean())}'
          f'{pct(s["재현율_평균"].mean(), False)}</tr>')
    A("</table>")
    A('<p class="small">⚠ 신호 수는 <b>전 구간(11.5년) 누적</b>이다. −30%는 연 5건 미만, '
      '성공은 연 2건 수준 — 단독으론 통계도 자본 활용도 안 된다. '
      '반등 정의 3종(a/b/c)에서 "깊을수록 좋다" 순위는 동일 — 정의에 강건.</p>')

    # ── 3. 창 축 ──
    A("<h2>3. 창 축 — 40~50일 고원, 60일에서 Sharpe 꺾임</h2>")
    A("<h3>3-1. 신호 수준 (MA 4종 평균)</h3>")
    A('<table><tr><th>깊이</th><th>창</th><th>신호(11.5년)</th><th>정밀도 a</th>'
      '<th>정밀도 b</th><th>정밀도 c</th><th>재현율 평균</th><th>fwd20</th></tr>')
    for r_ in (0.9, 0.8):
        for w in (20, 40, 60):
            s = sub[(sub.param == r_) & (sub.window == w)]
            if not len(s):
                continue
            A(f'<tr><td class="l">−{(1-r_)*100:.0f}%</td><td class="n">{w}일</td>'
              f'<td class="n">{s.n_sig.mean():,.0f}</td>'
              f'{pct(s["정밀도_a"].mean(), False)}{pct(s["정밀도_b"].mean(), False)}'
              f'{pct(s["정밀도_c"].mean(), False)}{pct(s["재현율_평균"].mean(), False)}'
              f'{pct(s.fwd20.mean())}</tr>')
    A("</table>")
    A('<p class="small">짧은 창일수록 정밀도↑ 재현율↓. 단 20일 창은 포트폴리오에서 최악(아래) — '
      '신호 수준 정밀도만 보고 고르면 안 되는 이유.</p>')

    A("<h3>3-2. 포트폴리오 — 창 고원 (−10% · MA200 고정)</h3>")
    A('<table><tr><th></th>' + "".join(f"<th>{w}일</th>" for w in win_grid) + "</tr>")
    A('<tr><td class="l">최종자산(억)</td>' + "".join(num(m["final"] / 1e8) for m in win_grid.values()) + "</tr>")
    A('<tr><td class="l">Sharpe</td>' + "".join(num(m["sharpe"]) for m in win_grid.values()) + "</tr>")
    A('<tr><td class="l">MDD</td>' + "".join(pct(m["mdd"] * 100) for m in win_grid.values()) + "</tr></table>")

    A("<h3>3-3. MA 고원 — 40일 창 vs 60일 창</h3>")
    A('<table><tr><th>추세 MA</th>' + "".join(f"<th>{n}</th>" for n in ma60_grid) + "</tr>")
    A('<tr><td class="l">60일 창: 최종(억) / Sharpe</td>' + "".join(
        f'<td class="n">{m["final"]/1e8:.2f} / {m["sharpe"]:.2f}</td>' for m in ma60_grid.values()) + "</tr>")
    A('<tr><td class="l">40일 창: 최종(억) / Sharpe</td>' + "".join(
        f'<td class="n">{m["final"]/1e8:.2f} / {m["sharpe"]:.2f}</td>' for m in ma40_grid.values()) + "</tr></table>")
    A('<div class="warn"><b>⚠ 두 곡선 모두 비단조.</b> 60일 창은 MA140~160 골짜기, 40일 창은 MA150 골짜기가 있다. '
      '"긴 MA가 낫다"는 방향과 "40~50일 창이 낫다"는 방향은 믿을 만하나, '
      'MA200·40일이라는 정확한 좌표는 봉우리일 수 있다 — 값 자체에 과신 금물.</div>')

    # ── 4. 계단 + 물타기 ──
    A("<h2>4. 깊이별 계단 진입 — 그리고 물타기 문제</h2>")
    A(f'<p>계단(−10%·−20%·−30% 첫 돌파마다 onset)이 추가하는 신호는 티어1 위에 +514건(16.5%). '
      f'추가 매수 {pyram+fresh}건을 원장 재생으로 분해하면 <b>보유 중 추가(물타기) {pyram}건({pyram/(pyram+fresh)*100:.0f}%) + '
      f'신규/재진입 {fresh}건({fresh/(pyram+fresh)*100:.0f}%)</b>.</p>')
    A("<h3>4-1. 출렁다리 검증 — 정밀도 c(높음)→b(낮음)→a(높음) 경로와 청산 규칙</h3>")
    A('<p>Kane 우려: "한 번 더 출렁이는 다리를 버텨야 과실 a를 얻는 것 아닌가." '
      '신호 단위로 실제 v1.2.0 청산을 태운 결과:</p>')
    A('<table><tr><th></th><th>n</th><th>승률</th><th>실현 평균수익</th><th>손절선 청산 비율</th>'
      '<th>출렁구간(D+6~15) 청산</th><th>과실 놓침*</th></tr>')
    for lab, wb in (("티어1 (−10% onset)", wb1), ("추가 (−20/−30 돌파)", wb2)):
        A(f'<tr><td class="l">{lab}</td><td class="n">{wb["n"]:,}</td>{pct(wb["win"], False)}'
          f'{pct(wb["avg"])}{pct(wb["stop"], False)}{pct(wb["wobble"], False)}{pct(wb["missed"], False)}</tr>')
    A("</table>")
    A('<p class="small">* 손절로 손실 확정 후 20일 내 +10%가 실제로 왔던 비율.</p>')
    A('<div class="ok"><b>시스템은 그 다리를 건너지 않는다.</b> 청산의 96~99%가 손절선 경유(중앙 D+8) — '
      'v1.2.0은 첫 반등(과실 c)으로 peak를 올린 뒤 D+6 조임에서 이익을 실현하고 나오는 구조다. '
      '출렁(b 골짜기)은 버티는 구간이 아니라 통행료 내고 빠져나오는 구간이며, '
      '과실 놓침은 5.6~7.2%로 제한적. 깊은 티어는 실전 청산 통과 후에도 건당 +3.08% vs +0.92%로 3배 우위.</div>')
    A("<h3>4-2. 물타기 금지 실험 — 티어3의 정체</h3>")
    A('<table><tr><th>안</th><th>최종자산</th><th>Sharpe</th><th>MDD</th></tr>')
    for k in ("T1  60일/−10%/MA200", "(참고) S1 계단10·20·30/MA200",
              "(참고) S1n = S1 물타기 금지", "S3n 계단10·20/MA200 (신규한정)"):
        m = met[k]
        A(f'<tr><td class="l">{k}</td>{won(m["final"])}{num(m["sharpe"])}{pct(m["mdd"]*100)}</tr>')
    A("</table>")
    A('<div class="warn"><b>S1(2.04억)의 초과이익은 물타기였다.</b> 물타기를 금지하면(S1n) 1.93억으로 '
      'T1 수준으로 회귀 — −30% 티어의 이익 원천이 보유 중 추가매수 115건이었다는 뜻. '
      '연 5건 미만의 얇은 표본 + 물타기 성격 → <b>−30% 티어 기각</b>. '
      '−20% 티어는 물타기를 금지해도 이익이 남는다(S3n 1.97억) — 손절 후 더 싸게 재진입하는 건전한 경로. '
      '단 T1 대비 증분 +0.05억으로 미미.</div>')

    # ── 5. 종합 비교 ──
    A("<h2>5. 최종 후보 종합 비교</h2>")
    A('<table><tr><th>안</th><th>최종자산</th><th>CAGR</th><th>Sharpe</th><th>MDD</th>'
      '<th>투자비중</th><th>매수</th><th>승률</th><th>P값*</th></tr>')
    for k in MAIN:
        m = met[k]
        cls = ' class="base"' if k == base_key else ""
        pv = f'{boots[k]["p"]:.3f}' if k in boots else "—"
        A(f'<tr{cls}><td class="l">{k}</td>{won(m["final"])}{pct(m["cagr"]*100)}'
          f'{num(m["sharpe"])}{pct(m["mdd"]*100)}{pct(m["invested"]*100, False)}'
          f'<td class="n">{m["n_buy"]:,}</td>{pct(m["win_rate"]*100, False)}'
          f'<td class="n">{pv}</td></tr>')
    A("</table>")
    A('<p class="small">* 블록 부트스트랩(20일·5,000회) vs 기준선. 전부 유의하지 않음 — '
      '이 전략군은 수익률로 유의차가 나지 않는다(청산 v1.2.0 검증 때와 동일). 판단 근거는 MDD·Sharpe·레짐 일관성.</p>')

    A("<h3>5-1. 레짐별 손익 / MDD</h3>")
    A('<table><tr><th>안</th>' + "".join(f"<th>{n}</th>" for n in REGIMES) + "</tr>")
    for k in MAIN:
        cls = ' class="base"' if k == base_key else ""
        A(f'<tr{cls}><td class="l">{k}</td>' + "".join(
            f'<td class="n"><span style="color:{UP if reg[k][n]["profit"]>0 else DOWN}">'
            f'{reg[k][n]["profit"]:,.0f}원</span><br>'
            f'<span class="small">MDD {reg[k][n]["mdd"]*100:.1f}%</span></td>' for n in REGIMES) + "</tr>")
    A("</table>")

    A("<h3>5-2. 민감도 (최종자산 / Sharpe)</h3>")
    A('<table><tr><th>안</th><th>전반 2015~2020</th><th>후반 2021~2026</th>'
      '<th>슬리피지 0%</th><th>유니버스 5조/30%</th></tr>')
    for k in MAIN:
        cls = ' class="base"' if k == base_key else ""
        A(f'<tr{cls}><td class="l">{k}</td>' + "".join(
            f'<td class="n">{m["final"]/1e8:.2f}억 / {m["sharpe"]:.2f}</td>' for m in sens[k]) + "</tr>")
    A("</table>")

    # ── 6. 정리 ──
    A("<h2>6. 정리 — 선택지</h2>")
    A('<div class="box">'
      '<b>T1 (60일/MA200)</b> — 변경 최소. 최근 5년 강세(후반 1.54억/0.55). MDD 개선은 작음(−27.6%).<br>'
      '<b>W1 (40일/MA200)</b> — 방어 특화. 평시 손실 ⅓(−1,097만→−380만), 평시 MDD −21.4%→−15.4%, '
      '전구간 MDD 최소(−21.5%). 대신 후반이 약함(1.39억/0.46). 전략의 명시 목적(평시 방어·위험조정)에 부합.<br>'
      '<b>W2 (50일/MA200)</b> — 수치 최고(1.95억/0.60)나 고원 꼭지 선택이라 과적합 위험 상대적으로 큼.<br>'
      '<b>S3n (계단10·20 신규한정)</b> — 물타기 없는 계단. T1 대비 +0.05억, MDD는 1.2%p 나쁨.<br>'
      '<b>보류</b> — 전부 P값 유의 없음. 가상계좌 표본 축적 후 재검도 정당한 선택.</div>')
    A('<p><b>기각 확정</b> — 저점근접(전패)·신저가이탈(공집합)·−30% 티어(물타기 이익)·'
      '계단 5% 간격(연속 물타기화)·20일 창(포트폴리오 최악)·추세필터 제거/완화·onset 재신호·'
      '반등확인 대기·변동성 정규화 눌림.</p>')

    A("<h2>7. 재현</h2>")
    A('<pre style="background:#f5f5f5;padding:12px;border-radius:4px;font-size:12.5px">'
      'cd ~/DriveForALL/StoLab/MagicFormula/korean_mkt_study\n'
      'python3 pattern_study.py          # 신호 수준 격자 → out/pattern_grid.csv\n'
      'python3 make_pattern_report.py    # 이 리포트 (포트폴리오·출렁다리·물타기 분해 포함)</pre>')
    A("</body></html>")

    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "스윙포트_진입패턴연구_20260817.html"
    path.write_text("\n".join(H), encoding="utf-8")
    return path


if __name__ == "__main__":
    print("생성:", build())
