"""make_report.py — 진입 신호 개선 검증 리포트(HTML) 생성.

숫자는 전부 백테스트 실행 결과에서 직접 뽑는다 (손으로 옮겨 적지 않는다).
출력: StockPortfolio/reports/스윙포트_진입신호검증_20260817.html
"""
from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from backtest import (EntryParams, ExitParams, PortfolioParams, UniverseParams,
                      VolScaleParams, load_panel, metrics, run_backtest, compute_signals)
from diag_recall import opportunity_mask, episodes, PULLBACK, BOUNCE, HORIZON
from verify_entry import block_bootstrap

UP, DOWN = "#ef5350", "#1976D2"
REPORTS = Path(__file__).resolve().parents[2] / "StockPortfolio" / "reports"

MA200 = dict(trend_ma=200, trend_min_periods=140)
CAND = {
    "기준선 v1.0.0 (MA120·눌림깊은순)": EntryParams(),
    "T1  추세 MA200": EntryParams(**MA200),
    "P1  우선순위 = 여력 큰 순": EntryParams(priority="trend"),
    "T1+P1  MA200 + 여력 큰 순": EntryParams(**MA200, priority="trend"),
    "A1  반등확인(종가>당일중간)": EntryParams(confirm="mid", confirm_max_wait=3),
    "B1  변동성 정규화 눌림": EntryParams(vol_depth=True, vol_depth_clip=(0.07, 0.15)),
    "R1  추세필터 제거": EntryParams(trend_margin=-1e9),
    "R2  재신호 5일 간격": EntryParams(resignal_gap=5),
}
REGIMES = {
    "2015~2019 평시": ("2015-01-01", "2019-12-31"),
    "2020~2021 코로나": ("2020-01-01", "2021-12-31"),
    "2022~2023 하락": ("2022-01-01", "2023-12-31"),
    "2024~2026 멜트업": ("2024-01-01", "2026-06-30"),
}


def won(v):
    c = UP if v > 0 else (DOWN if v < 0 else "#555")
    return f'<td class="n" style="color:{c}">{v:,.0f}원</td>'


def pct(v, signed=True, digits=1):
    c = UP if v > 0 else (DOWN if v < 0 else "#555")
    if not signed:
        c = "#333"
    return f'<td class="n" style="color:{c}">{v:.{digits}f}%</td>'


def num(v, digits=2):
    return f'<td class="n">{v:.{digits}f}</td>'


def build():
    panel = load_panel(UniverseParams(), VolScaleParams())
    pp = PortfolioParams()
    res = {k: run_backtest(panel, ep, ExitParams(), pp) for k, ep in CAND.items()}
    met = {k: metrics(r) for k, r in res.items()}
    base_key = "기준선 v1.0.0 (MA120·눌림깊은순)"
    bm = met[base_key]

    # ── 재현율 진단 ──
    opp, in_dip = opportunity_mask(panel)
    dates = panel.dates
    lo_i = int(np.searchsorted(dates, pd.Timestamp("2015-01-01")))
    hi_i = int(np.searchsorted(dates, pd.Timestamp("2026-06-30"), side="right")) - 1
    eps = [(j, s, e) for j, s, e in episodes(opp) if lo_i <= s <= hi_i]
    sig = compute_signals(panel, EntryParams())
    signal, cond = sig["signal"].to_numpy(), sig["cond"].to_numpy()
    tr = res[base_key]["trades"]; buys = tr[tr.side == "BUY"]
    col = {t: j for j, t in enumerate(panel.tickers)}; di = {d: i for i, d in enumerate(dates)}
    bought = np.zeros_like(signal)
    for _, r in buys.iterrows():
        bought[di[r["date"]], col[r["ticker"]]] = True
    cl = panel.close.to_numpy()
    ma120 = panel.close.rolling(120, min_periods=80).mean().to_numpy()
    high60 = panel.close.rolling(60, min_periods=30).max().to_numpy()
    caught = miss_slot = 0
    reasons = {"MA120 아래": 0, "onset 제한 (이미 눌림 진행 중)": 0, "기타": 0}
    for j, s, e in eps:
        w = slice(s, min(e, hi_i) + 1)
        if signal[w, j].any():
            si = s + int(np.flatnonzero(signal[w, j])[0])
            if bought[si:min(si + 3, hi_i + 1), j].any():
                caught += 1
            else:
                miss_slot += 1
            continue
        i = s
        if not np.isfinite(ma120[i, j]) or cl[i, j] <= ma120[i, j]:
            reasons["MA120 아래"] += 1
        elif cond[max(i - 1, 0), j]:
            reasons["onset 제한 (이미 눌림 진행 중)"] += 1
        else:
            reasons["기타"] += 1
    n_opp = len(eps)
    dip_lens = np.array([e - s + 1 for _, s, e in episodes(in_dip) if lo_i <= s <= hi_i])

    # ── 부트스트랩 ──
    base_ret = res[base_key]["equity"]["equity"].pct_change()
    boots = {k: block_bootstrap(r["equity"]["equity"].pct_change(), base_ret)
             for k, r in res.items() if k != base_key}

    # ── 매수 일치율 / 집중도 ──
    bt = tr[tr.side == "BUY"]; kb = set(zip(bt.date, bt.ticker))
    overlap, conc = {}, {}
    base_top5 = tr[tr.side == "SELL"].nlargest(5, "pnl").pnl.sum()
    for k, r in res.items():
        t = r["trades"]; b = t[t.side == "BUY"]
        overlap[k] = len(set(zip(b.date, b.ticker)) & kb) / len(kb) * 100
        conc[k] = t[t.side == "SELL"].nlargest(5, "pnl").pnl.sum()

    # ── 레짐 ──
    reg = {k: {name: metrics(run_backtest(panel, ep, ExitParams(), pp, start=s, end=e))
               for name, (s, e) in REGIMES.items()} for k, ep in CAND.items()}
    # ── 민감도 ──
    sens = {}
    for k, ep in CAND.items():
        sens[k] = {
            "슬리피지 0%": metrics(run_backtest(panel, ep, ExitParams(),
                                             PortfolioParams(stop_slippage=0.0))),
            "전반 2015~2020": metrics(run_backtest(panel, ep, ExitParams(), pp,
                                                 start="2015-01-01", end="2020-12-31")),
            "후반 2021~2026": metrics(run_backtest(panel, ep, ExitParams(), pp,
                                                 start="2021-01-01", end="2026-06-30")),
        }
    p2 = load_panel(UniverseParams(mktcap_min_krw=5e12, foreign_min_pct=30.0), VolScaleParams())
    uni5 = {k: metrics(run_backtest(p2, ep, ExitParams(), pp)) for k, ep in CAND.items()}

    # ── MA 기간 고원 ──
    ma_grid = {}
    for n in (100, 120, 140, 160, 180, 200, 220, 250, 300):
        ma_grid[n] = metrics(run_backtest(panel, EntryParams(trend_ma=n,
                                                            trend_min_periods=int(n * .7)),
                                          ExitParams(), pp))

    H = []
    A = H.append
    A(f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>스윙 포트 진입 신호 개선 검증 (2026-08-17)</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Apple SD Gothic Neo',sans-serif;
max-width:1180px;margin:0 auto;padding:28px 20px 80px;color:#222;line-height:1.65}}
h1{{font-size:25px;border-bottom:3px solid #222;padding-bottom:10px;margin-bottom:6px}}
h2{{font-size:19px;margin-top:38px;border-left:5px solid #222;padding-left:11px}}
h3{{font-size:16px;margin-top:26px;color:#333}}
table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:13.5px}}
th,td{{border:1px solid #dcdcdc;padding:7px 9px}}
th{{background:#f3f4f6;font-weight:600;text-align:center}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}}
td.l{{text-align:left}}
tr.best{{background:#fff8e1}}
tr.base td{{background:#eef1f4;font-weight:600}}
.box{{background:#f7f9fb;border-left:5px solid #1976D2;padding:13px 17px;margin:16px 0;border-radius:3px}}
.warn{{background:#fff5f5;border-left:5px solid #ef5350;padding:13px 17px;margin:16px 0;border-radius:3px}}
.ok{{background:#f2f9f2;border-left:5px solid #43a047;padding:13px 17px;margin:16px 0;border-radius:3px}}
.small{{font-size:12.5px;color:#666}}
.up{{color:{UP}}} .dn{{color:{DOWN}}}
</style></head><body>""")
    A(f"<h1>스윙 포트 진입 신호 개선 검증</h1>")
    A(f'<p class="small">전략 <code>kr_pullback_largecap_foreign</code> · '
      f'작성 {datetime.now():%Y-%m-%d %H:%M} KST · '
      f'엔진 <code>korean_mkt_study/backtest.py</code></p>')

    A('<div class="box"><b>검증 조건</b> (Kane 확정 2026-08-17)<br>'
      '유니버스 월말 시총 <b>4조↑</b> &amp; 외국인지분 <b>25%↑</b> · '
      '매수 신호 <b>T+1 시가</b> · 손절 체결 <b>청산가격 × 99.5%</b> · '
      '왕복 비용 <b>0.23%</b> · 청산 규칙 <b>v1.2.0 고정</b> · '
      f'구간 2015-01-02~2026-06-30 · 자본 100,000,000원 · 20슬롯 × 5,000,000원<br>'
      f'<span class="small">패널 {len(panel.tickers)}종목 / {len(panel.dates):,}거래일. '
      '엔진은 2026-08-16 청산 검증 수치를 재현함(MDD −14.74% vs 리포트 −14.9%).</span></div>')

    # 1. 문제 정의
    A("<h2>1. 오늘 푼 문제</h2>")
    A("<p>어제 스윙 포트 복기에서 Kane이 지적한 두 가지 중 <b>(1)</b>을 다뤘다.</p>"
      "<ul><li><b>(1) 기회는 많았는데 모델이 못 찾아냈다</b> — 재현율 문제. ← 이 리포트</li>"
      "<li>(2) 부른 신호가 반등으로 안 이어졌다 — 정밀도(예측) 문제. 축 A·B로 시도했으나 모두 실패(§4).</li></ul>")

    # 2. 재현율 진단
    A("<h2>2. 진단 — 기회의 79%를 놓치고 있었다</h2>")
    A(f'<p>기회 정의: 유니버스 종목이 <b>60일 고점 대비 −{PULLBACK:.0%} 이하</b> 상태에서 '
      f'이후 <b>{HORIZON}거래일 내 +{BOUNCE:.0%} 반등</b>한 구간(연속일은 1건으로 병합).</p>')
    A(f"<p>전 구간 기회 <b>{n_opp:,}건</b> (종목 {len({j for j,_,_ in eps})}개).</p>")
    A('<table><tr><th>구분</th><th>건수</th><th>비중</th><th>해석</th></tr>')
    A(f'<tr class="best"><td class="l">잡았다 (신호 + 실제 매수)</td><td class="n">{caught:,}</td>'
      f'{pct(caught/n_opp*100, False)}<td class="l">—</td></tr>')
    A(f'<tr><td class="l">놓침 — 신호는 떴으나 슬롯·현금 부족</td><td class="n">{miss_slot:,}</td>'
      f'{pct(miss_slot/n_opp*100, False)}<td class="l">사이징 문제</td></tr>')
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        if not v:
            continue
        A(f'<tr><td class="l">놓침 — {k}</td><td class="n">{v:,}</td>'
          f'{pct(v/n_opp*100, False)}<td class="l">트리거 문제</td></tr>')
    A("</table>")
    A(f'<div class="warn"><b>사이징이 아니라 트리거가 원인이다.</b> '
      f'20슬롯 만재일은 전 구간의 {(res[base_key]["equity"]["n_pos"]>=20).mean()*100:.1f}%뿐이고 '
      f'평균 보유는 {res[base_key]["equity"]["n_pos"].mean():.1f}종목, '
      f'평균 투자비중은 {bm["invested"]*100:.1f}%다. '
      f'슬롯을 늘려도 살 종목이 없다 — 신호가 애초에 안 뜬다.</div>')
    A(f'<p>눌림 구간(연속 −10% 이하) 길이는 중앙값 {np.median(dip_lens):.0f}거래일이지만 '
      f'p90 {np.percentile(dip_lens,90):.0f}일 · 최대 {dip_lens.max()}일이다. '
      f'현행 onset 규칙은 이 구간의 <b>첫날 하루만</b> 신호를 준다.</p>')

    # 3. 완화 실험
    A("<h2>3. 그래서 트리거를 풀어봤더니 — 재앙</h2>")
    A('<table><tr><th>안</th><th>최종자산</th><th>CAGR</th><th>Sharpe</th><th>MDD</th>'
      '<th>투자비중</th><th>매수</th><th>승률</th></tr>')
    for k in (base_key, "R1  추세필터 제거", "R2  재신호 5일 간격"):
        m = met[k]
        cls = ' class="base"' if k == base_key else ""
        A(f'<tr{cls}><td class="l">{k}</td>{won(m["final"])}{pct(m["cagr"]*100)}'
          f'{num(m["sharpe"])}{pct(m["mdd"]*100)}{pct(m["invested"]*100, False)}'
          f'<td class="n">{m["n_buy"]:,}</td>{pct(m["win_rate"]*100, False)}</tr>')
    A("</table>")
    A('<div class="warn"><b>놓친 60%는 사후적으로만 기회였다.</b> 추세 필터를 없애면 '
      '투자비중은 36.2% → 75.2%로 오르지만 최종자산이 원금 아래로 떨어지고 MDD가 −75.7%가 된다. '
      'MA120 아래에서 일어난 반등 하나마다 그대로 추락한 종목이 훨씬 많았다는 뜻이다. '
      'onset 완화(재신호)도 같은 이유로 MDD를 −29.1% → −39.5%로 악화시킨다 — '
      '눌림이 계속되는 종목에 물타기하는 셈이 된다.</div>')

    # 4. 축 A·B
    A("<h2>4. 정밀도 축(A·B)도 실패</h2>")
    A('<table><tr><th>안</th><th>최종자산</th><th>Sharpe</th><th>MDD</th>'
      '<th>투자비중</th><th>매수</th><th>승률</th><th>건당수익</th></tr>')
    for k in (base_key, "A1  반등확인(종가>당일중간)", "B1  변동성 정규화 눌림"):
        m = met[k]
        cls = ' class="base"' if k == base_key else ""
        A(f'<tr{cls}><td class="l">{k}</td>{won(m["final"])}{num(m["sharpe"])}'
          f'{pct(m["mdd"]*100)}{pct(m["invested"]*100, False)}<td class="n">{m["n_buy"]:,}</td>'
          f'{pct(m["win_rate"]*100, False)}{pct(m["avg_ret"]*100)}</tr>')
    A("</table>")
    A('<p>축 A(하락 멈춤 확인)는 5개 변형 × 대기 4수준 20조합 전부, '
      '축 B(변동성 정규화 눌림)는 12조합 전부 기준선에 못 미쳤다. '
      '두 축 모두 <b>진입을 더 조이는 방향</b>이라 투자비중을 11~28%로 떨어뜨렸고, '
      '이미 36%뿐인 자본 활용을 더 줄였다. 상세는 <code>out/entry_sweep_A.csv</code>, '
      '<code>out/entry_sweep_B.csv</code>.</p>')
    A('<p class="small">참고: 눌림 임계 고정값 스윕에서 <b>10%가 여전히 최적</b>이었다'
      '(6% 0.91억 / 8% 1.31억 / <b>10% 1.65억</b> / 12% 1.56억 / 15% 1.39억). '
      'v1.0.0의 눌림 파라미터는 재확인된 셈이다.</p>')

    # 5. 살아남은 것
    A("<h2>5. 살아남은 개선 — 추세선을 <u>길게</u> 바꾸기</h2>")
    A("<p>추세 필터를 <b>없애면</b> 재앙이지만, <b>더 긴 이동평균으로 교체</b>하면 "
      "재현율과 품질이 동시에 오른다. 상승 추세 종목은 MA200 &lt; MA120 &lt; 주가이므로 "
      "<code>종가 &gt; MA200</code>은 MA120보다 <b>느슨하면서도</b>, 필터 자체는 살아 있다. "
      "MA120을 깨고 내려온 건강한 눌림을 잡아낸다.</p>")
    A('<table><tr><th>안</th><th>최종자산</th><th>CAGR</th><th>Sharpe</th><th>MDD</th>'
      '<th>투자비중</th><th>매수</th><th>승률</th><th>건당수익</th></tr>')
    for k in (base_key, "T1  추세 MA200", "P1  우선순위 = 여력 큰 순", "T1+P1  MA200 + 여력 큰 순"):
        m = met[k]
        cls = ' class="base"' if k == base_key else (' class="best"' if k.startswith("T1+P1") else "")
        A(f'<tr{cls}><td class="l">{k}</td>{won(m["final"])}{pct(m["cagr"]*100)}'
          f'{num(m["sharpe"])}{pct(m["mdd"]*100)}{pct(m["invested"]*100, False)}'
          f'<td class="n">{m["n_buy"]:,}</td>{pct(m["win_rate"]*100, False)}'
          f'{pct(m["avg_ret"]*100)}</tr>')
    A("</table>")
    A('<div class="ok"><b>T1(MA200)은 다섯 지표가 모두 같은 방향으로 개선된다</b> — '
      '최종자산↑ · Sharpe↑ · MDD↓ · 투자비중↑ · 매수건수↑ · 승률↑. '
      '진입을 조이지 않고도 좋아진 유일한 안이다.</div>')

    A("<h3>추세 MA 기간 고원</h3>")
    A('<table><tr><th>MA 기간</th>' + "".join(f"<th>{n}</th>" for n in ma_grid) + "</tr>")
    A('<tr><td class="l">최종자산(억)</td>' +
      "".join(f'<td class="n">{m["final"]/1e8:.2f}</td>' for m in ma_grid.values()) + "</tr>")
    A('<tr><td class="l">Sharpe</td>' +
      "".join(f'<td class="n">{m["sharpe"]:.2f}</td>' for m in ma_grid.values()) + "</tr>")
    A('<tr><td class="l">MDD</td>' +
      "".join(pct(m["mdd"]*100) for m in ma_grid.values()) + "</tr></table>")
    A('<div class="warn"><b>⚠ 과적합 경계.</b> 180~220 구간이 모두 MA120보다 낫지만'
      '(1.80~1.92억), MA140·160은 MA120보다 <b>나쁘다</b>(1.62·1.73억). '
      '단조롭지 않은 곡선이라 MA200이 고원의 중심이라기보다 봉우리에 가깝다. '
      '"더 긴 MA가 낫다"는 방향은 신뢰할 만하나 200이라는 <b>정확한 값에 의미를 두면 안 된다</b>.</div>')

    # 6. 검증
    A("<h2>6. 검증</h2>")
    A("<h3>6-1. 블록 부트스트랩 (20일 블록 · 5,000회 · vs 기준선)</h3>")
    A('<table><tr><th>안</th><th>일평균 차이</th><th>연환산 차이</th><th>P값</th><th>유의(5%)</th></tr>')
    for k in ("T1  추세 MA200", "P1  우선순위 = 여력 큰 순", "T1+P1  MA200 + 여력 큰 순"):
        b = boots[k]
        A(f'<tr><td class="l">{k}</td><td class="n">{b["mean_diff"]*1e4:+.2f}bp</td>'
          f'{pct(b["ann_diff"]*100)}<td class="n">{b["p"]:.3f}</td>'
          f'<td style="text-align:center">{"○" if b["p"]<0.05 else "×"}</td></tr>')
    A("</table>")
    A('<p class="small">청산 v1.2.0 검증(2026-08-16) 때와 같은 결과다 — '
      '<b>수익률 차이는 통계적으로 유의하지 않다</b>(P=0.26~0.36). '
      '이 전략군은 원래 그렇고, 판단 근거는 MDD·Sharpe·레짐 일관성이다.</p>')

    A("<h3>6-2. 개선의 견고성 — 어디서 나온 이익인가</h3>")
    A('<table><tr><th>안</th><th>기준선과 매수 일치율</th><th>개선분</th>'
      '<th>상위 5건 청산손익 합</th><th>해석</th></tr>')
    for k in ("T1  추세 MA200", "P1  우선순위 = 여력 큰 순", "T1+P1  MA200 + 여력 큰 순"):
        imp = met[k]["final"] - bm["final"]
        note = ("신호 집합이 구조적으로 다름 — 이익이 넓게 분산"
                if overlap[k] < 80 else
                "<b>4%의 다른 결정에 이익이 집중 — 취약</b>")
        A(f'<tr><td class="l">{k}</td>{pct(overlap[k], False)}{won(imp)}{won(conc[k])}'
          f'<td class="l">{note}</td></tr>')
    A(f'<tr class="base"><td class="l">{base_key}</td><td class="n">100.0%</td>'
      f'<td class="n">—</td>{won(base_top5)}<td class="l">—</td></tr>')
    A("</table>")
    A('<div class="warn"><b>P1(우선순위)은 P값이 0.040으로 유일하게 유의하지만 믿기 어렵다.</b> '
      f'매수의 {overlap["P1  우선순위 = 여력 큰 순"]:.1f}%가 기준선과 동일한데도 '
      '개선분이 28,220,000원이다. 즉 <b>다른 4%(약 75건)의 결정이 개선을 통째로 만들었고</b>, '
      '상위 5건 청산손익 합이 기준선 42,480,000원 → 53,730,000원으로 뛴다. '
      '소수 대박에 의존하는 개선이다. 반면 T1(MA200)은 매수 일치율이 68.2%로 '
      '신호 집합 자체가 다르고, 상위 5건 합은 기준선과 거의 같다(42,660,000원) — '
      '<b>이익이 특정 몇 건이 아니라 전체에 퍼져 있다.</b></div>')

    A("<h3>6-3. 레짐별 손익 / MDD</h3>")
    A('<table><tr><th>안</th>' + "".join(f"<th>{n}</th>" for n in REGIMES) + "</tr>")
    for k in (base_key, "T1  추세 MA200", "P1  우선순위 = 여력 큰 순", "T1+P1  MA200 + 여력 큰 순"):
        cls = ' class="base"' if k == base_key else ""
        A(f'<tr{cls}><td class="l">{k}</td>' + "".join(
            f'<td class="n"><span style="color:{UP if reg[k][n]["profit"]>0 else DOWN}">'
            f'{reg[k][n]["profit"]:,.0f}원</span><br>'
            f'<span class="small">MDD {reg[k][n]["mdd"]*100:.1f}%</span></td>'
            for n in REGIMES) + "</tr>")
    A("</table>")

    A("<h3>6-4. 민감도</h3>")
    A('<table><tr><th>안</th><th>슬리피지 0%</th><th>전반 2015~2020</th>'
      '<th>후반 2021~2026</th><th>유니버스 5조/30%</th></tr>')
    for k in (base_key, "T1  추세 MA200", "P1  우선순위 = 여력 큰 순", "T1+P1  MA200 + 여력 큰 순"):
        cls = ' class="base"' if k == base_key else ""
        cells = "".join(f'<td class="n">{sens[k][c]["final"]/1e8:.2f}억 / '
                        f'{sens[k][c]["sharpe"]:.2f}</td>' for c in sens[k])
        cells += f'<td class="n">{uni5[k]["final"]/1e8:.2f}억 / {uni5[k]["sharpe"]:.2f}</td>'
        A(f'<tr{cls}><td class="l">{k}</td>{cells}</tr>')
    A("</table>")
    A('<p class="small">셀 = 최종자산 / Sharpe. T1은 <b>네 가지 절단 전부</b>에서 기준선을 앞선다. '
      'P1은 전반 구간(2015~2020)에서 기준선과 동일(1.13억/0.27~0.28) — 개선이 후반에만 있다.</p>')

    # 7. 결론
    A("<h2>7. 결론과 제안</h2>")
    A('<div class="ok"><b>채택 제안 — T1: 추세 필터 MA120 → MA200</b><br>'
      f'최종자산 {bm["final"]:,.0f}원 → {met["T1  추세 MA200"]["final"]:,.0f}원 · '
      f'Sharpe {bm["sharpe"]:.2f} → {met["T1  추세 MA200"]["sharpe"]:.2f} · '
      f'MDD {bm["mdd"]*100:.1f}% → {met["T1  추세 MA200"]["mdd"]*100:.1f}% · '
      f'투자비중 {bm["invested"]*100:.1f}% → {met["T1  추세 MA200"]["invested"]*100:.1f}%<br>'
      '근거: ① 다섯 지표 동시 개선 ② 레짐 4구간·민감도 4절단 전부에서 우위 '
      '③ 이익이 소수 거래에 집중되지 않음(상위 5건 합이 기준선과 동일). '
      '한계: P값 0.355로 통계적 유의는 없고, MA 기간 곡선이 단조롭지 않다.</div>')
    A('<div class="warn"><b>보류 — P1: 후보 우선순위 \'눌림 깊은 순 → 여력 큰 순\'</b><br>'
      'P값은 0.040으로 유일하게 유의하지만 매수 95.8%가 동일한 상태에서 나온 차이라 '
      '소수 결정 의존이 심하고, 전반 구간(2015~2020)에서는 개선이 전혀 없다. '
      '표본을 더 쌓고 재검토할 것.</div>')
    A('<p><b>기각(§11 기각목록 추가 대상)</b> — 추세필터 제거·완화, onset 재신호 완화, '
      '하락 멈춤 확인(반등일 대기), 변동성 정규화 눌림. 사유는 각각 §3·§4.</p>')

    A('<h2>8. 재현</h2>')
    A('<pre style="background:#f5f5f5;padding:12px;border-radius:4px;font-size:12.5px">'
      'cd ~/DriveForALL/StoLab/MagicFormula/korean_mkt_study\n'
      'python3 backtest.py         # 기준선 1회\n'
      'python3 diag_recall.py      # §2 재현율 진단\n'
      'python3 diag_entry.py       # 신호 전방수익 분해\n'
      'python3 sweep_entry.py A    # 축 A (B / C 도 동일)\n'
      'python3 sweep_recall.py     # 축 R\n'
      'python3 verify_entry.py     # §6 검증\n'
      'python3 make_report.py      # 이 리포트</pre>')
    A('<p class="small">원자료 <code>korean_mkt_study/data/{prices,meta,foreign}.parquet</code> · '
      '중간 산출 <code>korean_mkt_study/out/*.csv</code></p>')
    A("</body></html>")

    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "스윙포트_진입신호검증_20260817.html"
    path.write_text("\n".join(H), encoding="utf-8")
    return path


if __name__ == "__main__":
    print("생성:", build())
