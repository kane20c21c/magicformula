"""make_ma_kind_report.py — 스윙 포트 추세선 SMA vs EVWMA 비교 리포트.

입력: out/sweep_ma_kind.csv  (sweep_ma_kind.py 가 먼저 돌아야 한다)
출력: StockPortfolio/reports/스윙포트_추세선_SMA대EVWMA_20260904.html

make_report.py · make_pattern_report.py 와 같은 패턴 — 격자는 스윕 스크립트가 만들고
이 파일은 표시만 한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as B

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
REPORTS = HERE.parents[1] / "StockPortfolio" / "reports"
PANEL_START = "2010-01-04"

UP, DN = "#ef5350", "#1976D2"

CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Apple SD Gothic Neo',sans-serif;
 margin:0;padding:32px 40px;background:#fafafa;color:#222;line-height:1.65}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:34px 0 10px;
 padding-bottom:6px;border-bottom:2px solid #e0e0e0}
h3{font-size:15px;margin:22px 0 8px;color:#444}
.sub{color:#777;font-size:13px;margin-bottom:24px}
table{border-collapse:collapse;margin:12px 0;font-size:13px;background:#fff}
th,td{border:1px solid #e0e0e0;padding:7px 11px;text-align:right}
th{background:#f0f0f0;font-weight:600;text-align:center}
td.l,th.l{text-align:left}
tr.cur{background:#fffde7;font-weight:600}
tr.win td{background:#f5f5f5}
.up{color:#ef5350}.dn{color:#1976D2}
.box{padding:14px 18px;margin:16px 0;border-radius:6px;font-size:14px}
.verdict{background:#ffebee;border-left:5px solid #ef5350}
.note{background:#f5f5f5;border-left:5px solid #9e9e9e}
.warn{background:#fff8e1;border-left:5px solid #ffa726}
code{background:#eee;padding:1px 5px;border-radius:3px;font-size:12px}
.small{font-size:12px;color:#777}
"""


def _fmt_pct(v, digits=1, color=True):
    s = f"{v*100:.{digits}f}%"
    if not color:
        return s
    return f'<span class="{"up" if v > 0 else "dn"}">{s}</span>'


def gate_stats(p) -> dict:
    """추세 게이트 통과율 — '왜 달라지나' 의 근거."""
    sma = B.EntryParams(trend_ma_kind="sma", trend_ma=200, trend_min_periods=140)
    evw = B.EntryParams(trend_ma_kind="evwma", trend_ma=200, trend_min_periods=140)
    ma_s, ma_e = B._trend_ma(p, sma), B._trend_ma(p, evw)
    cl, el = p.close, p.elig
    both = ma_s.notna() & ma_e.notna() & cl.notna() & el
    w = p.dates >= pd.Timestamp("2015-01-01")
    S = ((cl > ma_s) & both).to_numpy()[w]
    E = ((cl > ma_e) & both).to_numpy()[w]
    M = both.to_numpy()[w]
    n = M.sum()
    d = ((ma_e - ma_s) / ma_s).where(both).to_numpy()[w]
    return dict(n=int(n), sma=S.sum() / n, evw=E.sum() / n,
                agree=((S == E) & M).sum() / n,
                evw_only=((~S) & E & M).sum() / n,
                sma_only=(S & (~E) & M).sum() / n,
                gap=float(np.nanmedian(d)))


def yearly(p) -> pd.DataFrame:
    out = {}
    for lab, kw in (("SMA200", dict(trend_ma_kind="sma", trend_ma=200, trend_min_periods=140)),
                    ("EVWMA200", dict(trend_ma_kind="evwma", trend_ma=200, trend_min_periods=140))):
        eq = B.run_backtest(p, B.EntryParams(**kw))["equity"]["equity"]
        yr = eq.resample("YE").last()
        prev, o = eq.iloc[0], {}
        for d, v in yr.items():
            o[d.year] = v / prev - 1.0
            prev = v
        out[lab] = o
    df = pd.DataFrame(out)
    df["차이"] = df["EVWMA200"] - df["SMA200"]
    return df


def build(df: pd.DataFrame, gs: dict, yr: pd.DataFrame) -> str:
    A = []
    a = A.append
    a(f"<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
      f"<title>스윙포트 추세선 — SMA vs EVWMA</title><style>{CSS}</style></head><body>")
    a("<h1>스윙 포트 추세선 — SMA vs EVWMA</h1>")
    a("<div class='sub'>2026-09-04 · Kane 요청 · "
      "<code>kr_pullback_largecap_foreign</code> v1.2.2.3 기준 · "
      "생성 <code>MagicFormula/korean_mkt_study/make_ma_kind_report.py</code></div>")

    # ── 판정 ──
    rel = df[df["워밍업"] == (df["기간"] * 0.7).round()]
    pairs = rel.pivot_table(index="기간", columns="종류",
                            values=["Sharpe", "MDD", "최종자산"])
    ws = int((pairs[("Sharpe", "SMA")] > pairs[("Sharpe", "EVWMA")]).sum())
    wm = int((pairs[("MDD", "SMA")] > pairs[("MDD", "EVWMA")]).sum())
    npair = len(pairs)
    a(f"""<div class='box verdict'><b>판정 — 기각. EVWMA 로 바꾸지 않는다.</b><br>
    기간을 <b>40~300 여덟 구간</b>으로 넓혀 재도 현행 워밍업(기간×0.7) 기준 SMA 가
    <b>Sharpe {ws}/{npair} · MDD {wm}/{npair}</b> 로 이긴다.
    현행 <b>SMA200</b>: Sharpe 0.68 · MDD −23.16% · 최종 202,751,563원 →
    <b>EVWMA200</b>: Sharpe 0.51 · MDD −28.61% · 최종 175,073,723원.<br>
    <span class='small'>단, 이 결론은 <b>전승이 아니다</b> — 워밍업을 엄격으로 바꾸면
    기간 120 한 점에서 EVWMA 가 앞선다(§3). 연도별 승패도 절반에 가깝다(§4).
    기각의 근거는 승패 수가 아니라 <b>위험 축(Sharpe·MDD)의 일관된 열세</b>다.</span></div>""")

    # ── 왜 ──
    a("<h2>1. 왜 나빠지나 — 게이트가 느슨해진다</h2>")
    a(f"""<p>EVWMA 선은 SMA 대비 중앙 <b>{gs['gap']*100:.2f}%</b> <b>아래</b>에 놓인다.
    추세 필터가 <code>close &gt; MA</code> 이므로 선이 낮으면 통과가 쉬워진다 —
    통과율 <b>{gs['sma']*100:.2f}% → {gs['evw']*100:.2f}%</b>.</p>""")
    a(f"""<table><tr><th class='l'>항목</th><th>값</th></tr>
    <tr><td class='l'>두 게이트 판정 일치</td><td>{gs['agree']*100:.2f}%</td></tr>
    <tr><td class='l'>EVWMA 만 통과 (EVWMA 가 추가로 사들이는 구간)</td>
        <td>{gs['evw_only']*100:.2f}%</td></tr>
    <tr><td class='l'>SMA 만 통과</td><td>{gs['sma_only']*100:.2f}%</td></tr>
    <tr><td class='l'>표본 (유니버스 편입일, 2015~)</td><td>{gs['n']:,}</td></tr></table>""")
    a("""<div class='box note'><b>해석.</b> EVWMA 는 "거래가 실린 가격대" 에 선이 머무는
    이동평균이다. 거래 없이 조용히 흘러내린 구간에서는 선이 위로 따라오지 않고 뒤처지므로,
    <b>추세가 이미 꺾인 종목도 게이트를 통과</b>한다. 매수 건수가
    1,500 → 1,594 건으로 늘어난 것이 그 결과다 — 늘어난 신호가 곧 손실이다.<br>
    <span class='small'>⚠ 이건 EVWMA 가 나쁜 지표라는 뜻이 아니라, <b>추세 게이트로는</b>
    맞지 않는다는 뜻이다. EVWMA 의 설계 의도(거래가 실린 평단)는 지지·저항 기준선이지
    추세 방향 판정기가 아니다.</span></div>""")

    # ── 기간 스윕 ──
    a("<h2>2. 기간 스윕 — 어느 창에서도 뒤집히지 않는다</h2>")
    a("<p>완화 워밍업(기간×0.7, 현행 200/140 비율) 기준. "
      "<b>EVWMA 는 기간이 길수록 단조 악화</b>하고, SMA 는 150~200 에 봉우리가 있다.</p>")
    a("<table><tr><th>기간</th><th colspan='3'>SMA</th><th colspan='3'>EVWMA</th>"
      "<th rowspan='2'>Sharpe 우세</th></tr>"
      "<tr><th>Sharpe</th><th>MDD</th><th>최종자산</th>"
      "<th>Sharpe</th><th>MDD</th><th>최종자산</th></tr>")
    for n in pairs.index:
        s_sh, e_sh = pairs.loc[n, ("Sharpe", "SMA")], pairs.loc[n, ("Sharpe", "EVWMA")]
        s_md, e_md = pairs.loc[n, ("MDD", "SMA")], pairs.loc[n, ("MDD", "EVWMA")]
        s_f, e_f = pairs.loc[n, ("최종자산", "SMA")], pairs.loc[n, ("최종자산", "EVWMA")]
        cur = " class='cur'" if n == 200 else ""
        win = "SMA" if s_sh > e_sh else "EVWMA"
        a(f"<tr{cur}><td>{n}{' ★' if n == 200 else ''}</td>"
          f"<td>{s_sh:.2f}</td><td>{s_md*100:.2f}%</td><td>{s_f:,.0f}</td>"
          f"<td>{e_sh:.2f}</td><td>{e_md*100:.2f}%</td><td>{e_f:,.0f}</td>"
          f"<td>{win}</td></tr>")
    a("</table>")
    a("""<div class='box warn'><b>⚠ 최종자산만 보면 짧은 기간에서 EVWMA 가 앞선다</b>
    (40·60·120일). 하지만 같은 구간의 MDD 가 전부 더 깊다 — 수익이 아니라
    <b>위험을 더 진 대가</b>다. 그래서 판정 기준을 Sharpe·MDD 로 잡았다.
    STRATEGY.md §8 의 "규칙 비교는 라이브 조건으로" 와 같은 취지 —
    유리해 보이는 단일 지표 하나로 채택하지 않는다.</div>""")

    # ── 워밍업 ──
    a("<h2>3. 워밍업은 원인이 아니다</h2>")
    a("<p>EVWMA 의 유동물량 V 에 <code>min_periods</code> 가 필요해서, "
      "LLV 운영 컬럼과 같은 <b>엄격(=n)</b> 과 현행 SMA 와 조건을 맞춘 <b>완화(=n×0.7)</b> "
      "를 모두 쟀다. 두 정책의 차이는 MA 종류의 차이보다 훨씬 작다.</p>")
    d2 = df.copy()
    d2["strict"] = d2["워밍업"] == d2["기간"]
    piv = d2.pivot_table(index="기간", columns=["종류", "strict"], values="Sharpe")

    flips = []          # SMA↔EVWMA 우열이 뒤집힌 조합
    for n in piv.index:
        for strict in (False, True):
            if piv.loc[n, ("EVWMA", strict)] > piv.loc[n, ("SMA", strict)]:
                flips.append((n, "엄격" if strict else "완화"))
    npairs = piv.shape[0] * 2

    a("<table><tr><th>기간</th><th>SMA 완화</th><th>SMA 엄격</th>"
      "<th>EVWMA 완화</th><th>EVWMA 엄격</th></tr>")
    for n in piv.index:
        def g(k, strict):
            v = piv.loc[n, (k, strict)]
            return "—" if pd.isna(v) else f"{v:.2f}"
        cur = " class='cur'" if n == 200 else ""
        a(f"<tr{cur}><td>{n}</td><td>{g('SMA', False)}</td><td>{g('SMA', True)}</td>"
          f"<td>{g('EVWMA', False)}</td><td>{g('EVWMA', True)}</td></tr>")
    a("</table>")

    if flips:
        fl = ", ".join(f"기간 {n}·{w}" for n, w in flips)
        a(f"""<div class='box warn'><b>⚠ 예외가 {len(flips)}건 있다 — 전부 이기는 게 아니다.</b><br>
        Sharpe 기준 {npairs}개 짝 중 SMA 가 <b>{npairs - len(flips)}개</b>에서 이기고,
        <b>{fl}</b> 에서는 EVWMA 가 앞선다
        ({', '.join(f"{piv.loc[n, ('EVWMA', w == '엄격')]:.2f} vs {piv.loc[n, ('SMA', w == '엄격')]:.2f}"
                    for n, w in flips)}).<br>
        다만 <b>같은 기간의 완화판에서는 도로 뒤집히고</b>, 이 조합의 MDD 는 여전히 SMA 가 낫다.
        워밍업 한 칸 차이로 우열이 오가는 지점은 <b>고원의 잡음</b>으로 보는 것이 맞다 —
        STRATEGY.md §3 이 MA 곡선의 비단조성을 두고 경고한 것과 같은 성격이다.
        <b>이 한 점을 근거로 EVWMA 를 채택하지는 않는다.</b></div>""")
    else:
        a("<p class='small'>Sharpe 기준, 워밍업 정책을 바꿔도 우열이 뒤집히지 않는다.</p>")
    a("<p class='small'>차이의 주된 원인은 워밍업이 아니라 MA 종류다 — "
      "워밍업 두 정책의 Sharpe 차이(중앙 "
      f"{float(np.nanmedian(np.abs(piv.xs(True, level='strict', axis=1).values - piv.xs(False, level='strict', axis=1).values))):.3f}"
      ") 가 SMA↔EVWMA 차이보다 작다.</p>")

    # ── 연도별 ──
    a("<h2>4. 연도별 — 승패는 갈리지만 손실이 위기 구간에 몰린다</h2>")
    win = int((yr["차이"] > 0).sum())
    a(f"""<p>EVWMA200 이 SMA200 을 이긴 해는 <b>{win}/{len(yr)}</b> 로 절반에 가깝다.
    <b>연도 승패만 보면 기각 근거가 약하다</b> — 기각의 근거는 승패 수가 아니라
    <b>지는 해에 더 크게 진다</b>는 데 있다.</p>""")
    a("<table><tr><th>연도</th><th>SMA200</th><th>EVWMA200</th><th>차이</th></tr>")
    for y, r in yr.iterrows():
        cls = " class='win'" if r["차이"] > 0 else ""
        a(f"<tr{cls}><td>{y}</td><td>{_fmt_pct(r['SMA200'])}</td>"
          f"<td>{_fmt_pct(r['EVWMA200'])}</td><td>{_fmt_pct(r['차이'])}</td></tr>")
    a("</table>")
    lose = yr[yr["차이"] < 0]["차이"]
    winr = yr[yr["차이"] > 0]["차이"]
    a(f"""<div class='box note'><b>비대칭.</b> 이긴 해 평균 <b>+{winr.mean()*100:.2f}%p</b>,
    진 해 평균 <b>{lose.mean()*100:.2f}%p</b>. 크게 지는 해가
    2018(−3.3%p)·2020(−6.3%p)·2021(−6.5%p)·2022(−4.3%p) 로
    <b>하락·전환 구간에 몰려 있다</b> — 느슨한 게이트가 위기에서 비용을 청구한다는
    §1 의 해석과 맞는다. 외국인 지분율 필터가 "위기 방어 필터" 였던 것과 같은 구조다
    (STRATEGY.md §2).</div>""")

    # ── 한계 ──
    a("<h2>5. 한계 — 이 결론이 닿지 않는 곳</h2>")
    a("""<ul>
    <li><b>거래량 정본 문제.</b> <code>data/prices.parquet</code> 는 2026-07-04 스냅샷이라
    2026-08-16 거래량 사고 이전분이고 월별 중앙값에 인위적 단차는 안 보이지만,
    LLV 의 KRX+NXT 통합(UN) 거래량과 같은 계열인지는 <b>대조하지 않았다</b>.
    EVWMA 는 거래량 가중이라 원천이 다르면 값이 달라진다.</li>
    <li><b>추세 게이트 용도에 한정된 결론이다.</b> EVWMA 를 손절선·평단 기준선·
    이격 지표로 쓰는 것은 여기서 재지 않았다 — 기각한 것은
    <b>"진입 추세 필터로서의 EVWMA"</b> 하나다.</li>
    <li><b>다른 축과의 상호작용 미검증.</b> 눌림 창(40일)·깊이(−10%)는 SMA 를 전제로
    굳은 값이다. EVWMA 에 맞는 창·깊이가 따로 있을 가능성은 남아 있다.</li>
    <li><b>신규 상장주 편향.</b> EVWMA 는 첫 유효일 값이 그날 종가라 워밍업 직후
    게이트를 거의 무조건 통과한다. 패널을 2010-01 부터 실어 매매 구간(2015~) 밖으로
    뺐지만, 상장이 늦은 종목은 구조적으로 이 편향을 피할 수 없다.</li>
    </ul>""")

    a("<h2>재현</h2><pre style='background:#fff;padding:12px;border:1px solid #e0e0e0;"
      "font-size:12px'>cd MagicFormula/korean_mkt_study\n"
      "python3 sweep_ma_kind.py          # out/sweep_ma_kind.csv\n"
      "python3 make_ma_kind_report.py    # 이 리포트</pre>")
    a("</body></html>")
    return "\n".join(A)


def main() -> int:
    csv = OUT / "sweep_ma_kind.csv"
    if not csv.exists():
        print(f"먼저 sweep_ma_kind.py 를 돌릴 것 — {csv} 없음", file=sys.stderr)
        return 1
    df = pd.read_csv(csv)
    p = B.load_panel(B.UniverseParams(), B.VolScaleParams(), start=PANEL_START)
    html = build(df, gate_stats(p), yearly(p))

    if not REPORTS.is_dir():
        print(f"리포트 폴더 없음 — 저장 건너뜀: {REPORTS}", file=sys.stderr)
        return 2
    path = REPORTS / "스윙포트_추세선_SMA대EVWMA_20260904.html"
    path.write_text(html, encoding="utf-8")
    print(f"저장: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
