"""k200_lead_study.py 결과 → 상위 10종목 엑셀 (out/k200_lead/KOSPI200_선행종목_20261123.xlsx)"""
import os, numpy as np, pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "out", "k200_lead")
L = pd.read_csv(f"{OUT}/yearly_long.csv", dtype={"ticker": str}); S = pd.read_csv(f"{OUT}/summary.csv", dtype={"ticker": str})
m = pd.read_parquet(f"{ROOT}/data_2015/meta.parquet", columns=["name"]).reset_index()
nm = m.sort_values("date").groupby("ticker")["name"].last()
S["name"] = S.ticker.map(nm)
top = S[(S.years >= 5) & (S.bal_slope > 0)].sort_values("bal_recent3", ascending=False).head(10)
yrs = sorted(L.year.unique())
P = L.pivot(index="ticker", columns="year", values="bal"); Pi = L.pivot(index="ticker", columns="year", values="ic")
base = L.groupby("year")[["k200_mom_bal", "k200_mom_ic", "k200_up_share"]].first()
med = L.groupby("year")[["bal", "ic"]].median()
nr, ns, nb = np.load(f"{OUT}/null.npy")
elig = S[S.years >= 5]
n_sig = int(((elig.bal_slope > 0) & (elig.bal_slope_p < 0.05)).sum())

RED, BLUE = "EF5350", "1976D2"
wb = Workbook()
def sheet(title, header, rows, fmts):
    ws = wb.create_sheet(title)
    ws.append(header)
    for c in ws[1]: c.font = Font(bold=True); c.fill = PatternFill("solid", fgColor="EEEEEE"); c.alignment = Alignment(horizontal="center", wrap_text=True)
    for r in rows:
        ws.append(r)
    for row in ws.iter_rows(min_row=2):
        for c, f in zip(row, fmts):
            if f is None: continue
            c.alignment = Alignment(horizontal="right")
            if isinstance(c.value, (int, float)) and not (isinstance(c.value, float) and np.isnan(c.value)):
                c.number_format = f
                if f.startswith("+"):
                    if c.value > 0: c.font = Font(color=RED)
                    elif c.value < 0: c.font = Font(color=BLUE)
            elif isinstance(c.value, float): c.value = None
    for i in range(1, len(header) + 1): ws.column_dimensions[get_column_letter(i)].width = 12
    ws.column_dimensions["B"].width = 16; ws.freeze_panes = "C2"
    return ws
wb.remove(wb.active)
pct, sp, num = "0.0%", "+0.00;-0.00;0.00", "0"
sheet("상위10 요약",
      ["티커", "종목명", "섹터", "관측 연도", "전 기간 균형적중률", "최근3년 균형적중률", "최근3년 적중률", "최근3년 상관",
       "균형적중률 기울기(%p/년)", "기울기 p값", "상관 기울기(/년)"],
      [[r.ticker, r["name"], r.sector, int(r.years), r.bal_all, r.bal_recent3, r.hit_recent3, r.ic_recent3,
        r.bal_slope * 100, r.bal_slope_p, r.ic_slope] for _, r in top.iterrows()],
      [None, None, None, num, pct, pct, pct, sp, "+0.00;-0.00;0.00", "0.000", sp])
rowsB = [[t, nm.get(t, "")] + [P.loc[t, y] for y in yrs] for t in top.ticker]
rowsB += [["(기준선)", "KOSPI200 모멘텀"] + [base.loc[y, "k200_mom_bal"] for y in yrs],
          ["(전체)", "유니버스 중앙값"] + [med.loc[y, "bal"] for y in yrs]]
sheet("연도별 균형적중률", ["티커", "종목명"] + [str(y) for y in yrs], rowsB, [None, None] + [pct] * len(yrs))
rowsC = [[t, nm.get(t, "")] + [Pi.loc[t, y] for y in yrs] for t in top.ticker]
rowsC += [["(기준선)", "KOSPI200 모멘텀"] + [base.loc[y, "k200_mom_ic"] for y in yrs],
          ["(전체)", "유니버스 중앙값"] + [med.loc[y, "ic"] for y in yrs],
          ["(참고)", "KOSPI200 10일 상승 비율"] + [base.loc[y, "k200_up_share"] for y in yrs]]
ws = sheet("연도별 상관", ["티커", "종목명"] + [str(y) for y in yrs], rowsC, [None, None] + [sp] * len(yrs))
for c in ws[ws.max_row][2:]: c.number_format = pct; c.font = Font(color="000000")
sheet("우연 수준", ["항목", "설명", "실제", "우연 중앙값", "우연 95%"],
      [["최근3년 균형적중률 1등", "262종목 중 최고값", S.bal_recent3[S.years >= 5].max(), np.median(nr), np.quantile(nr, .95)],
       ["균형적중률 기울기 1등(%p/년)", "5년 이상 종목 중 최고값", elig.bal_slope.max() * 100, np.median(ns) * 100, np.quantile(ns, .95) * 100],
       ["최근3년 ≥60% & 기울기 >1%p", "해당 종목 수", int(((S.bal_recent3 >= .6) & (S.bal_slope > .01)).sum()), np.median(nb), np.quantile(nb, .95)],
       ["기울기 >0 & p<0.05", f"5년 이상 {len(elig)}종목 중", n_sig, len(elig) * 0.025, None]],
      [None, None, "0.000", "0.000", "0.000"])
ws = wb.create_sheet("정의")
for line in ["신호 = 종목의 지난 10거래일 로그수익률 (t 종가까지)", "결과 = KOSPI200 의 다음 10거래일 로그수익률 (t 종가 → t+10 종가)",
             "적중률 = 부호가 같은 날 비율 · 균형적중률 = (지수 상승일 적중률 + 지수 하락일 적중률)/2 — 상승장 쏠림 보정",
             "상관 = Spearman 순위 상관 · 기준선 = KOSPI200 자기 자신의 지난 10일 수익률",
             "우연 수준 = KOSPI200 결과 계열을 무작위 날짜만큼 원형 이동해 200회 재계산",
             "선정 = 관측 5년 이상 & 균형적중률 기울기 > 0 중 최근 3년(2024~2026.10) 균형적중률 상위 10",
             "유니버스 = LLV universe_2026-11-16 (262종목, 현재 명단 → 과거 생존편향) · 자료 = data_2015 (2015-01-02~2026-10-29)",
             "⚠ 10일 구간이 겹쳐 연도당 독립 관측은 약 25개 — 한 해 적중률 표준오차 약 ±10%p"]:
    ws.append([line])
ws.column_dimensions["A"].width = 120
fn = f"{OUT}/KOSPI200_선행종목_20261123.xlsx"; wb.save(fn); print(fn)
top.to_csv(f"{OUT}/top10.csv", index=False, encoding="utf-8-sig")
print(top[["ticker", "name"]].to_string()); print("n_sig", n_sig, "of", len(elig))
