"""k200_nextday_study.py 결과 → 엑셀 (out/k200_nextday/KOSPI200_다음날_상관_20261123.xlsx)"""
import os, numpy as np, pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "out", "k200_nextday")
L = pd.read_csv(f"{OUT}/yearly_long.csv", dtype={"ticker": str}); S = pd.read_csv(f"{OUT}/summary.csv", dtype={"ticker": str})
nm = pd.read_parquet(f"{ROOT}/data_2015/meta.parquet", columns=["name"]).reset_index().sort_values("date").groupby("ticker")["name"].last()
S["name"] = S.ticker.map(nm); E = S[S.years >= 5]
top = E[E.ic_slope > 0].sort_values("ic_recent3", ascending=False).head(10)
neg = E.sort_values("ic_recent3").head(10)
yrs = sorted(L.year.unique()); P = L.pivot(index="ticker", columns="year", values="ic")
base = L.groupby("year")[["k200_mom_ic", "k200_mom_bal", "univ_med_ic"]].first()
N = np.load(f"{OUT}/null.npy")
RED, BLUE = "EF5350", "1976D2"; wb = Workbook(); wb.remove(wb.active)
def sheet(title, header, rows, fmts, widthB=22):
    ws = wb.create_sheet(title); ws.append(header)
    for c in ws[1]: c.font = Font(bold=True); c.fill = PatternFill("solid", fgColor="EEEEEE"); c.alignment = Alignment(horizontal="center", wrap_text=True)
    for r in rows: ws.append([None if isinstance(v, float) and np.isnan(v) else v for v in r])
    for row in ws.iter_rows(min_row=2):
        for c, f in zip(row, fmts):
            if f is None or not isinstance(c.value, (int, float)): continue
            c.alignment = Alignment(horizontal="right"); c.number_format = f
            if f.startswith("+"): c.font = Font(color=RED if c.value > 0 else BLUE if c.value < 0 else "000000")
    for i in range(1, len(header) + 1): ws.column_dimensions[get_column_letter(i)].width = 11
    ws.column_dimensions["B"].width = widthB; ws.freeze_panes = "C2"; return ws
sp, pct = "+0.000;-0.000;0.000", "0.0%"
H = ["티커", "종목명", "관측 연도", "전 기간 상관", "최근3년 상관", "최근3년 적중률", "최근3년 균형적중률", "상관 양수 연도 수",
     "상관 기울기(/년)", "기울기 p값", "시장 몫 제거 기울기 p값", "2026 제외 기울기 p값"]
F = [None, None, "0", sp, sp, pct, pct, "0", sp, "0.000", "0.000", "0.000"]
rowf = lambda r: [r.ticker, r["name"], int(r.years), r.ic_all, r.ic_recent3, r.hit_recent3, r.bal_recent3, int(r.pos_years),
                  r.ic_slope, r.ic_slope_p, r.ic_ex_slope_p, r.ic_slope_no26_p]
sheet("상위10 (양의 상관·상승 추세)", H, [rowf(r) for _, r in top.iterrows()], F)
yr_rows = [[t, nm.get(t, "")] + [P.loc[t, y] for y in yrs] for t in top.ticker]
yr_rows += [["(기준선)", "KOSPI200 자기 10일"] + [base.loc[y, "k200_mom_ic"] for y in yrs],
            ["(전체)", "유니버스 중앙값"] + [base.loc[y, "univ_med_ic"] for y in yrs]]
sheet("연도별 상관 (상위10)", ["티커", "종목명"] + [str(y) for y in yrs], yr_rows, [None, None] + [sp] * len(yrs))
sheet("참고 음의 상관 10", H, [rowf(r) for _, r in neg.iterrows()], F)
sheet("우연 수준", ["항목", "실제", "우연 중앙값", "우연 95%"],
      [["최근3년 상관 1등 (5년 이상 종목)", E.ic_recent3.max(), np.median(N[:, 0]), np.quantile(N[:, 0], .95)],
       ["상관 기울기 1등 (/년)", E.ic_slope.max(), np.median(N[:, 1]), np.quantile(N[:, 1], .95)],
       ["최근3년 상관 평균 (전 종목)", E.ic_recent3.mean(), np.median(N[:, 3]), None],
       ["기울기>0 & p<0.05 종목 수", int(((E.ic_slope > 0) & (E.ic_slope_p < .05)).sum()), len(E) * 0.025, None]],
      [None, sp, sp, sp], widthB=12).column_dimensions["A"].width = 34
ws = wb.create_sheet("정의")
for line in ["신호 = 종목의 지난 10거래일 로그수익률 (t 종가까지) · 결과 = KOSPI200 다음 거래일 로그수익률 (t → t+1)",
             "상관 = 연도별 Spearman 순위 상관 · 적중률 = 부호 일치 비율 · 균형적중률 = (지수 상승일 적중률 + 하락일 적중률)/2",
             "기준선 = KOSPI200 자기 자신의 지난 10일 수익률 · 우연 수준 = 결과 계열을 무작위 날짜만큼 원형 이동해 200회",
             "선정 = 관측 5년 이상 & 상관 기울기 > 0 중 최근 3년(2024~2026.10) 상관 상위 10",
             "유니버스 = LLV universe_2026-11-16 (262종목, 현재 명단 → 생존편향) · 자료 = data_2015 (2015-01-02~2026-10-29), 새 수집 없음",
             "한 해 상관 표준오차 ≈ 1/√250 ≈ 0.063"]:
    ws.append([line])
ws.column_dimensions["A"].width = 120
fn = f"{OUT}/KOSPI200_다음날_상관_20261123.xlsx"; wb.save(fn); print(fn)
pd.set_option("display.width", 250)
print(neg[["ticker", "name", "ic_all", "ic_recent3", "bal_recent3", "ic_slope", "ic_slope_p"]].round(3).to_string())
