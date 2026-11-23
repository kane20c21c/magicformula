"""k200_quarterly_select.py 결과 → 후보 엑셀 (out/k200_quarterly[_YYYY-MM]/KOSPI200_분기후보_{TAG}.xlsx). 다음 분기 결과는 보지 않는다."""
import os, sys, numpy as np, pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter
ROOT = os.path.dirname(os.path.abspath(__file__)); END = sys.argv[1] if len(sys.argv) > 1 else "2026-03-31"
OUT = os.path.join(ROOT, "out", "k200_quarterly" if END == "2026-03-31" else f"k200_quarterly_{END[:7]}")
TAG = "2026Q1" if END == "2026-03-31" else pd.Period(END, "Q").strftime("%YQ%q")
D = pd.read_csv(f"{OUT}/summary.csv", dtype={"ticker": str})
QH = pd.read_csv(f"{OUT}/quarterly_hit.csv", index_col=0); QH.index = QH.index.astype(str).str.zfill(6); QH = QH.drop(columns="ticker")
QB = pd.read_csv(f"{OUT}/quarterly_bal.csv", index_col=0); QB.index = QB.index.astype(str).str.zfill(6)
up = pd.read_csv(f"{OUT}/k200_up_share.csv", index_col=0).iloc[:, 0]
E = D[(D.quarters >= 20) & (D.slope_pp_per_q > 0)]
A = E.sort_values("bal_recent4", ascending=False).head(15)
B = E.sort_values("hit_recent4", ascending=False).head(15)
RED, BLUE = "EF5350", "1976D2"; wb = Workbook(); wb.remove(wb.active)
def sheet(title, header, rows, fmts):
    ws = wb.create_sheet(title); ws.append(header)
    for c in ws[1]: c.font = Font(bold=True); c.fill = PatternFill("solid", fgColor="EEEEEE"); c.alignment = Alignment(horizontal="center", wrap_text=True)
    for r in rows: ws.append([None if isinstance(v, float) and np.isnan(v) else v for v in r])
    for row in ws.iter_rows(min_row=2):
        for c, f in zip(row, fmts):
            if f and isinstance(c.value, (int, float)):
                c.alignment = Alignment(horizontal="right"); c.number_format = f
                if f.startswith("+"): c.font = Font(color=RED if c.value > 0 else BLUE if c.value < 0 else "000000")
    for i in range(1, len(header) + 1): ws.column_dimensions[get_column_letter(i)].width = 10
    ws.column_dimensions["B"].width = 18; ws.freeze_panes = "C2"
H = ["티커", "종목명", "관측 분기", "전 기간 적중률", "이전(~2025Q1) 적중률", "최근4분기 적중률", "최근4분기 균형적중률", "기울기(%p/분기)", "기울기 p값"]
F = [None, None, "0", "0.0%", "0.0%", "0.0%", "0.0%", "+0.00;-0.00;0.00", "0.000"]
rf = lambda r: [r.ticker, r["name"], int(r.quarters), r.hit_all, r.hit_prev, r.hit_recent4, r.bal_recent4, r.slope_pp_per_q, r.slope_p]
sheet("후보A 균형적중률순", H, [rf(r) for _, r in A.iterrows()], F)
sheet("후보B 적중률순", H, [rf(r) for _, r in B.iterrows()], F)
qc = list(QH.columns); allt = list(dict.fromkeys(list(A.ticker) + list(B.ticker)))
nm = D.set_index("ticker")["name"]
sheet("분기별 적중률", ["티커", "종목명"] + qc, [[t, nm[t]] + list(QH.loc[t, qc]) for t in allt] + [["(참고)", "KOSPI200 상승일 비율"] + list(up.values)], [None, None] + ["0%"] * len(qc))
sheet("분기별 균형적중률", ["티커", "종목명"] + qc, [[t, nm[t]] + list(QB.loc[t, qc]) for t in allt], [None, None] + ["0%"] * len(qc))
N = np.load(f"{OUT}/null.npy")
sheet("기준·우연", ["항목", "값", "", "", ""], [
    ["최근4분기 '항상 상승' 적중률", up.tail(4).mean()], 
    ["우연: 최근4분기 적중률 1등 중앙값", np.median(N[:, 0])], ["우연: 1등 95%", np.quantile(N[:, 0], .95)],
    ["우연: 10등 중앙값", np.median(N[:, 1])], ["우연: 10등 95%", np.quantile(N[:, 1], .95)]], [None, "0.0%"])
ws = wb.create_sheet("정의")
for l in ["신호 = 종목 지난 10거래일 수익률 방향 · 결과 = KOSPI200 다음 거래일 방향 · 분기 = 신호일 기준",
          f"구간 2020Q1~{TAG} (다음 날이 {END} 이후인 신호일 제외 — 다음 분기 눈가림) · 유니버스 262 (현재 명단, 생존편향) · 자료 data_2015",
          "적중률 = 부호 일치 비율(0 제외) · 균형적중률 = (지수 상승일 적중률 + 하락일 적중률)/2 — 상승장 쏠림 보정",
          "기울기 = 분기 번호에 대한 적중률 OLS · 후보 = 20분기 이상 관측 & 기울기 > 0 · 최근4분기 = 선정 마지막 4분기",
          "분기 하나 ≈ 60일 → 분기 적중률 표준오차 약 ±6.5%p"]:
    ws.append([l])
ws.column_dimensions["A"].width = 120
wb.save(f"{OUT}/KOSPI200_분기후보_{TAG}.xlsx"); A.to_csv(f"{OUT}/candidates_A.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); print(A.round(3).to_string())
