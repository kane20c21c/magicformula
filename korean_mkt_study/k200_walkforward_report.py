"""k200_lead_walkforward.py 결과 → 엑셀 (out/k200_walkforward/KOSPI200_선행종목_전진검증.xlsx)"""
import os, numpy as np, pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter
from scipy import stats
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "k200_walkforward")
Q = pd.read_csv(f"{OUT}/quarterly_results.csv"); P = pd.read_csv(f"{OUT}/monthly_picks.csv")
R = ["선행②", "선행③", "자기추세±1", "자기추세±2", "매매없음"]
wb = Workbook(); wb.remove(wb.active); RED, BLUE = "EF5350", "1976D2"
def sheet(t, H, rows, F):
    ws = wb.create_sheet(t); ws.append(H)
    for c in ws[1]: c.font = Font(bold=True); c.fill = PatternFill("solid", fgColor="EEEEEE"); c.alignment = Alignment(horizontal="center", wrap_text=True)
    for r in rows: ws.append([None if isinstance(v, float) and np.isnan(v) else v for v in r])
    for row in ws.iter_rows(min_row=2):
        for c, f in zip(row, F):
            if f and isinstance(c.value, (int, float)):
                c.alignment = Alignment(horizontal="right"); c.number_format = f
                if f.startswith("+"): c.font = Font(color=RED if c.value > 0 else BLUE if c.value < 0 else "000000")
    for i in range(1, len(H) + 1): ws.column_dimensions[get_column_letter(i)].width = 13
    ws.freeze_panes = "B2"; return ws
pp = "+0.0%;-0.0%;0.0%"
sheet("분기별 수익률", ["분기", "KOSPI200"] + R + ["②−자기±1", "선행② 균형적중"],
      [[r.분기, r.KOSPI200] + [r[x] for x in R] + [r["선행②"] - r["자기추세±1"], r["선행②_균형적중"]] for _, r in Q.iterrows()],
      [None, pp] + [pp] * 6 + ["0.0%"])
full = Q[Q.분기 != "2026Q4"]; st = []
for a, b in [("선행②", "자기추세±1"), ("선행③", "자기추세±2"), ("선행②", "매매없음"), ("선행②", "선행③")]:
    d = full[a] - full[b]; w = int((d > 0).sum())
    st.append([f"{a} − {b}", d.mean(), d.median(), f"{w}/{len(d)}", stats.ttest_1samp(d, 0).pvalue, stats.binomtest(w, len(d)).pvalue])
cum = (1 + Q.set_index("분기")[R]).cumprod().iloc[-1] - 1
st += [[], ["누적 (분기 복리, 2020Q1~2026-11-20)"] + [None] * 5] + [[x, cum[x], None, None, None, None] for x in R]
sheet("비교 통계", ["비교", "평균 차", "중앙 차", "이긴 분기", "t검정 p", "부호검정 p"], st, [None, pp, pp, None, "0.000", "0.000"]).column_dimensions["A"].width = 34
re = Q.dropna(subset=["선행②_실제ETF"])
sheet("합성 vs 실제 ETF", ["분기"] + sum([[x, x + " 실제ETF"] for x in R], []),
      [[r.분기] + sum([[r[x], r[x + "_실제ETF"]] for x in R], []) for _, r in re.iterrows()], [None] + [pp] * 10)
ws = sheet("월별 선행종목", list(P.columns), P.values.tolist(), [None, None, None, None, "0.0%", "0.0%", "0.0%", "0", "0"]); ws.column_dimensions["B"].width = ws.column_dimensions["C"].width = ws.column_dimensions["D"].width = 22
ws = wb.create_sheet("정의")
for l in (__import__("k200_lead_walkforward").__doc__ or "").splitlines() if False else open(os.path.join(os.path.dirname(OUT), "..", "k200_lead_walkforward.py")).read().split('"""')[1].splitlines():
    ws.append([l])
ws.column_dimensions["A"].width = 130
wb.save(f"{OUT}/KOSPI200_선행종목_전진검증.xlsx"); print("saved")
