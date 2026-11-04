"""표 묶음 → 엑셀 (케인 표시 규칙: 숫자 우측 정렬, % 소수 1자리, 양수 빨강 #ef5350 / 음수 파랑 #1976D2)."""
import numpy as np, pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
RED, BLUE, FONT = "ef5350", "1976D2", "Arial"
NUMFMT = {"int": "#,##0", "num2": "0.00", "num3": "0.000", "num4": "0.0000", "pct": "0.0%", "spct": "0.0%", "snum": "0.000"}
SIGNED = ("spct", "snum")

def write_tables(tables: dict, path, text_cols=("내용", "값", "설명")):
    """tables: {sheet: [(title, df, {col: fmt}), ...]}"""
    wb = Workbook(); wb.remove(wb.active)
    thin = Side(style="thin", color="BBBBBB"); hdr_fill = PatternFill("solid", fgColor="F2F2F2")
    for sheet, items in tables.items():
        ws = wb.create_sheet(sheet[:31]); r = 1; widths = {}
        for title, df, fmt in items:
            ws.cell(r, 1, title).font = Font(name=FONT, bold=True, size=11); r += 1
            for j, c in enumerate(df.columns, 1):
                cell = ws.cell(r, j, str(c)); cell.font = Font(name=FONT, bold=True); cell.fill = hdr_fill; cell.border = Border(bottom=thin)
                cell.alignment = Alignment(horizontal="center"); widths[j] = max(widths.get(j, 0), len(str(c)) * 1.8)
            r += 1
            for row in df.itertuples(index=False):
                for j, (c, v) in enumerate(zip(df.columns, row), 1):
                    kind = fmt.get(c, "int" if c in ("건수",) else None)
                    if isinstance(v, np.generic): v = v.item()
                    if isinstance(v, float) and np.isnan(v): v = None
                    if v is not None and not isinstance(v, (int, float, bool)): v = str(v)
                    cell = ws.cell(r, j, v); cell.font = Font(name=FONT)
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        cell.alignment = Alignment(horizontal="right")
                        if kind in NUMFMT: cell.number_format = NUMFMT[kind]
                        elif isinstance(v, float): cell.number_format = "0.000"
                        if kind in SIGNED:
                            if v > 0: cell.font = Font(name=FONT, color=RED)
                            elif v < 0: cell.font = Font(name=FONT, color=BLUE)
                        w = len(f"{v:,.3f}") if isinstance(v, float) else len(f"{v:,}")
                    else:
                        left = c in text_cols; cell.alignment = Alignment(horizontal="left" if left else "center", wrap_text=left)
                        w = min(len(str(v)) * (1.6 if left else 1.1), 100) if v is not None else 0
                    widths[j] = max(widths.get(j, 0), w)
                r += 1
            r += 1
        for j, w in widths.items(): ws.column_dimensions[get_column_letter(j)].width = max(8, w + 2)
    wb.save(path); return path
