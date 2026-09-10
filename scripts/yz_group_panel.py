#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yz_group_panel.py — YZ 고·중·저 그룹의 연도별 패널 (2023~2026)
===============================================================
케인 지시 2026-09-10: "매년 고중저 구분한 게 재미있을 것 같아. 23~26년 각각
고중저의 중앙값/평균, 연율%, 시총 중앙/평균, 그룹간의 이동, 그리고 수익율을."

`yz_group.py` 의 분할 로직을 그대로 써서(그 해 데이터만으로 균등 3분할) 연도별로
쌓고, 이동과 수익률을 붙인다.

⚠⚠ **수익률은 두 갈래를 반드시 갈라 본다 — 섞으면 오독한다.**

  ⓐ **같은 해 수익률** — 그룹을 그 해 데이터로 나눠 놓고 그 해 수익률을 보는 것이라
     **룩어헤드**다. "많이 움직인 종목이 많이 올랐다" 는 강세장에서 거의 동어반복이다.
     서술용으로만 쓰고 **전략 근거로 쓰면 안 된다.**

  ⓑ **익년 수익률** — 전년 그룹으로 다음 해를 보는 것이라 룩어헤드가 없다.
     **실전에서 의미가 있는 건 이쪽뿐이다.**

  실제로 둘의 결론이 다르다 (§결과 참조) — 그래서 갈라 놓는 것이다.

⚠ 전이 쌍이 **3개**(23→24, 24→25, 25→26)뿐이다. 어떤 검정도 할 수 없다 —
  이 표는 **관측 기록이지 근거가 아니다.**

실행
  python3 scripts/yz_group_panel.py [--years 2023 2024 2025 2026]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import yz_group as G                                     # noqa: E402

REPO = G.REPO
LABELS = G.LABELS
UP, DOWN = "FFEF5350", "FF1976D2"


def yearly(df: pd.DataFrame, years: list[int]) -> tuple[dict, pd.DataFrame, dict]:
    """연도별 그룹 + 특성표 + 그 해 수익률."""
    C, MC = G._piv(df, "Close"), G._piv(df, "MarketCap")
    grp, rows, ret = {}, [], {}
    for y in years:
        s, e = f"{y}-01-01", f"{y}-12-31"
        sig, days = G.yz_for(df, s, e)
        nd = len(C.loc[s:e])
        a = G.assign(sig, days, nd)
        grp[y] = a["그룹"]
        mc = MC.loc[s:e].median()

        # 그 해 수익률 — 첫 유효 종가 → 마지막 유효 종가 (기간수익률, 연율화 안 함)
        cy = C.loc[s:e]
        r = cy.apply(lambda col: (col.dropna().iloc[-1] / col.dropna().iloc[0] - 1)
                     if col.notna().sum() > 20 else np.nan)
        bmc = C[G.BM].loc[s:e].dropna()
        bm = bmc.iloc[-1] / bmc.iloc[0] - 1
        ret[y] = {"r": r, "ex": r - bm, "bm": float(bm), "n_days": nd,
                  "start": str(cy.index.min().date()), "end": str(cy.index.max().date())}

        for k in LABELS[::-1]:                            # 고 → 중 → 저 순서로 보기
            t = a[a["그룹"] == k]
            m = mc.reindex(t.index)
            rows.append({
                "연도": y, "거래일": nd, "그룹": k, "종목수": len(t),
                "YZ 중앙": t["YZ"].median(), "YZ 평균": t["YZ"].mean(),
                "연율% 중앙": t["연율%"].median(), "연율% 평균": t["연율%"].mean(),
                "시총(조) 중앙": m.median() / 1e12, "시총(조) 평균": m.mean() / 1e12,
            })
    return grp, pd.DataFrame(rows), ret


def transitions(grp: dict, years: list[int]) -> list[tuple[str, pd.DataFrame, float, int]]:
    out = []
    for a, b in zip(years, years[1:]):
        m = grp[a].notna() & grp[b].notna()
        ct = (pd.crosstab(grp[a][m], grp[b][m], normalize="index") * 100)
        ct = ct.reindex(index=LABELS[::-1], columns=LABELS[::-1]).fillna(0)
        ct.index = [f"{a}년 {i}" for i in ct.index]
        ct.columns = [f"→{b}년 {c}" for c in ct.columns]
        out.append((f"{a} → {b}", ct.reset_index().rename(columns={"index": "이전"}),
                    float((grp[a][m] == grp[b][m]).mean()), int(m.sum())))
    return out


def returns_table(grp: dict, ret: dict, years: list[int], lag: int) -> pd.DataFrame:
    """lag=0 → 같은 해(룩어헤드) / lag=1 → 익년(실전)."""
    rows = []
    for a in years:
        b = a + lag
        if b not in ret:
            continue
        R = ret[b]
        for k in LABELS[::-1]:
            t = grp[a][grp[a] == k].index
            rr, ee = R["r"].reindex(t).dropna(), R["ex"].reindex(t).dropna()
            if len(rr) < 5:
                continue
            rows.append({
                "그룹 기준": f"{a}년" if lag == 0 else f"{a} → {b}",
                "수익률 연도": b, f"BM({G.BM})": R["bm"], "그룹": k, "종목수": len(rr),
                "수익률 중앙": rr.median(), "수익률 평균": rr.mean(),
                "BM초과 중앙": ee.median(), "BM초과 평균": ee.mean(),
                "BM 이긴 비율": float((ee > 0).mean()),
            })
    return pd.DataFrame(rows)


def write_xlsx(out: Path, spec: pd.DataFrame, trans: list, same: pd.DataFrame,
               nxt: pd.DataFrame, ret: dict, years: list[int]) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    F = "Arial"
    hdr = PatternFill("solid", fgColor="FF37474F")
    thin = Side(style="thin", color="FFBDBDBD")
    med = Side(style="medium", color="FF37474F")
    FILL = {"고": "FFFFEBEE", "중": "FFF5F5F5", "저": "FFE3F2FD"}
    GCOL = {"고": UP, "중": "FF616161", "저": DOWN}
    wb = Workbook()
    first = [True]

    def sheet(name, widths):
        ws = wb.active if first[0] else wb.create_sheet(name)
        first[0] = False
        ws.title = name
        ws.sheet_view.showGridLines = False
        for j, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(j)].width = w
        return ws

    def note(ws, row, text, bold=False, size=10, color="FF000000"):
        c = ws.cell(row=row, column=1, value=text)
        c.font = Font(name=F, size=size, bold=bold, color=color)
        return row + 1

    def table(ws, dfx, r0, fmt, signed=frozenset(), gcol="그룹", band=3):
        for j, col in enumerate(dfx.columns, 1):
            c = ws.cell(row=r0, column=j, value=col)
            c.fill, c.font = hdr, Font(name=F, bold=True, color="FFFFFFFF", size=10)
            c.alignment = Alignment(horizontal="center", wrap_text=True)
            c.border = Border(bottom=med, left=thin, right=thin)
        ws.row_dimensions[r0].height = 30
        for i, (_, rec) in enumerate(dfx.iterrows()):
            g = rec.get(gcol) if gcol in dfx.columns else None
            for j, col in enumerate(dfx.columns, 1):
                v = rec[col]
                c = ws.cell(row=r0 + 1 + i, column=j,
                            value=(None if (isinstance(v, float) and np.isnan(v)) else v))
                col_color = "FF000000"
                if col in signed and isinstance(v, (int, float)) and not pd.isna(v):
                    col_color = UP if v > 0 else DOWN if v < 0 else "FF000000"
                elif col == gcol:
                    col_color = GCOL.get(g, "FF000000")
                c.font = Font(name=F, size=10, bold=(col == gcol), color=col_color)
                c.number_format = fmt.get(col, "General")
                c.alignment = Alignment(horizontal="left" if j == 1 else "right")
                c.border = Border(left=thin, right=thin,
                                  bottom=(med if (i + 1) % band == 0 else thin))
                if g in FILL:
                    c.fill = PatternFill("solid", fgColor=FILL[g])
        return r0 + len(dfx) + 1

    # ── 1. 연도별 특성 ──
    ws = sheet("연도별_특성", [7, 7, 6, 7, 10, 10, 11, 11, 12, 12])
    r = note(ws, 1, "YZ 고·중·저 그룹 — 연도별 특성", bold=True, size=14)
    r = note(ws, r, "각 연도를 **그 해 데이터만으로** 균등 3분할 · 생성기 "
                    "scripts/yz_group_panel.py", size=9, color="FF757575")
    r += 1
    r = table(ws, spec, r,
              {"연도": "0", "거래일": "#,##0", "종목수": "#,##0",
               "YZ 중앙": "0.0000", "YZ 평균": "0.0000",
               "연율% 중앙": "0.0", "연율% 평균": "0.0",
               "시총(조) 중앙": "#,##0.00", "시총(조) 평균": "#,##0.00"})
    r += 1
    for ln in [
        "읽기 — ⓐ **세 그룹이 다 같이 올라간다.** 고 그룹 연율% 중앙이 58.4 → 65.8 → "
        "66.2 → 107.7 로 오르는데 저 그룹도 25.8 → 33.5 → 34.4 → 56.2 로 같이 오른다.",
        "     시장 변동성이 통째로 움직이기 때문이다 — **2026년의 '저'(연율 56.2%)가 "
        "2023년의 '고'(58.4%)와 거의 같다.** 연도 간 비교는 여기서 무너진다.",
        "ⓑ **시총은 일관되게 고<중<저** 다 (중앙 기준 전 연도). 작은 회사가 더 흔들린다 — "
        "교과서적 사실의 재확인이지 발견이 아니다.",
        "ⓒ 시총 **평균**은 중앙과 순서가 다르다(2026 중 35.7조 > 고 24.0조). "
        "초대형주 몇 개가 평균을 끌어서다 — **평균이 아니라 중앙을 볼 것.**",
        "⚠ 2026년은 169거래일(8.3개월)이라 다른 해와 **거래일 수가 다르다.** "
        "연율%는 √252 로 연율화돼 비교 가능하지만, 뒤의 **수익률은 기간수익률**이라 아니다.",
    ]:
        r = note(ws, r, ln, size=9)

    # ── 2. 그룹 이동 ──
    ws = sheet("그룹_이동", [14, 12, 12, 12])
    r = note(ws, 1, "그룹 간 이동 — 전년 그룹이 다음 해 어디로 갔나 (%)", bold=True, size=14)
    r = note(ws, r, "각 연도를 그 해 데이터만으로 다시 3분할해 비교한다. 무작위면 "
                    "유지 33.3%.", size=9, color="FF757575")
    r += 1
    for lab, ct, same_r, n in trans:
        r = note(ws, r, f"{lab}   (n={n}종목, 그룹 유지 {same_r:.1%})", bold=True, size=11)
        r = table(ws, ct, r + 1, {c: "0.0" for c in ct.columns if c != "이전"},
                  gcol=None, band=3)
        r += 1
    for ln in [
        "읽기 — ⓐ 유지율이 72.0% → 60.3% → 58.8% 로 **떨어진다.** 다만 세 쌍뿐이라 "
        "추세인지 잡음인지 말할 수 없다.",
        "ⓑ ★ **고↔저 정반대로 건너뛰는 경우는 0~4.6% 로 거의 없다.** 이동은 대부분 "
        "인접 그룹(고↔중, 중↔저)이다.",
        "     즉 그룹이 바뀌어도 **한 칸씩 움직이지 순위가 뒤집히지는 않는다** — "
        "'느리게 움직이는 서열' 이라는 §10-5 결론과 정합적이다.",
        "ⓒ 가운데(중) 그룹이 가장 불안정하다 (유지 41~64%). 경계에 걸쳐 있으니 당연하다.",
    ]:
        r = note(ws, r, ln, size=9)

    # ── 3. 수익률 ──
    ws = sheet("수익률", [13, 9, 9, 6, 7, 11, 11, 12, 12, 12])
    r = note(ws, 1, "그룹별 수익률 — 두 갈래를 갈라서 본다", bold=True, size=14)
    r += 1
    for ln in [
        "⚠⚠ **섞어 읽으면 정반대 결론이 난다.**",
        "   ⓐ **같은 해** — 그룹을 그 해 데이터로 나눠 놓고 그 해 수익률을 본다 → "
        "**룩어헤드**. 강세장에서 '많이 움직인 종목이 많이 올랐다' 는 거의 동어반복이다.",
        "      서술용으로만 쓰고 **전략 근거로 쓰지 말 것.**",
        "   ⓑ **익년** — 전년 그룹으로 다음 해를 본다 → 룩어헤드 없음. "
        "**실전에서 의미가 있는 건 이쪽뿐이다.**",
    ]:
        r = note(ws, r, ln, size=9)
    r += 1
    fmt = {"수익률 연도": "0", "종목수": "#,##0",
           **{c: "0.0%" for c in ("BM(102110)", "수익률 중앙", "수익률 평균",
                                  "BM초과 중앙", "BM초과 평균", "BM 이긴 비율")}}
    sg = {"수익률 중앙", "수익률 평균", "BM초과 중앙", "BM초과 평균"}
    r = note(ws, r, "ⓐ 같은 해 (룩어헤드 있음 — 서술용)", bold=True, size=12)
    r = table(ws, same, r + 1, fmt, signed=sg)
    r += 1
    r = note(ws, r, "ⓑ 익년 (룩어헤드 없음 — 실전)", bold=True, size=12)
    r = table(ws, nxt, r + 1, fmt, signed=sg)
    r += 1
    for ln in [
        "★ 읽기 — **두 표의 결론이 다르다. 이게 이 시트의 요점이다.**",
        "   ⓐ 같은 해: 고 그룹이 **전 연도에서 압도적**이다 (2023 중앙 +83.6% vs 저 +8.7%, "
        "2025 +140.7% vs +31.5%). 하지만 이건 동어반복에 가깝다.",
        "   ⓑ 익년: **일관된 방향이 없다.**",
        "        2023→2024  저가 낫다 (BM초과 중앙 저 +13.5% vs 고 +3.2%, 승률 67.2% vs 53.8%)",
        "        2024→2025  고가 낫다 (수익률 중앙 고 +76.9% vs 저 +41.0%)",
        "        2025→2026  뒤죽박죽 (고 +22.0% · 저 +14.0% · 중 +10.9%)",
        "     ⇒ **작년 변동성 그룹으로 올해 수익률을 예측할 수 있다는 근거가 없다.** "
        "방향이 해마다 뒤집힌다.",
        "⚠ BM 초과가 대부분 음수인 것은 그룹의 문제가 아니다 — BM(102110)이 시총가중이라 "
        "반도체 대형주에 끌렸고, **중앙 종목은 원래 BM 에 진다**(cell12 §기준선과 같은 현상).",
        "⚠⚠ **전이 쌍이 3개뿐이다.** 어떤 통계 검정도 할 수 없다 — 이 표는 "
        "**관측 기록이지 근거가 아니다.**",
        "⚠ **생존편향**: 지금 살아 있는 종목만 있다. 크게 흔들리다 사라진 종목이 없으므로 "
        "**고 그룹이 특히 낙관적으로 치우친다.**",
        "⚠ 2026 수익률은 **8.3개월치**(169거래일)다 — 다른 해(242~245일)와 직접 비교 금지.",
    ]:
        r = note(ws, r, ln, size=9)

    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025, 2026])
    a = ap.parse_args()

    print("· 패널 로드")
    df = G.load_raw()
    print(f"· 연도별 그룹 {a.years}")
    grp, spec, ret = yearly(df, a.years)
    for y in a.years:
        print(f"  {y}: {ret[y]['start']}~{ret[y]['end']} ({ret[y]['n_days']}일) · "
              f"BM {ret[y]['bm']:+.1%} · " +
              " ".join(f"{k}{(grp[y]==k).sum()}" for k in LABELS[::-1]))
    trans = transitions(grp, a.years)
    for lab, _, s, n in trans:
        print(f"  이동 {lab}: 유지 {s:.1%} (n={n})")
    same = returns_table(grp, ret, a.years, 0)
    nxt = returns_table(grp, ret, a.years, 1)

    out = REPO / "docs" / "holdtiming" / "results" / "YZ그룹_연도별패널.xlsx"
    write_xlsx(out, spec, trans, same, nxt, ret, a.years)
    print(f"✅ Excel {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
