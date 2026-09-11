#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yz_cells_etf.py — 9셀 수익률표, 실제 ETF 기준 · 대형 2종목 제외
================================================================
케인 지시 2026-09-11:
  "1) 우리의 유니버스에서 영향이 너무 큰 2개 종목은 빼자. 삼성전자와 SK하이닉스.
   2) BM은 클로이가 계산하지 말고, 동일 가중 ETF 252000 하고 시총가중 102110 을 사용해."

종전(`yz_group_quarterly.py`)과 무엇이 다른가
  · 유니버스에서 **005930 삼성전자 · 000660 SK하이닉스** 제외
  · 시장 기준을 내가 만든 '전 종목 중앙' 이 아니라 **실제 상장 ETF 둘**로
      252000  TIGER 200동일가중   (동일가중)
      102110  TIGER 200           (시총가중)
  → 전제가 다르므로 **파일을 따로 둔다.** 같은 워크북에 섞으면 어느 시트가 어느
    기준인지 헷갈린다.

⚠ 252000 은 **2023-08-14 상장**이라 2023Q1~Q3 를 못 덮는다. 그래서 수익 분기가
  **2023Q4 이후인 전이 12쌍**만 쓴다 (종전 14쌍).
⚠ 그래서 종전 표와의 차이에는 '2종목 제외' 와 '기간 축소' 가 섞인다 —
  아래 ③ 표가 그 둘을 갈라 놓는다.

실행
  python3 scripts/yz_cells_etf.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import yz_group as G                                     # noqa: E402
import yz_group_quarterly as Q                           # noqa: E402

REPO = G.REPO
BM_EQ, BM_CAP = "252000", "102110"          # 동일가중 / 시총가중 ETF
DROP = ["005930", "000660"]                 # 케인 지시 제외 (삼성전자·SK하이닉스)
VOL3, RET3 = Q.VOL3, Q.RET3
UP, DOWN = "FFEF5350", "FF1976D2"


def load_eq() -> pd.Series:
    """252000 은 비코어라 tickers/ 에 있다 (core/extend 패널에 없음)."""
    p = G.STOLAB / "longlivevault" / "data" / "ohlcv" / "tickers" / f"{BM_EQ}.parquet"
    if not p.exists():
        raise SystemExit(
            f"❌ {BM_EQ} 없음: {p}\n"
            "   수집: cd $STOLAB/longlivevault && python3 -c \""
            "from stolab_data.data_service import get_ohlcv; "
            f"get_ohlcv('{BM_EQ}', start_date='20230101')\"")
    e = pd.read_parquet(p)
    e["Date"] = pd.to_datetime(e["Date"])
    return e.set_index("Date")["Close"].astype(float)


def build(df: pd.DataFrame, drop: list[str] | None = None) -> dict:
    drop = DROP if drop is None else drop
    piv = lambda c: df.pivot_table(index="Date", columns="Ticker",
                                   values=c, aggfunc="last").astype(float)
    O, H, L, C = (piv(c) for c in ("Open", "High", "Low", "Close"))
    excl = [BM_CAP] + list(drop)
    Cx, Ox, Hx, Lx = (t.drop(columns=excl, errors="ignore") for t in (C, O, H, L))
    o, cc = np.log(Ox / Cx.shift(1)), np.log(Cx / Ox)
    u, dn = np.log(Hx / Ox), np.log(Lx / Ox)
    rs = u * (u - cc) + dn * (dn - cc)
    EQ = load_eq()
    qs = sorted(set(C.index.to_period("Q")))

    def qret(q):
        s = C.index[C.index.to_period("Q") == q]
        return Cx.loc[s].apply(lambda c_: (c_.dropna().iloc[-1] / c_.dropna().iloc[0] - 1)
                               if c_.notna().sum() > 15 else np.nan)

    def bm(ser, q):
        """⚠ 그 분기 거래일의 95% 이상 덮을 때만 유효 — 부분 커버는 NaN."""
        s = C.index[C.index.to_period("Q") == q]
        x = ser.reindex(s).dropna()
        return float(x.iloc[-1] / x.iloc[0] - 1) if len(x) >= len(s) * 0.95 else np.nan

    rec, qinfo = [], []
    for a, b in zip(qs, qs[1:]):
        s = C.index[C.index.to_period("Q") == a]
        n = len(s)
        if n < Q.MIN_Q_DAYS:
            continue
        r_eq, r_cap = bm(EQ, b), bm(C[BM_CAP], b)
        if np.isnan(r_eq):
            continue                                     # 252000 미커버 분기 제외
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        sig = np.sqrt(o.loc[s].var(ddof=1) + k * cc.loc[s].var(ddof=1)
                      + (1 - k) * rs.loc[s].mean())
        ra, rb = qret(a), qret(b)
        ok = (sig.notna() & ra.notna() & rb.notna()
              & (Cx.loc[s].notna().sum() >= n * Q.MIN_COVER))
        gv, gr, rbb = Q._t3(sig[ok], VOL3), Q._t3(ra[ok], RET3), rb[ok]
        qinfo.append({"전분기": str(a), "수익 분기": str(b), "종목": int(ok.sum()),
                      "동일가중 252000": r_eq, "시총가중 102110": r_cap,
                      "종목 중앙": float(rbb.median()),
                      "괴리(시총−동일)": r_cap - r_eq})
        for cv in VOL3:
            for cr in RET3:
                for t_, v_ in rbb[(gv == cv) & (gr == cr)].items():
                    rec.append({"분기": str(b), "티커": t_, "변동성": cv, "전분기 수익": cr,
                                "r": v_, "vs동일": v_ - r_eq, "vs시총": v_ - r_cap})
    D, QI = pd.DataFrame(rec), pd.DataFrame(qinfo)

    rows = []
    for cv in VOL3[::-1]:
        for cr in RET3[::-1]:
            x = D[(D["변동성"] == cv) & (D["전분기 수익"] == cr)]
            rows.append({"변동성": cv, "전분기 수익": cr, "표본": len(x),
                         "중앙": float(x["r"].median()), "평균": float(x["r"].mean()),
                         "평균−중앙": float(x["r"].mean() - x["r"].median()),
                         "vs 동일가중 중앙": float(x["vs동일"].median()),
                         "vs 시총가중 중앙": float(x["vs시총"].median())})
    cells = pd.DataFrame(rows)
    cells.loc[len(cells)] = {
        "변동성": "전체", "전분기 수익": "—", "표본": len(D),
        "중앙": float(D["r"].median()), "평균": float(D["r"].mean()),
        "평균−중앙": float(D["r"].mean() - D["r"].median()),
        "vs 동일가중 중앙": float(D["vs동일"].median()),
        "vs 시총가중 중앙": float(D["vs시총"].median())}
    return {"cells": cells, "quarters": QI, "raw": D,
            "n_q": len(QI), "n_obs": len(D)}


def attribution(df: pd.DataFrame) -> pd.DataFrame:
    """종전 표와의 차이에서 '기간 축소' 와 '2종목 제외' 를 갈라 놓는다.

    ⚠ 이걸 안 하면 '삼성·하이닉스를 빼니 달라졌다' 고 오독하기 쉽다 —
      실제로는 기간이 14분기 → 12분기로 준 효과가 섞여 있다.
    """
    piv = lambda c: df.pivot_table(index="Date", columns="Ticker",
                                   values=c, aggfunc="last").astype(float)
    O, H, L, C = (piv(c) for c in ("Open", "High", "Low", "Close"))
    qs = sorted(set(C.index.to_period("Q")))
    q0 = pd.Period("2023Q4")

    def run(excl, qmin):
        Cx, Ox, Hx, Lx = (t.drop(columns=excl, errors="ignore") for t in (C, O, H, L))
        o, cc = np.log(Ox / Cx.shift(1)), np.log(Cx / Ox)
        u, dn = np.log(Hx / Ox), np.log(Lx / Ox)
        rs = u * (u - cc) + dn * (dn - cc)
        out = []
        for a, b in zip(qs, qs[1:]):
            if qmin and b < qmin:
                continue
            s = C.index[C.index.to_period("Q") == a]
            n = len(s)
            if n < Q.MIN_Q_DAYS:
                continue
            k = 0.34 / (1.34 + (n + 1) / (n - 1))
            sig = np.sqrt(o.loc[s].var(ddof=1) + k * cc.loc[s].var(ddof=1)
                          + (1 - k) * rs.loc[s].mean())
            qr = lambda q: Cx.loc[C.index[C.index.to_period("Q") == q]].apply(
                lambda c_: (c_.dropna().iloc[-1] / c_.dropna().iloc[0] - 1)
                if c_.notna().sum() > 15 else np.nan)
            ra, rb = qr(a), qr(b)
            ok = (sig.notna() & ra.notna() & rb.notna()
                  & (Cx.loc[s].notna().sum() >= n * Q.MIN_COVER))
            gv, gr = Q._t3(sig[ok], VOL3), Q._t3(ra[ok], RET3)
            for cv in VOL3:
                for cr in RET3:
                    for t_, v_ in rb[ok][(gv == cv) & (gr == cr)].items():
                        out.append({"변동성": cv, "전분기 수익": cr, "r": v_})
        return pd.DataFrame(out)

    cases = [("① 종전 (14분기 · 전 종목)", [BM_CAP], None),
             ("② 기간만 축소 (12분기 · 전 종목)", [BM_CAP], q0),
             ("③ 기간 + 2종목 제외 (12분기)", [BM_CAP] + DROP, q0)]
    res = {lab: run(e, q) for lab, e, q in cases}
    rows = []
    for lab in res:
        x = res[lab]
        rows.append({"구성": lab, "관측": len(x), "중앙": float(x["r"].median()),
                     "평균": float(x["r"].mean()),
                     "평균−중앙": float(x["r"].mean() - x["r"].median())})
    tot = pd.DataFrame(rows)

    gaps = []
    for cv in VOL3[::-1]:
        for cr in RET3[::-1]:
            g = []
            for lab in list(res)[1:]:
                x = res[lab]
                x = x[(x["변동성"] == cv) & (x["전분기 수익"] == cr)]
                g.append(float(x["r"].mean() - x["r"].median()))
            gaps.append({"셀": f"{cv}·{cr}", "② 전 종목": g[0], "③ 2종목 제외": g[1],
                         "차이(③−②)": g[1] - g[0]})
    return tot, pd.DataFrame(gaps)


def write_xlsx(out: Path, R: dict, tot: pd.DataFrame, gaps: pd.DataFrame) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    F = "Arial"
    hdr = PatternFill("solid", fgColor="FF37474F")
    thin = Side(style="thin", color="FFBDBDBD")
    med = Side(style="medium", color="FF37474F")
    FILL = {"고": "FFFFEBEE", "중": "FFF5F5F5", "저": "FFE3F2FD"}
    GC = {"고": UP, "중": "FF616161", "저": DOWN, "전체": "FF000000"}
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

    def table(ws, dfx, r0, fmt, signed=frozenset(), gcol=None, band=1):
        for j, col in enumerate(dfx.columns, 1):
            c = ws.cell(row=r0, column=j, value=col)
            c.fill, c.font = hdr, Font(name=F, bold=True, color="FFFFFFFF", size=10)
            c.alignment = Alignment(horizontal="center", wrap_text=True)
            c.border = Border(bottom=med, left=thin, right=thin)
        ws.row_dimensions[r0].height = 30
        for i, (_, rec) in enumerate(dfx.iterrows()):
            g = rec.get(gcol) if gcol and gcol in dfx.columns else None
            for j, col in enumerate(dfx.columns, 1):
                v = rec[col]
                c = ws.cell(row=r0 + 1 + i, column=j,
                            value=(None if (isinstance(v, float) and np.isnan(v)) else v))
                color = "FF000000"
                if col in signed and isinstance(v, (int, float)) and not pd.isna(v):
                    color = UP if v > 0 else DOWN if v < 0 else "FF000000"
                elif col == gcol:
                    color = GC.get(g, "FF000000")
                c.font = Font(name=F, size=10, color=color, bold=(col == gcol))
                c.number_format = fmt.get(col, "General")
                c.alignment = Alignment(horizontal="left" if j == 1 else "right")
                c.border = Border(left=thin, right=thin,
                                  bottom=(med if (i + 1) % band == 0 else thin))
                if g in FILL:
                    c.fill = PatternFill("solid", fgColor=FILL[g])
        return r0 + len(dfx) + 1

    QI = R["quarters"]
    ws = sheet("설정_분기별BM", [10, 11, 7, 16, 16, 12, 14])
    r = note(ws, 1, "9셀 수익률표 — 실제 ETF 기준 · 대형 2종목 제외", bold=True, size=14)
    r = note(ws, r, "생성기 scripts/yz_cells_etf.py   ·   케인 지시 2026-09-11",
             size=9, color="FF757575")
    r += 1
    for ln in [
        "종전(yz_group_quarterly.py)과 바뀐 것 둘",
        f"   ① 유니버스에서 **삼성전자(005930) · SK하이닉스(000660)** 제외",
        f"   ② 시장 기준을 내가 만든 '전 종목 중앙' 이 아니라 **실제 상장 ETF 둘**로 —",
        f"        {BM_EQ}  TIGER 200동일가중 (동일가중)   ·   {BM_CAP}  TIGER 200 (시총가중)",
        "",
        f"⚠ {BM_EQ} 은 **2023-08-14 상장**이라 2023Q1~Q3 를 못 덮는다. 그래서 수익 분기가 "
        f"**2023Q4 이후인 전이 {R['n_q']}쌍**만 쓴다 (종전 14쌍).",
        "   → 종전 표와의 차이에 '2종목 제외' 와 '기간 축소' 가 섞인다. 세 번째 시트가 "
        "그 둘을 갈라 놓는다.",
    ]:
        r = note(ws, r, ln, size=9, bold=ln and not ln.startswith(" "))
    r += 1
    r = note(ws, r, "분기별 두 ETF 수익률", bold=True, size=12)
    r = table(ws, QI, r + 1,
              {"종목": "#,##0", **{c: "0.0%" for c in
                                 ("동일가중 252000", "시총가중 102110", "종목 중앙",
                                  "괴리(시총−동일)")}},
              signed={"동일가중 252000", "시총가중 102110", "종목 중앙", "괴리(시총−동일)"})
    eqm, capm = QI["동일가중 252000"].mean(), QI["시총가중 102110"].mean()
    for ln in [
        f"   분기 평균 — 동일가중 **{eqm:+.1%}** · 시총가중 **{capm:+.1%}** · "
        f"종목 중앙 {QI['종목 중앙'].mean():+.1%}",
        "★ **두 ETF 의 괴리가 크다.** 같은 KOSPI200 을 담는데 가중만 다른 상품인데도 "
        f"분기 평균이 {capm - eqm:+.1%}p 벌어진다.",
        f"   극단은 2026Q2 — 동일가중 {QI.loc[QI['수익 분기'] == '2026Q2', '동일가중 252000'].iloc[0]:+.1%} "
        f"vs 시총가중 {QI.loc[QI['수익 분기'] == '2026Q2', '시총가중 102110'].iloc[0]:+.1%}. "
        "소수 대형주가 지수를 통째로 끌었다는 뜻이다.",
        "⇒ **'BM 대비' 라는 말이 어느 BM 이냐에 따라 완전히 달라진다.** 아래 표에 두 열을 "
        "나란히 둔 이유다.",
        f"   ※ 동일가중 ETF({eqm:+.1%})가 우리 유니버스 종목 중앙"
        f"({QI['종목 중앙'].mean():+.1%})보다 높다 — 252000 은 KOSPI200 **대형주 200개**의 "
        "동일가중이고, 우리 유니버스는 중소형을 포함한 204종목이라 구성이 다르다.",
    ]:
        r = note(ws, r, ln, size=9)

    ws = sheet("9셀_전기간", [7, 11, 8, 10, 10, 11, 15, 15])
    r = note(ws, 1, f"9셀 전 기간 — 전이 {R['n_q']}쌍 · 관측 {R['n_obs']:,}", bold=True, size=14)
    r = note(ws, r, "삼성전자·SK하이닉스 제외 · 다음 분기 수익률 원값",
             size=9, color="FF757575")
    r += 1
    r = table(ws, R["cells"], r,
              {"표본": "#,##0", **{c: "0.0%" for c in
                                 ("중앙", "평균", "평균−중앙", "vs 동일가중 중앙",
                                  "vs 시총가중 중앙")}},
              signed={"중앙", "평균", "vs 동일가중 중앙", "vs 시총가중 중앙"},
              gcol="변동성", band=3)
    r += 1
    for ln in [
        "읽기 — ⓐ **중앙으로 유일하게 두 ETF 를 다 이긴 칸은 고·중** "
        "(vs동일 +3.3% · vs시총 +1.6%). 나머지 8칸은 하나 이상에서 음수다.",
        "   ⚠ 다만 표본 170개이고 §11-3 부호검정이 p=0.791 이었다 — "
        "**한 칸을 골라 쓸 근거는 여전히 없다.**",
        "ⓑ **평균−중앙 격차는 여전히 고변동에서 크다** (고 8.6/14.8/6.4 vs 저 3.3/3.9/6.3). "
        "2종목을 빼도 이 구조는 남았다.",
        "ⓒ 전체 중앙 +3.5% 인데 vs동일 −1.7% · vs시총 −3.5% — "
        "**중앙 종목은 어느 ETF 로 재도 진다.** 시총가중 쪽이 더 크게 이긴다.",
    ]:
        r = note(ws, r, ln, size=9)

    ws = sheet("2종목제외_효과", [34, 10, 10, 10, 12])
    r = note(ws, 1, "'삼성전자·SK하이닉스 제외' 가 실제로 얼마나 바꿨나", bold=True, size=14)
    r = note(ws, r, "종전 표와의 차이에서 '기간 축소' 와 '2종목 제외' 를 갈라 놓는다 — "
                    "안 그러면 오독하기 쉽다", size=9, color="FF757575")
    r += 1
    r = table(ws, tot, r, {"관측": "#,##0",
                           **{c: "0.0%" for c in ("중앙", "평균", "평균−중앙")}},
              signed={"중앙", "평균"})
    r += 1
    r = note(ws, r, "평균−중앙 격차를 셀별로 (②→③ 이 순수한 '2종목 제외' 효과)",
             bold=True, size=11)
    r = table(ws, gaps, r + 1,
              {c: "0.0%" for c in ("② 전 종목", "③ 2종목 제외", "차이(③−②)")},
              signed={"차이(③−②)"}, band=3)
    r += 1
    for ln in [
        "★ **2종목을 빼도 거의 아무것도 바뀌지 않는다.** 전체 평균−중앙 6.9% → 6.8%, "
        "셀별 차이 −0.8%p ~ +0.7%p.",
        "   당연한 결과다 — 이 표의 통계는 **종목 200개의 중앙·평균**이라 2종목은 1% 비중이다. "
        "삼성·하이닉스의 영향력이 큰 곳은",
        "   **시총가중 지수**(거기선 두 종목이 큰 비중을 차지한다)이지 횡단면 순위 통계가 아니다.",
        "⇒ 케인이 걱정한 '영향이 너무 큰 2종목' 은 **BM 쪽 문제**였고, 그건 위 시트의 "
        "두 ETF 괴리로 드러난다. 유니버스에서 빼는 것으로는 해결되지 않는다.",
        "⚠ 종전 표(14분기)와 이 표(12분기)의 차이는 대부분 **기간 축소**에서 온다 "
        "(①→② 가 그 몫). 2종목 제외(②→③)는 거의 0 이다.",
    ]:
        r = note(ws, r, ln, size=9)

    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)


def main() -> int:
    print("· 패널 로드")
    df = G.load_raw()
    print(f"· 9셀 계산 (제외 {DROP} · BM {BM_EQ}/{BM_CAP})")
    R = build(df)
    print(f"  전이 {R['n_q']}쌍 · 관측 {R['n_obs']:,}")
    print("· 2종목 제외 효과 분해")
    tot, gaps = attribution(df)
    for _, x in tot.iterrows():
        print(f"  {x['구성']:<32} 관측 {x['관측']:>6,} · 중앙 {x['중앙']:>6.1%} · "
              f"평균 {x['평균']:>6.1%} · 격차 {x['평균−중앙']:>6.1%}")
    out = REPO / "docs" / "holdtiming" / "results" / "YZ9셀_ETF기준.xlsx"
    write_xlsx(out, R, tot, gaps)
    print(f"✅ Excel {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
