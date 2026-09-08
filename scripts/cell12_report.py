#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cell12_report.py — R²(60) × YZ_60 × RSI 12셀 리포트 (엑셀)
==========================================================
목적
  4셀 분류 축 개선(A안) 연구의 산출물. 롤링 R²(60거래일)와 Yang-Zhang 변동성
  YZ_60 을 **그날 전 종목 중앙값**으로 각각 2분할하고, RSI 를 과매도(<30) /
  중간 / 과매수(>70) 3분할해 만든 12개 셀의 구성과 선행 수익률을 정리한다.

종속변수
  BM(102110 TIGER 200) 대비 **초과** 선행수익률.
      excess(n) = (C[t+n]/C[t] - 1) - (BM[t+n]/BM[t] - 1)
  창 n = 5, 10, 20, 40, 60, 90, 120 거래일.

⚠ 읽을 때의 전제 (셀에도 적어 둔다)
  1. **생존편향** — 현재 살아 있는 207종목의 과거다. 상장폐지·급락 후 사라진
     종목이 없으므로 전 셀의 수익률이 낙관적으로 치우친다.
  2. **평균과 중앙값이 어긋난다** — 12셀 중 11셀의 중앙값이 음수인데 평균은
     양수인 셀이 여럿이다. 소수 대박이 평균을 끌어올린 것이므로 **중앙값을
     먼저 볼 것.** 종전 t검정은 전부 평균 기반이었다.
  3. **표본 겹침** — 20일 수익률을 매일 계산하므로 이웃한 관측이 독립이 아니다.
     건수가 커 보여도 독립 표본은 훨씬 적다.
  4. **분할 기준이 상대값**(그날 전 종목 중앙값)이라 셀 크기는 항상 대략
     4분의 1씩이다. "R²가 절대적으로 높다"는 뜻이 아니다.

실행
  PYTHONPATH=<pylibs> python3 scripts/cell12_report.py [--out <경로>]
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
REPO = HERE.parent.parent
STOLAB = REPO.parent
DEFAULT_OUT = REPO / "docs" / "holdtiming" / "results" / "cell12_R2xYZxRSI.xlsx"

WIN_R2 = 60                       # 롤링 R² 창 (거래일)
BM = "102110"                     # TIGER 200
FWD = [5, 10, 20, 40, 60, 90, 120]
RSI_LO, RSI_HI = 30.0, 70.0

QUAD = ["R²高·YZ高", "R²高·YZ低", "R²低·YZ高", "R²低·YZ低"]
RSI3 = ["과매도", "중간", "과매수"]
CELLS = [f"{q}·{r}" for q in QUAD for r in RSI3]
ERAS = {"23-24": ("2023-01-01", "2024-12-31"), "25-26": ("2025-01-01", "2026-12-31")}

UP, DOWN = "FFEF5350", "FF1976D2"   # Kane 전역 색 규약 (상승 빨강 / 하락 파랑)


# ─────────────────────────── 데이터 ───────────────────────────
def load_panel() -> dict[str, pd.DataFrame]:
    """LLV OHLCV 패널 → {티커: DataFrame(Date index)}."""
    sys.path.insert(0, str(STOLAB / "longlivevault"))
    vault = STOLAB / "longlivevault" / "data" / "ohlcv"
    parts = []
    for name in ("core.parquet", "extend.parquet"):
        p = vault / name
        if p.exists():
            parts.append(pd.read_parquet(p))
    if not parts:
        raise SystemExit(f"❌ OHLCV parquet 없음: {vault}")
    df = pd.concat(parts, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.zfill(6)
    return {t: g.sort_values("Date").set_index("Date") for t, g in df.groupby("Ticker")}


def wide(panel: dict[str, pd.DataFrame], col: str,
         idx: pd.DatetimeIndex, cols: list[str]) -> pd.DataFrame:
    """종목별 시리즈를 날짜×종목 행렬로. reindex 로 열 정렬을 강제한다 —
    전부 NaN 인 종목이 빠지면 다른 행렬과 모양이 어긋나 비교가 터진다."""
    return pd.DataFrame(
        {t: d[col] for t, d in panel.items() if col in d.columns}
    ).reindex(index=idx, columns=cols)


def build(panel: dict[str, pd.DataFrame]) -> dict:
    idx = sorted(set().union(*[d.index for d in panel.values()]))
    cols = sorted(panel)
    close = wide(panel, "Close", idx, cols).astype(float)
    yz = wide(panel, "YZ_60", idx, cols).astype(float)
    rsi = wide(panel, "RSI", idx, cols).astype(float)

    # 롤링 R²: corr(t, log C)². 단순회귀에서 R² = corr² 라 회귀를 풀 필요가 없다.
    y = np.log(close).replace([np.inf, -np.inf], np.nan)
    x = pd.Series(np.arange(len(y), dtype=float), index=y.index)
    r = y.apply(lambda c: x.rolling(WIN_R2, min_periods=WIN_R2).corr(c))
    r2 = r ** 2
    # 기울기 b = r · sd(y)/sd(x),  sd(x)² = (n²−1)/12  (등간격 x 의 닫힌 형태)
    slope = r.mul(y.rolling(WIN_R2, min_periods=WIN_R2).std()).div(
        np.sqrt((WIN_R2 * WIN_R2 - 1) / 12.0))
    past60 = close / close.shift(WIN_R2) - 1.0

    # 셀 배정 — 분할 기준은 **그날 전 종목 중앙값** (상대 분할)
    hi_r2 = r2.ge(r2.median(axis=1), axis=0)
    hi_yz = yz.ge(yz.median(axis=1), axis=0)
    rsi3 = pd.DataFrame(
        np.select([rsi < RSI_LO, rsi > RSI_HI], ["과매도", "과매수"], default="중간"),
        index=rsi.index, columns=rsi.columns).where(rsi.notna())
    quad = pd.DataFrame(
        np.where(hi_r2, "R²高·", "R²低·") + np.where(hi_yz, "YZ高", "YZ低"),
        index=r2.index, columns=r2.columns)
    cell = (quad + "·" + rsi3).where(r2.notna() & yz.notna() & rsi.notna())

    # BM 대비 초과 선행수익률
    if BM not in close.columns:
        raise SystemExit(f"❌ BM {BM} 이 패널에 없다 — 초과수익률을 낼 수 없다")
    bm = close[BM]
    fwd = {}
    for n in FWD:
        fwd[n] = (close.shift(-n) / close - 1.0).sub(bm.shift(-n) / bm - 1.0, axis=0)

    return dict(cell=cell, r2=r2, yz=yz, rsi=rsi, slope=slope, past60=past60, fwd=fwd)


# ─────────────────────────── 집계 ───────────────────────────
def compose(d: dict) -> pd.DataFrame:
    cell = d["cell"]
    tot = int(cell.notna().sum().sum())
    rows = []
    for c in CELLS:
        m = cell == c
        n = int(m.sum().sum())
        if n == 0:
            rows.append({"셀": c, "건수": 0})
            continue
        st = lambda f: d[f].where(m).stack()
        era = {k: int((cell.loc[a:b] == c).sum().sum()) for k, (a, b) in ERAS.items()}
        rows.append({
            "셀": c, "건수": n, "비중": n / tot,
            "종목수": int(m.any(axis=0).sum()),
            "R²": st("r2").median(), "YZ_60": st("yz").median(),
            "RSI": st("rsi").median(), "기울기": st("slope").median(),
            "상승비율": float((st("slope") > 0).mean()),
            "과거60일": st("past60").median(),
            "23-24": era["23-24"], "25-26": era["25-26"],
        })
    return pd.DataFrame(rows)


def returns(d: dict, stat: str, era: tuple[str, str] | None = None) -> pd.DataFrame:
    """셀 × 창 선행 초과수익률. stat = 'mean' | 'median'.

    ⚠ 맨 아래 **[전체 기준선]** 행이 이 표의 해석 기준이다. 셀을 0 과 비교하면
      안 된다 — 2023~2026 은 BM(코스피200)이 반도체 대형주에 끌려 매우 강했고
      시가총액 가중이라, **중앙 종목은 원래 BM 에 진다.** 그래서 대부분의 셀이
      음수로 나오는데 그건 셀의 성질이 아니라 기준선의 위치다.
      셀의 신호는 '음수냐'가 아니라 **'기준선보다 위냐 아래냐'** 로 읽는다.
    """
    cell = d["cell"] if era is None else d["cell"].loc[era[0]:era[1]]
    agg = (lambda s: s.mean()) if stat == "mean" else (lambda s: s.median())
    rows = []
    for c in CELLS + ["[전체 기준선]"]:
        m = cell.notna() if c == "[전체 기준선]" else (cell == c)
        row = {"셀": c, "건수": int(m.sum().sum())}
        for n in FWD:
            f = d["fwd"][n] if era is None else d["fwd"][n].loc[era[0]:era[1]]
            s = f.where(m).stack()
            row[f"{n}일"] = agg(s) if len(s) else np.nan
            if n == 20:
                row["20일_표본"] = int(len(s))
        rows.append(row)
    df = pd.DataFrame(rows)

    # 기준선 대비 초과분 (셀 − 전체). 부호가 곧 신호다.
    base = df[df["셀"] == "[전체 기준선]"].iloc[0]
    for n in FWD:
        df[f"Δ{n}일"] = df[f"{n}일"] - base[f"{n}일"]
    df.loc[df["셀"] == "[전체 기준선]", [f"Δ{n}일" for n in FWD]] = np.nan
    return df


# ─────────────────────────── 엑셀 ───────────────────────────
def write_xlsx(out: Path, comp: pd.DataFrame, tables: dict[str, pd.DataFrame],
               meta: dict) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    FONT = "Arial"
    hdr_fill = PatternFill("solid", fgColor="FF37474F")
    hdr_font = Font(name=FONT, bold=True, color="FFFFFFFF", size=10)
    band = PatternFill("solid", fgColor="FFF5F5F5")
    thin = Side(style="thin", color="FFBDBDBD")
    med = Side(style="medium", color="FF37474F")

    def style_header(ws, row, ncol):
        for j in range(1, ncol + 1):
            c = ws.cell(row=row, column=j)
            c.fill, c.font = hdr_fill, hdr_font
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            c.border = Border(bottom=med, left=thin, right=thin)
        ws.row_dimensions[row].height = 30

    def put_table(ws, df, r0, fmt, signed=frozenset(), widths=None, sep=None):
        """df 를 (r0, 1) 부터 쓴다. fmt = {열: number_format}.
        sep = 왼쪽에 굵은 구분선을 그을 열 이름 (블록 경계 표시용)."""
        for j, col in enumerate(df.columns, 1):
            ws.cell(row=r0, column=j, value=col)
        style_header(ws, r0, len(df.columns))
        sep_j = (list(df.columns).index(sep) + 1) if sep in list(df.columns) else None
        for i, (_, rec) in enumerate(df.iterrows()):
            r = r0 + 1 + i
            is_base = str(rec.iloc[0]).startswith("[")     # 기준선 행
            grp = i // 3                                   # 3행(RSI 3구분)씩 묶어 줄무늬
            for j, col in enumerate(df.columns, 1):
                v = rec[col]
                c = ws.cell(row=r, column=j,
                            value=(None if (isinstance(v, float) and np.isnan(v)) else v))
                c.font = Font(name=FONT, size=10,
                              bold=(j == 1 or is_base),
                              color=(UP if (col in signed and isinstance(v, (int, float))
                                            and not pd.isna(v) and v > 0)
                                     else DOWN if (col in signed and isinstance(v, (int, float))
                                                   and not pd.isna(v) and v < 0)
                                     else "FF000000"))
                c.number_format = fmt.get(col, "General")
                c.alignment = Alignment(horizontal="left" if j == 1 else "right")
                c.border = Border(left=(med if j == sep_j else thin), right=thin,
                                  top=(med if is_base else None),
                                  bottom=(med if (is_base or (i + 1) % 3 == 0) else thin))
                if is_base:
                    c.fill = PatternFill("solid", fgColor="FFFFF9C4")   # 기준선 강조
                elif grp % 2 == 1:
                    c.fill = band
        if widths:
            for j, w in enumerate(widths, 1):
                ws.column_dimensions[get_column_letter(j)].width = w
        return r0 + len(df) + 1

    def note(ws, row, text, bold=False, size=10, color="FF000000"):
        c = ws.cell(row=row, column=1, value=text)
        c.font = Font(name=FONT, size=size, bold=bold, color=color)
        c.alignment = Alignment(vertical="top", wrap_text=False)
        return row + 1

    wb = Workbook()

    # ── 시트 1: 읽는 법 ──────────────────────────────────────
    ws = wb.active
    ws.title = "읽는법"
    ws.column_dimensions["A"].width = 118
    ws.sheet_view.showGridLines = False
    r = 1
    r = note(ws, r, "R²(60) × YZ_60 × RSI  —  12셀 리포트", bold=True, size=16)
    r = note(ws, r, f"생성 {meta['generated_at']}   ·   {meta['source']}", size=9,
             color="FF757575"); r += 1

    for title, lines in [
        ("무엇을 나눈 표인가", [
            f"· R²(60)  = 최근 {WIN_R2}거래일 log(종가) 선형회귀의 결정계수. "
            "그날 전 종목 중앙값 기준으로 高/低 2분할.",
            "  높을수록 '한 방향으로 곧게' 움직인다는 뜻이다. 부드러움이 아니라 "
            "기울기 대비 잔차의 크기를 잰다.",
            "· YZ_60  = Yang-Zhang 변동성(60일). 같은 방식으로 高/低 2분할. 흔들림의 크기.",
            f"· RSI    = 과매도(<{RSI_LO:.0f}) / 중간 / 과매수(>{RSI_HI:.0f}) 3분할.",
            "  → 2 × 2 × 3 = 12셀. 셀은 종목의 속성이 아니라 **그날의 상태**다 "
            "(같은 종목이 날마다 다른 셀에 들어간다).",
        ]),
        ("종속변수 — 무엇으로 좋고 나쁨을 쟀나", [
            f"BM({BM} TIGER 200) 대비 **초과** 선행수익률.",
            "    excess(n) = (C[t+n]/C[t] − 1) − (BM[t+n]/BM[t] − 1)",
            f"창 n = {', '.join(str(n) for n in FWD)} 거래일. 시장 전체가 오른 몫을 빼고 "
            "종목 고유의 몫만 본다.",
        ]),
        ("★ 0 이 아니라 [전체 기준선] 과 비교할 것 — 이 표에서 가장 중요한 주의", [
            "수익률 시트의 노란 [전체 기준선] 행은 12셀을 합친 전체 관측의 값이다. "
            "20일 중앙값이 **−1.33%** 다.",
            "즉 이 기간에는 **중앙 종목이 원래 BM 에 진다.** 2023~2026 의 코스피200 이 "
            "반도체 대형주에 끌려 강했고 시가총액 가중이기 때문이다",
            "(평균으로 재면 기준선이 +0.12% 로 거의 0 인 것이 같은 사실의 뒷면이다 — "
            "평균 종목 ≈ BM, 중앙 종목 < BM).",
            "그래서 셀의 절대 수치가 음수인 것은 대개 셀의 성질이 아니라 기준선의 위치다. "
            "**신호는 오른쪽 Δ 블록의 부호**로 읽는다.",
            "실제로 기준선을 빼고 나니 평균과 중앙값의 모순이 대부분 사라졌다 — "
            "종전의 '11/12 셀이 음수' 는 셀의 문제가 아니었다.",
        ]),
        ("⚠ 그래도 남는 한계 (전부 실측에서 드러난 것)", [
            "1. 생존편향 — 지금 살아 있는 207종목의 과거다. 망해서 사라진 종목이 "
            "표본에 없으므로 전 셀이 낙관적이다. 기준선도 같이 낙관적이라",
            "   Δ 로 보면 상당 부분 상쇄되지만, 고모멘텀 종목이 사라진 편향은 "
            "과매수 셀에 특히 유리하게 남는다.",
            "2. 평균과 중앙값이 아직 갈리는 셀이 있다 — 특히 **YZ高·과매수**. "
            "평균 Δ는 +4%p 인데 중앙값 Δ는 ≈0 이다. 소수 대박이 만든 값이라",
            "   기대값으로 쓸 수는 있어도 '대체로 오른다' 고 읽으면 안 된다.",
            "3. 표본 겹침 — 20일 수익률을 매일 계산하므로 이웃 관측이 독립이 아니다. "
            "건수가 커 보여도 독립 표본은 훨씬 적다.",
            "4. 시대 쏠림 — 과매도 셀은 23-24 에, 과매수 셀은 25-26 에 몰려 있다"
            "(셀구성 시트의 마지막 두 열). 시대별 시트로 갈라 볼 것.",
            "5. 상대 분할 — 高/低는 그날 전 종목 중앙값 기준이라 셀 크기는 항상 대략 "
            "1/4씩이다. '절대적으로 높다' 가 아니다.",
        ]),
        ("소견 — 평균·중앙값이 **둘 다** 같은 부호인 것만 (Δ 기준, 20일)", [
            "◎ 과매도 4셀 전부 양(+) : Δ중앙 +1.9 ~ +3.3%p / Δ평균 +1.7 ~ +5.1%p. "
            "네 셀 모두, 거의 모든 창에서 양수다.",
            "   가장 강한 것은 **R²高·YZ高·과매도**(Δ중앙 +3.3%p, Δ평균 +5.1%p) — "
            "곧게 내리던 고변동 종목의 반등이다. 다만 1,060건뿐이다.",
            "◎ YZ低·과매수 2셀 전부 음(−) : R²高 Δ중앙 −1.3%p / R²低 −2.2%p 이고 "
            "**창이 길어질수록 나빠진다**(120일 −6.2 / −11.5%p).",
            "   변동성이 낮은데 RSI 가 과열이면 더 갈 힘이 없다는 뜻. "
            "회피·블랙리스트 후보로 가장 명확한 신호다.",
            "○ R²低·YZ高·중간 : Δ가 양쪽 다 양수이고 창이 길수록 커진다"
            "(20일 +0.1/+1.5%p → 120일 +1.6/+8.3%p). 표본도 35,616건으로 크다.",
            "△ YZ高·과매수 : 평균만 크게 양수, 중앙값은 ≈0. 위 한계 2 참조 — "
            "아직 규칙으로 쓰지 말 것.",
        ]),
        ("색 규약", [
            "양수 빨강(#EF5350) / 음수 파랑(#1976D2) — Kane 전역 지침. 숫자는 우측 정렬.",
        ]),
    ]:
        r = note(ws, r, title, bold=True, size=12)
        for ln in lines:
            r = note(ws, r, "   " + ln)
        r += 1

    # ── 시트 2: 셀 구성 ─────────────────────────────────────
    ws = wb.create_sheet("셀구성")
    ws.sheet_view.showGridLines = False
    r = note(ws, 1, "① 셀 구성 및 특성  —  값은 전부 중앙값", bold=True, size=13)
    r = note(ws, r, f"전 기간 유효 관측 {int(comp['건수'].sum()):,}건 (종목×일)   ·   "
                    f"'상승비율' = 회귀 기울기 b>0 인 비율", size=9, color="FF757575")
    end = put_table(
        ws, comp, r + 1,
        fmt={"건수": "#,##0", "비중": "0.0%", "종목수": "#,##0", "R²": "0.000",
             "YZ_60": "0.000", "RSI": "0", "기울기": "0.00000", "상승비율": "0%",
             "과거60일": "0.0%", "23-24": "#,##0", "25-26": "#,##0"},
        signed={"기울기", "과거60일"},
        widths=[20, 10, 8, 9, 8, 9, 7, 11, 10, 11, 10, 10])
    ws.freeze_panes = ws.cell(row=r + 2, column=2)
    end += 1
    for ln in [
        "읽기 — ① 표본이 치우쳐 있다: 중간 90.3% / 과매수 7.2% / 과매도 2.5%. "
        "과매도 셀은 824~1,406건뿐이다.",
        "② R²와 RSI가 이미 얽혀 있다: R²高·과매도의 '상승비율'이 2%, R²高·과매수가 98~99%. "
        "R²가 높은 종목이 RSI 극단에 가면 방향이 사실상 정해진다 — "
        "'방향' 축을 따로 둘 실익이 R²高 쪽에는 없다.",
        "③ 과거60일 수익률이 12셀을 −31.7% ~ +75.6% 로 거의 단조 정렬한다. "
        "YZ가 높을수록 양 끝이 더 벌어진다 — YZ가 진폭 배율 역할을 한다.",
        "④ R²低·YZ低·중간(41,298건, 24.6%)이 유일한 '진짜 횡보'다. "
        "과거60일 +0.4%, 기울기 ≈0, 상승 52%. 시장의 기본값이 여기다.",
        "⑤ 종목수가 전부 116~191이다. 어느 종목이든 12셀을 돌아다닌다 — "
        "셀은 고정 분류가 아니라 매일 다시 판정해야 하는 상태다.",
    ]:
        end = note(ws, end, ln, size=9)

    # ── 시트 3~: 수익률 ─────────────────────────────────────
    for name, (df, subtitle) in tables.items():
        ws = wb.create_sheet(name)
        ws.sheet_view.showGridLines = False
        r = note(ws, 1, f"{name}  —  BM(102110) 대비 초과 선행수익률", bold=True, size=13)
        r = note(ws, r, subtitle, size=9, color="FF757575")
        pct = [c for c in df.columns if c.endswith("일") and c != "20일_표본"]
        end = put_table(
            ws, df, r + 1,
            fmt={**{c: "0.00%" for c in pct}, "건수": "#,##0", "20일_표본": "#,##0"},
            signed=set(pct), sep=f"Δ{FWD[0]}일",
            widths=[[20, 10][j] if j < 2 else (10 if c == "20일_표본" else 9)
                    for j, c in enumerate(df.columns)])
        ws.freeze_panes = ws.cell(row=r + 2, column=3)
        end += 1
        for ln in [
            "⚠ 셀을 0 과 비교하지 말 것. 2023~2026 은 BM(코스피200)이 반도체 대형주에 "
            "끌려 매우 강했고 시가총액 가중이라 **중앙 종목은 원래 BM 에 진다.**",
            "   대부분의 셀이 음수인 것은 셀의 성질이 아니라 기준선의 위치다. "
            "노란 [전체 기준선] 행이 그 위치이고, 오른쪽 Δ 블록이 그 대비 초과분이다.",
            "   **신호는 Δ 의 부호다** — Δ가 양수면 그 셀은 시장의 중앙보다 낫다는 뜻.",
        ]:
            end = note(ws, end, ln, size=9)

    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    print("· 패널 로드")
    panel = load_panel()
    print(f"  {len(panel)}종목")
    print("· 지표·셀 계산")
    d = build(panel)
    print("· 집계")
    comp = compose(d)
    tables = {
        "수익률_중앙값": (returns(d, "median"), "전 기간 · 중앙값 — **먼저 볼 표**"),
        "수익률_평균": (returns(d, "mean"),
                     "전 기간 · 평균 — 소수 대박에 끌려간다. 중앙값과 함께 볼 것"),
        "시대별_23-24_중앙": (returns(d, "median", ERAS["23-24"]), "2023-01 ~ 2024-12 · 중앙값"),
        "시대별_25-26_중앙": (returns(d, "median", ERAS["25-26"]), "2025-01 ~ 2026-12 · 중앙값"),
    }
    print(f"· 엑셀 작성 → {args.out}")
    write_xlsx(args.out, comp, tables, {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M KST"),
        "source": "MagicFormula/scripts/cell12_report.py  (원자료: LLV core+extend parquet)",
    })
    print("✅ 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
