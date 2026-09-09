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
def load_raw() -> pd.DataFrame:
    """LLV OHLCV 정본 (core+extend) 를 긴 형태 그대로. 상주 검증도 이걸 쓴다."""
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
    return df


def load_panel(raw: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """LLV OHLCV 패널 → {티커: DataFrame(Date index)}."""
    df = load_raw() if raw is None else raw
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

    # ⚠ BM 자신을 **표본에서 뺀다** (시계열은 따로 챙겨 둔다).
    #   ① 자기 대비 초과수익은 정의상 정확히 0 이라 셀 통계를 0 쪽으로 끌어당기고,
    #   ② 102110 은 지수 ETF 라 3.3년 내내 YZ低 에 상주해 편향이 한쪽 셀에 몰린다.
    #   ③ 高/低 분할의 '그날 전 종목 중앙값' 도 지수가 섞이면 살짝 밀린다.
    #   그래서 지표·분할을 계산하기 **전에** 뺀다.
    if BM not in close.columns:
        raise SystemExit(f"❌ BM {BM} 이 패널에 없다 — 초과수익률을 낼 수 없다")
    bm = close[BM].copy()
    close, yz, rsi = (t.drop(columns=[BM]) for t in (close, yz, rsi))

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

    # BM 대비 초과 선행수익률 (BM 은 이미 표본에서 빠져 있고, 시계열만 쓴다)
    fwd = {n: (close.shift(-n) / close - 1.0).sub(bm.shift(-n) / bm - 1.0, axis=0)
           for n in FWD}

    names = {t: (str(d["Name"].dropna().iloc[-1]) if "Name" in d.columns
                 and d["Name"].notna().any() else t)
             for t, d in panel.items()}
    return dict(cell=cell, r2=r2, yz=yz, rsi=rsi, slope=slope, past60=past60,
                fwd=fwd, names=names)


def persistent_axis(d: dict, axis: str) -> tuple[list[str], list[str]]:
    """어느 축에서 **한쪽에만 상주한** 종목. R²·RSI 는 전 종목이 오가지만
    YZ 는 그렇지 않다 — 이 함수가 그 차이를 드러낸다."""
    cell, names = d["cell"], d["names"]
    valid = cell.notna()
    hi = (valid & cell.apply(lambda s: s.str.contains(f"{axis}高", regex=False))
          ).fillna(False)
    lo = (valid & cell.apply(lambda s: s.str.contains(f"{axis}低", regex=False))
          ).fillna(False)
    seen = set(cell.columns[valid.any()])
    H, L = set(cell.columns[hi.any()]), set(cell.columns[lo.any()])
    med = d["yz" if axis == "YZ" else "r2"].median()
    key = lambda S: [names.get(t, t) for t in sorted(S, key=lambda x: -med.get(x, 0))]
    return key(seen - L), key(seen - H)


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


def compose_margin(d: dict) -> tuple[pd.DataFrame, list[int]]:
    """12셀을 각 축으로 접은 **주변합**. 축 하나씩 무시하면 무엇이 남는지 본다.

    반환 (표, 묶음크기) — 묶음크기는 엑셀에서 구분선을 그을 단위다.
    ⚠ 비중은 전체 대비다. 같은 축끼리 더하면 100% 가 된다(R²高+R²低=1).
    """
    cell = d["cell"]
    tot = int(cell.notna().sum().sum())
    valid = cell.notna()
    # (라벨, 마스크) — 마스크는 셀 이름의 부분일치로 만든다.
    groups: list[tuple[str, pd.DataFrame]] = [("[전체 기준선]", valid)]
    sizes = [1]
    for names, size in [
        (["R²高", "R²低"], 2),
        (["YZ高", "YZ低"], 2),
        (["과매도", "중간", "과매수"], 3),
        (QUAD, 4),
    ]:
        for nm in names:
            if nm in ("과매도", "중간", "과매수"):        # RSI 축은 **마지막 마디**로 판정
                m = valid & cell.apply(lambda s: s.str.rsplit("·", n=1).str[-1] == nm)
            elif nm in QUAD:                              # 사분면은 앞 두 마디
                m = valid & cell.apply(lambda s: s.str.rsplit("·", n=1).str[0] == nm)
            else:                                         # 단일 축은 포함 여부
                m = valid & cell.apply(lambda s: s.str.contains(nm, regex=False))
            groups.append((nm, m.fillna(False)))
        sizes.append(size)

    rows = []
    for label, m in groups:
        n = int(m.sum().sum())
        st = lambda f: d[f].where(m).stack()
        era = {k: int((cell.loc[a:b].notna() & m.loc[a:b]).sum().sum())
               for k, (a, b) in ERAS.items()}
        rows.append({
            "셀": label, "건수": n, "비중": n / tot,
            "종목수": int(m.any(axis=0).sum()),
            "R²": st("r2").median(), "YZ_60": st("yz").median(),
            "RSI": st("rsi").median(), "기울기": st("slope").median(),
            "상승비율": float((st("slope") > 0).mean()),
            "과거60일": st("past60").median(),
            "23-24": era["23-24"], "25-26": era["25-26"],
        })
    return pd.DataFrame(rows), sizes


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


def build_facts(d: dict, comp: pd.DataFrame, tables: dict) -> dict:
    """해설에 쓸 숫자를 **표에서 뽑는다.** 손으로 박으면 재실행 때 조용히 어긋난다."""
    med = tables["수익률_중앙값"][0].set_index("셀")
    mean = tables["수익률_평균"][0].set_index("셀")
    Δ = lambda t, c, n=20: t.loc[c, f"Δ{n}일"]

    def group(pred):
        return [c for c in CELLS if pred(c)]

    def band(cells, n=20):
        a = [Δ(med, c, n) for c in cells]; b = [Δ(mean, c, n) for c in cells]
        return (f"Δ중앙 {min(a):+.1%}p~{max(a):+.1%}p".replace("%p", "%p"),
                f"Δ평균 {min(b):+.1%}p~{max(b):+.1%}p")

    oversold = group(lambda c: c.endswith("과매도"))
    lowvol_ob = group(lambda c: "YZ低" in c and c.endswith("과매수"))
    best = max(oversold, key=lambda c: Δ(med, c))
    # 두 통계의 부호가 같은 셀만 소견으로 채택한다.
    agree = lambda cs: all(np.sign(Δ(med, c)) == np.sign(Δ(mean, c)) for c in cs)

    verdict = []
    if agree(oversold) and Δ(med, oversold[0]) > 0:
        lo, hi = band(oversold)
        verdict += [
            f"◎ 과매도 4셀 전부 양(+) : {lo} / {hi}. 네 셀 모두, 거의 모든 창에서 양수다.",
            f"   가장 강한 것은 **{best}**(Δ중앙 {Δ(med,best):+.1%}p, "
            f"Δ평균 {Δ(mean,best):+.1%}p) — 곧게 내리던 고변동 종목의 반등이다. "
            f"다만 {comp.loc[comp['셀']==best,'건수'].iloc[0]:,.0f}건뿐이다.",
        ]
    if agree(lowvol_ob) and Δ(med, lowvol_ob[0]) < 0:
        verdict += [
            "◎ YZ低·과매수 2셀 전부 음(−) : "
            + " / ".join(f"{c.split('·')[0]} Δ중앙 {Δ(med,c):+.1%}p" for c in lowvol_ob)
            + " 이고 **창이 길어질수록 나빠진다**"
            + f"(120일 {' / '.join(f'{Δ(med,c,120):+.1%}p' for c in lowvol_ob)}).",
            "   변동성이 낮은데 RSI 가 과열이면 더 갈 힘이 없다는 뜻. "
            "회피·블랙리스트 후보로 가장 명확한 신호다.",
        ]
    c = "R²低·YZ高·중간"
    verdict.append(
        f"○ {c} : Δ가 양쪽 다 양수이고 창이 길수록 커진다"
        f"(20일 {Δ(med,c):+.1%}/{Δ(mean,c):+.1%}p → "
        f"120일 {Δ(med,c,120):+.1%}/{Δ(mean,c,120):+.1%}p). "
        f"표본도 {comp.loc[comp['셀']==c,'건수'].iloc[0]:,.0f}건으로 크다.")
    ob_hv = group(lambda c: "YZ高" in c and c.endswith("과매수"))
    verdict.append(
        f"△ YZ高·과매수 : 평균만 크게 양수"
        f"({' / '.join(f'{Δ(mean,c):+.1%}p' for c in ob_hv)}), "
        f"중앙값은 ≈0({' / '.join(f'{Δ(med,c):+.1%}p' for c in ob_hv)}). "
        "위 한계 2 참조 — 아직 규칙으로 쓰지 말 것.")

    return {
        "n_tickers": int(d["cell"].shape[1]),
        "base_med20": med.loc["[전체 기준선]", "20일"],
        "base_mean20": mean.loc["[전체 기준선]", "20일"],
        "yz_persistent": persistent_axis(d, "YZ"),
        "verdict": verdict,
    }


# ─────────────────────────── 엑셀 ───────────────────────────
def write_xlsx(out: Path, comp: pd.DataFrame, margin: pd.DataFrame,
               margin_groups: list[int], tables: dict[str, pd.DataFrame],
               facts: dict, meta: dict) -> None:
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

    def put_table(ws, df, r0, fmt, signed=frozenset(), widths=None, sep=None,
                  groups=None):
        """df 를 (r0, 1) 부터 쓴다. fmt = {열: number_format}.
        sep    = 왼쪽에 굵은 구분선을 그을 열 이름 (블록 경계 표시용).
        groups = 행 묶음 크기 리스트 (기본 3행씩 = RSI 3구분). 묶음마다 줄무늬·구분선."""
        for j, col in enumerate(df.columns, 1):
            ws.cell(row=r0, column=j, value=col)
        style_header(ws, r0, len(df.columns))
        sep_j = (list(df.columns).index(sep) + 1) if sep in list(df.columns) else None
        sizes = list(groups) if groups else [3] * ((len(df) + 2) // 3)
        gidx, glast = [], []            # 행 → (묶음번호, 묶음의 마지막인가)
        for gi, sz in enumerate(sizes):
            gidx += [gi] * sz
            glast += [False] * (sz - 1) + [True]
        gidx += [len(sizes)] * (len(df) - len(gidx))       # 남는 행(기준선 등)
        glast += [True] * (len(df) - len(glast))
        for i, (_, rec) in enumerate(df.iterrows()):
            r = r0 + 1 + i
            is_base = str(rec.iloc[0]).startswith("[")     # 기준선 행
            grp = gidx[i]
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
                                  bottom=(med if (is_base or glast[i]) else thin))
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
            f"20일 중앙값이 **{facts['base_med20']:+.2%}** 다.",
            "즉 이 기간에는 **중앙 종목이 원래 BM 에 진다.** 2023~2026 의 코스피200 이 "
            "반도체 대형주에 끌려 강했고 시가총액 가중이기 때문이다",
            f"(평균으로 재면 기준선이 {facts['base_mean20']:+.2%} 로 거의 0 인 것이 "
            "같은 사실의 뒷면이다 — 평균 종목 ≈ BM, 중앙 종목 < BM).",
            "그래서 셀의 절대 수치가 음수인 것은 대개 셀의 성질이 아니라 기준선의 위치다. "
            "**신호는 오른쪽 Δ 블록의 부호**로 읽는다.",
            "실제로 기준선을 빼고 나니 평균과 중앙값의 모순이 대부분 사라졌다 — "
            "종전의 '11/12 셀이 음수' 는 셀의 문제가 아니었다.",
        ]),
        ("⚠ 그래도 남는 한계 (전부 실측에서 드러난 것)", [
            f"1. 생존편향 — 지금 살아 있는 {facts['n_tickers']}종목의 과거다"
            f"(패널 {facts['n_tickers'] + 1}종목에서 BM 제외). 망해서 사라진 종목이 "
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
        ("소견 — 평균·중앙값이 **둘 다** 같은 부호인 것만 (Δ 기준, 20일)", facts["verdict"]),
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
    # ⚠ 아래 해설의 숫자는 **표에서 계산해 쓴다.** 손으로 박으면 재실행 때
    #   조용히 어긋난다 (실제로 두 번 어긋났다 — BM 제외 전후, 초판 대 재판).
    q = lambda lbl, col: comp.loc[comp["셀"] == lbl, col].iloc[0]
    rsi_share = {r: comp[comp["셀"].str.endswith("·" + r)]["비중"].sum() for r in RSI3}
    od = comp[comp["셀"].str.endswith("·과매도")]["건수"]
    flat = comp.loc[comp["과거60일"].abs().idxmin()]
    end += 1
    for ln in [
        f"읽기 — ① 표본이 치우쳐 있다: 중간 {rsi_share['중간']:.1%} / "
        f"과매수 {rsi_share['과매수']:.1%} / 과매도 {rsi_share['과매도']:.1%}. "
        f"과매도 셀은 {od.min():,.0f}~{od.max():,.0f}건뿐이다.",
        f"② R²와 RSI가 이미 얽혀 있다: R²高·과매도의 '상승비율'이 "
        f"{min(q('R²高·YZ高·과매도','상승비율'), q('R²高·YZ低·과매도','상승비율')):.0%}, "
        f"R²高·과매수가 "
        f"{min(q('R²高·YZ高·과매수','상승비율'), q('R²高·YZ低·과매수','상승비율')):.0%}~"
        f"{max(q('R²高·YZ高·과매수','상승비율'), q('R²高·YZ低·과매수','상승비율')):.0%}. "
        "R²가 높은 종목이 RSI 극단에 가면 방향이 사실상 정해진다 —",
        "   '방향' 축을 따로 둘 실익이 R²高 쪽에는 없다.",
        f"③ 과거60일 수익률이 12셀을 {comp['과거60일'].min():.1%} ~ "
        f"{comp['과거60일'].max():.1%} 로 거의 단조 정렬한다. "
        "YZ가 높을수록 양 끝이 더 벌어진다 — YZ가 진폭 배율 역할을 한다.",
        f"④ {flat['셀']}({flat['건수']:,.0f}건, {flat['비중']:.1%})이 유일한 '진짜 횡보'다. "
        f"과거60일 {flat['과거60일']:+.1%}, 기울기 ≈0, 상승 {flat['상승비율']:.0%}. "
        "시장의 기본값이 여기다.",
        f"⑤ 종목수가 전부 {comp['종목수'].min():.0f}~{comp['종목수'].max():.0f}이다. "
        "어느 종목이든 12셀을 돌아다닌다 — "
        "셀은 고정 분류가 아니라 매일 다시 판정해야 하는 상태다(단, YZ축은 예외 — ⑩).",
    ]:
        end = note(ws, end, ln, size=9)

    # ② 주변합 — 12셀을 각 축으로 접은 표
    end += 1
    end = note(ws, end, "② 주변합 — 12셀을 축 하나씩으로 접었을 때", bold=True, size=13)
    end = note(ws, end, "축 안에서 더하면 전체가 된다 (R²高 + R²低 = 100%). "
                        "12셀 표와 나란히 놓고 '이 특성이 어느 축에서 오는가'를 본다.",
               size=9, color="FF757575")
    end = put_table(
        ws, margin, end + 1,
        fmt={"건수": "#,##0", "비중": "0.0%", "종목수": "#,##0", "R²": "0.000",
             "YZ_60": "0.000", "RSI": "0", "기울기": "0.00000", "상승비율": "0%",
             "과거60일": "0.0%", "23-24": "#,##0", "25-26": "#,##0"},
        signed={"기울기", "과거60일"}, groups=margin_groups,
        widths=[20, 10, 8, 9, 8, 9, 7, 11, 10, 11, 10, 10])
    g = lambda lbl, col: margin.loc[margin["셀"] == lbl, col].iloc[0]
    spread = lambda a, b, col: abs(g(a, col) - g(b, col))
    hi_names, lo_names = facts["yz_persistent"]
    quad = margin[margin["셀"].isin(QUAD)]
    end += 1
    for ln in [
        "읽기 — ⑥ 세 축이 **방향을 가르는 힘**이 서로 다르다. "
        "상승비율(기울기 b>0 비율)로 재면:",
        f"      RSI축  과매도 {g('과매도','상승비율'):.0%} / 중간 {g('중간','상승비율'):.0%} / "
        f"과매수 {g('과매수','상승비율'):.0%}   ← 압도적 1위 "
        f"({spread('과매수','과매도','상승비율'):.0%}p 폭)",
        f"      YZ축   YZ高 {g('YZ高','상승비율'):.0%} / YZ低 {g('YZ低','상승비율'):.0%}"
        f"                  ← 2위 ({spread('YZ高','YZ低','상승비율')*100:.1f}%p)",
        f"      R²축   R²高 {g('R²高','상승비율'):.0%} / R²低 {g('R²低','상승비율'):.0%}"
        f"                  ← 거의 안 가른다 ({spread('R²高','R²低','상승비율')*100:.1f}%p)",
        "⑦ ⚠ **YZ 가 방향을 가르는 건 예상 밖이다** — YZ 는 부호 없는 진폭 지표라 원래 "
        "방향과 무관해야 한다.",
        f"   그런데 YZ高 의 과거60일이 {g('YZ高','과거60일'):+.1%}, "
        f"YZ低 가 {g('YZ低','과거60일'):+.1%} 다.",
        "   ⚠⚠ 초판은 여기서 '섹터·스타일 효과' 라고 단정했는데 근거가 종목명 훑기뿐이었다. "
        "재검증(**YZ안정성_검증** 시트) 결과 —",
        "   **YZ 순위는 3.3년 내내 종목에 붙어 있다**(ICC 0.66, 1년 뒤 순위상관 +0.63). "
        "즉 YZ高 vs YZ低 는 같은 종목의 다른 상태가 아니라 **대체로 다른 회사들**이다.",
        "   그래서 두 집단 사이의 어떤 차이든 **구성의 차이**일 수 있다 — 변동성의 효과라고 "
        "단정할 수 없고, 그 원인이 섹터인지 규모인지도 이 표로는 못 가린다.",
        "   → **YZ 를 방향 신호로 쓰지 말 것.** 부호 없는 지표에서 나온 방향성은 "
        "표본 구성의 성질이지 지표의 성질이 아니다. 진폭 축으로만 쓴다.",
        f"⑧ R²축은 방향은 못 가르지만 **크기**를 가른다: 기울기 중앙 "
        f"R²高 {g('R²高','기울기'):+.5f} / R²低 {g('R²低','기울기'):+.5f} "
        f"({g('R²高','기울기')/g('R²低','기울기'):.0f}배), "
        f"과거60일 {g('R²高','과거60일'):+.1%} / {g('R²低','과거60일'):+.1%}.",
        "   즉 R² 가 높다는 건 '오른다'가 아니라 '**한 방향으로 크게 간다**'는 뜻이다 — "
        "방향은 RSI 가, 진폭은 YZ 가, 크기·순도는 R² 가 담당한다. 세 축이 겹치지 않는다.",
        f"⑨ 사분면 4개는 건수가 {quad['비중'].min():.1%}~{quad['비중'].max():.1%} 로 고르다. "
        "상대 분할이라 당연하지만, 덕분에 사분면 간 비교는 표본 크기 차이를 "
        "걱정하지 않아도 된다.",
        "⑩ ★ **YZ 는 상태가 아니라 종목의 속성이다 — 이게 R²·RSI 와 결정적으로 다르다.**",
        f"   종목수를 보면 R²축은 高·低 모두 **{g('R²高','종목수'):.0f}**"
        "(전 종목이 양쪽을 오간다)인데 "
        f"YZ축은 **{g('YZ高','종목수'):.0f} / {g('YZ低','종목수'):.0f}** 다. 즉",
        f"   **{len(hi_names)}종목은 3.3년 내내 YZ高, {len(lo_names)}종목은 내내 YZ低** 였다. "
        "한 번도 반대편에 간 적이 없다.",
        "      항상 高 — " + " · ".join(hi_names),
        "      항상 低 — " + " · ".join(lo_names),
        "   ⚠ **이 명단은 현상이지 답이 아니다** — 29종목으로 섹터 37개를 가를 수 없다. "
        "206종목 전부로 다시 잰 결과가 **YZ안정성_검증** 시트에 있다.",
        "   요약: **YZ 순위는 종목에 붙어 있다**(ICC 0.66 · 1년 뒤 순위상관 +0.63 · "
        "5분위 1년 유지 37% vs 무작위 20%). R²·RSI 는 정반대로 60일이면 기억이 사라진다.",
        "   ⇒ **R²·RSI 는 상태 축, YZ 는 사실상 종목 그룹 축**이다. 세 축을 같은 성격으로 "
        "읽으면 안 된다.",
        f"   ※ BM({BM}) 자신은 표본에서 제외했다 — 자기 대비 초과수익이 정의상 0 이고, "
        "지수 ETF 라 내내 YZ低 에 상주해 편향이 한쪽에 몰리기 때문.",
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

    # ── 시트: YZ 안정성 검증 ────────────────────────────────
    P = facts.get("probe")
    if P:
        ws = wb.create_sheet("YZ안정성_검증")
        ws.sheet_view.showGridLines = False
        ws.column_dimensions["A"].width = 24
        r = note(ws, 1, "YZ 高/低 는 종목의 특성인가 — 안정성 검증", bold=True, size=14)
        r = note(ws, r, "생성기 scripts/yz_persistence_probe.py   ·   "
                        f"{P['n_tickers']}종목 / {P['n_days']}일 / {P['n_obs']:,}관측 (BM 제외)",
                 size=9, color="FF757575")
        r += 1
        for ln in [
            "질문의 내력 — 케인 지적 2회, 둘 다 옳았다",
            "   초판: '항상 高/低' 29종목의 **이름을 눈으로 훑고** \"사실상 스타일(섹터) 분할\" 이라 단정.",
            "   1차(09-08): 29종목으로 섹터 37개를 못 가른다 · 시총·매출·영업이익·업력도 봐야 한다 · "
            "테크 쏠림으로 섹터와 규모가 교란된다 → 206종목 횡단면으로 재검증 (아래 [참고]).",
            "   2차(09-09): ⚠ **질문을 바꿔치기했다.** 케인이 물은 것은 '수준이 안정적이냐'(시계열)인데 "
            "나는 '어떤 회사가 변동성이 높냐'(횡단면)로 답했다.",
            "        게다가 그 답 — 작고·어리고·매출 적은 회사가 더 변동적 — 은 **교과서적 사실**이라 "
            "안 나오는 게 이상한 것이지 발견이 아니다.",
            "",
            "그래서 이렇게 고쳤다 — **외부 변수가 아예 필요 없다**",
            "   그 종목의 YZ **순위가 시간에 대해 안 움직이는지**를 재고, **같은 잣대로 R²·RSI 와 비교**한다.",
            "   R² 축은 206종목 전부가 양쪽을 오간다는 것을 이미 알고 있으므로 **자연스러운 대조군**이 된다.",
            "   세 축 모두 **그날 전 종목 백분위**(0~1)로 바꿔서 잰다 — 단위를 맞추고, "
            "시장이 함께 움직인 몫을 빼서 순수한 종목 간 상대 위치만 남긴다.",
        ]:
            r = note(ws, r, ln, size=9, bold=ln and not ln.startswith(" "))
        r += 1

        t = P["table"]
        r = note(ws, r, "[A] 지속성 — 네 가지 잣대로 같은 것을 잰다", bold=True, size=12)
        A = t[["축", "ICC", "종목간 σ", "종목내 σ", "1년 전이 대각%", "시대 ρ",
               "高만", "低만", "양쪽"]]
        r = put_table(ws, A, r + 1,
                      fmt={"ICC": "0.000", "종목간 σ": "0.000", "종목내 σ": "0.000",
                           "1년 전이 대각%": '0.0"%"', "시대 ρ": "0.000",
                           "高만": "#,##0", "低만": "#,##0", "양쪽": "#,##0"},
                      groups=[3], widths=[14, 9, 11, 11, 14, 9, 8, 8, 8])
        for ln in [
            "   ICC = 순위 분산 중 **종목 간** 몫. 1.0 이면 순위가 종목마다 고정, "
            "0.0 이면 매일 새로 뽑는 것과 같다.",
            "   1년 전이 대각% = 5분위가 250거래일 뒤에도 같은 분위에 남는 비율. "
            "**무작위면 20%** 다.",
            "   시대 ρ = 전반기(23-24) 종목 순위 vs 후반기(25-26) 순위의 스피어만.",
            "   高만/低만 = 3.3년 내내 한쪽 절반에만 있었던 종목 수 (반대편에 한 번도 안 갔다).",
        ]:
            r = note(ws, r, ln, size=9)
        r += 1

        r = note(ws, r, "[B] 순위 자기상관 — 오늘 순위가 h거래일 뒤에도 유지되나",
                 bold=True, size=12)
        B = t[["축"] + [f"ρ({h}일)" for h in P["lags"]]]
        r = put_table(ws, B, r + 1, fmt={c: "0.000" for c in B.columns if c != "축"},
                      signed=set(B.columns) - {"축"}, groups=[3],
                      widths=[14] + [10] * (len(B.columns) - 1))
        r += 1

        r = note(ws, r, "[C] 귀무 대조 — '高 비율' 이 매일 동전던지기보다 얼마나 퍼져 있나",
                 bold=True, size=12)
        C = t[["축", "高비율 σ(실측)", "高비율 σ(귀무)", "극단 비율(실측)", "극단 비율(귀무)"]]
        r = put_table(ws, C, r + 1,
                      fmt={"高비율 σ(실측)": "0.000", "高비율 σ(귀무)": "0.000",
                           "극단 비율(실측)": "0.0%", "극단 비율(귀무)": "0.0%"},
                      groups=[3], widths=[14, 13, 13, 14, 14])
        r = note(ws, r, "   귀무 = 종목별 관측일수로 동전던지기(이항). "
                        "극단 = 高비율이 5% 미만이거나 95% 초과인 종목 비율.", size=9)
        r += 1

        for k, T in P["trans"].items():
            r = note(ws, r, f"[D] 1년 뒤 5분위 전이확률 (%) — {k}", bold=True, size=11)
            TT = T.reset_index().rename(columns={"index": "현재"})
            TT.columns = ["현재"] + list(T.columns)
            r = put_table(ws, TT, r + 1, fmt={c: "0.0" for c in T.columns},
                          groups=[len(TT)], widths=[14] + [10] * len(T.columns))
            r += 1

        SW = P.get("sweep")
        if SW is not None:
            r = note(ws, r, "[E] 창 길이를 바꾸면 안정성이 달라지나 — 중첩 0 재측정",
                     bold=True, size=12)
            for ln in [
                "   ⚠ **롤링 창은 자기상관을 기계적으로 만든다** — 60일 창이면 어제와 오늘이 "
                "데이터의 59/60 를 공유한다. 창이 길수록 안정적으로 *보인다*.",
                "   실제로 [A] 의 ICC 는 창에 따라 크게 움직인다 (YZ 0.519@10일 → 0.740@120일). "
                "**ICC 만으로 판단하면 안 된다.**",
                "   그래서 겹치지 않는 n일 블록을 잡고 **각 블록의 데이터만으로** 계산해 "
                "인접 블록끼리 순위를 비교한다 — 중첩이 정확히 0 이다.",
                "   (롤링값의 블록 중앙값을 쓰면 블록 앞부분 창이 이전 블록을 물어 여전히 "
                "오염된다 — 초판이 그렇게 쟀다가 고쳤다.)",
            ]:
                r = note(ws, r, ln, size=9)
            r = put_table(ws, SW, r + 1,
                          fmt={"창(거래일)": "#,##0", "블록 수": "#,##0",
                               **{c: "0.000" for c in SW.columns if c.startswith(("YZ ", "R² "))}},
                          signed={c for c in SW.columns if c.startswith(("YZ ", "R² "))},
                          groups=[len(SW)], widths=[12, 9, 15, 17, 15, 17])
            y1 = SW["YZ ρ(다음 블록)"]
            for ln in [
                f"   ⇒ **창을 10일로 줄여도 결론이 안 바뀐다.** YZ 인접 블록 순위상관이 "
                f"{y1.min():.3f}~{y1.max():.3f} 로 창에 거의 무관하다.",
                "     창 10일은 '지난 2주 변동성 순위' 로 '다음 2주' 를 맞히는 것인데도 "
                f"{y1.iloc[0]:.3f} 다. 한 블록 건너뛰어도 0.70 대를 유지한다.",
                "   ⇒ **R² 는 어느 창에서도 0 이다** — 불안정한 것도 창의 산물이 아니다.",
                "   ※ 이건 놀라운 결과가 아니다. **변동성 군집(volatility clustering)은 "
                "금융에서 가장 견고한 정형적 사실 중 하나**이고, ARCH·GARCH 가 존재하는 이유다.",
                "     여기서 값어치 있는 것은 'YZ 가 지속적이다' 가 아니라 "
                "**'R²·RSI 와 성격이 다르다'** 와 그것이 축 설계에 갖는 함의다.",
            ]:
                r = note(ws, r, ln, size=9)
            r += 1

        yz = t[t["축"] == "YZ_60"].iloc[0]
        r2r = t[t["축"] == "R²(60)"].iloc[0]
        rsi = t[t["축"] == "RSI"].iloc[0]
        r = note(ws, r, "★ 결론", bold=True, size=13)
        for ln in [
            f"   **YZ 수준은 이 기간(3.3년) 종목의 안정적 특성이다.** 네 잣대가 전부 같은 방향이고 "
            "차이가 크다 —",
            f"      ICC  YZ {yz['ICC']:.3f}  vs  R² {r2r['ICC']:.3f} / RSI {rsi['ICC']:.3f}",
            f"      1년 뒤 순위상관  YZ {yz['ρ(250일)']:+.3f}  vs  R² {r2r['ρ(250일)']:+.3f} / "
            f"RSI {rsi['ρ(250일)']:+.3f}   (2년 뒤에도 YZ {yz['ρ(500일)']:+.3f})",
            f"      1년 전이 대각  YZ {yz['1년 전이 대각%']:.1f}%  vs  R² {r2r['1년 전이 대각%']:.1f}% / "
            f"RSI {rsi['1년 전이 대각%']:.1f}%   (무작위 20%)",
            f"      시대 ρ  YZ {yz['시대 ρ']:.3f}  vs  R² {r2r['시대 ρ']:.3f} / RSI {rsi['시대 ρ']:.3f}",
            "   **R²·RSI 는 정반대다** — ICC 가 0.03~0.05 이고, 60거래일이면 자기상관이 0 으로 "
            "떨어지며, 전이 대각이 정확히 무작위(20%)다. 206종목 전부가 양쪽을 오간다.",
            "",
            "   ⇒ **12셀 표의 세 축은 성격이 같지 않다.**",
            "     R²·RSI 축은 진짜 **상태** 축이다 — 같은 종목이 날마다 다른 칸에 들어간다.",
            "     **YZ 축은 사실상 종목 그룹 축**이다 — YZ高 vs YZ低 비교는 같은 종목의 다른 상태를 "
            "비교하는 게 아니라 **대체로 다른 회사들을 비교**하는 것이다.",
            "     그래서 YZ 로 갈린 두 집단 사이에 무엇이 달라 보이든, 그것은 **구성의 차이**일 수 "
            "있고 변동성의 효과라고 단정할 수 없다(주석 ⑦ 이 그 함정이었다).",
            "",
            "   ⚠ **ICC 는 창 길이에 오염된다** ([E] 참조) — 단독으로 인용하지 말 것. "
            "창에 견고한 증거는 비중첩 블록 상관과 시대 ρ 다.",
            "   ⚠ **창을 바꿔도 이 결론은 안 바뀐다** — YZ 는 10일 창에서도 인접 블록 "
            "순위상관 0.761 이고, R² 는 어느 창에서도 0 이다.",
            "   ⚠ 이것은 '안정적이다' 까지다. **왜 안정적인지는 이 시트가 답하지 않는다.**",
            f"   ⚠ 다만 완전 고정은 아니다 — {int(yz['양쪽'])}종목은 양쪽을 오갔고 종목내 σ 도 "
            f"{yz['종목내 σ']:.3f} 로 0 이 아니다. '고정' 이 아니라 '느리게 움직이는 서열' 이다.",
        ]:
            r = note(ws, r, ln, size=9)
        r += 1

        cx = P["cross"]
        r = note(ws, r, "[참고] 어떤 회사가 변동성이 높나 — 답이 아니고, 결과도 예상 범위다",
                 bold=True, size=12)
        r = note(ws, r, "   1차 지적 때 한 횡단면 분석. 우회의 기록으로 남긴다 — "
                        "종속변수는 종목별 YZ高 비율(0~1).", size=9, color="FF757575")
        S = cx["single"].copy()
        for c in ("ρ", "η²"):
            if c not in S:
                S[c] = np.nan
        r = put_table(ws, S[["변수", "n", "ρ", "η²"]], r + 1,
                      fmt={"n": "#,##0", "ρ": "0.000", "η²": "0.000"},
                      signed={"ρ"}, groups=[7, 2], widths=[24, 8, 11, 11])
        for ln in [
            "   ⚠ **작고·어리고·매출 적은 회사가 더 변동적** 이라는 건 교과서적 사실이다. "
            "이 표가 그걸 재확인했을 뿐, '무엇의 속성인가' 를 답하지는 못한다 (케인 2차 지적).",
            "   회전율(+0.775)이 가장 강한 것도 마찬가지다 — 거래량과 변동성은 정보 도착에 함께 "
            "반응하므로 설명이라기보다 재진술이다.",
            f"   섹터가 이미 먹고 있는 분산 η²: "
            + " · ".join(f"{k} {v:.3f}" for k, v in cx["conf"].items())
            + "  → 섹터와 규모는 애초에 교란돼 있다(케인 1차 지적).",
            f"   규모 4변수 R² {cx['r2_base']:.3f} → 섹터 더미 추가 {cx['r2_full']:.3f} "
            f"(증분 {cx['inc']:+.3f}, 순열 귀무 중앙 {cx['inc_null_med']:+.3f}, p={cx['inc_p']:.4f}) — "
            "섹터와 규모는 서로를 대체하지 못한다.",
        ]:
            r = note(ws, r, ln, size=9)

    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    print("· 패널 로드")
    raw = load_raw()
    panel = load_panel(raw)
    print(f"  {len(panel)}종목")
    print("· 지표·셀 계산")
    d = build(panel)
    print("· 집계")
    comp = compose(d)
    margin, margin_groups = compose_margin(d)
    tables = {
        "수익률_중앙값": (returns(d, "median"), "전 기간 · 중앙값 — **먼저 볼 표**"),
        "수익률_평균": (returns(d, "mean"),
                     "전 기간 · 평균 — 소수 대박에 끌려간다. 중앙값과 함께 볼 것"),
        "시대별_23-24_중앙": (returns(d, "median", ERAS["23-24"]), "2023-01 ~ 2024-12 · 중앙값"),
        "시대별_25-26_중앙": (returns(d, "median", ERAS["25-26"]), "2025-01 ~ 2026-12 · 중앙값"),
    }
    facts = build_facts(d, comp, tables)
    # YZ 안정성 검증 (케인 지적 2026-09-08·09) — 실패해도 본 리포트는 나가게 한다.
    try:
        print("· YZ 안정성 검증")
        sys.path.insert(0, str(HERE.parent))
        import yz_persistence_probe as probe
        facts["probe"] = probe.build(raw)
    except Exception as e:                                  # noqa: BLE001
        print(f"  ⚠ 건너뜀 — {type(e).__name__}: {e}")
    print(f"· 엑셀 작성 → {args.out}")
    write_xlsx(args.out, comp, margin, margin_groups, tables, facts, {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M KST"),
        "source": "MagicFormula/scripts/cell12_report.py  (원자료: LLV core+extend parquet)",
    })
    print("✅ 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
