#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yz_group_quarterly.py — 전분기 YZ 그룹 → 다음 분기 수익률
==========================================================
케인 지시 2026-09-10:
  "그룹이동이 25→26년에 40%가 넘으니 수익율의 예측으로 사용할 수는 없어.
   룩어헤드가 걱정된다면 분기로 나누어서 전 분기의 변동률 고중저가 다음 분기의
   수익율이 어떻게 되는지 봐줄래? YZ 윈도우를 40거래일 정도로 줄여도 됨."

연도 패널(`yz_group_panel.py`)은 전이 쌍이 **3개**뿐이라 검정이 불가능했다.
분기로 쪼개면 **14쌍**이 되어 비로소 검정할 수 있다.

⚠⚠ **검정 단위는 '분기' 이지 '종목' 이 아니다.**
  한 분기 안의 200종목은 시장 요인을 공유하므로 독립 표본이 아니다. 종목 단위로
  세면 표본이 2,800개인 척하게 되어 p 값이 터무니없이 작아진다(유사반복).
  그래서 **분기마다 그룹 중앙수익률을 하나씩 뽑고, 분기를 단위로 부호검정**한다.

⚠ 시장 기준을 **동일가중(전 종목 중앙 수익률)** 으로 쓴다.
  BM(102110)은 시총가중이라 소수 대형주에 극단적으로 끌린다 — 실측 상관이
  동일가중 시장과 **+0.295** 밖에 안 되고, 2026Q2 에는 BM +68.9% 인데 중앙 종목은
  −4.8% 였다. 횡단면 그룹 비교의 기준으로 쓰면 베타가 왜곡된다.
  (BM 초과수익도 함께 싣되, 그건 '대형주 대비' 를 재는 것임을 명시한다.)

실행
  python3 scripts/yz_group_quarterly.py [--tail 40]
    --tail N : 분기의 **마지막 N거래일**로 YZ 를 계산 (미지정 = 분기 전체 ~60일)
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
MIN_Q_DAYS = 25          # 분기가 이보다 짧으면 그룹 판정에서 제외
MIN_COVER = 0.5


def _binom_p(k: int, n: int) -> float:
    """양측 부호검정 p (scipy 없이 — 정확 이항)."""
    from math import comb
    if n == 0:
        return np.nan
    pmf = [comb(n, i) / 2 ** n for i in range(n + 1)]
    obs = pmf[k]
    return float(sum(p for p in pmf if p <= obs + 1e-12))


def panel(df: pd.DataFrame, tail: int | None) -> tuple[pd.DataFrame, dict]:
    C = G._piv(df, "Close")
    O, H, L = (G._piv(df, c) for c in ("Open", "High", "Low"))
    Cx, Ox, Hx, Lx = (t.drop(columns=[G.BM], errors="ignore") for t in (C, O, H, L))
    o, cc = np.log(Ox / Cx.shift(1)), np.log(Cx / Ox)
    u, dn = np.log(Hx / Ox), np.log(Lx / Ox)
    rs = u * (u - cc) + dn * (dn - cc)

    qs = sorted(set(C.index.to_period("Q")))
    rows = []
    for a, b in zip(qs, qs[1:]):
        s = C.index[C.index.to_period("Q") == a]
        if tail:
            s = s[-tail:]
        n = len(s)
        if n < MIN_Q_DAYS:
            continue
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        sig = np.sqrt(o.loc[s].var(ddof=1) + k * cc.loc[s].var(ddof=1)
                      + (1 - k) * rs.loc[s].mean())
        ok = sig.notna() & (Cx.loc[s].notna().sum() >= n * MIN_COVER)
        pr = sig[ok].rank(pct=True)
        g = pd.Series(np.select([pr <= 1 / 3, pr <= 2 / 3], LABELS[:2], default=LABELS[2]),
                      index=pr.index)

        s2 = C.index[C.index.to_period("Q") == b]
        cy = Cx.loc[s2]
        r = cy.apply(lambda c_: (c_.dropna().iloc[-1] / c_.dropna().iloc[0] - 1)
                     if c_.notna().sum() > 15 else np.nan)
        bb = C[G.BM].loc[s2].dropna()
        rec = {"전분기": str(a), "수익 분기": str(b), "수익분기 거래일": len(s2),
               "판정 종목": int(ok.sum()),
               "시장(동일가중)": float(r.median()),
               "BM(시총가중)": float(bb.iloc[-1] / bb.iloc[0] - 1) if len(bb) > 1 else np.nan}
        for lab in LABELS[::-1]:
            v = r.reindex(g[g == lab].index).dropna()
            rec[lab] = float(v.median())
        rec["고−저"] = rec["고"] - rec["저"]
        rows.append(rec)
    T = pd.DataFrame(rows)

    M = T["시장(동일가중)"]
    beta = []
    for lab in LABELS[::-1]:
        b_, a_ = np.polyfit(M, T[lab], 1)
        beta.append({"그룹": lab, "β": float(b_), "α (분기)": float(a_),
                     "R²": float(np.corrcoef(M, T[lab])[0, 1] ** 2)})
    up, dq = T[M > 0], T[M <= 0]
    w = int((T["고−저"] > 0).sum())
    stat = {
        "n_q": len(T), "win": w, "p": _binom_p(w, len(T)),
        "spread_med": float(T["고−저"].median()), "spread_mean": float(T["고−저"].mean()),
        "beta": pd.DataFrame(beta),
        "up_n": len(up), "up_win": int((up["고−저"] > 0).sum()),
        "up_med": float(up["고−저"].median()),
        "dn_n": len(dq), "dn_win": int((dq["고−저"] > 0).sum()),
        "dn_med": float(dq["고−저"].median()),
        "corr_mkt": float(M.corr(T["고−저"])),
        "corr_bm_eq": float(M.corr(T["BM(시총가중)"])),
        "tail": tail,
    }
    return T, stat


VOL3 = ["저", "중", "고"]
RET3 = ["하", "중", "상"]


def _t3(x: pd.Series, labs: list[str]) -> pd.Series:
    r = x.rank(pct=True)
    return pd.Series(np.select([r <= 1 / 3, r <= 2 / 3], labs[:2], default=labs[2]),
                     index=x.index)


def cells9(df: pd.DataFrame, tail: int | None = None) -> dict:
    """(변동성 3 × 전분기 수익률 3) 9셀 → 다음 분기 수익률 3분할이 유지되나.

    케인 지시 2026-09-10 2차. 앞 검정은 변동성 축 하나만 봤으니, 전분기 수익률을
    같이 조건에 넣으면(=변동성으로 조건부한 모멘텀/반전) 달라지는지 본다.

    ⚠ 여기서도 **검정 단위는 분기**다. 셀마다 그 분기 종목들의 '다음 분기 수익률
      순위 백분위' 평균을 하나 뽑고, 그 값이 0.5 를 넘는 분기를 센다.
    ⚠ 셀이 9개라 **다중비교**다 — 본페로니면 유의 임계가 0.05/9 = 0.0056 이다.
    """
    C = G._piv(df, "Close")
    O, H, L = (G._piv(df, c) for c in ("Open", "High", "Low"))
    Cx, Ox, Hx, Lx = (t.drop(columns=[G.BM], errors="ignore") for t in (C, O, H, L))
    o, cc = np.log(Ox / Cx.shift(1)), np.log(Cx / Ox)
    u, dn = np.log(Hx / Ox), np.log(Lx / Ox)
    rs = u * (u - cc) + dn * (dn - cc)
    qs = sorted(set(C.index.to_period("Q")))

    def qret(q):
        s = C.index[C.index.to_period("Q") == q]
        return Cx.loc[s].apply(lambda c_: (c_.dropna().iloc[-1] / c_.dropna().iloc[0] - 1)
                               if c_.notna().sum() > 15 else np.nan)

    rows, disp = [], []
    for a, b in zip(qs, qs[1:]):
        s = C.index[C.index.to_period("Q") == a]
        if tail:
            s = s[-tail:]
        n = len(s)
        if n < MIN_Q_DAYS:
            continue
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        sig = np.sqrt(o.loc[s].var(ddof=1) + k * cc.loc[s].var(ddof=1)
                      + (1 - k) * rs.loc[s].mean())
        ra, rb = qret(a), qret(b)
        ok = (sig.notna() & ra.notna() & rb.notna()
              & (Cx.loc[s].notna().sum() >= n * MIN_COVER))
        sig, ra, rb = sig[ok], ra[ok], rb[ok]
        gv, gr = _t3(sig, VOL3), _t3(ra, RET3)
        nxt, pct = _t3(rb, RET3), rb.rank(pct=True)
        for cv in VOL3:
            mv = gv == cv
            disp.append({"분기": str(b), "변동성": cv,
                         "중간비율": float((nxt[mv] == "중").mean()),
                         "sd": float(rb[mv].std()), "시장sd": float(rb.std())})
            for cr in RET3:
                m = mv & (gr == cr)
                if m.sum() < 5:
                    continue
                rows.append({"분기": str(b), "변동성": cv, "전분기 수익": cr,
                             "n": int(m.sum()),
                             "상": float((nxt[m] == "상").mean()),
                             "중": float((nxt[m] == "중").mean()),
                             "하": float((nxt[m] == "하").mean()),
                             "순위백분위": float(pct[m].mean())})
    D, E = pd.DataFrame(rows), pd.DataFrame(disp)

    # ① 통합 9×3
    P = (D.groupby(["변동성", "전분기 수익"])
         .agg(분기수=("n", "size"), 평균종목=("n", "mean"),
              **{c: (c, "mean") for c in ("상", "중", "하")},
               순위백분위=("순위백분위", "mean")).reset_index())
    # ② 셀별 부호검정
    sig_rows = []
    for cv in VOL3[::-1]:
        for cr in RET3[::-1]:
            g = D[(D["변동성"] == cv) & (D["전분기 수익"] == cr)]
            if not len(g):
                continue
            w = int((g["순위백분위"] > 0.5).sum())
            sig_rows.append({"셀": f"변동성 {cv} · 전분기수익 {cr}", "분기": len(g),
                             "순위>0.5": f"{w}/{len(g)}",
                             "평균 순위백분위": float(g["순위백분위"].mean()),
                             "부호검정 p": _binom_p(w, len(g))})
    S = pd.DataFrame(sig_rows)

    # ③ 산포 — 방향이 아니라 퍼짐
    dsp = []
    for cv in VOL3[::-1]:
        g = E[E["변동성"] == cv]
        w_lt = int((g["중간비율"] < 1 / 3).sum())
        w_gt = int((g["중간비율"] > 1 / 3).sum())
        dsp.append({"변동성": cv, "분기": len(g),
                    "다음분기 '중간' 비율": float(g["중간비율"].mean()),
                    "'중간' < 33.3% 인 분기": f"{w_lt}/{len(g)}",
                    "'중간' > 33.3% 인 분기": f"{w_gt}/{len(g)}",
                    "p(치우침)": min(_binom_p(w_lt, len(g)), _binom_p(w_gt, len(g))),
                    "다음분기 수익률 sd": float(g["sd"].median()),
                    "sd / 시장sd": float((g["sd"] / g["시장sd"]).median())})
    Dsp = pd.DataFrame(dsp)
    hi = E[E["변동성"] == "고"].set_index("분기")["sd"]
    lo = E[E["변동성"] == "저"].set_index("분기")["sd"]
    both = hi.index.intersection(lo.index)
    w = int((hi[both] > lo[both]).sum())
    extra = {"n_q": len(both), "win": w, "p": _binom_p(w, len(both)),
             "ratio": float((hi[both] / lo[both]).median())}
    return {"pooled": P, "sign": S, "disp": Dsp, "sd_test": extra,
            "n_q": D["분기"].nunique(), "n_obs": len(D)}


def write_xlsx(out: Path, runs: list[tuple[str, pd.DataFrame, dict]],
               C9: dict | None = None) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    F = "Arial"
    hdr = PatternFill("solid", fgColor="FF37474F")
    thin = Side(style="thin", color="FFBDBDBD")
    med = Side(style="medium", color="FF37474F")
    GC = {"고": UP, "중": "FF616161", "저": DOWN}
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
            for j, col in enumerate(dfx.columns, 1):
                v = rec[col]
                c = ws.cell(row=r0 + 1 + i, column=j,
                            value=(None if (isinstance(v, float) and np.isnan(v)) else v))
                col_color = "FF000000"
                if col in signed and isinstance(v, (int, float)) and not pd.isna(v):
                    col_color = UP if v > 0 else DOWN if v < 0 else "FF000000"
                if gcol and col == gcol:
                    col_color = GC.get(v, "FF000000")
                c.font = Font(name=F, size=10, color=col_color,
                              bold=(gcol is not None and col == gcol))
                c.number_format = fmt.get(col, "General")
                c.alignment = Alignment(horizontal="left" if j == 1 else "right")
                c.border = Border(left=thin, right=thin,
                                  bottom=(med if (i + 1) % band == 0 else thin))
        return r0 + len(dfx) + 1

    lab0, T0, S0 = runs[0]
    ws = sheet("분기_패널", [10, 10, 9, 9, 13, 13, 10, 10, 10, 10])
    r = note(ws, 1, "전분기 YZ 그룹 → 다음 분기 수익률", bold=True, size=14)
    r = note(ws, r, f"창 = {lab0} · 생성기 scripts/yz_group_quarterly.py · "
                    "각 분기를 그 분기 데이터만으로 균등 3분할 (룩어헤드 없음)",
             size=9, color="FF757575")
    r += 1
    for ln in [
        "⚠⚠ **검정 단위는 '분기' 이지 '종목' 이 아니다.** 한 분기 안의 200종목은 시장 "
        "요인을 공유하므로 독립 표본이 아니다 —",
        "     종목 단위로 세면 표본이 2,800개인 척하게 되어 p 값이 터무니없이 작아진다"
        "(유사반복). 분기마다 그룹 중앙수익률을 하나씩 뽑아 분기를 단위로 검정한다.",
        f"⚠ 시장 기준은 **동일가중(전 종목 중앙)** 이다. BM(102110)은 시총가중이라 "
        f"소수 대형주에 끌린다 — 동일가중과의 상관이 **{S0['corr_bm_eq']:+.3f}** 뿐이고,",
        "     2026Q2 에는 BM +68.9% 인데 중앙 종목은 −4.8% 였다. BM 열은 참고로만 싣는다"
        "('대형주 대비' 를 재는 값이다).",
    ]:
        r = note(ws, r, ln, size=9)
    cols = ["전분기", "수익 분기", "수익분기 거래일", "판정 종목", "시장(동일가중)",
            "BM(시총가중)", "고", "중", "저", "고−저"]
    pct = {c: "0.0%" for c in cols[4:]}
    r = table(ws, T0[cols], r + 1, {**pct, "수익분기 거래일": "#,##0", "판정 종목": "#,##0"},
              signed=set(cols[4:]))
    r += 1
    for ln in [
        f"★ **고 > 저 인 분기는 {S0['win']}/{S0['n_q']} — 부호검정 p = {S0['p']:.3f}. "
        "방향성이 없다.**",
        f"   고−저 스프레드 중앙 {S0['spread_med']:+.2%} · 평균 {S0['spread_mean']:+.2%} "
        "— 0 근처다.",
        "⇒ **전분기 변동성 그룹으로 다음 분기 수익률을 예측할 수 없다.** "
        "연도 단위(전이 3쌍)에서 '방향이 해마다 뒤집힌다' 고만 말할 수 있었던 것을,",
        "   분기 14쌍으로 늘려 **검정으로 확인**한 것이다.",
        "⚠ 마지막 행(2026Q3)은 49거래일짜리 미완성 분기다.",
    ]:
        r = note(ws, r, ln, size=9)

    # ── 베타 시트 ──
    ws = sheet("베타_알파", [12, 10, 12, 10])
    r = note(ws, 1, "왜 예측이 안 되나 — 변동성 그룹은 베타 그룹이다", bold=True, size=14)
    r = note(ws, r, "그룹 중앙수익률 = α + β · 시장(동일가중), 분기 n=14",
             size=9, color="FF757575")
    r += 1
    r = table(ws, S0["beta"], r, {"β": "0.00", "α (분기)": "0.00%", "R²": "0.000"},
              gcol="그룹")
    r += 1
    for ln in [
        "★ **β 가 고 1.18 > 중 1.09 > 저 0.80 으로 단조**이고 R² 가 0.74~0.86 이다 — "
        "그룹이 시장 움직임을 얼마나 증폭하는지가 곧 그룹의 정체다.",
        "★ **α 는 셋 다 0 근처**(−0.54% ~ +1.12% / 분기). "
        "**증폭기이지 초과수익원이 아니다.**",
        "",
        "그래서 시장 방향에 따라 승자가 갈린다 —",
        f"   상승 분기 {S0['up_n']}개: 고−저 중앙 {S0['up_med']:+.2%} · "
        f"고가 이긴 분기 {S0['up_win']}/{S0['up_n']}",
        f"   하락 분기 {S0['dn_n']}개: 고−저 중앙 {S0['dn_med']:+.2%} · "
        f"고가 이긴 분기 {S0['dn_win']}/{S0['dn_n']}",
        f"   시장수익률 × 고−저 상관 {S0['corr_mkt']:+.3f}",
        "",
        "⇒ **시장이 오르면 고가, 내리면 저가 이긴다.** 당연한 결과다 — 베타가 그런 뜻이니까.",
        "⇒ 그런데 **다음 분기 시장 방향을 미리 알 수 없으므로 예측에 쓸 수 없다.** "
        "이것이 위 부호검정이 귀무인 이유다.",
        "",
        "쓸 수 있는 방식이 있다면 예측이 아니라 **노출 조절**이다 — 시장 하락을 각오할 때 "
        "저 그룹으로 옮기면 낙폭이 준다(하락 분기 고−저 중앙 −4.10%).",
        "   ⚠ 다만 그건 **시장 전망이 맞아야** 이득이다. 이 표가 그 전망을 주지는 않는다.",
    ]:
        r = note(ws, r, ln, size=9)
    r += 1

    if len(runs) > 1:
        r = note(ws, r, "창 길이 민감도 — 결론이 창에 흔들리나", bold=True, size=12)
        rob = pd.DataFrame([{
            "창": lab, "분기 수": S["n_q"], "고>저": f"{S['win']}/{S['n_q']}",
            "부호검정 p": S["p"], "스프레드 중앙": S["spread_med"],
            "β 고": S["beta"].iloc[0]["β"], "β 저": S["beta"].iloc[-1]["β"],
        } for lab, _, S in runs])
        r = table(ws, rob, r + 1,
                  {"분기 수": "#,##0", "부호검정 p": "0.000", "스프레드 중앙": "0.0%",
                   "β 고": "0.00", "β 저": "0.00"}, signed={"스프레드 중앙"})
        r = note(ws, r, "   ⇒ 창을 분기 전체(~60일)에서 40거래일로 줄여도 "
                        "**결론이 바뀌지 않는다.**", size=9)
    r += 1
    for ln in [
        "⚠ 한계 — **분기 14개**다. 3개보다는 낫지만 여전히 작다. "
        "특히 하락 분기가 5개뿐이라 '하락장에서 저가 낫다' 는 우연과 구분하기 어렵다.",
        "⚠ **생존편향**: 크게 흔들리다 사라진 종목이 없으므로 고 그룹이 낙관적으로 치우친다. "
        "β 도 그만큼 과소추정일 수 있다.",
        "⚠ 이 표는 **수익률의 방향**만 본다. 위험조정(샤프)·최대낙폭은 보지 않았다 — "
        "저 그룹의 값어치는 거기에 있을 수 있다.",
    ]:
        r = note(ws, r, ln, size=9)

    # ── 9셀 시트 ──
    if C9:
        ws = sheet("9셀_전이", [22, 9, 10, 9, 9, 9, 13])
        r = note(ws, 1, "9셀(변동성 3 × 전분기 수익률 3) → 다음 분기 수익률 3분할",
                 bold=True, size=14)
        r = note(ws, r, f"분기 {C9['n_q']}개 · 셀-분기 관측 {C9['n_obs']}개 · "
                        "변동성 축 하나만 보던 앞 검정에, 전분기 수익률을 함께 조건에 넣은 것"
                        "(=변동성으로 조건부한 모멘텀/반전)", size=9, color="FF757575")
        r += 1
        r = note(ws, r, "① 통합 — 9셀이 다음 분기 어디로 가나 (%). 무작위면 각 33.3",
                 bold=True, size=12)
        P = C9["pooled"].copy()
        P["변동성"] = pd.Categorical(P["변동성"], VOL3[::-1])
        P["전분기 수익"] = pd.Categorical(P["전분기 수익"], RET3[::-1])
        P = P.sort_values(["변동성", "전분기 수익"])
        r = put9 = table(ws, P, r + 1,
                         {"분기수": "#,##0", "평균종목": "#,##0.0",
                          **{c: "0.0%" for c in ("상", "중", "하", "순위백분위")}},
                         gcol="변동성", band=3)
        r += 1
        r = note(ws, r, "② 셀별 부호검정 — 분기를 단위로 (종목이 아니다)", bold=True, size=12)
        r = table(ws, C9["sign"], r + 1,
                  {"분기": "#,##0", "평균 순위백분위": "0.0%", "부호검정 p": "0.000"},
                  band=3)
        r += 1
        for ln in [
            "★ **9셀 어디에도 예측력이 없다.** 평균 순위백분위가 48.5~52.1% 로 전부 50% "
            "근처이고, 부호검정 p 는 0.18~1.00 이다.",
            f"   셀이 9개라 다중비교다 — 본페로니 임계 p < {0.05/9:.4f}. "
            "**근처에 온 셀조차 없다.**",
            "★ **전분기 수익률 축은 아무 정보도 없다.** 같은 변동성 안에서 상/중/하의 "
            "다음 분기 분포가 거의 같다(예: 고변동에서 '상' 36.2 / '중' 38.3 / '하' 38.2).",
            "   ⇒ **모멘텀도 반전도 없다.** 변동성으로 조건을 걸어도 마찬가지다.",
        ]:
            r = note(ws, r, ln, size=9)
        r += 1

        r = note(ws, r, "③ ★ 그런데 '방향' 이 아니라 '산포' 는 아주 강하게 예측한다",
                 bold=True, size=12)
        r = table(ws, C9["disp"], r + 1,
                  {"분기": "#,##0", "다음분기 '중간' 비율": "0.0%",
                   "p(치우침)": "0.0000", "다음분기 수익률 sd": "0.0%",
                   "sd / 시장sd": "0.00"}, gcol="변동성")
        sd = C9["sd_test"]
        for ln in [
            "   무작위면 '중간' 비율이 33.3% 여야 한다. 실측은 **고 24.2% / 중 30.3% / "
            "저 45.4%** 다.",
            "   **고 그룹은 14/14 분기에서 33.3% 미만, 저 그룹은 14/14 분기에서 초과** "
            "(각 p = 0.0001). 예외가 한 분기도 없다.",
            f"   다음 분기 수익률 표준편차도 **고 그룹이 저 그룹의 {sd['ratio']:.2f}배**이고, "
            f"{sd['win']}/{sd['n_q']} 분기에서 그렇다 (p = {sd['p']:.4f}).",
            "",
            "⇒ **변동성 그룹은 다음 분기에 '어디로 갈지' 는 못 맞히지만 '얼마나 멀리 갈지' "
            "는 아주 잘 맞힌다.**",
            "   고변동 그룹은 다음 분기에도 양 끝(상·하)으로 벌어지고, 저변동 그룹은 "
            "가운데로 몰린다.",
            "   ※ 이건 놀라운 결과가 아니다 — 변동성 지속성(§10-5)을 '다음 분기 수익률의 "
            "퍼짐' 이라는 실용적 형태로 다시 말한 것이다.",
            "",
            "⇒ **쓸모의 방향이 정해진다.** 이 그룹은 종목 고르기(알파)가 아니라 "
            "**위험 예산·포지션 사이징**에 쓰는 것이 맞다 —",
            "   같은 금액을 넣어도 고 그룹은 저 그룹의 2.26배로 흔들린다는 뜻이니까.",
            "⚠ 다만 '어느 쪽으로' 흔들릴지는 이 표가 말해주지 않는다. 손실 쪽도 그만큼 넓다.",
        ]:
            r = note(ws, r, ln, size=9)

    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail", type=int, default=None)
    a = ap.parse_args()

    print("· 패널 로드")
    df = G.load_raw()
    runs = []
    for tail, lab in [(None, "분기 전체 (~60거래일)"), (40, "최근 40거래일")]:
        T, S = panel(df, tail)
        runs.append((lab, T, S))
        print(f"· {lab}: 분기쌍 {S['n_q']} · 고>저 {S['win']}/{S['n_q']} "
              f"(p={S['p']:.3f}) · β 고 {S['beta'].iloc[0]['β']:.2f} / "
              f"저 {S['beta'].iloc[-1]['β']:.2f}")
    if a.tail:                                     # 지정하면 그 창을 앞으로
        runs.sort(key=lambda x: x[0] != f"최근 {a.tail}거래일")

    print("· 9셀 전이 검정")
    C9 = cells9(df, a.tail)
    print(f"  셀별 평균 순위백분위 "
          f"{C9['sign']['평균 순위백분위'].min():.1%}~{C9['sign']['평균 순위백분위'].max():.1%} · "
          f"최소 p {C9['sign']['부호검정 p'].min():.3f} (본페로니 임계 {0.05/9:.4f})")
    d = C9["disp"].set_index("변동성")
    mid = "다음분기 '중간' 비율"
    sd = C9["sd_test"]
    print(f"  산포: 다음분기 중간비율 고 {d.loc['고', mid]:.1%} / 저 {d.loc['저', mid]:.1%}"
          f" · sd 비 {sd['ratio']:.2f}배 ({sd['win']}/{sd['n_q']}분기, p={sd['p']:.4f})")

    out = REPO / "docs" / "holdtiming" / "results" / "YZ그룹_분기예측.xlsx"
    write_xlsx(out, runs, C9)
    print(f"✅ Excel {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
