# -*- coding: utf-8 -*-
"""regime_xlsx.py — 통일장 연구 산출물(CSV)을 엑셀로 (케인 2026-11-09 21:34 "CSV 한글이 깨져서 알아볼 수 없어, 엑셀로").

왜    CSV 를 BOM 없는 UTF-8 로 저장해 엑셀이 한글을 깨뜨렸다. 엑셀 파일을 따로 만들고, CSV 도 utf-8-sig(BOM)로 다시 쓴다.
표시  케인 전역 규칙 — 상승·+ 빨강 #EF5350 / 하락·− 파랑 #1976D2 · 숫자 오른쪽 정렬 ·
      원 = #,##0원 (소수점 없음) · 지수 = 0.00 · 변화율 = 0.0% · 차이 = 0.0%p
산출  out/regime/통일장_1단계_늘리기신호_YYYYMMDD.xlsx  (regime_expand.py 결과)
      out/regime/통일장_2d_노출통제_YYYYMMDD.xlsx       (regime_exposure.py 결과)
사용  python3 regime_xlsx.py [expand|exposure]   (인자 없으면 둘 다) — 두 스크립트 끝에서도 자동 호출
"""
import sys
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.formatting.rule import CellIsRule
from openpyxl.utils import get_column_letter

HERE = Path(__file__).resolve().parent; OUT = HERE / "out" / "regime"
FONT, UP, DN = "Arial", "EF5350", "1976D2"
NF = {"pct": "0.0%", "rate": "0.0%", "pp": '+0.0"%p";-0.0"%p";0.0"%p"', "won": '#,##0"원"', "idx": "0.00", "int": "#,##0", "num2": "0.00", "date": "yyyy-mm-dd"}
COLORED = {"pct", "pp", "won"}           # 부호에 따라 빨강/파랑
UNIV = {"swing": "메가갤럭시", "day": "유니버스"}

def rd(name): return pd.read_csv(OUT / name, encoding="utf-8-sig")

def conv(v, kind):
    if isinstance(v, (bool, np.bool_)) or (isinstance(v, str) and v in ("True", "False")):
        return "✓" if v in (True, "True") else "✗"
    if v is None or (isinstance(v, float) and np.isnan(v)): return None
    if kind in ("pct", "rate"): return float(v) / 100
    if kind == "date": return pd.Timestamp(v).to_pydatetime()
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    return v

def width(v):
    s = str(v) if v is not None else ""
    return sum(2 if ord(ch) > 0x1100 else 1 for ch in s)

def style_header(c):
    c.font = Font(name=FONT, bold=True, size=10); c.fill = PatternFill("solid", fgColor="E8EAED")
    c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    c.border = Border(bottom=Side(style="thin", color="999999"))

def color_rules(ws, rng):
    ws.conditional_formatting.add(rng, CellIsRule(operator="greaterThan", formula=["0"], font=Font(name=FONT, color=UP)))
    ws.conditional_formatting.add(rng, CellIsRule(operator="lessThan", formula=["0"], font=Font(name=FONT, color=DN)))

def write_table(wb, name, df, fmt, note=None):
    """열마다 서식 — fmt: {열 이름: pct|rate|pp|won|idx|int|num2|date} (없으면 글자)."""
    ws = wb.create_sheet(name); r0 = 1
    if note:
        ws.cell(1, 1, note).font = Font(name=FONT, italic=True, size=9, color="666666"); r0 = 2
    cols = list(df.columns); wmax = [width(h) for h in cols]
    for j, h in enumerate(cols, 1): style_header(ws.cell(r0, j, h))
    for i, row in enumerate(df.itertuples(index=False), r0 + 1):
        for j, v in enumerate(row, 1):
            kind = fmt.get(cols[j - 1]); x = conv(v, kind); c = ws.cell(i, j, x); c.font = Font(name=FONT, size=10)
            if kind and isinstance(x, (int, float)) and not isinstance(x, bool):
                c.number_format = NF[kind]; c.alignment = Alignment(horizontal="right")
            elif kind == "date" and x is not None: c.number_format = NF["date"]; c.alignment = Alignment(horizontal="center")
            elif x in ("✓", "✗"): c.alignment = Alignment(horizontal="center")
            wmax[j - 1] = max(wmax[j - 1], min(width(x if not isinstance(x, float) else f"{x:,.3f}"), 40))
    last = r0 + len(df)
    for j, h in enumerate(cols, 1):
        L = get_column_letter(j); ws.column_dimensions[L].width = max(8, wmax[j - 1] + 2)
        if fmt.get(h) in COLORED and len(df): color_rules(ws, f"{L}{r0 + 1}:{L}{last}")
    ws.freeze_panes = ws.cell(r0 + 1, 1); ws.auto_filter.ref = f"A{r0}:{get_column_letter(len(cols))}{last}"
    return ws

def write_rows(wb, name, df, rowfmt, note=None):
    """행마다 서식 (지표가 행, 후보가 열인 가로 비교표) — 첫 열 = 지표 이름."""
    ws = wb.create_sheet(name); r0 = 1
    if note:
        ws.cell(1, 1, note).font = Font(name=FONT, italic=True, size=9, color="666666"); r0 = 2
    cols = ["지표"] + list(df.columns)
    for j, h in enumerate(cols, 1): style_header(ws.cell(r0, j, h))
    for i, (lab, row) in enumerate(df.iterrows(), r0 + 1):
        kind = rowfmt.get(lab); ws.cell(i, 1, lab).font = Font(name=FONT, size=10, bold=True)
        for j, v in enumerate(row, 2):
            x = conv(v, kind); c = ws.cell(i, j, x); c.font = Font(name=FONT, size=10)
            if kind and isinstance(x, (int, float)) and not isinstance(x, bool): c.number_format = NF[kind]; c.alignment = Alignment(horizontal="right")
            else: c.alignment = Alignment(horizontal="center")
        if kind in COLORED: color_rules(ws, f"B{i}:{get_column_letter(len(cols))}{i}")
    ws.column_dimensions["A"].width = max(width(x) for x in cols[:1] + list(df.index)) + 2
    for j in range(2, len(cols) + 1): ws.column_dimensions[get_column_letter(j)].width = max(14, width(cols[j - 1]) + 2)
    ws.freeze_panes = ws.cell(r0 + 1, 2)
    return ws

def write_notes(wb, name, lines):
    ws = wb.active if wb.active.title == "Sheet" else wb.create_sheet(name); ws.title = name
    for i, (a, b) in enumerate(lines, 1):
        ca = ws.cell(i, 1, a); cb = ws.cell(i, 2, b)
        ca.font = Font(name=FONT, size=10, bold=bool(a)); cb.font = Font(name=FONT, size=10)
        cb.alignment = Alignment(wrap_text=True, vertical="top"); ca.alignment = Alignment(vertical="top")
    ws.column_dimensions["A"].width = 18; ws.column_dimensions["B"].width = 120
    return ws

def rebom(names):
    """CSV 를 utf-8-sig(BOM) 로 다시 저장 — 엑셀에서 열어도 한글이 안 깨지게."""
    for n in names:
        f = OUT / n
        if f.exists(): rd(n).to_csv(f, index=False, encoding="utf-8-sig")

# ── ① 늘리기 신호 (regime_expand.py) ──────────────────────────────────────
def backdrop():
    """참고: 신호에서 마지막 조건 하나를 뺀 '바탕 조건'의 모든 날 (결과를 본 뒤 확인한 것 — 미리 정한 기준 아님)."""
    import regime_expand as RX
    k, o, u = RX.load_base(); cal = k.index; kv, ov, uv = k.to_numpy(float), o.to_numpy(float), u.to_numpy(float)
    ib = int(np.searchsorted(cal, RX.B, side="right")) - 1; w = (cal >= RX.A) & (cal <= RX.B)
    elig = w & (np.arange(len(cal)) + RX.FWD <= ib)
    base = pd.DataFrame([dict(i=i, **RX.measure(i, kv, ov, uv)) for i in np.flatnonzero(elig)]).set_index("i")
    up = k > k.rolling(200).mean(); pull = ((k / k.rolling(20, min_periods=1).max() - 1) <= -0.05).astype(int).rolling(20, min_periods=1).max() > 0
    after = RX.gap_shock(k, o).astype(int).rolling(61, min_periods=1).max() > 0
    H = rd("expand_heat.csv").set_index("date")["heat_recon"]; H.index = pd.to_datetime(H.index); hot = H.reindex(cal).rolling(5, min_periods=1).max() >= 80
    C = {"모든 날": pd.Series(True, index=cal), "(가)의 바탕: 200일선 위 & 20일 안 −5% 눌림": up & pull,
         "(나1)의 바탕: 갭 쇼크 뒤 60일 안": after, "(열)의 바탕: 최근 5일 열량 ≥ 80": hot}
    rows = []
    for nm, c in C.items():
        b = base.loc[[i for i in np.flatnonzero(c.to_numpy(bool)) if i in base.index]]
        rows.append({"조건": nm, "날 수": len(b), "20일 수익 평균": b.수익20.mean(), "20일 수익 중앙": b.수익20.median(),
                     "20일 플러스 비율": (b.수익20 > 0).mean() * 100, "헛 진입 비율": b.헛진입.mean() * 100, "늦음 중앙": b.늦음.median()})
    return pd.DataFrame(rows)

def export_expand(day):
    rebom(["expand_summary.csv", "expand_events.csv", "expand_years.csv"])
    h = OUT / "expand_heat.csv"
    if h.exists():
        x = pd.read_csv(h, encoding="utf-8-sig", index_col=0); x.index.name = "date"; x.to_csv(h, encoding="utf-8-sig")
    wb = Workbook()
    write_notes(wb, "설명", [
        ("문서", "통일장 ① 단계 — 시장 층 '늘리기' 신호와 발열계 열량을 사건 기준으로 잰 결과 (regime_expand.py, 2026-11-09)"),
        ("기준선", "코스피 200. 신호 다음 날 시가 진입, h거래일 뒤 종가까지 수익 (h = 5·10·20)"),
        ("기간", "학습 2016-07-01 ~ 2026-05-29. 신호 뒤 20거래일이 학습 끝을 넘는 사건은 뺐다. 보류 구간(2026-06~) 미사용"),
        ("(가) 시장 눌림목", "코스피 200 > 200일선 AND 최근 20거래일 안에 '20일 최고 종가 대비 −5% 이하' AND 시장 브레드스 판정(±0.05)이 하락→상승 전환"),
        ("(나1) 투매 뒤 복귀", "갭 쇼크일(하루 −3% 이하 & 갭 몫 ≥ 40%) 뒤 60거래일 안에, 쇼크 뒤 최저 종가 대비 +10% 이상인 첫날"),
        ("(나2) 반등 룰 동시 다발", "같은 날 반등 룰 신호 종목 수 ≥ 4 (거래일의 1.44%) — 케인 11-09 탈락 결정"),
        ("(열) 열량 꺾임", "최근 5거래일 열량 최고 ≥ 80 AND 오늘 열량 ≤ 그 최고 − 5 AND 시장 브레드스 EMA10 > 5거래일 전"),
        ("사건", "신호가 켜진 날. 직전 사건 첫날 뒤 20거래일 안에 다시 켜지면 같은 사건"),
        ("늦음", "진입가 ÷ 직전 60거래일 최저 종가 − 1"),
        ("헛 진입", "진입 뒤 20거래일 안에 종가가 그 최저 종가 아래로 내려간 경우"),
        ("통과 기준", "① 사건 ≥ 20  ② 20일 수익 평균 > 기준, 사건 있는 연도 2/3 이상에서 그해 기준보다 높음  ③ 헛 진입 비율 ≤ 기준"),
        ("결과", "네 후보 모두 ① 탈락. (가)·(나1)·(열)은 ②·③ 통과 → 케인 결정: '표본 부족 후보'로 2-d 에 넣음. (나2) 탈락"),
        ("열량 검증", "발열계 191종목, 2015-10~ 재구성. LLV 정본 YZ20 과 상대오차 중앙 0.00%, 열량 상관 1.000 (겹침 717일). 종목은 현재 명단이라 생존 편향"),
        ("표시", "수익·차이는 + 빨강 / − 파랑. 비율은 색 없음"),
    ])
    S = rd("expand_summary.csv").set_index("신호")
    lab = {"사건": "사건 수", "신호일 수(학습)": "신호가 켜진 날 수", "수익5 평균": "5일 수익 평균", "기준5 평균": "기준 5일 평균", "수익10 평균": "10일 수익 평균",
           "기준10 평균": "기준 10일 평균", "수익20 평균": "20일 수익 평균", "기준20 평균": "기준 20일 평균", "수익20 중앙": "20일 수익 중앙", "기준20 중앙": "기준 20일 중앙",
           "20일 + 비율": "20일 플러스 비율", "기준 + 비율": "기준 플러스 비율", "늦음 중앙": "늦음 중앙", "헛진입%": "헛 진입 비율", "기준 헛진입%": "기준 헛 진입 비율",
           "유니버스20 평균": "유니버스 동일가중 20일 평균", "기준 유니버스20": "기준 유니버스 20일", "연도 + 비율": "연도 일관성(그해 기준보다 높은 해)", "연도 수": "사건 있는 연도 수",
           "①사건≥20": "① 사건 ≥ 20", "②수익": "② 수익", "③헛진입": "③ 헛 진입", "통과": "통과"}
    T = S.T.rename(index=lab)
    rf = {v: ("int" if k in ("사건", "신호일 수(학습)", "연도 수") else "rate" if k in ("20일 + 비율", "기준 + 비율", "헛진입%", "기준 헛진입%", "연도 + 비율", "늦음 중앙")
              else None if k.startswith(("①", "②", "③", "통과")) else "pct") for k, v in lab.items()}
    write_rows(wb, "요약", T, rf, note="학습 2016-07~2026-05 · 코스피 200 다음 날 시가 진입 · 기준 = 학습 구간 모든 거래일(2,410일)에 같은 방식으로 진입")
    E = rd("expand_events.csv").drop(columns=["i"]); E["헛진입"] = E["헛진입"].map({True: "예", False: "아니오", "True": "예", "False": "아니오"})
    E = E.rename(columns={"수익5": "5일 수익", "수익10": "10일 수익", "수익20": "20일 수익", "헛진입": "헛 진입", "유니버스20": "유니버스 동일가중 20일"})
    write_table(wb, "사건목록", E, {"날짜": "date", "코스피200": "idx", "5일 수익": "pct", "10일 수익": "pct", "20일 수익": "pct", "늦음": "rate", "유니버스 동일가중 20일": "pct"})
    Y = rd("expand_years.csv").rename(columns={"사건 20일": "사건 20일 수익 평균", "그해 기준 20일": "그해 기준 20일 평균", "차이": "차이"})
    Y["연도"] = Y["연도"].astype(str)
    write_table(wb, "연도별", Y, {"사건수": "int", "사건 20일 수익 평균": "pct", "그해 기준 20일 평균": "pct", "차이": "pp"})
    write_table(wb, "참고_바탕조건", backdrop(), {"날 수": "int", "20일 수익 평균": "pct", "20일 수익 중앙": "pct", "20일 플러스 비율": "rate", "헛 진입 비율": "rate", "늦음 중앙": "rate"},
                note="참고 — 결과를 본 뒤 확인한 것(미리 정한 기준 아님). 신호에서 마지막 조건 하나를 뺀 조건의 모든 날에 들어갔을 때")
    Hh = rd("expand_heat.csv").rename(columns={"date": "날짜", "heat_recon": "열량 (재구성, data_1104)", "heat_llv": "열량 (LLV 정본 YZ20 로 계산)"})
    write_table(wb, "열량", Hh, {"날짜": "date", "열량 (재구성, data_1104)": "idx", "열량 (LLV 정본 YZ20 로 계산)": "idx"},
                note="열량 = 종목별 YZ20 의 자기 직전 249일 대비 백분위의 중앙값 (50 = 평소). LLV 정본 쪽은 2023-11 부터 값이 있다")
    f = OUT / f"통일장_1단계_늘리기신호_{day}.xlsx"; wb.save(f); return f

# ── 2-d 노출 통제 (regime_exposure.py) ────────────────────────────────────
XF = {"최종 자산(원)": "won", "CAGR": "pct", "MDD": "pct", "Sharpe": "num2", "종목 투자비중": "rate", "총 투자비중": "rate", "청산수": "int",
      "구간 16.7-19": "pct", "구간 20-22": "pct", "구간 23-26.5": "pct", "무작위 CAGR 중앙": "pct", "무작위 CAGR 10%": "pct", "무작위 CAGR 90%": "pct",
      "무작위 MDD 중앙": "pct", "ΔCAGR": "pp", "ΔMDD": "pp", "구간 우위 수": "int", "무작위 이긴 비율": "rate", "무작위 ΔCAGR 중앙": "pp",
      "회수 비율": "pct", "SMA200 대비 CAGR": "pp", "기준 CAGR": "pct", "기준 MDD": "pct", "T": "int", "it": "int"}

def prep(df):
    df = df.copy()
    if "무대" in df: df["무대"] = df["무대"].map(UNIV).fillna(df["무대"])
    if "방식" in df: df["방식"] = df["방식"].fillna("")
    if "최종(억)" in df: df.insert(df.columns.get_loc("최종(억)"), "최종 자산(원)", (df["최종(억)"] * 1e8).round(0)); df = df.drop(columns="최종(억)")
    if "T" in df: df["T"] = df["T"].astype(int)
    return df

def stage_summary(J, stage):
    g = J[(J.단계 == stage) & ~J.장치.str.startswith("사후 상한")]; rows = []
    combos = [("메가갤럭시", 40), ("메가갤럭시", 60), ("유니버스", 40), ("유니버스", 60)]
    for (nm, md), h in g.groupby(["장치", "방식"], sort=False):
        r = {"장치": nm, "방식": md, "통과 조합 수": int(h.통과.astype(str).eq("True").sum())}
        for u, T in combos:
            z = h[(h.무대 == u) & (h["T"] == T)].iloc[0]; r[f"{u} T{T}"] = "✓" if str(z.통과) == "True" else "✗"
        r.update({"ΔCAGR 평균": h.ΔCAGR.mean(), "ΔMDD 평균": h.ΔMDD.mean(), "무작위 이긴 비율 평균": h["무작위 이긴 비율"].mean(), "구간 우위 수 평균": h["구간 우위 수"].mean()})
        if stage == "A": r.update({"회수 비율 평균": h["회수 비율"].mean(), "SMA200 대비 CAGR 평균": h["SMA200 대비 CAGR"].mean()})
        rows.append(r)
    return pd.DataFrame(rows)

def export_exposure(day):
    rebom([f"exposure_{a}_{u}.csv" for a in ("runs", "mc", "events") for u in ("swing", "day")] + ["exposure_summary.csv"])
    R = prep(pd.concat([rd(f"exposure_runs_{u}.csv") for u in ("swing", "day")], ignore_index=True))
    M = prep(pd.concat([rd(f"exposure_mc_{u}.csv") for u in ("swing", "day")], ignore_index=True))
    EV = prep(pd.concat([rd(f"exposure_events_{u}.csv") for u in ("swing", "day")], ignore_index=True))
    J = prep(rd("exposure_summary.csv"))
    wb = Workbook()
    write_notes(wb, "설명", [
        ("문서", "통일장 2-d — 시장 층 두 장치(줄이기·늘리기)를 스윙 포트에 붙인 결과 (regime_exposure.py, 2026-11-09)"),
        ("엔진", "stop_port.run_port — 스윙 라이브 방식: 1억 · 10슬롯 × 1,000만 원 고정 · 종목당 2슬롯 · 눌림 깊은 순 · 다음 날 시가 체결 · 실제 비용. 청산 = 버팀(손절 없음) + 시간 청산 T40 / T60"),
        ("무대", "메가갤럭시 = 4조·외인 30% 월별 / 유니버스 = LLV 217종목 상시 (50억 거래대금 필터 아님)"),
        ("기간", "학습 2016-07-01 ~ 2026-05-29. 보류 구간(2026-06~11-04) 미사용"),
        ("판정기 7개", "① K200 120일선 ② K200 200일선 ±2% ③ 실시간 zigzag 10% ④ C2 괴리형 퇴출 + 저점 +10% 복귀 ⑤ 갭 쇼크 게이트(갭 몫 ≥ 40%) · 시장 브레드스 ±0.05 · 평소 보유 ⑥ 같은 게이트 · 평소 SMA120 ⑦ 평소 C2 + 갭 쇼크·브레드스 하락이면 추가 퇴출"),
        ("줄이기 방식", "진입중단 = 하락 판정 동안 신규 매수 없음 / 슬롯절반 = 하락 동안 슬롯 상한 5 / 현금화 = 하락 판정 다음 날 보유 전량 시가×99.5% 매도 + 신규 매수 없음"),
        ("사후 상한", "zigzag 20% 사후 국면(실시간 아님)의 상승 국면에만 노출 — 넘을 수 없는 기준"),
        ("늘리기", "신호 (가) 눌림목 · (나1) 투매 뒤 복귀 · (열) 열량 꺾임 · 셋 중 하나 — 사건 뒤 20거래일 창. 수단: 슬롯확대 = 슬롯 상한 15 / ETF = 창 첫날 시가에 남는 현금 전부로 코스피 200(102110 대용) 매수, 창 끝 다음 날 매도, 종목 매수 현금이 모자라면 그만큼 팔아 바로 씀 [가정]"),
        ("반복", "후보마다 정해진 순서(라이브) 1회 + 후보 순서 무작위 30회 (seed 20261109+it, 후보끼리 같은 seed)"),
        ("판정 기준", "① CAGR ≥ 기준 ② MDD 가 기준보다 얕음 ③ 세 구간(2016.7~19 / 20~22 / 23~26.5) 중 둘 이상에서 기준보다 높음 — ①~③ 정해진 순서 1회 기준. ④ 무작위 30회 중 같은 seed 기준보다 CAGR 이 높은 비율 ≥ 2/3"),
        ("회수 비율", "(후보 CAGR − 기준) ÷ (같은 방식 사후 상한 CAGR − 기준)"),
        ("ΔMDD", "+ 는 낙폭이 얕아졌다는 뜻 (MDD 는 음수)"),
        ("주의", "갭 몫 40%·C2 +10% 는 2-c 에서 학습 구간을 보고 정한 값 — 보류 구간 확인 전에는 확정 못 함. ETF 이득은 2025-12~2026-05 상승장에 몰려 있음"),
        ("표시", "수익·차이·금액은 + 빨강 / − 파랑. 비율은 색 없음. 원은 소수점 없이"),
    ])
    write_table(wb, "단계A_요약", stage_summary(J, "A"), {"통과 조합 수": "int", "ΔCAGR 평균": "pp", "ΔMDD 평균": "pp", "무작위 이긴 비율 평균": "rate", "구간 우위 수 평균": "num2",
                                                      "회수 비율 평균": "pct", "SMA200 대비 CAGR 평균": "pp"}, note="줄이기 — 판정기 7 × 방식 3. 평균은 4개 조합(무대 2 × T40·T60)")
    write_table(wb, "단계B_요약", stage_summary(J, "B"), {"통과 조합 수": "int", "ΔCAGR 평균": "pp", "ΔMDD 평균": "pp", "무작위 이긴 비율 평균": "rate", "구간 우위 수 평균": "num2"},
                note="늘리기 (장치 없는 포트 위) — 신호 4 × 수단 2")
    B0 = R[(R.단계 == "기준") | R.장치.str.startswith("사후 상한")]
    write_table(wb, "기준_사후상한", B0, XF, note="장치 없음(기준)과 사후 상한 — 조합별")
    write_table(wb, "판정_전체", J.sort_values(["단계", "장치", "방식", "무대", "T"], kind="stable"), XF, note="후보 × 조합 전체 — ①~④ 와 통과 (✓/✗)")
    write_table(wb, "정해진순서_1회", R, XF, note="후보 × 조합별 정해진 순서(라이브) 1회 결과 + 무작위 30회 요약")
    write_table(wb, "무작위_30회", M, XF, note="런별 CAGR·MDD — 같은 it(seed) 끼리 짝지어 비교")
    EV = EV.rename(columns={"후보 20일%": "후보 20일 수익", "기준 20일%": "기준 20일 수익"}); EV["기여(원)"] = (EV.pop("기여(백만원)") * 1e6).round(0)
    write_table(wb, "늘리기_사건기여", EV, {"T": "int", "사건일": "date", "끝": "date", "후보 20일 수익": "pct", "기준 20일 수익": "pct", "기여(%p)": "pp", "기여(원)": "won"},
                note="늘리기 사건마다 사건일 → 21거래일 뒤 자산 변화의 (후보 − 기준). 정해진 순서 1회 기준")
    f = OUT / f"통일장_2d_노출통제_{day}.xlsx"; wb.save(f); return f

# ── 2-d 2차: 줄이기 위 늘리기 + 보류 구간 (regime_exposure.py run2/report2) ───────────
XF2 = {**XF, "줄이기 단독 CAGR": "pct", "줄이기 단독 MDD": "pct", "장치 없음 대비 ΔCAGR": "pp", "장치 없음 대비 ΔMDD": "pp",
       "보류 구간 수익": "pct", "보류 구간 MDD": "pct", "보류 구간 종목 투자비중": "rate", "보류 구간 청산수": "int", "보류 구간 거래일": "int",
       "현금 보유일 비중": "rate", "장치 없음 수익": "pct", "장치 없음 MDD": "pct", "Δ수익": "pp", "무작위 구간 수익 중앙": "pct", "장치 없음 무작위 중앙": "pct"}

def export_stage2(day):
    rdc = lambda n: prep(pd.concat([rd(f"{n}_{u}.csv") for u in ("swing", "day")], ignore_index=True))
    R, M, HR, HM = rdc("exposure2_runs"), rdc("exposure2_mc"), rdc("holdout_runs"), rdc("holdout_mc")
    J, HO = prep(rd("exposure2_summary.csv")), prep(rd("holdout_summary.csv"))
    for d in (R, M, J):
        for c in ("신호", "수단"):
            if c in d: d[c] = d[c].fillna("")
    wb = Workbook()
    write_notes(wb, "설명", [
        ("문서", "통일장 2-d 2차 — 고른 줄이기 4개 위에 늘리기를 얹은 결과 + 고른 줄이기 4개의 보류 구간 1회 확인 (regime_exposure.py run2/report2, 2026-11-10)"),
        ("고른 줄이기", "케인 11-10: ⑤ 갭쇼크·브레드스·평소보유 현금화 · ⑦ C2+갭쇼크 현금화 · ④ C2 진입중단 · ⑦ C2+갭쇼크 진입중단"),
        ("늘리기 얹기", "각 줄이기 위에 신호 4 × 수단 2. 늘리기는 그 줄이기가 '상승'일 때만 — 슬롯확대 = 창 & 상승이면 상한 15 / ETF = 창 & 상승이면 보유, 하락 판정이면 다음 날 시가 매도"),
        ("2차 비교 기준", "같은 줄이기 단독. 판정 기준 ①~④ 는 1차와 같고 ② 도 그대로 둔다 (케인 11-10)"),
        ("보류 구간", "2016-07-01 ~ 2026-11-04 를 끊지 않고 돌린 뒤 2026-06-01 ~ 11-04 구간만 잰다. 비교 = 장치 없음"),
        ("보류 판정 (실행 전 고정)", "H1 구간 수익 ≥ 장치 없음 · H2 구간 MDD(구간 시작 직전 자산부터)가 장치 없음보다 얕음 · H3 무작위 30회 중 구간 수익이 같은 seed 장치 없음보다 높은 비율 ≥ 2/3"),
        ("사후 상한", "보류 구간에서는 참고만 — zigzag 20% 국면은 끝 구간이 아직 확정되지 않았다"),
        ("표시", "수익·차이·금액은 + 빨강 / − 파랑. 비율은 색 없음. 원은 소수점 없이"),
    ])
    combos = [("메가갤럭시", 40), ("메가갤럭시", 60), ("유니버스", 40), ("유니버스", 60)]; rows = []
    for (g, sn, mn), h in J.groupby(["묶음", "신호", "수단"], sort=False):
        r = {"줄이기": g, "늘리기 신호": sn, "수단": mn, "통과 조합 수": int(h.통과.astype(str).eq("True").sum())}
        for u, T in combos:
            z = h[(h.무대 == u) & (h["T"] == T)]; r[f"{u} T{T}"] = ("✓" if str(z.통과.iloc[0]) == "True" else "✗") if len(z) else ""
        r.update({"ΔCAGR 평균 (줄이기 단독 대비)": h.ΔCAGR.mean(), "ΔMDD 평균": h.ΔMDD.mean(), "무작위 이긴 비율 평균": h["무작위 이긴 비율"].mean(),
                  "구간 우위 수 평균": h["구간 우위 수"].mean(), "장치 없음 대비 ΔCAGR 평균": h["장치 없음 대비 ΔCAGR"].mean(),
                  "① 통과": int(h["①"].astype(str).eq("True").sum()), "② 통과": int(h["②"].astype(str).eq("True").sum()),
                  "③ 통과": int(h["③"].astype(str).eq("True").sum()), "④ 통과": int(h["④"].astype(str).eq("True").sum())})
        rows.append(r)
    write_table(wb, "2차_요약", pd.DataFrame(rows), {"통과 조합 수": "int", "ΔCAGR 평균 (줄이기 단독 대비)": "pp", "ΔMDD 평균": "pp", "무작위 이긴 비율 평균": "rate",
                "구간 우위 수 평균": "num2", "장치 없음 대비 ΔCAGR 평균": "pp", "① 통과": "int", "② 통과": "int", "③ 통과": "int", "④ 통과": "int"},
                note="줄이기 위 늘리기 — 평균은 4개 조합. ①~④ 통과 = 4개 조합 중 그 기준을 넘은 수")
    write_table(wb, "보류구간_판정", HO, XF2, note="2026-06-01~11-04 · 장치 없음 대비 · H1~H3 와 통과 (✓/✗)")
    write_table(wb, "보류구간_결과", HR, XF2, note="정해진 순서 1회 — 보류 구간 수익·MDD·투자비중")
    write_table(wb, "2차_판정_전체", J, XF2, note="줄이기 × 늘리기 × 조합 전체 — ①~④ 와 통과")
    write_table(wb, "2차_정해진순서_1회", R, XF2)
    write_table(wb, "2차_무작위_30회", M, XF2)
    write_table(wb, "보류구간_무작위_30회", HM, XF2)
    f = OUT / f"통일장_2d_2차_보류구간_{day}.xlsx"; wb.save(f); return f

def main(which=None):
    day = datetime.now().strftime("%Y%m%d"); out = []
    if which in (None, "expand"): out.append(export_expand(day))
    if which in (None, "exposure"): out.append(export_exposure(day))
    if which in (None, "stage2") and (OUT / "exposure2_summary.csv").exists(): out.append(export_stage2(day))
    for f in out: print("저장", f)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
