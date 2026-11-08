# -*- coding: utf-8 -*-
"""surprise_xlsx.py — 급등 연구 결과를 엑셀로 (케인 2026-11-08). 표기 규칙: 상승·증가 #ef5350 / 하락·감소 #1976D2, 숫자 우측 정렬,
변화율 소수 첫째 자리. 비율·수익은 % 단위 숫자(35.9 = 35.9%), 대비는 %p 단위."""
import numpy as np, pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
RED, BLUE = "EF5350", "1976D2"
LABEL = {"n": "건수", "g5": "5일 전부 반납", "g10": "10일 전부 반납", "g20": "20일 전부 반납", "half5": "5일 절반 반납", "half10": "10일 절반 반납",
         "b5": "5일 고가 돌파", "b10": "10일 고가 돌파", "w20": "20일 승률", "du5": "5일 최고 종가", "du10": "10일 최고 종가",
         "dd5": "5일 최저 종가", "dd10": "10일 최저 종가", "med_du5": "5일 최고 종가(중앙)", "med_du10": "10일 최고 종가(중앙)",
         "med_dd5": "5일 최저 종가(중앙)", "med_dd10": "10일 최저 종가(중앙)", "a5": "5일 수익", "a10": "10일 수익", "a20": "20일 수익",
         "xb20": "20일 베타보정 초과", "f5": "5일 종가수익", "f10": "10일 종가수익", "f20": "20일 종가수익", "x5": "5일 지수초과", "x10": "10일 지수초과", "x20": "20일 지수초과",
         "nA": "A 건수", "nB": "B 건수"}
RET = {"du5", "du10", "dd5", "dd10", "med_du5", "med_du10", "med_dd5", "med_dd10", "a5", "a10", "a20", "xb20", "f5", "f10", "f20", "x5", "x10", "x20"}
HDR = PatternFill("solid", fgColor="EDEFF2"); THIN = Border(bottom=Side(style="thin", color="BBBBBB"))
B = Font(bold=True); R = Alignment(horizontal="right"); L = Alignment(horizontal="left"); C = Alignment(horizontal="center", wrap_text=True, vertical="center")

def _title(ws, row, text, size=12):
    ws.cell(row, 1, text).font = Font(bold=True, size=size); return row + 1

def _table(ws, row, df, color_cols=(), sig=None, idx_names=None, pct_fmt="0.0", signed=False):
    """df 를 row 부터 쓴다. 인덱스는 왼쪽 열. color_cols 의 값은 부호 색. sig: 같은 모양 bool DataFrame → 굵게."""
    df = df.copy(); idx = df.index
    names = idx_names or ([n or "" for n in idx.names] if isinstance(idx, pd.MultiIndex) else [idx.name or ""])
    cols = list(df.columns); c0 = 1
    for j, nme in enumerate(names): ws.cell(row, c0 + j, nme)
    for j, cname in enumerate(cols): ws.cell(row, c0 + len(names) + j, LABEL.get(cname, cname))
    for cc in range(c0, c0 + len(names) + len(cols)):
        x = ws.cell(row, cc); x.font = B; x.fill = HDR; x.alignment = C; x.border = THIN
    ws.row_dimensions[row].height = 32; row += 1
    fmt = ('+' + pct_fmt + ';-' + pct_fmt + ';' + pct_fmt) if signed else pct_fmt
    for i, (key, rec) in enumerate(df.iterrows()):
        keys = key if isinstance(key, tuple) else (key,)
        for j, k in enumerate(keys): ws.cell(row, c0 + j, str(k)).alignment = L
        for j, cname in enumerate(cols):
            v = rec[cname]; x = ws.cell(row, c0 + len(names) + j)
            if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v):
                x.value = float(v); x.alignment = R
                x.number_format = "#,##0" if cname in ("n", "nA", "nB") else fmt
                if cname in color_cols and v != 0: x.font = Font(color=RED if v > 0 else BLUE, bold=bool(sig is not None and sig.iloc[i][cname]))
                elif sig is not None and cname in sig.columns and sig.iloc[i][cname]: x.font = B
            elif isinstance(v, str): x.value = v; x.alignment = L
            else: x.value = "—"; x.alignment = R
        row += 1
    return row + 1

def _widths(ws, first=26, rest=10):
    ws.column_dimensions["A"].width = first
    for c in range(2, ws.max_column + 1): ws.column_dimensions[get_column_letter(c)].width = rest

def _contrast_block(ws, row, t, metrics):
    v = t.set_index("대비")[["nA", "nB"] + metrics]
    sig = pd.DataFrame(False, index=v.index, columns=v.columns)
    for m in metrics: sig[m] = ((t[m + "_lo"] > 0) | (t[m + "_hi"] < 0)).to_numpy()
    return _table(ws, row, v, color_cols=set(metrics), sig=sig, idx_names=["대비 (A − B)"], signed=True)

def write_step2(path, tag, n, baseline, descs, contrasts, r2, metrics, ref=False):
    wb = Workbook(); ws = wb.active; ws.title = "설명"
    lines = [("급등(서프라이즈) 연구 — 2단계 기본 대비" + (" · 참고 정의" if ref else ""), True),
             ("학습 구간 2015-01-02 ~ 2026-04-30 · 보류 2026-05-01 ~ (열지 않음)", False),
             (f"종목 급등 건수 {n:,}  ·  크기 " + ("+5 / +7 / +10% (참고)" if ref else "+6 / +9 / +14% (주, 같은 드묾)") +
              "  ·  거래량 " + ("1 / 2배 (참고)" if ref else "1.5 / 3배 (주)"), False),
             (f"기준선 (유니버스 아무 날, 다음 날 시가 진입, 비용 0.23%): 5일 {baseline[5]:+.2f}% · 10일 {baseline[10]:+.2f}% · 20일 {baseline[20]:+.2f}%", False),
             ("", False), ("지표 (pc = 급등 전날 종가, H = 급등일 고가, C = 급등일 종가)", True),
             ("전부 반납 g: k일 안 종가 ≤ pc 인 날이 있는 비율 (%)", False),
             ("절반 반납 half: k일 안 종가 ≤ pc + ½(C − pc) 비율 (%)", False),
             ("고가 돌파 b: k일 안 종가 > H 비율 (%) — 2차 상승", False),
             ("최고·최저 종가 du/dd: 이후 k일 최고·최저 종가 ÷ C − 1, 중앙값 (%)", False),
             ("수익 a: k일 뒤 종가 ÷ 다음 날 시가 − 1 − 0.23%, 평균 (%) · 승률 w20 = 20일 수익 > 0 비율", False),
             ("베타보정 초과 xb20: 20일 종가수익 − β60 × 코스피200 같은 기간 수익 (%)", False),
             ("", False), ("범위", True),
             ("단독 · 섹터 동반(업종 동료 평균 ≥ +3%) · 시장·반등형(코스피200 ≥ +2% & 직전 10일 안 −2% 이하 날 있음) · 시장·순수(그 밖 시장 동반)", False),
             ("형태 (갭 몫 φ = ln(1+갭)/ln(1+하루)): ① 갭하락→급등 <0 · ② 장중 주도 0~0.4 · ③ 섞임 0.4~0.6 · ④ 갭+추가 0.6~1 · ⑤ 갭상승→밀림 >1", False),
             ("", False), ("읽는 법", True),
             ("대비 시트: A − B, %p 단위. 빨강 = A가 큼, 파랑 = A가 작음. 굵게 = 날짜 묶음 부트스트랩 95% 구간이 0 밖", False),
             ("거래량·형태 대비는 같은 크기·형태(또는 크기·거래량) 칸 안에서 비교한 값을 칸 크기로 가중 평균", False),
             ("95% 구간 상·하한 전부는 '대비_95%구간' 시트", False),
             ("⚠ 전부 반납(g)은 급등 폭이 클수록 기계적으로 낮아진다 — 크기 비교는 절반 반납(half)을 함께 볼 것", False)]
    for i, (t, bold) in enumerate(lines, 1): ws.cell(i, 1, t).font = Font(bold=bold, size=13 if i == 1 else 11)
    ws.column_dimensions["A"].width = 120
    order = [("범위", "범위"), ("크기", "크기 (범위 안)"), ("거래량", "거래량 (범위 안)"), ("형태", "형태 (범위 안)")]
    for key, title in order:
        w = wb.create_sheet(key); r = _title(w, 1, f"{title} — 기술 통계")
        d = descs[key]
        if key == "범위": d = pd.concat([d, descs["전체"]])
        r = _table(w, r, d, color_cols=RET, idx_names=(["범위"] if key == "범위" else ["범위", key]))
        r = _title(w, r, f"{title} — 대비 (A − B, %p)"); r = _contrast_block(w, r, contrasts[key], metrics)
        _widths(w); w.freeze_panes = "B3" if key == "범위" else "C3"
    w = wb.create_sheet("설명력"); r = _title(w, 1, "설명력 R² (%) — 선형, 범주 더미")
    rr = r2.copy(); rr.index.name = "지표"; rr = rr.rename(index=LABEL)
    _table(w, r, rr, pct_fmt="0.00", idx_names=["지표"]); _widths(w, 22, 14)
    # 95% 구간 긴 표
    w = wb.create_sheet("대비_95%구간"); longs = []
    for key, t in contrasts.items():
        for _, rec in t.iterrows():
            for m in metrics:
                lo, hi = rec[m + "_lo"], rec[m + "_hi"]
                longs.append({"묶음": key, "대비": rec["대비"], "지표": LABEL.get(m, m), "A 건수": rec["nA"], "B 건수": rec["nB"],
                              "값 (%p)": rec[m], "95% 하한": lo, "95% 상한": hi,
                              "판정": ("유의" if (np.isfinite(lo) and (lo > 0 or hi < 0)) else ("—" if not np.isfinite(lo) else "0 포함"))})
    L_ = pd.DataFrame(longs)
    hdr = list(L_.columns)
    for j, h in enumerate(hdr, 1):
        x = w.cell(1, j, h); x.font = B; x.fill = HDR; x.alignment = C
    for i, rec in enumerate(L_.itertuples(index=False), 2):
        for j, v in enumerate(rec, 1):
            x = w.cell(i, j)
            if isinstance(v, (float, int, np.floating, np.integer)):
                if np.isfinite(v):
                    x.value = float(v); x.alignment = R
                    x.number_format = "#,##0" if hdr[j - 1].endswith("건수") else "+0.0;-0.0;0.0"
                    if hdr[j - 1] == "값 (%p)" and v != 0: x.font = Font(color=RED if v > 0 else BLUE, bold=rec[-1] == "유의")
                else: x.value = "—"; x.alignment = R
            else: x.value = v
    w.freeze_panes = "A2"; w.auto_filter.ref = f"A1:{get_column_letter(len(hdr))}{len(L_) + 1}"
    for j, wd in enumerate([8, 34, 18, 9, 9, 10, 10, 10, 8], 1): w.column_dimensions[get_column_letter(j)].width = wd
    if not ref: _scopes(wb)
    wb.save(path); print("엑셀 →", path)

def _scopes(wb):
    from stock_gapshock import OUT
    from stock_surprise import HOLD
    M = ["g5", "g10", "g20", "half5", "half10", "b5", "b10"]
    def d(x, by, xcols=True):
        g = x.groupby(by, observed=True); R_ = ["f5", "f10", "f20"] + (["x5", "x10", "x20"] if xcols else [])
        return pd.concat([g.size().rename("n"), g[M].mean() * 100, g[["du10", "dd10"]].median().add_prefix("med_") * 100, g[R_].mean() * 100], axis=1)
    s = pd.read_parquet(OUT / "sector_surprise_events.parquet"); s = s[s.date < HOLD].copy()
    s["size"] = pd.cut(s.r, [0.03, 0.045, 0.06, 1], right=False, labels=["작은 +3~+4.5", "중간 +4.5~+6", "큰 ≥+6"])
    i = pd.read_parquet(OUT / "index_surprise_events.parquet"); i = i[i.date < HOLD].copy()
    i["size"] = pd.cut(i.r, [0.02, 0.03, 0.05, 1], right=False, labels=["작은 +2~+3", "중간 +3~+5", "큰 ≥+5"])
    i["era"] = pd.cut(i.date.dt.year, [1999, 2007, 2014, 2026], labels=["2000–07", "2008–14", "2015–26.4"])
    w = wb.create_sheet("섹터·지수"); r = _title(w, 1, "섹터(KRX 업종지수) 급등 ≥ +3% — 범위 (수익은 종가 기준 f, 지수초과 x)")
    r = _table(w, r, d(s, "ctx"), color_cols=RET, idx_names=["범위"])
    r = _title(w, r, "섹터 급등 — 범위 × 크기"); r = _table(w, r, d(s, ["ctx", "size"]), color_cols=RET, idx_names=["범위", "크기"])
    r = _title(w, r, "코스피200 급등 ≥ +2% (2000~) — 크기"); r = _table(w, r, d(i, "size", False), color_cols=RET, idx_names=["크기"])
    r = _title(w, r, "코스피200 급등 — 시기"); r = _table(w, r, d(i, "era", False), color_cols=RET, idx_names=["시기"])
    _widths(w, 16, 10)

def write_step3(path, T, top10, base):
    wb = Workbook(); ws = wb.active; ws.title = "설명"
    lines = [("급등(서프라이즈) 연구 — 3단계 견고성 (학습 2015-01-02 ~ 2026-04-30)", True),
             ("주 정의: 크기 +6 / +9 / +14% · 거래량 1.5 / 3배 · 참고 정의 열 = +5 / +7 / +10% · 1 / 2배", False), ("", False),
             ("판정 기준 (결과 보기 전 고정)", True),
             ("① 날짜 묶음 부트스트랩 95% 구간이 0 밖 (600회)", False), ("② 시기 3개 (2015–19 / 2020–22 / 2023–26.4) 부호가 기대와 같음, 시기별 최소 30건", False),
             ("③ 몰림 상위 10일 ±5거래일 제외 후 부호 유지", False), ("④ 2026년 1~4월 제외 후 부호 유지 (케인 추가)", False),
             ("견고 = ①②③④ 모두 · 부분 = ①만 통과 · 약함 = ① 실패 · '없음' 주장은 95% 구간이 0을 포함하면 '없음 확인'", False), ("", False),
             ("단위: 비율·수익의 차이 %p (수준 비교 U1·U9 는 기준선 대비 %p). 빨강 = +, 파랑 = −", False),
             ("몰림 상위 10일: " + ", ".join(d.strftime("%Y-%m-%d") for d in top10), False),
             ("기준선 a20 (유니버스 아무 날): " + " · ".join(f"{k} {v:+.2f}%" for k, v in base.items()), False)]
    for i, (t, b) in enumerate(lines, 1): ws.cell(i, 1, t).font = Font(bold=b, size=13 if i == 1 else 11)
    ws.column_dimensions["A"].width = 120
    w = wb.create_sheet("견고성"); hdr = list(T.columns)
    lab = {"전체": "전체 (%p)", "하한": "95% 하한", "상한": "95% 상한", "nA": "A 건수", "nB": "B 건수"}
    for j, h in enumerate(hdr, 1):
        x = w.cell(1, j, lab.get(h, h)); x.font = B; x.fill = HDR; x.alignment = C
    w.row_dimensions[1].height = 32
    numc = {"전체", "하한", "상한", "2015–19", "2020–22", "2023–26.4", "몰림날 제외", "2026.1~4 제외", "참고 정의"}
    verdict_fill = {"견고": "E3F2E1", "부분": "FFF4D6", "약함": "F3F3F3", "없음 확인": "E3F2E1"}
    for i, rec in enumerate(T.itertuples(index=False), 2):
        for j, (h, v) in enumerate(zip(hdr, rec), 1):
            x = w.cell(i, j)
            if h in numc or h in ("nA", "nB", "시기 최소 n"):
                if isinstance(v, (float, int, np.floating, np.integer)) and np.isfinite(v):
                    x.value = float(v); x.alignment = R
                    x.number_format = "#,##0" if h in ("nA", "nB", "시기 최소 n") else "+0.0;-0.0;0.0"
                    if h in numc and v != 0: x.font = Font(color=RED if v > 0 else BLUE, bold=(h == "전체"))
                else: x.value = "—"; x.alignment = R
            else:
                x.value = v; x.alignment = C if h in ("①", "②", "③", "④", "기대", "지표", "판정", "참고 부호", "주장") else L
                if h == "판정": x.fill = PatternFill("solid", fgColor=verdict_fill.get(v, "FDE2E1")); x.font = B
    widths = {"주장": 6, "내용": 40, "지표": 8, "기대": 5}
    for j, h in enumerate(hdr, 1): w.column_dimensions[get_column_letter(j)].width = widths.get(h, 9 if h not in ("①", "②", "③", "④") else 4)
    w.freeze_panes = "E2"; wb.save(path); print("엑셀 →", path)

def write_generic(path, intro, sheets):
    """intro: [(텍스트, 굵게)] · sheets: [(시트명, [(제목, df, 색칠열, 부호표시, 굵게 DataFrame|None, 인덱스 이름|None)])]"""
    wb = Workbook(); ws = wb.active; ws.title = "설명"
    for i, (t, b) in enumerate(intro, 1): ws.cell(i, 1, t).font = Font(bold=b, size=13 if i == 1 else 11)
    ws.column_dimensions["A"].width = 130
    for name, blocks in sheets:
        w = wb.create_sheet(name); r = 1
        for title, df, cc, signed, sig, idxn in blocks:
            r = _title(w, r, title); r = _table(w, r, df, color_cols=set(cc), sig=sig, idx_names=idxn, signed=signed)
        _widths(w, 30, 11); w.freeze_panes = "B3"
    wb.save(path); print("엑셀 →", path)
