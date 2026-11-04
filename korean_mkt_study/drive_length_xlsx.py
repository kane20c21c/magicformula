"""상승 드라이브 길이 연구 — 결과 표를 엑셀로 (2026-11-04 케인 요청). 입력 out/crash/drive_len_up.parquet (drive_length.py)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
sys.path.insert(0, str(Path(__file__).resolve().parent))
from crash_study import OUT, load_index

RED, BLUE = "ef5350", "1976D2"
FONT = "Arial"
d = pd.read_parquet(OUT / "drive_len_up.parquet"); d["ge5"] = d.n >= 5; d["ge4"] = d.n >= 4
d["nb"] = pd.cut(d.n, [2, 3, 4, 5, 6, 7, 99], labels=["3", "4", "5", "6", "7", "8+"])
d["per_day"] = (1 + d.tot) ** (1 / d.n) - 1
k = load_index()["KOSPI"]; kk = pd.DataFrame({"c": k}); kk["dd60"] = kk.c / kk.c.rolling(60).max() - 1; kk["r5"] = kk.c / kk.c.shift(5) - 1; kk = kk.shift(1)
m = kk.reindex(d.date.values); d["k_dd60"] = m.dd60.values; d["k_r5"] = m.r5.values
d["kq"] = pd.qcut(d.kospi_ret20.rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
d["mq"] = pd.qcut(d.ma60_gap.rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])

# ---------- 표 만들기 (값 종류: 'pct' = 비율 0~1 → %, 'num', 'int', 'signed' = 수익률 부호색) ----------
def agg(g, with_r10=True):
    t = g.agg(건수=("n", "size"), n평균=("n", "mean"), P_n4=("ge4", "mean"), P_n5=("ge5", "mean"), 상승폭중앙=("tot", "median"))
    if with_r10: t["r10중앙"] = g.r10.median()
    return t.reset_index()
FMT = {"건수": "int", "n평균": "num3", "P_n4": "pct", "P_n5": "pct", "상승폭중앙": "spct", "r10중앙": "spct", "ρ": "num3"}

tables = {}  # sheet -> list of (title, df)
# 1. 길이 × 3변수
g = d.groupby("nb", observed=True)
t1 = g.agg(건수=("n", "size"), 상승폭중앙=("tot", "median"), 하루평균수익=("per_day", "median"), 첫날r=("r_d1", "median"), 마지막날r=("r_dn", "median"),
           속도A중앙=("speed", "median"), 거래량T중앙=("v_trend", "median"), 겹침ō중앙=("overlap", "median"), v_level중앙=("v_level", "median")).reset_index().rename(columns={"nb": "길이"})
rows1 = [("길이별 3변수 (드라이브 전 구간 D1..Dn 기준, 2016~2026 상승 드라이브)", t1,
          {"상승폭중앙": "spct", "하루평균수익": "spct", "첫날r": "spct", "마지막날r": "spct", "속도A중앙": "num2", "거래량T중앙": "num2", "겹침ō중앙": "num2", "v_level중앙": "num2"})]
for key, lab in (("speed", "속도 A"), ("v_trend", "거래량 T"), ("overlap", "겹침 ō")):
    ct = pd.crosstab(d.nb, d[key + "_c"], normalize="index"); q = d.groupby("nb", observed=True)[key].quantile([.25, .5, .75]).unstack(); q.columns = ["p25", "p50", "p75"]
    t = pd.concat([ct, q], axis=1).reset_index().rename(columns={"nb": "길이"})
    rows1.append((f"길이별 {lab} 유형 비율·분위", t, {**{c: "pct" for c in ct.columns}, "p25": "num2", "p50": "num2", "p75": "num2"}))
tables["1.길이×3변수"] = rows1

# 2. D0 설명 변수 — 10분위 + 회귀
vars_ = ["kospi_ret20", "breadth20", "kospi_ret60", "ma60_gap", "ret20", "ma20_gap", "ret5", "hi60_gap", "close_d0", "hi252_gap", "amt20", "mktcap", "sig60", "foreign", "sig20", "v5_20", "dn_run_pre"]
desc = {"kospi_ret20": "KOSPI 20일 수익(D0)", "breadth20": "MA20 위 종목 비율(D0)", "kospi_ret60": "KOSPI 60일 수익", "ma60_gap": "종목 종가/MA60−1", "ret20": "종목 20일 수익", "ma20_gap": "종목 종가/MA20−1",
        "ret5": "종목 5일 수익", "hi60_gap": "60일 고점 대비", "close_d0": "주가(원)", "hi252_gap": "252일 고점 대비", "amt20": "20일 평균 거래대금(원)", "mktcap": "시총(직전 월말, 원)", "sig60": "일수익 σ60",
        "foreign": "외국인 지분율(직전 월말)", "sig20": "일수익 σ20", "v5_20": "거래량 5일/20일", "dn_run_pre": "직전 하락 런 길이(일)"}
VF = {v: ("spct" if v in ("kospi_ret20", "kospi_ret60", "ma60_gap", "ret20", "ma20_gap", "ret5", "hi60_gap", "hi252_gap") else ("pct" if v in ("breadth20", "foreign") else ("int" if v in ("close_d0", "amt20", "mktcap", "dn_run_pre") else "num4"))) for v in vars_}
NB = 10
rho_rows = []; rows2 = []
for v in vars_:
    x = d[v]; ok = x.notna(); rho = spearmanr(x[ok], d.n[ok]).correlation
    b = pd.cut(x, [-1, 0, 1, 2, 3, 4, 5, 99], labels=["0", "1", "2", "3", "4", "5", "6+"]) if v == "dn_run_pre" else pd.qcut(x.rank(method="first"), NB, labels=[f"D{i}" for i in range(1, NB + 1)])
    t = agg(d.groupby(b, observed=True)); t.insert(1, "구간최소", d.groupby(b, observed=True)[v].min().values); t.insert(2, "구간중앙", d.groupby(b, observed=True)[v].median().values); t = t.rename(columns={t.columns[0]: "구간"})
    rho_rows.append({"변수": v, "설명": desc[v], "ρ": rho, "D1 P(n≥5)": t.P_n5.iloc[0], "D10 P(n≥5)": t.P_n5.iloc[-1], "최대−최소(pt)": (t.P_n5.max() - t.P_n5.min()) * 100})
    rows2.append((f"{v} — {desc[v]}  (Spearman ρ = {rho:+.3f}, 10분위)", t, {**FMT, "구간최소": VF[v], "구간중앙": VF[v]}))
rho_df = pd.DataFrame(rho_rows).sort_values("ρ")
tables["2.D0변수 10분위"] = [("Spearman ρ(변수 D0, 길이 n) 요약 — 전부 D0(드라이브 첫날 전 거래일) 종가 시점 정보. 최대−최소 = 10분위 중 P(n≥5) 최대와 최소의 차(pt)", rho_df, {"ρ": "num3", "D1 P(n≥5)": "pct", "D10 P(n≥5)": "pct", "최대−최소(pt)": "num2"})] + rows2

# 2b. 회귀 — 표준화(z) 입력. 거래대금·시총·주가는 log. 꼬리는 1·99 백분위로 윈저화
import statsmodels.api as sm
X = d[vars_].copy()
for v in ("amt20", "mktcap", "close_d0"): X[v] = np.log(X[v])
X = X.clip(X.quantile(0.01), X.quantile(0.99), axis=1)
Z = (X - X.mean()) / X.std(); okz = Z.notna().all(axis=1)
Z = Z[okz]; y5 = d.ge5[okz].astype(float); y4 = d.ge4[okz].astype(float); yn = d.n[okz].astype(float)
base5 = y5.mean()
uni = []
for v in vars_:
    m5 = sm.Logit(y5, sm.add_constant(Z[[v]])).fit(disp=0)
    mo = sm.OLS(yn, sm.add_constant(Z[[v]])).fit()
    b = m5.params[v]; c0 = m5.params["const"]
    p_lo = 1 / (1 + np.exp(-(c0 - b))); p_hi = 1 / (1 + np.exp(-(c0 + b)))
    uni.append({"변수": v, "설명": desc[v], "로짓계수(1σ)": b, "z": m5.tvalues[v], "오즈비(1σ)": np.exp(b), "P(n≥5)@−1σ": p_lo, "P(n≥5)@+1σ": p_hi, "효과폭(pt)": (p_hi - p_lo) * 100,
                "McFadden R²": m5.prsquared, "OLS n 계수(1σ)": mo.params[v], "OLS t": mo.tvalues[v], "OLS R²": mo.rsquared})
uni_df = pd.DataFrame(uni).sort_values("z")
multi_vars = ["kospi_ret20", "breadth20", "kospi_ret60", "ma60_gap", "ret5", "hi252_gap", "amt20", "mktcap", "sig60", "foreign", "v5_20", "dn_run_pre"]  # 상관 높은 쌍(ret20/ma20_gap/hi60_gap, sig20)은 뺌
mm5 = sm.Logit(y5, sm.add_constant(Z[multi_vars])).fit(disp=0); mm4 = sm.Logit(y4, sm.add_constant(Z[multi_vars])).fit(disp=0); mmo = sm.OLS(yn, sm.add_constant(Z[multi_vars])).fit()
multi = pd.DataFrame({"변수": multi_vars, "설명": [desc[v] for v in multi_vars],
                      "로짓계수 P(n≥5)": mm5.params[multi_vars].values, "z": mm5.tvalues[multi_vars].values, "오즈비(1σ)": np.exp(mm5.params[multi_vars].values),
                      "로짓계수 P(n≥4)": mm4.params[multi_vars].values, "z ": mm4.tvalues[multi_vars].values,
                      "OLS n 계수(1σ)": mmo.params[multi_vars].values, "OLS t": mmo.tvalues[multi_vars].values})
multi = multi.reindex(multi.z.abs().sort_values(ascending=False).index)
fitinfo = pd.DataFrame({"항목": ["표본", "기준 P(n≥5)", "기준 P(n≥4)", "다변량 로짓 McFadden R² (n≥5)", "다변량 로짓 McFadden R² (n≥4)", "다변량 OLS R² (n)", "입력 처리", "해석"],
                        "값": [f"{int(okz.sum()):,}건 (결측 제외)", f"{base5:.4f}", f"{y4.mean():.4f}", f"{mm5.prsquared:.4f}", f"{mm4.prsquared:.4f}", f"{mmo.rsquared:.4f}",
                               "변수마다 1·99 백분위 윈저화 → 평균0·표준편차1 로 표준화. amt20·mktcap·close_d0 는 log. 계수는 '1σ 증가 시' 효과",
                               "z 의 절대값이 2 를 넘으면 통계적으로 유의하나, R² 가 0.01 미만이면 설명력은 사실상 없음. 효과폭(pt)이 실질 크기"]})
corr = Z[multi_vars].corr(); corr.insert(0, "변수", corr.index)
tables["2b.회귀"] = [("단변량 — 로짓(P(n≥5)) · OLS(n), 변수별 각각 적합 (z 오름차순)", uni_df, {"로짓계수(1σ)": "num4", "z": "num2", "오즈비(1σ)": "num3", "P(n≥5)@−1σ": "pct", "P(n≥5)@+1σ": "pct", "효과폭(pt)": "num2", "McFadden R²": "num4", "OLS n 계수(1σ)": "num4", "OLS t": "num2", "OLS R²": "num4"}),
                   ("다변량 — 12변수 동시 적합 (|z| 내림차순). 상관 높은 중복 변수(ret20·ma20_gap·hi60_gap·sig20)는 제외", multi, {"로짓계수 P(n≥5)": "num4", "z": "num2", "오즈비(1σ)": "num3", "로짓계수 P(n≥4)": "num4", "z ": "num2", "OLS n 계수(1σ)": "num4", "OLS t": "num2"}),
                   ("적합 정보", fitinfo, {}),
                   ("다변량 입력 변수 간 상관(표준화 후)", corr.reset_index(drop=True), {c: "num2" for c in multi_vars})]

# 3. 범주
rows3 = []
for by, lab in (("cap_bucket", "시총 구간(전년말)"), ("market", "시장"), ("year", "연도")):
    t = agg(d.groupby(by, observed=True)); rows3.append((lab, t, FMT))
tables["3.시총·시장·연도"] = rows3

# 4. 시장 상태
d["ddq"] = pd.cut(d.k_dd60, [-1, -0.15, -0.10, -0.05, -0.02, 0.01], labels=["<-15%", "-15~-10%", "-10~-5%", "-5~-2%", ">-2%"])
rows4 = [("KOSPI 60일 고점 대비 낙폭(D0)", agg(d.groupby("ddq", observed=True)), FMT),
         ("KOSPI 20일 수익 5분위 × 직전 5일 KOSPI 반등 중(True)", agg(d.groupby(["kq", d.k_r5 > 0], observed=True)).rename(columns={"k_r5": "5일반등중"}), FMT),
         ("KOSPI 낙폭 <−15% 드라이브 — 연도별 (쇼크 반등 해에만 효과)", agg(d[d.k_dd60 < -0.15].groupby("year")), FMT),
         ("KOSPI 20일 수익 Q1(True) × 종목 MA60 이격 최하위 Q1(True)", agg(d.groupby([d.kq == "Q1", d.mq == "Q1"], observed=True)).rename(columns={"kq": "KOSPI_Q1", "mq": "MA60이격_Q1"}), FMT)]
haz = {}
for q, gq in d.groupby("kq", observed=True):
    haz[q] = {f"P(n≥{kk_+1}|n≥{kk_})": (gq.n >= kk_ + 1).sum() / (gq.n >= kk_).sum() for kk_ in (3, 4, 5, 6, 7)}
rows4.append(("지속률(하자드) — KOSPI 20일 수익 분위별", pd.DataFrame(haz).T.reset_index().rename(columns={"index": "KOSPI분위"}), {c: "pct" for c in pd.DataFrame(haz).T.columns}))
tables["4.시장상태"] = rows4

# 5. 메모
memo = pd.DataFrame({"항목": [
    "데이터", "드라이브", "3변수", "D0 규칙", "하루 어긋남", "섹터", "2026 절단", "건수", "생성"],
    "내용": [
    "korean_mkt_study/data_2015/prices.parquet (2015-01-02~2026-10-29), 동전주 제외(전년 평균 거래대금 10억·연말 시총 1,000억 OR 미달 제외), 2016~ 집계",
    "종가 수익률 부호가 3일 이상 연속 양(+)인 최대 구간. r=0 은 끊음. D1..Dn, D0 = D1 전 거래일",
    "속도 A=|r_Dn|/|r_D1| (감쇄<0.65/가속>1.8) · 거래량 T=V_Dn/V_D1 (상승형 감소<1.00/증가>1.90) · 겹침 ō=mean(D2..Dn 범위겹침) (계단<0.40/미끄럼>0.57). 여기선 드라이브 전 구간으로 계산 — 길이 n 은 사후값이므로 설명용이지 신호용 아님",
    "설명 변수는 전부 D0 종가 시점까지의 정보. 월말 스냅샷(시총·외국인)은 D1 보다 앞선 가장 최근 월말",
    "케인 지시(2026-11-03): 당일 조건으로 넣지 않고 하루 어긋나게 연결",
    "meta.parquet 의 sector 가 전부 '기타' 라 섹터 분석 불가 (data_2015 재구성 때 미채움)",
    "2026-10 말 진행 중이던 드라이브는 데이터 끝에서 잘려 길이가 과소평가됨",
    f"상승 드라이브 {len(d):,}건",
    "drive_length.py → out/crash/drive_len_up.parquet → drive_length_xlsx.py (2026-11-04)"]})
tables["5.메모"] = [("정의·주의사항", memo, {})]

# ---------- 쓰기 ----------
wb = Workbook(); wb.remove(wb.active)
thin = Side(style="thin", color="BBBBBB"); hdr_fill = PatternFill("solid", fgColor="F2F2F2")
NUMFMT = {"int": "#,##0", "num2": "0.00", "num3": "0.000", "num4": "0.0000", "pct": "0.0%", "spct": "0.0%"}
for sheet, items in tables.items():
    ws = wb.create_sheet(sheet); r = 1; widths = {}
    for title, df, fmt in items:
        ws.cell(r, 1, title).font = Font(name=FONT, bold=True, size=11); r += 1
        for j, c in enumerate(df.columns, 1):
            cell = ws.cell(r, j, str(c)); cell.font = Font(name=FONT, bold=True); cell.fill = hdr_fill; cell.border = Border(bottom=thin)
            cell.alignment = Alignment(horizontal="center"); widths[j] = max(widths.get(j, 0), len(str(c)) * 1.8)
        r += 1
        for row in df.itertuples(index=False):
            for j, (c, v) in enumerate(zip(df.columns, row), 1):
                kind = fmt.get(c, "int" if c == "건수" else None)
                if isinstance(v, (np.generic,)): v = v.item()
                if isinstance(v, float) and np.isnan(v): v = None
                if v is not None and not isinstance(v, (int, float, bool)): v = str(v)
                cell = ws.cell(r, j, v); cell.font = Font(name=FONT)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    cell.alignment = Alignment(horizontal="right")
                    if kind in NUMFMT: cell.number_format = NUMFMT[kind]
                    elif isinstance(v, float): cell.number_format = "0.000"
                    if kind == "spct" or c in ("ρ", "z", "z ", "OLS t", "로짓계수(1σ)", "로짓계수 P(n≥5)", "로짓계수 P(n≥4)", "OLS n 계수(1σ)"):
                        if v > 0: cell.font = Font(name=FONT, color=RED)
                        elif v < 0: cell.font = Font(name=FONT, color=BLUE)
                else:
                    cell.alignment = Alignment(horizontal="left" if sheet in ("5.메모", "2b.회귀") and c in ("내용", "값", "설명") else "center")
                widths[j] = max(widths.get(j, 0), min(len(str(v)) * (1.6 if sheet in ("5.메모", "2b.회귀") and c in ("내용", "값", "설명") else 1.1), 110) if v is not None else 0)
            r += 1
        r += 1
    for j, w in widths.items(): ws.column_dimensions[get_column_letter(j)].width = max(8, w + 2)
    ws.freeze_panes = None
out = OUT / "drive_length_up_20261104.xlsx"; wb.save(out); print("saved", out)
