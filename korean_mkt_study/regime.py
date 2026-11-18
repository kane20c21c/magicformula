import numpy as np, pandas as pd
B = "/Users/kane.youn/Dev/StoLab/MagicFormula/korean_mkt_study/data_idx/"
old = pd.read_parquet(B+"kospi200_2000_2014.parquet"); old = old.reset_index() if "Date" not in old.columns else old; old["Date"] = pd.to_datetime(old["Date"])
old = old.set_index("Date").Close.astype(float)
k = pd.read_parquet(B+"krx_indices.parquet"); new = k[k.name=="코스피 200"].set_index("date").sort_index().close.astype(float)
print(f"접합 확인: 2014 마지막 {old.index[-1].date()} {old.iloc[-1]:.2f} / 2015 첫날 {new.index[0].date()} {new.iloc[0]:.2f}")
px = pd.concat([old, new]).sort_index(); px = px[~px.index.duplicated()]
r = px.pct_change()
LBL = {(1,0): "조용한 상승", (1,1): "시끄러운 상승", (0,0): "조용한 하락", (0,1): "시끄러운 하락"}
ORDER = ["조용한 상승", "시끄러운 상승", "조용한 하락", "시끄러운 하락"]

def classify(trend, noise):
    vol = r.rolling(noise).std() * np.sqrt(252)
    med = vol.expanding(min_periods=250).median()          # 그때까지의 과거만 (미래정보 없음)
    if trend == "ma120":
        up = (px > px.rolling(120).mean())
    else:
        up = (px.pct_change(60) > 0)
    noisy = (vol > med)
    ok = vol.notna() & med.notna() & px.rolling(120).mean().notna()
    st = pd.Series([LBL[(int(u), int(n))] for u, n in zip(up, noisy)], index=px.index)[ok]
    return st, vol, med

def spells(st):
    g = (st != st.shift()).cumsum()
    return pd.DataFrame({"state": st.groupby(g).first(), "start": st.index.to_series().groupby(g).first(),
                         "end": st.index.to_series().groupby(g).last(), "len": st.groupby(g).size()})

def report(name, st, detail=True):
    st = st.loc["2015-01-02":]
    sp = spells(st)
    nxt = r.shift(-1).reindex(st.index)                     # 오늘 국면 → 내일 수익률
    print(f"\n===== {name}  ({st.index[0].date()} ~ {st.index[-1].date()}, {len(st)}일) =====")
    print("국면            비중   구간수  평균일  중앙값  최장  ≤3일비율 | 다음날기준 연수익  연σ  Sharpe(Rf0)")
    for s in ORDER:
        x = sp[sp.state == s]; y = nxt[st == s].dropna()
        print(f"{s:8s} {(st==s).mean()*100:6.1f}% {len(x):6d} {x.len.mean():7.1f} {x.len.median():6.0f} {x.len.max():5d} {(x.len<=3).mean()*100:7.1f}% |"
              f" {y.mean()*252*100:8.1f}% {y.std()*np.sqrt(252)*100:5.1f}% {y.mean()/y.std()*np.sqrt(252):6.2f}")
    if not detail: return sp
    T = pd.crosstab(st.shift().dropna(), st.iloc[1:], normalize="index").reindex(index=ORDER, columns=ORDER)
    print("\n하루 전환 확률 (행: 오늘 → 열: 내일, %)"); print((T*100).round(1).to_string())
    print("\n국면별 최장 구간 Top2")
    for s in ORDER:
        for _, row in sp[sp.state == s].nlargest(2, "len").iterrows():
            print(f"  {s:8s} {row.start.date()} ~ {row.end.date()}  {row.len}일")
    yr = pd.crosstab(st.index.year, st, normalize="index").reindex(columns=ORDER)*100
    print("\n연도별 국면 비중 (%)"); print(yr.round(0).astype(int).to_string())
    return sp

st1, vol, med = classify("ma120", 20)
print(f"20일 변동성 확장창 중앙값(문턱) 2015-01 {med.loc['2015-01-02']*100:.1f}%  →  2026-11 {med.iloc[-1]*100:.1f}%")
sp1 = report("기본: 120일 이평 × 20일 변동성", st1)
st2, _, _ = classify("ret60", 60)
report("변형: 60일 수익률 × 60일 변동성", st2, detail=False)
