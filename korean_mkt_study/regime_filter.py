import numpy as np, pandas as pd
exec(open(__import__("pathlib").Path(__file__).resolve().parent.joinpath("regime.py"), encoding="utf-8").read().split("def classify")[0])     # px, r, LBL, ORDER
ma = px.rolling(120).mean()
vol = r.rolling(20).std() * np.sqrt(252)
med = vol.expanding(min_periods=250).median()
ok = ma.notna() & vol.notna() & med.notna()
P, M, V, T = px[ok], ma[ok], vol[ok], med[ok]

def hyst(x, lo, hi, init):
    out, s = [], init
    for a, l, h in zip(x, lo, hi):
        if a > h: s = 1
        elif a < l: s = 0
        out.append(s)
    return pd.Series(out, index=x.index)

def confirm(raw, n):
    out, s, cand, cnt = [], int(raw.iloc[0]), None, 0
    for v in raw.astype(int):
        if v == s: cand, cnt = None, 0
        else:
            cnt = cnt + 1 if v == cand else 1; cand = v
            if cnt >= n: s, cand, cnt = v, None, 0
        out.append(s)
    return pd.Series(out, index=raw.index)

def to_state(up, noisy):
    return pd.Series([LBL[(int(u), int(n))] for u, n in zip(up, noisy)], index=up.index)

raw_up, raw_nz = (P > M).astype(int), (V > T).astype(int)
def make(kind, bd=0.02, bn=0.10, n=5):
    if kind == "raw": return to_state(raw_up, raw_nz)
    if kind == "hyst":
        return to_state(hyst(P, M*(1-bd), M*(1+bd), raw_up.iloc[0]), hyst(V, T*(1-bn), T*(1+bn), raw_nz.iloc[0]))
    return to_state(confirm(raw_up, n), confirm(raw_nz, n))

def stats(st):
    st = st.loc["2015-01-02":]
    g = (st != st.shift()).cumsum()
    sp = pd.DataFrame({"s": st.groupby(g).first(), "len": st.groupby(g).size(),
                       "a": st.index.to_series().groupby(g).first(), "b": st.index.to_series().groupby(g).last()})
    nxt = r.shift(-1).reindex(st.index)
    rows = {}
    for s in ORDER:
        x = sp[sp.s == s]; y = nxt[st == s].dropna()
        rows[s] = dict(비중=(st == s).mean()*100, 구간=len(x), 평균=x.len.mean(), 중앙=x.len.median(), 최장=x.len.max(),
                       짧은=(x.len <= 3).mean()*100, 연수익=y.mean()*252*100, 연σ=y.std()*np.sqrt(252)*100,
                       Sharpe=y.mean()/y.std()*np.sqrt(252))
    return pd.DataFrame(rows).T, sp

pd.set_option("display.width", 220)
fmt = {"비중":1,"평균":1,"중앙":0,"짧은":1,"연수익":1,"연σ":1,"Sharpe":2}
res = {}
for name, args in [("원본", ("raw",)), ("완충대 ±2%/±10%", ("hyst",)), ("5일 확인", ("conf",))]:
    t, sp = stats(make(*args)); res[name] = (t, sp)
    print(f"\n===== {name}  총 구간 {len(sp)}개 =====")
    print(t.round(fmt).to_string())
print("\n===== 완충대 민감도 (총 구간 수 / 3일 이하 비율 / 중앙값) =====")
for bd in (0.01, 0.02, 0.03):
    for bn in (0.05, 0.10, 0.20):
        t, sp = stats(make("hyst", bd, bn))
        print(f"  방향 ±{bd*100:.0f}%  소음 ±{bn*100:.0f}%  구간 {len(sp):3d}  3일이하 {(sp.len<=3).mean()*100:4.1f}%  중앙값 {sp.len.median():4.0f}일  평균 {sp.len.mean():5.1f}일")
for n in (3, 5, 10):
    t, sp = stats(make("conf", n=n))
    print(f"  {n}일 확인  구간 {len(sp):3d}  3일이하 {(sp.len<=3).mean()*100:4.1f}%  중앙값 {sp.len.median():4.0f}일  평균 {sp.len.mean():5.1f}일")
t, sp = res["완충대 ±2%/±10%"]
print("\n완충대 기준 국면별 최장 Top2")
for s in ORDER:
    for _, w in sp[sp.s == s].nlargest(2, "len").iterrows():
        print(f"  {s:8s} {w.a.date()} ~ {w.b.date()}  {w.len}일")
