# -*- coding: utf-8 -*-
"""surprise_step1.py — 급등 연구 1단계: 이벤트 표 분포 (결과 지표는 보지 않는다). 학습 구간만 + 보류 건수만.
비교 = 급락(stock_shock_events_ctx) 같은 학습 구간(~2026-04-30)."""
import numpy as np, pandas as pd
from pathlib import Path
from stock_gapshock import universe, D, OUT, LIMIT_CHANGE
from stock_surprise import HOLD
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
up = pd.read_parquet(OUT / "stock_surprise_events_ctx.parquet")
dn = pd.read_parquet(OUT / "stock_shock_events_ctx.parquet")
up["hold"] = up.date >= HOLD; dn["hold"] = dn.date >= HOLD
U, Dn = up[~up.hold], dn[~dn.hold]
def size(r, s):
    a = r.abs(); return pd.cut(a, [0.05, 0.07, 0.10, 1], right=False, labels=["작은", "중간", "큰"])
def shape_up(s): return pd.cut(s, [-np.inf, 0, 0.4, 0.6, 1, np.inf], right=False, labels=["①갭하락→급등", "②장중주도", "③섞임", "④갭+추가", "⑤갭상승→밀림"])
def shape_dn(s): return pd.cut(s, [-np.inf, 0, 0.4, 0.6, 1, np.inf], right=False, labels=["①갭상승→붕괴", "②장중주도", "③섞임", "④갭+추가", "⑤갭→반등"])
def vb(v): return pd.cut(v, [0, 1, 2, np.inf], right=False, labels=["저<1", "중1~2", "고≥2"])
for x, sh in ((U, shape_up), (Dn, shape_dn), (up, shape_up)):
    x["size"] = size(x.r, x); x["shape"] = sh(x.share); x["vb"] = vb(x.vol_n)
# 분모: 유니버스 정상 종목-일
p = pd.read_parquet(D / "prices.parquet").reset_index().sort_values(["ticker", "date"])
g = p.groupby("ticker", sort=False); p["r"] = p.close / g.close.shift(1) - 1; pv = g.volume.shift(1)
lim = np.where(p.date < LIMIT_CHANGE, 0.1505, 0.3005); p["year"] = p.date.dt.year
p = p[(pv > 0) & (p.volume > 0) & (p.open > 0) & (p.r.abs() <= lim) & (p.date < HOLD)]
p = p.merge(universe(p, pd.read_parquet(D / "meta.parquet")), on=["ticker", "year"])
N = len(p); print(f"분모 유니버스 정상 종목-일 (학습): {N:,}")
print("\n[1] 원분포 빈도 (제외 규칙 R4·R5 전, 분모 대비 %)")
rows = []
for t in (0.05, 0.07, 0.10):
    rows.append(dict(임계=f"±{t:.0%}", 급등_건=int((p.r >= t).sum()), 급등_pct=(p.r >= t).mean()*100, 급락_건=int((p.r <= -t).sum()), 급락_pct=(p.r <= -t).mean()*100))
t1 = pd.DataFrame(rows); t1["급등/급락"] = t1.급등_건 / t1.급락_건; print(t1.round(3).to_string(index=False))
fr = [(p.r <= -t).mean() for t in (0.05, 0.07, 0.10)]; q = np.array([-0.05, -0.07, -0.10]); qu = p.r.quantile([1 - f for f in fr]).values
print(f"  급락 −5/−7/−10% 와 같은 드묾({np.round(np.array(fr)*100,2)}%)  ·  같은 드묾의 급등 분위 = {np.round(qu*100,2)}%")
print(f"  상한가 마감(학습 이벤트 중): {int(U.limit_up.sum()):,}건 ({U.limit_up.mean():.1%})")
print(f"\n[2] 이벤트 표 (제외 규칙 적용·20일 경로·유니버스): 급등 학습 {len(U):,} · 보류 {int(up.hold.sum()):,}  |  급락 학습 {len(Dn):,}")
def tab(col, A=U, B=Dn):
    a = A[col].value_counts(); b = B[col].value_counts()
    t = pd.DataFrame({"급등_건": a, "급등_%": a / len(A) * 100, "급락_건": b, "급락_%": b / len(B) * 100}); return t.round(1)
print("\n[3] 범위"); print(tab("ctx").to_string())
print("\n[4] 크기"); print(tab("size").to_string())
print("\n[5] 형태"); print(tab("shape").reset_index().to_string(index=False))
print("\n[6] 거래량 (v 분위: 급등 / 급락)")
print(pd.DataFrame({"급등": U.vol_n.quantile([.1,.25,1/3,.5,2/3,.75,.9]), "급락": Dn.vol_n.quantile([.1,.25,1/3,.5,2/3,.75,.9])}).round(2).T.to_string())
print(tab("vb").to_string())
print("\n[7] 범위 × 크기 (급등 학습 건수)"); print(pd.crosstab(U.ctx, U["size"], margins=True).to_string())
print("\n[8] 범위 × 형태 (급등 학습, 행 %)"); print((pd.crosstab(U.ctx, U["shape"], normalize="index")*100).round(1).to_string())
print("\n[9] 범위 × 거래량 (급등 학습, 행 %)"); print((pd.crosstab(U.ctx, U.vb, normalize="index")*100).round(1).to_string())
print("\n[10] 크기 × 거래량 중앙값 v"); print(U.groupby(["ctx", "size"], observed=True).vol_n.median().unstack().round(2).to_string())
print("\n[11] 꼬리표: 직전 10일 같은 종목 급락 / 직전 10일 시장 −2% (범위별 %)")
print((U.groupby("ctx")[["pre_shock", "pre_mkt"]].mean()*100).round(1).to_string())
print("\n[12] 연도별 건수 (급등 / 급락)")
y = pd.DataFrame({"급등": U.date.dt.year.value_counts(), "급락": Dn.date.dt.year.value_counts(),
                  "급등_시장동반": U[U.ctx == "시장 동반"].date.dt.year.value_counts()}).fillna(0).astype(int).sort_index(); print(y.T.to_string())
print("\n[13] 몰림날 상위 10 (급등 학습)")
dd = U.groupby("date").agg(n=("ticker", "size"), k_r=("k_r", "first")).sort_values("n", ascending=False).head(10)
dd["k_r"] = (dd.k_r*100).round(2); print(dd.to_string())
md = U[U.ctx == "시장 동반"].date.nunique(); print(f"  시장 동반 급등 날짜 수: {md} (급락 시장 동반 날짜 {Dn[Dn.ctx=='시장 동반'].date.nunique()})")
top10 = U.groupby("date").size().nlargest(10).sum(); print(f"  상위 10일이 차지하는 비율: {top10/len(U):.1%}  (급락: {Dn.groupby('date').size().nlargest(10).sum()/len(Dn):.1%})")
sec = pd.read_parquet(OUT / "sector_surprise_events.parquet"); ix = pd.read_parquet(OUT / "index_surprise_events.parquet")
ss = pd.read_parquet(OUT / "sector_shock_events.parquet"); isx = pd.read_parquet(OUT / "index_shock_events.parquet")
S, I = sec[sec.date < HOLD], ix[ix.date < HOLD]; SS, IS = ss[ss.date < HOLD], isx[isx.date < HOLD]
print(f"\n[14] 섹터 급등 ≥+3%: 학습 {len(S):,} · 보류 {int((sec.date>=HOLD).sum())}  |  섹터 급락 학습 {len(SS):,}")
S = S.assign(size=pd.cut(S.r, [0.03, 0.045, 0.06, 1], right=False, labels=["작은", "중간", "큰"]), shape=shape_up(S.share))
print(pd.crosstab(S.ctx, S["size"], margins=True).to_string()); print((S["shape"].value_counts(normalize=True)*100).round(1).sort_index().to_string())
print(f"\n[15] 지수 급등 ≥+2% (2000~): 학습 {len(I)} · 보류 {int((ix.date>=HOLD).sum())}  |  지수 급락 학습 {len(IS)}")
I = I.assign(size=pd.cut(I.r, [0.02, 0.03, 0.05, 1], right=False, labels=["작은", "중간", "큰"]), shape=shape_up(I.share))
print(I["size"].value_counts().sort_index().to_string()); print((I["shape"].value_counts()).sort_index().to_string())
print("  지수 연대별:", I.date.dt.year.floordiv(1).pipe(lambda y: pd.cut(y, [1999, 2007, 2014, 2026], labels=["2000–07", "2008–14", "2015–26.4"])).value_counts().sort_index().to_dict())
