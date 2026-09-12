"""build_data_ext.py — 스윙 포트 데이터 스냅샷(2026-06-30)을 LLV panel 로 이어붙인 사본을 data_ext/ 에 만든다.

정본 data/ 는 건드리지 않는다. (Kane 2026-09-12: "스냅샷을 업데이트해서 진행")
  prices  : 2026-07-01~ LLV panel OHLCV (208종목). 이음새(마지막 공통일) 종가 비율로 스케일 —
            LLV 가 액면분할을 소급조정한 종목(3건)도 기존 시계열과 연속되게.
  meta    : 월말 시총 = 해당 월 마지막 거래일 LLV MarketCap. 나머지 필드는 종목의 마지막 meta 행 복사.
  foreign : 월말 외국인지분율 = 해당 월 마지막 유효값 LLV Foreign_Ratio (% 척도 동일 확인).
한계: LLV panel 208종목 밖의 종목은 연장되지 않는다 — 4조↑ 147종목은 전부 panel 안에 있어
      4조/30% 유니버스에는 영향 없음. 2조 유니버스는 누락 가능 (2026-07~ 구간 한정).
"""
import shutil
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
SRC, DST = HERE / "data", HERE / "data_ext"
LLV = HERE.parent.parent / "LongLiveVault" / "data" / "ohlcv" / "panel.parquet"
DST.mkdir(exist_ok=True)
for f in ("benchmark.parquet", "index_kospi.parquet"):
    shutil.copy(SRC / f, DST / f)

px = pd.read_parquet(SRC / "prices.parquet")
last_px_date = px.index.get_level_values("date").max()
pn = pd.read_parquet(LLV, columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume", "MarketCap", "Foreign_Ratio"])
pn = pn.rename(columns={"Date": "date", "Ticker": "ticker"})
new_end = pn.date.max()
print(f"snapshot ~{last_px_date.date()} → panel ~{new_end.date()}, panel tickers {pn.ticker.nunique()}")

# ── prices ──
seam_px = px.xs(last_px_date, level="date")["close"]
seam_pn = pn[pn.date == last_px_date].set_index("ticker")["Close"]
scale = (seam_px / seam_pn).dropna()
scale = scale[(scale > 0) & np.isfinite(scale)]
print(f"이음새 스케일 ≠1 종목: {(scale.sub(1).abs() > 0.001).sum()}건 → {scale[scale.sub(1).abs() > 0.001].round(3).to_dict()}")
ext = pn[pn.date > last_px_date].copy()
ext["s"] = ext.ticker.map(scale).fillna(1.0)
for c in ("Open", "High", "Low", "Close"):
    ext[c] = ext[c] * ext.s
ext = ext.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
ext = ext[["date", "ticker", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])
ext = ext.set_index(["date", "ticker"]).astype({c: px[c].dtype for c in ext.columns if c in px.columns})
px2 = pd.concat([px, ext]).sort_index()
px2.to_parquet(DST / "prices.parquet")
print(f"prices: +{len(ext):,} rows, {ext.index.get_level_values('ticker').nunique()} tickers, ~{px2.index.get_level_values('date').max().date()}")

# ── meta (월말) ──
meta = pd.read_parquet(SRC / "meta.parquet")
last_meta = meta.index.get_level_values("date").max()
month_ends = pd.date_range(last_meta + pd.Timedelta(days=1), new_end, freq="M")
tmpl = meta.groupby(level="ticker").last()
rows = []
for me in month_ends:
    d = pn[(pn.date <= me) & (pn.date > me - pd.Timedelta(days=10))].sort_values("date").groupby("ticker").last()
    d = d[d.MarketCap.notna()]
    t = tmpl.reindex(d.index)
    t["mktcap"] = d.MarketCap.astype("float64"); t["date"] = me
    rows.append(t.reset_index())
if rows:
    add = pd.concat(rows).set_index(["date", "ticker"])[meta.columns]
    meta2 = pd.concat([meta, add]).sort_index(); meta2.to_parquet(DST / "meta.parquet")
    print(f"meta: +{[m.date() for m in month_ends]} rows {len(add)}, ≥4조 at {month_ends[-1].date()}: {int((add.xs(month_ends[-1], level='date').mktcap >= 4e12).sum())}")
else:
    shutil.copy(SRC / "meta.parquet", DST / "meta.parquet")

# ── foreign (월말) ──
fo = pd.read_parquet(SRC / "foreign.parquet")
last_fo = fo.index.get_level_values("date").max()
month_ends = pd.date_range(last_fo + pd.Timedelta(days=1), new_end, freq="M")
rows = []
for me in month_ends:
    d = pn[(pn.date <= me) & (pn.date > me - pd.Timedelta(days=10)) & pn.Foreign_Ratio.notna()].sort_values("date").groupby("ticker").last()
    t = pd.DataFrame(index=d.index, columns=fo.columns, dtype="float64")
    t["foreign_ratio"] = d.Foreign_Ratio.astype("float64"); t["date"] = me
    rows.append(t.reset_index())
if rows:
    add = pd.concat(rows).set_index(["date", "ticker"])[fo.columns]
    fo2 = pd.concat([fo, add]).sort_index(); fo2.to_parquet(DST / "foreign.parquet")
    print(f"foreign: +{[m.date() for m in month_ends]} rows {len(add)}, ≥30% at {month_ends[-1].date()}: {int((add.xs(month_ends[-1], level='date').foreign_ratio >= 30).sum())}")
else:
    shutil.copy(SRC / "foreign.parquet", DST / "foreign.parquet")
print("done →", DST)
