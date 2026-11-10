# -*- coding: utf-8 -*-
"""extend_data_1104.py — 스윙 백테스트 데이터(data/)를 2026-11-04 까지 연장해 data_1104/ 에 만든다 (케인 2026-11-05).

정본 data/ (prices ~2026-06-30 · meta ~2026-05-31 · foreign ~2026-06-30) 는 건드리지 않는다.
연장 원천 (pykrx 가 아니라 LLV — data_2015 의 9-14 이후 종가 불일치를 피하려고):
  prices   2026-07-01~11-04  LLV panel.parquet (219종목, KIS 수정주가·KRX 정규장 OHLC) + 그 밖은 LLV data/raw KRX 원본(원주가)
           6-30 이음새 종가가 다르면(분할·감자) 과거 구간을 비율 보정 (rebuild_data_2015 와 같은 규칙)
  meta     월말 2026-06~10  KRX 원본 월 마지막 거래일: mktcap=MarketCap, value=Amount, shares=ListShrs, market, name
           is_pref=이름 끝 우/우B/우C, is_spac=이름에 스팩, is_managed=False, listing_date=NaT, sector='기타' (정본 관례)
  foreign  월말 2026-07~10  패널 종목 = LLV Foreign_Ratio(%) 그날 값, 패널 밖 = 2026-06 값 유지(ffill, 유니버스 4조·30% 판정엔 영향 없음 — 실측)
  kospi200_daily  LLV tickers/KOSPI200.parquet 로 연장 (구간 정의용)
사용: python3 extend_data_1104.py
"""
from __future__ import annotations
import glob, re, shutil
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
SRC, DST = HERE / "data", HERE / "data_1104"; DST.mkdir(exist_ok=True)
LLV = HERE.parents[1] / "LongLiveVault" / "data"
SEAM, END = pd.Timestamp("2026-06-30"), pd.Timestamp("2026-11-04")
MONTHS = ["2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]

def load_raw(a, b):
    fs = [f for f in glob.glob(str(LLV / "raw" / "krx_kos*_*.parquet")) if a <= re.search(r"(\d{8})", f).group(1) <= b]
    r = pd.concat([pd.read_parquet(f) for f in fs]); r["Date"] = pd.to_datetime(r.Date)
    return r.sort_values(["Ticker", "Date"])

def main():
    px = pd.read_parquet(SRC / "prices.parquet")
    raw = load_raw("20260630", END.strftime("%Y%m%d"))
    pan = pd.read_parquet(LLV / "ohlcv" / "panel.parquet", columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume", "Foreign_Ratio"])
    pan["Date"] = pd.to_datetime(pan.Date); pan = pan[(pan.Date >= SEAM) & (pan.Date <= END)]
    ptk = set(pan.Ticker)
    # ── prices 연장 ──
    ext_p = pan.rename(columns=str.lower)[["date", "ticker", "open", "high", "low", "close", "volume"]]
    ext_r = raw[~raw.Ticker.isin(ptk)].rename(columns=str.lower)[["date", "ticker", "open", "high", "low", "close", "volume"]]
    ext = pd.concat([ext_p, ext_r]); ext = ext[ext.close > 0]
    seam_new = ext[ext.date == SEAM].set_index("ticker").close
    seam_old = px.xs(SEAM, level="date").close
    both = seam_old.index.intersection(seam_new.index)
    ratio = (seam_new[both] / seam_old[both])
    adj = ratio[(ratio - 1).abs() > 0.002]
    print(f"이음새 {SEAM.date()}: 공통 {len(both)}종목, 불일치 {len(adj)}종목 → 과거 구간 비율 보정")
    for t, r in adj.sort_values().items(): print(f"   {t} {'(panel)' if t in ptk else '(raw)'} ×{r:.4f}")
    old = px.reset_index()
    old = old[old.date <= SEAM]
    for t, r in adj.items():
        m = old.ticker == t
        for c in ("open", "high", "low", "close"): old.loc[m, c] = old.loc[m, c] * r
        old.loc[m, "volume"] = old.loc[m, "volume"] / r
    new = ext[ext.date > SEAM]
    out = pd.concat([old, new]).sort_values(["ticker", "date"]).set_index(["date", "ticker"])
    out = out[~out.index.duplicated(keep="last")]
    out.to_parquet(DST / "prices.parquet")
    print(f"prices: {out.index.get_level_values('date').min().date()}~{out.index.get_level_values('date').max().date()}, 종목 {out.index.get_level_values('ticker').nunique()}, 행 {len(out):,}")
    # ── meta / foreign 월말 ──
    meta = pd.read_parquet(SRC / "meta.parquet"); fo = pd.read_parquet(SRC / "foreign.parquet")
    m_rows, f_rows = [], []
    fo_last = fo.xs(SEAM, level="date")
    for mo in MONTHS:
        label = pd.Timestamp(mo) + pd.offsets.MonthEnd(0)
        days = raw.Date[raw.Date.dt.strftime("%Y-%m") == mo]
        d = days.max(); r = raw[raw.Date == d].set_index("Ticker")
        name = r.Name.astype(str)
        is_pref = name.str.contains(r"(우|우B|우C)$", regex=True)
        is_spac = name.str.contains("스팩")
        fr = pan[pan.Date == d].set_index("Ticker").Foreign_Ratio
        fr = fr.reindex(r.index)
        miss = fr.isna()
        fr[miss] = fo_last.foreign_ratio.reindex(r.index[miss]).to_numpy()
        n_ff = int((miss & fr.notna()).sum())
        if mo != "2026-06" or label not in fo.index.get_level_values("date"):
            f_rows.append(pd.DataFrame({"date": label, "ticker": r.index, "foreign_ratio": fr.to_numpy(), "exhaustion": np.nan,
                                        "foreign_shares": (fr / 100 * r.ListShrs).to_numpy(), "listed_shares": r.ListShrs.to_numpy(), "limit_shares": np.nan}))
        if label not in meta.index.get_level_values("date"):
            m_rows.append(pd.DataFrame({"date": label, "ticker": r.index, "mktcap": r.MarketCap.astype(float).to_numpy(), "value": r.Amount.astype(float).to_numpy(),
                                        "shares": r.ListShrs.astype(float).to_numpy(), "foreign": (fr / 100).to_numpy(), "market": r.Market.to_numpy(),
                                        "is_pref": is_pref.to_numpy(), "is_spac": is_spac.to_numpy(), "is_managed": False, "listing_date": pd.NaT,
                                        "sector": "기타", "foreign_exhaustion": np.nan, "name": name.to_numpy()}))
        print(f"{mo}: 기준일 {d.date()} 종목 {len(r)}  지분율 패널 {int((~miss).sum())} / 6월값 유지 {n_ff} / 없음 {int(fr.isna().sum())}  "
              f"시총4조↑&외인30%↑ {int(((r.MarketCap >= 4e12) & (fr >= 30) & ~is_pref & ~is_spac).sum())}종목")
    meta2 = pd.concat([meta.reset_index()] + m_rows).set_index(["date", "ticker"]).sort_index()
    fo2 = pd.concat([fo.reset_index()] + f_rows).set_index(["date", "ticker"]).sort_index()
    for c in ("is_pref", "is_spac", "is_managed"): meta2[c] = meta2[c].astype(bool)
    meta2.to_parquet(DST / "meta.parquet"); fo2.to_parquet(DST / "foreign.parquet")
    print(f"meta ~{meta2.index.get_level_values('date').max().date()}  foreign ~{fo2.index.get_level_values('date').max().date()}")
    # ── kospi200_daily ──
    k = pd.read_parquet(SRC / "kospi200_daily.parquet") if (SRC / "kospi200_daily.parquet").exists() else pd.read_parquet(HERE / "data_2015" / "kospi200_daily.parquet")
    kl = pd.read_parquet(LLV / "ohlcv" / "tickers" / "KOSPI200.parquet"); kl["Date"] = pd.to_datetime(kl.Date); kl = kl.set_index("Date")
    k = k[k.index <= SEAM]
    add = kl[(kl.index > SEAM) & (kl.index <= END)][["Open", "High", "Low", "Close"]].astype(float)
    k2 = pd.concat([k, add]); k2.index.name = "Date"; k2.to_parquet(DST / "kospi200_daily.parquet")
    print(f"kospi200_daily ~{k2.index.max().date()}")
    for f in ("benchmark.parquet", "index_kospi.parquet"):
        if (SRC / f).exists(): shutil.copy(SRC / f, DST / f)

if __name__ == "__main__":
    main()
