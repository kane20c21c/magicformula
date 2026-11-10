# -*- coding: utf-8 -*-
"""breadth_study.py — 시장 국면 전환 신호 연구 1단계: 브레드스·츠바이크 (케인 2026-11-05).

목적: 시장 전체가 '투자할 수 있는 분위기'로 바뀌는지를 상승/하락 종목 수로 본다.
      2-1 = 지금 값이 츠바이크 브레드스 스러스트 근처인가.

정의 (2026-11-04 노트 + 11-05 케인 합의):
  시장 전체  data_2015/prices (2015-01-02~2026-09-11) + LLV data/raw KRX 원본(그 이후, KOSPI·KOSDAQ).
             동전주 제외 = crash_study.py 와 동일: 전년 일평균 거래대금(종가×거래량) ≥ 10억
             AND 전년 12월말 시총 ≥ 1,000억 (하나라도 미달이면 그해 제외, 신규상장 첫해 제외).
  유니버스   LLV panel.parquet 종목 (ETF 섹터 제외).
  상승/하락  전일 종가 대비 >0 / <0. 보합은 분모에서도 뺀다.
  비율       adv / (adv + dec),  EMA10 = ewm(span=10, adjust=False)
  츠바이크   EMA10 이 0.40 아래였던 날로부터 10거래일 안에 0.615 위로 → 그날이 신호일
사용: python3 breadth_study.py
"""
from __future__ import annotations
import glob, re, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data_2015"; OUT = HERE / "out" / "breadth"; OUT.mkdir(parents=True, exist_ok=True)
LLV = HERE.parents[1] / "LongLiveVault" / "data"
AMT_MIN, CAP_MIN = 1e9, 1e11
# ⚠ prices.parquet 은 2026-09-14(KRX 애프터마켓 개장) 이후 종가가 KRX 정규장 종가와 80% 어긋난다
#   (2026-11-05 실측: 10-21·10-29 각 82%·80.5% 불일치, LLV 패널·KRX 원본은 서로 100% 일치).
#   그래서 이음매를 개장 전 마지막 거래일로 당기고 그 뒤는 KRX 원본만 쓴다.
PX_END = pd.Timestamp("2026-09-11")
LO, HI, WIN = 0.40, 0.615, 10
sys.path.insert(0, str(HERE.parents[1] / "LongLiveVault"))

def eligibility(px):
    amt = (px.close * px.volume).groupby([px.ticker, px.date.dt.year]).mean().rename("amt_avg")
    amt.index.names = ["ticker", "year"]
    meta = pd.read_parquet(DATA / "meta.parquet", columns=["mktcap", "market", "is_pref", "is_spac"]).reset_index()
    ye = meta[meta.date.dt.month == 12].assign(year=lambda d: d.date.dt.year).set_index(["ticker", "year"])
    b = pd.concat([amt, ye[["mktcap", "market", "is_pref", "is_spac"]]], axis=1).reset_index()
    b["eligible"] = (b.amt_avg >= AMT_MIN) & (b.mktcap >= CAP_MIN)
    b["year"] = b.year + 1
    return b

def load_raw():
    fs = sorted(glob.glob(str(LLV / "raw" / "krx_kospi_*.parquet")) + glob.glob(str(LLV / "raw" / "krx_kosdaq_*.parquet")))
    r = pd.concat([pd.read_parquet(f, columns=["Date", "Ticker", "Close", "Volume"]) for f in fs])
    r = r.rename(columns={"Date": "date", "Ticker": "ticker", "Close": "close", "Volume": "volume"})
    r["date"] = pd.to_datetime(r.date); r = r[r.close > 0]
    return r

def daily_counts(df, elig_keys=None):
    df = df.sort_values(["ticker", "date"]).copy()
    df["chg"] = df.groupby("ticker").close.pct_change()
    # 전일 행이 연속 거래일이 아니면(원본 결측 구간) 비교하지 않는다
    df["prev_date"] = df.groupby("ticker").date.shift()
    if elig_keys is not None:
        k = pd.MultiIndex.from_arrays([df.ticker, df.date.dt.year])
        df = df[k.isin(elig_keys)]
    g = df.dropna(subset=["chg"]).groupby("date")
    c = pd.DataFrame({"adv": g.chg.apply(lambda s: (s > 0).sum()), "dec": g.chg.apply(lambda s: (s < 0).sum()),
                      "unch": g.chg.apply(lambda s: (s == 0).sum()), "n": g.chg.size(),
                      "ew_ret": g.chg.mean()})
    return c, df

def zweig(c, lo=LO, hi=HI, win=WIN):
    c = c.copy()
    c["ratio"] = c.adv / (c.adv + c.dec)
    c["ema10"] = c.ratio.ewm(span=10, adjust=False).mean()
    e = c.ema10.to_numpy(); sig = np.zeros(len(c), bool); low_day = [None] * len(c)
    last_sig = -999
    for i in range(len(c)):
        if e[i] > hi:
            j0 = max(0, i - win)
            lows = np.where(e[j0:i] < lo)[0]
            if len(lows) and i - last_sig > win:
                sig[i] = True; last_sig = i; low_day[i] = c.index[j0 + lows[-1]]
    c["thrust"] = sig; c["thrust_from_low"] = low_day
    c["min_ema_10d"] = c.ema10.rolling(WIN + 1).min()
    return c

def main():
    px = pd.read_parquet(DATA / "prices.parquet", columns=["close", "volume"]).reset_index()
    px["ticker"] = px.ticker.astype(str)
    el = eligibility(px)
    keys = pd.MultiIndex.from_frame(el.loc[el.eligible, ["ticker", "year"]])

    raw = load_raw()
    # 이어붙이기: prices 는 PX_END 까지, 원본은 그 이후 + 전일 비교용 PX_END 당일
    tail = raw[raw.date >= PX_END]
    joined = pd.concat([px.loc[px.date <= PX_END, ["date", "ticker", "close", "volume"]], tail[tail.date > PX_END]])
    # 이음매: PX_END 의 종가는 prices(수정주가) 쪽 — 원본(원주가)과 자본변동 종목에서 어긋날 수 있어 원본 쪽으로 교체
    seam = tail[tail.date == PX_END].set_index("ticker").close
    m = (joined.date == PX_END) & joined.ticker.isin(seam.index)
    joined.loc[m, "close"] = joined.loc[m, "ticker"].map(seam).to_numpy()
    mkt, _ = daily_counts(joined, keys)

    # 겹치는 구간 대조: 원본만으로 6~10월 계산 (원본은 2026-05 중순부터 연속)
    rc, _ = daily_counts(raw[raw.date >= "2026-06-01"], keys)
    pc = mkt.loc["2026-06-02":"2026-11-04"]
    cmp = pc[["adv", "dec"]].join(rc[["adv", "dec"]], rsuffix="_raw", how="inner")
    cmp["d_adv"] = cmp.adv - cmp.adv_raw; cmp["d_dec"] = cmp.dec - cmp.dec_raw

    # 유니버스 (LLV 패널, ETF 제외)
    sys.path.insert(0, str(HERE.parents[1] / "LongLiveVault"))
    from stolab_data.core_tickers import TICKER_LIST
    from stolab_data.extend_tickers import EXTEND_LIST
    etf = {t for t, _, s in list(TICKER_LIST) + list(EXTEND_LIST) if s == "ETF"}
    pan = pd.read_parquet(LLV / "ohlcv" / "panel.parquet", columns=["Date", "Ticker", "Close", "Volume"])
    pan = pan.rename(columns={"Date": "date", "Ticker": "ticker", "Close": "close", "Volume": "volume"})
    pan["date"] = pd.to_datetime(pan.date); pan = pan[~pan.ticker.isin(etf) & (pan.close > 0)]
    uni, _ = daily_counts(pan)

    mkt = zweig(mkt); uni = zweig(uni)
    mkt.to_csv(OUT / "breadth_market.csv"); uni.to_csv(OUT / "breadth_universe.csv"); cmp.to_csv(OUT / "seam_check.csv")

    pd.set_option("display.width", 200)
    yr = el[el.eligible].groupby("year").agg(n=("ticker", "size"), pref=("is_pref", "sum"), spac=("is_spac", "sum"))
    print("== 연도별 필터 통과 종목 수 ==\n", yr.loc[2016:].T.to_string())
    print("\n== 원천 대조 2026-06~11-04 (이어붙인 계열 − 원본 단독) ==")
    print(f"일수 {len(cmp)}, adv 차 |중앙| {cmp.d_adv.abs().median():.0f} 최대 {cmp.d_adv.abs().max():.0f}, dec 차 |중앙| {cmp.d_dec.abs().median():.0f} 최대 {cmp.d_dec.abs().max():.0f}")
    print(cmp[(cmp.d_adv.abs() + cmp.d_dec.abs()) > 5].to_string() if ((cmp.d_adv.abs() + cmp.d_dec.abs()) > 5).any() else "  5종목 넘게 다른 날 없음")
    cols = ["n", "adv", "dec", "unch", "ratio", "ema10", "min_ema_10d", "ew_ret"]
    print("\n== 시장 전체 2026-09-25~11-04 ==\n", mkt.loc["2026-09-25":, cols].round(3).to_string())
    print("\n== 유니버스 2026-09-25~11-04 ==\n", uni.loc["2026-09-25":, cols].round(3).to_string())
    for name, c in (("시장 전체", mkt), ("유니버스", uni)):
        t = c[c.thrust]
        print(f"\n== 츠바이크 신호일 — {name} ({c.index.min().date()}~) : {len(t)}건 ==")
        print(t[["thrust_from_low", "ratio", "ema10"]].round(3).to_string())

# ── 민감도 (케인 2026-11-05: 바닥 0.40~0.45 × 창 10~15일, 위쪽 0.615 고정) ──────────
ANSWER_LOWS = ["2026-03-31", "2026-07-30"]          # 정답지 저점 (고점 6/22·10월 초는 츠바이크 대상 아님)
FWD = (20, 60)

def sensitivity():
    mkt = pd.read_csv(OUT / "breadth_market.csv", index_col=0, parse_dates=True)
    mkt = mkt.loc["2016-01-04":]
    k = pd.read_parquet(LLV / "ohlcv" / "tickers" / "KOSPI.parquet"); k["Date"] = pd.to_datetime(k.Date)
    kospi = k.set_index("Date").Close.astype(float).reindex(mkt.index)
    ew = (1 + mkt.ew_ret.fillna(0)).cumprod()
    fwd = {}
    for h in FWD:
        fwd[f"k{h}"] = kospi.shift(-h) / kospi - 1
        fwd[f"e{h}"] = ew.shift(-h) / ew - 1
    fwd = pd.DataFrame(fwd)
    base = {c: (fwd[c].median(), (fwd[c] > 0).mean()) for c in fwd}
    idx = mkt.index
    rows, sigs = [], {}
    for lo in [0.40, 0.41, 0.42, 0.43, 0.44, 0.45]:
        for win in range(10, 16):
            z = zweig(mkt[["adv", "dec", "ew_ret", "n", "unch"]], lo=lo, win=win)
            d = z.index[z.thrust]; sigs[(lo, win)] = d
            r = {"lo": lo, "win": win, "n_sig": len(d)}
            for a in ANSWER_LOWS:
                a = pd.Timestamp(a); after = d[(d >= a) & (d <= a + pd.Timedelta(days=45))]
                r[a.strftime("%m/%d")] = f"+{idx.get_loc(after[0]) - idx.get_loc(a)}일" if len(after) else "-"
            f = fwd.reindex(d)
            for c in fwd:
                r[f"{c}_med"] = f[c].median(); r[f"{c}_pos"] = (f[c] > 0).mean()
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "sensitivity_grid.csv", index=False)
    pd.set_option("display.width", 220)
    print("== 기준선 (2016~ 아무 날이나) ==")
    for c, (m, p) in base.items(): print(f"  {c}: 중앙 {m*100:+.1f}%  플러스 {p*100:.0f}%")
    show = t.copy()
    for c in fwd:
        show[c] = show[f"{c}_med"].map(lambda v: f"{v*100:+.1f}%" if pd.notna(v) else "") + " / " + show[f"{c}_pos"].map(lambda v: f"{v*100:.0f}%" if pd.notna(v) else "")
    print("\n== 36조합 (수익률 = 신호일 종가 기준 중앙값 / 플러스 비율, 2026-11-04 까지 기간이 안 찬 신호는 제외) ==")
    print(show[["lo", "win", "n_sig", "03/31", "07/30"] + list(fwd.columns)].to_string(index=False))
    # 신호일 목록 — 조합별로 어떤 날이 추가되는지
    alld = sorted(set().union(*[set(v) for v in sigs.values()]))
    mat = pd.DataFrame({f"{lo:.2f}/{w}": [d in set(v) for d in alld] for (lo, w), v in sigs.items()}, index=alld)
    cnt = mat.sum(axis=1).rename("조합수")
    out = pd.concat([cnt, fwd.reindex(alld).round(3)], axis=1)
    out.to_csv(OUT / "sensitivity_signals.csv")
    print("\n== 한 번이라도 나온 신호일 (몇 개 조합에서 나왔나 / 이후 수익률) ==")
    print(out.to_string())

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sens":
        sensitivity()
    else:
        main()
