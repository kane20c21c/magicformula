# -*- coding: utf-8 -*-
"""backfill_prices_krx.py — data_ext 를 2026-07-01 ~ 2026-09-11 까지 전 유니버스로 백필 (pykrx 종목별 OHLCV)

왜: 정본 data/prices.parquet 는 2026-06-30 까지 2,796종목, build_data_ext 연장분은 LLV 208종목뿐이라
    7월부터 유니버스가 끊긴다. pykrx 종목별 일별 OHLCV 는 살아 있어(전종목 스냅샷·수급은 KRX 차단) 종목별로 받는다.
    pykrx 거래량은 KRX 단독 → 6월까지의 원천과 같아 거래량 이음새 문제도 해결.

하는 일:
  1) 2026-06 에 종가가 있던 종목 전부를 대상으로 pykrx.get_market_ohlcv (수정주가; 원주가 endpoint 차단) 20260630~END 호출
     (06-30 겹치는 하루로 이음새 검사: 종가 비율이 5% 넘게 다르면 새 구간을 비율로 스케일 — build_data_ext 와 같은 규칙)
  2) data_ext/prices.parquet 의 2026-07-01 이후 행(LLV 연장분)을 버리고 pykrx 행으로 교체
  3) data_ext/meta.parquet 을 7·8·9월로 이월: mktcap = 6월 mktcap × (월말 종가 / 6월말 종가), 나머지 컬럼은 6월 값 그대로
  4) 진행 캐시 data_ext/_krx_cache/<ticker>.parquet — 중단 후 재실행하면 이어서 받음
  5) 백업: data_ext/prices.parquet.bak_<날짜>, meta.parquet.bak_<날짜>

수급(investor_flow)은 KRX 차단으로 6/30 에 멈춘 채 둔다.
실행: python3 backfill_prices_krx.py [--end 20260911] [--sleep 0.15]
"""
import argparse, time, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
from pykrx import stock

HERE = Path(__file__).resolve().parent
EXT = HERE / "data_ext"
CACHE = EXT / "_krx_cache"
SEAM = "2026-06-30"
COLS = {"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch(ticker, start, end, sleep):
    f = CACHE / f"{ticker}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    for k in range(3):
        try:
            d = stock.get_market_ohlcv(start, end, ticker)   # 수정주가 (원주가 endpoint 는 KRX 차단)
            time.sleep(sleep)
            if d is None or len(d) == 0:
                d = pd.DataFrame(columns=list(COLS.values()))
            else:
                d = d.rename(columns=COLS)[list(COLS.values())].astype(float)
                d.index = pd.to_datetime(d.index); d.index.name = "date"
            d.to_parquet(f)
            return d
        except Exception as e:
            log(f"  {ticker} retry {k+1}: {repr(e)[:80]}"); time.sleep(2 + 3 * k)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default="20260911")
    ap.add_argument("--sleep", type=float, default=0.15)
    a = ap.parse_args()
    CACHE.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d")
    px = pd.read_parquet(EXT / "prices.parquet")
    d = px.index.get_level_values("date")
    base = px[d <= SEAM]
    june = base[(base.index.get_level_values("date") >= "2026-06-01") & base.close.notna()]
    tickers = sorted(june.index.get_level_values("ticker").unique())
    log(f"대상 {len(tickers)}종목 · {SEAM} 이후 기존 행 {int((d > SEAM).sum()):,}개 교체 예정")
    seam_close = base.xs(SEAM, level="date").close if SEAM in base.index.get_level_values("date") else None

    rows, bad, scaled = [], [], []
    t0 = time.time()
    for i, tk in enumerate(tickers, 1):
        d_ = fetch(tk, "20260630", a.end, a.sleep)
        if d_ is None:
            bad.append(tk); continue
        d_ = d_[d_.close > 0]
        if len(d_) == 0:
            continue
        if seam_close is not None and pd.Timestamp(SEAM) in d_.index and tk in seam_close.index and seam_close[tk] > 0:
            ratio = seam_close[tk] / d_.loc[SEAM, "close"]
            if abs(ratio - 1) > 0.05:
                d_[["open", "high", "low", "close"]] *= ratio
                d_["volume"] /= ratio
                scaled.append((tk, round(ratio, 4)))
        d_ = d_[d_.index > SEAM]
        d_["ticker"] = tk
        rows.append(d_.reset_index())
        if i % 200 == 0:
            log(f"{i}/{len(tickers)} ({time.time()-t0:.0f}s)")
    new = pd.concat(rows, ignore_index=True).set_index(["date", "ticker"]).sort_index()
    log(f"수신 {len(new):,}행 · 실패 {len(bad)} · 이음새 스케일 {len(scaled)}: {scaled[:10]}")

    if not (EXT / f"prices.parquet.bak_{stamp}").exists():
        shutil.copy(EXT / "prices.parquet", EXT / f"prices.parquet.bak_{stamp}")
    out = pd.concat([base, new[[c for c in base.columns if c in new.columns]]]).sort_index()
    for c in base.columns:
        if c not in out.columns:
            out[c] = np.nan
    out = out[base.columns]
    out.to_parquet(EXT / "prices.parquet")
    dd = out.index.get_level_values("date")
    log(f"prices.parquet 저장 {out.shape} · 끝 {dd.max().date()} · 7월 이후 종목 수 {out[dd > SEAM].index.get_level_values('ticker').nunique()}")

    # meta 이월 — 정본 meta 는 전 종목 2026-05-31, 6/30 은 LLV 207종목만. 종목별 "마지막 월말 행"에서 6/30·7/31·8/31 로 이월,
    #            mktcap 은 그 월말 종가 대비 비율로 스케일, 나머지 컬럼은 그대로.
    meta = pd.read_parquet(EXT / "meta.parquet")
    if not (EXT / f"meta.parquet.bak_{stamp}").exists():
        shutil.copy(EXT / "meta.parquet", EXT / f"meta.parquet.bak_{stamp}")
    md = meta.index.get_level_values("date")
    meta = meta[md <= SEAM].sort_index()
    last = meta.groupby(level="ticker").tail(1)
    base_dt = pd.Series(last.index.get_level_values("date"), index=last.index.get_level_values("ticker"))
    log(f"meta 종목별 마지막 월말 분포: {base_dt.value_counts().sort_index().tail(3).to_dict()}")
    cl = out.close.unstack("ticker").ffill()
    ends = [pd.Timestamp(SEAM)] + [ts for ts in pd.date_range(SEAM, a.end, freq="ME") if ts > pd.Timestamp(SEAM)]  # 6/30, 7/31, 8/31
    add = []
    for ts in ends:
        tks = base_dt[base_dt < ts].index
        tks = [t for t in tks if t in cl.columns]
        if not tks:
            continue
        r = last.reset_index(level="date", drop=True).loc[tks].copy()
        c_now = cl.loc[:ts].iloc[-1].reindex(tks)
        c_base = pd.Series(np.nan, index=tks)
        for bd in base_dt.loc[tks].unique():
            row = cl.loc[:bd].iloc[-1]
            sel = [t for t in tks if base_dt[t] == bd]
            c_base.loc[sel] = row.reindex(sel).values
        ratio = (c_now / c_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        r["mktcap"] = r["mktcap"] * ratio.values
        r["date"] = ts
        add.append(r.reset_index().set_index(["date", "ticker"]))
    meta2 = pd.concat([meta] + add).sort_index()
    meta2 = meta2[~meta2.index.duplicated(keep="first")]
    meta2.to_parquet(EXT / "meta.parquet")
    log(f"meta.parquet 저장 {meta2.shape} · 월별 종목 수 {meta2.groupby(level='date').size().tail(4).to_dict()}")
    if bad:
        (EXT / f"_krx_failed_{stamp}.txt").write_text("\n".join(bad))
        log(f"실패 종목 {len(bad)}개 → _krx_failed_{stamp}.txt")
    log(f"완료 ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
