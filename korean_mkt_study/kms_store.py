# -*- coding: utf-8 -*-
"""kms_store.py — 연구 정본 데이터 세트(data_2015/) 관리 도구 (케인 2026-11-15, 한 방향 원칙).

연구 데이터는 **KRX 에서 직접** 받는다 (LLV raw·패널을 원천으로 쓰지 않는다 — 운영과 연구를 분리).
  · KRX Open API (data-dbg) — 일별 시세(원주가, 정규장)·시총·상장주식수·소속부·지수. 날짜당 시장별 1콜.
  · pykrx (KRX 사이트) — 외국인 보유·펀더멘털·투자자 순매수 (Open API 에 없다).
  · 한국투자 — pykrx 가 막히면 투자자 수급 대비책 (rebuild_data_2015.step_flow_kis 재사용), 종목 마스터(관리종목 현재값).
시간대 제한 없음 (케인 2026-11-15 — 연구도 낮에 한다). KRX Open API 키는 LLV .env 의 KRX_AUTH_KEY 를 읽기만 한다.

서브커맨드:
  fix-regular   2026-09-14(KRX 애프터마켓 개장) ~ 세트 끝의 고가·저가·종가를 KRX 정규장 값으로 교정.
                pykrx 수정주가가 이 구간 종가에 애프터마켓 가격을 섞어 준다 (LLV CLAUDE.md '백필 가격' 절과 같은 사고).
                실측(2026-11-15): 10-21 종가 일치 14% · 시가 99.9% · 거래량 99.9% → 시가 비율(저장 시가 ÷ KRX 시가)로
                권리 조정을 유지한 채 고가·저가·종가 = KRX × 비율. 시가·거래량은 그대로.
  managed       meta.is_managed 를 코스닥 KRX 소속부 '관리종목(소속부없음)' 으로 채운다 (월 마지막 거래일 기준, 2015~).
                코스피는 KRX 에 이 칸이 없어(전부 빈칸) 채울 수 없다 → managed_known=False 로 남긴다.
                ⚠ 소속부는 한 칸 라벨이라 관리+투자주의환기가 '환기'로 보이는 경우가 있다(2026-11-13 4건) — 하한값.
  manifest      세트별 MANIFEST.json 갱신 (기간·종목 수·행 수·파일 크기).

사용:  python3 kms_store.py fix-regular [--dry-run]
       python3 kms_store.py managed [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STORE = HERE / "data_2015"                      # 연구 정본 (2015-01-02 ~) — kms_data.SETS["current"]
CACHE = HERE / "_cache_krx"                     # KRX Open API 응답 캐시 (날짜·시장별, 재조회 방지)
LLV_ENV = HERE.parents[1] / "LongLiveVault" / ".env"
AFTERMARKET_OPEN = pd.Timestamp("2026-09-14")
EVENT_TOL = 0.003     # 기준가 ÷ 전일 종가 가 이보다 1 에서 멀면 권리락(수정 사건)으로 본다

_URL = {"kospi": "https://data-dbg.krx.co.kr/svc/apis/sto/stk_bydd_trd",
        "kosdaq": "https://data-dbg.krx.co.kr/svc/apis/sto/ksq_bydd_trd"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _krx_key() -> str:
    k = os.environ.get("KRX_AUTH_KEY")
    if k:
        return k
    for line in LLV_ENV.read_text(encoding="utf-8").splitlines():
        if line.startswith("KRX_AUTH_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("KRX_AUTH_KEY 없음 (환경변수 또는 LongLiveVault/.env)")


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).replace({"": None, "-": None}), errors="coerce")


def krx_daily(day, market: str, retries: int = 3) -> pd.DataFrame:
    """KRX Open API 일별 시세 (원주가). 캐시 우선. 열: Ticker Name Market Sect Open High Low Close Base Volume Amount MarketCap ListShrs."""
    import requests
    d = pd.Timestamp(day).strftime("%Y%m%d")
    CACHE.mkdir(exist_ok=True)
    cf = CACHE / f"{market}_{d}.parquet"
    if cf.exists():
        return pd.read_parquet(cf)
    last = None
    for i in range(retries):
        try:
            r = requests.get(_URL[market], headers={"AUTH_KEY": _krx_key()}, params={"basDd": d}, timeout=40)
            r.raise_for_status()
            rows = r.json().get("OutBlock_1", [])
            break
        except Exception as e:                                # noqa: BLE001
            last = e
            time.sleep(10 * (i + 1))
    else:
        raise RuntimeError(f"KRX {market} {d} 실패: {last}")
    if not rows:
        return pd.DataFrame()
    j = pd.DataFrame(rows)
    out = pd.DataFrame({
        "Ticker": j["ISU_CD"].astype(str).str.zfill(6), "Name": j["ISU_NM"], "Market": j["MKT_NM"],
        "Sect": j.get("SECT_TP_NM", pd.Series("", index=j.index)).fillna(""),
        "Open": _num(j["TDD_OPNPRC"]), "High": _num(j["TDD_HGPRC"]), "Low": _num(j["TDD_LWPRC"]),
        "Close": _num(j["TDD_CLSPRC"]), "Change": _num(j["CMPPREVDD_PRC"]),
        "Volume": _num(j["ACC_TRDVOL"]), "Amount": _num(j["ACC_TRDVAL"]),
        "MarketCap": _num(j["MKTCAP"]), "ListShrs": _num(j["LIST_SHRS"]),
    })
    out["Base"] = out["Close"] - out["Change"]                # 기준가 (권리락 반영) — 연장 도구의 수정계수 재료
    out.to_parquet(cf, index=False)
    return out


def _backup(p: Path) -> Path:
    b = p.with_name(p.name + f".bak_{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(p, b)
    return b


# ── fix-regular ────────────────────────────────────────────────────────────
def fix_regular(dry_run: bool = False, start=AFTERMARKET_OPEN) -> dict:
    p = STORE / "prices.parquet"
    px = pd.read_parquet(p)
    dates = sorted(d for d in px.index.get_level_values("date").unique() if d >= pd.Timestamp(start))
    log(f"fix-regular: {STORE.name}/prices {len(dates)}거래일 ({dates[0].date()} ~ {dates[-1].date()})")
    stats = {"days": len(dates), "rows": 0, "changed_close": 0, "no_krx": 0, "zero_open": 0}
    px = px.copy()
    for d in dates:
        k = pd.concat([krx_daily(d, m) for m in ("kospi", "kosdaq")], ignore_index=True).drop_duplicates("Ticker")
        k = k.set_index("Ticker")
        idx = px.index[px.index.get_level_values("date") == d]
        sub = px.loc[idx]
        tk = sub.index.get_level_values("ticker")
        kk = k.reindex(tk)
        ok = kk["Open"].notna().to_numpy() & (kk["Open"].to_numpy() > 0) & (sub["open"].to_numpy() > 0)
        stats["no_krx"] += int(kk["Open"].isna().sum())
        stats["zero_open"] += int((kk["Open"].fillna(-1).to_numpy() == 0).sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(ok, sub["open"].to_numpy() / kk["Open"].to_numpy(), np.nan)
        new_close = np.where(ok, kk["Close"].to_numpy() * ratio, sub["close"].to_numpy())
        stats["changed_close"] += int((np.abs(new_close - sub["close"].to_numpy()) > 1e-6 * np.abs(sub["close"].to_numpy())).sum())
        stats["rows"] += int(ok.sum())
        for col, kc in (("high", "High"), ("low", "Low"), ("close", "Close")):
            px.loc[idx, col] = np.where(ok, kk[kc].to_numpy() * ratio, sub[col].to_numpy())
    log(f"  교정 대상 {stats['rows']:,}행 · 종가가 바뀐 행 {stats['changed_close']:,} · KRX 에 없음 {stats['no_krx']} · KRX 시가 0 {stats['zero_open']}")
    if not dry_run:
        b = _backup(p)
        tmp = p.with_suffix(".parquet.tmp"); px.to_parquet(tmp); os.replace(tmp, p)
        log(f"  저장 {p} (백업 {b.name})")
        _note(f"fix-regular {dates[0].date()}~{dates[-1].date()}: 고가·저가·종가 = KRX 정규장 × 시가 비율, {stats['rows']:,}행 (종가 변경 {stats['changed_close']:,})")
    return stats


# ── extend-prices (KRX 원주가 + 기준가 수정계수) ──────────────────────────────
def krx_days(start, end) -> list:
    """start 다음 날 ~ end 의 평일 중 KRX 가 응답한(거래일) 날짜. 빈 응답 = 휴장."""
    out = []
    for d in pd.bdate_range(pd.Timestamp(start) + pd.Timedelta(days=1), pd.Timestamp(end)):
        if not krx_daily(d, "kospi").empty:
            out.append(d)
    return out


def adjust_extension(store_px: pd.DataFrame, raw: pd.DataFrame, seam: pd.Timestamp, raw_seam: pd.DataFrame):
    """저장 수정주가(~seam) 뒤에 KRX 원주가(raw, seam 다음 날~)를 이어 붙일 때의 수정계수.

    계수 f_t = 기준가_t ÷ 전일 원종가 (권리락이 없으면 정확히 1). 결과:
      · 새 행 t   = 원주가 × Π(t 이후 계수)        · 거래량 ÷ 같은 값
      · 저장 이력 = 저장값 × Π(새 구간 전체 계수) ÷ s   (s = 저장 seam 종가 ÷ 원 seam 종가 — 저장본이 이미 반영한
                    seam 이후 권리락을 두 번 곱하지 않게)
    raw 열: date Ticker Open High Low Close Volume Base.  반환: (new_rows[date,ticker 색인], hist_mult{ticker: 배수})
    """
    raw = raw.sort_values(["Ticker", "date"])
    seam_close = raw_seam.set_index("Ticker")["Close"] if not raw_seam.empty else pd.Series(dtype=float)
    st_seam = store_px.xs(seam, level="date")["close"] if seam in store_px.index.get_level_values("date") else pd.Series(dtype=float)
    new_parts, hist_mult = [], {}
    for tk, g in raw.groupby("Ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        prev = np.r_[seam_close.get(tk, np.nan), g["Close"].to_numpy()[:-1]]
        base = g["Base"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where((prev > 0) & (base > 0), base / prev, 1.0)
        f = np.where(np.abs(f - 1) > EVENT_TOL, f, 1.0)       # 0.3% 미만 = 잡음으로 본다 (스팩 기준가 0.15% 차 실측)
        after = np.r_[np.cumprod(f[::-1])[::-1][1:], 1.0]          # Π_{j>i} f_j
        total = float(np.prod(f))
        m = after
        g2 = g.copy()
        for c in ("Open", "High", "Low", "Close"):
            g2[c] = g[c].to_numpy(dtype=float) * m
        g2["Volume"] = g["Volume"].to_numpy(dtype=float) / m
        new_parts.append(g2)
        rs, ss = seam_close.get(tk, np.nan), st_seam.get(tk, np.nan)
        sfac = (ss / rs) if (rs and rs > 0 and ss and ss > 0) else 1.0
        hm = total / sfac
        if abs(hm - 1) > EVENT_TOL:                             # pykrx 수정가 반올림 잔차(~0.03%)로 전 이력을 흔들지 않는다
            hist_mult[tk] = hm
    new = pd.concat(new_parts, ignore_index=True) if new_parts else pd.DataFrame(columns=raw.columns)
    new = new.rename(columns={"Ticker": "ticker", "Open": "open", "High": "high", "Low": "low", "Close": "close",
                              "Volume": "volume"})
    new = new.set_index(["date", "ticker"])[["open", "high", "low", "close", "volume"]].sort_index()
    return new, hist_mult


def extend_prices(to, dry_run: bool = False) -> dict:
    """연구 정본 prices 를 to 까지 KRX Open API 로 연장. 종목 = 기존 종목 + 새 구간에 처음 나온 신규상장(원본 관례)."""
    p = STORE / "prices.parquet"
    px = pd.read_parquet(p)
    seam = px.index.get_level_values("date").max()
    days = krx_days(seam, to)
    if not days:
        log(f"extend-prices: {seam.date()} 이후 {pd.Timestamp(to).date()} 까지 새 거래일 없음"); return {"days": 0}
    known = set(px.index.get_level_values("ticker").unique())
    raw_seam = pd.concat([krx_daily(seam, m) for m in ("kospi", "kosdaq")]).drop_duplicates("Ticker")
    seam_tk = set(raw_seam.Ticker)
    frames = []
    for d in days:
        k = pd.concat([krx_daily(d, m) for m in ("kospi", "kosdaq")]).drop_duplicates("Ticker")
        k["date"] = d
        frames.append(k)
    raw = pd.concat(frames, ignore_index=True)
    first_seen = raw.groupby("Ticker")["date"].min()
    newly = {t for t in first_seen.index if t not in known and t not in seam_tk}
    raw = raw[raw.Ticker.isin(known | newly)]
    new, hm = adjust_extension(px, raw[["date", "Ticker", "Open", "High", "Low", "Close", "Volume", "Base"]],
                               seam, raw_seam[raw_seam.Ticker.isin(known)])
    px2 = px.copy()
    if hm:
        tk = px2.index.get_level_values("ticker")
        mult = tk.map(hm).fillna(1.0).to_numpy()
        for c in ("open", "high", "low", "close"):
            px2[c] = px2[c].to_numpy() * mult
        px2["volume"] = px2["volume"].to_numpy() / mult
    allv = pd.concat([px2, new]).sort_index()
    assert not allv.index.duplicated().any()
    st = {"seam": str(seam.date()), "days": len(days), "end": str(days[-1].date()), "new_rows": len(new),
          "new_tickers": sorted(newly), "hist_adjusted": {k: round(v, 6) for k, v in sorted(hm.items())}}
    log(f"extend-prices: {seam.date()} → {days[-1].date()} ({len(days)}거래일) · 새 행 {len(new):,} · 신규상장 {len(newly)} · "
        f"과거 수정(권리락) {len(hm)}종목 {list(st['hist_adjusted'].items())[:5]}")
    if not dry_run:
        b = _backup(p)
        tmp = p.with_suffix(".parquet.tmp"); allv.to_parquet(tmp); os.replace(tmp, p)
        log(f"  저장 {p} (백업 {b.name})")
        _note(f"extend-prices {seam.date()}→{days[-1].date()}: KRX 원주가+기준가 수정계수, 새 행 {len(new):,}, 신규상장 {len(newly)}, 과거 수정 {len(hm)}종목")
    return st


# ── managed ────────────────────────────────────────────────────────────────
def month_last_trading_days() -> dict:
    px_dates = pd.read_parquet(STORE / "prices.parquet", columns=["close"]).index.get_level_values("date").unique()
    s = pd.Series(sorted(px_dates))
    return {k: v for k, v in s.groupby(s.dt.to_period("M")).max().items()}


def fill_managed(dry_run: bool = False) -> dict:
    p = STORE / "meta.parquet"
    meta = pd.read_parquet(p)
    labels = sorted(meta.index.get_level_values("date").unique())
    ltd = month_last_trading_days()
    meta = meta.copy()
    meta["is_managed"] = False
    meta["managed_known"] = meta["market"].astype(str).str.upper().eq("KOSDAQ")
    counts = {}
    for lab in labels:
        per = pd.Timestamp(lab).to_period("M")
        day = ltd.get(per)
        if day is None:
            continue
        k = krx_daily(day, "kosdaq")
        if k.empty:
            log(f"  {per}: KRX 빈 응답 — 건너뜀"); continue
        mg = set(k.loc[k.Sect.astype(str).str.startswith("관리종목"), "Ticker"])
        idx = meta.index[(meta.index.get_level_values("date") == lab)
                         & meta.index.get_level_values("ticker").isin(mg)]
        meta.loc[idx, "is_managed"] = True
        counts[str(per)] = (len(mg), len(idx))
    tot = sum(v[1] for v in counts.values())
    log(f"managed: {len(counts)}개월 · 코스닥 관리 표시 {tot:,}행 (월별 KRX/meta 매칭 예: "
        f"{list(counts.items())[:2]} … {list(counts.items())[-2:]})")
    if not dry_run:
        b = _backup(p)
        tmp = p.with_suffix(".parquet.tmp"); meta.to_parquet(tmp); os.replace(tmp, p)
        log(f"  저장 {p} (백업 {b.name})")
        _note(f"managed: is_managed = 코스닥 KRX 소속부(월 마지막 거래일) {len(counts)}개월 {tot:,}행 · 코스피 managed_known=False")
    return counts


# ── MANIFEST ───────────────────────────────────────────────────────────────
def _note(text: str):
    """data_2015/MANIFEST.json 의 history 에 한 줄 남긴다."""
    mf = STORE / "MANIFEST.json"
    j = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {}
    j.setdefault("history", []).append({"at": datetime.now().strftime("%Y-%m-%d %H:%M"), "by": "kms_store.py", "what": text})
    mf.write_text(json.dumps(j, ensure_ascii=False, indent=1), encoding="utf-8")


def file_stats(p: Path) -> dict:
    import pyarrow.parquet as pq
    sch = pq.read_schema(p)
    names = list(sch.names)
    out = {"rows": pq.ParquetFile(p).metadata.num_rows, "mb": round(p.stat().st_size / 1e6, 1), "columns": names}
    dcol = next((c for c in ("date", "Date") if c in names), None)
    tcol = next((c for c in ("ticker", "Code") if c in names), None)
    if dcol:
        t = pq.read_table(p, columns=[dcol] + ([tcol] if tcol else [])).to_pandas().reset_index()
        dd = pd.to_datetime(t[dcol])
        out.update(start=str(dd.min().date()), end=str(dd.max().date()), n_dates=int(dd.nunique()))
        if tcol:
            out["n_tickers"] = int(t[tcol].nunique())
    return out


def write_manifest(folder: Path, meta: dict) -> Path:
    mf = folder / "MANIFEST.json"
    old = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {}
    files = {f.name: file_stats(f) for f in sorted(folder.glob("*.parquet"))}
    j = {**old, **meta, "files": files, "manifest_updated": datetime.now().strftime("%Y-%m-%d %H:%M")}
    mf.write_text(json.dumps(j, ensure_ascii=False, indent=1), encoding="utf-8")
    return mf


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fix-regular", "managed", "manifest"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "fix-regular":
        fix_regular(a.dry_run)
    elif a.cmd == "managed":
        fill_managed(a.dry_run)
    else:
        import kms_data
        for name, spec in kms_data.SETS.items():
            if spec["path"].exists():
                print(write_manifest(spec["path"], {k: v for k, v in spec.items() if k != "path"}))


if __name__ == "__main__":
    main()
