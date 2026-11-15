# -*- coding: utf-8 -*-
"""extend_store.py — 연구 정본(data_2015/)을 필요할 때 늘린다 (케인 2026-11-15, 한 방향 원칙 · 정기화하지 않음).

    python3 extend_store.py --to 2026-11-13                 # 전 단계
    python3 extend_store.py --to 2026-11-13 --steps prices  # 일부만 (prices indices monthly flow deriv)
    python3 extend_store.py --to 2026-11-13 --dry-run       # prices 는 계산만, 나머지 단계는 건너뜀

원천은 전부 KRX 다 — LLV 데이터(raw·패널·options)를 원천으로 쓰지 않는다.
  prices    KRX Open API 일별 원주가 + 기준가 수정계수 (kms_store.extend_prices — 9-14 이후 pykrx 수정가 종가 오염을 피한다)
  indices   pykrx(KRX 사이트) 지수 — index_kospi·benchmark 월말, kospi200_daily (rebuild_data_2015.step_indices)
  monthly   pykrx(KRX 사이트) 월말 meta·foreign·fundamentals (rebuild_data_2015.build_month) + is_managed = 코스닥 KRX 소속부.
            마지막 저장 월은 다시 만든다 (2026-10-31 라벨은 10-29 기준이었다 — DATA_2015.md)
  flow      pykrx(KRX 사이트) 전종목×투자자 순매수 거래대금 (rebuild_data_2015.flow_day).
            ⚠ pykrx 가 막히면 **자동으로 한국투자**(종목별 투자자매매동향, 백만원 반올림 — rebuild_data_2015.step_flow_kis)
  deriv     KRX Open API 파생 일별 (LLV krx_fetcher.fetch_option_daily 를 **라이브러리로만** 호출 — LLV 데이터는 안 읽음).
            KRX drv API 는 평일 08~18시에만 응답한다 (KRX 쪽 제한)
시간대 제한 없음 (케인 2026-11-15). 각 단계는 저장 전에 백업(.bak_YYYYmmdd_HHMMSS)을 남기고 MANIFEST.json history 에 한 줄 적는다.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LLV = HERE.parents[1] / "LongLiveVault"
KST = timezone(timedelta(hours=9))


def _load_env():
    """LLV .env 의 KRX_ID·KRX_PW·KRX_AUTH_KEY (+ KIS 키) 를 환경변수로 (이미 있으면 그대로). 값은 출력하지 않는다."""
    f = LLV / ".env"
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()
sys.path.insert(0, str(HERE))
import kms_store as K                       # noqa: E402
import rebuild_data_2015 as R               # noqa: E402  (import 시 KRX_ID/PW 를 환경에서 치운다 — 위에서 먼저 실어 둔다)

log = K.log
STORE = K.STORE


def _wire(to: pd.Timestamp):
    """rebuild_data_2015 의 전역을 연구 정본·연장 캐시로 돌린다 (원 스크립트의 2026-10-29 고정값을 덮는다)."""
    R.SRC = R.DST = STORE
    R.CACHE = HERE / "_cache_ext"; R.CACHE.mkdir(exist_ok=True)
    R.FLOW_CACHE = R.CACHE / f"flow_kis_{to:%Y%m%d}"; R.FLOW_CACHE.mkdir(exist_ok=True)
    R.START, R.END = pd.Timestamp("2015-01-02"), to


def _save(name: str, df: pd.DataFrame, what: str):
    p = STORE / f"{name}.parquet"
    if p.exists():
        K._backup(p)
    tmp = p.with_suffix(".parquet.tmp"); df.to_parquet(tmp); os.replace(tmp, p)
    K._note(what)
    log(f"  저장 {p.name} — {what}")


# ── monthly ────────────────────────────────────────────────────────────────
def _month_complete(month: pd.Period, to: pd.Timestamp) -> bool:
    me = month.to_timestamp(how="end").normalize()
    if to >= me:
        return True
    return not K.krx_days(to, me)                 # to 이후 그달 남은 거래일이 없으면 완결


def step_monthly(to: pd.Timestamp):
    meta = pd.read_parquet(STORE / "meta.parquet")
    last_lab = meta.index.get_level_values("date").max()
    months = [m for m in pd.period_range(last_lab.to_period("M"), to.to_period("M"), freq="M") if _month_complete(m, to)]
    if not months:
        log("monthly: 새로 완결된 달 없음"); return
    R.krx_login()
    parts = [R.build_month(str(m)) for m in months]
    for k in ("meta", "foreign", "fundamentals"):
        old = pd.read_parquet(STORE / f"{k}.parquet")
        new = pd.concat([p[k] for p in parts])
        if k == "meta":
            new["managed_known"] = new["market"].astype(str).str.upper().eq("KOSDAQ")
            for p in parts:
                lab = p["meta"].index.get_level_values("date")[0]
                sect = K.krx_daily(p["day"], "kosdaq")
                mg = set(sect.loc[sect.Sect.astype(str).str.startswith("관리종목"), "Ticker"]) if not sect.empty else set()
                idx = new.index[(new.index.get_level_values("date") == lab) & new.index.get_level_values("ticker").isin(mg)]
                new.loc[idx, "is_managed"] = True
        new = new[[c for c in old.columns if c in new.columns]]
        for c in old.columns:
            if c in new and str(old[c].dtype) != str(new[c].dtype):
                try:
                    new[c] = new[c].astype(old[c].dtype)
                except Exception:                                   # noqa: BLE001
                    pass
        labs = new.index.get_level_values("date").unique()
        allv = pd.concat([old[~old.index.get_level_values("date").isin(labs)], new]).sort_index()
        assert not allv.index.duplicated().any()
        _save(k, allv, f"monthly {k}: {', '.join(str(m) for m in months)} (pykrx 월 마지막 거래일"
                       + (" · is_managed=코스닥 KRX 소속부)" if k == "meta" else ")"))


# ── flow ───────────────────────────────────────────────────────────────────
def step_flow(to: pd.Timestamp):
    old = pd.read_parquet(STORE / "investor_flow.parquet")
    last_old = old.index.get_level_values("date").max()
    px = pd.read_parquet(STORE / "prices.parquet", columns=["volume"])
    days = sorted(d for d in px.index.get_level_values("date").unique() if last_old < d <= to)
    if not days:
        log("flow: 새 거래일 없음"); return
    try:
        R.krx_login()
        parts = [R.flow_day(d) for d in days]
        src = "pykrx"
    except (Exception, SystemExit) as e:                            # noqa: BLE001
        log(f"  ⚠ pykrx 수급 실패 ({type(e).__name__}: {e}) → 한국투자 대비책으로 전환")
        R.step_flow_kis(budget_sec=1e9)                              # 저장·대조·정지일 제거까지 원 스크립트가 한다
        K._note(f"flow {last_old.date()}→{to.date()}: 한국투자 대비책(pykrx 실패) — 백만원 반올림")
        return
    new = pd.concat(parts)
    tk = set(px.index.get_level_values("ticker").unique())
    new = new[new.index.get_level_values("ticker").isin(tk)]        # 원본 관례: prices 종목만
    pv = px["volume"].reindex(new.index)
    new = new[~(pv.isna() | (pv == 0))]                             # 원본 관례: 거래정지일 행 없음
    allv = pd.concat([old, new]).sort_index()
    assert not allv.index.duplicated().any()
    _save("investor_flow", allv, f"flow {last_old.date()}→{days[-1].date()}: {src} 원 단위, {len(new):,}행")


# ── deriv ──────────────────────────────────────────────────────────────────
def step_deriv(to: pd.Timestamp):
    import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
    sys.path.insert(0, str(LLV))
    from stolab_data.krx_fetcher import fetch_option_daily           # 라이브러리 호출만 (KRX 로 직접 나간다)
    p = STORE / "derivatives_all.parquet"
    base = pq.read_table(p)
    have = pd.Timestamp(pc.max(base["Date"]).as_py())
    days = K.krx_days(have, to)
    parts = []
    for d in days:
        for kind in ("opt", "fut"):
            df = fetch_option_daily(d.strftime("%Y%m%d"), kind)
            if df.empty:
                raise RuntimeError(f"KRX 파생 {kind} {d.date()} 빈 응답 — drv API 는 평일 08~18시에만 응답한다")
            df["Kind"] = kind
            sess = df["Name"].str.extract(r"\((정규|주간|야간)\)\s*$")[0].replace({"주간": "정규"})
            if kind == "opt":
                sess = sess.fillna(pd.Series(np.where(df["IV"] > 0, "정규", "야간"), index=df.index))
            df["Session"] = sess.fillna("미구분")
            df = df[~((df["Volume"] == 0) & (df["OI"] == 0))]
            df["Date"] = pd.to_datetime(df["Date"])
            parts.append(pa.Table.from_pandas(df[R.DERIV_COLS], preserve_index=False).cast(base.schema))
    if not parts:
        log("deriv: 새 거래일 없음"); return
    allv = pa.concat_tables([base] + parts)
    allv = allv.take(pc.sort_indices(allv, sort_keys=[("Date", "ascending"), ("Kind", "ascending"), ("Code", "ascending"), ("Session", "ascending")]))
    K._backup(p)
    pq.write_table(allv, p.with_suffix(".parquet.tmp"), compression="zstd"); os.replace(p.with_suffix(".parquet.tmp"), p)
    K._note(f"deriv {have.date()}→{days[-1].date()}: KRX drv 일별 {len(days)}일")
    log(f"  저장 derivatives_all — {have.date()}→{days[-1].date()} ({len(days)}일)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", help="연장 끝 날짜 YYYY-MM-DD (기본 = 어제 KST)")
    ap.add_argument("--steps", nargs="+", default=["prices", "indices", "monthly", "flow", "deriv"],
                    choices=["prices", "indices", "monthly", "flow", "deriv"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    to = pd.Timestamp(a.to) if a.to else pd.Timestamp(datetime.now(KST).date() - timedelta(days=1))
    _wire(to)
    log(f"연구 정본 연장 → {to.date()} · 단계 {a.steps}{' (dry-run)' if a.dry_run else ''}")
    t0 = time.time()
    if "prices" in a.steps:
        K.extend_prices(to, dry_run=a.dry_run)
    if a.dry_run:
        log("dry-run — prices 외 단계는 건너뜀"); return
    if "indices" in a.steps:
        R.krx_login(); R.step_indices(); K._note(f"indices → {to.date()} (pykrx 지수, 전 구간 재계산)")
    if "monthly" in a.steps:
        step_monthly(to)
    if "flow" in a.steps:
        step_flow(to)
    if "deriv" in a.steps:
        step_deriv(to)
    K.write_manifest(STORE, {})
    log(f"완료 ({int(time.time() - t0)}초) — MANIFEST 갱신")
    log("백업: 월 1회 자동(정기 조정 주말). 지금 올리려면 ~/Dev/StoLab/StoLab-meta/scripts/research_backup_monthly.sh --force")


if __name__ == "__main__":
    main()
