# -*- coding: utf-8 -*-
"""rebuild_data_2015.py — 스윙 연구 데이터 세트를 기간 2015-01-02 ~ 2026-10-29 로 재구성해 data_2015/ 에 만든다.
(케인 지시 2026-10-30. 정본 data/ · 연장본 data_ext/ 는 건드리지 않는다.)

원천 정의는 data/ 저장본과 pykrx 를 대조해 역추적한 것이다 (원 생성 스크립트는 "Kane repo" 에 있어 이 리포엔 없다):
  prices          pykrx 종목별 수정주가 OHLCV. 연장은 이음새(마지막 공통일) 종가 대조 → 불일치(분할 등) 시 과거 구간을 비율 보정.
  investor_flow   pykrx 전종목×투자자 일별 **순매수 거래대금(원)** — 개인/기관(기관합계)/외국인(외국인+기타외국인)/기타법인.
  meta            월 마지막 거래일 get_market_cap(시총·거래대금·상장주식수) + 외국인 지분율/100 + 한도소진률/100,
                  market(KOSPI/KOSDAQ), is_pref(이름 끝 우/우B/우C), is_spac(이름에 스팩), is_managed=False(원본 전구간 미수집),
                  listing_date=NaT, sector='기타' (원본 자리표시자 유지). 날짜 라벨 = 달력 월말.
  foreign         월말 외국인 보유수량/상장주식수/한도수량 + 지분율(%)·한도소진률(%).
                  ⚠ 원본 지분율은 pykrx float16 잔재(48.28125 류). 새 행은 보유/상장 으로 정확히 계산 (차이 ≤0.002%p).
  fundamentals    월말 get_market_fundamental(BPS PER PBR EPS DIV DPS) + ROE = EPS/BPS (BPS≤0 → NaN).
  index_kospi     KOSPI(1001) 월말 종가 + 월수익률.   benchmark  KOSPI200(1028) 월수익률.  (전구간 재생성 — 원본 2026-06 행이 월중 값으로 틀려 있었다)
  kospi200_daily  KOSPI200 일봉, FDR KS200 포맷(Close UpDown Comp Change Open High Low Volume Amount MarCap). 전구간 재생성.
  derivatives_all LLV 통합본 — 미니에서 복사해 합친다 (별도 단계).
  dart_factors · universe_snapshots  생성 규칙 미상 — 케인이 규칙을 주면 재생성 (아직 미처리).

사용:  PYTHONPATH=... python3 rebuild_data_2015.py monthly | indices | flow | prices | trim | report
       (각 단계는 멱등 — 캐시 _cache_2015/ 에 중간 결과를 두고 이어서 돈다)
실행 환경: LLV .env 의 KRX_ID/KRX_PW 가 환경변수로 실려 있어야 pykrx 로그인이 된다.
"""
from __future__ import annotations
import sys, time, json, re, warnings
from pathlib import Path
from datetime import date
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
SRC, EXT, DST = HERE / "data", HERE / "data_ext", HERE / "data_2015"
CACHE = HERE / "_cache_2015"; CACHE.mkdir(exist_ok=True); DST.mkdir(exist_ok=True)
START, END = pd.Timestamp("2015-01-02"), pd.Timestamp("2026-10-29")
NEW_MONTHS = ["2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]   # data/ 월말 스냅샷은 2026-05 까지

# ─────────────────────────── KRX 로그인 세션 재사용 ───────────────────────────
# pykrx(KRX 로그인 포크)는 import 시점에 KRX_ID/KRX_PW 로 로그인한다. 프로세스마다 새로 로그인하면 KRX 가
# 로그인 엔드포인트를 잠시 막는다(2026-10-30 실측: 10분간 ~10회 → HTML 에러페이지). 그래서 자격증명을
# import 전에 잠시 치우고, 로그인 쿠키를 파일에 저장해 55분간 재사용한다.
import os
_KRX_ID, _KRX_PW = os.environ.pop("KRX_ID", None), os.environ.pop("KRX_PW", None)
SESSION_FILE = Path(os.environ.get("KRX_SESSION_FILE", "/tmp/krx_session.json"))

def krx_login(max_age_sec: int = 55 * 60):
    from pykrx.website.comm import auth
    if SESSION_FILE.exists():
        try:
            j = json.loads(SESSION_FILE.read_text())
            if time.time() - j["login_time"] < max_age_sec:
                k = auth.KRXSession(); k.cookies = j["cookies"]; k.login_time = j["login_time"]; k.expiry_time = j["login_time"] + 3600; k.is_authenticated = True
                for name, info in j["cookies"].items():
                    k.session.cookies.set(name, info["value"], domain=info.get("domain") or None, path=info.get("path") or "/")
                auth.set_auth_session(k); log(f"KRX 세션 재사용 (로그인 후 {int(time.time()-j['login_time'])}초)"); return k
        except Exception as e:
            log(f"세션 파일 무시: {e}")
    if not (_KRX_ID and _KRX_PW):
        raise SystemExit("KRX_ID/KRX_PW 환경변수 필요")
    k = auth.KRXSession()
    try:
        ok = k.refresh(_KRX_ID, _KRX_PW)
    except Exception as e:
        ok = False; log(f"KRX 로그인 응답 이상: {type(e).__name__}")
    if not ok:
        raise SystemExit("KRX 로그인 실패 — 로그인 엔드포인트가 잠시 막힌 상태일 수 있다. 몇 분 뒤 다시 시도")
    SESSION_FILE.write_text(json.dumps({"login_time": k.login_time, "cookies": k.cookies}))
    auth.set_auth_session(k); log("KRX 새 로그인 → 세션 저장"); return k

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def ymd(ts): return pd.Timestamp(ts).strftime("%Y%m%d")

def last_trading_day(month: str) -> pd.Timestamp:
    """달력 월말 이전 마지막 거래일 (기간 끝 END 를 넘지 않음)."""
    from pykrx import stock
    me = min(pd.Timestamp(month) + pd.offsets.MonthEnd(0), END)
    return pd.Timestamp(stock.get_nearest_business_day_in_a_week(ymd(me), prev=True))

def month_label(month: str) -> pd.Timestamp:
    return pd.Timestamp(month) + pd.offsets.MonthEnd(0)

def listing(day: pd.Timestamp) -> pd.DataFrame:
    """해당일 KOSPI+KOSDAQ 상장 명부 (KONEX 제외) — ticker, name, market."""
    from pykrx import stock
    # 종목명은 그 시점 사명이어야 한다 (is_pref·is_spac 판정 근거). get_market_ticker_name 은 오늘 기준 사명이라
    # 2026-05 재현에서 33건이 어긋났다 → 같은 날짜의 전종목 수급 응답(종목명 포함)에서 가져온다.
    frames = []
    for mk in ("KOSPI", "KOSDAQ"):
        tl = stock.get_market_ticker_list(ymd(day), market=mk)
        nm = stock.get_market_net_purchases_of_equities(ymd(day), ymd(day), mk, "개인")["종목명"]
        d = pd.DataFrame({"name": nm.reindex(tl)}, index=pd.Index(tl, name="ticker")); d["market"] = mk
        miss = d["name"].isna()
        if miss.any():
            d.loc[miss, "name"] = [stock.get_market_ticker_name(t) for t in d.index[miss]]
        frames.append(d)
    return pd.concat(frames)[~pd.concat(frames).index.duplicated()]

# ─────────────────────────── monthly: meta · foreign · fundamentals ───────────────────────────
def build_month(month: str):
    from pykrx import stock
    f = CACHE / f"month_{month}.pkl"
    if f.exists():
        return pd.read_pickle(f)
    day = last_trading_day(month); lab = month_label(month)
    log(f"[{month}] 기준 거래일 {day.date()} → 라벨 {lab.date()}")
    lst = listing(day)
    cap = stock.get_market_cap(ymd(day), market="ALL")
    exh = stock.get_exhaustion_rates_of_foreign_investment(ymd(day), market="ALL")
    fnd = stock.get_market_fundamental(ymd(day), market="ALL")
    cap, exh, fnd = (x[x.index.isin(lst.index)] for x in (cap, exh, fnd))
    # meta
    meta = pd.DataFrame(index=cap.index)
    meta["mktcap"] = cap["시가총액"].astype("int64"); meta["value"] = cap["거래대금"].astype("int64"); meta["shares"] = cap["상장주식수"].astype("int64")
    ratio = (exh["보유수량"] / exh["상장주식수"] * 100).where(exh["상장주식수"] > 0)
    exhr = (exh["보유수량"] / exh["한도수량"] * 100).where(exh["한도수량"] > 0, 0.0)   # 한도수량 0 → 원본 관례 0.0
    meta["foreign"] = (ratio / 100).reindex(meta.index)
    meta["market"] = lst["market"].reindex(meta.index)
    name = lst["name"].reindex(meta.index)
    meta["is_pref"] = name.str.contains(r"우(?:B|C)?$", regex=True).fillna(False)
    meta["is_spac"] = name.str.contains("스팩").fillna(False)
    meta["is_managed"] = False
    meta["listing_date"] = pd.NaT
    meta["sector"] = "기타"
    meta["foreign_exhaustion"] = (exhr / 100).reindex(meta.index)
    meta["name"] = name
    # foreign
    fo = pd.DataFrame({"foreign_ratio": ratio, "exhaustion": exhr,
                       "foreign_shares": exh["보유수량"].astype("int64"), "listed_shares": exh["상장주식수"].astype("int64"),
                       "limit_shares": exh["한도수량"].astype("int64")})
    # fundamentals
    fu = fnd[["BPS", "PER", "PBR", "EPS", "DIV", "DPS"]].copy()
    fu["BPS"] = fu["BPS"].astype("int32"); fu["EPS"] = fu["EPS"].astype("int32"); fu["DPS"] = fu["DPS"].astype("int32")
    fu["ROE"] = np.where(fu["BPS"] > 0, fu["EPS"] / fu["BPS"].replace(0, np.nan), np.nan)
    out = {}
    for k, d in (("meta", meta), ("foreign", fo), ("fundamentals", fu)):
        d = d.copy(); d.index.name = "ticker"; d["date"] = lab
        out[k] = d.reset_index().set_index(["date", "ticker"]).sort_index()
    out["day"] = day
    pd.to_pickle(out, f)
    log(f"[{month}] meta {len(meta)} / foreign {len(fo)} / fundamentals {len(fu)} 종목")
    return out

def step_monthly():
    parts = [build_month(m) for m in NEW_MONTHS]
    for k in ("meta", "foreign", "fundamentals"):
        old = pd.read_parquet(SRC / f"{k}.parquet")
        old = old[old.index.get_level_values("date") >= START]
        new = pd.concat([p[k] for p in parts])
        # 컬럼 순서·dtype 을 원본에 맞춘다
        new = new[old.columns]
        for c in old.columns:
            if str(old[c].dtype) != str(new[c].dtype):
                try: new[c] = new[c].astype(old[c].dtype)
                except Exception as e: log(f"  dtype 유지 실패 {k}.{c}: {old[c].dtype} vs {new[c].dtype} ({e})")
        # 원본이 이미 가진 달(foreign 은 2026-06 까지 있다)은 새 계산값으로 교체 — 교체 전 차이를 기록
        ov = old.index.get_level_values("date").isin(new.index.get_level_values("date").unique())
        if ov.any():
            o, n = old[ov], new.reindex(old[ov].index)
            num = [c for c in o.columns if o[c].dtype.kind in "fiu"]
            mism = {c: int((~np.isclose(o[c].astype(float), n[c].astype(float), rtol=1e-9, atol=0.005, equal_nan=True)).sum()) for c in num}
            log(f"  {k}: 원본과 겹치는 {ov.sum()}행({sorted(set(o.index.get_level_values('date').date))}) 을 새 값으로 교체 — 숫자컬럼 불일치 {mism}, 새 값에 없는 종목 {int(n[num[0]].isna().sum() - o[num[0]].isna().sum())}")
            old = old[~ov]
        allv = pd.concat([old, new]).sort_index()
        assert not allv.index.duplicated().any()
        allv.to_parquet(DST / f"{k}.parquet")
        log(f"{k}: {len(old):,} + {len(new):,} = {len(allv):,} 행, {allv.index.get_level_values('date').min().date()} ~ {allv.index.get_level_values('date').max().date()}")
    # 검증: 2026-05 를 같은 로직으로 만들어 원본과 대조
    v = build_month("2026-05")
    for k in ("meta", "foreign", "fundamentals"):
        old = pd.read_parquet(SRC / f"{k}.parquet").xs(pd.Timestamp("2026-05-31"), level="date")
        new = v[k].xs(pd.Timestamp("2026-05-31"), level="date")
        common = old.index.intersection(new.index)
        log(f"  검증 {k} 2026-05: 원본 {len(old)} / 재현 {len(new)} / 공통 {len(common)} | 원본만 {sorted(old.index.difference(new.index))[:5]} / 재현만 {sorted(new.index.difference(old.index))[:5]}")
        for c in old.columns:
            a, b = old.loc[common, c], new.loc[common, c]
            if a.dtype.kind in "fiu" and b.dtype.kind in "fiu":
                tol = 1e-6 if c not in ("foreign_ratio", "exhaustion", "foreign", "foreign_exhaustion") else 0.005
                diff = ~np.isclose(a.astype(float), b.astype(float), rtol=1e-9, atol=tol, equal_nan=True)
            elif c == "listing_date":
                continue
            else:
                diff = (a.astype(str) != b.astype(str))
            if diff.sum():
                ex = common[diff][:3]
                log(f"    ⚠ {k}.{c}: 불일치 {diff.sum()}/{len(common)}  예 {[(t, a[t], b[t]) for t in ex]}")

# ─────────────────────────── indices ───────────────────────────
def step_indices():
    from pykrx import stock
    s0, s1 = "20141201", ymd(END)
    k = stock.get_index_ohlcv(s0, s1, "1001", "m")["종가"]; k2 = stock.get_index_ohlcv(s0, s1, "1028", "m")["종가"]
    def monthly(ser):
        ser = ser.copy(); ser.index = pd.to_datetime(ser.index) + pd.offsets.MonthEnd(0); return ser
    k, k2 = monthly(k), monthly(k2)
    ik = pd.DataFrame({"KOSPI_close": k, "KOSPI_ret": k.pct_change()}); ik = ik[ik.index >= START]; ik.index.name = "date"
    bm = pd.DataFrame({"BENCH_KOSPI200": k2.pct_change()}); bm = bm[bm.index >= START]; bm.index.name = "date"
    ik.to_parquet(DST / "index_kospi.parquet"); bm.to_parquet(DST / "benchmark.parquet")
    for name, new in (("index_kospi", ik), ("benchmark", bm)):
        old = pd.read_parquet(SRC / f"{name}.parquet"); old = old[old.index >= START]
        c = old.index.intersection(new.index); d = (old.loc[c] - new.loc[c]).abs()
        bad = d[(d > 1e-4).any(axis=1)]
        log(f"{name}: {len(new)} 행 {new.index.min().date()}~{new.index.max().date()} | 원본 대조 {len(c)}개월 중 불일치 {len(bad)}: {[str(x.date()) for x in bad.index]}")
    d = stock.get_index_ohlcv(ymd(START), s1, "1028")
    d.index = pd.to_datetime(d.index); d.index.name = "Date"
    kd = pd.DataFrame({"Close": d["종가"].astype(float)})
    kd["Comp"] = kd["Close"].diff().round(2); kd["Change"] = (kd["Comp"] / kd["Close"].shift(1)).round(4)
    kd["UpDown"] = np.select([kd["Comp"] > 0, kd["Comp"] < 0], [1, 2], 3).astype("int64")
    for a, b in (("Open", "시가"), ("High", "고가"), ("Low", "저가"), ("Volume", "거래량"), ("Amount", "거래대금"), ("MarCap", "상장시가총액")):
        kd[a] = d[b].astype(float)
    kd = kd[["Close", "UpDown", "Comp", "Change", "Open", "High", "Low", "Volume", "Amount", "MarCap"]]
    # 첫 행 Comp/Change 는 2014-12-30 종가로
    prev = stock.get_index_ohlcv("20141220", "20141231", "1028")["종가"].iloc[-1]
    kd.iloc[0, kd.columns.get_loc("Comp")] = round(kd["Close"].iloc[0] - prev, 2)
    kd.iloc[0, kd.columns.get_loc("Change")] = round((kd["Close"].iloc[0] - prev) / prev, 4)
    kd.iloc[0, kd.columns.get_loc("UpDown")] = 1 if kd["Comp"].iloc[0] > 0 else (2 if kd["Comp"].iloc[0] < 0 else 3)
    kd.to_parquet(DST / "kospi200_daily.parquet")
    old = pd.read_parquet(EXT / "kospi200_daily.parquet"); old = old[old.index >= START]
    c = old.index.intersection(kd.index)
    mism = {col: int((~np.isclose(old.loc[c, col].astype(float), kd.loc[c, col].astype(float), rtol=1e-6, atol=1e-2)).sum()) for col in kd.columns}
    log(f"kospi200_daily: {len(kd)} 행 {kd.index.min().date()}~{kd.index.max().date()} | FDR 원본 대조 {len(c)}일 — 컬럼별 불일치 {mism}")

# ─────────────────────────── investor_flow ───────────────────────────
FLOW_INV = ("개인", "기관합계", "외국인", "기타외국인", "기타법인")
def flow_day(day: pd.Timestamp) -> pd.DataFrame:
    """하루치 전종목 순매수 거래대금 (KOSPI+KOSDAQ). 캐시."""
    from pykrx import stock
    f = CACHE / f"flow_{ymd(day)}.pkl"
    if f.exists(): return pd.read_pickle(f)
    cols = {}
    for mk in ("KOSPI", "KOSDAQ"):
        for inv in FLOW_INV:
            d = stock.get_market_net_purchases_of_equities(ymd(day), ymd(day), mk, inv)
            cols.setdefault(inv, []).append(d["순매수거래대금"])
    df = pd.DataFrame({inv: pd.concat(v) for inv, v in cols.items()})
    df = df[~df.index.duplicated()]
    out = pd.DataFrame({"개인": df["개인"], "기관": df["기관합계"], "외국인": df["외국인"] + df["기타외국인"], "기타법인": df["기타법인"]}).fillna(0).astype("int64")   # 투자자별 응답에 빠진 종목 = 순매수 0
    out.index.name = "ticker"; out["date"] = day
    out = out.reset_index().set_index(["date", "ticker"]).sort_index()
    pd.to_pickle(out, f); return out

def step_flow(budget_sec: float = 160):
    t0 = time.time()
    old = pd.read_parquet(SRC / "investor_flow.parquet")
    old = old[old.index.get_level_values("date") >= START]
    last_old = old.index.get_level_values("date").max()
    days = pd.read_parquet(DST / "kospi200_daily.parquet").index
    todo = [d for d in days if d > last_old and d <= END]
    # 검증: 원본 마지막 날(2026-06-30)을 같은 로직으로 받아 대조
    v = flow_day(last_old); o = old.xs(last_old, level="date"); n = v.xs(last_old, level="date")
    c = o.index.intersection(n.index)
    log(f"검증 investor_flow {last_old.date()}: 원본 {len(o)} / 재현 {len(n)} / 공통 {len(c)} | 컬럼별 불일치 " +
        str({k: int((o.loc[c, k] != n.loc[c, k]).sum()) for k in o.columns}) + f" | 원본만 {sorted(o.index.difference(n.index))[:5]} 재현만 {sorted(n.index.difference(o.index))[:5]}")
    parts, left = [], 0
    for d in todo:
        if time.time() - t0 > budget_sec and not (CACHE / f"flow_{ymd(d)}.pkl").exists():
            left += 1; continue
        parts.append(flow_day(d))
    if left:
        log(f"시간 예산 소진 — 남은 날짜 {left}개, 다시 실행하면 이어서 받는다"); return
    new = pd.concat(parts); allv = pd.concat([old, new]).sort_index()
    assert not allv.index.duplicated().any()
    allv.to_parquet(DST / "investor_flow.parquet")
    log(f"investor_flow: {len(old):,} + {len(new):,} = {len(allv):,} 행, ~{allv.index.get_level_values('date').max().date()}, 신규 구간 종목수 {new.index.get_level_values('ticker').nunique()} / 일수 {len(todo)}")

# ─────────────────────────── prices ───────────────────────────
PX_CACHE = CACHE / "px"; PX_CACHE.mkdir(exist_ok=True)
def px_fetch(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    from pykrx import stock
    f = PX_CACHE / f"{ticker}.pkl"
    if f.exists(): return pd.read_pickle(f)
    for i in range(3):
        try:
            d = stock.get_market_ohlcv(ymd(start), ymd(end), ticker); break
        except Exception as e:
            if i == 2: log(f"  {ticker} 실패: {e}"); return None
            time.sleep(2)
    d = d.rename(columns={"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume"})[["open", "high", "low", "close", "volume"]].astype(float)
    d.index = pd.to_datetime(d.index); d.index.name = "date"
    pd.to_pickle(d, f); time.sleep(0.05); return d

def step_prices(budget_sec: float = 165):
    """data_ext/prices (~2026-09-11) 를 이음새 방식으로 END 까지 연장. 이음새 종가가 다르면(분할·병합) 과거를 비율 보정."""
    t0 = time.time()
    base = pd.read_parquet(EXT / "prices.parquet")
    base = base[base.index.get_level_values("date") >= START]
    seam = base.index.get_level_values("date").max()
    alive = base.xs(seam, level="date")
    log(f"기준: data_ext/prices {len(base):,}행, 이음새 {seam.date()}, 생존 종목 {len(alive)}")
    # 신규 상장: 월말 meta(2026-09·10)에 있는데 prices 에 없는 보통주(우선주·스팩 제외)
    # ⚠ 원본 prices 는 KOSPI+KOSDAQ 보통주 가운데 ~230종목(소형 지주사 등)을 미상의 규칙으로 빼놓았다. 지금 그 종목을
    #   넣으면 2026-09 부터만 존재하는 불연속 시계열이 되므로, meta 에 2026-09 이후 **처음** 등장한 종목만 신규상장으로 더한다.
    meta = pd.read_parquet(DST / "meta.parquet")
    first = meta.reset_index().groupby("ticker")["date"].min()
    recent = meta[meta.index.get_level_values("date") >= pd.Timestamp("2026-09-30")]
    recent = recent[~recent.is_pref & ~recent.is_spac]
    known = set(base.index.get_level_values("ticker").unique())
    cand = set(recent.index.get_level_values("ticker")) - known
    newborn = sorted(t for t in cand if first.get(t, pd.Timestamp.min) >= pd.Timestamp("2026-09-30") and t.endswith("0"))
    todo = list(alive.index) + newborn
    done = sum((PX_CACHE / f"{t}.pkl").exists() for t in todo)
    log(f"대상 {len(todo)} (신규상장 후보 {len(newborn)}: {newborn[:8]}{'…' if len(newborn) > 8 else ''}) / 캐시 {done}")
    parts, adj, left = [], [], 0
    for t in todo:
        cached = (PX_CACHE / f"{t}.pkl").exists()
        if not cached and time.time() - t0 > budget_sec:
            left += 1; continue
        d = px_fetch(t, seam, END)
        if d is None or d.empty: continue
        if t in alive.index:
            if seam not in d.index:
                log(f"  {t}: 이음새 {seam.date()} 행 없음 — 연장 생략"); continue
            old_c, new_c = float(alive.loc[t, "close"]), float(d.loc[seam, "close"])
            if old_c > 0 and new_c > 0 and abs(new_c / old_c - 1) > 0.005:
                adj.append((t, old_c, new_c, new_c / old_c))
            d = d[d.index > seam]
        d = d.copy(); d["ticker"] = t; parts.append(d.reset_index().set_index(["date", "ticker"]))
    if left:
        log(f"시간 예산 소진 — 남은 종목 {left}, 다시 실행하면 이어서 받는다 (경과 {int(time.time()-t0)}s)"); return
    new = pd.concat(parts).sort_index()
    out = base.copy()
    if adj:
        log(f"이음새 종가 불일치 {len(adj)}종목 — 과거 구간을 비율 보정 (pykrx 수정주가 = 오늘 기준):")
        for t, o, n, r in adj[:12]: log(f"    {t}: {o:,.0f} → {n:,.0f} (×{r:.4f})")
        if len(adj) > 12: log(f"    … 외 {len(adj)-12}종목 (전체 목록 _prices_seam_adjust.csv)")
        ratio = out.index.get_level_values("ticker").map({t: r for t, o, n, r in adj}).fillna(1.0).to_numpy()
        for c in ("open", "high", "low", "close"): out[c] = out[c].to_numpy() * ratio
        out["volume"] = out["volume"].to_numpy() / ratio
        clean = sum(abs(r - round(r)) < 0.002 or abs(1/r - round(1/r)) < 0.002 for *_, r in adj)
        log(f"    정수배(액면분할·병합형) {clean} / 비정수배(감자·증자형) {len(adj)-clean}")
        pd.DataFrame(adj, columns=["ticker", "old_close", "new_close", "ratio"]).to_csv(DST / "_prices_seam_adjust.csv", index=False)
    allv = pd.concat([out, new]).sort_index()
    assert not allv.index.duplicated().any()
    allv.to_parquet(DST / "prices.parquet")
    dts = allv.index.get_level_values("date")
    log(f"prices: {len(out):,} + {len(new):,} = {len(allv):,} 행, {dts.min().date()} ~ {dts.max().date()}, 종목 {allv.index.get_level_values('ticker').nunique()} | 신규 구간 거래일 {new.index.get_level_values('date').nunique()}")

# ─────────────────────────── trim / report ───────────────────────────
def _trim_copy(name: str, src_dir: Path):
    d = pd.read_parquet(src_dir / f"{name}.parquet")
    lvl = "date" if "date" in d.index.names else ("Date" if "Date" in d.index.names else None)
    dts = d.index.get_level_values(lvl) if lvl else d["Date"]
    d = d[(dts >= START) & (dts <= END + pd.offsets.MonthEnd(0))] if name in ("meta", "foreign", "fundamentals", "dart_factors", "universe_snapshots", "index_kospi", "benchmark") else d[(dts >= START) & (dts <= END)]
    d.to_parquet(DST / f"{name}.parquet"); return d

def step_trim():
    """아직 연장하지 못한 파일을 2015~ 로 재단해 세트를 채운다 (덮어쓰지 않음 — 이미 만든 파일은 건너뜀)."""
    for name, src_dir in (("investor_flow", SRC), ("dart_factors", SRC), ("universe_snapshots", SRC)):
        if (DST / f"{name}.parquet").exists():
            log(f"{name}: 이미 있음 — 건너뜀"); continue
        d = _trim_copy(name, src_dir); log(f"{name}: 재단 사본 {len(d):,} 행 (연장 전 — 원본 data/ 범위 그대로)")
    # investor_flow 는 원본 관례대로 prices 종목 집합으로 제한한다 (원본 2026-06-30 단면 2,176 = 전부 prices 종목)
    if (DST / "prices.parquet").exists() and (DST / "investor_flow.parquet").exists():
        px_t = set(pd.read_parquet(DST / "prices.parquet", columns=["close"]).index.get_level_values("ticker").unique())
        f = pd.read_parquet(DST / "investor_flow.parquet"); n0 = len(f)
        f = f[f.index.get_level_values("ticker").isin(px_t)]
        if len(f) != n0:
            f.to_parquet(DST / "investor_flow.parquet"); log(f"investor_flow: prices 종목 밖 {n0-len(f):,} 행 제거 → {len(f):,}")

def step_report():
    rows = []
    for f in sorted(DST.glob("*.parquet")):
        d = pd.read_parquet(f)
        lvl = "date" if "date" in d.index.names else ("Date" if "Date" in d.index.names else None)
        dts = pd.to_datetime(d.index.get_level_values(lvl)) if lvl else pd.to_datetime(d["Date"])
        tk = d.index.get_level_values("ticker").nunique() if "ticker" in d.index.names else (d["Code"].nunique() if "Code" in d.columns else "-")
        na = int(d.isna().sum().sum())
        rows.append((f.stem, f"{len(d):,}", str(dts.min().date()), str(dts.max().date()), dts.nunique(), tk, ", ".join(map(str, d.columns))[:70], f"{na:,}"))
    rep = pd.DataFrame(rows, columns=["파일", "행", "시작", "끝", "일자수", "종목수", "컬럼", "결측셀"])
    print(rep.to_string(index=False)); rep.to_csv(DST / "_report.csv", index=False)

# ─────────────────────────── derivatives_all ───────────────────────────
LLV_OPT = HERE.parents[1] / "LongLiveVault" / "data" / "options"
DERIV_COLS = ["Date", "Kind", "Session", "Underlying", "Type", "Code", "Name", "Open", "High", "Low", "Close", "Change", "IV", "BasePrice", "Volume", "Amount", "OI"]
def _daily_to_all(f: Path) -> pd.DataFrame:
    d = pd.read_parquet(f); d["Kind"] = f.name[:3]
    # 옵션은 "(정규)/(야간)", 선물은 "(주간)/(야간)" — 통합본은 주간→정규. 표기 없음(코스닥위클리 옵션·일부 선물)은 미구분.
    sess = d["Name"].str.extract(r"\((정규|주간|야간)\)\s*$")[0].replace({"주간": "정규"})
    if d["Kind"].iloc[0] == "opt":   # 표기 없는 옵션(코스닥위클리)은 통합본 규칙대로 IV>0 → 정규 / IV=0 → 야간 (2026-08-11·12 대조 100%)
        sess = sess.fillna(pd.Series(np.where(d["IV"] > 0, "정규", "야간"), index=d.index))
    d["Session"] = sess.fillna("미구분")
    d = d[~((d["Volume"] == 0) & (d["OI"] == 0))]
    return d[DERIV_COLS]

def step_deriv():
    """LLV 통합본(에어 사본, ~2026-08-12) + 일별 opt/fut 파일(에어 ~08-24, 미니 사본은 환경변수 DERIV_EXTRA_DIR) → data_2015.
    ⚠ 380만 행 문자열 테이블은 VM(3.9GB)에서 pandas 로 들면 OOM — pyarrow 로만 다룬다."""
    import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
    base = pq.read_table(LLV_OPT / "derivatives_all.parquet")
    assert base.column_names == DERIV_COLS, base.column_names
    base = base.filter(pc.and_(pc.greater_equal(base["Date"], pa.scalar(START.to_pydatetime())), pc.less_equal(base["Date"], pa.scalar(END.to_pydatetime()))))
    have_max = pd.Timestamp(pc.max(base["Date"]).as_py())
    dirs = [LLV_OPT] + ([Path(os.environ["DERIV_EXTRA_DIR"])] if os.environ.get("DERIV_EXTRA_DIR") else [])
    files = {}
    for dd in dirs:
        for f in sorted(dd.glob("*_20*.parquet")):
            day = pd.Timestamp(f.stem.split("_")[1])
            if day > have_max and day <= END: files.setdefault(day, {})[f.name[:3]] = f    # 같은 날짜는 뒤 디렉토리(미니)가 우선
    parts = [pa.Table.from_pandas(_daily_to_all(f), preserve_index=False).cast(base.schema) for day in sorted(files) for f in files[day].values()]
    allv = pa.concat_tables([base] + parts) if parts else base
    allv = allv.take(pc.sort_indices(allv, sort_keys=[("Date", "ascending"), ("Kind", "ascending"), ("Code", "ascending"), ("Session", "ascending")]))
    pq.write_table(allv, DST / "derivatives_all.parquet", compression="zstd")
    days = pd.to_datetime(pc.unique(allv["Date"]).to_pandas()).sort_values()
    bd = pd.bdate_range(START, END); missing = bd.difference(days)
    log(f"derivatives_all: {allv.num_rows:,} 행, {days.min().date()} ~ {days.max().date()}, 일수 {len(days)} | 통합본 {base.num_rows:,} + 일별 {len(files)}일 | 기간 내 평일 중 빈 날 {len(missing)} (마지막 날 이후 {int((bd > days.max()).sum())}일 포함)")
    old = pq.read_table(SRC / "derivatives_all.parquet", columns=["Date"])["Date"]
    for day in pd.to_datetime(pc.unique(old).to_pandas()).sort_values()[-3:]:
        sc = pa.scalar(day.to_pydatetime())
        log(f"  대조 {day.date()}: 구 사본 {pc.sum(pc.equal(old, sc)).as_py()} / 새 {pc.sum(pc.equal(allv['Date'], sc)).as_py()}")

# ─────────────────────────── investor_flow — KIS 경로 (케인 결정 2026-10-30 19:18) ───────────────────────────
# KRX Data Marketplace 차단으로 pykrx 전종목 수급을 못 받아, KIS 종목별 투자자매매동향(일별) FHPTJ04160001
# (investor-trade-by-stock-daily, FID_INPUT_DATE_1 기준일 ~ 과거 30거래일/콜) 로 연장한다.
#  · 시장구분 **J(KRX)** — 2026-06-30 삼성전자·카카오·KR모터스 대조에서 저장본(pykrx)과 백만원 반올림 수준 일치.
#    UN(KRX+NXT) 은 NXT 물량이 섞여 삼성전자 외국인이 −8,685억 vs −6,972억 로 달랐다 → 이력과 같은 정의 = KRX 정규장만.
#  · 단위 백만원(반올림) → ×1,000,000 원. 소형주의 100만원 미만 순매수는 0 이 된다 (원본은 원 단위).
#  · 외국인 = frgn_ntby_tr_pbmn (등록+비등록 합) · 기관 = orgn(기관계) · 기타법인 = etc_corp.
KIS_FLOW_URL = "/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily"
FLOW_CACHE = CACHE / "flow_kis"; FLOW_CACHE.mkdir(exist_ok=True)
_KF = None
def _kis():
    global _KF
    if _KF is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("kf", HERE.parents[1] / "LongLiveVault" / "stolab_data" / "kis_fetcher.py")
        _KF = importlib.util.module_from_spec(spec); spec.loader.exec_module(_KF)
    return _KF

def kis_flow_ticker(ticker: str, start: pd.Timestamp, end: pd.Timestamp, sess) -> pd.DataFrame:
    """end 기준일부터 start 까지 거슬러 받는다 (콜당 30거래일). 마지막 페이지의 start 이전 행도 남겨 대조에 쓴다."""
    f = FLOW_CACHE / f"{ticker}.pkl"
    if f.exists(): return pd.read_pickle(f)
    kf = _kis(); tok = kf._get_token(); h = kf._build_headers(tok, "FHPTJ04160001")
    cols = {"prsn_ntby_tr_pbmn": "개인", "orgn_ntby_tr_pbmn": "기관", "frgn_ntby_tr_pbmn": "외국인", "etc_corp_ntby_tr_pbmn": "기타법인"}
    rows, cursor = [], end
    for _ in range(8):
        body = None
        for attempt in range(5):
            try:
                r = sess.get(kf._BASE_URL + KIS_FLOW_URL, headers=h, timeout=10,
                             params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker, "FID_INPUT_DATE_1": ymd(cursor), "FID_ORG_ADJ_PRC": "", "FID_ETC_CLS_CODE": ""})
                body = r.json()
                if body.get("rt_cd") == "0": break
                if "초당" in (body.get("msg1") or "") or body.get("msg_cd") == "EGW00201": time.sleep(0.6); continue
                log(f"  {ticker}: KIS 거부 {body.get('msg_cd')} {body.get('msg1')}"); body = None; break
            except Exception as e:
                time.sleep(1.0 * (attempt + 1)); body = None
        if not body: break
        o = pd.DataFrame(body.get("output2") or [])
        if o.empty: break
        o = o[["stck_bsop_date"] + list(cols)].rename(columns=cols)
        o["date"] = pd.to_datetime(o["stck_bsop_date"]); o = o.drop(columns="stck_bsop_date")
        rows.append(o)
        if o["date"].min() <= start or len(o) < 20: break
        cursor = o["date"].min() - pd.Timedelta(days=1)
        time.sleep(0.05)
    if not rows:
        d = pd.DataFrame(columns=["개인", "기관", "외국인", "기타법인"]); d.index.name = "date"
    else:
        d = pd.concat(rows).drop_duplicates("date").set_index("date").sort_index()
        d = d[d.index <= end]
        for c in ("개인", "기관", "외국인", "기타법인"):
            d[c] = (pd.to_numeric(d[c].replace({"": "0", "-": "0"}), errors="coerce").fillna(0) * 1_000_000).round().astype("int64")
    pd.to_pickle(d, f); time.sleep(0.04); return d

def step_flow_kis(budget_sec: float = 165):
    import requests
    t0 = time.time()
    old = pd.read_parquet(DST / "investor_flow.parquet")
    last_old = old.index.get_level_values("date").max(); start = last_old + pd.Timedelta(days=1)
    px = pd.read_parquet(DST / "prices.parquet", columns=["close"])
    alive = sorted(px[px.index.get_level_values("date") >= start].index.get_level_values("ticker").unique())
    done = sum((FLOW_CACHE / f"{t}.pkl").exists() for t in alive)
    log(f"investor_flow(KIS): 원본 끝 {last_old.date()} → {start.date()}~{END.date()}, 대상 {len(alive)}종목 / 캐시 {done}")
    # 겹침 대조용: 원본 마지막 45일만 종목별로 잘라 둔다 (종목마다 550만 행을 다시 훑으면 1종목/초로 느려진다 — 실측)
    tail = old[old.index.get_level_values("date") >= last_old - pd.Timedelta(days=45)]
    tail_by = {t: g.droplevel("ticker") for t, g in tail.groupby(level="ticker")}
    sess = requests.Session(); parts, overlap, left = [], [], 0
    for t in alive:
        if not (FLOW_CACHE / f"{t}.pkl").exists() and time.time() - t0 > budget_sec:
            left += 1; continue
        d = kis_flow_ticker(t, start, END, sess)
        if d.empty: continue
        ov = d[d.index <= last_old]
        if not ov.empty and t in tail_by:
            o = tail_by[t].reindex(ov.index).dropna()
            if not o.empty:
                diff = (o - ov.loc[o.index]).abs().max().max(); overlap.append((t, len(o), diff))
        n = d[d.index >= start].copy(); n["ticker"] = t
        parts.append(n.reset_index().set_index(["date", "ticker"]))
    if left:
        log(f"시간 예산 소진 — 남은 종목 {left}, 다시 실행하면 이어서 (경과 {int(time.time()-t0)}s, 이번에 받은 {len(parts)})"); return
    new = pd.concat(parts).sort_index()
    # 원본(pykrx 전종목 스냅샷) 관례: 거래정지일 행 없음 (2026-06 원본 45,576행 중 거래량 0 행 0). KIS 는 정지일에도 0 행을 주므로
    # 그날 거래량이 0 이거나 가격 행이 없는 행은 뺀다. 거래가 있는데 네 값이 모두 0 인 행(백만원 반올림, ~1.9%)은 실제 값이라 남긴다.
    pv = pd.read_parquet(DST / "prices.parquet", columns=["volume"])["volume"].reindex(new.index)
    halt = pv.isna() | (pv == 0)
    log(f"  거래정지일 행 제거 {int(halt.sum()):,} (전부 0 {int((halt & (new == 0).all(axis=1)).sum()):,}) · 거래 있는데 전부 0(반올림) {int((~halt & (new == 0).all(axis=1)).sum()):,} 유지")
    new = new[~halt]
    allv = pd.concat([old[old.index.get_level_values("date") <= last_old], new]).sort_index()
    assert not allv.index.duplicated().any()
    allv.to_parquet(DST / "investor_flow.parquet")
    ovd = pd.DataFrame(overlap, columns=["ticker", "rows", "maxdiff"])
    log(f"investor_flow: {len(old):,} + {len(new):,} = {len(allv):,} 행, ~{allv.index.get_level_values('date').max().date()} | 신규 종목 {new.index.get_level_values('ticker').nunique()} · 거래일 {new.index.get_level_values('date').nunique()}")
    log(f"  겹침 대조(≤{last_old.date()}): {len(ovd)}종목 {int(ovd.rows.sum())}행, |차| ≤ 100만원(반올림 한도) {int((ovd.maxdiff <= 1_000_000).sum())} / 초과 {int((ovd.maxdiff > 1_000_000).sum())} → {ovd[ovd.maxdiff > 1_000_000].head(5).values.tolist()}")
    ovd.to_csv(DST / "_flow_kis_overlap_check.csv", index=False)

if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "report"
    if step not in ("prices", "trim", "report", "deriv", "flow_kis"): krx_login()          # 종목별 OHLCV 는 로그인 없이 된다
    {"monthly": step_monthly, "indices": step_indices, "flow": step_flow, "prices": step_prices, "trim": step_trim, "report": step_report, "deriv": step_deriv, "flow_kis": step_flow_kis}[step]()
