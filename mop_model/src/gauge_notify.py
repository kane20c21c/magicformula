# -*- coding: utf-8 -*-
"""
gauge_notify.py — 섹터 오버나이트 게이지 통지 (메일 + Pushover)
================================================================
StockPortfolio app/paper_day/notify.py 와 동일 계약 (스윙/데이 포트 방식):
    send_email(subject, html) -> bool
    send_push(title, message) -> bool
전부 fail-safe — 발송 실패가 게이지 JSON 저장에 영향을 주지 않는다.

환경변수 (기존 .env 재사용, 새 키 없음):
    GMAIL_USER / GMAIL_APP_PW / ALERT_EMAIL   ← StockPortfolio/.env
    PUSHOVER_USER_KEY / PUSHOVER_API_TOKEN    ← MorningBrief/.env 폴백 포함

표기 규칙 (Kane): 기준 50% 대비 위 = 빨강 #ef5350, 아래 = 파랑 #1976D2.
확률 표시는 소수점 첫째 자리 xx.x%.
"""
from __future__ import annotations

import json
import logging
import os
import smtplib
import urllib.parse
import urllib.request
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

logger = logging.getLogger(__name__)

RED = "#ef5350"
BLUE = "#1976D2"

_STOLAB = Path(os.path.expanduser("~/DriveForALL/StoLab"))
_ENV_CANDIDATES = [
    _STOLAB / "StockPortfolio" / ".env",     # GMAIL_* / ALERT_EMAIL
    _STOLAB / "MorningBrief" / ".env",       # PUSHOVER_* 폴백 (실존 폴더명 — 공백 없음)
]
_env_loaded = False


def _load_env() -> None:
    """dotenv 있으면 사용, 없으면 수동 파싱. 이미 설정된 환경변수는 덮지 않는다."""
    global _env_loaded
    if _env_loaded:
        return
    for p in _ENV_CANDIDATES:
        if not p.exists():
            continue
        try:
            from dotenv import load_dotenv
            load_dotenv(p, override=False)
        except ImportError:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    _env_loaded = True


def send_email(subject: str, html: str) -> bool:
    _load_env()
    user = os.getenv("GMAIL_USER", "")
    pw = os.getenv("GMAIL_APP_PW", "")
    recipients = [a.strip() for a in os.getenv("ALERT_EMAIL", "").split(",") if a.strip()]
    if not (user and pw and recipients):
        logger.warning("[gauge.notify] GMAIL_USER/GMAIL_APP_PW/ALERT_EMAIL 미설정 — 메일 생략")
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"데이 포트 <{user}>"
        msg["To"] = ", ".join(recipients)
        msg.attach(MIMEText(html, "html", "utf-8"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(user, pw)
            s.sendmail(user, recipients, msg.as_string())
        logger.info("[gauge.notify] 메일 발송 완료: %s", subject)
        return True
    except Exception as e:
        logger.error("[gauge.notify] 메일 발송 실패(%s): %s", subject, e)
        return False


_PUSHOVER_URL = "https://api.pushover.net/1/messages.json"

# ── Pushover 앱 분리 (2026-08-18 Kane 지시) ──────────────────────────────
# 섹터 게이지는 **손절경보 앱**으로 보낸다 (Kane 확정 — 15:10 잠정 게이지는
# 장 마감 전에 즉시 봐야 하는 정보라 우선순위가 높아야 한다. 초안의 매매신호
# → 브리핑 → 손절경보 순으로 바뀌었다).
# 손절경보·시스템 앱은 **앱 단위로 priority 가 high(1) 로 강제**된다 —
# 정본은 MB push_sender.HIGH_PRIORITY_APPS, 아래 자립 경로도 같은 규칙을 지킨다.
# 앱→토큰 라우팅 정본도 MB push_sender 이고 여기서는 위임한다.
# 못 읽으면 자립 경로로 내려가 종전처럼 단일 토큰으로 보낸다.
APP_STOP = "stop"
APP_BRIEF = "brief"
_HIGH_PRIORITY_APPS = frozenset({"stop", "system"})
_APP_ENV_FALLBACK = {"watch": "PUSHOVER_TOKEN_WATCH",
                     "signal": "PUSHOVER_TOKEN_SIGNAL",
                     "stop": "PUSHOVER_TOKEN_STOP",
                     "brief": "PUSHOVER_TOKEN_BRIEF",
                     "system": "PUSHOVER_TOKEN_SYSTEM"}

_MB_UNLOADED = object()
_mb_send_push = _MB_UNLOADED


def _load_mb_send_push():
    """MB lib/push_sender 를 단독 로드 (lib/__init__ 의 무거운 import 우회).

    sys.modules 에 'lib' 같은 흔한 이름을 심지 않고 전용 이름으로 감싼다.
    실패는 한 번만 판정하고 기억한다.
    """
    global _mb_send_push
    if _mb_send_push is not _MB_UNLOADED:
        return _mb_send_push
    _mb_send_push = None
    try:
        import importlib.util
        import sys
        import types

        mb_root = Path(os.getenv("MORNING_BRIEF_PATH")
                       or (_STOLAB / "MorningBrief"))
        lib_dir = mb_root / "scripts" / "lib"
        if not (lib_dir / "push_sender.py").exists():
            logger.error("[gauge.notify] MB push_sender 없음(%s) — 자립 경로 사용",
                         lib_dir)
            return None
        pkg_name = "_mb_lib"
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(lib_dir)]
            sys.modules[pkg_name] = pkg
        mod_name = f"{pkg_name}.push_sender"
        if mod_name in sys.modules:
            _mb_send_push = sys.modules[mod_name].send_push
            return _mb_send_push
        spec = importlib.util.spec_from_file_location(
            mod_name, lib_dir / "push_sender.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        _mb_send_push = mod.send_push
    except Exception as e:
        logger.error("[gauge.notify] MB push_sender 로드 실패(%s) — 자립 경로 사용", e)
        _mb_send_push = None
    return _mb_send_push


def send_push(title: str, message: str, app: str = APP_STOP,
              priority: int = 0) -> bool:
    _load_env()
    mb = _load_mb_send_push()
    if mb is not None:
        try:
            return bool(mb(title, message, priority=priority, html=True, app=app))
        except Exception as e:
            logger.error("[gauge.notify] MB 위임 실패(%s) — 자립 경로 재시도", e)
    user_key = os.getenv("PUSHOVER_USER_KEY", "")
    legacy = os.getenv("PUSHOVER_API_TOKEN", "")
    api_token = os.getenv(_APP_ENV_FALLBACK.get(app, ""), "") or legacy
    if not (user_key and api_token):
        logger.warning("[gauge.notify] PUSHOVER 키 미설정 — 푸시 생략")
        return False
    if app in _HIGH_PRIORITY_APPS:
        priority = max(priority, 1)
    payload = {
        "token": api_token, "user": user_key,
        "title": title[:250], "message": message[:1024], "html": 1,
        "priority": max(-2, min(priority, 1)),
    }
    try:
        data = urllib.parse.urlencode(payload).encode("utf-8")
        req = urllib.request.Request(_PUSHOVER_URL, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        if body.get("status") == 1:
            logger.info("[gauge.notify] 푸시 발송 완료: %s", title)
            return True
        logger.error("[gauge.notify] Pushover 응답 오류: %s", body)
        return False
    except Exception as e:
        logger.error("[gauge.notify] 푸시 발송 실패(%s): %s", title, e)
        return False


# ════════════════════════════════════════════════════════
# 본문 빌더
# ════════════════════════════════════════════════════════
def _pct1(v) -> str:
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _color(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "#666"
    return RED if f > 0.5 else (BLUE if f < 0.5 else "#666")


def _span(v) -> str:
    return f"<span style='color:{_color(v)};font-weight:600'>{_pct1(v)}</span>"


def _font(v) -> str:
    """Pushover 는 <font color> 만 지원 (<span style> 미지원)."""
    return f'<font color="{_color(v)}">{_pct1(v)}</font>'


_TABLE = ('style="border-collapse:collapse;font-family:-apple-system,'
          "'Noto Sans KR',sans-serif;font-size:13px\"")
_TH = ('style="padding:6px 10px;border-bottom:2px solid #ccc;text-align:right;'
       'color:#666;font-weight:600"')
_TH_L = _TH.replace("right", "left")
_TD = 'style="padding:6px 10px;border-bottom:1px solid #eee;text-align:right"'
_TD_L = _TD.replace("right", "left")


def build_gauge_email(doc: dict) -> tuple[str, str]:
    d = doc.get("as_of", "")
    ptime = doc.get("price_time", "")
    sets = doc.get("sets", [])

    subject = f"[데이 포트] 섹터 오버나이트 게이지 — {d} {ptime} 잠정"

    # 요약 테이블 (8세트)
    rows = "".join(
        f"<tr><td {_TD_L}><b>{s['name']}</b></td>"
        f"<td {_TD}>{_span(s.get('weighted_p'))}</td>"
        f"<td {_TD}>{_pct1(s.get('mean_p'))}</td>"
        f"<td {_TD}>{s.get('n_scored', 0)}/{s.get('n_members', 0)}</td>"
        f"<td {_TD_L}>{s.get('top_member', {}).get('name', '—')} "
        f"{_pct1(s.get('top_member', {}).get('p'))}</td></tr>"
        for s in sets)
    body = (f"<p><b>{ptime}</b> 잠정 종가 기준, 갭1(익일시가/당일종가) "
            f"<b>상대 상승확률</b>의 세트 내 지정 가중평균입니다.<br>"
            f"<span style='font-size:12px;color:#777'>확률은 MOp 유니버스 "
            f"{doc.get('universe_count', '—')}종목 내 상대 백분위 — 50% = 시장 중앙. "
            f"동시호가(15:20~15:30) 미반영 잠정치이며 확정 신호는 16:20 정본.</span></p>"
            f"<table {_TABLE}><tr><th {_TH_L}>세트</th><th {_TH}>가중확률</th>"
            f"<th {_TH}>단순평균</th><th {_TH}>종목</th><th {_TH_L}>세트 내 1위</th></tr>"
            f"{rows}</table>")

    # 세트별 상세
    for s in sets:
        mrows = "".join(
            f"<tr><td {_TD_L}>{m['name']} <span style='color:#999'>{m['ticker']}</span></td>"
            f"<td {_TD}>{round(m['close']):,}원</td>"
            f"<td {_TD}>{m['weight'] * 100:.1f}%</td>"
            f"<td {_TD}>{_span(m.get('p'))}</td></tr>"
            for m in s.get("members", []) if m.get("p") is not None)
        excl = [m for m in s.get("members", []) if m.get("p") is None]
        body += (f"<h3 style='font-size:13px;margin:16px 0 4px'>{s['name']} "
                 f"— {_span(s.get('weighted_p'))}</h3>"
                 f"<table {_TABLE}><tr><th {_TH_L}>종목</th><th {_TH}>잠정가</th>"
                 f"<th {_TH}>가중치</th><th {_TH}>확률</th></tr>{mrows}</table>")
        if excl:
            names = ", ".join(f"{m['name']}({m['ticker']})" for m in excl)
            body += (f"<p style='font-size:11px;color:#b45309'>⚠ 점수 없음(제외): "
                     f"{names}</p>")

    html = f"""<html><body style="font-family:-apple-system,'Noto Sans KR',sans-serif;color:#222">
<h2 style="font-size:16px">🤖 데이 포트 — 섹터 오버나이트 게이지 (잠정)</h2>
{body}
<p style="color:#999;font-size:11px;margin-top:18px">
mop_ml gauge · 15:10 잠정 종가 → 15:15 보고 · 정본 신호는 16:20 run_daily</p>
</body></html>"""
    return subject, html


def build_gauge_push(doc: dict) -> tuple[str, str]:
    d = doc.get("as_of", "")
    sets = doc.get("sets", [])
    lines = "\n".join(
        f"<b>{s['name']}</b> {_font(s.get('weighted_p'))}" for s in sets)
    return f"🤖 섹터 게이지 ({d} 잠정)", lines


__all__ = ["send_email", "send_push", "build_gauge_email", "build_gauge_push"]
