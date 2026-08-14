#!/usr/bin/env python3
"""
send_fever_mail.py — 발열률 일일 메일 발송
==========================================
daily_WW_wf.py 가 만든 산출물(온도일지 · 상위10 · 그래프 PNG)을 읽어
HTML 메일 1통을 보낸다. 계산은 하지 않는다 (읽기 전용).

SMTP: StockPortfolio / MorningBrief 의 .env 를 그대로 재사용
      GMAIL_USER / GMAIL_APP_PW / ALERT_EMAIL (SP app/paper/notify.py 와 동일 계약)

사용법:
    python3 send_fever_mail.py              # 발송
    python3 send_fever_mail.py --dry-run    # 발송 없이 output/발열률_메일.html 로 미리보기
"""
from __future__ import annotations

import argparse
import os
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
FM   = os.path.dirname(BASE)
OUT  = os.path.join(FM, "output")
STOLAB = os.path.dirname(os.path.dirname(FM))

TEMP_LOG = os.path.join(OUT, "나우캐스트_온도일지.csv")
TOP_CSV  = os.path.join(OUT, "발열률_상위10.csv")
GRAPH    = os.path.join(OUT, "발열률_온도그래프.png")
PREVIEW  = os.path.join(OUT, "발열률_메일.html")

UP, DOWN, FLAT = "#ef5350", "#1976D2", "#9e9e9e"
KO_WD = ["월", "화", "수", "목", "금", "토", "일"]


def load_env():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for p in (os.path.join(STOLAB, "StockPortfolio", ".env"),
              os.path.join(STOLAB, "MorningBrief", ".env")):
        if os.path.exists(p):
            load_dotenv(p, override=False)


def won(v):
    return f"{int(round(v)):,}원" if pd.notna(v) else "—"


def pct_span(v, suffix="%"):
    if pd.isna(v):
        return f'<span style="color:{FLAT}">—</span>'
    c = UP if v > 0 else (DOWN if v < 0 else FLAT)
    return f'<span style="color:{c}">{v:+.1f}{suffix}</span>'


def build_html(temp, top, cid="fevergraph"):
    cur = temp.iloc[-1]
    prev = temp.iloc[-2] if len(temp) >= 2 else cur
    d = pd.Timestamp(cur["date"])
    dstr = f"{d:%Y.%m.%d}({KO_WD[d.weekday()]})"
    dtemp = cur["hot60"] - prev["hot60"]
    spread = cur["hot60"] - cur["hot100"]          # 음수 = 냉각 진행 (2026-08-14 Kane)
    judged = int(cur["장기60"] + cur["저변60"] + cur["단기60"])

    TH = ("padding:7px 8px;border-bottom:2px solid #c8ccd0;font-size:12px;"
          "color:#5f5f5f;font-weight:600;")
    TD = "padding:7px 8px;border-bottom:1px solid #e6e8ea;font-size:13px;"
    NUM = "text-align:right;font-variant-numeric:tabular-nums;"

    scols = [c for c in top.columns if c.startswith("Vblk ")]     # D-2, D-1, D+0
    rcols = [c for c in top.columns if c.startswith("순위 ")]
    labels = ["D-2", "D-1", "D+0"]
    close_hdr = f"종가<div style='font-weight:400;color:#8a8a8a;font-size:10px'>{d:%-m월 %-d일}</div>"

    heads = [("순위", 1), ("종목", 0), ("국면", 0)]
    heads += [(f"{lab}<div style='font-weight:400;color:#8a8a8a;font-size:10px'>"
               f"{c[5:]}</div>", 1) for lab, c in zip(labels, scols)]
    heads += [("전체순위", 1), ("p4단독", 1), (close_hdr, 1), ("등락률", 1), ("", 0)]
    head = "".join(f'<th style="{TH}text-align:{"right" if r else "left"}">{h}</th>'
                   for h, r in heads)

    rows = []
    for _, r in top.iterrows():
        flags = ""
        if r["과열"]:
            flags += f'<span title="과열" style="color:{UP}">●</span> '
        if r["재가열"]:
            flags += '<span title="재가열" style="color:#ef6c00">◆</span>'
        newmark = (f'<span style="color:{UP};font-weight:700">↑</span>'
                   if r["신규"] else "")
        chg = r["등락률"]
        ccol = UP if pd.notna(chg) and chg > 0 else (
            DOWN if pd.notna(chg) and chg < 0 else FLAT)

        score_cells = []
        for i, c in enumerate(scols):
            v = r[c]
            txt = f"{v:.1f}" if pd.notna(v) else "—"
            if i == len(scols) - 1:                  # D+0 만 볼드
                score_cells.append(f'<td style="{TD}{NUM}font-weight:700">{txt}</td>')
            else:
                score_cells.append(f'<td style="{TD}{NUM}color:#8a8a8a">{txt}</td>')

        rows.append(
            "<tr>"
            f'<td style="{TD}{NUM}">{r["순위"]}</td>'
            f'<td style="{TD}">{r["종목명"]} {newmark}'
            f'<div style="color:#8a8a8a;font-size:11px">{r["티커"]} · {r["섹터"]}</div></td>'
            f'<td style="{TD}">{r["국면"]}</td>'
            + "".join(score_cells) +
            f'<td style="{TD}{NUM}">{int(r[rcols[-1]])}</td>'
            f'<td style="{TD}{NUM}color:#8a8a8a">{int(r["p4단독"])}</td>'
            f'<td style="{TD}{NUM}">{won(r["종가"])}</td>'
            f'<td style="{TD}{NUM}color:{ccol}">'
            f'{f"{chg:+.1f}%" if pd.notna(chg) else "—"}</td>'
            f'<td style="{TD}text-align:center">{flags}</td>'
            "</tr>")

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"></head>
<body style="margin:0;padding:18px;background:#ffffff;color:#1f1f1f;
             font-family:-apple-system,BlinkMacSystemFont,'Apple SD Gothic Neo',sans-serif">
  <div style="max-width:900px;margin:0 auto">
    <h2 style="margin:0 0 2px;font-size:19px">🌡 발열률 — {dstr}</h2>
    <div style="color:#6b6b6b;font-size:12px;margin-bottom:14px">
      시장 온도 나우캐스트(W60/M4) + 발열계 상위 10 · 관측 지평 20~60거래일
    </div>

    <table style="width:100%;border-collapse:collapse;background:#f4f6f8;
                  border-radius:8px;margin-bottom:16px">
      <tr>
        <td style="padding:14px 16px;width:34%">
          <div style="color:#6b6b6b;font-size:11px">시장 온도 (W60/M4)</div>
          <div style="font-size:26px;font-weight:700">{cur['hot60']:.1f}%
            <span style="font-size:14px">{pct_span(dtemp, '%p')}</span></div>
        </td>
        <td style="padding:14px 16px;width:33%">
          <div style="color:#6b6b6b;font-size:11px">스프레드 (W60−W100)</div>
          <div style="font-size:26px;font-weight:700">{pct_span(spread, '%p')}</div>
          <div style="color:#8a8a8a;font-size:11px">음수 = 냉각 진행</div>
        </td>
        <td style="padding:14px 16px;width:33%">
          <div style="color:#6b6b6b;font-size:11px">TIGER 200</div>
          <div style="font-size:22px;font-weight:700">{won(cur['tiger_close'])}</div>
        </td>
      </tr>
    </table>

    <div style="color:#5f5f5f;font-size:12px;margin-bottom:10px">
      구성: 장기 {int(cur['장기60'])} · 저변 {int(cur['저변60'])} ·
      단기 {int(cur['단기60'])} · 유보 {int(cur['유보60'])} (판정 {judged}종목)
      &nbsp;|&nbsp; 역사적 중앙값 29%
    </div>

    <img src="cid:{cid}" style="width:100%;border:1px solid #e6e8ea;border-radius:8px;margin-bottom:18px">

    <h3 style="font-size:15px;margin:0 0 8px">발열계 상위 10</h3>
    <table style="width:100%;border-collapse:collapse">
      <thead><tr>{head}</tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>

    <div style="color:#6b6b6b;font-size:11px;margin-top:14px;line-height:1.6">
      ● 과열(마지막 상승파가 자기 이력 95백분위 이상) · ◆ 재가열(최근 60거래일 내
      장기 요동 이력, 현재는 아님) · ↑ 전일 대비 상위10 신규 진입<br>
      발열계 = 추세 33 : 눌림진폭 17 : 눌림거래량 17 : YZ 33 (백분위 합성).
      <b>관측 명단이지 매수 지시가 아니다</b> — 지평 20~60거래일, 장기 보유용 아님.
      급등장에선 순위가 베타(YZ) 순서로 쏠린다.<br>
      온도는 절대 레벨보다 방향·상대 변화로 읽을 것. 일간 등락은 노이즈.
    </div>
  </div>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    temp = pd.read_csv(TEMP_LOG)
    top = pd.read_csv(TOP_CSV, dtype={"티커": str}).fillna(
        {"과열": "", "재가열": "", "신규": "", "섹터": ""})

    html = build_html(temp, top)
    if args.dry_run:
        with open(PREVIEW, "w", encoding="utf-8") as f:
            f.write(html.replace(f'src="cid:fevergraph"',
                                 f'src="{os.path.basename(GRAPH)}"'))
        print(f"미리보기 → {PREVIEW}")
        return

    load_env()
    user, pw = os.getenv("GMAIL_USER", ""), os.getenv("GMAIL_APP_PW", "")
    to = [a.strip() for a in os.getenv("ALERT_EMAIL", "").split(",") if a.strip()]
    if not (user and pw and to):
        print("⚠ GMAIL_USER/GMAIL_APP_PW/ALERT_EMAIL 미설정 — 발송 생략")
        return

    d = pd.Timestamp(temp.iloc[-1]["date"])
    msg = MIMEMultipart("related")
    msg["Subject"] = (f"[발열률] {d:%m/%d} 온도 {temp.iloc[-1]['hot60']:.0f}% · "
                      f"{top.iloc[0]['종목명']} 외 9")
    msg["From"] = f"발열률 <{user}>"
    msg["To"] = ", ".join(to)
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(html, "html", "utf-8"))
    msg.attach(alt)
    if os.path.exists(GRAPH):
        with open(GRAPH, "rb") as f:
            img = MIMEImage(f.read())
        img.add_header("Content-ID", "<fevergraph>")
        img.add_header("Content-Disposition", "inline",
                       filename="발열률_온도그래프.png")
        msg.attach(img)
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(user, pw)
            s.sendmail(user, to, msg.as_string())
        print(f"메일 발송 완료 → {', '.join(to)}")
    except Exception as e:
        print(f"⚠ 메일 발송 실패: {e}")


if __name__ == "__main__":
    main()
