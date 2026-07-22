"""
Build and deliver the daily alert.

The message is written to be impossible to misread as a crystal ball. Each
name shows the conditional historical up-rate WITH its sample size and
confidence interval, the stock's own base rate, the lift, and the FDR
q-value. The footer states plainly that these are historical frequencies,
not predictions.
"""

from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from .engine import ScreenResult

log = logging.getLogger(__name__)


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def format_alert(results: list[ScreenResult], meta: dict, cfg) -> str:
    now = datetime.now(ZoneInfo(cfg.timezone))
    metric = "cierre" if cfg.move_metric == "close" else "máximo"
    thr = f"{100 * cfg.up_threshold:.2f}%".rstrip("0").rstrip(".")

    lines = []
    lines.append("📊 SCREENER NASDAQ 100 — tasas base condicionales")
    lines.append(now.strftime("%Y-%m-%d %H:%M ") + "COT")
    lines.append(
        f"Evento: {metric} de la 1ª hora ≥ apertura +{thr} "
        f"({cfg.first_hour_minutes} min)"
    )
    lines.append(
        f"Universo escaneado: {meta.get('scanned', 0)} · "
        f"con muestra: {meta.get('with_sample', 0)} · "
        f"significativos (FDR {cfg.fdr_alpha:g}): {meta.get('fdr_significant', 0)}"
    )
    lines.append("─" * 34)

    if not results:
        lines.append("Hoy ningún nombre supera los filtros de honestidad")
        lines.append("(muestra mínima, lift y control FDR). Sin señal ≠ error:")
        lines.append("es el sistema negándose a inventar una.")
    else:
        lines.append("Ordenado por límite inferior del IC (conservador):")
        lines.append("")
        for i, r in enumerate(results, 1):
            e = r.estimate
            cond = ", ".join(r.condition.values())
            gap = f"{100 * r.today_gap:+.2f}%" if r.today_gap is not None else "n/d"
            lines.append(f"{i}. {r.ticker}")
            lines.append(
                f"   sube {_pct(e.point)}  (n={e.n}, IC95% {_pct(e.lo)}–{_pct(e.hi)})"
            )
            lines.append(
                f"   base {_pct(e.base_rate)}  ·  lift {100 * e.lift:+.0f}pp  ·  q={r.qvalue:.3f}"
            )
            lines.append(f"   condición hoy: {cond}  ·  gap {gap}")
            lines.append("")

    lines.append("─" * 34)
    lines.append(
        "Estas son FRECUENCIAS HISTÓRICAS bajo condiciones parecidas, "
        "no probabilidades del futuro. n = días comparables; el IC es la "
        "incertidumbre real. No es asesoría de inversión."
    )
    return "\n".join(lines)


def send_telegram(text: str, cfg) -> bool:
    if not (cfg.telegram_bot_token and cfg.telegram_chat_id):
        return False
    try:
        import requests

        url = f"https://api.telegram.org/bot{cfg.telegram_bot_token}/sendMessage"
        resp = requests.post(
            url,
            json={"chat_id": cfg.telegram_chat_id, "text": text,
                  "disable_web_page_preview": True},
            timeout=20,
        )
        ok = resp.status_code == 200
        if not ok:
            log.error("Telegram send failed: %s %s", resp.status_code, resp.text[:200])
        return ok
    except Exception as e:  # noqa: BLE001
        log.error("Telegram send error: %s", e)
        return False


def send_email(text: str, cfg) -> bool:
    if not (cfg.smtp_host and cfg.email_to and cfg.email_from):
        return False
    try:
        import smtplib
        from email.mime.text import MIMEText

        msg = MIMEText(text, _charset="utf-8")
        msg["Subject"] = "Screener Nasdaq 100 — alerta diaria"
        msg["From"] = cfg.email_from
        msg["To"] = cfg.email_to
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=20) as s:
            s.starttls()
            if cfg.smtp_user:
                s.login(cfg.smtp_user, cfg.smtp_password)
            s.sendmail(cfg.email_from, [cfg.email_to], msg.as_string())
        return True
    except Exception as e:  # noqa: BLE001
        log.error("Email send error: %s", e)
        return False


def dispatch(text: str, cfg) -> None:
    """Send on every configured channel; always echo to stdout/logs."""
    print(text)
    sent_tg = send_telegram(text, cfg)
    sent_mail = send_email(text, cfg)
    if not (sent_tg or sent_mail):
        log.warning("No alert channel configured — printed to stdout only. "
                    "Set TELEGRAM_* or SMTP_* secrets to receive the alert.")
