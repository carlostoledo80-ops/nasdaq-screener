"""
Entry point run by the scheduler each weekday morning.

    python -m nasdaq_screener.run_daily

It fetches the current Nasdaq 100, screens every name for the conditional
first-hour up-rate, applies the honesty gates, and sends the alert. A day
with zero survivors is a valid, honest outcome — it sends a "no signal"
message rather than manufacturing one.
"""

from __future__ import annotations

import logging
import sys

from .alert import dispatch, format_alert
from .config import CONFIG
from .data import make_provider
from .engine import screen_universe
from .universe import get_ndx100


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("run_daily")

    cfg = CONFIG
    try:
        tickers = get_ndx100()
        log.info("Universe: %d tickers.", len(tickers))
        provider = make_provider(cfg)
        results, meta = screen_universe(tickers, provider, cfg)
        log.info("Screen complete: %s", meta)
        text = format_alert(results, meta, cfg)
        dispatch(text, cfg)
        return 0
    except Exception as e:  # noqa: BLE001
        log.exception("Daily run failed: %s", e)
        # Best-effort failure ping so silence never means "all clear".
        try:
            dispatch(f"⚠️ Screener Nasdaq 100 falló hoy: {e}", cfg)
        except Exception:  # noqa: BLE001
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
