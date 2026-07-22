"""
Every knob you'll want to turn lives here.

The defaults are deliberately conservative and honest. Read the comments
before changing them — a couple of these choices are the difference
between a real signal and a curve-fit mirage.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # ------------------------------------------------------------------ universe
    # Nasdaq 100 constituents are fetched live (see universe.py). If the live
    # fetch fails, we fall back to a bundled snapshot so the job never dies.
    universe: str = "NDX100"

    # ------------------------------------------------------------- outcome we score
    # "Up move" = the FIRST HOUR (open -> open+first_hour_minutes).
    #   move_metric = "high"  -> did the intraday HIGH reach open*(1+threshold)?
    #                            (touch-based; matches how a momentum entry fills)
    #   move_metric = "close" -> was the first-hour CLOSE >= open*(1+threshold)?
    #                            (stricter; the move had to hold)
    first_hour_minutes: int = 60
    move_metric: str = "high"

    # Threshold for what counts as an "up move".
    #
    # NOTE ON 2%: a >=2% move in the first hour for a large-cap Nasdaq 100 name
    # is genuinely uncommon (unconditional base rate is often ~8-18% depending
    # on the volatility regime). That's fine — a rare outcome leaves MORE room
    # for a condition to add lift — but per-condition sample sizes get thin, so
    # min_samples matters more. Start at 1.0% to keep samples usable, raise to
    # 0.02 once you've confirmed the pipeline on your data.
    up_threshold: float = 0.010

    # ------------------------------------------------------------------- history
    # yfinance's FREE intraday history is the binding constraint:
    #   - 1-minute bars: last ~30 days only
    #   - 60-minute bars: last ~730 days (~2y)   <-- what we use for first-hour truth
    #   - daily bars: full history
    # So out of the box you get ~2 years of REAL first-hour outcomes, not 5.
    # To get a true 5-year sample you must plug a paid minute provider
    # (Polygon / Alpaca / Databento) into data.py — the interface is ready for it.
    lookback_days: int = 720
    data_provider: str = "yfinance"  # "yfinance" | "polygon" | "alpaca"

    # ------------------------------------------------------- conditioning features
    # Features are computed from data available at run time (~8:20 COT), i.e.
    # PRIOR-session data plus whatever premarket exists. Each is binned; a day
    # "matches today" when it shares today's bin across the active features.
    # Fewer active features => bigger samples but coarser conditioning. Three
    # is a sane default; do not crank this up without watching n collapse.
    active_features: tuple[str, ...] = (
        "gap_bin",          # today's premarket gap vs prior close, bucketed

    )
    # Optional extras you can add to active_features:
    #   "rsi_bin", "range_pos_bin", "dow" (day of week), "volume_bin"

    # --------------------------------------------------------------- honesty gates
    confidence: float = 0.95        # CI level on every rate
    min_samples: int = 30           # a condition with fewer matches is NOT ranked
    fdr_alpha: float = 0.10         # Benjamini-Hochberg false-discovery rate
    # Primary ranking score = Wilson LOWER bound of the conditional up-rate.
    # This penalizes small samples automatically and is our main defense
    # against surfacing noise. Do not switch this to the point estimate.
    rank_by: str = "wilson_lower"

    # Only include a name in the alert if it clears ALL of these:
    require_fdr_significant: bool = True   # survived BH across the universe
    min_lift: float = 0.05                 # conditional rate at least +5pp over base
    top_n: int = 5                         # how many names the alert lists

    # ------------------------------------------------------------------ costs
    # Used by backtest.py only, to report NET-of-cost hit rates. A 2% gross
    # first-hour move can be far less net. Round-trip cost in return terms.
    roundtrip_cost: float = 0.0010  # 10 bps; set to your real fill/slippage

    # ------------------------------------------------------------------ alert
    timezone: str = "America/Bogota"     # Colombia, UTC-5, no DST
    send_hour_local: int = 8             # informational; the CRON does the timing
    send_minute_local: int = 20
    # Channels: any you configure will be used; unconfigured ones are skipped.
    telegram_bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))
    email_to: str = field(default_factory=lambda: os.getenv("ALERT_EMAIL_TO", ""))
    email_from: str = field(default_factory=lambda: os.getenv("ALERT_EMAIL_FROM", ""))
    smtp_host: str = field(default_factory=lambda: os.getenv("SMTP_HOST", ""))
    smtp_port: int = field(default_factory=lambda: int(os.getenv("SMTP_PORT", "587")))
    smtp_user: str = field(default_factory=lambda: os.getenv("SMTP_USER", ""))
    smtp_password: str = field(default_factory=lambda: os.getenv("SMTP_PASSWORD", ""))

    # polygon/alpaca keys if you upgrade the data provider
    polygon_api_key: str = field(default_factory=lambda: os.getenv("POLYGON_API_KEY", ""))
    alpaca_key: str = field(default_factory=lambda: os.getenv("ALPACA_KEY_ID", ""))
    alpaca_secret: str = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", ""))


CONFIG = Config()
