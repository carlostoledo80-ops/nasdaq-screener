"""
Data access — fully automatic from the internet.

Default provider is yfinance (no API key, free). It gives us three things:
  1. daily bars (full history)        -> for building conditioning features
  2. 60-minute bars (~2 years)        -> for the REAL first-hour outcome truth
  3. a live premarket/last snapshot   -> for today's gap feature at run time

The Provider interface is intentionally small so you can drop in Polygon /
Alpaca / Databento later to get a genuine 5-year minute history without
touching the rest of the system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger(__name__)

_ET = "America/New_York"


@dataclass
class Snapshot:
    last: float | None
    prev_close: float | None

    @property
    def gap(self) -> float | None:
        if self.last and self.prev_close:
            return self.last / self.prev_close - 1.0
        return None


class Provider:
    def daily(self, ticker: str, lookback_days: int) -> pd.DataFrame:  # noqa: D401
        raise NotImplementedError

    def first_hour(self, ticker: str) -> pd.DataFrame:
        """DataFrame indexed by date with columns: open, fh_high, fh_close."""
        raise NotImplementedError

    def snapshot(self, ticker: str) -> Snapshot:
        raise NotImplementedError


class YFinanceProvider(Provider):
    def __init__(self, first_hour_minutes: int = 60) -> None:
        import yfinance  # imported lazily so the module loads without it installed

        self._yf = yfinance
        self._fh_min = first_hour_minutes

    def daily(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        # Pull generous daily history; features need warmup (RSI, 20d MA, ATR).
        period = f"{max(lookback_days + 120, 400)}d"
        df = self._yf.download(
            ticker, period=period, interval="1d",
            auto_adjust=False, progress=False, threads=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna(how="all")

    def first_hour(self, ticker: str) -> pd.DataFrame:
        # 60m bars: Yahoo serves ~730 days. The 09:30 ET bar is the first hour.
        df = self._yf.download(
            ticker, period="730d", interval="60m",
            auto_adjust=False, progress=False, threads=False, prepost=False,
        )
        if df.empty:
            return pd.DataFrame(columns=["open", "fh_high", "fh_close"])
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        idx = df.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        idx = idx.tz_convert(_ET)
        df = df.set_index(idx)
        # keep only the regular-session opening bar (09:30 ET)
        first = df[(df.index.hour == 9) & (df.index.minute == 30)].copy()
        out = pd.DataFrame(
            {
                "open": first["Open"].values,
                "fh_high": first["High"].values,
                "fh_close": first["Close"].values,
            },
            index=first.index.normalize().tz_localize(None),
        )
        out.index.name = "date"
        return out[~out.index.duplicated(keep="first")]

    def snapshot(self, ticker: str) -> Snapshot:
        try:
            t = self._yf.Ticker(ticker)
            fi = getattr(t, "fast_info", {}) or {}
            last = fi.get("last_price") or fi.get("lastPrice")
            prev = fi.get("previous_close") or fi.get("previousClose")
            # premarket-aware fallback if fast_info is thin
            if last is None:
                intraday = self._yf.download(
                    ticker, period="2d", interval="1m",
                    prepost=True, progress=False, threads=False,
                )
                if not intraday.empty:
                    if isinstance(intraday.columns, pd.MultiIndex):
                        intraday.columns = intraday.columns.get_level_values(0)
                    last = float(intraday["Close"].dropna().iloc[-1])
            return Snapshot(
                last=float(last) if last else None,
                prev_close=float(prev) if prev else None,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("snapshot failed for %s: %s", ticker, e)
            return Snapshot(last=None, prev_close=None)


def make_provider(cfg) -> Provider:
    p = cfg.data_provider.lower()
    if p == "yfinance":
        return YFinanceProvider(cfg.first_hour_minutes)
    if p in ("polygon", "alpaca"):
        raise NotImplementedError(
            f"Provider '{p}' interface is stubbed. Implement daily()/first_hour()/"
            "snapshot() against your key to unlock true 5-year minute history."
        )
    raise ValueError(f"unknown data_provider: {cfg.data_provider}")
