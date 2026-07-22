"""
The screening engine.

For each ticker it reconstructs, from real first-hour bars, the historical
record of "did the first hour close/high clear +threshold vs the open?".
It then looks at TODAY's feature bins, finds the historical days that match
those bins, and reports the conditional up-rate with full uncertainty.

Across the whole universe it applies Benjamini-Hochberg FDR control and
ranks survivors by the Wilson lower bound. Nothing here claims to predict
the future; it reports how often, under matched past conditions, the move
happened — and how sure we can be that that's more than noise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import Provider
from .features import bin_features, compute_indicators, today_feature_bins
from .stats import RateEstimate, benjamini_hochberg, estimate_conditional_rate

log = logging.getLogger(__name__)


@dataclass
class ScreenResult:
    ticker: str
    estimate: RateEstimate
    condition: dict[str, str]   # today's matched bins (active features only)
    today_gap: float | None
    qvalue: float = 1.0
    fdr_pass: bool = False


def build_dataset(daily: pd.DataFrame, first_hour: pd.DataFrame, cfg) -> pd.DataFrame:
    """Join prior-session feature bins with each day's first-hour outcome."""
    if daily.empty or first_hour.empty:
        return pd.DataFrame()

    ind = compute_indicators(daily)
    binned = bin_features(ind)
    binned.index = pd.to_datetime(binned.index).normalize()

    fh = first_hour.copy()
    fh.index = pd.to_datetime(fh.index).normalize()

    df = binned.join(fh, how="inner").dropna(subset=["open", "fh_high", "fh_close"])
    if df.empty:
        return df

    target = df["open"] * (1.0 + cfg.up_threshold)
    if cfg.move_metric == "close":
        df["outcome"] = (df["fh_close"] >= target).astype(int)
    else:  # "high" (touch-based)
        df["outcome"] = (df["fh_high"] >= target).astype(int)
    # realized P&L proxy for a "buy the open, exit at first-hour close" hold,
    # used by the backtest for net-of-cost expectancy.
    df["fh_ret"] = df["fh_close"] / df["open"] - 1.0
    return df


def _condition_key(row: dict, features: tuple[str, ...]) -> tuple:
    return tuple(row.get(f, "unknown") for f in features)


def screen_ticker(ticker: str, provider: Provider, cfg) -> ScreenResult | None:
    try:
        daily = provider.daily(ticker, cfg.lookback_days)
        fh = provider.first_hour(ticker)
        ds = build_dataset(daily, fh, cfg)
        if ds.empty:
            return None

        # unconditional base rate over the SAME sample as the conditional
        base_successes = int(ds["outcome"].sum())
        base_n = int(len(ds))
        if base_n == 0:
            return None

        # today's condition
        snap = provider.snapshot(ticker)
        today_bins = today_feature_bins(daily, snap.gap)
        if not today_bins:
            return None
        key = _condition_key(today_bins, cfg.active_features)

        # historical days that match today's bins across the active features
        mask = np.ones(len(ds), dtype=bool)
        for f, v in zip(cfg.active_features, key):
            mask &= (ds[f].values == v)
        cond = ds.loc[mask]
        cond_n = int(len(cond))
        if cond_n < cfg.min_samples:
            log.info("%s: only %d matched days (< %d) — skipped.",
                     ticker, cond_n, cfg.min_samples)
            return None
        cond_successes = int(cond["outcome"].sum())

        est = estimate_conditional_rate(
            cond_successes, cond_n, base_successes, base_n, cfg.confidence
        )
        condition = {f: v for f, v in zip(cfg.active_features, key)}
        return ScreenResult(ticker=ticker, estimate=est,
                            condition=condition, today_gap=snap.gap)
    except Exception as e:  # noqa: BLE001
        log.warning("screen_ticker failed for %s: %s", ticker, e)
        return None


def _rank_key(r: ScreenResult, cfg):
    e = r.estimate
    primary = e.lo if cfg.rank_by == "wilson_lower" else e.point
    return (primary, e.lift, e.n)


def screen_universe(tickers: list[str], provider: Provider, cfg) -> tuple[list[ScreenResult], dict]:
    """Screen every ticker, FDR-correct across the universe, rank survivors."""
    raw: list[ScreenResult] = []
    for tk in tickers:
        res = screen_ticker(tk, provider, cfg)
        if res is not None:
            raw.append(res)

    meta = {"scanned": len(tickers), "with_sample": len(raw)}
    if not raw:
        return [], meta

    # FDR across all tested tickers
    pvals = np.array([r.estimate.p_value for r in raw])
    rejected, qvals = benjamini_hochberg(pvals, cfg.fdr_alpha)
    for r, q, rej in zip(raw, qvals, rejected):
        r.qvalue = float(q)
        r.fdr_pass = bool(rej)

    # apply honesty gates
    survivors = [
        r for r in raw
        if (r.estimate.lift >= cfg.min_lift)
        and (r.fdr_pass or not cfg.require_fdr_significant)
    ]
    survivors.sort(key=lambda r: _rank_key(r, cfg), reverse=True)

    meta["fdr_significant"] = int(sum(r.fdr_pass for r in raw))
    meta["survivors"] = len(survivors)
    return survivors[: cfg.top_n], meta
