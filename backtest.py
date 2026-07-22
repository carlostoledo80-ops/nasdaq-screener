"""
Walk-forward validation — the honesty check on the whole idea.

The daily screener is only worth running if the conditioning generalizes.
This module learns "favorable" condition keys on a TRAIN slice and then
measures, on a held-out TEST slice it never saw, whether those conditions
actually delivered a higher first-hour up-rate — and whether the edge
survives transaction costs.

Run:
    python -m nasdaq_screener.backtest                 # samples the universe
    python -m nasdaq_screener.backtest AAPL MSFT NVDA  # specific tickers

Read the output honestly: if OOS lift hovers near zero or the net
expectancy is negative, the pretty daily percentages are noise. Better to
know that than to trade it.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CONFIG
from .data import make_provider
from .engine import build_dataset
from .stats import wilson_interval
from .universe import get_ndx100

log = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    ticker: str
    train_base: float
    test_base: float
    flagged_test_rate: float
    oos_lift: float
    n_flagged_test: int
    net_expectancy: float      # mean first-hour return on flagged test days, net of cost
    gross_expectancy: float


def _favorable_keys(train: pd.DataFrame, cfg) -> set[tuple]:
    """Condition keys that beat the train base rate with enough sample."""
    base = train["outcome"].mean()
    keys = set()
    grp = train.groupby(list(cfg.active_features))["outcome"]
    for key, s in grp:
        n = len(s)
        if n < cfg.min_samples:
            continue
        lo, _, _ = wilson_interval(int(s.sum()), n, cfg.confidence)
        if lo > base + 0.0 and (s.mean() - base) >= cfg.min_lift:
            keys.add(key if isinstance(key, tuple) else (key,))
    return keys


def walk_forward(ticker: str, provider, cfg, train_frac: float = 0.6) -> WalkForwardResult | None:
    daily = provider.daily(ticker, cfg.lookback_days)
    fh = provider.first_hour(ticker)
    ds = build_dataset(daily, fh, cfg)
    if len(ds) < max(120, cfg.min_samples * 3):
        return None
    ds = ds.sort_index()
    split = int(len(ds) * train_frac)
    train, test = ds.iloc[:split], ds.iloc[split:]
    if test.empty:
        return None

    fav = _favorable_keys(train, cfg)
    if not fav:
        return WalkForwardResult(ticker, train["outcome"].mean(),
                                 test["outcome"].mean(), float("nan"), float("nan"),
                                 0, float("nan"), float("nan"))

    test_keys = list(zip(*[test[f].values for f in cfg.active_features]))
    flag = np.array([k in fav for k in test_keys])
    flagged = test.loc[flag]
    n_flag = int(len(flagged))
    if n_flag == 0:
        return WalkForwardResult(ticker, train["outcome"].mean(),
                                 test["outcome"].mean(), float("nan"), float("nan"),
                                 0, float("nan"), float("nan"))

    flagged_rate = flagged["outcome"].mean()
    test_base = test["outcome"].mean()
    gross = flagged["fh_ret"].mean()
    net = gross - cfg.roundtrip_cost
    return WalkForwardResult(
        ticker=ticker,
        train_base=train["outcome"].mean(),
        test_base=test_base,
        flagged_test_rate=flagged_rate,
        oos_lift=flagged_rate - test_base,
        n_flagged_test=n_flag,
        net_expectancy=net,
        gross_expectancy=gross,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = CONFIG
    provider = make_provider(cfg)

    tickers = sys.argv[1:] or get_ndx100()[:25]  # sample to keep runtime sane
    rows = []
    for tk in tickers:
        try:
            r = walk_forward(tk, provider, cfg)
        except Exception as e:  # noqa: BLE001
            log.warning("walk_forward failed for %s: %s", tk, e)
            r = None
        if r is not None:
            rows.append(r)

    if not rows:
        print("No tickers produced enough data for walk-forward validation.")
        return 0

    df = pd.DataFrame([r.__dict__ for r in rows])
    valid = df.dropna(subset=["oos_lift"])
    print("\n=== Walk-forward (train 60% / test 40%) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if not valid.empty:
        print("\n--- Aggregate over tickers with flagged test days ---")
        print(f"mean OOS lift      : {valid['oos_lift'].mean():+.4f}")
        print(f"median OOS lift    : {valid['oos_lift'].median():+.4f}")
        print(f"mean net expectancy: {valid['net_expectancy'].mean():+.4f}  "
              f"(gross {valid['gross_expectancy'].mean():+.4f}, "
              f"cost {cfg.roundtrip_cost:.4f})")
        print(f"share OOS lift > 0 : {(valid['oos_lift'] > 0).mean():.2f}")
        print("\nIf mean OOS lift and net expectancy are not comfortably "
              "positive, treat the daily signal as noise.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
