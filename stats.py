"""
Honest statistics for conditional base rates.

The whole point of this module is to NOT overstate what the data says.
We never report a bare point estimate as if it were a probability of the
future. Every conditional rate comes with:

  - a sample size (n) — how many historical days actually matched
  - a Wilson confidence interval — the uncertainty around the estimate
  - a comparison against the unconditional base rate — is there real lift?
  - multiple-testing control — because scanning 100 tickers and taking the
    max is exactly how you fool yourself with noise.

Ranking is done by the LOWER bound of the Wilson interval, not the point
estimate. That single choice does most of the anti-overfitting work: a
flashy 90% rate on n=6 gets a low lower bound and sinks; a solid 60% on
n=180 floats to the top. It is a conservative, defensible score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# z-scores for common two-sided confidence levels
_Z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}


def z_for(conf: float) -> float:
    if conf in _Z:
        return _Z[conf]
    # fall back to an accurate inverse-normal for arbitrary levels
    from scipy.stats import norm

    return float(norm.ppf(1 - (1 - conf) / 2))


@dataclass(frozen=True)
class RateEstimate:
    """A conditional up-rate with its uncertainty."""

    successes: int
    n: int
    conf: float
    point: float  # observed frequency
    lo: float  # Wilson lower bound  <-- primary ranking score
    hi: float  # Wilson upper bound
    base_rate: float  # unconditional frequency for the same ticker/outcome
    lift: float  # point - base_rate
    p_value: float  # one-sided P(rate > base_rate) under H0

    @property
    def enough_sample(self) -> bool:
        return self.n > 0


def wilson_interval(successes: int, n: int, conf: float = 0.95) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Far better than the naive normal approximation for small n or rates near
    0/1, which is exactly the regime a stock screener lives in. Returns
    (low, high, point_estimate).
    """
    if n <= 0:
        return (0.0, 0.0, 0.0)
    z = z_for(conf)
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi, phat)


def one_sided_binomial_p(successes: int, n: int, p0: float) -> float:
    """P(X >= successes | Binomial(n, p0)) — evidence the true rate exceeds p0.

    p0 is the unconditional base rate. A small p-value means the conditional
    up-rate is unlikely to be this high by chance alone *relative to the
    stock's own baseline*. This is what we FDR-correct across the universe.
    """
    if n <= 0:
        return 1.0
    from scipy.stats import binomtest

    p0 = min(max(p0, 1e-9), 1 - 1e-9)
    return float(binomtest(successes, n, p0, alternative="greater").pvalue)


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR control.

    Returns (rejected_mask, qvalues). Controls the expected proportion of
    false discoveries among the tickers we flag, which is the right error
    to control when we test the whole Nasdaq 100 every morning. Without
    this, the 'best' name every day is mostly luck.
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    # BH critical values and step-up rejection
    crit = (np.arange(1, m + 1) / m) * alpha
    below = ranked <= crit
    if below.any():
        kmax = np.max(np.where(below)[0])
        thresh = ranked[kmax]
    else:
        thresh = -1.0
    rejected = p <= thresh
    # monotone q-values
    q_sorted = ranked * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.clip(q_sorted, 0, 1)
    return rejected, q


def estimate_conditional_rate(
    cond_successes: int,
    cond_n: int,
    base_successes: int,
    base_n: int,
    conf: float = 0.95,
) -> RateEstimate:
    """Bundle a conditional up-rate with its full honest context."""
    lo, hi, point = wilson_interval(cond_successes, cond_n, conf)
    base_rate = (base_successes / base_n) if base_n > 0 else 0.0
    p_val = one_sided_binomial_p(cond_successes, cond_n, base_rate)
    return RateEstimate(
        successes=cond_successes,
        n=cond_n,
        conf=conf,
        point=point,
        lo=lo,
        hi=hi,
        base_rate=base_rate,
        lift=point - base_rate,
        p_value=p_val,
    )
