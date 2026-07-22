"""
Conditioning features — what makes 'today' comparable to past days.

Discipline that keeps this honest:
  * Every feature for session d is known BEFORE d's open. Indicators are
    computed on daily closes and then shifted by one day, so a feature on
    row d reflects information through the close of d-1.
  * The one exception is `gap` (d's open vs d-1 close). Historically we use
    the official open as the gap; live at ~8:20 COT we use the premarket
    last price. Premarket gap and official-open gap can differ — a known,
    documented approximation, not a hidden leak.
  * Bins are either fixed economic thresholds (gap, trend, RSI, range,
    volume) or TRAILING-window percentiles (volatility regime). No bin edge
    is ever learned from the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------- indicators
def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / n, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr_pct(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    return atr / close


def compute_indicators(daily: pd.DataFrame) -> pd.DataFrame:
    """Return per-date features, each aligned to be known BEFORE that day's open."""
    if daily.empty:
        return pd.DataFrame()
    d = daily.copy()
    close, high, low, vol = d["Close"], d["High"], d["Low"], d["Volume"]
    prev_close = close.shift(1)

    gap = d["Open"] / prev_close - 1.0                    # uses today's open (documented)
    trend_5d = (close.shift(1) / close.shift(6) - 1.0)    # momentum through prior close
    atr_pct = _atr_pct(d).shift(1)                        # vol as of prior close
    rsi = _rsi(close).shift(1)
    range_pos = ((close - low) / (high - low).replace(0, np.nan)).shift(1)  # prior-day range
    vol_ratio = (vol.shift(1) / vol.rolling(20).mean().shift(1))

    out = pd.DataFrame(
        {
            "gap": gap,
            "trend_5d": trend_5d,
            "atr_pct": atr_pct,
            "rsi": rsi,
            "range_pos": range_pos,
            "vol_ratio": vol_ratio,
            "dow": pd.Series(d.index.dayofweek, index=d.index),
        }
    )
    return out


# ---------------------------------------------------------------------- binning
def _gap_bin(g: float) -> str:
    if g <= -0.01:  return "gap_dn_big"
    if g <= -0.0025: return "gap_dn"
    if g < 0.0025:  return "flat"
    if g < 0.01:    return "gap_up"
    return "gap_up_big"


def _trend_bin(t: float) -> str:
    if t <= -0.03: return "trend_dn_strong"
    if t <= -0.01: return "trend_dn"
    if t < 0.01:   return "trend_neutral"
    if t < 0.03:   return "trend_up"
    return "trend_up_strong"


def _rsi_bin(r: float) -> str:
    if r < 30:  return "rsi_oversold"
    if r < 45:  return "rsi_weak"
    if r < 55:  return "rsi_neutral"
    if r < 70:  return "rsi_strong"
    return "rsi_overbought"


def _range_pos_bin(p: float) -> str:
    if p < 0.33: return "closed_low"
    if p < 0.66: return "closed_mid"
    return "closed_high"


def _vol_ratio_bin(v: float) -> str:
    if v < 0.8:  return "vol_light"
    if v < 1.2:  return "vol_normal"
    if v < 2.0:  return "vol_high"
    return "vol_surge"


def _dow_bin(x: float) -> str:
    return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][int(x)]


def _vol_regime_series(atr_pct: pd.Series, window: int = 252) -> pd.Series:
    """Trailing-percentile regime for volatility — no look-ahead."""
    pct = atr_pct.rolling(window, min_periods=60).apply(
        lambda w: float((w[-1] >= w).mean()), raw=True
    )

    def to_bin(p: float) -> str:
        if np.isnan(p): return "volr_unknown"
        if p < 0.34: return "volr_low"
        if p < 0.67: return "volr_mid"
        return "volr_high"

    return pct.map(to_bin)


def bin_features(ind: pd.DataFrame) -> pd.DataFrame:
    if ind.empty:
        return ind
    b = pd.DataFrame(index=ind.index)
    b["gap_bin"] = ind["gap"].map(lambda x: _gap_bin(x) if pd.notna(x) else "unknown")
    b["trend_bin"] = ind["trend_5d"].map(lambda x: _trend_bin(x) if pd.notna(x) else "unknown")
    b["vol_regime_bin"] = _vol_regime_series(ind["atr_pct"])
    b["rsi_bin"] = ind["rsi"].map(lambda x: _rsi_bin(x) if pd.notna(x) else "unknown")
    b["range_pos_bin"] = ind["range_pos"].map(lambda x: _range_pos_bin(x) if pd.notna(x) else "unknown")
    b["volume_bin"] = ind["vol_ratio"].map(lambda x: _vol_ratio_bin(x) if pd.notna(x) else "unknown")
    b["dow"] = ind["dow"].map(_dow_bin)
    return b


def today_feature_bins(daily: pd.DataFrame, live_gap: float | None) -> dict[str, str]:
    """Feature bins for the UPCOMING session, using latest data + live gap.

    We reuse the historical indicator pipeline, then override the last row's
    gap with the live premarket gap (the only feature that references the
    session being predicted).
    """
    ind = compute_indicators(daily)
    if ind.empty:
        return {}
    binned = bin_features(ind)
    last = binned.iloc[-1].to_dict()
    if live_gap is not None:
        last["gap_bin"] = _gap_bin(live_gap)
    return last
