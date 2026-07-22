"""
Nasdaq 100 constituents.

Fetched live so the universe stays current as the index reconstitutes.
If the live fetch fails (network hiccup, layout change), we fall back to a
bundled snapshot so the daily job never dies silently. The snapshot will
drift over time — the live path is the source of truth.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Offline fallback only. Not guaranteed current — the live fetch overrides it.
_FALLBACK_NDX100 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "AVGO", "TSLA",
    "COST", "NFLX", "AMD", "PEP", "ADBE", "LIN", "CSCO", "TMUS", "QCOM", "INTU",
    "AMAT", "TXN", "AMGN", "ISRG", "CMCSA", "HON", "BKNG", "MU", "VRTX", "LRCX",
    "ADI", "REGN", "PANW", "ADP", "KLAC", "SBUX", "GILD", "MDLZ", "SNPS", "CDNS",
    "MELI", "CRWD", "PYPL", "MAR", "CTAS", "ORLY", "ASML", "NXPI", "ABNB", "PDD",
    "CEG", "MRVL", "WDAY", "FTNT", "DASH", "MNST", "ADSK", "ROP", "CHTR", "PCAR",
    "KDP", "AEP", "PAYX", "ODFL", "FANG", "TEAM", "ROST", "CPRT", "FAST", "EA",
    "KHC", "GEHC", "DDOG", "CSGP", "VRSK", "XEL", "CTSH", "BKR", "LULU", "EXC",
    "TTWO", "IDXX", "ON", "CCEP", "ANSS", "ZS", "DXCM", "MCHP", "WBD", "CDW",
    "GFS", "BIIB", "ILMN", "MDB", "TTD", "ARM", "SMCI", "MRNA", "WBA", "DLTR",
]


def get_ndx100() -> list[str]:
    """Return current Nasdaq 100 tickers. Live first, bundled fallback second."""
    try:
        import pandas as pd

        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        for t in tables:
            cols = {str(c).strip().lower() for c in t.columns}
            if "ticker" in cols or "symbol" in cols:
                col = "Ticker" if "Ticker" in t.columns else (
                    "Symbol" if "Symbol" in t.columns else None
                )
                if col is None:
                    for c in t.columns:
                        if str(c).strip().lower() in ("ticker", "symbol"):
                            col = c
                            break
                tickers = [
                    str(x).strip().upper().replace(".", "-")
                    for x in t[col].dropna().tolist()
                ]
                # sanity: NDX100 should give ~100 clean symbols
                tickers = [x for x in tickers if 1 <= len(x) <= 6 and x.isalnum() or "-" in x]
                if 90 <= len(tickers) <= 110:
                    log.info("Fetched %d Nasdaq 100 tickers live.", len(tickers))
                    return sorted(set(tickers))
        raise ValueError("no constituents table matched expected layout")
    except Exception as e:  # noqa: BLE001
        log.warning("Live universe fetch failed (%s); using bundled fallback.", e)
        return sorted(set(_FALLBACK_NDX100))
