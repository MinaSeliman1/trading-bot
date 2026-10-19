"""Hand-built bar scenarios used ONLY by the test-suite.

These are controlled fixtures (not market data, never used for performance
numbers): every price below was chosen by hand so that expected results can be
computed on paper.
"""
import datetime as dt

import numpy as np

from fcr_fvg.data import RTH_MINUTES, Session


def mk_session(bars: dict, base: float = 100.0, date=dt.date(2024, 3, 12), expected_n=RTH_MINUTES,
               or_bar=(100.0, 101.0, 99.0, 100.0), missing=()) -> Session:
    """bars: {idx: (o, h, l, c)}. Unspecified bars after the opening range are flat at `base`.
    Bars 0-4 default to `or_bar` (=> H=101, L=99)."""
    o = np.full(RTH_MINUTES, np.nan); h = o.copy(); l = o.copy(); c = o.copy(); v = o.copy()
    for i in range(expected_n):
        if i < 5:
            b = or_bar
        else:
            b = (base, base, base, base)
        o[i], h[i], l[i], c[i] = b
        v[i] = 1000.0
    for i, b in bars.items():
        o[i], h[i], l[i], c[i] = b
    for i in missing:
        o[i] = h[i] = l[i] = c[i] = np.nan
    return Session(date, o, h, l, c, v, expected_n=expected_n)


def mirror(s: Session, axis: float = 200.0) -> Session:
    """Price reflection: turns a LONG scenario into the exact SHORT counterpart."""
    return Session(s.date, axis - s.o, axis - s.l, axis - s.h, axis - s.c, s.v.copy(), s.expected_n)


# Canonical LONG scenario (SPY-like, tick 0.01). H=101, L=99.
#  idx 10-12 : FVG (c1.high=100.90, c3.low=101.00)  -> zone [100.90, 101.00]
#  idx 13    : retest (bearish bar, low 100.95 inside zone)
#  idx 14    : bullish engulfing (signal bar) ; low 100.85
#  idx 15    : entry bar, open 102.00
#  idx 16    : target bar
LONG_BARS = {
    10: (100.50, 100.90, 100.40, 100.80),
    11: (100.90, 101.60, 100.90, 101.50),
    12: (101.50, 102.00, 101.00, 101.90),
    13: (101.80, 101.80, 100.95, 101.00),
    14: (100.98, 102.20, 100.85, 102.00),
    15: (102.00, 102.30, 101.90, 102.20),
    16: (102.20, 105.60, 102.10, 105.50),
}


def long_session(**over):
    bars = dict(LONG_BARS)
    bars.update(over)
    return mk_session(bars)
