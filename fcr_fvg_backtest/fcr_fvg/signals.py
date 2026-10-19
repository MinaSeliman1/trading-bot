"""Signal detection: Opening range -> breakout FVG -> retest -> engulfing candle.

Index convention: bar `i` is the 1-minute bar that OPENS at 09:30 + i minutes.
The loop variable `t` is the bar that has JUST CLOSED. Nothing at an index > t
is ever read while processing t (enforced by a truncation test in tests/).

Interpretation choices (all documented in README.md):
  * The 3 bars of the FVG pattern all start at or after 09:35 (index >= 5).
  * c2 must OPEN <= H (LONG) / >= L (SHORT) AND CLOSE strictly beyond the level
    (strict "body crosses the level" reading).
  * An FVG exists only once c3 has closed (bar t == c3). It becomes testable
    from bar t+1. It stays active for `fvg_ttl_bars` bars: t+1 .. t+ttl.
  * Retest: a later bar whose [low, high] intersects the closed zone.
  * Confirmation: engulfing bar on the retest bar itself or on the following
    `confirm_extra_bars` bars; the engulfed bar must be the immediately
    preceding bar and must be present.
  * Several FVG candidates can be alive simultaneously (possibly both
    directions). Every confirmed setup is emitted in chronological order; the
    engine takes the first one that survives the execution checks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Session
from .params import OR_BARS, StrategyParams


@dataclass(frozen=True)
class Signal:
    direction: int            # +1 LONG, -1 SHORT
    or_high: float
    or_low: float
    c1_idx: int
    fvg_idx: int              # index of c3 (FVG complete when this bar closes)
    zone_lo: float            # LONG: High(c1)  SHORT: High(c3)
    zone_hi: float            # LONG: Low(c3)   SHORT: Low(c1)
    retest_idx: int
    signal_idx: int           # engulfing bar (closes -> order sent; fill at signal_idx + 1 open)
    ext_low: float            # lowest low  over retest_idx..signal_idx (present bars)
    ext_high: float           # highest high over retest_idx..signal_idx


@dataclass
class _Cand:
    direction: int
    c1: int
    formed: int
    zone_lo: float
    zone_hi: float
    state: str = "wait"       # wait -> retest
    retest: int = -1
    ext_lo: float = np.inf
    ext_hi: float = -np.inf
    dead: bool = False


def _is_engulfing(direction: int, o, c, t: int) -> bool:
    p = t - 1
    if p < 0 or not (np.isfinite(o[p]) and np.isfinite(o[t])):
        return False
    if direction == 1:
        return (c[p] < o[p] and c[t] > o[t]          # bearish then bullish
                and o[t] <= c[p] and c[t] >= o[p])  # body engulfs previous body
    return (c[p] > o[p] and c[t] < o[t]
            and o[t] >= c[p] and c[t] <= o[p])


def opening_range(s: Session) -> tuple[float, float] | None:
    """H and L from the five 1-minute bars 09:30-09:34 only."""
    if not np.isfinite(s.o[:OR_BARS]).all():
        return None
    return float(np.max(s.h[:OR_BARS])), float(np.min(s.l[:OR_BARS]))


def scan_session(s: Session, p: StrategyParams) -> list[Signal]:
    rng = opening_range(s)
    if rng is None:
        return []
    H, L = rng
    o, h, l, c = s.o, s.h, s.l, s.c
    # fill bar (t+1) must start before the cutoff -> t <= cutoff-2 ; and exist in the session
    last_t = min(p.entry_cutoff_idx - 2, s.expected_n - 2)
    cands: list[_Cand] = []
    out: list[Signal] = []

    for t in range(OR_BARS, last_t + 1):
        here = bool(np.isfinite(o[t]))

        # ---- 1. advance candidates created strictly before t -------------
        for cd in cands:
            if cd.dead:
                continue
            if cd.state == "wait":
                if t - cd.formed > p.fvg_ttl_bars:
                    cd.dead = True
                    continue
                if not here:
                    continue
                if l[t] <= cd.zone_hi and h[t] >= cd.zone_lo:     # range intersects zone
                    cd.state = "retest"
                    cd.retest = t
                else:
                    continue
            # state == "retest"
            if t - cd.retest > p.confirm_extra_bars:
                cd.dead = True
                continue
            if not here:
                continue
            cd.ext_lo = min(cd.ext_lo, l[t])
            cd.ext_hi = max(cd.ext_hi, h[t])
            if _is_engulfing(cd.direction, o, c, t):
                out.append(Signal(cd.direction, H, L, cd.c1, cd.formed, cd.zone_lo, cd.zone_hi,
                                  cd.retest, t, float(cd.ext_lo), float(cd.ext_hi)))
                cd.dead = True

        # ---- 2. FVG whose third bar is t (needs t-2 >= 5, i.e. all bars after 09:35) --------
        c1, c2 = t - 2, t - 1
        if c1 >= OR_BARS and here and np.isfinite(o[c1]) and np.isfinite(o[c2]):
            if o[c2] <= H and c[c2] > H and l[t] > h[c1]:                   # bullish
                cands.append(_Cand(1, c1, t, float(h[c1]), float(l[t])))
            elif o[c2] >= L and c[c2] < L and h[t] < l[c1]:                 # bearish
                cands.append(_Cand(-1, c1, t, float(h[t]), float(l[c1])))
        cands = [cd for cd in cands if not cd.dead]
    return out
