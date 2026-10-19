import numpy as np
import pytest

from fcr_fvg.params import StrategyParams
from fcr_fvg.signals import opening_range, scan_session
from tests.helpers import LONG_BARS, long_session, mirror, mk_session

P = StrategyParams()


def test_opening_range_uses_only_first_five_minutes():
    s = long_session()
    s.h[5] = 500.0           # a spike at 09:35 must NOT enter H
    s.l[5] = -5.0
    assert opening_range(s) == (101.0, 99.0)


def test_opening_range_requires_all_five_bars():
    s = long_session(); s.o[3] = s.h[3] = s.l[3] = s.c[3] = np.nan
    assert opening_range(s) is None
    assert scan_session(s, P) == []


def test_canonical_long_signal():
    sigs = scan_session(long_session(), P)
    assert len(sigs) == 1
    g = sigs[0]
    assert (g.direction, g.c1_idx, g.fvg_idx, g.retest_idx, g.signal_idx) == (1, 10, 12, 13, 14)
    assert (g.zone_lo, g.zone_hi) == (100.90, 101.00)
    assert g.ext_low == 100.85 and g.or_high == 101.0


def test_canonical_short_is_exact_mirror():
    sigs = scan_session(mirror(long_session()), P)
    assert len(sigs) == 1
    g = sigs[0]
    assert (g.direction, g.fvg_idx, g.retest_idx, g.signal_idx) == (-1, 12, 13, 14)
    assert g.zone_lo == pytest.approx(99.00) and g.zone_hi == pytest.approx(99.10)  # [High(c3), Low(c1)]


def test_breakout_without_fvg_is_not_a_signal():
    bars = dict(LONG_BARS)
    bars[12] = (101.5, 102.0, 100.80, 101.9)      # c3.low <= c1.high -> no gap
    assert scan_session(mk_session(bars), P) == []


def test_gap_equal_to_high_is_not_a_gap():
    bars = dict(LONG_BARS)
    bars[12] = (101.5, 102.0, 100.90, 101.9)      # Low(c3) == High(c1): strict inequality required
    assert scan_session(mk_session(bars), P) == []


def test_second_candle_must_open_at_or_below_H():
    bars = dict(LONG_BARS)
    bars[11] = (101.05, 101.60, 100.95, 101.5)    # opens ABOVE H=101
    assert scan_session(mk_session(bars), P) == []


def test_second_candle_must_close_strictly_above_H():
    bars = dict(LONG_BARS)
    bars[11] = (100.90, 101.60, 100.90, 101.00)   # close == H
    assert scan_session(mk_session(bars), P) == []


def test_fvg_must_form_after_0935():
    # pattern whose c1 is bar 4 (inside the opening range) must not count
    bars = {
        4: (100.5, 100.90, 100.4, 100.8),
        5: (100.9, 101.6, 100.9, 101.5),
        6: (101.5, 102.0, 101.0, 101.9),
        7: (101.8, 101.8, 100.95, 101.0),
        8: (100.98, 102.2, 100.85, 102.0),
    }
    s = mk_session(bars, or_bar=(100.0, 101.0, 99.0, 100.0))
    s.o[4], s.h[4], s.l[4], s.c[4] = bars[4]
    assert scan_session(s, P) == []


def test_retest_on_third_candle_itself_does_not_count():
    bars = dict(LONG_BARS)
    # nothing touches the zone after c3
    bars[13] = (101.9, 102.0, 101.5, 101.6)
    bars[14] = (101.6, 102.2, 101.5, 102.0)
    assert scan_session(mk_session(bars), P) == []


@pytest.mark.parametrize("retest_idx,expect", [(27, True), (28, False)])
def test_fvg_ttl_15_bars(retest_idx, expect):
    # FVG completes at bar 12 -> retest allowed on bars 13..27 (15 bars), not 28
    bars = {k: v for k, v in LONG_BARS.items() if k <= 12}
    for i in range(13, retest_idx):
        bars[i] = (102.0, 102.1, 101.9, 102.0)            # stays away from the zone
    bars[retest_idx] = (101.8, 101.8, 100.95, 101.0)      # bearish retest
    bars[retest_idx + 1] = (100.98, 102.2, 100.85, 102.0) # bullish engulfing
    sigs = scan_session(mk_session(bars), P)
    assert (len(sigs) == 1) is expect


@pytest.mark.parametrize("m,expect", [(0, True), (1, True), (2, True), (3, False)])
def test_confirmation_window_is_retest_bar_plus_two(m, expect):
    """Retest bar k=13. The engulfing bar is k+m. Allowed: m in {0,1,2}."""
    bars = {k: v for k, v in LONG_BARS.items() if k <= 11}
    bars[12] = (101.90, 102.00, 101.00, 101.00)               # c3 bearish (Low 101.00 > High(c1) 100.90)
    if m == 0:
        bars[13] = (100.98, 102.20, 100.95, 102.00)           # retest AND engulfs c3 on the same bar
    else:
        bars[13] = (101.50, 101.50, 100.95, 101.00)           # bearish retest
        for i in range(14, 13 + m):
            bars[i] = (101.10, 101.10, 101.00, 101.05)        # bearish, no engulfing
        bars[13 + m] = (101.00, 102.20, 100.90, 102.00)       # bullish engulfing of the previous bar
    sigs = scan_session(mk_session(bars), P)
    assert (len(sigs) == 1) is expect
    if expect:
        assert sigs[0].retest_idx == 13 and sigs[0].signal_idx == 13 + m


def test_engulfing_requires_previous_bar_opposite_colour_and_body_engulf():
    base = dict(LONG_BARS)
    # previous bar bullish -> not an engulfing
    b = dict(base); b[13] = (101.0, 101.8, 100.95, 101.8)
    assert scan_session(mk_session(b), P) == []
    # open above previous close -> fails Open(cur) <= Close(prev)
    b = dict(base); b[14] = (101.05, 102.2, 100.85, 102.0)
    assert scan_session(mk_session(b), P) == []
    # close below previous open -> fails Close(cur) >= Open(prev)
    b = dict(base); b[14] = (100.98, 102.2, 100.85, 101.79)
    assert scan_session(mk_session(b), P) == []
    # equality on both boundaries is accepted (<= and >=)
    b = dict(base); b[14] = (101.00, 102.2, 100.85, 101.80)
    assert len(scan_session(mk_session(b), P)) == 1


def test_entry_cutoff_1100():
    # Same pattern shifted so that the signal bar is 10:58 (idx 88) -> fill 10:59 OK ; idx 89 -> fill 11:00 refused
    def shifted(k):
        bars = {i + k: v for i, v in LONG_BARS.items() if i <= 14}
        return mk_session(bars)
    assert len(scan_session(shifted(74), P)) == 1      # signal idx 88
    assert len(scan_session(shifted(75), P)) == 0      # signal idx 89 -> fill idx 90 == 11:00


def test_missing_bar_breaks_consecutive_pattern():
    s = long_session()
    s.o[11] = s.h[11] = s.l[11] = s.c[11] = np.nan     # c2 missing: the FVG is not formed
    assert scan_session(s, P) == []


def test_missing_previous_bar_blocks_engulfing():
    s = long_session()
    s.o[13] = s.h[13] = s.l[13] = s.c[13] = np.nan     # retest bar absent -> no retest, no engulf
    assert scan_session(s, P) == []


def test_no_lookahead_signal_independent_of_future_bars():
    """The signal at bar t must be identical when every bar after t is destroyed."""
    full = long_session()
    sig = scan_session(full, P)[0]
    cut = long_session()
    for i in range(sig.signal_idx + 1, 390):
        cut.o[i] = cut.h[i] = cut.l[i] = cut.c[i] = np.nan
    sig2 = scan_session(cut, P)
    assert len(sig2) == 1 and sig2[0] == sig


def test_no_signal_before_0938():
    # fastest conceivable chain: c1=5,c2=6,c3=7, retest/engulf=8 -> signal bar idx 8 (09:38), fill 09:39
    bars = {
        5: (100.50, 100.90, 100.40, 100.80),
        6: (100.90, 101.60, 100.90, 101.50),
        7: (101.50, 102.00, 101.00, 101.90),
        8: (100.98, 102.2, 100.85, 102.0),
    }
    s = mk_session(bars)
    # engulf needs a bearish previous bar: bar 7 bullish -> make a variant where c3 is bearish
    bars[7] = (101.90, 102.00, 101.00, 101.00)
    s = mk_session(bars)
    sigs = scan_session(s, P)
    assert len(sigs) == 1 and sigs[0].signal_idx == 8 and sigs[0].fvg_idx == 7
    assert all(g.signal_idx >= 8 for g in sigs)
