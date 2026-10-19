import numpy as np
import pytest

from fcr_fvg.execution import Rejection, Trade, simulate_trade, size_position
from fcr_fvg.instruments import ES, NQ, QQQ, SPY
from fcr_fvg.params import StrategyParams
from fcr_fvg.signals import scan_session
from tests.helpers import LONG_BARS, long_session, mirror, mk_session

EQ = 100_000.0


def run(s, inst=SPY, **kw):
    p = StrategyParams(**kw)
    sig = scan_session(s, p)[0]
    return simulate_trade(s, sig, inst, p, EQ), p


def test_long_variant_a_hand_computed_take_profit():
    t, _ = run(long_session(), stop_variant="A")
    assert isinstance(t, Trade)
    # entry: open 102.00 + 1 tick slippage
    assert t.entry_price == pytest.approx(102.01)
    # stop A: min low over retest..signal = 100.85, minus 1 tick
    assert t.stop == pytest.approx(100.84)
    risk = 102.01 - 100.84                                  # 1.17
    assert t.target == pytest.approx(102.01 + 3 * risk)     # 105.52
    assert t.exit_reason == "tp" and t.exit_price == pytest.approx(105.52 - 0.01)
    # sizing: floor(500 / (1.18 + 0.01)) = 420 shares
    assert t.units == 420
    assert t.commissions == pytest.approx(2 * 420 * 0.005)  # 2.10 per side
    gross = (105.51 - 102.01) * 420                         # 1470.00
    assert t.net_pnl == pytest.approx(gross - 4.20)
    assert t.r_multiple == pytest.approx((gross - 4.20) / (420 * 1.17))
    assert t.risk_pct_equity < 0.005
    assert t.entry_time.strftime("%H:%M") == "09:45" and t.signal_time.strftime("%H:%M") == "09:45"
    assert t.retest_time.strftime("%H:%M") == "09:43"


def test_long_variant_b_uses_fvg_boundary():
    t, _ = run(long_session(), stop_variant="B")
    assert t.stop == pytest.approx(100.89)                  # High(c1) 100.90 - 1 tick
    assert t.risk_ticks == 112                              # 102.01 - 100.89


def test_short_mirror_has_identical_pnl():
    tl, _ = run(long_session(), stop_variant="A")
    ts, _ = run(mirror(long_session()), stop_variant="A")
    assert ts.direction == "SHORT"
    assert ts.entry_price == pytest.approx(200 - 102.00 - 0.01)
    assert ts.stop == pytest.approx(200 - 100.85 + 0.01)
    assert ts.units == tl.units
    assert ts.net_pnl == pytest.approx(tl.net_pnl)
    assert ts.r_multiple == pytest.approx(tl.r_multiple)


def test_stop_and_target_in_same_bar_stop_first():
    s = long_session(**{"16": None}) if False else long_session()
    s.h[16], s.l[16] = 105.60, 100.50                       # bar touches both TP (105.52) and stop (100.84)
    t, _ = run(s, stop_variant="A")
    assert t.exit_reason == "stop"
    assert t.exit_price == pytest.approx(100.84 - 0.01)
    assert t.net_pnl < 0 and t.r_multiple < -1.0            # costs make the loss slightly worse than -1R


def test_gap_through_stop_fills_at_open_not_stop():
    s = long_session()
    s.o[16], s.h[16], s.l[16], s.c[16] = 100.00, 100.10, 99.50, 99.80   # opens far below stop 100.84
    t, _ = run(s, stop_variant="A")
    assert t.exit_reason == "gap_stop"
    assert t.exit_price == pytest.approx(100.00 - 0.01)     # worse than the stop level
    assert t.r_multiple < -1.5


def test_gap_through_target_is_not_credited_with_better_price():
    s = long_session()
    s.o[16], s.h[16], s.l[16], s.c[16] = 110.0, 111.0, 109.0, 110.5     # opens above target
    t, _ = run(s, stop_variant="A")
    assert t.exit_reason == "tp" and t.exit_price == pytest.approx(105.52 - 0.01)


def test_end_of_day_exit_at_last_bar_close_with_slippage():
    s = long_session()
    s.h[16], s.l[16], s.c[16] = 103.0, 102.1, 102.5          # neither stop nor target
    for i in range(17, 390):                                 # market stays between stop and target all day
        s.o[i] = s.h[i] = s.l[i] = s.c[i] = 102.5
    t, _ = run(s, stop_variant="A")
    assert t.exit_reason == "eod"
    assert t.exit_price == pytest.approx(102.50 - 0.01)
    assert t.exit_time.strftime("%H:%M") == "16:00"


def test_early_close_session_exits_at_last_bar_of_the_shortened_session():
    s = mk_session(dict(LONG_BARS), expected_n=210)          # 13:00 close
    s.h[16], s.l[16], s.c[16] = 103.0, 102.1, 102.5
    for i in range(17, 210):
        s.o[i] = s.h[i] = s.l[i] = s.c[i] = 102.5
    t, _ = run(s, stop_variant="A")
    assert t.exit_reason == "eod" and t.exit_time.strftime("%H:%M") == "13:00"


def test_entry_bar_is_checked_for_exits():
    s = long_session()
    s.l[15] = 100.50                                         # entry bar trades through the stop
    t, _ = run(s, stop_variant="A")
    assert t.exit_reason == "stop" and t.exit_idx == 15
    t2, _ = run(s, stop_variant="A", check_exit_on_entry_bar=False)
    assert t2.exit_idx == 16


def test_invalid_stop_is_rejected():
    s = long_session()
    # force the entry bar to open BELOW the FVG-based stop -> stop >= entry
    s.o[15], s.h[15], s.l[15], s.c[15] = 100.50, 100.70, 100.40, 100.60
    r, _ = run(s, stop_variant="B")
    assert isinstance(r, Rejection) and r.reason == "invalid_stop"


def test_missing_entry_bar_rejects_instead_of_inventing_a_price():
    s = long_session()
    s.o[15] = s.h[15] = s.l[15] = s.c[15] = np.nan
    r, _ = run(s)
    assert isinstance(r, Rejection) and r.reason == "entry_bar_missing"


@pytest.mark.parametrize("slip", [1, 2, 3, 5])
def test_slippage_scales_costs_only(slip):
    def sess():
        s = long_session()
        s.h[16] = 107.0                      # target stays reachable for every slippage level
        return s
    t0, _ = run(sess(), slippage_ticks=0)
    t, _ = run(sess(), slippage_ticks=slip)
    # same trade structure ; entry and exit both worse by `slip` ticks
    assert t.entry_price == pytest.approx(102.00 + slip * 0.01)
    assert t.exit_reason == t0.exit_reason == "tp"
    risk = t.entry_price - 100.84
    assert t.target == pytest.approx(t.entry_price + 3 * risk)
    assert t.exit_price == pytest.approx(t.target - slip * 0.01)
    assert t.slippage_cost == pytest.approx(2 * slip * 0.01 * t.units)
    assert t.r_multiple < t0.r_multiple


# ------------------------------- sizing ------------------------------------
def test_size_etf_budget_rule():
    p = StrategyParams()
    u, capped, why = size_position(SPY, EQ, 500.0, 50, p)   # risk 0.50 $/share
    # per unit = (50+1)*0.01 + 0.01 = 0.52 -> floor(500/0.52) = 961 ; notional 480k > 4x -> capped to 800
    assert u == 800 and capped


def test_size_etf_without_cap():
    p = StrategyParams()
    u, capped, _ = size_position(SPY, EQ, 100.0, 117, p)
    assert u == 420 and not capped


def test_futures_reject_when_one_contract_exceeds_risk_budget():
    p = StrategyParams()
    # ES: 12 points = 48 ticks -> (48+1)*12.5 + 4.5 = 617 > 500 -> not tradable at 0.5% of 100k
    u, _, why = size_position(ES, EQ, 5000.0, 48, p)
    assert u == 0 and why == "risk_budget_below_one_unit"
    # ES: 4 points = 16 ticks -> (16+1)*12.5+4.5 = 217 -> 2 contracts
    u, _, _ = size_position(ES, EQ, 5000.0, 16, p)
    assert u == 2


def test_nq_value_per_tick_is_5_dollars_and_point_20():
    assert NQ.tick_value == 5.0 and NQ.point_value == 20.0
    assert ES.tick_value == 12.5 and ES.point_value == 50.0
    u, _, _ = size_position(NQ, EQ, 18000.0, 20, StrategyParams(max_margin_usage=1.0))   # 5 pts -> 21*5+4.5 = 109.5 -> 4 (margin 25k each)
    assert u == 4


def test_futures_margin_cap():
    p = StrategyParams(max_margin_usage=0.1)                 # only 10k of margin usable -> < 1 ES contract
    u, _, why = size_position(ES, EQ, 5000.0, 4, p)
    assert u == 0 and why == "exposure_cap_below_one_unit"


ES_BARS = {
    10: (5001.00, 5003.50, 5000.50, 5003.00),
    11: (5003.50, 5007.00, 5003.50, 5006.00),     # opens at/below H=5004, closes above
    12: (5006.00, 5009.00, 5004.50, 5008.50),     # Low(c3) 5004.50 > High(c1) 5003.50 -> zone [5003.50, 5004.50]
    13: (5008.00, 5008.00, 5004.00, 5004.50),     # bearish retest
    14: (5004.25, 5010.00, 5003.75, 5009.00),     # bullish engulfing
    15: (5009.00, 5011.00, 5008.50, 5010.00),     # entry bar
    16: (5010.00, 5030.00, 5009.50, 5028.00),     # target bar
}


def es_session():
    return mk_session(ES_BARS, base=5000.0, or_bar=(5000.0, 5004.0, 4996.0, 5000.0))


def test_es_trade_hand_computed_uses_point_value_50_and_tick_12_5():
    t, _ = run(es_session(), inst=ES, stop_variant="A")
    assert isinstance(t, Trade)
    assert t.entry_price == 5009.25 and t.stop == 5003.50 and t.target == 5026.50
    assert t.risk_ticks == 23 and t.units == 1
    assert t.exit_reason == "tp" and t.exit_price == 5026.25
    assert t.commissions == pytest.approx(4.50)
    assert t.net_pnl == pytest.approx((5026.25 - 5009.25) * 50 - 4.50)     # 845.50
    assert t.risk_usd_nominal == pytest.approx(5.75 * 50)                  # 287.50
    assert t.r_multiple == pytest.approx(845.5 / 287.5)


def test_es_stop_out_loss_in_dollars():
    s = es_session()
    s.l[16] = 5000.0                                                        # stop 5003.50 traded through (bar also reaches target)
    t, _ = run(s, inst=ES, stop_variant="A")
    assert t.exit_reason == "stop" and t.exit_price == 5003.25
    assert t.net_pnl == pytest.approx((5003.25 - 5009.25) * 50 - 4.50)     # -304.50
    assert t.r_multiple == pytest.approx(-304.5 / 287.5)
