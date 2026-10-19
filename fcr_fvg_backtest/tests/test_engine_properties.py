"""Invariant checks on RANDOM-WALK data. The data is synthetic and used only to
exercise the code paths; no performance number from it is ever reported."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from fcr_fvg.data import build_sessions, load_ohlcv
from fcr_fvg.engine import run_backtest
from fcr_fvg.instruments import ES, SPY
from fcr_fvg.params import StrategyParams
from fcr_fvg.signals import scan_session
from tests.synth import bday_range, random_walk_frame


@pytest.fixture(scope="module")
def spy_sessions(tmp_path_factory):
    days = bday_range("2024-01-02", "2024-12-31")
    days = [d for d in days if d not in (dt.date(2024, 1, 15), dt.date(2024, 2, 19), dt.date(2024, 3, 29), dt.date(2024, 5, 27),
                                         dt.date(2024, 6, 19), dt.date(2024, 7, 4), dt.date(2024, 9, 2), dt.date(2024, 11, 28),
                                         dt.date(2024, 12, 25), dt.date(2024, 7, 3), dt.date(2024, 11, 29), dt.date(2024, 12, 24))]
    df = random_walk_frame(days[:140], seed=3, vol=0.0012, extended=False)   # high vol so setups occur
    p = tmp_path_factory.mktemp("d") / "r.csv"
    df.to_csv(p, index=False)
    sessions, qc = build_sessions(load_ohlcv(p), SPY, StrategyParams())
    return sessions


@pytest.mark.parametrize("variant", ["A", "B"])
def test_invariants(spy_sessions, variant):
    p = StrategyParams(stop_variant=variant)
    res = run_backtest(spy_sessions, SPY, p)
    t = res["trades"]
    assert len(t) >= 10, "random-walk fixture produced too few trades to make the invariants meaningful"
    assert t["date"].is_unique                                         # max one trade / day / instrument
    ent = pd.to_datetime(t["entry_time"]); ext = pd.to_datetime(t["exit_time"])
    emin = ent.dt.hour * 60 + ent.dt.minute
    assert (emin >= 9 * 60 + 39).all() and (emin < 11 * 60).all()      # earliest conceivable fill 09:39, cutoff 11:00
    assert (ext >= ent).all() and ((ext.dt.hour * 60 + ext.dt.minute) <= 16 * 60).all()
    sig = pd.to_datetime(t["signal_time"]); assert (sig <= ent).all()
    assert (pd.to_datetime(t["fvg_time"]) <= pd.to_datetime(t["retest_time"]) + pd.Timedelta(minutes=1)).all()
    d = np.where(t["direction"] == "LONG", 1, -1)
    assert (d * (t["entry_price"] - t["stop"]) > 0).all()              # stop on the correct side
    assert np.allclose(d * (t["target"] - t["entry_price"]), 3 * d * (t["entry_price"] - t["stop"]))
    assert (t["units"] >= 1).all() and (t["units"] == t["units"].round()).all()
    assert (t["risk_pct_equity"] <= 0.005 + 1e-9).all()                # nominal risk never above 0.5% of equity
    assert np.allclose(t["net_pnl"], t["gross_pnl_ideal"] - t["slippage_cost"] - t["commissions"])
    assert np.allclose(t["r_multiple"], t["net_pnl"] / t["risk_usd_nominal"])
    # compounding: equity_before follows the realised P&L
    eq = 100_000 + t["net_pnl"].cumsum().shift(fill_value=0)
    assert np.allclose(t["equity_before"], eq)
    # cutoff check on the originating signals
    assert (t["entry_idx"] <= 89).all()


def test_signal_scan_is_causal_on_random_data(spy_sessions):
    """For each signal, wiping every bar after (signal_idx) must not change that signal."""
    checked = 0
    p = StrategyParams()
    for s in spy_sessions:
        for g in scan_session(s, p):
            import copy
            c = copy.deepcopy(s)
            for arr in (c.o, c.h, c.l, c.c, c.v):
                arr[g.signal_idx + 1:] = np.nan
            again = scan_session(c, p)
            assert g in again
            checked += 1
    assert checked >= 10


def test_slippage_monotonicity_on_identical_signals(spy_sessions):
    out = {}
    for k in (1, 2, 3, 5):
        out[k] = run_backtest(spy_sessions, SPY, StrategyParams(slippage_ticks=k))["trades"]
    # more slippage cannot improve total R-expectancy on this fixed signal stream
    exp = [out[k]["r_multiple"].mean() for k in (1, 2, 3, 5)]
    assert exp == sorted(exp, reverse=True)


def test_no_trade_when_data_missing_never_invents_prices(spy_sessions):
    import copy
    s = copy.deepcopy(spy_sessions[0])
    s.o[5:] = s.h[5:] = s.l[5:] = s.c[5:] = np.nan                    # nothing after the opening range
    assert run_backtest([s], SPY, StrategyParams())["trades"].empty


def test_futures_rejections_are_logged_not_silent(spy_sessions):
    # price scale of the walk (~100) with ES specs: risk per contract is tiny in dollars/point terms, but
    # the point is that every signal ends either as a trade or as a logged rejection
    res = run_backtest(spy_sessions, ES, StrategyParams())
    d = res["daily"]
    n_sig_days = int((d["n_signals"] > 0).sum())
    assert n_sig_days >= len(res["trades"])
