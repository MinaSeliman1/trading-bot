import numpy as np
import pandas as pd
import pytest

from bot.backtest_engine import SimTrade, compute_metrics
from bot.pead_backtest import (
    build_event_trades, build_portfolio_event_trades, equity_curve_from_trades, daily_returns_from_equity,
)


def _synthetic_bars(n=300, seed=1):
    idx = pd.bdate_range("2023-01-01", periods=n, tz="UTC")
    close = 100 + np.cumsum(np.random.default_rng(seed).normal(0.05, 1.0, n))
    return pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5, "close": close}, index=idx)


def test_threshold_filters_events_correctly():
    bars = _synthetic_bars()
    idx = bars.index
    events = pd.DataFrame({
        "surprise_pct": [8.0, -3.0, 12.0, 2.0, 15.0],
        "report_time": ["post-market", "pre-market", "post-market", "post-market", "pre-market"],
    }, index=pd.DatetimeIndex([idx[20], idx[80], idx[140], idx[200], idx[260]]))

    trades = build_event_trades("TEST", events, bars, surprise_threshold_pct=5.0, hold_days=20, slippage_pct=0.001)
    assert len(trades) == 3  # only the 8%, 12%, 15% surprises clear the 5% bar


def test_entry_timing_never_precedes_public_information():
    """post-market news: first tradeable bar is the NEXT trading day (market
    was already closed when it broke). pre-market news: the SAME day's open
    is tradeable since the market opens after the announcement."""
    bars = _synthetic_bars()
    idx = bars.index
    events = pd.DataFrame({
        "surprise_pct": [10.0, 10.0],
        "report_time": ["post-market", "pre-market"],
    }, index=pd.DatetimeIndex([idx[20], idx[260]]))

    trades = build_event_trades("TEST", events, bars, surprise_threshold_pct=5.0, hold_days=20, slippage_pct=0.0)
    post_trade = next(t for t in trades if abs((t.entry_time - idx[20]).days) <= 3)
    pre_trade = next(t for t in trades if abs((t.entry_time - idx[260]).days) <= 3)

    assert post_trade.entry_time > idx[20]
    assert pre_trade.entry_time == idx[260]


def test_events_outside_window_are_excluded():
    bars = _synthetic_bars()
    idx = bars.index
    events = pd.DataFrame({"surprise_pct": [10.0, 10.0], "report_time": ["post-market", "post-market"]},
                          index=pd.DatetimeIndex([idx[20], idx[220]]))

    trades = build_event_trades("TEST", events, bars, surprise_threshold_pct=5.0, hold_days=20, slippage_pct=0.0,
                                window_start=idx[100], window_end=idx[299])
    assert len(trades) == 1
    assert trades[0].entry_time > idx[100]


def test_equity_curve_handles_overlapping_hold_periods_without_duplicate_index():
    """Regression test for the bug found during the real run: hold periods
    from 'different windows' that overlap in calendar time must not produce
    duplicate timestamps when combined into one equity curve."""
    t1 = SimTrade("A", "long", pd.Timestamp("2023-01-10", tz="UTC"), 100, 100, 10000,
                 exit_time=pd.Timestamp("2023-04-05", tz="UTC"), exit_price=105, pnl=500)
    t2 = SimTrade("B", "long", pd.Timestamp("2023-04-10", tz="UTC"), 50, 200, 10000,
                 exit_time=pd.Timestamp("2023-05-01", tz="UTC"), exit_price=52, pnl=400)

    equity = equity_curve_from_trades([t1, t2], 100000)
    assert equity.index.duplicated().sum() == 0
    metrics = compute_metrics(equity, [t1, t2], 100000)  # must not raise
    assert metrics["trade_count"] == 2
    assert metrics["final_equity"] == 100900.0


def _staggered_universe(n_tickers=5, n_bars=200):
    idx = pd.bdate_range("2023-01-01", periods=n_bars, tz="UTC")
    tickers = [chr(ord("A") + i) for i in range(n_tickers)]
    bars_by_ticker, earnings_by_ticker = {}, {}
    for i, t in enumerate(tickers):
        close = 100 + np.cumsum(np.random.default_rng(i).normal(0, 0.5, n_bars))
        bars_by_ticker[t] = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5, "close": close},
                                         index=idx)
        surprise = 5.0 + i * 2  # increasing surprise magnitude: A=5% ... last=highest
        earnings_by_ticker[t] = pd.DataFrame({"surprise_pct": [surprise], "report_time": ["post-market"]},
                                             index=[idx[50]])  # all report the SAME day
    return tickers, earnings_by_ticker, bars_by_ticker


def test_portfolio_priority_favors_larger_surprises_when_capacity_limited():
    tickers, earnings_by_ticker, bars_by_ticker = _staggered_universe()
    trades = build_portfolio_event_trades(tickers, earnings_by_ticker, bars_by_ticker, surprise_threshold_pct=5.0,
                                          hold_days=20, slippage_pct=0.0, starting_capital=100000,
                                          max_gross_exposure=0.2, max_new_positions_per_day=10)
    assert sorted(t.instrument for t in trades) == ["D", "E"]  # highest two surprises win


def test_portfolio_daily_new_position_cap():
    tickers, earnings_by_ticker, bars_by_ticker = _staggered_universe()
    trades = build_portfolio_event_trades(tickers, earnings_by_ticker, bars_by_ticker, surprise_threshold_pct=5.0,
                                          hold_days=20, slippage_pct=0.0, starting_capital=100000,
                                          max_gross_exposure=10.0, max_new_positions_per_day=2)
    assert len(trades) == 2


def test_portfolio_gross_exposure_cap_holds_across_full_timeline():
    n = 300
    idx = pd.bdate_range("2023-01-01", periods=n, tz="UTC")
    tickers = [f"T{i}" for i in range(10)]
    bars_by_ticker, earnings_by_ticker = {}, {}
    for i, t in enumerate(tickers):
        close = 100 + np.cumsum(np.random.default_rng(i).normal(0, 0.5, n))
        bars_by_ticker[t] = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5, "close": close},
                                         index=idx)
        earnings_by_ticker[t] = pd.DataFrame({"surprise_pct": [8.0], "report_time": ["post-market"]},
                                             index=[idx[50 + i]])  # staggered across ~10 business days

    cap_multiple = 1.5
    trades = build_portfolio_event_trades(tickers, earnings_by_ticker, bars_by_ticker, surprise_threshold_pct=5.0,
                                          hold_days=60, slippage_pct=0.0, starting_capital=100000,
                                          max_gross_exposure=cap_multiple, max_new_positions_per_day=3)

    events = []
    for t in trades:
        events.append((t.entry_time, "open", t))
        events.append((t.exit_time, "close", t))
    events.sort(key=lambda e: (e[0], e[1] == "open"))

    open_positions, max_notional = {}, 0.0
    for ts, kind, t in events:
        if kind == "open":
            open_positions[id(t)] = t
            max_notional = max(max_notional, sum(p.qty * p.entry_price for p in open_positions.values()))
        else:
            open_positions.pop(id(t), None)

    assert max_notional <= cap_multiple * 100000 * 1.01


def test_regime_filter_blocks_entries_during_blocked_regime():
    n = 200
    idx = pd.bdate_range("2023-01-01", periods=n, tz="UTC")
    tickers = ["A", "B"]
    bars_by_ticker, earnings_by_ticker = {}, {}
    for i, t in enumerate(tickers):
        close = 100 + np.cumsum(np.random.default_rng(i).normal(0, 0.5, n))
        bars_by_ticker[t] = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5, "close": close},
                                         index=idx)
        earnings_by_ticker[t] = pd.DataFrame({"surprise_pct": [8.0], "report_time": ["post-market"]},
                                             index=[idx[50 + i * 30]])

    regime_labels = pd.Series("BULLISH", index=idx.normalize())
    regime_labels.loc[idx[50:55].normalize()] = "CORRECTION"  # covers A's entry day only

    unfiltered = build_portfolio_event_trades(tickers, earnings_by_ticker, bars_by_ticker, 5.0, 20, 0.0, 100000)
    filtered = build_portfolio_event_trades(tickers, earnings_by_ticker, bars_by_ticker, 5.0, 20, 0.0, 100000,
                                            regime_labels=regime_labels)

    assert len(unfiltered) == 2
    assert len(filtered) == 1
    assert filtered[0].instrument == "B"


def test_slippage_always_works_against_the_trader():
    """A silent slippage-direction bug (e.g. buy filled BELOW raw price)
    would make the backtest look better than reality without any obvious
    symptom -- worth a dedicated, explicit check rather than relying on
    it being incidentally correct in other tests."""
    bars = _synthetic_bars()
    idx = bars.index
    events = pd.DataFrame({"surprise_pct": [10.0], "report_time": ["post-market"]}, index=[idx[20]])

    no_slip = build_event_trades("TEST", events, bars, surprise_threshold_pct=5.0, hold_days=20, slippage_pct=0.0)
    with_slip = build_event_trades("TEST", events, bars, surprise_threshold_pct=5.0, hold_days=20, slippage_pct=0.01)
    assert len(no_slip) == 1 and len(with_slip) == 1

    assert with_slip[0].entry_price > no_slip[0].entry_price  # buy fill is worse (higher) with slippage
    assert with_slip[0].exit_price < no_slip[0].exit_price    # sell fill is worse (lower) with slippage
    assert with_slip[0].pnl < no_slip[0].pnl                  # net effect always hurts, never helps


def test_nan_surprise_pct_is_excluded_like_a_below_threshold_event():
    """NaN surprise_pct (old pre-analyst-coverage quarters, or near-zero
    estimate quarters nulled by earnings_data.py) must not accidentally
    pass a >=threshold check -- NaN comparisons are always False in Python,
    but worth a direct regression test since this is a silent-failure risk."""
    bars = _synthetic_bars()
    idx = bars.index
    events = pd.DataFrame({"surprise_pct": [float("nan"), 10.0], "report_time": ["post-market", "post-market"]},
                          index=pd.DatetimeIndex([idx[20], idx[100]]))
    trades = build_event_trades("TEST", events, bars, surprise_threshold_pct=5.0, hold_days=20, slippage_pct=0.0)
    assert len(trades) == 1
    assert trades[0].entry_time > idx[100]


def test_event_near_end_of_data_with_insufficient_hold_period_is_excluded():
    """An event whose hold period would run past the end of available price
    data must be skipped, not crash or silently truncate the hold."""
    bars = _synthetic_bars(n=100)
    idx = bars.index
    events = pd.DataFrame({"surprise_pct": [10.0], "report_time": ["post-market"]}, index=[idx[95]])
    trades = build_event_trades("TEST", events, bars, surprise_threshold_pct=5.0, hold_days=20, slippage_pct=0.0)
    assert len(trades) == 0


def test_daily_returns_from_equity_basic_sanity():
    equity = pd.Series([100000.0, 101000.0, 99000.0], index=pd.bdate_range("2023-01-01", periods=3, tz="UTC"))
    returns = daily_returns_from_equity(equity)
    assert len(returns) == 2
    assert returns.iloc[0] == pytest.approx(0.01)
    assert returns.iloc[1] == pytest.approx(99000.0 / 101000.0 - 1)


def test_daily_returns_from_equity_handles_empty_series():
    assert daily_returns_from_equity(pd.Series(dtype=float)).empty


def test_real_price_marking_gives_realistic_sharpe_unlike_linear_interpolation():
    """Regression test for a severe bug found during the full-universe PEAD
    run: marking unrealized P&L via a linear ramp from entry to exit price
    (the old default) produces a near-zero-variance equity curve during
    every open position, inflating Sharpe by 1-2 orders of magnitude and
    hiding intra-trade drawdown entirely, versus marking to the real daily
    close price (now the default when bars_by_ticker is supplied)."""
    from bot.backtest_engine import compute_metrics

    rng = np.random.default_rng(0)
    n = 250
    idx = pd.bdate_range("2023-01-01", periods=n, tz="UTC")
    daily_ret = rng.normal(0.0008, 0.02, n)
    price = 100 * np.cumprod(1 + daily_ret)
    bars = pd.DataFrame({"close": price}, index=idx)

    entry_price, exit_price = price[10], price[200]
    qty = 10000 / entry_price
    pnl = (exit_price - entry_price) * qty
    trade = SimTrade("X", "long", idx[10], entry_price, qty, 10000, exit_time=idx[200], exit_price=exit_price, pnl=pnl)

    eq_real = equity_curve_from_trades([trade], 100000, bars_by_ticker={"X": bars})
    eq_fallback = equity_curve_from_trades([trade], 100000)  # no bars_by_ticker -- linear interpolation fallback

    m_real = compute_metrics(eq_real, [trade], 100000)
    m_fallback = compute_metrics(eq_fallback, [trade], 100000)

    assert eq_real.iloc[-1] == pytest.approx(eq_fallback.iloc[-1])  # same final P&L either way
    assert m_fallback["sharpe"] > 10 * m_real["sharpe"]  # interpolation is wildly optimistic
    assert m_fallback["max_drawdown"] == 0.0  # a straight ramp to a positive endpoint never dips
    assert m_real["max_drawdown"] > 0.0       # the real price path does show an intra-trade drawdown
    assert 0.1 < m_real["sharpe"] < 2.0        # realistic order of magnitude for a single noisy trade
