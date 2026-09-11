import numpy as np
import pandas as pd

from bot.backtest_engine import (
    SimTrade, _open_position, _close_trade, compute_metrics,
    generate_walk_forward_windows, simulate_instrument, simulate_portfolio, InstrumentSpec,
    _mean_reversion_trendfilter_signal, simulate_momentum_breakout_mtf, is_oos_deviation_ok,
    simulate_mean_reversion_macrogate, cross_sectional_momentum_backtest, build_price_matrix,
)


def test_hard_stop_caps_loss_at_exactly_one_percent_when_filled_at_stop():
    equity = 100000.0
    atr_val = 2.0
    qty = (equity * 0.01) / atr_val  # the exact sizing formula simulate_instrument uses
    pos = _open_position("SPY", "long", qty=qty, raw_price=100.0, ts=pd.Timestamp("2026-01-01"),
                          equity=equity, atr_val=atr_val, trail_mult=None, regime=None, slippage_pct=0.0)
    assert pos.hard_stop == 98.0
    trade = _close_trade(pos, pd.Timestamp("2026-01-01 00:15"), raw_price=pos.hard_stop, reason="stop", slippage_pct=0.0)
    assert abs(trade.pnl / trade.equity_at_entry * 100 - (-1.0)) < 1e-9


def test_walk_forward_windows_roll_forward_by_test_months():
    windows = generate_walk_forward_windows(pd.Timestamp("2020-01-01"), pd.Timestamp("2024-01-01"),
                                             train_months=12, test_months=3)
    assert len(windows) >= 4
    assert windows[1]["train_start"] == windows[0]["train_start"] + pd.DateOffset(months=3)
    assert windows[0]["train_end"] == windows[0]["test_start"]


def test_compute_metrics_basic_sanity():
    idx = pd.date_range("2024-01-01", periods=110, freq="D")
    vals = list(range(100, 90, -1)) + list(np.linspace(90, 130, 100))
    eq = pd.Series(vals, index=idx, dtype=float) * 1000
    trades = [
        SimTrade("SPY", "long", idx[0], 100, 10, 100000, exit_time=idx[5], exit_price=95, pnl=-500, exit_reason="stop"),
        SimTrade("SPY", "long", idx[20], 95, 10, 100000, exit_time=idx[30], exit_price=110, pnl=1500, exit_reason="signal"),
    ]
    m = compute_metrics(eq, trades, 100000)
    assert m["trade_count"] == 2
    assert abs(m["win_rate"] - 0.5) < 1e-9
    assert m["max_drawdown"] > 0
    assert m["profit_factor"] > 1


def test_strategy_a_binary_correlation_filter_blocks_btc_long():
    def make_mr_bars(start, n_flat=30, n_hold=40):
        flat = np.full(n_flat, 100.0) + np.sin(np.linspace(0, 10, n_flat)) * 0.2
        hold = np.full(n_hold, 90.0)
        close = np.concatenate([flat, hold])
        idx = pd.date_range(start, periods=len(close), freq="15min")
        return pd.DataFrame({"open": close, "high": close + 0.3, "low": close - 0.3, "close": close,
                              "volume": np.random.default_rng(1).uniform(1000, 2000, len(close))}, index=idx)

    spy_bars = make_mr_bars("2024-01-01 09:30")
    qqq_bars = make_mr_bars("2024-01-01 09:30")

    n_btc = 30
    btc_close = np.full(n_btc, 100.0)
    btc_vol = np.full(n_btc, 1000.0)
    btc_close[-1] = 120.0
    btc_vol[-1] = 5000.0
    btc_idx = pd.date_range(end="2024-01-01 18:00", periods=n_btc, freq="1h")
    btc_bars = pd.DataFrame({"open": btc_close, "high": btc_close + 0.5, "low": btc_close - 0.5,
                              "close": btc_close, "volume": btc_vol}, index=btc_idx)

    instruments = {
        "SPY": InstrumentSpec("SPY", "us_equity", "mean_reversion", "15Min"),
        "QQQ": InstrumentSpec("QQQ", "us_equity", "mean_reversion", "15Min"),
        "BTCUSD": InstrumentSpec("BTC/USD", "crypto", "momentum_breakout", "1Hour"),
    }
    bars_by_key = {"SPY": spy_bars, "QQQ": qqq_bars, "BTCUSD": btc_bars}
    params_by_key = {
        "SPY": {"sma_period": 20, "std_dev_mult": 1.5},
        "QQQ": {"sma_period": 20, "std_dev_mult": 1.8},
        "BTCUSD": {"lookback": 20, "volume_mult": 1.5, "trailing_atr_mult": 2.0},
    }

    result_a = simulate_portfolio("A", instruments, bars_by_key, params_by_key, 100000, 0.0)
    btc_trades_a = [t for t in result_a["trades"] if t.instrument == "BTCUSD"]
    assert len(btc_trades_a) == 0

    result_b = simulate_portfolio("B", instruments, bars_by_key, params_by_key, 100000, 0.0)
    btc_trades_b = [t for t in result_b["trades"] if t.instrument == "BTCUSD"]
    assert len(btc_trades_b) == 1


def test_active_start_warms_up_slow_indicator_before_test_window():
    """A 200-period EMA cannot warm up from a cold start inside a short test
    window (e.g. ~100 bars). active_start must let the indicator warm up on
    bars before it while still gating entries to only fire from active_start on."""
    cfg = InstrumentSpec("GLD", "us_equity", "trend_following", "4Hour")
    params = {"fast_ema": 50, "slow_ema": 200, "trailing_atr_mult": 3.0}

    flat = np.full(250, 100.0)         # long enough to fully warm up EMA-200 before the ramp starts
    ramp = np.linspace(100, 140, 100)  # crossover happens somewhere in here, at/after active_start
    close = np.concatenate([flat, ramp])
    bars = pd.DataFrame({"open": close, "high": close + 0.3, "low": close - 0.3, "close": close,
                          "volume": 1000}, index=pd.date_range("2024-01-01", periods=len(close), freq="4h"))

    active_start = bars.index[250]  # start of the ramp -- only 100 bars remain, nowhere near 200-bar warmup cold
    short_window = bars[bars.index >= active_start]

    # cold-started on just the short window: EMA-200 never finishes warming up -> no trades possible
    cold_trades, _ = simulate_instrument("GLD", cfg, short_window, "trend_following", params, 100000, 0.0)
    assert len(cold_trades) == 0

    # same short window, but with warmup bars supplied and active_start gating entries
    warm_trades, warm_equity = simulate_instrument("GLD", cfg, bars, "trend_following", params, 100000, 0.0,
                                                     active_start=active_start)
    assert len(warm_trades) >= 1
    assert all(t.entry_time >= active_start for t in warm_trades)
    assert not warm_equity.empty
    assert warm_equity.index.min() >= active_start


def test_stop_cooldown_suppresses_immediate_reentry():
    """Without a cooldown, mean reversion re-enters the instant it's stopped
    out if price is still beyond the band -- the exact whipsaw failure mode
    diagnosed in a persistent decline. A cooldown must reduce trade count and
    enforce a minimum bar gap after every stop."""
    cfg = InstrumentSpec("SPY", "us_equity", "mean_reversion", "15Min")
    n = 60
    close = np.concatenate([np.full(40, 100.0), np.linspace(100, 90, 20)])
    bars = pd.DataFrame({"open": close, "high": close + 0.2, "low": close - 0.2, "close": close, "volume": 1500},
                         index=pd.date_range("2024-01-01", periods=n, freq="15min"))
    params = {"sma_period": 20, "std_dev_mult": 1.5}

    no_cooldown, _ = simulate_instrument("SPY", cfg, bars, "mean_reversion", params, 100000, 0.0)
    with_cooldown, _ = simulate_instrument("SPY", cfg, bars, "mean_reversion", params, 100000, 0.0,
                                            stop_cooldown_bars=5)

    assert len(with_cooldown) < len(no_cooldown)

    for stop_trade in [t for t in with_cooldown if t.exit_reason == "stop"]:
        later_entries = [t for t in with_cooldown if t.entry_time > stop_trade.exit_time]
        if later_entries:
            gap_bars = (min(t.entry_time for t in later_entries) - stop_trade.exit_time) / pd.Timedelta(minutes=15)
            assert gap_bars >= 5 - 1e-9


def test_mean_reversion_trendfilter_only_trades_with_the_trend():
    pre_up = {"sma": [100] * 10, "upper": [105] * 10, "lower": [95] * 10, "trend_up": [True] * 10}
    pre_down = {"sma": [100] * 10, "upper": [105] * 10, "lower": [95] * 10, "trend_up": [False] * 10}

    assert _mean_reversion_trendfilter_signal(pre_up, 5, 90, None, None, False) == "long"
    assert _mean_reversion_trendfilter_signal(pre_down, 5, 90, None, None, False) is None
    assert _mean_reversion_trendfilter_signal(pre_down, 5, 110, None, None, False) == "short"
    assert _mean_reversion_trendfilter_signal(pre_up, 5, 110, None, None, False) is None


def test_momentum_breakout_mtf_requires_4h_trend_confirmation():
    cfg = InstrumentSpec("BTC/USD", "crypto", "momentum_breakout", "1Hour")
    n1h = 200
    close1h = np.full(n1h, 100.0)
    vol1h = np.full(n1h, 1000.0)
    close1h[-1], vol1h[-1] = 120.0, 5000.0
    bars_1h = pd.DataFrame({"high": close1h + 0.5, "low": close1h - 0.5, "close": close1h, "volume": vol1h},
                            index=pd.date_range(end="2024-03-01", periods=n1h, freq="1h"))
    params = {"lookback": 20, "volume_mult": 1.5, "trailing_atr_mult": 2.0}

    bars_4h_up = pd.DataFrame({"close": np.linspace(90, 110, 60)},
                               index=pd.date_range(end="2024-03-01", periods=60, freq="4h"))
    trades_confirmed, _ = simulate_momentum_breakout_mtf("BTCUSD", cfg, bars_1h, bars_4h_up, params, 100000, 0.0)
    assert len(trades_confirmed) >= 1

    bars_4h_down = pd.DataFrame({"close": np.linspace(110, 90, 60)},
                                 index=pd.date_range(end="2024-03-01", periods=60, freq="4h"))
    trades_blocked, _ = simulate_momentum_breakout_mtf("BTCUSD", cfg, bars_1h, bars_4h_down, params, 100000, 0.0)
    assert len(trades_blocked) == 0


def test_is_oos_deviation_rule():
    assert is_oos_deviation_ok(1.0, 1.2) is True     # 20% deviation, within 30% tolerance
    assert is_oos_deviation_ok(1.0, 0.5) is False    # 50% deviation, rejected
    assert is_oos_deviation_ok(2.0, -0.5) is False   # sign flip, way over tolerance
    assert is_oos_deviation_ok(float("nan"), 1.0) is False


def test_strategy_a_gross_exposure_never_exceeds_cap():
    """Five instruments each independently breaching their entry threshold on
    the same day used to be able to stack to 500-600% gross notional (found
    via the sanity-check trade dump) since each position was only capped
    individually at 100% of equity. Must now be capped in aggregate."""
    from bot.backtest_engine import STRATEGY_A_MAX_GROSS_EXPOSURE

    def make_dip_bars(seed, n_flat=30, n_hold=20):
        rng = np.random.default_rng(seed)
        flat = np.full(n_flat, 100.0) + np.sin(np.linspace(0, 10, n_flat)) * 0.2
        hold = np.full(n_hold, 90.0)
        close = np.concatenate([flat, hold])
        idx = pd.date_range("2024-01-01 09:30", periods=len(close), freq="15min")
        return pd.DataFrame({"open": close, "high": close + 0.3, "low": close - 0.3, "close": close,
                              "volume": rng.uniform(1000, 2000, len(close))}, index=idx)

    instruments = {
        "SPY": InstrumentSpec("SPY", "us_equity", "mean_reversion", "15Min"),
        "QQQ": InstrumentSpec("QQQ", "us_equity", "mean_reversion", "15Min"),
        "GLD": InstrumentSpec("GLD", "us_equity", "mean_reversion", "15Min"),
        "USO": InstrumentSpec("USO", "us_equity", "mean_reversion", "15Min"),
    }
    bars_by_key = {k: make_dip_bars(i) for i, k in enumerate(instruments)}
    params_by_key = {k: {"sma_period": 20, "std_dev_mult": 1.5} for k in instruments}

    result = simulate_portfolio("A", instruments, bars_by_key, params_by_key, 100000, 0.0)

    # reconstruct the actual open-position timeline from entry/exit intervals
    events = []
    for t in result["trades"]:
        events.append((t.entry_time, "open", t))
        events.append((t.exit_time, "close", t))
    events.sort(key=lambda e: (e[0], e[1] == "open"))  # process closes before opens at the same timestamp

    open_positions = {}
    for _, kind, t in events:
        if kind == "open":
            open_positions[t.instrument] = t
            gross = sum(abs(p.qty * p.entry_price) for p in open_positions.values())
            assert gross <= STRATEGY_A_MAX_GROSS_EXPOSURE * 100000 * 1.01  # small tolerance for equity growth mid-run
        else:
            open_positions.pop(t.instrument, None)


def test_macro_gate_blocks_new_longs_when_risk_off():
    class Cfg:
        pass

    cfg = Cfg()
    up = np.linspace(100, 130, 250)          # warms up above its own rising 200-SMA
    dip = np.linspace(130, 110, 20)          # sustained multi-bar decline -> breaches the 20-period band
    close = np.concatenate([up, dip])
    bars = pd.DataFrame({"high": close + 0.3, "low": close - 0.3, "close": close, "volume": 1500},
                         index=pd.date_range("2024-01-01", periods=len(close), freq="15min"))
    params = {"sma_period": 20, "std_dev_mult": 1.5, "trend_sma_period": 200}

    gate_off = pd.Series(False, index=bars.index)
    gate_on = pd.Series(True, index=bars.index)

    trades_off, _ = simulate_mean_reversion_macrogate("SPY", cfg, bars, gate_off, params, 100000, 0.0)
    trades_on, _ = simulate_mean_reversion_macrogate("SPY", cfg, bars, gate_on, params, 100000, 0.0)

    assert len(trades_off) == 0
    assert len(trades_on) >= 1
    assert all(t.direction == "long" for t in trades_on)


def test_cross_sectional_momentum_picks_the_outperformer():
    n = 400
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    rng = np.random.default_rng(3)
    winner = 100 * (1.002 ** np.arange(n)) * (1 + rng.normal(0, 0.002, n))
    losers = {f"L{i}": 100 * (0.9995 ** np.arange(n)) * (1 + rng.normal(0, 0.002, n)) for i in range(4)}
    price_matrix = pd.DataFrame({"WINNER": winner, **losers}, index=idx)

    trades, equity = cross_sectional_momentum_backtest(price_matrix, lookback_days=63, top_frac=0.2,
                                                         starting_capital=100000, slippage_pct=0.0, min_names=1)
    assert not equity.empty
    winner_trades = [t for t in trades if t.instrument == "XSMOM:WINNER"]
    assert len(winner_trades) >= 1
    # the clear winner should never be sold at a loss over most rebalances given its strong steady uptrend
    assert sum(1 for t in winner_trades if t.pnl > 0) >= len(winner_trades) // 2
