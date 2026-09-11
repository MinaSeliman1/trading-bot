"""
The backtest engine reimplements Strategy A's signal math as vectorized
arrays for performance (see bot/backtest_engine.py's module docstring).
These tests prove that reimplementation produces IDENTICAL bar-by-bar
signals to the actual live Strategy A classes in bot/strategies/, so the
backtest is truly testing Strategy A as specified/deployed, not a drifted
copy of it.
"""
import numpy as np
import pandas as pd

from bot.strategies.mean_reversion import MeanReversionStrategy
from bot.strategies.momentum_breakout import MomentumBreakoutStrategy
from bot.strategies.trend_following import TrendFollowingStrategy
from bot.backtest_engine import (
    precompute_mean_reversion, precompute_momentum_breakout, precompute_trend_following,
    _mean_reversion_signal, _momentum_signal, _trend_signal,
)


def test_mean_reversion_signal_matches_live_strategy():
    rng = np.random.default_rng(4)
    n = 300
    close = 100 + np.sin(np.linspace(0, 30, n)) * 3 + rng.normal(0, 0.2, n)
    bars = pd.DataFrame({"high": close + 0.3, "low": close - 0.3, "close": close,
                          "volume": rng.uniform(1000, 2000, n)},
                         index=pd.date_range("2024-01-01", periods=n, freq="15min"))

    params = {"sma_period": 20, "std_dev_mult": 1.5}
    live = MeanReversionStrategy(**params)
    pre = precompute_mean_reversion(bars, params["sma_period"], params["std_dev_mult"])
    close_vals = bars["close"].values

    direction = None
    for i in range(params["sma_period"] + 1, len(bars)):
        sub = bars.iloc[: i + 1]
        live_sig = live.evaluate(sub, direction)
        vec_sig = _mean_reversion_signal(pre, i, close_vals[i], direction, None, False)
        assert live_sig == vec_sig, f"divergence at bar {i}: live={live_sig} vec={vec_sig}"
        if vec_sig == "exit":
            direction = None
        elif vec_sig in ("long", "short"):
            direction = vec_sig


def test_momentum_breakout_signal_matches_live_strategy():
    rng = np.random.default_rng(6)
    n = 200
    close = 100 + rng.normal(0, 0.5, n).cumsum() * 0.3
    volume = rng.uniform(1000, 2000, n)
    # inject a couple of clean breakouts with volume confirmation
    close[100] = close[:100].max() + 10
    volume[100] = 5000
    close[150] = close[:150].min() - 10
    volume[150] = 5000
    bars = pd.DataFrame({"high": close + 0.3, "low": close - 0.3, "close": close, "volume": volume},
                         index=pd.date_range("2024-01-01", periods=n, freq="1h"))

    params = {"lookback": 20, "volume_mult": 1.5}
    live = MomentumBreakoutStrategy(**params)
    pre = precompute_momentum_breakout(bars, params["lookback"], params["volume_mult"])

    direction = None
    for i in range(params["lookback"] + 1, len(bars)):
        sub = bars.iloc[: i + 1]
        live_sig = live.evaluate(sub, direction)
        vec_sig = _momentum_signal(pre, i, direction, allow_short=True)
        assert live_sig == vec_sig, f"divergence at bar {i}: live={live_sig} vec={vec_sig}"
        if vec_sig == "exit":
            direction = None
        elif vec_sig in ("long", "short"):
            direction = vec_sig


def test_trend_following_signal_matches_live_strategy():
    rng = np.random.default_rng(8)
    flat = np.full(60, 100.0)
    ramp = np.linspace(100, 140, 60)
    down = np.linspace(140, 110, 40)
    close = np.concatenate([flat, ramp, down]) + rng.normal(0, 0.1, 160)
    bars = pd.DataFrame({"close": close}, index=pd.date_range("2024-01-01", periods=len(close), freq="4h"))

    params = {"fast_ema": 10, "slow_ema": 30}
    live = TrendFollowingStrategy(**params)
    pre = precompute_trend_following(bars, params["fast_ema"], params["slow_ema"])

    direction = None
    for i in range(params["slow_ema"] + 1, len(bars)):
        sub = bars.iloc[: i + 1]
        live_sig = live.evaluate(sub, direction)
        vec_sig = _trend_signal(pre, i, direction, None, False)
        assert live_sig == vec_sig, f"divergence at bar {i}: live={live_sig} vec={vec_sig}"
        if vec_sig == "exit":
            direction = None
        elif vec_sig in ("long", "short"):
            direction = vec_sig
