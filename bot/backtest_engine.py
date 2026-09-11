"""
Shared backtest/walk-forward/Monte Carlo engine used to evaluate BOTH
Strategy A (baseline) and Strategy B (regime + covariance + vol targeting)
through the identical pipeline: same data, same walk-forward split dates,
same cost scenarios, same starting capital.

Design notes (documented here since they're load-bearing assumptions for the
whole comparison -- see comparison_report.md's methodology section for the
user-facing version):

- Per-instrument indicators (SMA/std bands, breakout levels, EMA cross, ATR,
  ADX/regime) are precomputed once as vectorized arrays. The sequential event
  loop then does O(1) lookups per bar instead of recomputing rolling windows
  at every step -- this is what makes walk-forward grid search tractable.
- Stops are checked against bar CLOSE (matching how the live bot polls),
  not intrabar high/low. This is consistent with the rest of this codebase
  but is more conservative than an intrabar-fill assumption would be.
- Walk-forward parameter search optimizes each instrument's threshold
  parameter independently (single-instrument backtest, common ATR-1%-risk
  sizing proxy for both strategies) rather than jointly across the whole
  5-instrument portfolio -- jointly optimizing a 5-instrument parameter grid
  is combinatorially intractable in-session. Final performance figures use
  each strategy's REAL sizing logic in a full shared-equity portfolio run.
- Parameters are optimized once at a fixed mid-point slippage assumption
  (0.10%); the three cost scenarios are then evaluated as a sensitivity pass
  over those fixed parameters, not re-optimized per cost level.
- BTC/USD momentum breakout NEVER shorts in either strategy's backtest --
  Alpaca does not support shorting crypto, so an A that could short BTC would
  be modeling an edge that could never be realized live. Both strategies are
  held to the same real constraint so the comparison isn't confounded by it.
- Strategy B's volatility-targeting scale-up cap is modeled at its
  post-validation steady-state (1.5x cap) throughout the backtest, not the
  conservative 1.0x-only cap mandated for a live system's first 4-6 weeks --
  a historical backtest has no "early rollout window" concept.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from bot.regime_detector import RegimeDetector, calculate_adx, classify_regime, RANGING, TRENDING

logger = logging.getLogger(__name__)

# Strategy A's original spec bounds each position at up to 100% of equity
# notional individually, with no cross-instrument cap -- with several
# instruments open simultaneously that allows unrealistic gross leverage.
# Capped at Alpaca's real standard margin multiplier (confirmed live via
# TradingClient account.multiplier == 4) so the backtest can't assume more
# buying power than an actual account would have.
STRATEGY_A_MAX_GROSS_EXPOSURE = 4.0


@dataclass
class InstrumentSpec:
    symbol: str
    asset_class: str          # "us_equity" or "crypto"
    strategy_kind: str        # "mean_reversion" | "momentum_breakout" | "trend_following"
    timeframe: str             # "15Min" | "1Hour" | "4Hour"
    fractionable: bool = True


# ---------------------------------------------------------------------- #
# indicator precomputation
# ---------------------------------------------------------------------- #
def atr_series(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = bars["high"], bars["low"], bars["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def precompute_mean_reversion(bars: pd.DataFrame, sma_period: int, std_dev_mult: float) -> dict:
    close = bars["close"]
    sma = close.rolling(sma_period).mean()
    std = close.rolling(sma_period).std()
    return {"sma": sma.values, "upper": (sma + std_dev_mult * std).values, "lower": (sma - std_dev_mult * std).values}


def precompute_momentum_breakout(bars: pd.DataFrame, lookback: int, volume_mult: float) -> dict:
    prior_high = bars["high"].shift(1).rolling(lookback).max()
    prior_low = bars["low"].shift(1).rolling(lookback).min()
    prior_avg_vol = bars["volume"].shift(1).rolling(lookback).mean()
    breakout_up = (bars["close"] > prior_high) & (bars["volume"] >= volume_mult * prior_avg_vol)
    breakout_down = (bars["close"] < prior_low) & (bars["volume"] >= volume_mult * prior_avg_vol)
    return {"breakout_up": breakout_up.fillna(False).values, "breakout_down": breakout_down.fillna(False).values}


def precompute_trend_following(bars: pd.DataFrame, fast_ema: int, slow_ema: int) -> dict:
    close = bars["close"]
    fast = close.ewm(span=fast_ema, adjust=False).mean()
    slow = close.ewm(span=slow_ema, adjust=False).mean()
    crossed_up = (fast.shift(1) <= slow.shift(1)) & (fast > slow)
    crossed_down = (fast.shift(1) >= slow.shift(1)) & (fast < slow)
    valid = ~close.rolling(slow_ema).mean().isna()  # warm-up guard
    return {
        "crossed_up": (crossed_up & valid).fillna(False).values,
        "crossed_down": (crossed_down & valid).fillna(False).values,
    }


def precompute_regime(bars: pd.DataFrame, adx_period: int = 14) -> np.ndarray:
    adx = calculate_adx(bars, adx_period)
    return np.array([classify_regime(v) for v in adx.values], dtype=object)


# ---------------------------------------------------------------------- #
# strategy logic variants (exploration, not part of the core A/B comparison)
# ---------------------------------------------------------------------- #
def precompute_mean_reversion_trendfilter(bars: pd.DataFrame, sma_period: int, std_dev_mult: float,
                                            trend_sma_period: int = 200) -> dict:
    """Mean reversion variant: only fade in the direction of the long-term
    trend -- buy dips when price is above the 200-SMA, sell rallies when
    price is below it. Never fade AGAINST the underlying trend."""
    pre = precompute_mean_reversion(bars, sma_period, std_dev_mult)
    trend_sma = bars["close"].rolling(trend_sma_period).mean()
    pre["trend_up"] = (bars["close"] > trend_sma).fillna(False).values
    return pre


def _mean_reversion_trendfilter_signal(pre, i, price, current_direction, regime, use_regime) -> Optional[str]:
    sma, upper, lower = pre["sma"][i], pre["upper"][i], pre["lower"][i]
    if sma != sma:
        return None
    trend_up = pre["trend_up"][i]
    if use_regime and current_direction is not None and regime == TRENDING:
        return "exit"
    if current_direction is None:
        if use_regime and regime != RANGING:
            return None
        if price <= lower and trend_up:
            return "long"
        if price >= upper and not trend_up:
            return "short"
        return None
    if current_direction == "long" and price >= sma:
        return "exit"
    if current_direction == "short" and price <= sma:
        return "exit"
    return None


def precompute_trend_following_volfilter(bars: pd.DataFrame, fast_ema: int, slow_ema: int,
                                           atr_period: int = 14, vol_percentile_min: float = 0.25) -> dict:
    """Trend following variant: skip new entries when ATR is in the bottom
    `vol_percentile_min` of its own history so far -- a "dead market" filter,
    the idea being a trend-following crossover in an abnormally quiet market
    is more likely noise than the start of a real move."""
    pre = precompute_trend_following(bars, fast_ema, slow_ema)
    atr_vals = atr_series(bars, atr_period)
    pct_rank = atr_vals.expanding().rank(pct=True)
    pre["vol_ok"] = (pct_rank >= vol_percentile_min).fillna(False).values
    return pre


def _trend_volfilter_signal(pre, i, current_direction, regime, use_regime) -> Optional[str]:
    up, down = pre["crossed_up"][i], pre["crossed_down"][i]
    if current_direction is None:
        if use_regime and regime != TRENDING:
            return None
        if not pre["vol_ok"][i]:
            return None
        if up:
            return "long"
        if down:
            return "short"
        return None
    if current_direction == "long" and down:
        return "exit"
    if current_direction == "short" and up:
        return "exit"
    return None


def simulate_mean_reversion_macrogate(key: str, cfg, bars: pd.DataFrame, macro_risk_on: pd.Series, params: dict,
                                        starting_capital: float, slippage_pct: float, active_start=None,
                                        stop_cooldown_bars: int = 0, base_risk_pct: float = 0.01) -> tuple:
    """Mean reversion + SMA200 trend filter (this instrument's own trend), PLUS
    a portfolio-level macro gate: no new LONG entries anywhere while SPY is
    below its own 200-day SMA (`macro_risk_on`, computed once from SPY daily
    bars and reindexed onto this instrument's bar index via ffill). Shorts are
    not gated by the macro filter -- only new longs are blocked, matching the
    "risk-on allows longs" framing. Needs its own loop since it consumes an
    external gate series not derivable from `bars` alone."""
    pre = precompute_mean_reversion_trendfilter(bars, params["sma_period"], params["std_dev_mult"],
                                                  params.get("trend_sma_period", 200))
    gate = macro_risk_on.reindex(bars.index, method="ffill").fillna(False).values

    atr = atr_series(bars, 14).values
    close = bars["close"].values
    idx = bars.index

    trades, position, equity, records = [], None, starting_capital, []
    trail_mult = params.get("trailing_atr_mult")
    min_bars = max(params["sma_period"], params.get("trend_sma_period", 200), 14) + 1
    cooldown_until = -1

    for i in range(min_bars, len(bars)):
        price, a = close[i], atr[i]

        if position is not None:
            position.update_trailing(price, a)
            if position.stop_hit(price):
                trades.append(_close_trade(position, idx[i], price, "stop", slippage_pct))
                equity += trades[-1].pnl
                position = None
                cooldown_until = i + stop_cooldown_bars

        current_direction = position.direction if position else None
        signal = _mean_reversion_trendfilter_signal(pre, i, price, current_direction, None, False)
        if signal == "long" and current_direction is None and not gate[i]:
            signal = None  # macro gate blocks new longs when SPY is below its 200-day SMA

        if signal == "exit" and position is not None:
            trades.append(_close_trade(position, idx[i], price, "signal", slippage_pct))
            equity += trades[-1].pnl
            position = None
        elif signal in ("long", "short") and position is not None and position.direction != signal:
            trades.append(_close_trade(position, idx[i], price, "flip", slippage_pct))
            equity += trades[-1].pnl
            position = None

        is_active = active_start is None or idx[i] >= active_start
        in_cooldown = i < cooldown_until
        if position is None and signal in ("long", "short") and a == a and a > 0 and is_active and not in_cooldown:
            qty = (equity * base_risk_pct) / a
            if qty > 0:
                position = _open_position(key, signal, qty, price, idx[i], equity, a, trail_mult, None, slippage_pct)

        if is_active:
            records.append((idx[i], equity + _unrealized(position, price)))

    if position is not None:
        trades.append(_close_trade(position, idx[-1], close[-1], "end_of_data", slippage_pct))

    equity_series = pd.Series({t: v for t, v in records}).sort_index() if records else pd.Series(dtype=float)
    return trades, equity_series


# ---------------------------------------------------------------------- #
# simulation primitives
# ---------------------------------------------------------------------- #
def apply_slippage(price: float, side: str, slippage_pct: float) -> float:
    return price * (1 + slippage_pct) if side == "buy" else price * (1 - slippage_pct)


ENTRY_SIDE = {"long": "buy", "short": "sell"}
EXIT_SIDE = {"long": "sell", "short": "buy"}


@dataclass
class SimTrade:
    instrument: str
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    equity_at_entry: float
    regime_at_entry: Optional[str] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None

    @property
    def return_pct(self) -> float:
        if self.pnl is None or self.equity_at_entry <= 0:
            return 0.0
        return self.pnl / self.equity_at_entry


@dataclass
class SimPosition:
    instrument: str
    direction: str
    qty: float
    entry_price: float
    entry_time: pd.Timestamp
    equity_at_entry: float
    atr_at_entry: float
    hard_stop: float
    trail_mult: Optional[float]
    extreme_price: float
    regime_at_entry: Optional[str] = None
    trail_stop: Optional[float] = None

    def current_stop(self) -> float:
        if self.trail_stop is None:
            return self.hard_stop
        return max(self.hard_stop, self.trail_stop) if self.direction == "long" else min(self.hard_stop, self.trail_stop)

    def update_trailing(self, price: float, atr: float) -> None:
        if self.trail_mult is None or atr != atr or atr <= 0:
            return
        if self.direction == "long":
            self.extreme_price = max(self.extreme_price, price)
            candidate = self.extreme_price - self.trail_mult * atr
        else:
            self.extreme_price = min(self.extreme_price, price)
            candidate = self.extreme_price + self.trail_mult * atr
        if self.trail_stop is None:
            self.trail_stop = candidate
        elif self.direction == "long":
            self.trail_stop = max(self.trail_stop, candidate)
        else:
            self.trail_stop = min(self.trail_stop, candidate)

    def stop_hit(self, price: float) -> bool:
        stop = self.current_stop()
        return price <= stop if self.direction == "long" else price >= stop


class EWMAVolTracker:
    """O(1)-update EWMA volatility of a daily return stream."""

    def __init__(self, span: int = 20):
        self.alpha = 2 / (span + 1)
        self._var = None

    def update(self, daily_return: float) -> None:
        r2 = daily_return ** 2
        self._var = r2 if self._var is None else (1 - self.alpha) * self._var + self.alpha * r2

    @property
    def annualized_vol(self) -> float:
        if self._var is None:
            return float("nan")
        return float(np.sqrt(self._var) * np.sqrt(252))


# ---------------------------------------------------------------------- #
# single-instrument backtest (grid-search proxy: common ATR-1%-risk sizing)
# ---------------------------------------------------------------------- #
def simulate_instrument(key: str, cfg, bars: pd.DataFrame, strategy_kind: str, params: dict,
                         starting_capital: float, slippage_pct: float, use_regime: bool = False,
                         base_risk_pct: float = 0.01, active_start=None, stop_cooldown_bars: int = 0) -> tuple:
    """Single-instrument, isolated-capital backtest used only to rank
    parameter choices during walk-forward optimization. Returns (trades, equity_series).

    `bars` may extend earlier than the period you actually want results for --
    pass `active_start` (a timestamp) to compute indicators (SMA/EMA/ATR/ADX)
    over the full history in `bars` while only allowing NEW entries and
    returning equity records from `active_start` onward. This matters most
    for slow-warming indicators (e.g. a 200-period EMA on 4-hour bars) which
    would otherwise never finish warming up inside a short out-of-sample
    test window if computed from a cold start at the window boundary.

    `stop_cooldown_bars`: after a stop-loss exit, block new entries on this
    instrument for this many bars. Strategy A has no regime filter, so
    without a cooldown it will re-enter immediately after a stop-out if
    price is still beyond the signal threshold -- exactly the repeated
    buy-the-still-falling-knife pattern that devastates it in a trend. Zero
    by default (matches the original spec's "no cooldown" baseline); pass a
    positive value to test the fairness-adjusted variant."""
    if strategy_kind == "mean_reversion":
        pre = precompute_mean_reversion(bars, params["sma_period"], params["std_dev_mult"])
    elif strategy_kind == "momentum_breakout":
        pre = precompute_momentum_breakout(bars, params["lookback"], params["volume_mult"])
    elif strategy_kind == "trend_following":
        pre = precompute_trend_following(bars, params["fast_ema"], params["slow_ema"])
    elif strategy_kind == "mean_reversion_trendfilter":
        pre = precompute_mean_reversion_trendfilter(bars, params["sma_period"], params["std_dev_mult"],
                                                      params.get("trend_sma_period", 200))
    elif strategy_kind == "trend_following_volfilter":
        pre = precompute_trend_following_volfilter(bars, params["fast_ema"], params["slow_ema"],
                                                     vol_percentile_min=params.get("vol_percentile_min", 0.25))
    else:
        raise ValueError(strategy_kind)

    atr = atr_series(bars, 14).values
    regime = precompute_regime(bars, 14) if use_regime else None
    close = bars["close"].values
    idx = bars.index

    trades = []
    position = None
    equity = starting_capital
    records = []
    trail_mult = params.get("trailing_atr_mult")
    min_bars = max(params.get("sma_period", 0), params.get("slow_ema", 0), params.get("lookback", 0),
                    params.get("trend_sma_period", 0), 14) + 1
    cooldown_until = -1

    for i in range(min_bars, len(bars)):
        price = close[i]
        a = atr[i]
        cur_regime = regime[i] if regime is not None else None

        if position is not None:
            position.update_trailing(price, a)
            if position.stop_hit(price):
                trades.append(_close_trade(position, idx[i], price, "stop", slippage_pct))
                equity += trades[-1].pnl
                position = None
                cooldown_until = i + stop_cooldown_bars

        current_direction = position.direction if position else None
        if strategy_kind == "mean_reversion":
            signal = _mean_reversion_signal(pre, i, price, current_direction, cur_regime, use_regime)
        elif strategy_kind == "momentum_breakout":
            signal = _momentum_signal(pre, i, current_direction, allow_short=not use_regime)
        elif strategy_kind == "trend_following":
            signal = _trend_signal(pre, i, current_direction, cur_regime, use_regime)
        elif strategy_kind == "mean_reversion_trendfilter":
            signal = _mean_reversion_trendfilter_signal(pre, i, price, current_direction, cur_regime, use_regime)
        else:
            signal = _trend_volfilter_signal(pre, i, current_direction, cur_regime, use_regime)

        if signal == "exit" and position is not None:
            trades.append(_close_trade(position, idx[i], price, "signal", slippage_pct))
            equity += trades[-1].pnl
            position = None
        elif signal in ("long", "short") and position is not None and position.direction != signal:
            trades.append(_close_trade(position, idx[i], price, "flip", slippage_pct))
            equity += trades[-1].pnl
            position = None

        is_active = active_start is None or idx[i] >= active_start
        in_cooldown = i < cooldown_until
        if position is None and signal in ("long", "short") and a == a and a > 0 and is_active and not in_cooldown:
            qty = (equity * base_risk_pct) / a
            if qty > 0:
                position = _open_position(key, signal, qty, price, idx[i], equity, a, trail_mult, cur_regime, slippage_pct)

        if is_active:
            records.append((idx[i], equity + _unrealized(position, price)))

    if position is not None:
        trades.append(_close_trade(position, idx[-1], close[-1], "end_of_data", slippage_pct))

    equity_series = pd.Series({t: v for t, v in records}).sort_index()
    return trades, equity_series


def _mean_reversion_signal(pre, i, price, current_direction, regime, use_regime) -> Optional[str]:
    sma, upper, lower = pre["sma"][i], pre["upper"][i], pre["lower"][i]
    if sma != sma:
        return None
    if use_regime and current_direction is not None and regime == TRENDING:
        return "exit"
    if current_direction is None:
        if use_regime and regime != RANGING:
            return None
        if price <= lower:
            return "long"
        if price >= upper:
            return "short"
        return None
    if current_direction == "long" and price >= sma:
        return "exit"
    if current_direction == "short" and price <= sma:
        return "exit"
    return None


def _momentum_signal(pre, i, current_direction, allow_short: bool) -> Optional[str]:
    up, down = pre["breakout_up"][i], pre["breakout_down"][i]
    if current_direction is None:
        if up:
            return "long"
        if down and allow_short:
            return "short"
        return None
    if current_direction == "long" and down:
        return "exit"
    if current_direction == "short" and up:
        return "exit"
    return None


def _trend_signal(pre, i, current_direction, regime, use_regime) -> Optional[str]:
    up, down = pre["crossed_up"][i], pre["crossed_down"][i]
    if current_direction is None:
        if use_regime and regime != TRENDING:
            return None
        if up:
            return "long"
        if down:
            return "short"
        return None
    if current_direction == "long" and down:
        return "exit"
    if current_direction == "short" and up:
        return "exit"
    return None


def _open_position(key, direction, qty, raw_price, ts, equity, atr_val, trail_mult, regime, slippage_pct) -> SimPosition:
    fill = apply_slippage(raw_price, ENTRY_SIDE[direction], slippage_pct)
    hard_stop = fill - atr_val if direction == "long" else fill + atr_val
    return SimPosition(instrument=key, direction=direction, qty=qty, entry_price=fill, entry_time=ts,
                        equity_at_entry=equity, atr_at_entry=atr_val, hard_stop=hard_stop, trail_mult=trail_mult,
                        extreme_price=fill, regime_at_entry=regime)


def _close_trade(position: SimPosition, ts, raw_price, reason, slippage_pct) -> SimTrade:
    fill = apply_slippage(raw_price, EXIT_SIDE[position.direction], slippage_pct)
    pnl = (fill - position.entry_price) * position.qty if position.direction == "long" else (position.entry_price - fill) * position.qty
    return SimTrade(instrument=position.instrument, direction=position.direction, entry_time=position.entry_time,
                     entry_price=position.entry_price, qty=position.qty, equity_at_entry=position.equity_at_entry,
                     regime_at_entry=position.regime_at_entry, exit_time=ts, exit_price=fill, pnl=pnl, exit_reason=reason)


def _unrealized(position: Optional[SimPosition], price: float) -> float:
    if position is None:
        return 0.0
    return (price - position.entry_price) * position.qty if position.direction == "long" else (position.entry_price - price) * position.qty


def simulate_momentum_breakout_mtf(key: str, cfg, bars_1h: pd.DataFrame, bars_4h: pd.DataFrame, params: dict,
                                     starting_capital: float, slippage_pct: float, active_start=None,
                                     stop_cooldown_bars: int = 0, base_risk_pct: float = 0.01) -> tuple:
    """Momentum breakout variant: the 1h breakout only fires when confirmed by
    the 4h trend (EMA(mtf_fast) > EMA(mtf_slow) on the 4h timeframe). Long-only,
    matching the rest of this codebase's no-short-crypto policy. Needs its own
    loop (not simulate_instrument's dispatch) since it consumes two bar series."""
    pre = precompute_momentum_breakout(bars_1h, params["lookback"], params["volume_mult"])
    fast_4h = bars_4h["close"].ewm(span=params.get("mtf_fast_ema", 20), adjust=False).mean()
    slow_4h = bars_4h["close"].ewm(span=params.get("mtf_slow_ema", 50), adjust=False).mean()
    trend_up_1h = (fast_4h > slow_4h).reindex(bars_1h.index, method="ffill").fillna(False).values

    atr = atr_series(bars_1h, 14).values
    close = bars_1h["close"].values
    idx = bars_1h.index

    trades, position, equity, records = [], None, starting_capital, []
    trail_mult = params.get("trailing_atr_mult")
    min_bars = params["lookback"] + 1
    cooldown_until = -1

    for i in range(min_bars, len(bars_1h)):
        price, a = close[i], atr[i]

        if position is not None:
            position.update_trailing(price, a)
            if position.stop_hit(price):
                trades.append(_close_trade(position, idx[i], price, "stop", slippage_pct))
                equity += trades[-1].pnl
                position = None
                cooldown_until = i + stop_cooldown_bars

        current_direction = position.direction if position else None
        up, down = pre["breakout_up"][i], pre["breakout_down"][i]
        signal = None
        if current_direction is None:
            if up and trend_up_1h[i]:
                signal = "long"
        elif current_direction == "long" and down:
            signal = "exit"

        if signal == "exit" and position is not None:
            trades.append(_close_trade(position, idx[i], price, "signal", slippage_pct))
            equity += trades[-1].pnl
            position = None

        is_active = active_start is None or idx[i] >= active_start
        in_cooldown = i < cooldown_until
        if position is None and signal == "long" and a == a and a > 0 and is_active and not in_cooldown:
            qty = (equity * base_risk_pct) / a
            if qty > 0:
                position = _open_position(key, "long", qty, price, idx[i], equity, a, trail_mult, None, slippage_pct)

        if is_active:
            records.append((idx[i], equity + _unrealized(position, price)))

    if position is not None:
        trades.append(_close_trade(position, idx[-1], close[-1], "end_of_data", slippage_pct))

    equity_series = pd.Series({t: v for t, v in records}).sort_index() if records else pd.Series(dtype=float)
    return trades, equity_series


# ---------------------------------------------------------------------- #
# metrics
# ---------------------------------------------------------------------- #
def daily_returns_from_equity(equity_series: pd.Series) -> pd.Series:
    if equity_series.empty:
        return equity_series
    daily = equity_series.resample("1D").last().ffill().dropna()
    return daily.pct_change().dropna()


def compute_metrics(equity_series: pd.Series, trades: list, starting_capital: float) -> dict:
    daily_ret = daily_returns_from_equity(equity_series)

    if len(daily_ret) >= 2 and daily_ret.std() > 0:
        sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
        downside = daily_ret[daily_ret < 0]
        sortino = float(daily_ret.mean() / downside.std() * np.sqrt(252)) if len(downside) >= 2 and downside.std() > 0 else float("nan")
    else:
        sharpe, sortino = float("nan"), float("nan")

    if not equity_series.empty:
        running_peak = equity_series.cummax()
        drawdown = (running_peak - equity_series) / running_peak.replace(0, np.nan)
        max_dd = float(drawdown.max()) if not drawdown.empty else float("nan")
        recovery_days = _recovery_time_days(equity_series, running_peak)
    else:
        max_dd, recovery_days = float("nan"), float("nan")

    closed = [t for t in trades if t.pnl is not None]
    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl <= 0]
    win_rate = len(wins) / len(closed) if closed else float("nan")
    gross_win = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else float("nan")

    total_return = (equity_series.iloc[-1] / starting_capital - 1) if not equity_series.empty else float("nan")

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "recovery_days": recovery_days,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "trade_count": len(closed),
        "total_return": total_return,
        "final_equity": float(equity_series.iloc[-1]) if not equity_series.empty else starting_capital,
    }


def _recovery_time_days(equity_series: pd.Series, running_peak: pd.Series) -> float:
    """Days from the trough of the WORST drawdown episode back to a new peak.
    NaN if the series never recovers within the available data."""
    drawdown = (running_peak - equity_series) / running_peak.replace(0, np.nan)
    if drawdown.empty or drawdown.max() <= 0:
        return 0.0
    trough_idx = drawdown.idxmax()
    peak_value_at_trough = running_peak.loc[trough_idx]
    after = equity_series.loc[trough_idx:]
    recovered = after[after >= peak_value_at_trough]
    if recovered.empty:
        return float("nan")
    recovery_idx = recovered.index[0]
    return float((recovery_idx - trough_idx).total_seconds() / 86400)


# ---------------------------------------------------------------------- #
# walk-forward windows
# ---------------------------------------------------------------------- #
def generate_walk_forward_windows(start: pd.Timestamp, end: pd.Timestamp, train_months: int = 12,
                                    test_months: int = 3) -> list:
    """Rolling windows: [train_start, train_end) then [train_end, test_end).
    Steps forward by `test_months` each iteration -- identical schedule used
    for both strategies."""
    windows = []
    train_start = pd.Timestamp(start)
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > end:
            break
        windows.append({"train_start": train_start, "train_end": train_end, "test_start": train_end, "test_end": test_end})
        train_start = train_start + pd.DateOffset(months=test_months)
    return windows


# ---------------------------------------------------------------------- #
# walk-forward parameter optimization (per-instrument grid search)
# ---------------------------------------------------------------------- #
def optimize_parameters(key: str, cfg, bars: pd.DataFrame, strategy_kind: str, param_grid: list,
                         train_start: pd.Timestamp, train_end: pd.Timestamp, slippage_pct: float,
                         starting_capital: float, use_regime: bool = False, stop_cooldown_bars: int = 0) -> dict:
    """Grid search over `param_grid` (list of param dicts), maximizing in-sample
    Sharpe on [train_start, train_end). Returns the best param dict (falls back
    to the first grid entry if no candidate produces any trades). Selection
    criterion is Sharpe only -- never win rate, which isn't correlated with
    profitability (see grid_search_landscape for profit-factor/stability
    reporting used in the broader exploration)."""
    train_bars = bars[(bars.index >= train_start) & (bars.index < train_end)]
    if train_bars.empty:
        return param_grid[0]

    best_params, best_sharpe = param_grid[0], -np.inf
    for params in param_grid:
        try:
            trades, equity = simulate_instrument(key, cfg, train_bars, strategy_kind, params,
                                                  starting_capital, slippage_pct, use_regime=use_regime,
                                                  stop_cooldown_bars=stop_cooldown_bars)
        except Exception:
            logger.exception("Grid search failed for %s params=%s", key, params)
            continue
        if not trades:
            continue
        metrics = compute_metrics(equity, trades, starting_capital)
        sharpe = metrics["sharpe"]
        if sharpe == sharpe and sharpe > best_sharpe:
            best_sharpe, best_params = sharpe, params

    return best_params


def grid_search_landscape(key: str, cfg, bars: pd.DataFrame, strategy_kind: str, param_grid: list,
                           windows: list, slippage_pct: float, starting_capital: float,
                           use_regime: bool = False, stop_cooldown_bars: int = 0) -> list:
    """For every (param, window) pair, compute BOTH in-sample and out-of-sample
    metrics (OOS uses the train-period-as-warmup fix so slow indicators are hot
    by test_start). This is the raw landscape used for plateau detection
    (stable-across-windows parameter selection) and the anti-overfitting check
    (reject IS/OOS Sharpe divergence > 30%) -- not just picking a single winner."""
    rows = []
    for params in param_grid:
        for w_idx, w in enumerate(windows):
            train_bars = bars[(bars.index >= w["train_start"]) & (bars.index < w["train_end"])]
            is_trades, is_equity = simulate_instrument(key, cfg, train_bars, strategy_kind, params,
                                                        starting_capital, slippage_pct, use_regime=use_regime,
                                                        stop_cooldown_bars=stop_cooldown_bars)
            is_metrics = compute_metrics(is_equity, is_trades, starting_capital)

            oos_warmup_bars = bars[(bars.index >= w["train_start"]) & (bars.index < w["test_end"])]
            oos_trades, oos_equity = simulate_instrument(key, cfg, oos_warmup_bars, strategy_kind, params,
                                                          starting_capital, slippage_pct, use_regime=use_regime,
                                                          active_start=w["test_start"],
                                                          stop_cooldown_bars=stop_cooldown_bars)
            oos_metrics = compute_metrics(oos_equity, oos_trades, starting_capital)

            rows.append({
                "params": params, "window": w_idx,
                "is_sharpe": is_metrics["sharpe"], "oos_sharpe": oos_metrics["sharpe"],
                "oos_profit_factor": oos_metrics["profit_factor"], "oos_trade_count": oos_metrics["trade_count"],
                "oos_max_drawdown": oos_metrics["max_drawdown"], "oos_win_rate": oos_metrics["win_rate"],
            })
    return rows


def is_oos_deviation_ok(is_sharpe: float, oos_sharpe: float, max_deviation: float = 0.30) -> bool:
    """Anti-overfitting rule: reject a combination whose OOS Sharpe deviates
    more than `max_deviation` (30% by default) from its IS Sharpe -- a big raw
    number that doesn't hold up out-of-sample is overfitting, not edge."""
    if is_sharpe != is_sharpe or oos_sharpe != oos_sharpe:
        return False
    if abs(is_sharpe) < 1e-9:
        return abs(oos_sharpe) < 1e-9
    return abs(oos_sharpe - is_sharpe) / abs(is_sharpe) <= max_deviation


# ---------------------------------------------------------------------- #
# daily covariance schedule (Strategy B only)
# ---------------------------------------------------------------------- #
def compute_daily_covariance_schedule(daily_returns: pd.DataFrame, spy_col: str = "SPY",
                                        lookback_days: int = 60, min_history: int = 30) -> dict:
    """One shrunk-covariance snapshot per trading day, using only data STRICTLY
    BEFORE that day (no lookahead). Keyed by pandas Timestamp (date-normalized)."""
    detector = RegimeDetector(corr_lookback_days=lookback_days)
    schedule = {}
    dates = daily_returns.index
    for i in range(min_history, len(dates)):
        as_of = dates[i]
        window = daily_returns.iloc[:i]  # strictly before `as_of`
        result = detector.covariance_matrices(window, spy_col=spy_col)
        if result["normal_covariance"] is None:
            continue
        schedule[pd.Timestamp(as_of.date())] = result
    return schedule


def _lookup_schedule(schedule: dict, date, max_lookback_days: int = 7):
    key = pd.Timestamp(date)
    for _ in range(max_lookback_days):
        if key in schedule:
            return schedule[key]
        key = key - pd.Timedelta(days=1)
    return None


# ---------------------------------------------------------------------- #
# full portfolio backtest (real per-strategy sizing/gating)
# ---------------------------------------------------------------------- #
def simulate_portfolio(mode: str, instruments: dict, bars_by_key: dict, params_by_key: dict,
                        starting_capital: float, slippage_pct: float, daily_cov_schedule: dict = None,
                        risk_decisions_path: str = None, active_start=None, stop_cooldown_bars: int = 0) -> dict:
    """`mode`: "A" or "B". `instruments[key]` needs .strategy_kind, .fractionable.
    Returns {"trades": [...], "equity": pd.Series, "regime_log": {key: array}}.

    `bars_by_key` may extend earlier than the period you want results for --
    pass `active_start` to compute indicators over the full history while
    only allowing new entries and returning equity records from
    `active_start` onward (see simulate_instrument's docstring for why).

    `stop_cooldown_bars` only applies to mode "A" (see simulate_instrument's
    docstring) -- mode "B" gets equivalent protection from regime gating, so
    it doesn't need an explicit cooldown for a fair comparison."""
    from bot.correlation_manager import CorrelationManager, log_sizing_decision
    from bot.kill_switch import KillSwitch
    from bot.risk_manager_b import VolTargetRiskManager

    use_regime = mode == "B"
    pre_by_key, atr_by_key, regime_by_key, min_bar_by_key = {}, {}, {}, {}

    for key, bars in bars_by_key.items():
        kind = instruments[key].strategy_kind
        params = params_by_key[key]
        if kind == "mean_reversion":
            pre_by_key[key] = precompute_mean_reversion(bars, params["sma_period"], params["std_dev_mult"])
            min_bar_by_key[key] = params["sma_period"] + 1
        elif kind == "momentum_breakout":
            pre_by_key[key] = precompute_momentum_breakout(bars, params["lookback"], params["volume_mult"])
            min_bar_by_key[key] = params["lookback"] + 1
        else:
            pre_by_key[key] = precompute_trend_following(bars, params["fast_ema"], params["slow_ema"])
            min_bar_by_key[key] = params["slow_ema"] + 1
        atr_by_key[key] = atr_series(bars, 14).values
        if use_regime:
            regime_by_key[key] = precompute_regime(bars, 14)

    events = []
    for key, bars in bars_by_key.items():
        for i in range(min_bar_by_key[key], len(bars)):
            events.append((bars.index[i], key, i))
    events.sort(key=lambda e: e[0])

    positions: dict = {}
    trades: list = []
    realized_pnl = 0.0
    last_price: dict = {}
    equity_records = []

    kill_switch = KillSwitch() if use_regime else None
    vol_tracker = EWMAVolTracker(20) if use_regime else None
    risk_manager_b = VolTargetRiskManager() if use_regime else None
    corr_manager = CorrelationManager(target_vol=risk_manager_b.target_vol) if use_regime else None
    last_date_seen = None
    last_daily_equity = starting_capital
    current_vol_cut = 1.0
    cooldown_until_by_key: dict = {}

    def running_equity():
        unreal = sum(_unrealized(p, last_price.get(p.instrument, p.entry_price)) for p in positions.values())
        return starting_capital + realized_pnl + unreal

    def gross_notional():
        return sum(abs(p.qty * last_price.get(p.instrument, p.entry_price)) for p in positions.values())

    def current_weights():
        eq = running_equity()
        if eq <= 0:
            return {}
        out = {}
        for k, p in positions.items():
            w = (p.qty * last_price.get(k, p.entry_price)) / eq
            out[k] = w if p.direction == "long" else -w
        return out

    for ts, key, i in events:
        cfg = instruments[key]
        bars = bars_by_key[key]
        price = float(bars["close"].iloc[i])
        a = float(atr_by_key[key][i])
        last_price[key] = price
        cur_regime = regime_by_key[key][i] if use_regime else None

        if use_regime:
            date = ts.date()
            if date != last_date_seen:
                eq_now = running_equity()
                if last_date_seen is not None and last_daily_equity > 0:
                    daily_ret = eq_now / last_daily_equity - 1
                    vol_tracker.update(daily_ret)
                    current_vol_cut = kill_switch.update_volatility(vol_tracker.annualized_vol, risk_manager_b.target_vol)
                last_daily_equity = eq_now
                last_date_seen = date
            kill_switch.update_drawdown(running_equity())

        position = positions.get(key)
        if position is not None:
            position.update_trailing(price, a)
            if position.stop_hit(price):
                trade = _close_trade(position, ts, price, "stop", slippage_pct)
                trades.append(trade)
                realized_pnl += trade.pnl
                del positions[key]
                position = None
                if mode == "A":
                    cooldown_until_by_key[key] = i + stop_cooldown_bars

        current_direction = position.direction if position else None
        kind = cfg.strategy_kind
        if kind == "mean_reversion":
            signal = _mean_reversion_signal(pre_by_key[key], i, price, current_direction, cur_regime, use_regime)
        elif kind == "momentum_breakout":
            signal = _momentum_signal(pre_by_key[key], i, current_direction, allow_short=False)
        else:
            signal = _trend_signal(pre_by_key[key], i, current_direction, cur_regime, use_regime)

        if signal == "exit" and position is not None:
            trade = _close_trade(position, ts, price, "signal", slippage_pct)
            trades.append(trade)
            realized_pnl += trade.pnl
            del positions[key]
            position = None
        elif signal in ("long", "short") and position is not None and position.direction != signal:
            trade = _close_trade(position, ts, price, "flip", slippage_pct)
            trades.append(trade)
            realized_pnl += trade.pnl
            del positions[key]
            position = None

        is_active = active_start is None or ts >= active_start
        in_cooldown = mode == "A" and i < cooldown_until_by_key.get(key, -1)
        if key not in positions and signal in ("long", "short") and a == a and a > 0 and is_active and not in_cooldown:
            equity = running_equity()
            trail_mult = params_by_key[key].get("trailing_atr_mult")

            if mode == "A":
                if key == "BTCUSD" and signal == "long" and _is_long(positions, "SPY") and _is_long(positions, "QQQ"):
                    pass  # binary correlation filter blocks this entry
                else:
                    qty = (equity * 0.01) / a
                    qty = min(qty, equity / price) if price > 0 else 0.0
                    # Portfolio-level exposure cap: the original spec only ever bounds a SINGLE
                    # position at up to 100% of equity notional, with no cross-instrument cap --
                    # with up to 5 instruments open at once that allowed gross exposure to reach
                    # 500-600% of equity in practice (found via the sanity-check trade dump), an
                    # amount of leverage no real account actually has. Capped at Alpaca's real
                    # margin multiplier (4x, confirmed live via account.multiplier) so the backtest
                    # can't take on leverage that couldn't actually be filled.
                    room = max(0.0, STRATEGY_A_MAX_GROSS_EXPOSURE * equity - gross_notional())
                    if price > 0:
                        qty = min(qty, room / price)
                    qty = qty if cfg.fractionable else float(int(qty))
                    if qty > 0:
                        positions[key] = _open_position(key, signal, qty, price, ts, equity, a, trail_mult, cur_regime, slippage_pct)

            else:  # mode == "B"
                if kill_switch.can_open_new_positions():
                    hist = atr_by_key[key][:i + 1]
                    atr_percentile = float((hist <= a).mean()) if len(hist) > 0 else None
                    drawdown = kill_switch.current_drawdown(equity)
                    scale = risk_manager_b.scaling_factor(vol_tracker.annualized_vol, drawdown,
                                                           instrument_vol_percentile=atr_percentile, conservative_mode=False)
                    scale *= current_vol_cut
                    base_qty = risk_manager_b.position_size(equity, a, price, scale, gross_notional(), fractionable=cfg.fractionable)

                    cov = _lookup_schedule(daily_cov_schedule, ts.date()) if daily_cov_schedule else None
                    approved_qty = base_qty
                    if cov is not None and base_qty > 0:
                        candidate_weight = (base_qty * price / equity) * (1 if signal == "long" else -1)
                        result = corr_manager.size_new_position(key, candidate_weight, current_weights(),
                                                                  cov["normal_covariance"], cov["stress_covariance"])
                        approved_qty = base_qty * result["approved_scale"]
                        if risk_decisions_path:
                            log_sizing_decision(risk_decisions_path, key, candidate_weight, current_weights(), result)

                    if approved_qty > 0 and not kill_switch.check_duplicate_order(key, signal, now=ts):
                        positions[key] = _open_position(key, signal, approved_qty, price, ts, equity, a, trail_mult, cur_regime, slippage_pct)

        if is_active:
            equity_records.append((ts, running_equity()))

    for key in list(positions.keys()):
        bars = bars_by_key[key]
        trade = _close_trade(positions[key], bars.index[-1], float(bars["close"].iloc[-1]), "end_of_data", slippage_pct)
        trades.append(trade)
        realized_pnl += trade.pnl
        del positions[key]

    equity_series = pd.Series({t: v for t, v in equity_records}).sort_index() if equity_records else pd.Series(dtype=float)
    return {"trades": trades, "equity": equity_series, "regime_log": regime_by_key}


def _is_long(positions: dict, key: str) -> bool:
    pos = positions.get(key)
    return pos is not None and pos.direction == "long"


# ---------------------------------------------------------------------- #
# Monte Carlo (independent trade-sequence bootstrap, per strategy)
# ---------------------------------------------------------------------- #
def monte_carlo_bootstrap(trades: list, starting_capital: float, n_resamples: int = 1000, seed: int = None) -> dict:
    """Resample the trade-return sequence with replacement `n_resamples` times,
    reconstruct a synthetic compounding equity path per draw, and report the
    Sharpe/max-drawdown DISTRIBUTIONS (not just point estimates). Annualization
    uses the trade frequency implied by the original backtest's span."""
    closed = [t for t in trades if t.pnl is not None]
    empty = {"sharpe_p5": float("nan"), "sharpe_p50": float("nan"), "sharpe_p95": float("nan"),
             "max_dd_p5": float("nan"), "max_dd_p50": float("nan"), "max_dd_p95": float("nan"),
             "n_trades": len(closed), "sharpe_samples": [], "max_dd_samples": []}
    if len(closed) < 5:
        return empty

    returns = np.array([t.return_pct for t in closed])
    span_days = max((closed[-1].exit_time - closed[0].entry_time).total_seconds() / 86400, 1.0)
    trades_per_year = len(closed) / (span_days / 365.25)

    rng = np.random.default_rng(seed)
    n = len(returns)
    sharpes, max_dds, total_rets = [], [], []
    for _ in range(n_resamples):
        sample = rng.choice(returns, size=n, replace=True)
        equity_path = np.concatenate([[starting_capital], starting_capital * np.cumprod(1 + sample)])
        sharpes.append(float(sample.mean() / sample.std() * np.sqrt(trades_per_year)) if sample.std() > 0 else 0.0)
        running_peak = np.maximum.accumulate(equity_path)
        dd = (running_peak - equity_path) / np.where(running_peak == 0, np.nan, running_peak)
        max_dds.append(float(np.nanmax(dd)))
        total_rets.append(float(equity_path[-1] / starting_capital - 1))

    sharpes, max_dds = np.array(sharpes), np.array(max_dds)
    return {
        "sharpe_p5": float(np.percentile(sharpes, 5)), "sharpe_p50": float(np.percentile(sharpes, 50)),
        "sharpe_p95": float(np.percentile(sharpes, 95)),
        "max_dd_p5": float(np.percentile(max_dds, 5)), "max_dd_p50": float(np.percentile(max_dds, 50)),
        "max_dd_p95": float(np.percentile(max_dds, 95)),
        "n_trades": n, "sharpe_samples": sharpes.tolist(), "max_dd_samples": max_dds.tolist(),
    }


# ---------------------------------------------------------------------- #
# paired block bootstrap significance test (A vs B, same resampled dates)
# ---------------------------------------------------------------------- #
def paired_bootstrap_significance(daily_returns_a: pd.Series, daily_returns_b: pd.Series,
                                    n_resamples: int = 1000, block_size: int = 10, seed: int = None) -> dict:
    """Block-bootstrap the SAME resampled date indices for both strategies each
    draw (paired), so the comparison reflects 'what if this stretch of market
    history had played out' for both at once, not two independent resamples."""
    aligned = pd.concat([daily_returns_a.rename("a"), daily_returns_b.rename("b")], axis=1).dropna()
    if len(aligned) < block_size * 2:
        return {"p_value": float("nan"), "observed_diff": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "significant_at_5pct": False, "n_obs": len(aligned), "diff_samples": []}

    a, b = aligned["a"].values, aligned["b"].values
    n = len(a)

    def sharpe(x):
        return float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0.0

    observed_diff = sharpe(b) - sharpe(a)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    diffs = []
    for _ in range(n_resamples):
        idx = []
        for _ in range(n_blocks):
            start = rng.integers(0, max(n - block_size, 1))
            idx.extend(range(start, min(start + block_size, n)))
        idx = np.array(idx[:n])
        diffs.append(sharpe(b[idx]) - sharpe(a[idx]))
    diffs = np.array(diffs)

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    p_value = min(float((diffs <= 0).mean()) * 2 if observed_diff >= 0 else float((diffs >= 0).mean()) * 2, 1.0)

    return {
        "observed_sharpe_a": sharpe(a), "observed_sharpe_b": sharpe(b), "observed_diff": float(observed_diff),
        "ci_low": float(ci_low), "ci_high": float(ci_high), "p_value": p_value,
        "significant_at_5pct": bool(ci_low > 0 or ci_high < 0), "n_obs": n, "diff_samples": diffs.tolist(),
    }


# ---------------------------------------------------------------------- #
# regime-period classification and attribution
# ---------------------------------------------------------------------- #
def classify_regime_periods(spy_daily_bars: pd.DataFrame, correction_dd_threshold: float = 0.10,
                              vol_shock_percentile: float = 0.90, bullish_return_window: int = 60,
                              vol_window: int = 20) -> pd.Series:
    """Label each calendar day BULLISH / CORRECTION / VOL_SHOCK / NEUTRAL using
    SPY as the market proxy: drawdown from peak > 10% -> CORRECTION; else
    realized vol in the top 10th historical percentile -> VOL_SHOCK; else a
    positive 60-day return -> BULLISH; else NEUTRAL."""
    close = spy_daily_bars["close"]
    returns = close.pct_change()
    drawdown = (close.cummax() - close) / close.cummax()
    realized_vol = returns.rolling(vol_window).std() * np.sqrt(252)
    vol_threshold = realized_vol.quantile(vol_shock_percentile)
    rolling_return = close.pct_change(bullish_return_window)

    labels = []
    for dd, vol, ret in zip(drawdown, realized_vol, rolling_return):
        if dd == dd and dd > correction_dd_threshold:
            labels.append("CORRECTION")
        elif vol == vol and vol_threshold == vol_threshold and vol > vol_threshold:
            labels.append("VOL_SHOCK")
        elif ret == ret and ret > 0:
            labels.append("BULLISH")
        else:
            labels.append("NEUTRAL")
    return pd.Series(labels, index=close.index)


def regime_breakdown(equity_series: pd.Series, trades: list, regime_labels: pd.Series) -> dict:
    empty_stats = {"days": 0, "total_return": float("nan"), "sharpe": float("nan"), "trade_count": 0}
    if equity_series.empty:
        return {regime: dict(empty_stats) for regime in ["BULLISH", "CORRECTION", "VOL_SHOCK", "NEUTRAL"]}

    daily_eq = equity_series.resample("1D").last().ffill().dropna()
    daily_ret = daily_eq.pct_change().dropna()
    label_by_date = {ts.date(): label for ts, label in regime_labels.items()}
    ret_labels = pd.Series([label_by_date.get(ts.date()) for ts in daily_ret.index], index=daily_ret.index)

    out = {}
    for regime in ["BULLISH", "CORRECTION", "VOL_SHOCK", "NEUTRAL"]:
        mask = ret_labels == regime
        rets = daily_ret[mask]
        days_this_regime = {ts.date() for ts in daily_ret.index[mask]}
        trade_count = sum(1 for t in trades if t.entry_time.date() in days_this_regime)
        if len(rets) < 2:
            out[regime] = {"days": int(mask.sum()), "total_return": float("nan"), "sharpe": float("nan"), "trade_count": trade_count}
            continue
        total_return = float((1 + rets).prod() - 1)
        sharpe_val = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else float("nan")
        out[regime] = {"days": int(mask.sum()), "total_return": total_return, "sharpe": sharpe_val, "trade_count": trade_count}
    return out


# ---------------------------------------------------------------------- #
# cross-sectional momentum (fundamentally different from the per-instrument
# time-series signals above -- ranks a basket against itself, not each
# instrument against its own history)
# ---------------------------------------------------------------------- #
def build_price_matrix(daily_bars_by_ticker: dict) -> pd.DataFrame:
    """Align each ticker's daily close onto a shared business-day index,
    forward-filled for any single-ticker gaps."""
    closes = {t: df["close"] for t, df in daily_bars_by_ticker.items() if not df.empty}
    if not closes:
        return pd.DataFrame()
    master_index = sorted(set().union(*[c.index for c in closes.values()]))
    return pd.DataFrame({t: c.reindex(master_index, method="ffill") for t, c in closes.items()})


def cross_sectional_momentum_backtest(price_matrix: pd.DataFrame, lookback_days: int, top_frac: float,
                                        starting_capital: float, slippage_pct: float, active_start=None,
                                        min_names: int = 3) -> tuple:
    """Monthly-rebalanced, long-only, equal-weighted top-`top_frac` momentum:
    at each month-end, rank all tickers by trailing `lookback_days` return,
    liquidate the whole book, and reinvest equal-weighted into the new top
    basket. `active_start`: trades before this date are computed (for
    lookback warmup) but not recorded, same convention as simulate_instrument."""
    dates = price_matrix.index
    if len(dates) < lookback_days + 21:
        return [], pd.Series(dtype=float)

    month_key = pd.Series([(d.year, d.month) for d in dates], index=dates)
    rebalance_dates = set(pd.Series(dates, index=dates).groupby(month_key).max().tolist())

    trades = []
    cash = starting_capital
    holdings: dict = {}
    entry_price: dict = {}
    entry_time: dict = {}
    equity_at_entry = starting_capital
    records = []

    for date in dates:
        row = price_matrix.loc[date]
        mtm = cash + sum(holdings.get(t, 0.0) * row[t] for t in holdings if t in row.index and not pd.isna(row[t]))
        is_active = active_start is None or date >= active_start
        if is_active:
            records.append((date, mtm))

        if date not in rebalance_dates:
            continue
        pos = dates.get_loc(date)
        if pos < lookback_days:
            continue
        lookback_date = dates[pos - lookback_days]
        trailing_ret = (price_matrix.loc[date] / price_matrix.loc[lookback_date] - 1).dropna()
        if len(trailing_ret) < min_names:
            continue
        n_top = max(min_names, int(len(trailing_ret) * top_frac))
        top_tickers = set(trailing_ret.sort_values(ascending=False).head(n_top).index)

        for t, qty in list(holdings.items()):
            raw = row.get(t)
            if raw is None or pd.isna(raw):
                continue
            fill = apply_slippage(raw, "sell", slippage_pct)
            pnl = (fill - entry_price[t]) * qty
            cash += qty * fill
            if is_active:
                trades.append(SimTrade("XSMOM:" + t, "long", entry_time[t], entry_price[t], qty,
                                        equity_at_entry, exit_time=date, exit_price=fill, pnl=pnl, exit_reason="rebalance"))
            del holdings[t]; del entry_price[t]; del entry_time[t]

        if top_tickers:
            equity_at_entry = cash
            per_name = cash / len(top_tickers)
            for t in top_tickers:
                raw = row.get(t)
                if raw is None or pd.isna(raw) or raw <= 0:
                    continue
                fill = apply_slippage(raw, "buy", slippage_pct)
                qty = per_name / fill
                holdings[t], entry_price[t], entry_time[t] = qty, fill, date
                cash -= qty * fill

    final_date = dates[-1]
    final_row = price_matrix.loc[final_date]
    for t, qty in list(holdings.items()):
        raw = final_row.get(t)
        if raw is None or pd.isna(raw):
            continue
        fill = apply_slippage(raw, "sell", slippage_pct)
        pnl = (fill - entry_price[t]) * qty
        trades.append(SimTrade("XSMOM:" + t, "long", entry_time[t], entry_price[t], qty, equity_at_entry,
                                exit_time=final_date, exit_price=fill, pnl=pnl, exit_reason="end_of_data"))

    equity_series = pd.Series({t: v for t, v in records}).sort_index() if records else pd.Series(dtype=float)
    return trades, equity_series
