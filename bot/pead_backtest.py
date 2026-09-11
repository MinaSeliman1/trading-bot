"""
Preliminary PEAD (post-earnings announcement drift) backtest: long on a
significant positive earnings surprise, hold for a fixed number of trading
days, exit. Event-driven (not a rolling-bar signal), so it gets its own
small engine rather than being shoehorned into backtest_engine.py's
per-bar dispatch -- but reuses the same primitives (SimTrade, slippage,
compute_metrics, Monte Carlo, the 30% anti-overfitting rule) for consistency
with the rest of this project's methodology.
"""
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from bot.backtest_engine import SimTrade, apply_slippage, compute_metrics, monte_carlo_bootstrap, is_oos_deviation_ok

logger = logging.getLogger(__name__)

POSITION_NOTIONAL = 10_000.0  # fixed $ per trade -- preliminary edge-existence test, not portfolio-constrained sizing

# Portfolio-level caps, same spirit as Strategy B's exposure control: total
# gross exposure across ALL simultaneously-open PEAD positions is capped as
# a multiple of capital (not just a per-position cap), and the number of
# NEW positions opened on any single entry day is also capped -- when more
# qualifying signals compete for limited capacity than can be taken, the
# larger surprises win priority, the rest are skipped outright (not
# partial-sized), matching "prioritize, don't take everything."
PEAD_MAX_GROSS_EXPOSURE = 1.5     # x starting capital -- lower end of the requested 1.5-2x range
PEAD_MAX_NEW_POSITIONS_PER_DAY = 3


def _raw_candidates(symbol: str, earnings: pd.DataFrame, daily_bars: pd.DataFrame,
                     surprise_threshold_pct: float, hold_days: int, window_start=None, window_end=None,
                     early_exit_by_date: dict = None) -> list:
    """Qualifying earnings events for one ticker, resolved to concrete entry/exit
    dates and raw (pre-slippage) prices -- the part of event detection that's
    identical whether trades are sized independently or portfolio-aware.

    `early_exit_by_date`: optional {date -> bool} market-wide signal (e.g. SPY
    close below its own 200-day SMA), checked once per calendar day strictly
    AFTER entry. If it's True on some day before the normal hold_days exit,
    the trade exits there instead -- never later than the hold_days exit,
    never on the entry day itself. Unlike the (rejected) entry-day regime
    gate, this reacts to the regime trajectory across the WHOLE hold period,
    not just its state at entry."""
    if earnings.empty or daily_bars.empty:
        return []

    candidates = []
    bar_dates = daily_bars.index
    for report_time, row in earnings.iterrows():
        if window_start is not None and report_time < window_start:
            continue
        if window_end is not None and report_time >= window_end:
            continue
        if row["surprise_pct"] != row["surprise_pct"] or row["surprise_pct"] < surprise_threshold_pct:
            continue

        report_date = report_time.normalize()
        same_day_ok = str(row.get("report_time", "")).lower() == "pre-market"
        entry_candidates = bar_dates[bar_dates.normalize() >= report_date] if same_day_ok else \
                            bar_dates[bar_dates.normalize() > report_date]
        if len(entry_candidates) == 0:
            continue
        entry_idx = bar_dates.get_loc(entry_candidates[0])
        planned_exit_idx = entry_idx + hold_days
        if planned_exit_idx >= len(bar_dates):
            continue  # not enough future data to complete the hold period

        exit_idx = planned_exit_idx
        if early_exit_by_date:
            for i in range(entry_idx + 1, planned_exit_idx + 1):
                if early_exit_by_date.get(bar_dates[i].date(), False):
                    exit_idx = i
                    break

        entry_ts, exit_ts = bar_dates[entry_idx], bar_dates[exit_idx]
        raw_entry = float(daily_bars["open"].iloc[entry_idx])
        raw_exit = float(daily_bars["close"].iloc[exit_idx])
        if raw_entry <= 0 or raw_exit <= 0:
            continue

        candidates.append({"symbol": symbol, "entry_time": entry_ts, "exit_time": exit_ts,
                           "raw_entry": raw_entry, "raw_exit": raw_exit, "surprise_pct": row["surprise_pct"],
                           "early_exit": exit_idx < planned_exit_idx})
    return candidates


def build_event_trades(symbol: str, earnings: pd.DataFrame, daily_bars: pd.DataFrame,
                        surprise_threshold_pct: float, hold_days: int, slippage_pct: float,
                        window_start=None, window_end=None) -> list:
    """One long trade per qualifying earnings event, sized independently at a
    fixed notional with NO awareness of other open positions -- entry at the
    next tradeable day's open (same day if pre-market since the market
    hasn't opened yet, next trading day if post-market/unknown -- never
    before the surprise was public), exit `hold_days` trading days later at
    the close. `window_start`/`window_end` restrict which EVENT dates are
    eligible (for train/test splitting) -- the exit can fall after
    window_end. See `build_portfolio_event_trades` for the exposure-capped
    version."""
    candidates = _raw_candidates(symbol, earnings, daily_bars, surprise_threshold_pct, hold_days,
                                  window_start, window_end)
    trades = []
    for c in candidates:
        fill_entry = apply_slippage(c["raw_entry"], "buy", slippage_pct)
        fill_exit = apply_slippage(c["raw_exit"], "sell", slippage_pct)
        qty = POSITION_NOTIONAL / fill_entry
        pnl = (fill_exit - fill_entry) * qty
        trades.append(SimTrade(symbol, "long", c["entry_time"], fill_entry, qty, POSITION_NOTIONAL,
                               exit_time=c["exit_time"], exit_price=fill_exit, pnl=pnl, exit_reason="hold_period_end"))
    return trades


def build_portfolio_event_trades(tickers: list, earnings_by_ticker: dict, bars_by_ticker: dict,
                                   surprise_threshold_pct: float, hold_days: int, slippage_pct: float,
                                   starting_capital: float, window_start=None, window_end=None,
                                   max_gross_exposure: float = PEAD_MAX_GROSS_EXPOSURE,
                                   max_new_positions_per_day: int = PEAD_MAX_NEW_POSITIONS_PER_DAY,
                                   position_notional: float = POSITION_NOTIONAL,
                                   regime_labels: pd.Series = None,
                                   blocked_regimes: frozenset = frozenset({"CORRECTION"}),
                                   early_exit_by_date: dict = None) -> list:
    """Portfolio-aware version: processes ALL tickers' qualifying events
    together in chronological order, capping aggregate gross notional
    exposure across simultaneously-open positions and the number of NEW
    positions opened on any single entry day. When more qualifying signals
    compete for capacity than can be taken on a given day, larger surprises
    get priority and the rest are skipped outright.

    `regime_labels`: optional daily Series of regime labels (e.g. from
    `backtest_engine.classify_regime_periods` on SPY) -- if given, no new
    entry is taken on a day whose regime is in `blocked_regimes` (default:
    block CORRECTION only, matching the diagnosed failure mode). Positions
    already open are unaffected; this only gates NEW entries.

    `early_exit_by_date`: optional {date -> bool} market-wide signal (see
    `_raw_candidates`) -- cuts a position's hold short if it fires on some
    day between entry and the planned hold_days exit. Passed straight
    through to `_raw_candidates`; independent of `regime_labels` (entry
    gate) above -- the two can be combined or used separately."""
    candidates = []
    for t in tickers:
        candidates += _raw_candidates(t, earnings_by_ticker.get(t, pd.DataFrame()), bars_by_ticker.get(t, pd.DataFrame()),
                                       surprise_threshold_pct, hold_days, window_start, window_end,
                                       early_exit_by_date)
    if not candidates:
        return []

    if regime_labels is not None:
        label_by_date = {ts.date(): label for ts, label in regime_labels.items()}
        candidates = [c for c in candidates if label_by_date.get(c["entry_time"].date()) not in blocked_regimes]
        if not candidates:
            return []

    by_day = defaultdict(list)
    for c in candidates:
        by_day[c["entry_time"]].append(c)

    # a position's exit day may fall on a date with no NEW candidates -- iterate
    # the full calendar of relevant dates (all candidate entry days + all
    # candidate exit days) so exposure is released exactly when it should be,
    # not only when the next candidate happens to show up.
    all_relevant_days = sorted(set(by_day.keys()) | {c["exit_time"] for c in candidates})

    trades = []
    open_positions = []  # each: dict with symbol/entry_time/exit_time/fill_entry/raw_exit/qty/notional/equity_at_entry

    for day in all_relevant_days:
        still_open = []
        for p in open_positions:
            if p["exit_time"] <= day:
                fill_exit = apply_slippage(p["raw_exit"], "sell", slippage_pct)
                pnl = (fill_exit - p["fill_entry"]) * p["qty"]
                trades.append(SimTrade(p["symbol"], "long", p["entry_time"], p["fill_entry"], p["qty"],
                                       p["equity_at_entry"], exit_time=p["exit_time"], exit_price=fill_exit,
                                       pnl=pnl, exit_reason="hold_period_end"))
            else:
                still_open.append(p)
        open_positions = still_open

        todays_candidates = sorted(by_day.get(day, []), key=lambda c: -c["surprise_pct"])
        if not todays_candidates:
            continue

        current_gross = sum(p["notional"] for p in open_positions)
        new_positions_today = 0
        for c in todays_candidates:
            if new_positions_today >= max_new_positions_per_day:
                logger.debug("Daily new-position cap reached on %s, skipping remaining lower-priority signals", day)
                break
            room = max_gross_exposure * starting_capital - current_gross
            if room < position_notional * 0.5:  # not enough room for at least half a normal position
                logger.debug("Gross exposure cap reached on %s, skipping remaining signals", day)
                break
            notional = min(position_notional, room)
            fill_entry = apply_slippage(c["raw_entry"], "buy", slippage_pct)
            qty = notional / fill_entry
            open_positions.append({"symbol": c["symbol"], "entry_time": c["entry_time"], "exit_time": c["exit_time"],
                                   "fill_entry": fill_entry, "raw_exit": c["raw_exit"], "qty": qty,
                                   "notional": notional, "equity_at_entry": starting_capital})
            current_gross += notional
            new_positions_today += 1

    for p in open_positions:  # close anything still open at the end of the range
        fill_exit = apply_slippage(p["raw_exit"], "sell", slippage_pct)
        pnl = (fill_exit - p["fill_entry"]) * p["qty"]
        trades.append(SimTrade(p["symbol"], "long", p["entry_time"], p["fill_entry"], p["qty"], p["equity_at_entry"],
                               exit_time=p["exit_time"], exit_price=fill_exit, pnl=pnl, exit_reason="end_of_data"))

    return trades


def equity_curve_from_trades(trades: list, starting_capital: float, bars_by_ticker: dict = None) -> pd.Series:
    """Mark-to-market daily equity from a set of possibly-overlapping event
    trades: realized P&L accumulates at each exit.

    If `bars_by_ticker` (symbol -> daily bars DataFrame with a 'close'
    column) is given, unrealized P&L during each trade's holding period is
    marked to the REAL daily closing price -- this is the correct method
    and should be used for any real analysis, since it preserves genuine
    day-to-day price volatility. Without it (or for a symbol missing from
    the dict), falls back to linearly interpolating from entry to exit
    price day-by-day.

    The interpolation fallback DOES NOT produce a valid Sharpe/drawdown: a
    straight-line ramp to a known endpoint has almost no day-to-day
    variance, which inflates Sharpe by orders of magnitude and hides any
    intra-trade drawdown entirely (verified empirically: >50x Sharpe
    inflation, 0% vs 2%+ max drawdown, for the identical trade). Every
    real PEAD run must pass `bars_by_ticker` -- the fallback exists only
    for tests/callers that don't have price data and don't need valid
    risk metrics from the result."""
    if not trades:
        return pd.Series(dtype=float)

    all_dates = sorted(set(d for t in trades for d in (t.entry_time, t.exit_time)))
    daily_index = pd.bdate_range(all_dates[0], all_dates[-1])

    close_by_symbol = {}
    if bars_by_ticker:
        for symbol, bars in bars_by_ticker.items():
            if bars is None or bars.empty:
                continue
            s = bars["close"].copy()
            s.index = s.index.normalize()
            s = s[~s.index.duplicated(keep="last")]
            close_by_symbol[symbol] = s.reindex(daily_index.union(s.index)).sort_index().ffill().reindex(daily_index)

    realized = pd.Series(0.0, index=daily_index)
    for t in trades:
        exit_day = pd.Timestamp(t.exit_time).normalize()
        pos = daily_index.searchsorted(exit_day)
        if pos < len(daily_index) and daily_index[pos] == exit_day:
            realized.iloc[pos] += t.pnl

    cum_realized = realized.cumsum()

    unrealized = pd.Series(0.0, index=daily_index)
    for t in trades:
        entry_day = pd.Timestamp(t.entry_time).normalize()
        exit_day = pd.Timestamp(t.exit_time).normalize()
        span = daily_index[(daily_index >= entry_day) & (daily_index < exit_day)]
        if len(span) == 0:
            continue

        closes = close_by_symbol.get(t.instrument)
        marked = closes.reindex(span) if closes is not None else None
        if marked is not None and marked.notna().all():
            unrealized.loc[span] += (marked.to_numpy() - t.entry_price) * t.qty
        else:
            frac = np.linspace(0, 1, len(span), endpoint=False)
            unrealized.loc[span] += frac * t.pnl

    equity = starting_capital + cum_realized + unrealized
    return equity


def daily_returns_from_equity(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return equity
    return equity.pct_change().dropna()
