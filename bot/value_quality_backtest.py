"""
Value/quality factor backtest: monthly rebalance, long the top
quintile/decile by composite score (bot/value_quality_signal.py), equal
weight within the basket, realistic transaction costs.

No leverage anywhere in this construction -- the portfolio is always
exactly 100% invested across the selected basket, equal-weighted, never
more. This is the "aggregate exposure cap" lesson from lessons_learned.md
satisfied BY CONSTRUCTION (unlike PEAD's event-driven entries, this
strategy has no mechanism that could stack exposure beyond capital, so no
separate cap parameter is needed here -- documented explicitly rather than
silently assumed, per the same lesson about documenting assumptions).

Each holding period (one ticker, one rebalance interval) is modeled as one
SimTrade and reuses bot/pead_backtest.py's `equity_curve_from_trades` (marks
to real daily closes, not interpolation -- see the mark-to-market bug
lesson in lessons_learned.md) and bot/backtest_engine.py's `compute_metrics`
/ `monte_carlo_bootstrap`, for consistency with the rest of this project.
"""
import logging

import pandas as pd

from bot.backtest_engine import SimTrade, apply_slippage, compute_metrics, monte_carlo_bootstrap
from bot.pead_backtest import equity_curve_from_trades
from bot.value_quality_signal import as_of_cross_section, build_composite_cross_section, monthly_rebalance_dates

logger = logging.getLogger(__name__)


def _price_on_or_after(price_bars: pd.DataFrame, date: pd.Timestamp):
    future = price_bars[price_bars.index >= date]
    if future.empty:
        return None, None
    return float(future["close"].iloc[0]), future.index[0]


def select_basket(scored_cross_section: pd.DataFrame, fraction: float = 0.2) -> list:
    """Top `fraction` of the ranked universe by composite_score -- 0.2 for
    quintile (default), 0.1 for decile. Ties at the cutoff are all
    included (no arbitrary tie-breaking), so the realized basket size can
    exceed round(n*fraction) by a small amount when composite scores tie
    exactly -- acceptable and documented, not silently truncated."""
    valid = scored_cross_section.dropna(subset=["composite_score"])
    if valid.empty:
        return []
    cutoff = valid["composite_score"].quantile(1 - fraction)
    return valid.loc[valid["composite_score"] >= cutoff, "symbol"].tolist()


def run_backtest(tickers: list, ticker_metrics: dict, sector_by_symbol: dict, price_bars_by_ticker: dict,
                  start: pd.Timestamp, end: pd.Timestamp, starting_capital: float = 100_000.0,
                  slippage_pct: float = 0.0010, basket_fraction: float = 0.2, min_group_size: int = 8) -> dict:
    rebalance_dates = monthly_rebalance_dates(start, end)
    if len(rebalance_dates) < 2:
        return {"trades": [], "baskets": []}

    trades, baskets = [], []
    for i in range(len(rebalance_dates) - 1):
        this_date, next_date = rebalance_dates[i], rebalance_dates[i + 1]
        cs = as_of_cross_section(ticker_metrics, this_date)
        if cs.empty:
            continue
        scored = build_composite_cross_section(cs, sector_by_symbol, min_group_size=min_group_size)
        basket = select_basket(scored, basket_fraction)
        baskets.append({"date": this_date, "basket": basket, "universe_size": len(scored)})
        if not basket:
            continue

        notional_per_name = starting_capital / len(basket)
        for symbol in basket:
            bars = price_bars_by_ticker.get(symbol)
            if bars is None or bars.empty:
                continue
            entry_price, entry_date = _price_on_or_after(bars, this_date)
            exit_price, exit_date = _price_on_or_after(bars, next_date)
            if entry_price is None or exit_price is None:
                continue
            fill_entry = apply_slippage(entry_price, "buy", slippage_pct)
            fill_exit = apply_slippage(exit_price, "sell", slippage_pct)
            qty = notional_per_name / fill_entry
            pnl = (fill_exit - fill_entry) * qty
            trades.append(SimTrade(symbol, "long", entry_date, fill_entry, qty, notional_per_name,
                                   exit_time=exit_date, exit_price=fill_exit, pnl=pnl,
                                   exit_reason="rebalance"))

    equity = equity_curve_from_trades(trades, starting_capital, price_bars_by_ticker)
    metrics = compute_metrics(equity, trades, starting_capital)
    mc = monte_carlo_bootstrap(trades, starting_capital, n_resamples=1000, seed=42)

    return {"trades": trades, "baskets": baskets, "equity": equity, "metrics": metrics, "mc": mc,
           "rebalance_dates": rebalance_dates}


def cost_sensitivity(tickers: list, ticker_metrics: dict, sector_by_symbol: dict, price_bars_by_ticker: dict,
                      start: pd.Timestamp, end: pd.Timestamp, starting_capital: float = 100_000.0,
                      basket_fraction: float = 0.2, min_group_size: int = 8,
                      slippage_scenarios=(0.0005, 0.0010, 0.0015)) -> dict:
    """Reruns the full backtest at each slippage scenario (basket
    selection is identical across scenarios -- only fills change -- but
    rerunning end to end is simplest and matches this project's existing
    PEAD scripts' approach)."""
    out = {}
    for slip in slippage_scenarios:
        r = run_backtest(tickers, ticker_metrics, sector_by_symbol, price_bars_by_ticker, start, end,
                         starting_capital, slippage_pct=slip, basket_fraction=basket_fraction,
                         min_group_size=min_group_size)
        equity = r["equity"]
        out[slip] = float(equity.iloc[-1] / starting_capital - 1) if not equity.empty else float("nan")
    return out


def regime_slice_metrics(trades: list, price_bars_by_ticker: dict, starting_capital: float,
                          sub_start: pd.Timestamp, sub_end: pd.Timestamp) -> dict:
    """Metrics for just the trades ENTERED within [sub_start, sub_end) --
    for reporting performance during a specific regime/recession window,
    same pattern as run_pead_early_exit_2016_2020.py's sub-period
    breakdown. Not a re-optimization -- the basket/trades were already
    fixed by `run_backtest`; this only slices which ones to report on."""
    sub_trades = [t for t in trades if sub_start <= t.entry_time < sub_end]
    if not sub_trades:
        return {"trade_count": 0}
    equity = equity_curve_from_trades(sub_trades, starting_capital, price_bars_by_ticker)
    return compute_metrics(equity, sub_trades, starting_capital)


def viability_verdict(metrics: dict, mc: dict, cost_returns: dict) -> dict:
    sharpe, dd95 = metrics.get("sharpe"), mc.get("max_dd_p95")
    cost_15 = cost_returns.get(0.0015)
    checks = {
        "sharpe_ge_0.5": (sharpe == sharpe and sharpe >= 0.5),
        "mc_dd95_le_20pct": (dd95 == dd95 and dd95 <= 0.20),
        "cost_15bps_positive": (cost_15 == cost_15 and cost_15 > 0),
    }
    return {**checks, "all_pass": all(checks.values())}


if __name__ == "__main__":
    # Mechanics self-test on SYNTHETIC price+metrics data -- proves the
    # rebalance/basket-selection/trade-construction/equity-curve plumbing
    # runs correctly end to end. NOT a backtest result -- see
    # exploration_log_value_quality.md for why the real run is still
    # pending (sector data collection blocked on Alpha Vantage's quota).
    import numpy as np
    rng = np.random.default_rng(7)
    tickers = [f"T{i}" for i in range(20)]
    sector_by_symbol = {t: {"sector": "Sector" + str(i % 2), "industry": "Industry" + str(i % 4)}
                        for i, t in enumerate(tickers)}

    dates = pd.date_range("2019-01-31", "2020-12-31", freq="QE", tz="UTC")
    ticker_metrics, price_bars_by_ticker = {}, {}
    daily_index = pd.date_range("2019-01-01", "2021-02-01", freq="B", tz="UTC")
    for t in tickers:
        n = len(dates)
        ticker_metrics[t] = pd.DataFrame({
            "symbol": t, "fiscal_date_ending": dates, "available_date": dates + pd.Timedelta(days=35),
            "price_used_date": dates, "price": rng.uniform(20, 200, n),
            "price_to_book": rng.uniform(0.5, 8, n), "price_to_earnings": rng.uniform(5, 40, n),
            "net_margin_ttm": rng.uniform(0.02, 0.35, n), "debt_to_equity": rng.uniform(0.1, 3.5, n),
            "earnings_stability": rng.uniform(-1, 0, n),
        })
        walk = np.cumsum(rng.normal(0, 1, len(daily_index)))
        price_bars_by_ticker[t] = pd.DataFrame({"close": 100 + walk}, index=daily_index)

    result = run_backtest(tickers, ticker_metrics, sector_by_symbol, price_bars_by_ticker,
                          pd.Timestamp("2019-02-01", tz="UTC"), pd.Timestamp("2020-12-31", tz="UTC"))
    print(f"Trades: {len(result['trades'])}, baskets: {len(result['baskets'])}")
    print("Metrics:", result["metrics"])
    print("Basket sizes over time:", [(b["date"].date(), len(b["basket"])) for b in result["baskets"]])
