"""
Exact same walk-forward setup as run_pead_capped.py (same 23 tickers, same
grid, same 7 windows, same 1.5x exposure cap / 3-per-day / priority rule)
plus a market regime gate: no new entries on days SPY is in CORRECTION
regime (reusing backtest_engine.classify_regime_periods, the same
classifier used for the diagnostic). Decisive test: does this eliminate
windows 0/1's losses without destroying the rest of the performance?
"""
import logging
from datetime import datetime, timedelta, timezone

import numpy as np

import config
from bot.data_client import DataClient
from bot.earnings_data import AlphaVantageEarningsClient
from bot.pead_backtest import build_portfolio_event_trades, equity_curve_from_trades
from bot.backtest_engine import (
    compute_metrics, monte_carlo_bootstrap, generate_walk_forward_windows, is_oos_deviation_ok,
    classify_regime_periods,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pead_regime_filtered")

LOG_PATH = "exploration_log_pead.md"
STARTING_CAPITAL = 100_000.0
BASELINE_SLIPPAGE = 0.0010
TICKERS = ["JPM", "BAC", "GS", "JNJ", "UNH", "PFE", "ABBV", "PG", "KO", "WMT",
           "MCD", "CAT", "HON", "UPS", "XOM", "CVX", "VZ", "DIS", "HD", "COST",
           "LIN", "DUK", "V"]
THRESHOLD_GRID = [3.0, 5.0, 7.0, 10.0]
HOLD_DAYS_GRID = [20, 40, 60]
MAX_GROSS_EXPOSURE = 1.5
MAX_NEW_POSITIONS_PER_DAY = 3
SLIPPAGE_SCENARIOS = [0.0005, 0.0010, 0.0015]


def log_md(text: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    logger.info(text.split("\n")[0][:200])


def main():
    dc = DataClient(config.ALPACA_API_KEY, config.ALPACA_API_SECRET, config.ALPACA_DATA_FEED)
    av = AlphaVantageEarningsClient(config.ALPHA_VANTAGE_API_KEY)
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=365 * 6)

    bars_by_ticker, earnings_by_ticker = {}, {}
    for t in TICKERS:
        bars_by_ticker[t] = dc.get_bars(t, "us_equity", "1Day", start, end)
        earnings_by_ticker[t] = av.get_quarterly_surprises(t)  # cached, no new API calls

    spy_daily = dc.get_bars("SPY", "us_equity", "1Day", start, end)  # cached from round 2
    regime_labels = classify_regime_periods(spy_daily)
    logger.info("Regime labels: %s", regime_labels.value_counts().to_dict())

    price_start = max(b.index.min() for b in bars_by_ticker.values())
    price_end = min(b.index.max() for b in bars_by_ticker.values())
    event_start = min(e.index.min() for e in earnings_by_ticker.values() if not e.empty)
    event_end = max(e.index.max() for e in earnings_by_ticker.values() if not e.empty)
    overall_start = max(price_start, event_start)
    overall_end = min(price_end, event_end)
    windows = generate_walk_forward_windows(overall_start, overall_end, train_months=24, test_months=6)

    log_md(f"\n## Regime-filtered re-test: block new entries when SPY is in CORRECTION\n")
    log_md(f"- Identical setup to the capped-only run (same 23 tickers, same grid, same 7 windows, "
          f"same 1.5x exposure cap / 3-per-day / surprise-priority). Only addition: no new position "
          f"opens on a day SPY's regime label (via `classify_regime_periods`, same classifier used "
          f"for the diagnostic) is CORRECTION.")

    window_results = []
    oos_trades_all = []

    for w_idx, w in enumerate(windows):
        best_params, best_is_sharpe = None, -np.inf
        for thresh in THRESHOLD_GRID:
            for hold in HOLD_DAYS_GRID:
                is_trades = build_portfolio_event_trades(TICKERS, earnings_by_ticker, bars_by_ticker, thresh, hold,
                                                          BASELINE_SLIPPAGE, STARTING_CAPITAL,
                                                          w["train_start"], w["train_end"],
                                                          MAX_GROSS_EXPOSURE, MAX_NEW_POSITIONS_PER_DAY,
                                                          regime_labels=regime_labels)
                if len(is_trades) < 5:
                    continue
                is_equity = equity_curve_from_trades(is_trades, STARTING_CAPITAL)
                is_metrics = compute_metrics(is_equity, is_trades, STARTING_CAPITAL)
                if is_metrics["sharpe"] == is_metrics["sharpe"] and is_metrics["sharpe"] > best_is_sharpe:
                    best_is_sharpe, best_params = is_metrics["sharpe"], (thresh, hold)

        if best_params is None:
            log_md(f"- Window {w_idx}: no param combo produced >=5 IS trades, skipped.")
            continue

        thresh, hold = best_params
        oos_trades = build_portfolio_event_trades(TICKERS, earnings_by_ticker, bars_by_ticker, thresh, hold,
                                                   BASELINE_SLIPPAGE, STARTING_CAPITAL,
                                                   w["test_start"], w["test_end"],
                                                   MAX_GROSS_EXPOSURE, MAX_NEW_POSITIONS_PER_DAY,
                                                   regime_labels=regime_labels)
        oos_equity = equity_curve_from_trades(oos_trades, STARTING_CAPITAL)
        oos_metrics = compute_metrics(oos_equity, oos_trades, STARTING_CAPITAL)
        deviation_ok = is_oos_deviation_ok(best_is_sharpe, oos_metrics["sharpe"], 0.30)

        log_md(f"- Window {w_idx}: params thresh={thresh}% hold={hold}d, IS Sharpe={best_is_sharpe:.3f}, "
              f"OOS Sharpe={oos_metrics['sharpe']:.3f}, OOS trades={oos_metrics['trade_count']}, "
              f"{'OK' if deviation_ok else 'FLAG (>30% IS/OOS deviation)'}")

        window_results.append({"window": w_idx, "params": best_params, "deviation_ok": deviation_ok})
        oos_trades_all.extend(oos_trades)

    stitched = equity_curve_from_trades(oos_trades_all, STARTING_CAPITAL)
    overall_metrics = compute_metrics(stitched, oos_trades_all, STARTING_CAPITAL)
    mc = monte_carlo_bootstrap(oos_trades_all, STARTING_CAPITAL, n_resamples=1000, seed=42)
    n_flagged = sum(1 for wr in window_results if not wr["deviation_ok"])

    log_md(f"\n### Combined out-of-sample result (exposure-capped + regime-filtered)\n")
    log_md(f"- Sharpe={overall_metrics['sharpe']:.3f}  Sortino={overall_metrics['sortino']:.3f}  "
          f"MaxDD={overall_metrics['max_drawdown']*100:.1f}%  MC_DD95={mc['max_dd_p95']*100:.1f}%  "
          f"PF={overall_metrics['profit_factor']:.3f}  WinRate={overall_metrics['win_rate']*100:.1f}%  "
          f"Trades={overall_metrics['trade_count']}  ({n_flagged}/{len(window_results)} windows flagged)")

    cost_returns = {}
    for slip in SLIPPAGE_SCENARIOS:
        all_slip_trades = []
        for wr in window_results:
            thresh, hold = wr["params"]
            w = windows[wr["window"]]
            all_slip_trades += build_portfolio_event_trades(TICKERS, earnings_by_ticker, bars_by_ticker, thresh, hold,
                                                             slip, STARTING_CAPITAL, w["test_start"], w["test_end"],
                                                             MAX_GROSS_EXPOSURE, MAX_NEW_POSITIONS_PER_DAY,
                                                             regime_labels=regime_labels)
        s = equity_curve_from_trades(all_slip_trades, STARTING_CAPITAL)
        cost_returns[slip] = float(s.iloc[-1] / STARTING_CAPITAL - 1) if not s.empty else float("nan")
    log_md(f"- Cost sensitivity: " + ", ".join(f"{k*100:.2f}%->{v*100:.1f}%" for k, v in cost_returns.items()))

    log_md(f"\n**Comparison table**")
    log_md(f"| | Uncapped | Capped only | Capped + regime-filtered |")
    log_md(f"|---|---|---|---|")
    log_md(f"| Sharpe | 2.93 | 2.41 | {overall_metrics['sharpe']:.2f} |")
    log_md(f"| MC_DD95 | 80.1% | 13.0% | {mc['max_dd_p95']*100:.1f}% |")
    log_md(f"| Trades | 131 | 120 | {overall_metrics['trade_count']} |")
    log_md(f"| Windows flagged | 6/7 | 6/7 | {n_flagged}/{len(window_results)} |")
    log_md(f"| Cost@0.15% survival | +17.3% | +14.7% | {cost_returns[0.0015]*100:+.1f}% |")

    return overall_metrics, mc, window_results


if __name__ == "__main__":
    main()
