"""
Re-run of the exact same walk-forward PEAD methodology as run_pead_preliminary.py
(same 23 tickers, same original grid [3,5,7,10]% x [20,40,60]d, same 7 windows)
but through build_portfolio_event_trades (gross exposure capped at 1.5x
capital, max 3 new positions/day, priority by surprise magnitude) instead of
independent per-ticker sizing. Answers directly: does the 80.1% MC_DD95 drop
once uncontrolled position stacking is capped, or does it persist (pointing
at the underlying PEAD signal itself, not the sizing)?
"""
import logging
from datetime import datetime, timedelta, timezone

import numpy as np

import config
from bot.data_client import DataClient
from bot.earnings_data import AlphaVantageEarningsClient
from bot.pead_backtest import build_portfolio_event_trades, equity_curve_from_trades
from bot.backtest_engine import compute_metrics, monte_carlo_bootstrap, generate_walk_forward_windows, is_oos_deviation_ok

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pead_capped")

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

    price_start = max(b.index.min() for b in bars_by_ticker.values())
    price_end = min(b.index.max() for b in bars_by_ticker.values())
    event_start = min(e.index.min() for e in earnings_by_ticker.values() if not e.empty)
    event_end = max(e.index.max() for e in earnings_by_ticker.values() if not e.empty)
    overall_start = max(price_start, event_start)
    overall_end = min(price_end, event_end)
    windows = generate_walk_forward_windows(overall_start, overall_end, train_months=24, test_months=6)

    log_md(f"\n## Re-test with portfolio-level exposure cap "
          f"(max_gross_exposure={MAX_GROSS_EXPOSURE}x, max_new_positions_per_day={MAX_NEW_POSITIONS_PER_DAY}, "
          f"priority by surprise magnitude)\n")
    log_md(f"- Identical setup to the original run: same 23 tickers, same grid {THRESHOLD_GRID} x "
          f"{HOLD_DAYS_GRID}, same 7 walk-forward windows (24mo train / 6mo test). Only change: "
          f"trades now go through `build_portfolio_event_trades` instead of independent per-ticker sizing.")

    window_results = []
    oos_trades_all = []

    for w_idx, w in enumerate(windows):
        best_params, best_is_sharpe = None, -np.inf
        for thresh in THRESHOLD_GRID:
            for hold in HOLD_DAYS_GRID:
                is_trades = build_portfolio_event_trades(TICKERS, earnings_by_ticker, bars_by_ticker, thresh, hold,
                                                          BASELINE_SLIPPAGE, STARTING_CAPITAL,
                                                          w["train_start"], w["train_end"],
                                                          MAX_GROSS_EXPOSURE, MAX_NEW_POSITIONS_PER_DAY)
                if len(is_trades) < 5:
                    continue
                is_equity = equity_curve_from_trades(is_trades, STARTING_CAPITAL, bars_by_ticker)
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
                                                   MAX_GROSS_EXPOSURE, MAX_NEW_POSITIONS_PER_DAY)
        oos_equity = equity_curve_from_trades(oos_trades, STARTING_CAPITAL, bars_by_ticker)
        oos_metrics = compute_metrics(oos_equity, oos_trades, STARTING_CAPITAL)
        deviation_ok = is_oos_deviation_ok(best_is_sharpe, oos_metrics["sharpe"], 0.30)

        log_md(f"- Window {w_idx}: params thresh={thresh}% hold={hold}d, IS Sharpe={best_is_sharpe:.3f}, "
              f"OOS Sharpe={oos_metrics['sharpe']:.3f}, OOS trades={oos_metrics['trade_count']}, "
              f"{'OK' if deviation_ok else 'FLAG (>30% IS/OOS deviation)'}")

        window_results.append({"window": w_idx, "params": best_params, "deviation_ok": deviation_ok})
        oos_trades_all.extend(oos_trades)

    stitched = equity_curve_from_trades(oos_trades_all, STARTING_CAPITAL, bars_by_ticker)
    overall_metrics = compute_metrics(stitched, oos_trades_all, STARTING_CAPITAL)
    mc = monte_carlo_bootstrap(oos_trades_all, STARTING_CAPITAL, n_resamples=1000, seed=42)

    n_flagged = sum(1 for wr in window_results if not wr["deviation_ok"])
    log_md(f"\n### Combined out-of-sample result (exposure-capped)\n")
    log_md(f"- Sharpe={overall_metrics['sharpe']:.3f}  Sortino={overall_metrics['sortino']:.3f}  "
          f"MaxDD={overall_metrics['max_drawdown']*100:.1f}%  **MC_DD95={mc['max_dd_p95']*100:.1f}%**  "
          f"PF={overall_metrics['profit_factor']:.3f}  WinRate={overall_metrics['win_rate']*100:.1f}%  "
          f"Trades={overall_metrics['trade_count']}  ({n_flagged}/{len(window_results)} windows flagged)")

    log_md(f"\n**Comparison: uncapped MC_DD95 was 80.1% (131 trades) -> capped MC_DD95 is "
          f"{mc['max_dd_p95']*100:.1f}% ({overall_metrics['trade_count']} trades).**")

    if mc["max_dd_p95"] == mc["max_dd_p95"] and mc["max_dd_p95"] < 0.80 * 0.7:  # meaningfully lower, not just noise
        log_md(f"\n**CONCLUSION: the drawdown problem was substantially structural/sizing-driven -- "
              f"capping exposure meaningfully reduced tail risk. This points at engineering (position "
              f"sizing discipline), not the underlying PEAD signal, as the primary issue.**")
    elif mc["max_dd_p95"] == mc["max_dd_p95"] and mc["max_dd_p95"] <= 0.20:
        log_md(f"\n**CONCLUSION: capping exposure resolved the drawdown problem entirely -- MC_DD95 now "
              f"clears the 20% viability bar.**")
    else:
        log_md(f"\n**CONCLUSION: MC_DD95 remains high even after capping exposure -- the tail risk is "
              f"NOT primarily a sizing/stacking artifact. This is a more concerning signal about the "
              f"underlying PEAD signal itself (e.g. a genuinely fat-tailed per-trade return distribution "
              f"even for isolated, uncorrelated positions).**")

    return overall_metrics, mc


if __name__ == "__main__":
    main()
