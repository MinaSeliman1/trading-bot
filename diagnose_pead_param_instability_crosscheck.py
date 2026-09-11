"""
Cross-check for diagnose_pead_param_instability.py's finding: on the full-97
universe, thresh=10%/hold=20d had the worst-or-near-worst IS Sharpe rank in
early windows yet turned out to be the most stable, highest OOS Sharpe combo
in EVERY one of the 5 windows (pooled mean OOS Sharpe 1.50 vs 0.41-1.17 for
every other grid cell). That finding was itself spotted by looking at OOS
outcomes across the same 5 windows -- same trap the project has flagged
before (picking a "winner" after seeing the test result). This script checks
whether the same pattern reproduces on the 74-ticker NEW-ONLY universe
(disjoint tickers, same time periods) -- if thresh=10/hold=20 is ALSO the
standout there, that is real cross-sectional corroboration, not restating
the same cherry-pick under a new name.
"""
import logging
from datetime import datetime, timedelta, timezone

import numpy as np

import config
from bot.data_client import DataClient
from bot.earnings_data import AlphaVantageEarningsClient
from bot.pead_backtest import build_portfolio_event_trades, equity_curve_from_trades
from bot.backtest_engine import compute_metrics, generate_walk_forward_windows
from run_pead_viability_check import discover_tickers, MAX_GROSS_EXPOSURE, MAX_NEW_POSITIONS_PER_DAY
from run_pead_sensitivity import ORIGINAL_23

logging.basicConfig(level=logging.WARNING)

STARTING_CAPITAL = 100_000.0
BASELINE_SLIPPAGE = 0.0010
THRESHOLD_GRID = [3.0, 5.0, 7.0, 10.0]
HOLD_DAYS_GRID = [20, 40, 60]
TRAIN_MONTHS, TEST_MONTHS = 24, 6


def main():
    all_tickers = discover_tickers()
    tickers = [t for t in all_tickers if t not in ORIGINAL_23]  # new-only 74

    dc = DataClient(config.ALPACA_API_KEY, config.ALPACA_API_SECRET, config.ALPACA_DATA_FEED)
    av = AlphaVantageEarningsClient(config.ALPHA_VANTAGE_API_KEY)
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=365 * 6)

    bars_by_ticker, earnings_by_ticker = {}, {}
    for t in tickers:
        bars_by_ticker[t] = dc.get_bars(t, "us_equity", "1Day", start, end)
        earnings_by_ticker[t] = av.get_quarterly_surprises(t)
    tickers = [t for t in tickers if not bars_by_ticker[t].empty and not earnings_by_ticker[t].empty]

    price_start = max(b.index.min() for b in bars_by_ticker.values() if not b.empty)
    price_end = min(b.index.max() for b in bars_by_ticker.values() if not b.empty)
    event_start = min(e.index.min() for e in earnings_by_ticker.values() if not e.empty)
    event_end = max(e.index.max() for e in earnings_by_ticker.values() if not e.empty)
    overall_start, overall_end = max(price_start, event_start), min(price_end, event_end)
    windows = generate_walk_forward_windows(overall_start, overall_end, TRAIN_MONTHS, TEST_MONTHS)

    print(f"\n{len(tickers)} usable NEW-ONLY tickers, {len(windows)} windows over "
          f"{overall_start.date()} to {overall_end.date()}\n")

    pooled = {}  # (thresh,hold) -> list of oos sharpes across windows
    for w_idx, w in enumerate(windows):
        for thresh in THRESHOLD_GRID:
            for hold in HOLD_DAYS_GRID:
                oos_trades = build_portfolio_event_trades(tickers, earnings_by_ticker, bars_by_ticker, thresh, hold,
                                                           BASELINE_SLIPPAGE, STARTING_CAPITAL,
                                                           w["test_start"], w["test_end"],
                                                           MAX_GROSS_EXPOSURE, MAX_NEW_POSITIONS_PER_DAY)
                oos_equity = equity_curve_from_trades(oos_trades, STARTING_CAPITAL, bars_by_ticker)
                oos_metrics = compute_metrics(oos_equity, oos_trades, STARTING_CAPITAL)
                pooled.setdefault((thresh, hold), []).append(
                    (oos_metrics["sharpe"], oos_metrics["trade_count"]))

    print(f"{'thresh':>6} {'hold':>4} | per-window OOS Sharpe" + " " * 8 + "| mean   std   min   n_total")
    rows = []
    for (thresh, hold), vals in pooled.items():
        sharpes = [v[0] for v in vals if v[0] == v[0]]
        n_total = sum(v[1] for v in vals)
        mean_s, std_s, min_s = np.mean(sharpes), np.std(sharpes), np.min(sharpes)
        rows.append((thresh, hold, sharpes, mean_s, std_s, min_s, n_total))
    rows.sort(key=lambda r: r[3], reverse=True)
    for thresh, hold, sharpes, mean_s, std_s, min_s, n_total in rows:
        sharpe_str = ", ".join(f"{s:6.3f}" for s in sharpes)
        print(f"{thresh:>5.1f}% {hold:>4} | {sharpe_str} | {mean_s:5.3f} {std_s:5.3f} {min_s:6.3f} {n_total:6d}")


if __name__ == "__main__":
    main()
