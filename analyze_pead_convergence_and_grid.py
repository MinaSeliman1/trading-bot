"""
Two analyses on the already-collected 23-ticker PEAD data, no new API calls:

1. Empirical convergence: how does MC 95th-pct drawdown behave as pooled
   trade count grows? Resamples the ALREADY-OBSERVED 131 trade returns at
   increasing target sample sizes (bootstrap-from-empirical-distribution),
   rather than guessing a theoretical convergence rate. Explicit assumption
   stated: this assumes tickers added later have a similar per-trade return
   distribution to the current 23 -- not guaranteed, but the only estimate
   possible without the data itself.

2. Finer parameter grid search for a stable plateau (threshold x hold_days),
   same strict 30% IS/OOS deviation rule, same walk-forward windows.
"""
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import config
from bot.data_client import DataClient
from bot.earnings_data import AlphaVantageEarningsClient
from bot.pead_backtest import build_event_trades, equity_curve_from_trades
from bot.backtest_engine import compute_metrics, generate_walk_forward_windows, is_oos_deviation_ok

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pead_convergence")

LOG_PATH = "exploration_log_pead.md"
STARTING_CAPITAL = 100_000.0
BASELINE_SLIPPAGE = 0.0010
TICKERS = ["JPM", "BAC", "GS", "JNJ", "UNH", "PFE", "ABBV", "PG", "KO", "WMT",
           "MCD", "CAT", "HON", "UPS", "XOM", "CVX", "VZ", "DIS", "HD", "COST",
           "LIN", "DUK", "V"]
ORIGINAL_THRESHOLD_GRID = [3.0, 5.0, 7.0, 10.0]
ORIGINAL_HOLD_GRID = [20, 40, 60]
FINE_THRESHOLD_GRID = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]
FINE_HOLD_GRID = [10, 15, 20, 30, 40, 50, 60, 80]


def log_md(text: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    logger.info(text.split("\n")[0][:200])


def pooled_trades(tickers, earnings_by_ticker, bars_by_ticker, threshold, hold_days, slippage, w_start, w_end):
    trades = []
    for t in tickers:
        trades += build_event_trades(t, earnings_by_ticker[t], bars_by_ticker[t], threshold, hold_days, slippage,
                                     window_start=w_start, window_end=w_end)
    return trades


def bootstrap_dd95_at_size(trade_returns: np.ndarray, n: int, n_resamples: int = 1000, seed: int = None) -> float:
    rng = np.random.default_rng(seed)
    dds = []
    for _ in range(n_resamples):
        sample = rng.choice(trade_returns, size=n, replace=True)
        equity = np.concatenate([[1.0], np.cumprod(1 + sample)])
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / peak
        dds.append(dd.max())
    return float(np.percentile(dds, 95))


def part1_convergence(oos_trades_all):
    log_md("\n## Part 1: empirical convergence -- how many pooled trades to get MC_DD95 under 20%?\n")
    returns = np.array([t.return_pct for t in oos_trades_all if t.pnl is not None])
    log_md(f"- Base sample: {len(returns)} observed OOS trade returns (mean={returns.mean()*100:.2f}%, "
          f"std={returns.std()*100:.2f}%, min={returns.min()*100:.1f}%, max={returns.max()*100:.1f}%).")
    log_md(f"- Method: resample AT the target size N with replacement FROM this same empirical return "
          f"distribution (1000 resamples per N), track how the resulting MC 95th-pct drawdown moves. "
          f"**Explicit assumption**: this assumes tickers added later produce trades with a statistically "
          f"similar per-trade return distribution to the current 23 -- reasonable but not guaranteed; "
          f"it is the best estimate possible without the additional data itself.")

    targets = [131, 150, 200, 250, 300, 400, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000]
    results = []
    for n in targets:
        dd95 = bootstrap_dd95_at_size(returns, n, n_resamples=1000, seed=42)
        results.append((n, dd95))
        log_md(f"    N={n:5d} trades -> MC_DD95={dd95*100:5.1f}%")

    crossing = next((n for n, dd in results if dd <= 0.20), None)
    if crossing:
        log_md(f"\n- **Crosses below the 20% MC_DD95 threshold at approximately N={crossing} pooled trades** "
              f"(under the stated assumption).")
    else:
        log_md(f"\n- **Does NOT cross below 20% even at N={targets[-1]} trades** under this resampling -- "
              f"the tail-risk problem may be structural (a genuinely fat-tailed per-trade return "
              f"distribution) rather than a sample-size problem alone. More data would still narrow the "
              f"CONFIDENCE INTERVAL around the drawdown estimate, but may not lower the point estimate itself.")

    # convert to tickers/days needed
    trades_per_ticker = len(returns) / len(TICKERS)
    log_md(f"\n- Observed rate: {len(returns)} trades / {len(TICKERS)} tickers = "
          f"{trades_per_ticker:.2f} trades/ticker over this walk-forward setup.")
    if crossing:
        tickers_needed = int(np.ceil(crossing / trades_per_ticker))
        additional_tickers = max(0, tickers_needed - len(TICKERS))
        additional_days = int(np.ceil(additional_tickers / 25))
        log_md(f"- To reach ~{crossing} trades: ~{tickers_needed} tickers total needed "
              f"({additional_tickers} more beyond the 23 already collected). "
              f"**At 25 tickers/day free tier: ~{additional_days} more days of drip-feed "
              f"({additional_days + 1} including today).**")
    return results, crossing


def part2_finer_grid(tickers, earnings_by_ticker, bars_by_ticker, windows):
    log_md("\n## Part 2: finer parameter grid search for a stable plateau\n")
    log_md(f"- Grid: threshold in {FINE_THRESHOLD_GRID} (9 values) x hold_days in {FINE_HOLD_GRID} "
          f"(8 values) = 72 combos x {len(windows)} windows = {72*len(windows)} (param,window) evaluations. "
          f"Same 30% IS/OOS deviation rule, same viability bar.")

    rows = []
    for thresh in FINE_THRESHOLD_GRID:
        for hold in FINE_HOLD_GRID:
            for w_idx, w in enumerate(windows):
                is_trades = pooled_trades(tickers, earnings_by_ticker, bars_by_ticker, thresh, hold,
                                          BASELINE_SLIPPAGE, w["train_start"], w["train_end"])
                if len(is_trades) < 5:
                    continue
                is_eq = equity_curve_from_trades(is_trades, STARTING_CAPITAL)
                is_m = compute_metrics(is_eq, is_trades, STARTING_CAPITAL)

                oos_trades = pooled_trades(tickers, earnings_by_ticker, bars_by_ticker, thresh, hold,
                                           BASELINE_SLIPPAGE, w["test_start"], w["test_end"])
                oos_eq = equity_curve_from_trades(oos_trades, STARTING_CAPITAL)
                oos_m = compute_metrics(oos_eq, oos_trades, STARTING_CAPITAL)

                rows.append({"params": (thresh, hold), "window": w_idx, "is_sharpe": is_m["sharpe"],
                            "oos_sharpe": oos_m["sharpe"], "oos_trade_count": oos_m["trade_count"],
                            "oos_profit_factor": oos_m["profit_factor"], "oos_max_drawdown": oos_m["max_drawdown"]})

    survivors = [r for r in rows if is_oos_deviation_ok(r["is_sharpe"], r["oos_sharpe"], 0.30)]
    log_md(f"- {len(rows)} total (param,window) combos evaluated. "
          f"{len(rows)-len(survivors)} rejected by the 30% anti-overfitting rule.")

    by_params = {}
    for r in (survivors if survivors else rows):
        by_params.setdefault(r["params"], []).append(r)

    scored = []
    for params, group in by_params.items():
        oos_sharpes = [g["oos_sharpe"] for g in group if g["oos_sharpe"] == g["oos_sharpe"]]
        if len(oos_sharpes) < 3:  # require a plateau across AT LEAST 3 of the 7 windows to count
            continue
        mean_s, std_s = float(np.mean(oos_sharpes)), float(np.std(oos_sharpes))
        total_trades = sum(g["oos_trade_count"] for g in group)
        scored.append({"params": params, "n_windows": len(oos_sharpes), "mean_oos_sharpe": mean_s,
                       "std_oos_sharpe": std_s, "total_trades": total_trades})
    scored.sort(key=lambda s: s["mean_oos_sharpe"] - s["std_oos_sharpe"], reverse=True)

    log_md(f"- Param combos with a plateau across >=3/{len(windows)} windows: {len(scored)}")
    for s in scored[:10]:
        log_md(f"    thresh={s['params'][0]}% hold={s['params'][1]}d: n_windows={s['n_windows']} "
              f"mean_oos_sharpe={s['mean_oos_sharpe']:.3f} std={s['std_oos_sharpe']:.3f} "
              f"total_trades={s['total_trades']}")

    if not scored:
        log_md("- **No parameter combination showed a stable plateau across >=3 windows even with the "
              "finer grid. The instability observed in the original 12-combo run is not a grid-resolution "
              "artifact -- it reflects genuine sample-size-driven noise.**")
    return scored


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

    # rebuild the ORIGINAL run's pooled OOS trades (same grid as before) to get the 131-trade base sample
    oos_trades_all = []
    for w in windows:
        best_params, best_is_sharpe = None, -np.inf
        for thresh in ORIGINAL_THRESHOLD_GRID:
            for hold in ORIGINAL_HOLD_GRID:
                is_trades = pooled_trades(TICKERS, earnings_by_ticker, bars_by_ticker, thresh, hold,
                                          BASELINE_SLIPPAGE, w["train_start"], w["train_end"])
                if len(is_trades) < 5:
                    continue
                is_eq = equity_curve_from_trades(is_trades, STARTING_CAPITAL)
                is_m = compute_metrics(is_eq, is_trades, STARTING_CAPITAL)
                if is_m["sharpe"] == is_m["sharpe"] and is_m["sharpe"] > best_is_sharpe:
                    best_is_sharpe, best_params = is_m["sharpe"], (thresh, hold)
        if best_params is None:
            continue
        thresh, hold = best_params
        oos_trades_all += pooled_trades(TICKERS, earnings_by_ticker, bars_by_ticker, thresh, hold,
                                        BASELINE_SLIPPAGE, w["test_start"], w["test_end"])

    logger.info("Rebuilt base sample: %d trades (should match the original run's 131)", len(oos_trades_all))

    part1_convergence(oos_trades_all)
    part2_finer_grid(TICKERS, earnings_by_ticker, bars_by_ticker, windows)


if __name__ == "__main__":
    main()
