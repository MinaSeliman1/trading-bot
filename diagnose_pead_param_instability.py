"""
Diagnostic: WHY do the optimal (threshold, hold_days) params never converge
across PEAD walk-forward windows? run_pead_viability_check.py only ever logs
the argmax-IS-Sharpe combo per window -- this script dumps the FULL 12-cell
grid (all thresholds x hold_days) for every window, with IS/OOS Sharpe and
trade counts for EVERY combo, plus the train/test period's dominant market
regime. That's enough to distinguish two competing explanations:

  (a) small-sample noise: per-cell trade counts are so low that IS Sharpe
      estimates are unstable, and the argmax is picking noise, not signal
      (the top few combos are within noise of each other).
  (b) regime dependency: the TRUE best combo genuinely shifts with the
      market regime during train/test (a structural, not a sampling, cause).

Reuses the exact same engine/grid/windows as run_pead_viability_check.py's
"full universe, 24mo train/6mo test" config -- the headline result the
project's conclusions are anchored on (Sharpe=0.884, 3/5 windows flagged).
"""
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import config
from bot.data_client import DataClient
from bot.earnings_data import AlphaVantageEarningsClient
from bot.pead_backtest import build_portfolio_event_trades, equity_curve_from_trades
from bot.backtest_engine import (
    compute_metrics, generate_walk_forward_windows, is_oos_deviation_ok, classify_regime_periods,
)
from run_pead_viability_check import discover_tickers, MAX_GROSS_EXPOSURE, MAX_NEW_POSITIONS_PER_DAY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pead_diagnose")

STARTING_CAPITAL = 100_000.0
BASELINE_SLIPPAGE = 0.0010
THRESHOLD_GRID = [3.0, 5.0, 7.0, 10.0]
HOLD_DAYS_GRID = [20, 40, 60]
TRAIN_MONTHS, TEST_MONTHS = 24, 6


def regime_share(regime_labels: pd.Series, start, end) -> dict:
    sub = regime_labels[(regime_labels.index >= start) & (regime_labels.index < end)]
    if sub.empty:
        return {}
    counts = sub.value_counts(normalize=True) * 100
    return {k: round(v, 1) for k, v in counts.items()}


def main():
    tickers = discover_tickers()
    dc = DataClient(config.ALPACA_API_KEY, config.ALPACA_API_SECRET, config.ALPACA_DATA_FEED)
    av = AlphaVantageEarningsClient(config.ALPHA_VANTAGE_API_KEY)
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=365 * 6)

    bars_by_ticker, earnings_by_ticker = {}, {}
    for t in tickers:
        bars_by_ticker[t] = dc.get_bars(t, "us_equity", "1Day", start, end)
        earnings_by_ticker[t] = av.get_quarterly_surprises(t)
    tickers = [t for t in tickers if not bars_by_ticker[t].empty and not earnings_by_ticker[t].empty]

    spy_bars = dc.get_bars("SPY", "us_equity", "1Day", start, end)
    regime_labels = classify_regime_periods(spy_bars)

    price_start = max(b.index.min() for b in bars_by_ticker.values() if not b.empty)
    price_end = min(b.index.max() for b in bars_by_ticker.values() if not b.empty)
    event_start = min(e.index.min() for e in earnings_by_ticker.values() if not e.empty)
    event_end = max(e.index.max() for e in earnings_by_ticker.values() if not e.empty)
    overall_start, overall_end = max(price_start, event_start), min(price_end, event_end)
    windows = generate_walk_forward_windows(overall_start, overall_end, TRAIN_MONTHS, TEST_MONTHS)

    print(f"\n{len(tickers)} usable tickers, {len(windows)} windows over {overall_start.date()} to {overall_end.date()}\n")

    all_window_rows = []
    for w_idx, w in enumerate(windows):
        train_regime = regime_share(regime_labels, w["train_start"], w["train_end"])
        test_regime = regime_share(regime_labels, w["test_start"], w["test_end"])
        print(f"=== Window {w_idx}: train {w['train_start'].date()}..{w['train_end'].date()}  "
              f"test {w['test_start'].date()}..{w['test_end'].date()} ===")
        print(f"  train regime mix: {train_regime}")
        print(f"  test  regime mix: {test_regime}")

        rows = []
        for thresh in THRESHOLD_GRID:
            for hold in HOLD_DAYS_GRID:
                is_trades = build_portfolio_event_trades(tickers, earnings_by_ticker, bars_by_ticker, thresh, hold,
                                                          BASELINE_SLIPPAGE, STARTING_CAPITAL,
                                                          w["train_start"], w["train_end"],
                                                          MAX_GROSS_EXPOSURE, MAX_NEW_POSITIONS_PER_DAY)
                is_equity = equity_curve_from_trades(is_trades, STARTING_CAPITAL, bars_by_ticker)
                is_metrics = compute_metrics(is_equity, is_trades, STARTING_CAPITAL)

                oos_trades = build_portfolio_event_trades(tickers, earnings_by_ticker, bars_by_ticker, thresh, hold,
                                                           BASELINE_SLIPPAGE, STARTING_CAPITAL,
                                                           w["test_start"], w["test_end"],
                                                           MAX_GROSS_EXPOSURE, MAX_NEW_POSITIONS_PER_DAY)
                oos_equity = equity_curve_from_trades(oos_trades, STARTING_CAPITAL, bars_by_ticker)
                oos_metrics = compute_metrics(oos_equity, oos_trades, STARTING_CAPITAL)

                rows.append({
                    "thresh": thresh, "hold": hold,
                    "is_sharpe": is_metrics["sharpe"], "is_n": is_metrics["trade_count"],
                    "oos_sharpe": oos_metrics["sharpe"], "oos_n": oos_metrics["trade_count"],
                })

        rows.sort(key=lambda r: (r["is_sharpe"] if r["is_sharpe"] == r["is_sharpe"] else -np.inf), reverse=True)
        print(f"  {'thresh':>6} {'hold':>4} | {'IS Sharpe':>9} {'IS n':>5} | {'OOS Sharpe':>10} {'OOS n':>5}  IS-rank")
        for rank, r in enumerate(rows):
            marker = "  <- IS argmax (this is what the viability check picks)" if rank == 0 else ""
            print(f"  {r['thresh']:>5.1f}% {r['hold']:>4} | {r['is_sharpe']:>9.3f} {r['is_n']:>5} | "
                  f"{r['oos_sharpe']:>10.3f} {r['oos_n']:>5}  #{rank+1}{marker}")

        is_sharpes = [r["is_sharpe"] for r in rows if r["is_sharpe"] == r["is_sharpe"]]
        oos_by_is_rank = [r["oos_sharpe"] for r in rows]
        best_oos_rank = int(np.argmax([s if s == s else -np.inf for s in oos_by_is_rank])) + 1
        spread = (is_sharpes[0] - is_sharpes[min(2, len(is_sharpes) - 1)]) if len(is_sharpes) >= 2 else float("nan")
        print(f"  -> IS-argmax rank among OOS outcomes: #{best_oos_rank} of {len(rows)} "
              f"(1 = IS pick was also the best OOS choice)")
        print(f"  -> IS Sharpe gap, rank1 vs rank3: {spread:.3f}  |  min IS n across grid: "
              f"{min(r['is_n'] for r in rows)}  max: {max(r['is_n'] for r in rows)}\n")

        all_window_rows.append({"window": w_idx, "rows": rows, "train_regime": train_regime,
                                 "test_regime": test_regime, "best_oos_rank": best_oos_rank})

    # ------------------------------------------------------------------ #
    # synthesis
    # ------------------------------------------------------------------ #
    print("=== Synthesis across all windows ===")
    ranks = [w["best_oos_rank"] for w in all_window_rows]
    print(f"IS-argmax's rank among the OOS-sorted grid, per window: {ranks}")
    print(f"(if IS selection carried real information, this should cluster near #1; "
          f"random selection over 12 cells would average ~6.5)")

    all_is_ns = [r["is_n"] for w in all_window_rows for r in w["rows"]]
    print(f"\nIS trade count per grid cell across all windows/combos: "
          f"min={min(all_is_ns)} median={int(np.median(all_is_ns))} max={max(all_is_ns)}")

    print("\nDominant train regime vs the window's IS-argmax (thresh, hold):")
    for w in all_window_rows:
        top = max(w["train_regime"], key=w["train_regime"].get) if w["train_regime"] else "N/A"
        best = w["rows"][0]
        print(f"  Window {w['window']}: train regime dominant={top} {w['train_regime']} "
              f"-> IS picked thresh={best['thresh']}% hold={best['hold']}d")


if __name__ == "__main__":
    main()
