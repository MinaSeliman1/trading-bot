import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import config
from bot.data_client import DataClient
from bot.earnings_data import AlphaVantageEarningsClient
from bot.pead_backtest import build_event_trades, equity_curve_from_trades, daily_returns_from_equity
from bot.backtest_engine import (
    compute_metrics, monte_carlo_bootstrap, generate_walk_forward_windows, is_oos_deviation_ok,
)
from bot.compare_strategies import SLIPPAGE_SCENARIOS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pead_preliminary")

LOG_PATH = "exploration_log_pead.md"
STARTING_CAPITAL = 100_000.0
BASELINE_SLIPPAGE = 0.0010
TICKERS = ["JPM", "BAC", "GS", "JNJ", "UNH", "PFE", "ABBV", "PG", "KO", "WMT",
           "MCD", "CAT", "HON", "UPS", "XOM", "CVX", "VZ", "DIS", "HD", "COST",
           "LIN", "DUK", "V"]
THRESHOLD_GRID = [3.0, 5.0, 7.0, 10.0]
HOLD_DAYS_GRID = [20, 40, 60]


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

    total_events = sum(len(e[(e.index >= overall_start) & (e.index <= overall_end)]) for e in earnings_by_ticker.values())
    log_md(f"\n## Preliminary PEAD backtest: {len(TICKERS)} tickers, tradeable window "
          f"{overall_start.date()} to {overall_end.date()}\n")
    log_md(f"- {total_events} total quarterly earnings events across all tickers in the tradeable window "
          f"(price data available for the hold period).")
    log_md(f"- Signal: long-only, raw surprise_pct >= threshold (grid: {THRESHOLD_GRID}), "
          f"hold {HOLD_DAYS_GRID} trading days (grid search per window, walk-forward optimized). "
          f"Fixed $10,000 notional per trade (no compounding/portfolio sizing yet -- preliminary "
          f"edge-existence check only, as requested).")

    windows = generate_walk_forward_windows(overall_start, overall_end, train_months=24, test_months=6)
    log_md(f"- Walk-forward: 24mo train / 6mo test (wider than the usual 12/3 given quarterly event "
          f"frequency -- ensures a meaningful event count per window). {len(windows)} windows generated.")

    window_results = []
    oos_trades_all = []

    for w_idx, w in enumerate(windows):
        best_params, best_is_sharpe = None, -np.inf
        for thresh in THRESHOLD_GRID:
            for hold in HOLD_DAYS_GRID:
                is_trades = pooled_trades(TICKERS, earnings_by_ticker, bars_by_ticker, thresh, hold,
                                          BASELINE_SLIPPAGE, w["train_start"], w["train_end"])
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
        oos_trades = pooled_trades(TICKERS, earnings_by_ticker, bars_by_ticker, thresh, hold, BASELINE_SLIPPAGE,
                                   w["test_start"], w["test_end"])
        oos_equity = equity_curve_from_trades(oos_trades, STARTING_CAPITAL)
        oos_metrics = compute_metrics(oos_equity, oos_trades, STARTING_CAPITAL)
        deviation_ok = is_oos_deviation_ok(best_is_sharpe, oos_metrics["sharpe"], 0.30)

        log_md(f"- Window {w_idx} (train {w['train_start'].date()}-{w['train_end'].date()}, "
              f"test {w['test_start'].date()}-{w['test_end'].date()}): best params thresh={thresh}% hold={hold}d, "
              f"IS Sharpe={best_is_sharpe:.3f}, OOS Sharpe={oos_metrics['sharpe']:.3f}, "
              f"OOS trades={oos_metrics['trade_count']}, {'OK' if deviation_ok else 'FLAG (>30% IS/OOS deviation)'}")

        window_results.append({"window": w_idx, "params": best_params, "is_sharpe": best_is_sharpe,
                               "oos_metrics": oos_metrics, "deviation_ok": deviation_ok,
                               "test_start": w["test_start"], "test_end": w["test_end"]})
        oos_trades_all.extend(oos_trades)

    # NOTE: unlike the bar-based backtest, PEAD hold periods (up to 60 trading
    # days) routinely extend past a window's test_end boundary, so consecutive
    # windows' OWN equity curves overlap in calendar time -- stitching them
    # with _stitch_equity_curves (built for strictly non-overlapping windows)
    # produces duplicate index entries and crashes downstream metrics. The
    # underlying TRADES are already correctly non-duplicated across windows
    # (event filtering uses non-overlapping [test_start, test_end) date
    # ranges), so build one equity curve from the complete combined trade
    # list instead of stitching per-window curves.
    stitched = equity_curve_from_trades(oos_trades_all, STARTING_CAPITAL)
    overall_metrics = compute_metrics(stitched, oos_trades_all, STARTING_CAPITAL)
    mc = monte_carlo_bootstrap(oos_trades_all, STARTING_CAPITAL, n_resamples=1000, seed=42)

    log_md(f"\n### Combined out-of-sample result across all {len(windows)} windows\n")
    log_md(f"- Sharpe={overall_metrics['sharpe']:.3f}  Sortino={overall_metrics['sortino']:.3f}  "
          f"MaxDD={overall_metrics['max_drawdown']*100:.1f}%  MC_DD95={mc['max_dd_p95']*100:.1f}%  "
          f"PF={overall_metrics['profit_factor']:.3f}  WinRate={overall_metrics['win_rate']*100:.1f}%  "
          f"Trades={overall_metrics['trade_count']}")

    # cost sensitivity: re-run OOS at each slippage using each window's already-chosen params (no re-optimization)
    cost_returns = {}
    for slip in SLIPPAGE_SCENARIOS:
        all_slip_trades = []
        for wr in window_results:
            thresh, hold = wr["params"]
            all_slip_trades += pooled_trades(TICKERS, earnings_by_ticker, bars_by_ticker, thresh, hold, slip,
                                              wr["test_start"], wr["test_end"])
        s = equity_curve_from_trades(all_slip_trades, STARTING_CAPITAL)
        cost_returns[slip] = float(s.iloc[-1] / STARTING_CAPITAL - 1) if not s.empty else float("nan")
    log_md(f"- Cost sensitivity: " + ", ".join(f"{k*100:.2f}%->{v*100:.1f}%" for k, v in cost_returns.items()))

    n_flagged = sum(1 for wr in window_results if not wr["deviation_ok"])
    sharpe = overall_metrics["sharpe"]
    dd95 = mc["max_dd_p95"]
    cost_15 = cost_returns[0.0015]
    viable = (sharpe == sharpe and sharpe >= 0.5 and dd95 == dd95 and dd95 <= 0.20 and
             cost_15 == cost_15 and cost_15 > 0)
    log_md(f"\n**Viability check: Sharpe>=0.5: {sharpe>=0.5 if sharpe==sharpe else 'N/A'} | "
          f"MC_DD95<=20%: {dd95<=0.20 if dd95==dd95 else 'N/A'} | edge survives 0.15% slippage: "
          f"{cost_15>0 if cost_15==cost_15 else 'N/A'} | {n_flagged}/{len(window_results)} windows flagged "
          f"for IS/OOS deviation >30%**")
    log_md(f"\n**PRELIMINARY VERDICT: {'PASSES all 3 viability criteria' if viable else 'DOES NOT pass all 3 viability criteria'}**")

    return window_results, overall_metrics, mc, cost_returns


if __name__ == "__main__":
    main()
