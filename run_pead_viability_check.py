"""
General-purpose PEAD viability check: dynamically discovers ALL tickers
currently cached in earnings_cache/ (no hardcoded list), runs the same
walk-forward methodology as run_pead_capped.py (capped-only engine --
exposure cap 1.5x, max 3 new positions/day, priority by surprise
magnitude, NO regime filter -- the regime-filter path was tested and
rejected, see exploration_log_pead.md), and reports the same 3 viability
criteria (Sharpe OOS >= 0.5, MC_DD95 <= 20%, edge survives 0.15%
slippage). Designed to run immediately once the full drip-feed batch is
collected -- no changes needed, just re-run.
"""
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import config
from bot.data_client import DataClient
from bot.earnings_data import AlphaVantageEarningsClient
from bot.pead_backtest import build_portfolio_event_trades, equity_curve_from_trades
from bot.backtest_engine import compute_metrics, monte_carlo_bootstrap, generate_walk_forward_windows, is_oos_deviation_ok

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pead_viability")

LOG_PATH = "exploration_log_pead.md"
CACHE_DIR = Path("earnings_cache")
STARTING_CAPITAL = 100_000.0
BASELINE_SLIPPAGE = 0.0010
THRESHOLD_GRID = [3.0, 5.0, 7.0, 10.0]
HOLD_DAYS_GRID = [20, 40, 60]
MAX_GROSS_EXPOSURE = 1.5
MAX_NEW_POSITIONS_PER_DAY = 3
SLIPPAGE_SCENARIOS = [0.0005, 0.0010, 0.0015]
MIN_EVENTS_PER_TICKER = 3  # exclude a ticker if it has too few qualifying-window events to be meaningful


def log_md(text: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    logger.info(text.split("\n")[0][:200])


def discover_tickers() -> list:
    return sorted(p.stem.replace("_earnings", "") for p in CACHE_DIR.glob("*_earnings.json"))


def run_viability_check(tickers=None, label="capped-only, all cached tickers", train_months=24, test_months=6,
                        quiet=False):
    tickers = tickers or discover_tickers()
    dc = DataClient(config.ALPACA_API_KEY, config.ALPACA_API_SECRET, config.ALPACA_DATA_FEED)
    av = AlphaVantageEarningsClient(config.ALPHA_VANTAGE_API_KEY)
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=365 * 6)

    bars_by_ticker, earnings_by_ticker = {}, {}
    for t in tickers:
        bars_by_ticker[t] = dc.get_bars(t, "us_equity", "1Day", start, end)
        earnings_by_ticker[t] = av.get_quarterly_surprises(t)  # cached, no new API calls

    usable_tickers = [t for t in tickers if not bars_by_ticker[t].empty and not earnings_by_ticker[t].empty]
    if len(usable_tickers) < len(tickers):
        skipped = set(tickers) - set(usable_tickers)
        logger.warning("Skipping %d tickers with missing price or earnings data: %s", len(skipped), skipped)
    tickers = usable_tickers

    price_start = max(b.index.min() for b in bars_by_ticker.values() if not b.empty)
    price_end = min(b.index.max() for b in bars_by_ticker.values() if not b.empty)
    event_start = min(e.index.min() for e in earnings_by_ticker.values() if not e.empty)
    event_end = max(e.index.max() for e in earnings_by_ticker.values() if not e.empty)
    overall_start = max(price_start, event_start)
    overall_end = min(price_end, event_end)
    windows = generate_walk_forward_windows(overall_start, overall_end, train_months=train_months,
                                            test_months=test_months)

    _log = (lambda t: None) if quiet else log_md
    _log(f"\n## Viability check: {label} ({len(tickers)} tickers)\n")
    _log(f"- Tickers: {tickers}")
    _log(f"- Same methodology throughout: exposure cap {MAX_GROSS_EXPOSURE}x, max "
        f"{MAX_NEW_POSITIONS_PER_DAY} new positions/day, priority by surprise magnitude, NO regime "
        f"filter (see 'DECISION: regime-filter path rejected' section -- not being carried forward). "
        f"Grid {THRESHOLD_GRID} x {HOLD_DAYS_GRID}, walk-forward {train_months}mo train / {test_months}mo test, "
        f"{len(windows)} windows over {overall_start.date()} to {overall_end.date()}.")

    window_results = []
    oos_trades_all = []

    for w_idx, w in enumerate(windows):
        best_params, best_is_sharpe = None, -np.inf
        for thresh in THRESHOLD_GRID:
            for hold in HOLD_DAYS_GRID:
                is_trades = build_portfolio_event_trades(tickers, earnings_by_ticker, bars_by_ticker, thresh, hold,
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
            _log(f"- Window {w_idx}: no param combo produced >=5 IS trades, skipped.")
            continue

        thresh, hold = best_params
        oos_trades = build_portfolio_event_trades(tickers, earnings_by_ticker, bars_by_ticker, thresh, hold,
                                                   BASELINE_SLIPPAGE, STARTING_CAPITAL,
                                                   w["test_start"], w["test_end"],
                                                   MAX_GROSS_EXPOSURE, MAX_NEW_POSITIONS_PER_DAY)
        oos_equity = equity_curve_from_trades(oos_trades, STARTING_CAPITAL, bars_by_ticker)
        oos_metrics = compute_metrics(oos_equity, oos_trades, STARTING_CAPITAL)
        deviation_ok = is_oos_deviation_ok(best_is_sharpe, oos_metrics["sharpe"], 0.30)

        _log(f"- Window {w_idx}: params thresh={thresh}% hold={hold}d, IS Sharpe={best_is_sharpe:.3f}, "
              f"OOS Sharpe={oos_metrics['sharpe']:.3f}, OOS trades={oos_metrics['trade_count']}, "
              f"{'OK' if deviation_ok else 'FLAG (>30% IS/OOS deviation)'}")

        window_results.append({"window": w_idx, "params": best_params, "deviation_ok": deviation_ok})
        oos_trades_all.extend(oos_trades)

    if not oos_trades_all:
        _log("- No OOS trades produced across any window -- cannot compute viability metrics.")
        return None

    stitched = equity_curve_from_trades(oos_trades_all, STARTING_CAPITAL, bars_by_ticker)
    overall_metrics = compute_metrics(stitched, oos_trades_all, STARTING_CAPITAL)
    mc = monte_carlo_bootstrap(oos_trades_all, STARTING_CAPITAL, n_resamples=1000, seed=42)
    n_flagged = sum(1 for wr in window_results if not wr["deviation_ok"])

    _log(f"\n### Combined out-of-sample result ({label})\n")
    _log(f"- Sharpe={overall_metrics['sharpe']:.3f}  Sortino={overall_metrics['sortino']:.3f}  "
          f"MaxDD={overall_metrics['max_drawdown']*100:.1f}%  MC_DD95={mc['max_dd_p95']*100:.1f}%  "
          f"PF={overall_metrics['profit_factor']:.3f}  WinRate={overall_metrics['win_rate']*100:.1f}%  "
          f"Trades={overall_metrics['trade_count']}  ({n_flagged}/{len(window_results)} windows flagged)")

    cost_returns = {}
    for slip in SLIPPAGE_SCENARIOS:
        all_slip_trades = []
        for wr in window_results:
            thresh, hold = wr["params"]
            w = windows[wr["window"]]
            all_slip_trades += build_portfolio_event_trades(tickers, earnings_by_ticker, bars_by_ticker, thresh, hold,
                                                             slip, STARTING_CAPITAL, w["test_start"], w["test_end"],
                                                             MAX_GROSS_EXPOSURE, MAX_NEW_POSITIONS_PER_DAY)
        s = equity_curve_from_trades(all_slip_trades, STARTING_CAPITAL, bars_by_ticker)
        cost_returns[slip] = float(s.iloc[-1] / STARTING_CAPITAL - 1) if not s.empty else float("nan")
    _log(f"- Cost sensitivity: " + ", ".join(f"{k*100:.2f}%->{v*100:.1f}%" for k, v in cost_returns.items()))

    sharpe, dd95, cost_15 = overall_metrics["sharpe"], mc["max_dd_p95"], cost_returns[0.0015]
    viable = (sharpe == sharpe and sharpe >= 0.5 and dd95 == dd95 and dd95 <= 0.20 and
             cost_15 == cost_15 and cost_15 > 0)
    _log(f"\n**Viability check: Sharpe>=0.5: {sharpe>=0.5 if sharpe==sharpe else 'N/A'} | "
          f"MC_DD95<=20%: {dd95<=0.20 if dd95==dd95 else 'N/A'} | edge survives 0.15% slippage: "
          f"{cost_15>0 if cost_15==cost_15 else 'N/A'} | {n_flagged}/{len(window_results)} windows flagged "
          f"for IS/OOS deviation >30%**")
    _log(f"\n**VERDICT ({label}): {'PASSES all 3 viability criteria' if viable else 'DOES NOT pass all 3 viability criteria'}**")

    return {"overall_metrics": overall_metrics, "mc": mc, "cost_returns": cost_returns,
           "window_results": window_results, "viable": viable, "tickers": tickers}


if __name__ == "__main__":
    run_viability_check()
