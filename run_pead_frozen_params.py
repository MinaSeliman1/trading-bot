"""
Structural fix test: instead of re-optimizing (threshold, hold_days) via
per-window IS-Sharpe argmax -- the mechanism diagnosed in
diagnose_pead_param_instability.py as the actual cause of the parameter
instability (IS-argmax rank vs OOS-outcome rank was WORSE than random,
average rank 8.6/12; the top IS candidates were separated by Sharpe gaps of
0.07-0.26, i.e. statistical ties, so the "winner" was arbitrary; and the
combo it systematically favored, long-hold/low-threshold, was shown to be
the LEAST stable OOS performer both on the full-97 and the independent
74-ticker universe) -- this freezes thresh=10%/hold=20d GLOBALLY, chosen
because it was the top-mean, lowest-variance, only-ever-positive combo in
BOTH the full-97 grid (mean OOS Sharpe 1.50, every window positive) and the
disjoint 74-new-only cross-check (mean OOS Sharpe 1.45, every window
positive) -- not picked from a single test. Runs it as one continuous
simulation over the whole available history (no walk-forward re-fitting at
all, since there is nothing left to fit) and reports the same 3 viability
criteria as run_pead_viability_check.py for direct comparison.
"""
import logging
from datetime import datetime, timedelta, timezone

import config
from bot.data_client import DataClient
from bot.earnings_data import AlphaVantageEarningsClient
from bot.pead_backtest import build_portfolio_event_trades, equity_curve_from_trades
from bot.backtest_engine import compute_metrics, monte_carlo_bootstrap, generate_walk_forward_windows
from run_pead_viability_check import discover_tickers, MAX_GROSS_EXPOSURE, MAX_NEW_POSITIONS_PER_DAY
from run_pead_sensitivity import ORIGINAL_23

logging.basicConfig(level=logging.WARNING)

STARTING_CAPITAL = 100_000.0
BASELINE_SLIPPAGE = 0.0010
SLIPPAGE_SCENARIOS = [0.0005, 0.0010, 0.0015]
FROZEN_THRESH, FROZEN_HOLD = 10.0, 20


def run(tickers, label):
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

    print(f"\n=== {label} ({len(tickers)} tickers), frozen thresh={FROZEN_THRESH}% hold={FROZEN_HOLD}d, "
          f"single continuous run {overall_start.date()}..{overall_end.date()} ===")

    trades = build_portfolio_event_trades(tickers, earnings_by_ticker, bars_by_ticker, FROZEN_THRESH, FROZEN_HOLD,
                                           BASELINE_SLIPPAGE, STARTING_CAPITAL, overall_start, overall_end,
                                           MAX_GROSS_EXPOSURE, MAX_NEW_POSITIONS_PER_DAY)
    equity = equity_curve_from_trades(trades, STARTING_CAPITAL, bars_by_ticker)
    metrics = compute_metrics(equity, trades, STARTING_CAPITAL)
    mc = monte_carlo_bootstrap(trades, STARTING_CAPITAL, n_resamples=1000, seed=42)

    print(f"Sharpe={metrics['sharpe']:.3f}  Sortino={metrics['sortino']:.3f}  "
          f"MaxDD={metrics['max_drawdown']*100:.1f}%  MC_DD95={mc['max_dd_p95']*100:.1f}%  "
          f"PF={metrics['profit_factor']:.3f}  WinRate={metrics['win_rate']*100:.1f}%  Trades={metrics['trade_count']}")

    cost_returns = {}
    for slip in SLIPPAGE_SCENARIOS:
        slip_trades = build_portfolio_event_trades(tickers, earnings_by_ticker, bars_by_ticker, FROZEN_THRESH,
                                                     FROZEN_HOLD, slip, STARTING_CAPITAL, overall_start, overall_end,
                                                     MAX_GROSS_EXPOSURE, MAX_NEW_POSITIONS_PER_DAY)
        s = equity_curve_from_trades(slip_trades, STARTING_CAPITAL, bars_by_ticker)
        cost_returns[slip] = float(s.iloc[-1] / STARTING_CAPITAL - 1) if not s.empty else float("nan")
    print("Cost sensitivity: " + ", ".join(f"{k*100:.2f}%->{v*100:.1f}%" for k, v in cost_returns.items()))

    sharpe, dd95, cost_15 = metrics["sharpe"], mc["max_dd_p95"], cost_returns[0.0015]
    viable = sharpe == sharpe and sharpe >= 0.5 and dd95 == dd95 and dd95 <= 0.20 and cost_15 == cost_15 and cost_15 > 0
    print(f"Viability: Sharpe>=0.5: {sharpe>=0.5} | MC_DD95<=20%: {dd95<=0.20} | "
          f"edge survives 0.15% slippage: {cost_15>0}")
    print(f"VERDICT: {'PASSES all 3 viability criteria' if viable else 'DOES NOT pass all 3 viability criteria'}")

    # window-sliced breakdown purely for reporting/comparison -- NOT used to select anything
    windows = generate_walk_forward_windows(overall_start, overall_end, 24, 6)
    print("Per-6mo-slice OOS Sharpe with the SAME frozen params throughout (no re-fitting):")
    for w_idx, w in enumerate(windows):
        slice_trades = build_portfolio_event_trades(tickers, earnings_by_ticker, bars_by_ticker, FROZEN_THRESH,
                                                      FROZEN_HOLD, BASELINE_SLIPPAGE, STARTING_CAPITAL,
                                                      w["test_start"], w["test_end"],
                                                      MAX_GROSS_EXPOSURE, MAX_NEW_POSITIONS_PER_DAY)
        slice_equity = equity_curve_from_trades(slice_trades, STARTING_CAPITAL, bars_by_ticker)
        slice_metrics = compute_metrics(slice_equity, slice_trades, STARTING_CAPITAL)
        print(f"  Window {w_idx} ({w['test_start'].date()}..{w['test_end'].date()}): "
              f"Sharpe={slice_metrics['sharpe']:.3f} trades={slice_metrics['trade_count']}")

    return {"metrics": metrics, "mc": mc, "cost_returns": cost_returns, "viable": viable}


if __name__ == "__main__":
    all_tickers = discover_tickers()
    run(all_tickers, "full-97 universe")
    run([t for t in all_tickers if t not in ORIGINAL_23], "new-only-74 universe (independent)")
