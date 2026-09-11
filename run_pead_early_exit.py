"""
Final PEAD iteration (per explicit decision: last one -- if this doesn't
CLEARLY pass all 3 viability criteria, not borderline, PEAD is retired):
mid-hold early exit instead of entry-only gating. The entry-day regime
filter was rejected earlier (exploration_log_pead.md: "DECISION: regime-filter
path rejected") because it only looked at entry-day state and the trade-level
forensics showed it wasn't causal. This tests the log's own suggested
alternative: "a more principled version would need to account for the regime
trajectory across the WHOLE hold period ... or use a smoother, less binary
signal (e.g. SPY vs its own SMA200, which changes slowly)".

Mechanism: thresh=10%/hold=20d frozen (established as the only combo that
was OOS-stable across every window on both the full-97 and independent
74-ticker universes -- see diagnose_pead_param_instability*.py). Added: if
SPY's close drops below its own 200-day SMA on any day during a position's
hold, exit that day instead of waiting for hold_days. No parameters tuned
here (200d SMA is a standard convention, not fit on this data).

Evaluated as ONE continuous simulation over the full available history
(2021-08-03 to 2026-07-01, including the 2021-2022 correction that the
project's usual 5-window walk-forward slices don't cover) -- baseline
(no early exit) vs early-exit, on both universes, side by side.
"""
import logging
from datetime import datetime, timedelta, timezone

import config
from bot.data_client import DataClient
from bot.earnings_data import AlphaVantageEarningsClient
from bot.pead_backtest import build_portfolio_event_trades, equity_curve_from_trades
from bot.backtest_engine import compute_metrics, monte_carlo_bootstrap, classify_regime_periods, regime_breakdown
from run_pead_viability_check import discover_tickers, MAX_GROSS_EXPOSURE, MAX_NEW_POSITIONS_PER_DAY
from run_pead_sensitivity import ORIGINAL_23

logging.basicConfig(level=logging.WARNING)

STARTING_CAPITAL = 100_000.0
BASELINE_SLIPPAGE = 0.0010
SLIPPAGE_SCENARIOS = [0.0005, 0.0010, 0.0015]
FROZEN_THRESH, FROZEN_HOLD = 10.0, 20


def verdict_line(sharpe, dd95, cost_15):
    checks = [
        ("Sharpe>=0.5", sharpe, sharpe >= 0.5 if sharpe == sharpe else False, f"{sharpe:.3f}", "margin=%.3f" % (sharpe - 0.5) if sharpe == sharpe else "N/A"),
        ("MC_DD95<=20%", dd95, dd95 <= 0.20 if dd95 == dd95 else False, f"{dd95*100:.1f}%", "margin=%.1fpp" % ((0.20 - dd95) * 100) if dd95 == dd95 else "N/A"),
        ("cost-15bps survival>0", cost_15, cost_15 > 0 if cost_15 == cost_15 else False, f"{cost_15*100:.1f}%", "N/A"),
    ]
    all_pass = all(c[2] for c in checks)
    for name, _, ok, val, margin in checks:
        print(f"    {name}: {'PASS' if ok else 'FAIL'} (value={val}, {margin})")
    return all_pass


def run(tickers, label, early_exit_by_date):
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

    results = {}
    for variant_name, eeb in [("baseline (no early exit)", None), ("with mid-hold early exit", early_exit_by_date)]:
        trades = build_portfolio_event_trades(tickers, earnings_by_ticker, bars_by_ticker, FROZEN_THRESH, FROZEN_HOLD,
                                               BASELINE_SLIPPAGE, STARTING_CAPITAL, overall_start, overall_end,
                                               MAX_GROSS_EXPOSURE, MAX_NEW_POSITIONS_PER_DAY,
                                               early_exit_by_date=eeb)
        equity = equity_curve_from_trades(trades, STARTING_CAPITAL, bars_by_ticker)
        metrics = compute_metrics(equity, trades, STARTING_CAPITAL)
        mc = monte_carlo_bootstrap(trades, STARTING_CAPITAL, n_resamples=1000, seed=42)

        cost_returns = {}
        for slip in SLIPPAGE_SCENARIOS:
            slip_trades = build_portfolio_event_trades(tickers, earnings_by_ticker, bars_by_ticker, FROZEN_THRESH,
                                                         FROZEN_HOLD, slip, STARTING_CAPITAL, overall_start, overall_end,
                                                         MAX_GROSS_EXPOSURE, MAX_NEW_POSITIONS_PER_DAY,
                                                         early_exit_by_date=eeb)
            s = equity_curve_from_trades(slip_trades, STARTING_CAPITAL, bars_by_ticker)
            cost_returns[slip] = float(s.iloc[-1] / STARTING_CAPITAL - 1) if not s.empty else float("nan")

        print(f"\n=== {label} -- {variant_name} ({len(tickers)} tickers, "
              f"{overall_start.date()}..{overall_end.date()}) ===")
        print(f"  Sharpe={metrics['sharpe']:.3f}  Sortino={metrics['sortino']:.3f}  "
              f"MaxDD={metrics['max_drawdown']*100:.1f}%  MC_DD95={mc['max_dd_p95']*100:.1f}%  "
              f"PF={metrics['profit_factor']:.3f}  WinRate={metrics['win_rate']*100:.1f}%  Trades={metrics['trade_count']}")
        print("  Cost sensitivity: " + ", ".join(f"{k*100:.2f}%->{v*100:.1f}%" for k, v in cost_returns.items()))
        all_pass = verdict_line(metrics["sharpe"], mc["max_dd_p95"], cost_returns[0.0015])
        print(f"  VERDICT: {'PASSES all 3' if all_pass else 'FAILS at least 1'}")

        results[variant_name] = {"trades": trades, "equity": equity, "metrics": metrics, "mc": mc,
                                  "cost_returns": cost_returns, "viable": all_pass}
    return results


def print_regime_breakdown(label, trades, equity, regime_labels):
    bd = regime_breakdown(equity, trades, regime_labels)
    print(f"  Regime breakdown ({label}):")
    for regime, stats in bd.items():
        print(f"    {regime:<11} days={stats['days']:>4}  return={stats['total_return']*100 if stats['total_return']==stats['total_return'] else float('nan'):>7.1f}%  "
              f"sharpe={stats['sharpe']:>7.2f}  trades={stats['trade_count']}")


if __name__ == "__main__":
    dc = DataClient(config.ALPACA_API_KEY, config.ALPACA_API_SECRET, config.ALPACA_DATA_FEED)
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=365 * 7)  # extra year of buffer so SMA200 is defined at overall_start
    spy_bars = dc.get_bars("SPY", "us_equity", "1Day", start, end)
    sma200 = spy_bars["close"].rolling(200).mean()
    below_sma200 = spy_bars["close"] < sma200
    early_exit_by_date = {ts.date(): bool(v) for ts, v in below_sma200.items() if v == v}
    n_flagged_days = sum(early_exit_by_date.values())
    print(f"SPY SMA200 early-exit signal: {n_flagged_days}/{len(early_exit_by_date)} days flagged "
          f"({n_flagged_days/len(early_exit_by_date)*100:.1f}%) over {spy_bars.index.min().date()}..{spy_bars.index.max().date()}")

    regime_labels = classify_regime_periods(spy_bars)

    all_tickers = discover_tickers()
    universes = [(all_tickers, "full-97 universe"),
                 ([t for t in all_tickers if t not in ORIGINAL_23], "new-only-74 universe (independent)")]

    for tickers, label in universes:
        res = run(tickers, label, early_exit_by_date)
        for variant_name, r in res.items():
            print_regime_breakdown(f"{label} / {variant_name}", r["trades"], r["equity"], regime_labels)
