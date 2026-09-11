"""
Second-recession test requested: same exact frozen mechanism as
run_pead_early_exit.py (thresh=10%/hold=20d + SPY-SMA200 mid-hold early
exit, NO new parameters), applied to the earliest period where our price
data source (Alpaca) has ANY coverage at all.

2007-2009 is NOT reachable: verified directly against the Alpaca API
(bot/data_client.py's only equity price source) on the SIP feed --
bisected year by year for both JPM and SPY, zero rows for every year
2005-2015, 252 real rows starting exactly 2016-01-04. This is a hard
account/API floor, not a subscription/coverage nuance -- Alpha Vantage
earnings data is NOT the bottleneck (92/97 of our cached tickers have
earnings history well before 2010, most back to 1996).

Separately: the project's default feed (config.ALPACA_DATA_FEED="iex")
has an even later floor for individual stocks -- IEX-sourced historical
bars for names like JPM return ZERO rows before ~2020-07, even though SIP
covers the same symbol back to 2016-01-04 (verified directly). This script
therefore requests the SIP feed explicitly (bypassing the project default,
which would otherwise silently make this test look like "no data" for a
reason that has nothing to do with the real 2016 account floor) and
fetches with use_cache=False so this one-off deep-history pull never
writes into the shared data_cache/ parquet files (keyed by symbol+
timeframe only, not by feed, and relied on by every other script in this
project) -- keeping this investigation fully isolated from the rest of
the pipeline.

The earliest genuinely usable period is therefore 2016-01-04 onward. This
script uses 2016-11-01 (leaving ~200 trading days of runway so SPY's own
200-day SMA is already valid at window start) through 2020-07-26 (the day
before the already-tested 2020-2026 dataset begins, so there is no overlap
between this "second recession" period and the first test). This window
contains TWO independent, differently-flavored stress episodes never used
to build or tune this mechanism: the Q4 2018 correction (~Sept-Dec 2018,
S&P -19.8%) and the COVID crash (Feb-Mar 2020, S&P -34%, the fastest crash
in modern market history) -- arguably a harder combined test than a single
2007-2009 window would have been.
"""
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

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
PRICE_FLOOR = datetime(2016, 1, 4, tzinfo=timezone.utc)
WINDOW_START = pd.Timestamp("2016-11-01", tz="UTC")
WINDOW_END = pd.Timestamp("2020-07-26", tz="UTC")
FETCH_END = datetime(2020, 9, 1, tzinfo=timezone.utc)  # room for hold_days to complete past window_end


def verdict_line(sharpe, dd95, cost_15):
    checks = [
        ("Sharpe>=0.5", sharpe >= 0.5 if sharpe == sharpe else False, f"{sharpe:.3f}"),
        ("MC_DD95<=20%", dd95 <= 0.20 if dd95 == dd95 else False, f"{dd95*100:.1f}%"),
        ("cost-15bps survival>0", cost_15 > 0 if cost_15 == cost_15 else False, f"{cost_15*100:.1f}%"),
    ]
    for name, ok, val in checks:
        print(f"    {name}: {'PASS' if ok else 'FAIL'} (value={val})")
    return all(c[1] for c in checks)


def main():
    all_tickers = discover_tickers()
    dc = DataClient(config.ALPACA_API_KEY, config.ALPACA_API_SECRET, "sip")
    av = AlphaVantageEarningsClient(config.ALPHA_VANTAGE_API_KEY)

    bars_by_ticker, earnings_by_ticker, coverage = {}, {}, {}
    for t in all_tickers:
        try:
            b = dc.get_bars(t, "us_equity", "1Day", PRICE_FLOOR, FETCH_END, use_cache=False)
        except Exception:
            b = pd.DataFrame()  # e.g. symbol has zero bars anywhere in range (real "no coverage", not a bug)
        e = av.get_quarterly_surprises(t)
        bars_by_ticker[t] = b
        earnings_by_ticker[t] = e
        coverage[t] = {
            "has_bars": not b.empty,
            "bar_start": b.index.min() if not b.empty else None,
            "full_coverage_from_window_start": (not b.empty) and b.index.min() <= WINDOW_START,
            "has_earnings_in_window": (not e.empty) and ((e.index >= WINDOW_START) & (e.index < WINDOW_END)).any(),
        }

    n_total = len(all_tickers)
    n_with_bars = sum(1 for c in coverage.values() if c["has_bars"])
    n_full_coverage = sum(1 for c in coverage.values() if c["full_coverage_from_window_start"])
    n_with_events = sum(1 for c in coverage.values() if c["full_coverage_from_window_start"] and c["has_earnings_in_window"])

    print(f"\n=== Data coverage report for {WINDOW_START.date()}..{WINDOW_END.date()} ===")
    print(f"Total cached tickers: {n_total}")
    print(f"With ANY price bars in [{PRICE_FLOOR.date()}, {FETCH_END.date()}]: {n_with_bars}")
    print(f"With FULL price coverage from window start ({WINDOW_START.date()}): {n_full_coverage}")
    print(f"...of those, with >=1 qualifying-window earnings event inside the window: {n_with_events}")
    missing = [t for t, c in coverage.items() if not c["full_coverage_from_window_start"]]
    if missing:
        print(f"Tickers WITHOUT full coverage from window start (likely IPO'd after {WINDOW_START.date()}): {missing}")
        for t in missing:
            print(f"  {t}: bar_start={coverage[t]['bar_start']}")

    usable_tickers = [t for t, c in coverage.items() if c["full_coverage_from_window_start"] and not earnings_by_ticker[t].empty]
    print(f"\nUsable tickers for this test: {len(usable_tickers)} of {n_total}\n")

    spy_bars = dc.get_bars("SPY", "us_equity", "1Day", PRICE_FLOOR, FETCH_END, use_cache=False)
    sma200 = spy_bars["close"].rolling(200).mean()
    below_sma200 = spy_bars["close"] < sma200
    early_exit_by_date = {ts.date(): bool(v) for ts, v in below_sma200.items() if v == v}
    regime_labels = classify_regime_periods(spy_bars)
    regime_in_window = regime_labels[(regime_labels.index >= WINDOW_START) & (regime_labels.index < WINDOW_END)]
    print("Regime mix over the test window:", (regime_in_window.value_counts(normalize=True) * 100).round(1).to_dict())

    for universe_tickers, label in [(usable_tickers, "full universe"),
                                     ([t for t in usable_tickers if t not in ORIGINAL_23], "new-only subset")]:
        print(f"\n########## {label}: {len(universe_tickers)} tickers ##########")
        for variant_name, eeb in [("baseline (no early exit)", None), ("with mid-hold early exit", early_exit_by_date)]:
            trades = build_portfolio_event_trades(universe_tickers, earnings_by_ticker, bars_by_ticker,
                                                   FROZEN_THRESH, FROZEN_HOLD, BASELINE_SLIPPAGE, STARTING_CAPITAL,
                                                   WINDOW_START, WINDOW_END, MAX_GROSS_EXPOSURE,
                                                   MAX_NEW_POSITIONS_PER_DAY, early_exit_by_date=eeb)
            equity = equity_curve_from_trades(trades, STARTING_CAPITAL, bars_by_ticker)
            metrics = compute_metrics(equity, trades, STARTING_CAPITAL)
            mc = monte_carlo_bootstrap(trades, STARTING_CAPITAL, n_resamples=1000, seed=42)

            cost_returns = {}
            for slip in SLIPPAGE_SCENARIOS:
                st = build_portfolio_event_trades(universe_tickers, earnings_by_ticker, bars_by_ticker,
                                                   FROZEN_THRESH, FROZEN_HOLD, slip, STARTING_CAPITAL,
                                                   WINDOW_START, WINDOW_END, MAX_GROSS_EXPOSURE,
                                                   MAX_NEW_POSITIONS_PER_DAY, early_exit_by_date=eeb)
                s = equity_curve_from_trades(st, STARTING_CAPITAL, bars_by_ticker)
                cost_returns[slip] = float(s.iloc[-1] / STARTING_CAPITAL - 1) if not s.empty else float("nan")

            print(f"\n  --- {variant_name} ---")
            print(f"  Sharpe={metrics['sharpe']:.3f}  Sortino={metrics['sortino']:.3f}  "
                  f"MaxDD={metrics['max_drawdown']*100:.1f}%  MC_DD95={mc['max_dd_p95']*100:.1f}%  "
                  f"PF={metrics['profit_factor']:.3f}  WinRate={metrics['win_rate']*100:.1f}%  Trades={metrics['trade_count']}")
            print("  Cost sensitivity: " + ", ".join(f"{k*100:.2f}%->{v*100:.1f}%" for k, v in cost_returns.items()))
            all_pass = verdict_line(metrics["sharpe"], mc["max_dd_p95"], cost_returns[0.0015])
            print(f"  VERDICT: {'PASSES all 3' if all_pass else 'FAILS at least 1'}")

            bd = regime_breakdown(equity, trades, regime_labels)
            print("  Regime breakdown:")
            for regime, stats in bd.items():
                ret = stats['total_return'] * 100 if stats['total_return'] == stats['total_return'] else float("nan")
                print(f"    {regime:<11} days={stats['days']:>4}  return={ret:>7.1f}%  "
                      f"sharpe={stats['sharpe']:>7.2f}  trades={stats['trade_count']}")

            # sub-period breakdown: Q4 2018 correction and COVID crash specifically
            for sub_label, sub_start, sub_end in [
                ("Q4-2018 correction", "2018-09-20", "2018-12-26"),
                ("COVID crash", "2020-02-19", "2020-04-07"),
            ]:
                sub_trades = [tr for tr in trades if pd.Timestamp(sub_start, tz="UTC") <= tr.entry_time < pd.Timestamp(sub_end, tz="UTC")]
                print(f"    [{sub_label}] entries in window: {len(sub_trades)}, "
                      f"total pnl={sum(tr.pnl for tr in sub_trades):.0f}")


if __name__ == "__main__":
    main()
