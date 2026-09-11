"""
Independent validation of round 2's Direction 2 finding (mean reversion +
SMA200 trend filter, daily bars, broad equity universe -- combined Sharpe
1.90 on a hand-picked 56-ticker universe).

Two independent checks, run with NO re-optimization of the discovered
signal itself:

1. FRESH UNIVERSE, same time period: an objectively-drawn set of tickers
   (S&P 500 pool minus the 56 already tested, fixed-seed random sample --
   documented in bot/sp500_pool.py, not curated) run through the exact same
   methodology (grid search, 30% anti-overfitting filter, same viability
   bar) on the SAME 2022-06 to now period.

2. FRESH TIME WINDOW, same tickers + already-chosen params: the 7 original
   winning tickers, using the EXACT parameters already selected for them
   (no re-fitting), evaluated on 2020-07-27 to 2022-06-03 -- a period that
   was never touched by any train or test window in the original
   walk-forward (whose earliest train_start was 2022-06-03). Pure holdout.

Pass/fail gate (fixed in advance, not adjusted after seeing results):
combined Sharpe on the fresh universe must stay > 0.5 AND not drop more
than 50% from the original 1.90 (i.e. must stay > 0.95).
"""
import json
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import config
from bot.data_client import DataClient
from bot.backtest_engine import (
    InstrumentSpec, generate_walk_forward_windows, grid_search_landscape, is_oos_deviation_ok,
    compute_metrics, monte_carlo_bootstrap, daily_returns_from_equity, simulate_instrument,
)
from bot.compare_strategies import _stitch_equity_curves, STARTING_CAPITAL, BASELINE_SLIPPAGE, SLIPPAGE_SCENARIOS
from bot.explore_v2 import BROAD_EQUITY_UNIVERSE, fetch_universe, isolated_final, cost_survival, plateau_score, survivors, TRAIN_MONTHS, TEST_MONTHS
from bot.sp500_pool import SP500_POOL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate_v2")

LOG_PATH = "exploration_log_v2.md"
RANDOM_SEED = 20260703  # fixed, documented: today's date at time of this validation, for reproducibility
FRESH_UNIVERSE_SIZE = 56  # match the original universe size for a fair comparison
ORIGINAL_COMBINED_SHARPE = 1.896
GATE_MIN_ABSOLUTE = 0.5
GATE_MIN_RELATIVE = 0.5 * ORIGINAL_COMBINED_SHARPE  # must not drop more than 50% -> stay above this

ORIGINAL_WINNERS = {
    "META": {"sma_period": 20, "std_dev_mult": 2.0, "trend_sma_period": 200},
    "NVDA": {"sma_period": 20, "std_dev_mult": 1.5, "trend_sma_period": 200},
    "MRK": {"sma_period": 20, "std_dev_mult": 2.0, "trend_sma_period": 200},
    "COP": {"sma_period": 20, "std_dev_mult": 1.5, "trend_sma_period": 200},
    "NKE": {"sma_period": 20, "std_dev_mult": 1.5, "trend_sma_period": 200},
    "PYPL": {"sma_period": 20, "std_dev_mult": 2.0, "trend_sma_period": 200},
    "GE": {"sma_period": 20, "std_dev_mult": 1.5, "trend_sma_period": 200},
}


def log_md(text: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    logger.info(text.split("\n")[0][:200])


def draw_fresh_universe() -> list:
    already_tested = set(BROAD_EQUITY_UNIVERSE)
    remaining_pool = [t for t in SP500_POOL if t not in already_tested]
    rng = np.random.default_rng(RANDOM_SEED)
    n = min(FRESH_UNIVERSE_SIZE, len(remaining_pool))
    drawn = rng.choice(remaining_pool, size=n, replace=False)
    return sorted(drawn.tolist())


def combined_portfolio(passers: list, slippage_pct: float = BASELINE_SLIPPAGE):
    """passers: list of dicts with key, spec, bars, windows, params."""
    daily_ret_series = {}
    all_trades = []
    for r in passers:
        segments = []
        for w in r["windows"]:
            oos_bars = r["bars"][(r["bars"].index >= w["train_start"]) & (r["bars"].index < w["test_end"])]
            t, eq = simulate_instrument(r["key"], r["spec"], oos_bars, "mean_reversion_trendfilter", r["params"],
                                        STARTING_CAPITAL, slippage_pct, active_start=w["test_start"])
            if not eq.empty:
                segments.append(eq)
            all_trades.extend(t)
        stitched = _stitch_equity_curves(segments, STARTING_CAPITAL)
        daily_ret_series[r["key"]] = daily_returns_from_equity(stitched)
    aligned = pd.DataFrame(daily_ret_series)
    combined_ret = aligned.mean(axis=1, skipna=True).dropna()
    if combined_ret.empty:
        return None, None, None
    combined_equity = STARTING_CAPITAL * (1 + combined_ret).cumprod()
    sharpe = float(combined_ret.mean() / combined_ret.std() * np.sqrt(252)) if combined_ret.std() > 0 else float("nan")
    metrics = compute_metrics(combined_equity, all_trades, STARTING_CAPITAL)
    mc = monte_carlo_bootstrap(all_trades, STARTING_CAPITAL, n_resamples=1000, seed=42)
    return sharpe, metrics, mc


def check_1_fresh_universe(dc, start_date, end_date) -> dict:
    log_md("\n## Independent validation, check 1: objectively-drawn fresh ticker universe\n")
    fresh_tickers = draw_fresh_universe()
    already_tested = set(BROAD_EQUITY_UNIVERSE)
    remaining_pool_size = len([t for t in SP500_POOL if t not in already_tested])
    log_md(f"- Sampling method: `bot/sp500_pool.py`'s fixed S&P 500 reference pool ({len(SP500_POOL)} tickers) "
           f"minus the {len(already_tested)} tickers already tested in round 2 direction 2 "
           f"({remaining_pool_size} remaining), drawn via `np.random.default_rng(seed={RANDOM_SEED})` "
           f"without replacement, n={FRESH_UNIVERSE_SIZE}. Seed and full ticker list documented here for reproducibility.")
    log_md(f"- Drawn tickers: {fresh_tickers}")

    daily_bars = fetch_universe(dc, fresh_tickers, "us_equity", "1Day", start_date, end_date, "fresh S&P500 draw")

    grid = [{"sma_period": 20, "std_dev_mult": m, "trend_sma_period": 200} for m in (1.5, 2.0)]
    results = []
    for key, bars in daily_bars.items():
        windows = generate_walk_forward_windows(bars.index.min(), bars.index.max(), TRAIN_MONTHS, TEST_MONTHS)
        if len(windows) < 3:
            continue
        spec = InstrumentSpec(key, "us_equity", "mean_reversion_trendfilter", "1Day")
        landscape = grid_search_landscape(key, spec, bars, "mean_reversion_trendfilter", grid, windows,
                                          BASELINE_SLIPPAGE, STARTING_CAPITAL)
        surv = survivors(landscape)
        plateau = plateau_score(surv if surv else landscape)
        if not plateau:
            continue
        best = plateau[0]
        metrics, mc, trades = isolated_final(key, spec, bars, windows, best["params"])
        cr = cost_survival(key, spec, bars, windows, best["params"])
        passes = (metrics["sharpe"] == metrics["sharpe"] and metrics["sharpe"] >= 0.5 and
                  mc["max_dd_p95"] == mc["max_dd_p95"] and mc["max_dd_p95"] <= 0.20 and
                  cr[0.0015] == cr[0.0015] and cr[0.0015] > 0)
        results.append({"key": key, "spec": spec, "bars": bars, "windows": windows, "params": best["params"],
                        "metrics": metrics, "mc": mc, "cost_returns": cr, "passes_all_3": passes})

    n_pass = sum(1 for r in results if r["passes_all_3"])
    log_md(f"- Fetched {len(daily_bars)}/{len(fresh_tickers)} tickers successfully. "
           f"{len(results)} produced a surviving anti-overfitting plateau. "
           f"**{n_pass} of {len(daily_bars)} passed all 3 viability criteria** "
           f"(original universe: 7 of 56 = 12.5%; this universe: {n_pass}/{len(daily_bars)} = "
           f"{n_pass/max(len(daily_bars),1)*100:.1f}%).")
    passers = [r for r in results if r["passes_all_3"]]
    for r in sorted(passers, key=lambda r: r["metrics"]["sharpe"], reverse=True):
        log_md(f"    PASS: {r['key']} Sharpe={r['metrics']['sharpe']:.3f} PF={r['metrics']['profit_factor']:.3f} "
               f"MaxDD={r['metrics']['max_drawdown']*100:.1f}% trades={r['metrics']['trade_count']}")

    if passers:
        sharpe, metrics, mc = combined_portfolio(passers)
        log_md(f"- **Combined equal-weight portfolio of {len(passers)} passers on the FRESH universe: "
               f"Sharpe={sharpe:.3f}** (original universe combined Sharpe was {ORIGINAL_COMBINED_SHARPE:.3f}) "
               f"MaxDD={metrics['max_drawdown']*100:.1f}% MC_DD95={mc['max_dd_p95']*100:.1f}% "
               f"PF={metrics['profit_factor']:.3f} trades={metrics['trade_count']}")
    else:
        sharpe, metrics, mc = None, None, None
        log_md("- No tickers passed all 3 criteria on the fresh universe -- no combined portfolio to build. "
               "Combined Sharpe is effectively 0/undefined for gate purposes.")

    gate_sharpe = sharpe if sharpe is not None and sharpe == sharpe else 0.0
    passes_gate = gate_sharpe > GATE_MIN_ABSOLUTE and gate_sharpe > GATE_MIN_RELATIVE
    log_md(f"- **Gate check**: combined Sharpe {gate_sharpe:.3f} vs required > {GATE_MIN_ABSOLUTE} "
           f"AND > {GATE_MIN_RELATIVE:.3f} (50% of original {ORIGINAL_COMBINED_SHARPE:.3f}). "
           f"**{'PASSES' if passes_gate else 'FAILS'} the pre-declared gate.**")

    return {"fresh_tickers": fresh_tickers, "n_tested": len(daily_bars), "n_pass": n_pass,
            "passers": [p["key"] for p in passers], "combined_sharpe": sharpe,
            "combined_metrics": metrics, "combined_mc": mc, "passes_gate": passes_gate}


def check_2_fresh_time_window(dc) -> dict:
    log_md("\n## Independent validation, check 2: genuinely unseen time window (pure holdout)\n")
    holdout_start = datetime(2020, 7, 27, tzinfo=timezone.utc)
    holdout_end = datetime(2022, 6, 3, tzinfo=timezone.utc)
    log_md(f"- Holdout window: {holdout_start.date()} to {holdout_end.date()} (~22.7 months) -- this is "
           f"BEFORE the earliest train_start (2022-06-03) used anywhere in the original walk-forward, so "
           f"it was never seen as train OR test data in any prior step.")
    log_md(f"- Using the 7 original winning tickers' EXACT already-selected parameters, no re-fitting: "
           f"{ORIGINAL_WINNERS}")

    daily_bars = fetch_universe(dc, list(ORIGINAL_WINNERS.keys()), "us_equity", "1Day",
                                holdout_start, holdout_end, "holdout period")

    daily_ret_series = {}
    all_trades = []
    per_ticker = []
    for key, params in ORIGINAL_WINNERS.items():
        bars = daily_bars.get(key)
        if bars is None or bars.empty:
            log_md(f"    {key}: no data in holdout window, skipped")
            continue
        spec = InstrumentSpec(key, "us_equity", "mean_reversion_trendfilter", "1Day")
        trades, equity = simulate_instrument(key, spec, bars, "mean_reversion_trendfilter", params,
                                             STARTING_CAPITAL, BASELINE_SLIPPAGE)
        metrics = compute_metrics(equity, trades, STARTING_CAPITAL)
        log_md(f"    {key}: Sharpe={metrics['sharpe']:.3f} PF={metrics['profit_factor']:.3f} "
               f"MaxDD={metrics['max_drawdown']*100:.1f}% trades={metrics['trade_count']}")
        per_ticker.append({"key": key, "metrics": metrics})
        if not equity.empty:
            daily_ret_series[key] = daily_returns_from_equity(equity)
        all_trades.extend(trades)

    if daily_ret_series:
        aligned = pd.DataFrame(daily_ret_series)
        combined_ret = aligned.mean(axis=1, skipna=True).dropna()
        combined_equity = STARTING_CAPITAL * (1 + combined_ret).cumprod()
        sharpe = float(combined_ret.mean() / combined_ret.std() * np.sqrt(252)) if combined_ret.std() > 0 else float("nan")
        metrics = compute_metrics(combined_equity, all_trades, STARTING_CAPITAL)
        log_md(f"- **Combined portfolio on the unseen 2020-2022 holdout: Sharpe={sharpe:.3f}** "
               f"(original 2022-2026 combined Sharpe was {ORIGINAL_COMBINED_SHARPE:.3f}) "
               f"MaxDD={metrics['max_drawdown']*100:.1f}% PF={metrics['profit_factor']:.3f} "
               f"trades={metrics['trade_count']}")
        return {"combined_sharpe": sharpe, "combined_metrics": metrics, "per_ticker": per_ticker}
    else:
        log_md("- No usable data in the holdout window for any ticker.")
        return {"combined_sharpe": None, "combined_metrics": None, "per_ticker": per_ticker}


def main():
    dc = DataClient(config.ALPACA_API_KEY, config.ALPACA_API_SECRET, config.ALPACA_DATA_FEED)
    end_date = datetime.now(timezone.utc) - timedelta(minutes=20)
    start_date = end_date - timedelta(days=365 * 4 + 30)

    check1 = check_1_fresh_universe(dc, start_date, end_date)
    check2 = check_2_fresh_time_window(dc)

    with open("validation_results.json", "w") as f:
        json.dump({
            "check1_fresh_universe": {k: v for k, v in check1.items() if k not in ("combined_metrics", "combined_mc")},
            "check1_passes_gate": check1["passes_gate"],
            "check2_fresh_time_combined_sharpe": check2["combined_sharpe"],
        }, f, indent=2, default=str)

    return check1, check2


if __name__ == "__main__":
    main()
