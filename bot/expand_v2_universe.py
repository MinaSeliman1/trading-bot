"""
Step 2A: cautious universe expansion, triggered because the fresh-universe
gate check passed. NO re-optimization of the signal itself -- same exact
grid, same 30% anti-overfitting rule, same viability bar, applied to all
tickers in bot/sp500_pool.py not already tested in either the original
Direction 2 batch (56) or the independent validation draw (56). This finds
how many MORE tickers pass the SAME unchanged filter, not tickers chosen to
improve the headline number.
"""
import json
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import config
from bot.data_client import DataClient
from bot.backtest_engine import (
    InstrumentSpec, generate_walk_forward_windows, grid_search_landscape, compute_metrics,
    monte_carlo_bootstrap, daily_returns_from_equity, simulate_instrument,
)
from bot.compare_strategies import _stitch_equity_curves, STARTING_CAPITAL, BASELINE_SLIPPAGE
from bot.explore_v2 import BROAD_EQUITY_UNIVERSE, fetch_universe, isolated_final, cost_survival, plateau_score, survivors, TRAIN_MONTHS, TEST_MONTHS
from bot.sp500_pool import SP500_POOL
from bot.validate_v2_independent import draw_fresh_universe, ORIGINAL_COMBINED_SHARPE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("expand_v2")

LOG_PATH = "exploration_log_v2.md"


def log_md(text: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    logger.info(text.split("\n")[0][:200])


def main():
    already_tested = set(BROAD_EQUITY_UNIVERSE) | set(draw_fresh_universe())
    expansion_universe = sorted([t for t in SP500_POOL if t not in already_tested])

    log_md("\n## Step 2A: cautious universe expansion (unchanged criteria, no re-optimization)\n")
    log_md(f"- Universe: all {len(SP500_POOL)} tickers in `bot/sp500_pool.py` minus the "
           f"{len(already_tested)} already tested (56 original + 56 independent-validation draw) "
           f"= {len(expansion_universe)} new tickers. Same exact grid ({{sma_period:20, "
           f"std_dev_mult in [1.5,2.0], trend_sma_period:200}}), same 30% anti-overfitting rule, "
           f"same viability bar (Sharpe>=0.5, MC_DD95<=20%, edge survives 0.15% slippage). "
           f"No parameter changes of any kind from what was already validated.")

    dc = DataClient(config.ALPACA_API_KEY, config.ALPACA_API_SECRET, config.ALPACA_DATA_FEED)
    end_date = datetime.now(timezone.utc) - timedelta(minutes=20)
    start_date = end_date - timedelta(days=365 * 4 + 30)

    daily_bars = fetch_universe(dc, expansion_universe, "us_equity", "1Day", start_date, end_date, "expansion")

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
    passers = [r for r in results if r["passes_all_3"]]
    log_md(f"- Fetched {len(daily_bars)}/{len(expansion_universe)} tickers. {len(results)} produced a "
           f"surviving plateau. **{n_pass} of {len(daily_bars)} new tickers passed all 3 criteria.**")
    for r in sorted(passers, key=lambda r: r["metrics"]["sharpe"], reverse=True):
        log_md(f"    PASS: {r['key']} Sharpe={r['metrics']['sharpe']:.3f} PF={r['metrics']['profit_factor']:.3f} "
               f"MaxDD={r['metrics']['max_drawdown']*100:.1f}% trades={r['metrics']['trade_count']}")

    # cumulative multiple-testing accounting across all rounds so far
    total_tested = 56 + 56 + len(daily_bars)
    total_pass = 7 + 2 + n_pass
    log_md(f"\n- **Cumulative multiple-testing accounting**: {total_tested} total tickers tested across all "
           f"rounds (56 original Direction 2 + 56 independent validation draw + {len(daily_bars)} expansion), "
           f"{total_pass} total passed all 3 viability criteria at any point ({total_pass/total_tested*100:.1f}% "
           f"overall hit rate). This is the honest denominator for judging whether ANY individual pass rate "
           f"is more than chance would produce.")

    if not passers:
        log_md("- No new tickers passed. No combined-portfolio diversification test to run for this batch.")
        return {"n_new_pass": 0, "total_tested": total_tested, "total_pass": total_pass}

    # combined portfolio of ALL passers found so far across all 3 batches (cumulative), not just this batch
    daily_ret_series = {}
    all_trades = []
    for r in passers:
        segments = []
        for w in r["windows"]:
            oos_bars = r["bars"][(r["bars"].index >= w["train_start"]) & (r["bars"].index < w["test_end"])]
            t, eq = simulate_instrument(r["key"], r["spec"], oos_bars, "mean_reversion_trendfilter", r["params"],
                                        STARTING_CAPITAL, BASELINE_SLIPPAGE, active_start=w["test_start"])
            if not eq.empty:
                segments.append(eq)
            all_trades.extend(t)
        stitched = _stitch_equity_curves(segments, STARTING_CAPITAL)
        daily_ret_series[r["key"]] = daily_returns_from_equity(stitched)

    aligned = pd.DataFrame(daily_ret_series)
    combined_ret = aligned.mean(axis=1, skipna=True).dropna()
    combined_equity = STARTING_CAPITAL * (1 + combined_ret).cumprod()
    sharpe = float(combined_ret.mean() / combined_ret.std() * np.sqrt(252)) if combined_ret.std() > 0 else float("nan")
    metrics = compute_metrics(combined_equity, all_trades, STARTING_CAPITAL)
    mc = monte_carlo_bootstrap(all_trades, STARTING_CAPITAL, n_resamples=1000, seed=42)
    log_md(f"- **Combined portfolio of this batch's {len(passers)} new passers alone: Sharpe={sharpe:.3f}** "
           f"MaxDD={metrics['max_drawdown']*100:.1f}% MC_DD95={mc['max_dd_p95']*100:.1f}% "
           f"PF={metrics['profit_factor']:.3f} trades={metrics['trade_count']}")

    with open("expansion_results.json", "w") as f:
        json.dump({"n_new_tested": len(daily_bars), "n_new_pass": n_pass, "new_passers": [p["key"] for p in passers],
                   "combined_sharpe_new_batch": sharpe, "total_tested": total_tested, "total_pass": total_pass},
                  f, indent=2, default=str)

    return {"n_new_pass": n_pass, "combined_sharpe": sharpe, "total_tested": total_tested, "total_pass": total_pass}


if __name__ == "__main__":
    main()
