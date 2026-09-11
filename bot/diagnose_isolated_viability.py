"""
Diagnostic run BEFORE any parameter-grid tightening or re-entry-cooldown work:
does each instrument-strategy's BASE logic (Strategy A style -- no regime
filter, no portfolio-level correlation/exposure effects) have ANY standalone
edge at all when tested in isolation, on its own capital, walk-forward
optimized on its own parameters?

This answers a different question than the A-vs-B portfolio comparison: not
"is B's added complexity worth it" but "does the underlying signal (SMA/std
bands, breakout, EMA cross) have predictive value on this instrument over
this period at all" -- before spending more effort tuning parameters or
fairness details around a signal that might have no edge to find.

Uses the SAME walk-forward window schedule and parameter grids as
compare_strategies.py, and the same isolated single-instrument backtest
(simulate_instrument) already used there for grid-search parameter ranking --
just run standalone and reported without stitching into a portfolio.
"""
import logging
from datetime import datetime, timedelta, timezone

import config
from bot.data_client import DataClient
from bot.backtest_engine import (
    generate_walk_forward_windows, optimize_parameters, simulate_instrument,
    compute_metrics, monte_carlo_bootstrap,
)
from bot.compare_strategies import (
    INSTRUMENTS, PARAM_GRIDS, STARTING_CAPITAL, BASELINE_SLIPPAGE,
    TRAIN_MONTHS, TEST_MONTHS, SLIPPAGE_SCENARIOS, YEARS_OF_HISTORY, _stitch_equity_curves,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("diagnose_isolated")

VIABILITY_SHARPE_MIN = 0.5
VIABILITY_MC_DD95_MAX = 0.20


def diagnose_instrument(key: str, spec, bars, windows: list) -> dict:
    oos_segments, oos_trades, window_params = [], [], []

    for w in windows:
        params = optimize_parameters(key, spec, bars, spec.strategy_kind, PARAM_GRIDS[key],
                                      w["train_start"], w["train_end"], BASELINE_SLIPPAGE,
                                      STARTING_CAPITAL, use_regime=False)
        window_params.append(params)
        # Include the train period as warmup so slow indicators (e.g. a 200-period EMA on
        # 4-hour GLD/USO bars) are already hot by test_start, not cold-started at the boundary.
        oos_warmup_bars = bars[(bars.index >= w["train_start"]) & (bars.index < w["test_end"])]
        trades, equity = simulate_instrument(key, spec, oos_warmup_bars, spec.strategy_kind, params,
                                              STARTING_CAPITAL, BASELINE_SLIPPAGE, use_regime=False,
                                              active_start=w["test_start"])
        if not equity.empty:
            oos_segments.append(equity)
        oos_trades.extend(trades)

    stitched = _stitch_equity_curves(oos_segments, STARTING_CAPITAL)
    metrics = compute_metrics(stitched, oos_trades, STARTING_CAPITAL)
    mc = monte_carlo_bootstrap(oos_trades, STARTING_CAPITAL, n_resamples=1000, seed=42)

    cost_returns = {}
    for slip in SLIPPAGE_SCENARIOS:
        segs = []
        for w, params in zip(windows, window_params):
            oos_warmup_bars = bars[(bars.index >= w["train_start"]) & (bars.index < w["test_end"])]
            _, equity = simulate_instrument(key, spec, oos_warmup_bars, spec.strategy_kind, params,
                                             STARTING_CAPITAL, slip, use_regime=False,
                                             active_start=w["test_start"])
            if not equity.empty:
                segs.append(equity)
        stitched_slip = _stitch_equity_curves(segs, STARTING_CAPITAL)
        cost_returns[slip] = float(stitched_slip.iloc[-1] / STARTING_CAPITAL - 1) if not stitched_slip.empty else float("nan")

    sharpe = metrics["sharpe"]
    dd95 = mc["max_dd_p95"]
    cost_15 = cost_returns[0.0015]

    viable_sharpe = sharpe == sharpe and sharpe >= VIABILITY_SHARPE_MIN
    viable_dd = dd95 == dd95 and dd95 <= VIABILITY_MC_DD95_MAX
    viable_cost = cost_15 == cost_15 and cost_15 > 0
    is_viable = viable_sharpe and viable_dd and viable_cost

    return {
        "key": key, "metrics": metrics, "mc": mc, "cost_returns": cost_returns,
        "viable_sharpe": viable_sharpe, "viable_dd": viable_dd, "viable_cost": viable_cost,
        "is_viable": is_viable, "window_params": window_params, "n_windows": len(windows),
    }


def main():
    dc = DataClient(config.ALPACA_API_KEY, config.ALPACA_API_SECRET, config.ALPACA_DATA_FEED)
    end_date = datetime.now(timezone.utc) - timedelta(minutes=20)
    start_date = end_date - timedelta(days=365 * YEARS_OF_HISTORY + 30)

    results = []
    for key, spec in INSTRUMENTS.items():
        bars = dc.get_bars(spec.symbol, spec.asset_class, spec.timeframe, start_date, end_date)
        windows = generate_walk_forward_windows(bars.index.min(), bars.index.max(), TRAIN_MONTHS, TEST_MONTHS)
        logger.info("%s: %d cached bars, %d walk-forward windows", key, len(bars), len(windows))
        result = diagnose_instrument(key, spec, bars, windows)
        results.append(result)
        m, mc = result["metrics"], result["mc"]
        logger.info("%s -> OOS Sharpe=%.3f  MaxDD=%.1f%%  MC_DD95=%.1f%%  WinRate=%.1f%%  ProfitFactor=%.3f  "
                    "Trades=%d  Cost@0.15%%=%.1f%%  VIABLE=%s",
                    key, m["sharpe"], m["max_drawdown"] * 100, mc["max_dd_p95"] * 100, m["win_rate"] * 100,
                    m["profit_factor"], m["trade_count"], result["cost_returns"][0.0015] * 100, result["is_viable"])

    print("\n=== ISOLATED VIABILITY SUMMARY (ranked by OOS Sharpe) ===")
    ranked = sorted(results, key=lambda r: r["metrics"]["sharpe"] if r["metrics"]["sharpe"] == r["metrics"]["sharpe"] else -999,
                     reverse=True)
    for r in ranked:
        m, mc = r["metrics"], r["mc"]
        print(f"{r['key']:8s} Sharpe={m['sharpe']:7.3f}  MaxDD={m['max_drawdown']*100:6.1f}%  "
              f"MC_DD95={mc['max_dd_p95']*100:6.1f}%  WinRate={m['win_rate']*100:5.1f}%  "
              f"PF={m['profit_factor']:.3f}  Trades={m['trade_count']:4d}  "
              f"Cost0.15%={r['cost_returns'][0.0015]*100:7.1f}%  VIABLE={r['is_viable']}")

    return results


if __name__ == "__main__":
    main()
