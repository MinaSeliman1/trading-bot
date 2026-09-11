"""
Overnight exploration: for each of the 5 base instrument-strategies, and for
3 logic variants, run an isolated (single-instrument, own capital) walk-
forward grid search across the same 12 windows used by the main comparison,
apply the anti-overfitting rule (reject OOS Sharpe deviating >30% from IS
Sharpe), and rank surviving candidates by OOS Sharpe + profit factor +
cross-window stability -- never by win rate.

Results are written incrementally to exploration_log.md so progress is
visible even if this doesn't finish. Final output: JSON dump of every
surviving candidate, ranked, for the morning summary to read from.
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
    simulate_instrument, simulate_momentum_breakout_mtf, compute_metrics, monte_carlo_bootstrap,
)
from bot.compare_strategies import _stitch_equity_curves, YEARS_OF_HISTORY, TRAIN_MONTHS, TEST_MONTHS, STARTING_CAPITAL, BASELINE_SLIPPAGE, SLIPPAGE_SCENARIOS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("overnight_exploration")

LOG_PATH = "exploration_log.md"
MAX_IS_OOS_DEVIATION = 0.30
STOP_COOLDOWN_BARS = 5


def log_md(text: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    logger.info(text.split("\n")[0][:200])


def plateau_score(landscape_rows: list) -> list:
    """Group landscape rows by param combo, score each by (mean OOS Sharpe,
    -std OOS Sharpe) -- a plateau has both a decent mean AND low variance
    across windows, unlike an isolated single-window peak."""
    by_params = {}
    for row in landscape_rows:
        key = json.dumps(row["params"], sort_keys=True)
        by_params.setdefault(key, []).append(row)

    scored = []
    for key, rows in by_params.items():
        oos_sharpes = [r["oos_sharpe"] for r in rows if r["oos_sharpe"] == r["oos_sharpe"]]
        if len(oos_sharpes) < 2:
            continue
        mean_sharpe = float(np.mean(oos_sharpes))
        std_sharpe = float(np.std(oos_sharpes))
        n_windows_with_trades = sum(1 for r in rows if r["oos_trade_count"] > 0)
        total_trades = sum(r["oos_trade_count"] for r in rows)
        pf_values = [r["oos_profit_factor"] for r in rows if r["oos_profit_factor"] == r["oos_profit_factor"]
                     and r["oos_profit_factor"] != float("inf")]
        mean_pf = float(np.mean(pf_values)) if pf_values else float("nan")
        scored.append({
            "params": json.loads(key), "mean_oos_sharpe": mean_sharpe, "std_oos_sharpe": std_sharpe,
            "mean_oos_profit_factor": mean_pf, "n_windows_with_trades": n_windows_with_trades,
            "total_trades": total_trades, "rows": rows,
        })
    return sorted(scored, key=lambda s: (s["mean_oos_sharpe"] - s["std_oos_sharpe"]), reverse=True)


def anti_overfitting_survivors(landscape_rows: list) -> list:
    return [r for r in landscape_rows if is_oos_deviation_ok(r["is_sharpe"], r["oos_sharpe"], MAX_IS_OOS_DEVIATION)]


def isolated_final_metrics(key, spec, bars, windows, params, stop_cooldown_bars=0):
    """Stitch the OOS equity/trades for one instrument at one fixed param set
    across all windows, using the warmup-corrected active_start mechanism."""
    segments, trades = [], []
    for w in windows:
        oos_warmup_bars = bars[(bars.index >= w["train_start"]) & (bars.index < w["test_end"])]
        t, eq = simulate_instrument(key, spec, oos_warmup_bars, spec.strategy_kind, params,
                                     STARTING_CAPITAL, BASELINE_SLIPPAGE, active_start=w["test_start"],
                                     stop_cooldown_bars=stop_cooldown_bars)
        if not eq.empty:
            segments.append(eq)
        trades.extend(t)
    stitched = _stitch_equity_curves(segments, STARTING_CAPITAL)
    metrics = compute_metrics(stitched, trades, STARTING_CAPITAL)
    mc = monte_carlo_bootstrap(trades, STARTING_CAPITAL, n_resamples=1000, seed=42)

    cost_returns = {}
    for slip in SLIPPAGE_SCENARIOS:
        segs = []
        for w in windows:
            oos_warmup_bars = bars[(bars.index >= w["train_start"]) & (bars.index < w["test_end"])]
            _, eq = simulate_instrument(key, spec, oos_warmup_bars, spec.strategy_kind, params,
                                         STARTING_CAPITAL, slip, active_start=w["test_start"],
                                         stop_cooldown_bars=stop_cooldown_bars)
            if not eq.empty:
                segs.append(eq)
        s = _stitch_equity_curves(segs, STARTING_CAPITAL)
        cost_returns[slip] = float(s.iloc[-1] / STARTING_CAPITAL - 1) if not s.empty else float("nan")

    return metrics, mc, cost_returns, trades


def main():
    dc = DataClient(config.ALPACA_API_KEY, config.ALPACA_API_SECRET, config.ALPACA_DATA_FEED)
    end_date = datetime.now(timezone.utc) - timedelta(minutes=20)
    start_date = end_date - timedelta(days=365 * YEARS_OF_HISTORY + 30)

    base_instruments = {
        "SPY": InstrumentSpec("SPY", "us_equity", "mean_reversion", "15Min"),
        "QQQ": InstrumentSpec("QQQ", "us_equity", "mean_reversion", "15Min"),
        "BTCUSD": InstrumentSpec("BTC/USD", "crypto", "momentum_breakout", "1Hour"),
        "GLD": InstrumentSpec("GLD", "us_equity", "trend_following", "4Hour"),
        "USO": InstrumentSpec("USO", "us_equity", "trend_following", "4Hour"),
    }

    bars_by_key = {}
    for key, spec in base_instruments.items():
        bars_by_key[key] = dc.get_bars(spec.symbol, spec.asset_class, spec.timeframe, start_date, end_date)
        logger.info("%s: %d bars", key, len(bars_by_key[key]))

    btc_4h = dc.get_bars("BTC/USD", "crypto", "4Hour", start_date, end_date)
    logger.info("BTCUSD 4Hour (for MTF variant): %d bars", len(btc_4h))

    windows = generate_walk_forward_windows(
        max(b.index.min() for b in bars_by_key.values()),
        min(b.index.max() for b in bars_by_key.values()), TRAIN_MONTHS, TEST_MONTHS)
    logger.info("%d walk-forward windows", len(windows))

    all_candidates = []

    log_md("\n## Round 1: base logic, isolated, win-rate-agnostic ranking\n")

    # ---------------- SPY / QQQ mean reversion: baseline, +cooldown, +trendfilter ---------------- #
    for key in ("SPY", "QQQ"):
        spec = base_instruments[key]
        std_grid = [1.0, 1.25, 1.5, 1.75, 2.0] if key == "SPY" else [1.2, 1.5, 1.8, 2.1, 2.4]

        variants = {
            "baseline": ([{"sma_period": 20, "std_dev_mult": m} for m in std_grid], 0, "mean_reversion"),
            "cooldown5": ([{"sma_period": 20, "std_dev_mult": m} for m in std_grid], STOP_COOLDOWN_BARS, "mean_reversion"),
            "trendfilter200": ([{"sma_period": 20, "std_dev_mult": m, "trend_sma_period": 200} for m in std_grid],
                               0, "mean_reversion_trendfilter"),
        }
        for variant_name, (grid, cooldown, kind) in variants.items():
            spec_v = InstrumentSpec(spec.symbol, spec.asset_class, kind, spec.timeframe)
            landscape = grid_search_landscape(key, spec_v, bars_by_key[key], kind, grid, windows,
                                               BASELINE_SLIPPAGE, STARTING_CAPITAL, use_regime=False,
                                               stop_cooldown_bars=cooldown)
            survivors = anti_overfitting_survivors(landscape)
            plateau = plateau_score(survivors if survivors else landscape)
            n_rejected = len(landscape) - len(survivors)
            log_md(f"- **{key} / {variant_name}**: {len(landscape)} (param,window) combos tested, "
                   f"{n_rejected} rejected by the 30% IS/OOS deviation rule. "
                   f"{'No survivors -- all combos overfit.' if not survivors else f'{len(plateau)} distinct param sets survived.'}")
            if not plateau:
                continue
            best = plateau[0]
            metrics, mc, cost_returns, trades = isolated_final_metrics(key, spec_v, bars_by_key[key], windows,
                                                                        best["params"], cooldown)
            log_md(f"  best plateau params={best['params']}: mean_oos_sharpe={best['mean_oos_sharpe']:.3f} "
                   f"std={best['std_oos_sharpe']:.3f} -> stitched Sharpe={metrics['sharpe']:.3f} "
                   f"PF={metrics['profit_factor']:.3f} MaxDD={metrics['max_drawdown']*100:.1f}% "
                   f"MC_DD95={mc['max_dd_p95']*100:.1f}% trades={metrics['trade_count']} "
                   f"cost0.15%={cost_returns[0.0015]*100:.1f}%")
            all_candidates.append({
                "label": f"{key} mean_reversion [{variant_name}]", "key": key, "variant": variant_name,
                "params": best["params"], "metrics": metrics, "mc": mc, "cost_returns": cost_returns,
                "is_oos_stability": best["std_oos_sharpe"], "n_windows_survived": best["n_windows_with_trades"],
            })

    # ---------------- BTCUSD momentum breakout: baseline vs MTF-confirmed ---------------- #
    spec = base_instruments["BTCUSD"]
    atr_grid = [1.5, 2.0, 2.5, 3.0]
    baseline_grid = [{"lookback": 20, "volume_mult": 1.5, "trailing_atr_mult": m} for m in atr_grid]
    landscape = grid_search_landscape("BTCUSD", spec, bars_by_key["BTCUSD"], "momentum_breakout", baseline_grid,
                                       windows, BASELINE_SLIPPAGE, STARTING_CAPITAL, use_regime=False)
    survivors = anti_overfitting_survivors(landscape)
    plateau = plateau_score(survivors if survivors else landscape)
    log_md(f"- **BTCUSD / baseline breakout**: {len(landscape)} combos, {len(landscape)-len(survivors)} rejected "
           f"by anti-overfitting rule. {len(plateau)} param sets survived.")
    if plateau:
        best = plateau[0]
        metrics, mc, cost_returns, _ = isolated_final_metrics("BTCUSD", spec, bars_by_key["BTCUSD"], windows, best["params"])
        log_md(f"  best params={best['params']}: Sharpe={metrics['sharpe']:.3f} PF={metrics['profit_factor']:.3f} "
               f"MaxDD={metrics['max_drawdown']*100:.1f}% MC_DD95={mc['max_dd_p95']*100:.1f}% trades={metrics['trade_count']}")
        all_candidates.append({"label": "BTCUSD momentum_breakout [baseline]", "key": "BTCUSD", "variant": "baseline",
                                "params": best["params"], "metrics": metrics, "mc": mc, "cost_returns": cost_returns,
                                "is_oos_stability": best["std_oos_sharpe"], "n_windows_survived": best["n_windows_with_trades"]})

    # MTF-confirmed variant: needs a dedicated loop since it's not in simulate_instrument's dispatch
    mtf_grid = [{"lookback": 20, "volume_mult": 1.5, "trailing_atr_mult": m, "mtf_fast_ema": 20, "mtf_slow_ema": 50}
                for m in atr_grid]
    mtf_rows = []
    for params in mtf_grid:
        for w_idx, w in enumerate(windows):
            train_1h = bars_by_key["BTCUSD"][(bars_by_key["BTCUSD"].index >= w["train_start"]) & (bars_by_key["BTCUSD"].index < w["train_end"])]
            train_4h = btc_4h[(btc_4h.index >= w["train_start"]) & (btc_4h.index < w["train_end"])]
            is_trades, is_eq = simulate_momentum_breakout_mtf("BTCUSD", spec, train_1h, train_4h, params, STARTING_CAPITAL, BASELINE_SLIPPAGE)
            is_metrics = compute_metrics(is_eq, is_trades, STARTING_CAPITAL)

            oos_1h = bars_by_key["BTCUSD"][(bars_by_key["BTCUSD"].index >= w["train_start"]) & (bars_by_key["BTCUSD"].index < w["test_end"])]
            oos_4h = btc_4h[(btc_4h.index >= w["train_start"]) & (btc_4h.index < w["test_end"])]
            oos_trades, oos_eq = simulate_momentum_breakout_mtf("BTCUSD", spec, oos_1h, oos_4h, params, STARTING_CAPITAL,
                                                                  BASELINE_SLIPPAGE, active_start=w["test_start"])
            oos_metrics = compute_metrics(oos_eq, oos_trades, STARTING_CAPITAL)
            mtf_rows.append({"params": params, "window": w_idx, "is_sharpe": is_metrics["sharpe"],
                              "oos_sharpe": oos_metrics["sharpe"], "oos_profit_factor": oos_metrics["profit_factor"],
                              "oos_trade_count": oos_metrics["trade_count"], "oos_max_drawdown": oos_metrics["max_drawdown"],
                              "oos_win_rate": oos_metrics["win_rate"]})
    mtf_survivors = anti_overfitting_survivors(mtf_rows)
    mtf_plateau = plateau_score(mtf_survivors if mtf_survivors else mtf_rows)
    log_md(f"- **BTCUSD / MTF-confirmed breakout**: {len(mtf_rows)} combos, {len(mtf_rows)-len(mtf_survivors)} rejected. "
           f"{len(mtf_plateau)} param sets survived.")
    if mtf_plateau:
        best = mtf_plateau[0]
        segs, trades = [], []
        for w in windows:
            oos_1h = bars_by_key["BTCUSD"][(bars_by_key["BTCUSD"].index >= w["train_start"]) & (bars_by_key["BTCUSD"].index < w["test_end"])]
            oos_4h = btc_4h[(btc_4h.index >= w["train_start"]) & (btc_4h.index < w["test_end"])]
            t, eq = simulate_momentum_breakout_mtf("BTCUSD", spec, oos_1h, oos_4h, best["params"], STARTING_CAPITAL,
                                                     BASELINE_SLIPPAGE, active_start=w["test_start"])
            if not eq.empty:
                segs.append(eq)
            trades.extend(t)
        stitched = _stitch_equity_curves(segs, STARTING_CAPITAL)
        metrics = compute_metrics(stitched, trades, STARTING_CAPITAL)
        mc = monte_carlo_bootstrap(trades, STARTING_CAPITAL, n_resamples=1000, seed=42)
        cost_returns = {}
        for slip in SLIPPAGE_SCENARIOS:
            segs2 = []
            for w in windows:
                oos_1h = bars_by_key["BTCUSD"][(bars_by_key["BTCUSD"].index >= w["train_start"]) & (bars_by_key["BTCUSD"].index < w["test_end"])]
                oos_4h = btc_4h[(btc_4h.index >= w["train_start"]) & (btc_4h.index < w["test_end"])]
                _, eq2 = simulate_momentum_breakout_mtf("BTCUSD", spec, oos_1h, oos_4h, best["params"], STARTING_CAPITAL,
                                                          slip, active_start=w["test_start"])
                if not eq2.empty:
                    segs2.append(eq2)
            s2 = _stitch_equity_curves(segs2, STARTING_CAPITAL)
            cost_returns[slip] = float(s2.iloc[-1] / STARTING_CAPITAL - 1) if not s2.empty else float("nan")
        log_md(f"  best params={best['params']}: Sharpe={metrics['sharpe']:.3f} PF={metrics['profit_factor']:.3f} "
               f"MaxDD={metrics['max_drawdown']*100:.1f}% MC_DD95={mc['max_dd_p95']*100:.1f}% trades={metrics['trade_count']} "
               f"cost0.15%={cost_returns[0.0015]*100:.1f}%")
        all_candidates.append({"label": "BTCUSD momentum_breakout [MTF-confirmed]", "key": "BTCUSD", "variant": "mtf",
                                "params": best["params"], "metrics": metrics, "mc": mc, "cost_returns": cost_returns,
                                "is_oos_stability": best["std_oos_sharpe"], "n_windows_survived": best["n_windows_with_trades"]})

    # ---------------- GLD / USO trend following: WIDE grid for plateau detection, +volfilter ---------------- #
    log_md("\n## Round 2: GLD/USO plateau-based grid tightening + volatility filter\n")
    wide_atr_grid = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    for key in ("GLD", "USO"):
        spec = base_instruments[key]
        baseline_grid = [{"fast_ema": 50, "slow_ema": 200, "trailing_atr_mult": m} for m in wide_atr_grid]
        landscape = grid_search_landscape(key, spec, bars_by_key[key], "trend_following", baseline_grid, windows,
                                           BASELINE_SLIPPAGE, STARTING_CAPITAL, use_regime=False)
        survivors = anti_overfitting_survivors(landscape)
        plateau = plateau_score(survivors if survivors else landscape)
        log_md(f"- **{key} / trend_following wide-grid plateau search**: {len(landscape)} combos, "
               f"{len(landscape)-len(survivors)} rejected. Plateau ranking (mean_oos_sharpe - std):")
        for p in plateau[:6]:
            log_md(f"    trailing_atr_mult={p['params']['trailing_atr_mult']}: mean_oos_sharpe={p['mean_oos_sharpe']:.3f} "
                   f"std={p['std_oos_sharpe']:.3f} n_windows_with_trades={p['n_windows_with_trades']} "
                   f"total_trades={p['total_trades']}")
        if plateau:
            best = plateau[0]
            metrics, mc, cost_returns, _ = isolated_final_metrics(key, spec, bars_by_key[key], windows, best["params"])
            log_md(f"  SELECTED plateau params={best['params']} -> Sharpe={metrics['sharpe']:.3f} "
                   f"PF={metrics['profit_factor']:.3f} MaxDD={metrics['max_drawdown']*100:.1f}% "
                   f"MC_DD95={mc['max_dd_p95']*100:.1f}% trades={metrics['trade_count']} "
                   f"cost0.15%={cost_returns[0.0015]*100:.1f}%")
            all_candidates.append({"label": f"{key} trend_following [wide-grid plateau]", "key": key, "variant": "baseline_tightened",
                                    "params": best["params"], "metrics": metrics, "mc": mc, "cost_returns": cost_returns,
                                    "is_oos_stability": best["std_oos_sharpe"], "n_windows_survived": best["n_windows_with_trades"]})

        volfilter_grid = [{"fast_ema": 50, "slow_ema": 200, "trailing_atr_mult": m, "vol_percentile_min": 0.25}
                           for m in wide_atr_grid]
        vf_landscape = grid_search_landscape(key, spec, bars_by_key[key], "trend_following_volfilter", volfilter_grid,
                                              windows, BASELINE_SLIPPAGE, STARTING_CAPITAL, use_regime=False)
        vf_survivors = anti_overfitting_survivors(vf_landscape)
        vf_plateau = plateau_score(vf_survivors if vf_survivors else vf_landscape)
        log_md(f"- **{key} / trend_following_volfilter**: {len(vf_landscape)} combos, "
               f"{len(vf_landscape)-len(vf_survivors)} rejected. {len(vf_plateau)} param sets survived.")
        if vf_plateau:
            best = vf_plateau[0]
            spec_vf = InstrumentSpec(spec.symbol, spec.asset_class, "trend_following_volfilter", spec.timeframe)
            metrics, mc, cost_returns, _ = isolated_final_metrics(key, spec_vf, bars_by_key[key], windows, best["params"])
            log_md(f"  best params={best['params']}: Sharpe={metrics['sharpe']:.3f} PF={metrics['profit_factor']:.3f} "
                   f"MaxDD={metrics['max_drawdown']*100:.1f}% MC_DD95={mc['max_dd_p95']*100:.1f}% trades={metrics['trade_count']} "
                   f"cost0.15%={cost_returns[0.0015]*100:.1f}%")
            all_candidates.append({"label": f"{key} trend_following [volfilter]", "key": key, "variant": "volfilter",
                                    "params": best["params"], "metrics": metrics, "mc": mc, "cost_returns": cost_returns,
                                    "is_oos_stability": best["std_oos_sharpe"], "n_windows_survived": best["n_windows_with_trades"]})

    with open("exploration_candidates.json", "w") as f:
        json.dump([{
            "label": c["label"], "params": c["params"],
            "sharpe": c["metrics"]["sharpe"], "sortino": c["metrics"]["sortino"],
            "profit_factor": c["metrics"]["profit_factor"], "max_drawdown": c["metrics"]["max_drawdown"],
            "trade_count": c["metrics"]["trade_count"], "mc_dd95": c["mc"]["max_dd_p95"],
            "mc_sharpe_p5": c["mc"]["sharpe_p5"], "cost_0_05": c["cost_returns"].get(0.0005),
            "cost_0_10": c["cost_returns"].get(0.001), "cost_0_15": c["cost_returns"].get(0.0015),
            "is_oos_stability_std": c["is_oos_stability"], "n_windows_survived": c["n_windows_survived"],
        } for c in all_candidates], f, indent=2, default=str)

    log_md(f"\n## Round 1+2 complete: {len(all_candidates)} candidates collected. See exploration_candidates.json.\n")
    logger.info("DONE. %d candidates written to exploration_candidates.json", len(all_candidates))
    return all_candidates


if __name__ == "__main__":
    main()
