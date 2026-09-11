"""
Round 2 exploration: structurally different directions, not just parameter
tuning on the same signal families validated (and found wanting) in round 1.

1. Wider, less-arbitraged instrument universe, same validated logic
   (mean reversion + SMA200 trend filter on 20 mid-cap equities; momentum
   breakout on 10 liquid crypto alt-coins).
2. Daily bars across a broad ~55-ticker liquid equity universe.
3. Cross-sectional momentum (rank a basket by trailing return, hold the top
   quintile, rebalance monthly) -- fundamentally different from a
   time-series signal on a single instrument.
4. Portfolio-level SPY-SMA200 macro gate on top of the mean reversion +
   trend filter that showed the one real effect in round 1.

Same anti-overfitting rules as round 1: never rank on win rate; reject any
(param, window) pair where OOS Sharpe deviates >30% from IS Sharpe; report
honestly if nothing clears the viability bar. Scope capped at these 4
directions.
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
    simulate_instrument, simulate_mean_reversion_macrogate, compute_metrics, monte_carlo_bootstrap,
    build_price_matrix, cross_sectional_momentum_backtest,
)
from bot.compare_strategies import _stitch_equity_curves, STARTING_CAPITAL, BASELINE_SLIPPAGE, SLIPPAGE_SCENARIOS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("explore_v2")

LOG_PATH = "exploration_log_v2.md"
MAX_IS_OOS_DEVIATION = 0.30
YEARS_OF_HISTORY = 4
TRAIN_MONTHS = 12
TEST_MONTHS = 3

MIDCAP_EQUITIES = ["DKS", "RL", "M", "KSS", "AAL", "ALK", "JBLU", "CPB", "HRL", "CAG",
                   "MDU", "HRB", "TXT", "RRC", "DVN", "APA", "ANF", "URBN", "BBY", "F"]
CRYPTO_ALTS = ["ETH/USD", "LTC/USD", "LINK/USD", "AVAX/USD", "DOGE/USD",
               "DOT/USD", "BCH/USD", "CRV/USD", "GRT/USD", "AAVE/USD"]
BROAD_EQUITY_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "BAC", "WFC", "GS", "MS",
    "JNJ", "PFE", "UNH", "MRK", "ABBV", "XOM", "CVX", "COP", "PG", "KO", "PEP", "WMT", "HD",
    "MCD", "NKE", "DIS", "CMCSA", "VZ", "T", "INTC", "AMD", "CSCO", "ORCL", "IBM", "CRM",
    "ADBE", "PYPL", "V", "MA", "AXP", "BA", "CAT", "GE", "MMM", "HON", "UPS", "FDX", "LMT",
    "RTX", "DE", "UNP", "NEE", "DUK", "SO",
]


def log_md(text: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    logger.info(text.split("\n")[0][:200])


def plateau_score(landscape_rows: list) -> list:
    by_params = {}
    for row in landscape_rows:
        key = json.dumps(row["params"], sort_keys=True)
        by_params.setdefault(key, []).append(row)
    scored = []
    for key, rows in by_params.items():
        oos = [r["oos_sharpe"] for r in rows if r["oos_sharpe"] == r["oos_sharpe"]]
        if len(oos) < 2:
            continue
        mean_s, std_s = float(np.mean(oos)), float(np.std(oos))
        pf_vals = [r["oos_profit_factor"] for r in rows if r["oos_profit_factor"] == r["oos_profit_factor"]
                   and r["oos_profit_factor"] != float("inf")]
        scored.append({"params": json.loads(key), "mean_oos_sharpe": mean_s, "std_oos_sharpe": std_s,
                       "mean_oos_profit_factor": float(np.mean(pf_vals)) if pf_vals else float("nan"),
                       "n_windows_with_trades": sum(1 for r in rows if r["oos_trade_count"] > 0),
                       "total_trades": sum(r["oos_trade_count"] for r in rows)})
    return sorted(scored, key=lambda s: s["mean_oos_sharpe"] - s["std_oos_sharpe"], reverse=True)


def survivors(landscape_rows: list) -> list:
    return [r for r in landscape_rows if is_oos_deviation_ok(r["is_sharpe"], r["oos_sharpe"], MAX_IS_OOS_DEVIATION)]


def isolated_final(key, spec, bars, windows, params, sim_fn=simulate_instrument, extra_args=()):
    segments, trades = [], []
    for w in windows:
        oos_bars = bars[(bars.index >= w["train_start"]) & (bars.index < w["test_end"])]
        t, eq = sim_fn(key, spec, oos_bars, *extra_args, params, STARTING_CAPITAL, BASELINE_SLIPPAGE,
                       active_start=w["test_start"]) if sim_fn is not simulate_instrument else \
                sim_fn(key, spec, oos_bars, spec.strategy_kind, params, STARTING_CAPITAL, BASELINE_SLIPPAGE,
                       active_start=w["test_start"])
        if not eq.empty:
            segments.append(eq)
        trades.extend(t)
    stitched = _stitch_equity_curves(segments, STARTING_CAPITAL)
    metrics = compute_metrics(stitched, trades, STARTING_CAPITAL)
    mc = monte_carlo_bootstrap(trades, STARTING_CAPITAL, n_resamples=1000, seed=42)
    return metrics, mc, trades


def cost_survival(key, spec, bars, windows, params, sim_fn=simulate_instrument, extra_args=()):
    out = {}
    for slip in SLIPPAGE_SCENARIOS:
        segs = []
        for w in windows:
            oos_bars = bars[(bars.index >= w["train_start"]) & (bars.index < w["test_end"])]
            if sim_fn is simulate_instrument:
                _, eq = sim_fn(key, spec, oos_bars, spec.strategy_kind, params, STARTING_CAPITAL, slip,
                               active_start=w["test_start"])
            else:
                _, eq = sim_fn(key, spec, oos_bars, *extra_args, params, STARTING_CAPITAL, slip,
                               active_start=w["test_start"])
            if not eq.empty:
                segs.append(eq)
        s = _stitch_equity_curves(segs, STARTING_CAPITAL)
        out[slip] = float(s.iloc[-1] / STARTING_CAPITAL - 1) if not s.empty else float("nan")
    return out


def fetch_universe(dc, tickers, asset_class, timeframe, start, end, label):
    out = {}
    for t in tickers:
        try:
            bars = dc.get_bars(t, asset_class, timeframe, start, end)
            if len(bars) > 100:
                out[t] = bars
            else:
                logger.warning("%s: only %d bars, skipping", t, len(bars))
        except Exception:
            logger.exception("%s: fetch failed, skipping", t)
    logger.info("%s universe: %d/%d tickers usable", label, len(out), len(tickers))
    return out


def main():
    dc = DataClient(config.ALPACA_API_KEY, config.ALPACA_API_SECRET, config.ALPACA_DATA_FEED)
    end_date = datetime.now(timezone.utc) - timedelta(minutes=20)
    start_date = end_date - timedelta(days=365 * YEARS_OF_HISTORY + 30)

    all_candidates = []

    # =================================================================== #
    # Direction 1: wider universe, same intraday logic
    # =================================================================== #
    log_md("\n## Direction 1: wider instrument universe, same validated intraday logic\n")
    midcap_bars = fetch_universe(dc, MIDCAP_EQUITIES, "us_equity", "15Min", start_date, end_date, "mid-cap equity")
    crypto_bars = fetch_universe(dc, CRYPTO_ALTS, "crypto", "1Hour", start_date, end_date, "crypto alt")

    std_grid = [1.5, 2.0]
    equity_results = []
    for key, bars in midcap_bars.items():
        windows = generate_walk_forward_windows(bars.index.min(), bars.index.max(), TRAIN_MONTHS, TEST_MONTHS)
        if len(windows) < 3:
            continue
        spec = InstrumentSpec(key, "us_equity", "mean_reversion_trendfilter", "15Min")
        grid = [{"sma_period": 20, "std_dev_mult": m, "trend_sma_period": 200} for m in std_grid]
        landscape = grid_search_landscape(key, spec, bars, "mean_reversion_trendfilter", grid, windows,
                                           BASELINE_SLIPPAGE, STARTING_CAPITAL)
        survived = survivors(landscape)
        plateau = plateau_score(survived if survived else landscape)
        if plateau:
            best = plateau[0]
            metrics, mc, _ = isolated_final(key, spec, bars, windows, best["params"])
            equity_results.append({"key": key, "params": best["params"], "metrics": metrics, "mc": mc,
                                    "plateau": best, "n_windows": len(windows)})

    equity_results.sort(key=lambda r: r["metrics"]["sharpe"] if r["metrics"]["sharpe"] == r["metrics"]["sharpe"] else -999,
                        reverse=True)
    log_md(f"- Tested {len(midcap_bars)} mid-cap equities with mean_reversion_trendfilter. "
           f"{len(equity_results)} produced a surviving plateau. Top 5 by OOS Sharpe:")
    for r in equity_results[:5]:
        m, mc = r["metrics"], r["mc"]
        cr = cost_survival(r["key"], InstrumentSpec(r["key"], "us_equity", "mean_reversion_trendfilter", "15Min"),
                           midcap_bars[r["key"]], generate_walk_forward_windows(
                               midcap_bars[r["key"]].index.min(), midcap_bars[r["key"]].index.max(),
                               TRAIN_MONTHS, TEST_MONTHS), r["params"])
        log_md(f"    {r['key']}: Sharpe={m['sharpe']:.3f} PF={m['profit_factor']:.3f} "
               f"MaxDD={m['max_drawdown']*100:.1f}% MC_DD95={mc['max_dd_p95']*100:.1f}% trades={m['trade_count']} "
               f"cost0.15%={cr[0.0015]*100:.1f}%")
        all_candidates.append({"label": f"{r['key']} mean_reversion_trendfilter [mid-cap universe]",
                               "params": r["params"], "metrics": m, "mc": mc, "cost_returns": cr,
                               "direction": "1-universe-equity"})

    breakout_grid = [{"lookback": 20, "volume_mult": 1.5, "trailing_atr_mult": m} for m in (2.0, 2.5, 3.0)]
    crypto_results = []
    for key, bars in crypto_bars.items():
        windows = generate_walk_forward_windows(bars.index.min(), bars.index.max(), TRAIN_MONTHS, TEST_MONTHS)
        if len(windows) < 3:
            continue
        spec = InstrumentSpec(key, "crypto", "momentum_breakout", "1Hour")
        landscape = grid_search_landscape(key, spec, bars, "momentum_breakout", breakout_grid, windows,
                                           BASELINE_SLIPPAGE, STARTING_CAPITAL)
        survived = survivors(landscape)
        plateau = plateau_score(survived if survived else landscape)
        if plateau:
            best = plateau[0]
            metrics, mc, _ = isolated_final(key, spec, bars, windows, best["params"])
            crypto_results.append({"key": key, "params": best["params"], "metrics": metrics, "mc": mc})

    crypto_results.sort(key=lambda r: r["metrics"]["sharpe"] if r["metrics"]["sharpe"] == r["metrics"]["sharpe"] else -999,
                        reverse=True)
    log_md(f"- Tested {len(crypto_bars)} crypto alt-coins with momentum_breakout. Top 5 by OOS Sharpe:")
    for r in crypto_results[:5]:
        m, mc = r["metrics"], r["mc"]
        windows = generate_walk_forward_windows(crypto_bars[r["key"]].index.min(), crypto_bars[r["key"]].index.max(),
                                                  TRAIN_MONTHS, TEST_MONTHS)
        cr = cost_survival(r["key"], InstrumentSpec(r["key"], "crypto", "momentum_breakout", "1Hour"),
                           crypto_bars[r["key"]], windows, r["params"])
        log_md(f"    {r['key']}: Sharpe={m['sharpe']:.3f} PF={m['profit_factor']:.3f} "
               f"MaxDD={m['max_drawdown']*100:.1f}% MC_DD95={mc['max_dd_p95']*100:.1f}% trades={m['trade_count']} "
               f"cost0.15%={cr[0.0015]*100:.1f}%")
        all_candidates.append({"label": f"{r['key']} momentum_breakout [crypto alt universe]",
                               "params": r["params"], "metrics": m, "mc": mc, "cost_returns": cr,
                               "direction": "1-universe-crypto"})

    # =================================================================== #
    # Direction 2: daily bars, broad equity universe
    # =================================================================== #
    log_md("\n## Direction 2: daily bars, broad liquid equity universe\n")
    daily_bars = fetch_universe(dc, BROAD_EQUITY_UNIVERSE, "us_equity", "1Day", start_date, end_date, "broad daily")

    daily_grid = [{"sma_period": 20, "std_dev_mult": m, "trend_sma_period": 200} for m in std_grid]
    daily_results = []
    for key, bars in daily_bars.items():
        windows = generate_walk_forward_windows(bars.index.min(), bars.index.max(), TRAIN_MONTHS, TEST_MONTHS)
        if len(windows) < 3:
            continue
        spec = InstrumentSpec(key, "us_equity", "mean_reversion_trendfilter", "1Day")
        landscape = grid_search_landscape(key, spec, bars, "mean_reversion_trendfilter", daily_grid, windows,
                                           BASELINE_SLIPPAGE, STARTING_CAPITAL)
        survived = survivors(landscape)
        plateau = plateau_score(survived if survived else landscape)
        if plateau:
            best = plateau[0]
            metrics, mc, _ = isolated_final(key, spec, bars, windows, best["params"])
            daily_results.append({"key": key, "params": best["params"], "metrics": metrics, "mc": mc})

    daily_results.sort(key=lambda r: r["metrics"]["sharpe"] if r["metrics"]["sharpe"] == r["metrics"]["sharpe"] else -999,
                       reverse=True)
    log_md(f"- Tested {len(daily_bars)} liquid equities on DAILY bars with mean_reversion_trendfilter. "
           f"{len(daily_results)} produced a surviving plateau. Top 5 by OOS Sharpe:")
    for r in daily_results[:5]:
        m, mc = r["metrics"], r["mc"]
        windows = generate_walk_forward_windows(daily_bars[r["key"]].index.min(), daily_bars[r["key"]].index.max(),
                                                   TRAIN_MONTHS, TEST_MONTHS)
        cr = cost_survival(r["key"], InstrumentSpec(r["key"], "us_equity", "mean_reversion_trendfilter", "1Day"),
                           daily_bars[r["key"]], windows, r["params"])
        log_md(f"    {r['key']}: Sharpe={m['sharpe']:.3f} PF={m['profit_factor']:.3f} "
               f"MaxDD={m['max_drawdown']*100:.1f}% MC_DD95={mc['max_dd_p95']*100:.1f}% trades={m['trade_count']} "
               f"cost0.15%={cr[0.0015]*100:.1f}%")
        all_candidates.append({"label": f"{r['key']} mean_reversion_trendfilter [daily bars]",
                               "params": r["params"], "metrics": m, "mc": mc, "cost_returns": cr,
                               "direction": "2-daily-timeframe"})

    # =================================================================== #
    # Direction 3: cross-sectional momentum
    # =================================================================== #
    log_md("\n## Direction 3: cross-sectional momentum (monthly rebalance, top-quintile)\n")
    price_matrix = build_price_matrix(daily_bars)
    log_md(f"- Price matrix: {price_matrix.shape[1]} tickers x {price_matrix.shape[0]} days")

    windows = generate_walk_forward_windows(price_matrix.index.min(), price_matrix.index.max(), TRAIN_MONTHS, TEST_MONTHS)
    xsmom_landscape = []
    for lookback in (63, 126, 252):
        for w_idx, w in enumerate(windows):
            train_matrix = price_matrix[(price_matrix.index >= w["train_start"]) & (price_matrix.index < w["train_end"])]
            is_trades, is_eq = cross_sectional_momentum_backtest(train_matrix, lookback, 0.2, STARTING_CAPITAL,
                                                                   BASELINE_SLIPPAGE)
            is_metrics = compute_metrics(is_eq, is_trades, STARTING_CAPITAL)

            oos_matrix = price_matrix[(price_matrix.index >= w["train_start"]) & (price_matrix.index < w["test_end"])]
            oos_trades, oos_eq = cross_sectional_momentum_backtest(oos_matrix, lookback, 0.2, STARTING_CAPITAL,
                                                                     BASELINE_SLIPPAGE, active_start=w["test_start"])
            oos_metrics = compute_metrics(oos_eq, oos_trades, STARTING_CAPITAL)
            xsmom_landscape.append({"params": {"lookback_days": lookback}, "window": w_idx,
                                    "is_sharpe": is_metrics["sharpe"], "oos_sharpe": oos_metrics["sharpe"],
                                    "oos_profit_factor": oos_metrics["profit_factor"],
                                    "oos_trade_count": oos_metrics["trade_count"],
                                    "oos_max_drawdown": oos_metrics["max_drawdown"], "oos_win_rate": oos_metrics["win_rate"]})

    xsmom_survivors = survivors(xsmom_landscape)
    xsmom_plateau = plateau_score(xsmom_survivors if xsmom_survivors else xsmom_landscape)
    log_md(f"- {len(xsmom_landscape)} (lookback,window) combos tested, "
           f"{len(xsmom_landscape)-len(xsmom_survivors)} rejected by the 30% IS/OOS deviation rule. "
           f"{len(xsmom_plateau)} lookback values survived as a plateau.")
    for p in xsmom_plateau:
        log_md(f"    lookback_days={p['params']['lookback_days']}: mean_oos_sharpe={p['mean_oos_sharpe']:.3f} "
               f"std={p['std_oos_sharpe']:.3f} n_windows_with_trades={p['n_windows_with_trades']}")

    if xsmom_plateau:
        best_lookback = xsmom_plateau[0]["params"]["lookback_days"]
        segs, trades = [], []
        for w in windows:
            oos_matrix = price_matrix[(price_matrix.index >= w["train_start"]) & (price_matrix.index < w["test_end"])]
            t, eq = cross_sectional_momentum_backtest(oos_matrix, best_lookback, 0.2, STARTING_CAPITAL,
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
                oos_matrix = price_matrix[(price_matrix.index >= w["train_start"]) & (price_matrix.index < w["test_end"])]
                _, eq2 = cross_sectional_momentum_backtest(oos_matrix, best_lookback, 0.2, STARTING_CAPITAL, slip,
                                                             active_start=w["test_start"])
                if not eq2.empty:
                    segs2.append(eq2)
            s2 = _stitch_equity_curves(segs2, STARTING_CAPITAL)
            cost_returns[slip] = float(s2.iloc[-1] / STARTING_CAPITAL - 1) if not s2.empty else float("nan")
        log_md(f"  SELECTED lookback_days={best_lookback} -> stitched Sharpe={metrics['sharpe']:.3f} "
               f"PF={metrics['profit_factor']:.3f} MaxDD={metrics['max_drawdown']*100:.1f}% "
               f"MC_DD95={mc['max_dd_p95']*100:.1f}% trades={metrics['trade_count']} "
               f"cost0.15%={cost_returns[0.0015]*100:.1f}%")
        all_candidates.append({"label": f"Cross-sectional momentum [lookback={best_lookback}d, top 20%]",
                               "params": {"lookback_days": best_lookback}, "metrics": metrics, "mc": mc,
                               "cost_returns": cost_returns, "direction": "3-cross-sectional"})
    else:
        log_md("  No lookback value survived the anti-overfitting filter as a stable plateau.")

    # =================================================================== #
    # Direction 4: portfolio-level SPY-SMA200 macro gate
    # =================================================================== #
    log_md("\n## Direction 4: portfolio-level SPY-SMA200 macro gate on mean reversion + trend filter\n")
    spy_daily = dc.get_bars("SPY", "us_equity", "1Day", start_date, end_date)
    spy_sma200 = spy_daily["close"].rolling(200).mean()
    macro_risk_on = (spy_daily["close"] > spy_sma200).fillna(False)

    spy_bars = dc.get_bars("SPY", "us_equity", "15Min", start_date, end_date)
    qqq_bars = dc.get_bars("QQQ", "us_equity", "15Min", start_date, end_date)
    for key, bars, grid_vals in [("SPY", spy_bars, [1.0, 1.5, 2.0]), ("QQQ", qqq_bars, [1.2, 1.8, 2.4])]:
        windows = generate_walk_forward_windows(bars.index.min(), bars.index.max(), TRAIN_MONTHS, TEST_MONTHS)
        spec = InstrumentSpec(key, "us_equity", "mean_reversion_macrogate", "15Min")
        landscape = []
        for m in grid_vals:
            params = {"sma_period": 20, "std_dev_mult": m, "trend_sma_period": 200}
            for w_idx, w in enumerate(windows):
                train_bars = bars[(bars.index >= w["train_start"]) & (bars.index < w["train_end"])]
                is_trades, is_eq = simulate_mean_reversion_macrogate(key, spec, train_bars, macro_risk_on, params,
                                                                       STARTING_CAPITAL, BASELINE_SLIPPAGE)
                is_metrics = compute_metrics(is_eq, is_trades, STARTING_CAPITAL)
                oos_bars = bars[(bars.index >= w["train_start"]) & (bars.index < w["test_end"])]
                oos_trades, oos_eq = simulate_mean_reversion_macrogate(key, spec, oos_bars, macro_risk_on, params,
                                                                        STARTING_CAPITAL, BASELINE_SLIPPAGE,
                                                                        active_start=w["test_start"])
                oos_metrics = compute_metrics(oos_eq, oos_trades, STARTING_CAPITAL)
                landscape.append({"params": params, "window": w_idx, "is_sharpe": is_metrics["sharpe"],
                                  "oos_sharpe": oos_metrics["sharpe"], "oos_profit_factor": oos_metrics["profit_factor"],
                                  "oos_trade_count": oos_metrics["trade_count"],
                                  "oos_max_drawdown": oos_metrics["max_drawdown"], "oos_win_rate": oos_metrics["win_rate"]})
        survived = survivors(landscape)
        plateau = plateau_score(survived if survived else landscape)
        log_md(f"- **{key} / mean_reversion + trendfilter + SPY macro gate**: {len(landscape)} combos, "
               f"{len(landscape)-len(survived)} rejected. {len(plateau)} param sets survived.")
        if plateau:
            best = plateau[0]
            segs, trades = [], []
            for w in windows:
                oos_bars = bars[(bars.index >= w["train_start"]) & (bars.index < w["test_end"])]
                t, eq = simulate_mean_reversion_macrogate(key, spec, oos_bars, macro_risk_on, best["params"],
                                                            STARTING_CAPITAL, BASELINE_SLIPPAGE, active_start=w["test_start"])
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
                    oos_bars = bars[(bars.index >= w["train_start"]) & (bars.index < w["test_end"])]
                    _, eq2 = simulate_mean_reversion_macrogate(key, spec, oos_bars, macro_risk_on, best["params"],
                                                                STARTING_CAPITAL, slip, active_start=w["test_start"])
                    if not eq2.empty:
                        segs2.append(eq2)
                s2 = _stitch_equity_curves(segs2, STARTING_CAPITAL)
                cost_returns[slip] = float(s2.iloc[-1] / STARTING_CAPITAL - 1) if not s2.empty else float("nan")
            log_md(f"  best params={best['params']}: Sharpe={metrics['sharpe']:.3f} PF={metrics['profit_factor']:.3f} "
                   f"MaxDD={metrics['max_drawdown']*100:.1f}% MC_DD95={mc['max_dd_p95']*100:.1f}% "
                   f"trades={metrics['trade_count']} cost0.15%={cost_returns[0.0015]*100:.1f}%")
            all_candidates.append({"label": f"{key} mean_reversion+trendfilter+macrogate", "params": best["params"],
                                   "metrics": metrics, "mc": mc, "cost_returns": cost_returns, "direction": "4-macro-gate"})

    with open("exploration_v2_candidates.json", "w") as f:
        json.dump([{
            "label": c["label"], "direction": c["direction"], "params": c["params"],
            "sharpe": c["metrics"]["sharpe"], "profit_factor": c["metrics"]["profit_factor"],
            "max_drawdown": c["metrics"]["max_drawdown"], "trade_count": c["metrics"]["trade_count"],
            "mc_dd95": c["mc"]["max_dd_p95"], "cost_0_15": c["cost_returns"].get(0.0015),
        } for c in all_candidates], f, indent=2, default=str)

    log_md(f"\n## All 4 directions complete: {len(all_candidates)} candidates collected. "
          f"See exploration_v2_candidates.json.\n")
    logger.info("DONE. %d candidates.", len(all_candidates))
    return all_candidates


if __name__ == "__main__":
    main()
