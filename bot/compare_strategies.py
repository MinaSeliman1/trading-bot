"""
Orchestrates the objective Strategy A (baseline) vs Strategy B (regime +
covariance + vol targeting) comparison: same data, same walk-forward split
dates, same cost scenarios, same starting capital, both walk-forward
optimized on their own parameters. Produces comparison_report.md and
comparison_equity_curves.png.

Historical backtest only -- no paper trading is started here.
"""
import logging
import sys
from datetime import datetime, timedelta, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from bot.data_client import DataClient
from bot.backtest_engine import (
    InstrumentSpec, generate_walk_forward_windows, optimize_parameters, simulate_portfolio,
    compute_metrics, compute_daily_covariance_schedule, monte_carlo_bootstrap,
    paired_bootstrap_significance, classify_regime_periods, regime_breakdown,
    daily_returns_from_equity,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("compare_strategies")

STARTING_CAPITAL = 100000.0
BASELINE_SLIPPAGE = 0.0010
SLIPPAGE_SCENARIOS = [0.0005, 0.0010, 0.0015]
YEARS_OF_HISTORY = 4
TRAIN_MONTHS = 12
TEST_MONTHS = 3
MC_RESAMPLES = 1000
BOOTSTRAP_RESAMPLES = 1000
# Strategy A has no regime filter; without a cooldown it re-enters immediately
# after a stop-out if price is still beyond the signal threshold -- the exact
# whipsaw pattern diagnosed against real SPY/QQQ data. Strategy B gets
# equivalent protection from regime gating, so only A needs this to make the
# comparison fair. Fixed, not walk-forward optimized (deliberately -- this is
# a fairness correction, not a tunable edge).
STRATEGY_A_STOP_COOLDOWN_BARS = 5

INSTRUMENTS = {
    "SPY": InstrumentSpec("SPY", "us_equity", "mean_reversion", "15Min"),
    "QQQ": InstrumentSpec("QQQ", "us_equity", "mean_reversion", "15Min"),
    "BTCUSD": InstrumentSpec("BTC/USD", "crypto", "momentum_breakout", "1Hour"),
    "GLD": InstrumentSpec("GLD", "us_equity", "trend_following", "4Hour"),
    "USO": InstrumentSpec("USO", "us_equity", "trend_following", "4Hour"),
}

PARAM_GRIDS = {
    "SPY": [{"sma_period": 20, "std_dev_mult": m} for m in (1.0, 1.25, 1.5, 1.75, 2.0)],
    "QQQ": [{"sma_period": 20, "std_dev_mult": m} for m in (1.2, 1.5, 1.8, 2.1, 2.4)],
    "BTCUSD": [{"lookback": 20, "volume_mult": 1.5, "trailing_atr_mult": m} for m in (1.5, 2.0, 2.5, 3.0)],
    "GLD": [{"fast_ema": 50, "slow_ema": 200, "trailing_atr_mult": m} for m in (2.0, 2.5, 3.0, 3.5)],
    "USO": [{"fast_ema": 50, "slow_ema": 200, "trailing_atr_mult": m} for m in (2.0, 2.5, 3.0, 3.5)],
}


def fetch_all_data(dc: DataClient, end_date: datetime):
    start_date = end_date - timedelta(days=365 * YEARS_OF_HISTORY + 30)
    bars_by_key, daily_by_key = {}, {}
    for key, spec in INSTRUMENTS.items():
        logger.info("Fetching %s %s bars for %s ...", spec.symbol, spec.timeframe, key)
        bars = dc.get_bars(spec.symbol, spec.asset_class, spec.timeframe, start_date, end_date)
        bars_by_key[key] = bars
        logger.info("  -> %d bars (%s to %s)", len(bars), bars.index.min() if len(bars) else None,
                    bars.index.max() if len(bars) else None)

        daily = dc.get_bars(spec.symbol, spec.asset_class, "1Day", start_date, end_date)
        daily_by_key[key] = daily
    return bars_by_key, daily_by_key


def build_daily_returns_matrix(daily_by_key: dict) -> pd.DataFrame:
    closes = {k: v["close"] for k, v in daily_by_key.items() if not v.empty}
    master_index = closes["SPY"].index
    aligned = pd.DataFrame({k: v.reindex(master_index, method="ffill") for k, v in closes.items()})
    return aligned.pct_change().dropna(how="all")


def _stitch_equity_curves(segments: list, starting_capital: float) -> pd.Series:
    chained, carry = [], starting_capital
    for seg in segments:
        if seg.empty:
            continue
        normalized = seg / seg.iloc[0] * carry
        chained.append(normalized)
        carry = normalized.iloc[-1]
    if not chained:
        return pd.Series(dtype=float)
    return pd.concat(chained).sort_index()


def _window_bars(bars_by_key: dict, start, end) -> dict:
    return {key: bars[(bars.index >= start) & (bars.index < end)] for key, bars in bars_by_key.items()}


def run_walk_forward(bars_by_key: dict, cov_schedule: dict, windows: list) -> dict:
    window_results = []
    oos_a_segments, oos_b_segments = [], []
    oos_trades_a, oos_trades_b = [], []

    for w_idx, w in enumerate(windows):
        logger.info("Window %d/%d: train %s->%s, test %s->%s", w_idx + 1, len(windows),
                    w["train_start"].date(), w["train_end"].date(), w["test_start"].date(), w["test_end"].date())

        params_a, params_b = {}, {}
        for key, spec in INSTRUMENTS.items():
            grid = PARAM_GRIDS[key]
            params_a[key] = optimize_parameters(key, spec, bars_by_key[key], spec.strategy_kind, grid,
                                                 w["train_start"], w["train_end"], BASELINE_SLIPPAGE,
                                                 STARTING_CAPITAL, use_regime=False,
                                                 stop_cooldown_bars=STRATEGY_A_STOP_COOLDOWN_BARS)
            params_b[key] = optimize_parameters(key, spec, bars_by_key[key], spec.strategy_kind, grid,
                                                 w["train_start"], w["train_end"], BASELINE_SLIPPAGE,
                                                 STARTING_CAPITAL, use_regime=True)

        train_bars = _window_bars(bars_by_key, w["train_start"], w["train_end"])
        # OOS runs get the FULL train+test span so slow indicators (e.g. a 200-period EMA
        # on 4-hour GLD/USO bars) are already warmed up by test_start, instead of cold-starting
        # at the window boundary where a 3-month test window may not contain 200 bars at all.
        oos_warmup_bars = _window_bars(bars_by_key, w["train_start"], w["test_end"])

        res_a_is = simulate_portfolio("A", INSTRUMENTS, train_bars, params_a, STARTING_CAPITAL, BASELINE_SLIPPAGE,
                                       stop_cooldown_bars=STRATEGY_A_STOP_COOLDOWN_BARS)
        res_a_oos = simulate_portfolio("A", INSTRUMENTS, oos_warmup_bars, params_a, STARTING_CAPITAL, BASELINE_SLIPPAGE,
                                        active_start=w["test_start"], stop_cooldown_bars=STRATEGY_A_STOP_COOLDOWN_BARS)
        res_b_is = simulate_portfolio("B", INSTRUMENTS, train_bars, params_b, STARTING_CAPITAL, BASELINE_SLIPPAGE,
                                       daily_cov_schedule=cov_schedule)
        res_b_oos = simulate_portfolio("B", INSTRUMENTS, oos_warmup_bars, params_b, STARTING_CAPITAL, BASELINE_SLIPPAGE,
                                        daily_cov_schedule=cov_schedule, active_start=w["test_start"])

        window_results.append({
            "window": w_idx, **w, "params_a": params_a, "params_b": params_b,
            "a_is": compute_metrics(res_a_is["equity"], res_a_is["trades"], STARTING_CAPITAL),
            "a_oos": compute_metrics(res_a_oos["equity"], res_a_oos["trades"], STARTING_CAPITAL),
            "b_is": compute_metrics(res_b_is["equity"], res_b_is["trades"], STARTING_CAPITAL),
            "b_oos": compute_metrics(res_b_oos["equity"], res_b_oos["trades"], STARTING_CAPITAL),
        })

        if not res_a_oos["equity"].empty:
            oos_a_segments.append(res_a_oos["equity"])
        if not res_b_oos["equity"].empty:
            oos_b_segments.append(res_b_oos["equity"])
        oos_trades_a.extend(res_a_oos["trades"])
        oos_trades_b.extend(res_b_oos["trades"])

    return {
        "window_results": window_results,
        "stitched_a": _stitch_equity_curves(oos_a_segments, STARTING_CAPITAL),
        "stitched_b": _stitch_equity_curves(oos_b_segments, STARTING_CAPITAL),
        "oos_trades_a": oos_trades_a,
        "oos_trades_b": oos_trades_b,
    }


def run_cost_sensitivity(window_results: list, bars_by_key: dict, cov_schedule: dict) -> list:
    rows = []
    for slip in SLIPPAGE_SCENARIOS:
        eq_a_segments, eq_b_segments = [], []
        for wr in window_results:
            oos_warmup_bars = _window_bars(bars_by_key, wr["train_start"], wr["test_end"])
            res_a = simulate_portfolio("A", INSTRUMENTS, oos_warmup_bars, wr["params_a"], STARTING_CAPITAL, slip,
                                        active_start=wr["test_start"], stop_cooldown_bars=STRATEGY_A_STOP_COOLDOWN_BARS)
            res_b = simulate_portfolio("B", INSTRUMENTS, oos_warmup_bars, wr["params_b"], STARTING_CAPITAL, slip,
                                        daily_cov_schedule=cov_schedule, active_start=wr["test_start"])
            if not res_a["equity"].empty:
                eq_a_segments.append(res_a["equity"])
            if not res_b["equity"].empty:
                eq_b_segments.append(res_b["equity"])
        stitched_a = _stitch_equity_curves(eq_a_segments, STARTING_CAPITAL)
        stitched_b = _stitch_equity_curves(eq_b_segments, STARTING_CAPITAL)
        ret_a = float(stitched_a.iloc[-1] / STARTING_CAPITAL - 1) if not stitched_a.empty else float("nan")
        ret_b = float(stitched_b.iloc[-1] / STARTING_CAPITAL - 1) if not stitched_b.empty else float("nan")
        rows.append({"slippage_pct": slip, "total_return_a": ret_a, "total_return_b": ret_b})
        logger.info("Cost scenario %.2f%%: A total return %.2f%%, B total return %.2f%%",
                    slip * 100, ret_a * 100, ret_b * 100)
    return rows


def plot_equity_curves(stitched_a: pd.Series, stitched_b: pd.Series, path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    if not stitched_a.empty:
        ax.plot(stitched_a.index, stitched_a.values, label="Strategy A (baseline)", linewidth=1.4)
    if not stitched_b.empty:
        ax.plot(stitched_b.index, stitched_b.values, label="Strategy B (regime + covariance)", linewidth=1.4)
    ax.set_title("Out-of-sample equity curves (stitched across walk-forward windows)")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _param_instability(window_results: list, key: str, strategy: str, param_name: str) -> bool:
    values = [wr[f"params_{strategy}"][key].get(param_name) for wr in window_results]
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return False
    grid_values = sorted({p.get(param_name) for p in PARAM_GRIDS[key] if param_name in p})
    if len(grid_values) < 2:
        return False
    span = max(values) - min(values)
    full_span = grid_values[-1] - grid_values[0]
    return full_span > 0 and (span / full_span) > 0.75


def run_comparison():
    if not config.ALPACA_API_KEY or not config.ALPACA_API_SECRET:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET in .env")

    dc = DataClient(config.ALPACA_API_KEY, config.ALPACA_API_SECRET, config.ALPACA_DATA_FEED)
    end_date = datetime.now(timezone.utc) - timedelta(minutes=20)

    bars_by_key, daily_by_key = fetch_all_data(dc, end_date)
    for key, bars in bars_by_key.items():
        if bars.empty:
            raise RuntimeError(f"No data returned for {key}; aborting comparison")

    daily_returns = build_daily_returns_matrix(daily_by_key)
    logger.info("Building daily covariance schedule (Ledoit-Wolf shrinkage, 60-day lookback)...")
    cov_schedule = compute_daily_covariance_schedule(daily_returns, spy_col="SPY")
    logger.info("Covariance schedule: %d daily snapshots", len(cov_schedule))

    overall_start = max(b.index.min() for b in bars_by_key.values())
    overall_end = min(b.index.max() for b in bars_by_key.values())
    windows = generate_walk_forward_windows(overall_start, overall_end, TRAIN_MONTHS, TEST_MONTHS)
    logger.info("Generated %d walk-forward windows (%s -> %s)", len(windows), overall_start.date(), overall_end.date())
    if not windows:
        raise RuntimeError("Not enough historical data to form even one walk-forward window")

    wf = run_walk_forward(bars_by_key, cov_schedule, windows)
    window_results, stitched_a, stitched_b = wf["window_results"], wf["stitched_a"], wf["stitched_b"]
    oos_trades_a, oos_trades_b = wf["oos_trades_a"], wf["oos_trades_b"]

    logger.info("Running cost-sensitivity sweep across slippage scenarios...")
    cost_rows = run_cost_sensitivity(window_results, bars_by_key, cov_schedule)

    logger.info("Running Monte Carlo bootstrap (%d resamples) for both strategies...", MC_RESAMPLES)
    mc_a = monte_carlo_bootstrap(oos_trades_a, STARTING_CAPITAL, n_resamples=MC_RESAMPLES, seed=42)
    mc_b = monte_carlo_bootstrap(oos_trades_b, STARTING_CAPITAL, n_resamples=MC_RESAMPLES, seed=42)

    logger.info("Running paired block-bootstrap significance test...")
    daily_ret_a = daily_returns_from_equity(stitched_a)
    daily_ret_b = daily_returns_from_equity(stitched_b)
    significance = paired_bootstrap_significance(daily_ret_a, daily_ret_b, n_resamples=BOOTSTRAP_RESAMPLES,
                                                  block_size=10, seed=7)

    logger.info("Classifying regime periods and attributing performance...")
    regime_labels = classify_regime_periods(daily_by_key["SPY"])
    regime_a = regime_breakdown(stitched_a, oos_trades_a, regime_labels)
    regime_b = regime_breakdown(stitched_b, oos_trades_b, regime_labels)

    overall_a = compute_metrics(stitched_a, oos_trades_a, STARTING_CAPITAL)
    overall_b = compute_metrics(stitched_b, oos_trades_b, STARTING_CAPITAL)

    plot_equity_curves(stitched_a, stitched_b, "comparison_equity_curves.png")
    logger.info("Saved comparison_equity_curves.png")

    write_report(
        window_results=window_results, overall_a=overall_a, overall_b=overall_b,
        mc_a=mc_a, mc_b=mc_b, significance=significance, regime_a=regime_a, regime_b=regime_b,
        cost_rows=cost_rows, data_range=(overall_start, overall_end), n_windows=len(windows),
    )
    logger.info("Saved comparison_report.md")

    return {
        "overall_a": overall_a, "overall_b": overall_b, "mc_a": mc_a, "mc_b": mc_b,
        "significance": significance, "regime_a": regime_a, "regime_b": regime_b, "cost_rows": cost_rows,
    }


def _fmt(x, pct=False, digits=3):
    if x is None or x != x:
        return "n/a"
    if x == float("inf"):
        return "inf"
    return f"{x*100:.{digits}f}%" if pct else f"{x:.{digits}f}"


def write_report(window_results, overall_a, overall_b, mc_a, mc_b, significance, regime_a, regime_b,
                  cost_rows, data_range, n_windows):
    lines = []
    lines.append("# Strategy A (baseline) vs Strategy B (regime + covariance + vol targeting)\n")
    lines.append(f"Data range: {data_range[0].date()} to {data_range[1].date()} "
                 f"({n_windows} walk-forward windows, {TRAIN_MONTHS}mo train / {TEST_MONTHS}mo test, rolling forward "
                 f"by {TEST_MONTHS}mo). Starting capital: ${STARTING_CAPITAL:,.0f}. "
                 f"Baseline cost assumption: {BASELINE_SLIPPAGE*100:.2f}% slippage.\n")

    lines.append("## Methodology and key assumptions\n")
    lines.append(
        "- Both strategies pass through the identical pipeline: same cached historical data, same walk-forward "
        "split dates, same starting capital, same cost scenarios.\n"
        "- Strategy A and Strategy B each optimize their OWN parameters independently per window (std-dev "
        "threshold for mean reversion, ATR trailing multiplier for momentum/trend) via in-sample grid search; "
        "ADX regime cutoffs (20/25) are fixed, matching the original walk-forward spec.\n"
        "- Parameter thresholds are optimized on a single-instrument backtest with a common ATR-1%-risk sizing "
        "proxy (both strategies), to keep the grid search tractable across 5 instruments x many windows. Final "
        "performance figures use each strategy's REAL sizing logic (A: fixed ATR-risk + binary correlation "
        "filter; B: volatility targeting + covariance-aware sizing + kill switches) in a full shared-equity "
        "portfolio simulation.\n"
        "- Parameters are optimized once at the baseline 0.10% slippage; the 0.05%/0.10%/0.15% cost scenarios "
        "are a sensitivity pass over those fixed parameters, not independently re-optimized.\n"
        "- BTC/USD momentum breakout never shorts in either strategy -- Alpaca does not support shorting crypto, "
        "so both strategies are held to that real constraint identically.\n"
        "- Strategy B's volatility-targeting scale-up cap is modeled at its post-validation steady-state (1.5x), "
        "not the conservative 1.0x-only cap mandated for a live system's first 4-6 weeks -- a historical backtest "
        "has no \"early rollout\" period.\n"
        "- Stops are checked against bar close, not intrabar high/low, consistent with how the live bot polls.\n"
        "- Strategy B's kill switches (drawdown halt, vol-breach size cut) are active in this backtest since "
        "they're part of the Phase 1-6 system as specified. This means part of any drawdown improvement B shows "
        "could come from the kill switches rather than from regime/covariance intelligence specifically -- a "
        "confound worth isolating via an ablation run if the headline result here looks close.\n"
    )

    lines.append("## Summary: out-of-sample, stitched across all walk-forward windows\n")
    lines.append("| Metric | Strategy A | Strategy B |")
    lines.append("|---|---|---|")
    for label, k, pct in [
        ("Sharpe", "sharpe", False), ("Sortino", "sortino", False),
        ("Max drawdown", "max_drawdown", True), ("Recovery (days)", "recovery_days", False),
        ("Win rate", "win_rate", True), ("Profit factor", "profit_factor", False),
        ("Trade count", "trade_count", False), ("Total return", "total_return", True),
        ("Final equity", "final_equity", False),
    ]:
        va, vb = overall_a.get(k), overall_b.get(k)
        lines.append(f"| {label} | {_fmt(va, pct)} | {_fmt(vb, pct)} |")
    lines.append("")

    lines.append("## In-sample vs out-of-sample Sharpe per window\n")
    lines.append("| Window | Train | Test | A IS | A OOS | B IS | B OOS | A OOS degradation | B OOS degradation |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for wr in window_results:
        a_is, a_oos = wr["a_is"]["sharpe"], wr["a_oos"]["sharpe"]
        b_is, b_oos = wr["b_is"]["sharpe"], wr["b_oos"]["sharpe"]
        a_flag = "FLAG" if (a_oos == a_oos and a_oos < 0.5) else ""
        b_flag = "FLAG" if (b_oos == b_oos and b_oos < 0.5) else ""
        lines.append(f"| {wr['window']} | {wr['train_start'].date()} | {wr['test_start'].date()} | "
                     f"{_fmt(a_is)} | {_fmt(a_oos)} | {_fmt(b_is)} | {_fmt(b_oos)} | {a_flag} | {b_flag} |")
    lines.append("")

    unstable = []
    for key in INSTRUMENTS:
        param_name = "std_dev_mult" if key in ("SPY", "QQQ") else "trailing_atr_mult"
        if _param_instability(window_results, key, "a", param_name):
            unstable.append(f"A/{key}")
        if _param_instability(window_results, key, "b", param_name):
            unstable.append(f"B/{key}")
    lines.append(f"**Parameter stability**: {'unstable optimal parameters across windows for: ' + ', '.join(unstable) if unstable else 'optimal parameters were reasonably stable across windows for both strategies.'}\n")

    lines.append("## Monte Carlo (1000 trade-sequence resamples, out-of-sample trades)\n")
    lines.append("| | Strategy A | Strategy B |")
    lines.append("|---|---|---|")
    lines.append(f"| Sharpe 5th pct | {_fmt(mc_a['sharpe_p5'])} | {_fmt(mc_b['sharpe_p5'])} |")
    lines.append(f"| Sharpe median | {_fmt(mc_a['sharpe_p50'])} | {_fmt(mc_b['sharpe_p50'])} |")
    lines.append(f"| Sharpe 95th pct | {_fmt(mc_a['sharpe_p95'])} | {_fmt(mc_b['sharpe_p95'])} |")
    lines.append(f"| Max DD 5th pct | {_fmt(mc_a['max_dd_p5'], pct=True)} | {_fmt(mc_b['max_dd_p5'], pct=True)} |")
    lines.append(f"| Max DD median | {_fmt(mc_a['max_dd_p50'], pct=True)} | {_fmt(mc_b['max_dd_p50'], pct=True)} |")
    lines.append(f"| Max DD 95th pct | {_fmt(mc_a['max_dd_p95'], pct=True)} | {_fmt(mc_b['max_dd_p95'], pct=True)} |")
    lines.append(f"| N trades resampled | {mc_a['n_trades']} | {mc_b['n_trades']} |")
    lines.append("")
    not_viable_a = mc_a["max_dd_p95"] == mc_a["max_dd_p95"] and mc_a["max_dd_p95"] > 0.20
    not_viable_b = mc_b["max_dd_p95"] == mc_b["max_dd_p95"] and mc_b["max_dd_p95"] > 0.20
    lines.append(f"**Viability flag (95th pct MC drawdown > 20%)**: A={'FLAG' if not_viable_a else 'ok'}, B={'FLAG' if not_viable_b else 'ok'}\n")

    lines.append("## Statistical significance: paired block bootstrap on daily returns\n")
    lines.append(f"- Observed annualized Sharpe: A = {_fmt(significance['observed_sharpe_a'])}, "
                 f"B = {_fmt(significance['observed_sharpe_b'])}\n"
                 f"- Observed Sharpe difference (B - A): {_fmt(significance['observed_diff'])}\n"
                 f"- 95% bootstrap CI on the difference: [{_fmt(significance['ci_low'])}, {_fmt(significance['ci_high'])}]\n"
                 f"- Two-sided p-value: {_fmt(significance['p_value'], digits=4)}\n"
                 f"- **Statistically significant at 5%: {'YES' if significance['significant_at_5pct'] else 'NO'}** "
                 f"(n={significance['n_obs']} paired daily observations, {BOOTSTRAP_RESAMPLES} resamples, block size 10 days)\n")

    lines.append("## Performance by regime period (SPY-derived: bullish / correction / vol shock / neutral)\n")
    lines.append("| Regime | Days | A return | A Sharpe | A trades | B return | B Sharpe | B trades |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for regime in ["BULLISH", "CORRECTION", "VOL_SHOCK", "NEUTRAL"]:
        ra, rb = regime_a[regime], regime_b[regime]
        lines.append(f"| {regime} | {ra['days']} | {_fmt(ra['total_return'], pct=True)} | {_fmt(ra['sharpe'])} | "
                     f"{ra['trade_count']} | {_fmt(rb['total_return'], pct=True)} | {_fmt(rb['sharpe'])} | {rb['trade_count']} |")
    lines.append("")

    lines.append("## Transaction cost sensitivity (out-of-sample total return, fixed walk-forward parameters)\n")
    lines.append("| Slippage | A total return | B total return |")
    lines.append("|---|---|---|")
    for row in cost_rows:
        lines.append(f"| {row['slippage_pct']*100:.2f}% | {_fmt(row['total_return_a'], pct=True)} | "
                     f"{_fmt(row['total_return_b'], pct=True)} |")
    lines.append("")
    edge_gone_a = cost_rows[-1]["total_return_a"] == cost_rows[-1]["total_return_a"] and cost_rows[-1]["total_return_a"] < 0
    edge_gone_b = cost_rows[-1]["total_return_b"] == cost_rows[-1]["total_return_b"] and cost_rows[-1]["total_return_b"] < 0
    lines.append(f"**Edge survives 0.15% slippage**: A={'NO' if edge_gone_a else 'yes'}, B={'NO' if edge_gone_b else 'yes'}\n")

    lines.append("## Conclusion\n")
    lines.append(_write_conclusion(overall_a, overall_b, significance, mc_a, mc_b))

    lines.append("\n![Equity curves](comparison_equity_curves.png)\n")

    with open("comparison_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_conclusion(overall_a, overall_b, significance, mc_a, mc_b) -> str:
    sharpe_a, sharpe_b = overall_a.get("sharpe"), overall_b.get("sharpe")
    dd_a, dd_b = overall_a.get("max_drawdown"), overall_b.get("max_drawdown")
    significant = significance.get("significant_at_5pct")
    diff = significance.get("observed_diff")

    parts = []
    if sharpe_a == sharpe_a and sharpe_b == sharpe_b:
        better = "B" if sharpe_b > sharpe_a else "A"
        parts.append(f"Out-of-sample, Strategy {better} shows the higher Sharpe ratio "
                     f"(A={_fmt(sharpe_a)}, B={_fmt(sharpe_b)}).")
    if dd_a == dd_a and dd_b == dd_b:
        safer = "B" if dd_b < dd_a else "A"
        parts.append(f"Strategy {safer} shows the smaller max drawdown (A={_fmt(dd_a, pct=True)}, B={_fmt(dd_b, pct=True)}).")

    if not significant:
        parts.append(
            "However, the paired bootstrap significance test does NOT reject the null hypothesis that A and B "
            "perform the same -- the 95% confidence interval on the Sharpe difference includes zero. On this "
            "data, the gap between the two strategies is not distinguishable from noise. "
            "**The added complexity of Strategy B (regime detection, covariance shrinkage, volatility "
            "targeting, kill switches) is NOT clearly justified by a measurable risk-adjusted return "
            "improvement over the simpler baseline** -- Strategy A delivers comparable performance with "
            "substantially less engineering risk, fewer moving parts to maintain, and fewer ways to silently "
            "misbehave in production."
        )
    else:
        direction = "in favor of B" if diff > 0 else "in favor of A"
        parts.append(
            f"The paired bootstrap significance test DOES reject the null hypothesis at the 5% level, "
            f"{direction} -- the gap between strategies on this data is unlikely to be pure noise. "
            + ("This is evidence the added complexity of Strategy B is earning its keep on a risk-adjusted "
               "basis, though the regime-breakdown and Monte Carlo tables above should be checked to confirm "
               "the edge isn't concentrated in a single regime or a lucky trade cluster before trusting it "
               "going forward."
               if diff > 0 else
               "This is evidence the added complexity of Strategy B is NOT paying for itself on this data -- "
               "the simpler baseline outperformed with less engineering risk."))

    return " ".join(parts)


if __name__ == "__main__":
    run_comparison()
