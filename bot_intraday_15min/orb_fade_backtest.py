"""
bot_intraday_15min/orb_fade_backtest.py
=========================================
Rigorous backtest of the "Opening Range Fade" 15-minute intraday strategy,
following the same validation standard used for PEAD and value/quality:

  - Multi-ticker (not single-symbol curve-fit)
  - Walk-forward split (train window vs out-of-sample window)
  - Monte Carlo resampling for drawdown estimate (MC_DD95)
  - Explicit regime check where data allows
  - Transaction cost / slippage stress test

STRATEGY LOGIC (validated informally on SPY, 60 days, free data — see
exploration notes):
  1. For each ticker, each day: the first 15-minute candle after the open
     defines a range [orb_low, orb_high].
  2. If price later breaks ABOVE orb_high -> enter SHORT (fade the breakout).
     If price later breaks BELOW orb_low  -> enter LONG (fade the breakout).
  3. Stop loss = 1 range-width beyond entry. Take profit = TP_MULT range-widths
     back toward/through the opening range.
  4. Max one trade per ticker per day. Close at end of day if neither hit.

DATA SOURCE: Alpaca minute bars, resampled to 15-min bars, via the existing
bot/data_client.py wrapper (reuse the same client used for PEAD/value-quality
so caching and rate-limit handling stay consistent).

KNOWN DATA LIMITATION (see lessons_learned.md, piège #7): Alpaca IEX feed has
no reliable individual-stock intraday data before mid-2020. This backtest
will therefore only cover regimes from mid-2020 onward unless a SIP feed
subscription is available. COVID crash (Mar 2020) may be partially in range;
the 2018 correction is NOT reachable with this data source for intraday bars.
This must be stated explicitly in any viability conclusion — do not claim
multi-regime validation you cannot actually support with the data used.

USAGE:
  Run inside the Claude Code Desktop project (same folder structure as
  bot/pead_backtest.py). Requires bot/data_client.py and .env with Alpaca keys
  already configured (both already exist in this project).

  python bot_intraday_15min/orb_fade_backtest.py

STOPPING RULES (same discipline as PEAD/value-quality — copy into any
scheduled routine prompt if this is run unattended):
  - Do NOT wire into bot/main.py or any paper/live trading regardless of
    result quality, without explicit user approval.
  - If Sharpe OOS < 0.5 or MC_DD95 > 20%, or the walk-forward split shows
    a sign flip (positive in-sample, negative out-of-sample), document as
    REJECTED in exploration_log_intraday_orb.md and stop — do not keep
    re-tuning parameters to force a pass (see piège #4: argmax selection
    bias performs worse than random OOS).
"""

import os

# MUST be set before bot.data_client is first imported -- isolates this
# script's 1-min bar cache from the shared data_cache/ used by PEAD/
# value-quality (which cache different timeframes for the same symbols;
# a dedicated directory avoids any provenance ambiguity, same pattern as
# run_value_quality_viability.py).
os.environ.setdefault("DATA_CACHE_DIR", "data_cache_intraday")

from datetime import datetime, time as dtime, timedelta, timezone

import numpy as np
import pandas as pd

import config
from bot.data_client import DataClient

# Wired to the real bot/data_client.py interface (verified against the
# actual file, not assumed): DataClient.get_bars(symbol, asset_class,
# timeframe, start, end, use_cache=True). "1Min" was NOT previously
# supported by _TIMEFRAME_MAP/_TIMEFRAME_SECONDS -- added there (additive
# change, does not affect existing "15Min"/"1Hour"/"4Hour"/"1Day" callers).
#
# FEED CHOICE, VERIFIED LIVE (not assumed from the file's original
# docstring): bisected actual Alpaca minute-bar depth before writing this.
#   - IEX (this project's default feed): 0 rows for SPY/AAPL on every
#     sampled weekday in 2018-2020; real data from January 2021 onward.
#     So the docstring's "mid-2020" estimate was in the right ballpark but
#     not exact -- the true IEX floor is ~Jan 2021, verified here.
#   - SIP: real minute-bar data back to January 2016 (matches this
#     project's already-established SIP floor for DAILY bars in PEAD/
#     value-quality -- same account-level floor, not timeframe-specific).
# This script uses SIP for the same reason PEAD's 2016-2020 test and the
# value/quality backtest do: it is the only feed with enough depth to test
# more than one market regime. See exploration_log_intraday_orb.md for the
# full verification and what this changes vs. the user's original request
# (which assumed only mid-2020-onward was reachable).
PRICE_FEED = "sip"

TICKERS = [
    # Reuse the same liquid, large-cap universe already validated for data
    # availability in the PEAD/value-quality work, to avoid re-discovering
    # data quality issues from scratch.
    "SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA",
    "JPM", "CAT", "XOM", "JNJ",
]

TP_MULTIPLES = [1.0, 1.5, 2.0]
COMMISSION_BPS = 0.5  # as originally coded -- optimistic single-cost assumption
COST_SENSITIVITY_BPS = [5.0, 10.0, 15.0]  # this project's standard cost-stress convention (0.05/0.10/0.15%)
LOOKBACK_DAYS = 730  # ~2 years; primary walk-forward window, per the original plan
WALK_FORWARD_SPLIT = 0.6  # 60% train / 40% out-of-sample, chronological
N_MONTE_CARLO = 2000
# Bonus regime check (see PRICE_FEED comment above) -- NOT part of the primary
# 60/40 walk-forward, evaluated separately, does not affect the primary
# verdict's stop-rule decision.
Q4_2018_WINDOW = (datetime(2018, 8, 1, tzinfo=timezone.utc), datetime(2019, 2, 1, tzinfo=timezone.utc))


def resample_to_15min(minute_df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min OHLCV bars to 15-min bars. Caller must already have
    filtered to regular market hours -- this function only bins time, it
    does not know which bars are pre/post-market."""
    ohlc = {
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }
    df = minute_df.resample("15min", origin="start_day").agg(ohlc).dropna()
    return df


def fetch_15min_bars(dc: DataClient, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetches 1-min bars via bot/data_client.py (SIP feed, see PRICE_FEED),
    filters to regular trading hours (9:30-16:00 America/New_York -- Alpaca
    returns extended-hours bars too, which would corrupt the opening-range
    definition if left in), then resamples to 15-min. 9:30 ET = 570 minutes
    after local midnight = an exact multiple of 15, so `origin="start_day"`
    bins align exactly on the real market open once bars are in local time."""
    minute_bars = dc.get_bars(symbol, "us_equity", "1Min", start, end)
    if minute_bars.empty:
        return pd.DataFrame()
    local = minute_bars.tz_convert("America/New_York")
    times = local.index.time
    regular_hours = (times >= dtime(9, 30)) & (times <= dtime(15, 59))
    local = local[regular_hours]
    if local.empty:
        return pd.DataFrame()
    return resample_to_15min(local)


def run_orb_fade(df: pd.DataFrame, tp_mult: float, commission_bps=COMMISSION_BPS) -> pd.DataFrame:
    """Same core logic validated in the SPY prototype, generalized per ticker."""
    trades = []
    df = df.copy()
    df["date"] = df.index.date
    for day, day_df in df.groupby("date"):
        day_df = day_df.sort_index()
        if len(day_df) < 3:
            continue
        opening = day_df.iloc[0]
        orb_high, orb_low = opening["high"], opening["low"]
        orb_range = orb_high - orb_low
        if orb_range <= 0:
            continue

        entered = False
        direction = entry_price = stop = target = None
        for idx, row in day_df.iloc[1:].iterrows():
            if not entered:
                if row["high"] > orb_high:
                    direction, entry_price = "short", orb_high
                    entered = True
                elif row["low"] < orb_low:
                    direction, entry_price = "long", orb_low
                    entered = True
                else:
                    continue

                if direction == "long":
                    stop = entry_price - orb_range
                    target = entry_price + tp_mult * orb_range
                else:
                    stop = entry_price + orb_range
                    target = entry_price - tp_mult * orb_range
                continue

            exit_price = exit_reason = None
            if direction == "long":
                if row["low"] <= stop:
                    exit_price, exit_reason = stop, "SL"
                elif row["high"] >= target:
                    exit_price, exit_reason = target, "TP"
            else:
                if row["high"] >= stop:
                    exit_price, exit_reason = stop, "SL"
                elif row["low"] <= target:
                    exit_price, exit_reason = target, "TP"

            if exit_price is not None:
                pnl_pct = ((exit_price - entry_price) if direction == "long"
                           else (entry_price - exit_price)) / entry_price
                pnl_pct -= commission_bps / 10000
                trades.append({"date": day, "direction": direction,
                                "reason": exit_reason, "pnl_pct": pnl_pct})
                entered = False
                break

        if entered:
            last_price = day_df.iloc[-1]["close"]
            pnl_pct = ((last_price - entry_price) if direction == "long"
                       else (entry_price - last_price)) / entry_price
            pnl_pct -= commission_bps / 10000
            trades.append({"date": day, "direction": direction,
                            "reason": "EOD", "pnl_pct": pnl_pct})

    return pd.DataFrame(trades)


def monte_carlo_drawdown(daily_returns: pd.Series, n_sims=N_MONTE_CARLO) -> float:
    """Block-bootstrap resample of daily returns to estimate MC_DD95.
    Uses block resampling (not iid) to respect autocorrelation/clustering,
    per lessons_learned.md piège #6."""
    if len(daily_returns) < 10:
        return np.nan
    block_size = 5
    n_days = len(daily_returns)
    rets = daily_returns.values
    max_dds = []
    for _ in range(n_sims):
        blocks = []
        while sum(len(b) for b in blocks) < n_days:
            start = np.random.randint(0, max(1, n_days - block_size))
            blocks.append(rets[start:start + block_size])
        sim_rets = np.concatenate(blocks)[:n_days]
        equity = (1 + sim_rets).cumprod()
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dds.append(dd.min())
    return np.percentile(max_dds, 95) * -1  # positive number, 95th pct worst DD


def evaluate(trades_df: pd.DataFrame, label: str) -> dict:
    if trades_df.empty:
        print(f"{label}: NO TRADES")
        return {"label": label, "n": 0, "sharpe": np.nan, "trade_level_sharpe": np.nan, "mc_dd95": np.nan}
    daily = trades_df.groupby("date")["pnl_pct"].sum()
    n = len(trades_df)
    win_rate = (trades_df["pnl_pct"] > 0).mean()
    avg_ret = trades_df["pnl_pct"].mean()
    std = trades_df["pnl_pct"].std()
    # As originally coded: per-TRADE mean/std annualized with sqrt(252).
    # Kept for transparency, but this overstates significance whenever more
    # than one trade can land on the same day (up to 12 tickers here) --
    # sqrt(252) assumes 252 INDEPENDENT observations/year, not up to 12x
    # that many same-day, non-independent trades. NOT used for the
    # stop-rule verdict -- see `sharpe` (daily-return-based) below, which
    # matches how Sharpe is computed everywhere else in this project
    # (bot/backtest_engine.py's compute_metrics, always off daily returns).
    trade_level_sharpe = (avg_ret / std) * np.sqrt(252) if std > 0 else np.nan
    sharpe = (daily.mean() / daily.std()) * np.sqrt(252) if len(daily) >= 2 and daily.std() > 0 else np.nan
    mc_dd95 = monte_carlo_drawdown(daily)
    print(f"{label}: n={n} win_rate={win_rate:.1%} sharpe(daily)={sharpe:.2f} "
          f"sharpe(per-trade,orig)={trade_level_sharpe:.2f} MC_DD95={mc_dd95:.1%}")
    return {"label": label, "n": n, "win_rate": win_rate,
            "sharpe": sharpe, "trade_level_sharpe": trade_level_sharpe, "mc_dd95": mc_dd95,
            "avg_ret": avg_ret, "n_days": len(daily)}


def fetch_universe(dc: DataClient, tickers: list, start: datetime, end: datetime) -> dict:
    bars_by_ticker = {}
    for t in tickers:
        bars = fetch_15min_bars(dc, t, start, end)
        print(f"  {t}: {len(bars)} 15-min bars"
              f"{'' if bars.empty else f' ({bars.index.min().date()} to {bars.index.max().date()})'}")
        if not bars.empty:
            bars_by_ticker[t] = bars
    return bars_by_ticker


def split_chronological(bars_by_ticker: dict, split_fraction: float = WALK_FORWARD_SPLIT):
    """ONE global cutoff date across the whole universe (not a per-ticker
    row-count split) -- keeps train/test synchronized across tickers, and
    matches the docstring's "split chronologically" description."""
    all_start = min(b.index.min() for b in bars_by_ticker.values())
    all_end = max(b.index.max() for b in bars_by_ticker.values())
    cutoff = all_start + (all_end - all_start) * split_fraction
    train = {t: b[b.index < cutoff] for t, b in bars_by_ticker.items()}
    test = {t: b[b.index >= cutoff] for t, b in bars_by_ticker.items()}
    return train, test, cutoff


def run_universe_orb(bars_by_ticker: dict, tp_mult: float, commission_bps: float = COMMISSION_BPS) -> pd.DataFrame:
    all_trades = []
    for symbol, bars in bars_by_ticker.items():
        if bars.empty:
            continue
        t = run_orb_fade(bars, tp_mult, commission_bps)
        if not t.empty:
            t["symbol"] = symbol
            all_trades.append(t)
    return pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()


def apply_stop_rule(is_result: dict, oos_result: dict) -> dict:
    """Exactly the criteria specified: OOS Sharpe < 0.5, OR MC_DD95 > 20%,
    OR a sign flip (IS avg return positive, OOS negative, or vice versa).
    Uses the daily-return-based Sharpe (see `evaluate`), not the
    original per-trade one."""
    reasons = []
    oos_sharpe = oos_result.get("sharpe", np.nan)
    oos_dd95 = oos_result.get("mc_dd95", np.nan)
    is_avg, oos_avg = is_result.get("avg_ret", np.nan), oos_result.get("avg_ret", np.nan)

    if oos_result.get("n", 0) == 0:
        reasons.append("no OOS trades produced")
    else:
        if not (oos_sharpe == oos_sharpe) or oos_sharpe < 0.5:
            reasons.append(f"OOS Sharpe {oos_sharpe:.3f} < 0.5")
        if oos_dd95 == oos_dd95 and oos_dd95 > 0.20:
            reasons.append(f"MC_DD95 {oos_dd95:.1%} > 20%")
        if is_avg == is_avg and oos_avg == oos_avg and ((is_avg > 0) != (oos_avg > 0)):
            reasons.append(f"sign flip: IS avg_ret={is_avg:.5f}, OOS avg_ret={oos_avg:.5f}")

    return {"rejected": len(reasons) > 0, "reasons": reasons}


def log_md(text: str) -> None:
    with open("exploration_log_intraday_orb.md", "a", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)


def main():
    dc = DataClient(config.ALPACA_API_KEY, config.ALPACA_API_SECRET, PRICE_FEED)
    # SIP real-time embargo verified live: querying with end=now() (0 offset)
    # returns "subscription does not permit querying recent SIP data"; 1hr
    # offset already works. Using a generous 24h buffer (irrelevant for a
    # 730-day backtest) rather than chasing the exact boundary.
    now = datetime.now(timezone.utc) - timedelta(hours=24)
    primary_start = now - timedelta(days=LOOKBACK_DAYS)

    log_md(f"\n## Run -- {now.isoformat()}\n")
    log_md(f"- Feed: {PRICE_FEED} (see module docstring: IEX floor for minute bars verified "
          f"~Jan 2021 live, SIP verified back to Jan 2016 -- same account-level floor as "
          f"the daily-bar SIP depth already established for PEAD/value-quality)")
    log_md(f"- Primary window: {primary_start.date()} to {now.date()} ({LOOKBACK_DAYS} days), "
          f"60/40 chronological split, universe: {TICKERS}")

    print("Fetching primary-window 15-min bars (SIP)...")
    bars_by_ticker = fetch_universe(dc, TICKERS, primary_start, now)
    if len(bars_by_ticker) < len(TICKERS):
        missing = set(TICKERS) - set(bars_by_ticker.keys())
        log_md(f"- **No data at all for: {sorted(missing)}** -- excluded from this run, not treated as zero-return.")
    if not bars_by_ticker:
        log_md("- **No tickers returned any data -- cannot proceed.**")
        return None

    train, test, cutoff = split_chronological(bars_by_ticker)
    log_md(f"- Split cutoff: {cutoff.date()} (train: {min(b.index.min() for b in bars_by_ticker.values()).date()} "
          f"to {cutoff.date()}, test: {cutoff.date()} to {max(b.index.max() for b in bars_by_ticker.values()).date()})")

    log_md(f"\n### Per-TP-multiple results (all {len(TP_MULTIPLES)} reported independently -- "
          f"no in-sample argmax selection, per lessons_learned.md piège #4)\n")
    per_tp_results = {}
    for tp in TP_MULTIPLES:
        is_trades = run_universe_orb(train, tp)
        oos_trades = run_universe_orb(test, tp)
        is_result = evaluate(is_trades, f"TP={tp}x IS")
        oos_result = evaluate(oos_trades, f"TP={tp}x OOS")
        verdict = apply_stop_rule(is_result, oos_result)
        per_tp_results[tp] = {"is": is_result, "oos": oos_result, "verdict": verdict}

        log_md(f"- **TP={tp}x**: IS n={is_result['n']} sharpe={is_result.get('sharpe', float('nan')):.3f} | "
              f"OOS n={oos_result['n']} sharpe={oos_result.get('sharpe', float('nan')):.3f} "
              f"MC_DD95={oos_result.get('mc_dd95', float('nan')):.1%} | "
              f"**{'REJECTED' if verdict['rejected'] else 'not rejected'}**"
              f"{(' (' + '; '.join(verdict['reasons']) + ')') if verdict['reasons'] else ''}")

    all_rejected = all(r["verdict"]["rejected"] for r in per_tp_results.values())
    any_passed = any(not r["verdict"]["rejected"] for r in per_tp_results.values())

    log_md(f"\n### Cost sensitivity (OOS only, best-behaved TP multiple by OOS Sharpe among "
          f"non-rejected, or TP={TP_MULTIPLES[0]}x if all rejected -- reported for transparency, "
          f"NOT used to pick a winner)\n")
    reference_tp = (max((tp for tp, r in per_tp_results.items() if not r["verdict"]["rejected"]),
                        key=lambda tp: per_tp_results[tp]["oos"].get("sharpe", -np.inf), default=None)
                    or TP_MULTIPLES[0])
    for cost_bps in COST_SENSITIVITY_BPS:
        oos_trades_cost = run_universe_orb(test, reference_tp, commission_bps=cost_bps)
        r = evaluate(oos_trades_cost, f"TP={reference_tp}x OOS @ {cost_bps}bps")
        log_md(f"- {cost_bps}bps: OOS sharpe={r.get('sharpe', float('nan')):.3f} "
              f"avg_ret/trade={r.get('avg_ret', float('nan')):.5f}")

    log_md(f"\n### PRIMARY VERDICT\n")
    if all_rejected:
        log_md(f"**REJECTED.** Every TP multiple tested ({TP_MULTIPLES}) triggers at least one stop-rule "
              f"criterion on the primary 60/40 walk-forward split. Per the stop rules given: not retuning "
              f"parameters to force a pass. This strategy, as specified, does not clear this project's bar.")
    elif any_passed:
        passed = [tp for tp, r in per_tp_results.items() if not r["verdict"]["rejected"]]
        log_md(f"**NOT REJECTED for TP multiple(s) {passed} on the primary window.** Caveat, not a green "
              f"light: selecting a specific TP multiple only after seeing its OOS result is itself a form "
              f"of post-hoc selection (lessons_learned.md piège #4) -- this is a preliminary, single-window "
              f"result, not a validated candidate. Per the explicit rule: NOT wired into bot/main.py, NO "
              f"paper/live trading regardless of this result.")

    log_md(f"\n### Data coverage limitation (as specified)\n")
    log_md(f"Alpaca's IEX feed (this project's default) has no reliable individual-stock minute-bar data "
          f"before approximately January 2021 (verified live: zero rows for SPY/AAPL on sampled weekdays "
          f"in 2018-2020, real data from Jan 2021 onward) -- so on IEX alone, the 2018 correction is NOT "
          f"reachable for intraday validation, and the primary window above (last {LOOKBACK_DAYS} days) "
          f"cannot by itself constitute multi-regime validation.")

    return per_tp_results


def run_bonus_regime_check():
    """NOT part of the primary verdict -- see module docstring PRICE_FEED
    comment. This project already established (PEAD, value/quality) that
    Alpaca's SIP feed has real depth back to Jan 2016, which this script's
    own live verification confirms ALSO holds for minute bars (unlike IEX).
    That means the Q4 2018 correction IS reachable for THIS strategy too,
    contrary to the assumption in the original file docstring -- checked
    here as an additional, clearly-separated data point, not a
    replacement for the primary result or its stop-rule verdict."""
    dc = DataClient(config.ALPACA_API_KEY, config.ALPACA_API_SECRET, PRICE_FEED)
    start, end = Q4_2018_WINDOW

    log_md(f"\n## Bonus regime check: Q4 2018 correction ({start.date()} to {end.date()}) -- SIP-verified reachable\n")
    log_md(f"- Not part of the primary 60/40 walk-forward or its stop-rule verdict above. Included because "
          f"live verification (see module docstring) showed SIP reaches back to Jan 2016 for minute bars, "
          f"same as this project's established daily-bar SIP floor -- so unlike the IEX-only assumption in "
          f"the original file, this specific regime IS testable, and testing it costs nothing extra once "
          f"SIP is already being used for the primary run.")

    bars_by_ticker = fetch_universe(dc, TICKERS, start, end)
    if not bars_by_ticker:
        log_md("- No data returned for this window -- cannot evaluate.")
        return None

    for tp in TP_MULTIPLES:
        trades = run_universe_orb(bars_by_ticker, tp)
        result = evaluate(trades, f"TP={tp}x Q4-2018")
        log_md(f"- TP={tp}x: n={result['n']} sharpe={result.get('sharpe', float('nan')):.3f} "
              f"MC_DD95={result.get('mc_dd95', float('nan')):.1%}")
    return None


if __name__ == "__main__":
    main()
    run_bonus_regime_check()
