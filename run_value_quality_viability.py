"""
Value/quality factor viability check: assembles whatever's currently fully
cached (earnings + balance_sheet + income_statement + overview/sector) and
runs the full pipeline -- composite score, backtest, cost sensitivity,
multi-regime breakdown, viability verdict. Designed to be re-run as-is as
the fundamentals+sector drip-feed progresses; always uses whatever subset
of the 97-ticker universe is currently complete, no edits needed.

Refuses to run a "real" backtest below MIN_UNIVERSE_SIZE tickers -- a
quintile/decile on a handful of names is not statistically meaningful and
reporting a number from it would repeat exactly the small-sample mistake
this project's lessons_learned.md warns against.

Uses the SIP feed (not this project's IEX default) for price history, in
an ISOLATED cache directory (data_cache_sip/, via DATA_CACHE_DIR override
below) -- SIP has real coverage back to 2016-01-04 (verified), unlike IEX
which has no individual-equity history before ~2020-07 on this account.
Isolated so this doesn't mix provenance with the shared data_cache/ used
by the rest of the project. Range chosen to cover both the Q4-2018
correction and the COVID-2020 crash, per the user's explicit multi-regime
requirement.
"""
import os

os.environ["DATA_CACHE_DIR"] = "data_cache_sip"  # MUST be set before bot.data_client is first imported

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import config
from bot.data_client import DataClient
from bot.fundamentals_data import SECEdgarFundamentalsClient
from bot.value_quality_signal import build_ticker_metrics
from bot.value_quality_backtest import run_backtest, cost_sensitivity, regime_slice_metrics, viability_verdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vq_viability")

LOG_PATH = "exploration_log_value_quality.md"
FUNDAMENTALS_CACHE_DIR = Path("fundamentals_cache")
TICKER_UNIVERSE_DIR = Path("earnings_cache")  # historical name -- just the ticker list source, no earnings used anymore
STARTING_CAPITAL = 100_000.0
PRICE_START = datetime(2016, 1, 4, tzinfo=timezone.utc)  # SIP floor, verified against the live API
BASKET_FRACTION = 0.2  # quintile
MIN_GROUP_SIZE = 8
MIN_UNIVERSE_SIZE = 40  # below this, refuse to report a backtest result -- see module docstring
REGIME_WINDOWS = {
    "Q4-2018 correction": (pd.Timestamp("2018-09-20", tz="UTC"), pd.Timestamp("2018-12-26", tz="UTC")),
    "COVID crash": (pd.Timestamp("2020-02-19", tz="UTC"), pd.Timestamp("2020-04-07", tz="UTC")),
    "Recent (last 18mo)": (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=548), pd.Timestamp.now(tz="UTC")),
}


def log_md(text: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    logger.info(text.split("\n")[0][:200])


def discover_usable_tickers() -> list:
    """A ticker is usable once it has: SEC EDGAR fundamentals cached
    ({t}_secfacts.json) AND Alpha Vantage sector cached ({t}_overview.json).
    Earnings data is no longer used anywhere in this pipeline -- the ticker
    LIST still comes from earnings_cache/ purely because that's where the
    original 97-ticker universe (from the PEAD work) happens to be recorded,
    not because earnings data itself is needed here."""
    tickers = []
    for p in sorted(TICKER_UNIVERSE_DIR.glob("*_earnings.json")):
        t = p.stem.replace("_earnings", "")
        if ((FUNDAMENTALS_CACHE_DIR / f"{t}_secfacts.json").exists() and
                (FUNDAMENTALS_CACHE_DIR / f"{t}_overview.json").exists()):
            tickers.append(t)
    return tickers


def main():
    tickers = discover_usable_tickers()
    log_md(f"\n## Viability check attempt -- {pd.Timestamp.now(tz='UTC').isoformat()}\n")
    log_md(f"- Usable tickers (SEC EDGAR fundamentals + Alpha Vantage sector both cached): "
          f"{len(tickers)} of 97")

    if len(tickers) < MIN_UNIVERSE_SIZE:
        log_md(f"- **Below MIN_UNIVERSE_SIZE ({MIN_UNIVERSE_SIZE}) -- refusing to run a backtest.** "
              f"A quintile/decile on {len(tickers)} tickers is not statistically meaningful (see "
              f"lessons_learned.md). Waiting for the drip-feed to progress further before attempting "
              f"a real run.")
        return None

    dc = DataClient(config.ALPACA_API_KEY, config.ALPACA_API_SECRET, "sip")
    fundamentals_client = SECEdgarFundamentalsClient()

    ticker_metrics, price_bars_by_ticker, sector_by_symbol = {}, {}, {}
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    for t in tickers:
        pit, missing_concepts = fundamentals_client.get_quarterly_fundamentals(t)
        if missing_concepts:
            logger.warning("%s: SEC EDGAR has no data for concepts %s (left as explicit NaN, not guessed)",
                           t, missing_concepts)
        bars = dc.get_bars(t, "us_equity", "1Day", PRICE_START, end)
        if pit.empty or bars.empty:
            continue
        metrics = build_ticker_metrics(t, pit, bars)
        if metrics.empty:
            continue
        ticker_metrics[t] = metrics
        price_bars_by_ticker[t] = bars
        sector_by_symbol[t] = fundamentals_client.get_sector_industry(t, config.ALPHA_VANTAGE_API_KEY)

    usable = list(ticker_metrics.keys())
    log_md(f"- After building point-in-time metrics: {len(usable)} tickers usable "
          f"(dropped {len(tickers) - len(usable)} with empty metrics/price)")
    if len(usable) < MIN_UNIVERSE_SIZE:
        log_md(f"- Fell below MIN_UNIVERSE_SIZE after metrics construction -- refusing to run.")
        return None

    price_start = min(b.index.min() for b in price_bars_by_ticker.values())
    price_end = max(b.index.max() for b in price_bars_by_ticker.values())
    log_md(f"- Backtest span: {price_start.date()} to {price_end.date()}")

    result = run_backtest(usable, ticker_metrics, sector_by_symbol, price_bars_by_ticker,
                          price_start, price_end, STARTING_CAPITAL, slippage_pct=0.0010,
                          basket_fraction=BASKET_FRACTION, min_group_size=MIN_GROUP_SIZE)
    metrics, mc, trades = result["metrics"], result["mc"], result["trades"]
    basket_sizes = [len(b["basket"]) for b in result["baskets"]]

    log_md(f"- Basket size over time: min={min(basket_sizes) if basket_sizes else 'N/A'} "
          f"max={max(basket_sizes) if basket_sizes else 'N/A'} "
          f"(quintile of ~{len(usable)} tickers)")
    log_md(f"- Sharpe={metrics['sharpe']:.3f}  Sortino={metrics['sortino']:.3f}  "
          f"MaxDD={metrics['max_drawdown']*100:.1f}%  MC_DD95={mc['max_dd_p95']*100:.1f}%  "
          f"PF={metrics['profit_factor']:.3f}  WinRate={metrics['win_rate']*100:.1f}%  "
          f"Trades={metrics['trade_count']}")

    costs = cost_sensitivity(usable, ticker_metrics, sector_by_symbol, price_bars_by_ticker,
                             price_start, price_end, STARTING_CAPITAL, BASKET_FRACTION, MIN_GROUP_SIZE)
    log_md(f"- Cost sensitivity: " + ", ".join(f"{k*100:.2f}%->{v*100:.1f}%" for k, v in costs.items()))

    verdict = viability_verdict(metrics, mc, costs)
    log_md(f"- Viability (full span): {verdict}")

    log_md("\n### Regime breakdown\n")
    regime_verdicts = {}
    for label, (rstart, rend) in REGIME_WINDOWS.items():
        rmetrics = regime_slice_metrics(trades, price_bars_by_ticker, STARTING_CAPITAL, rstart, rend)
        log_md(f"- {label} ({rstart.date()} to {rend.date()}): {rmetrics}")
        regime_verdicts[label] = rmetrics

    return {"metrics": metrics, "mc": mc, "costs": costs, "verdict": verdict, "regime_verdicts": regime_verdicts,
           "n_tickers": len(usable)}


if __name__ == "__main__":
    main()
