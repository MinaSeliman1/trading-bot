"""
Sector drip-feed: fetches Alpha Vantage OVERVIEW (Sector/Industry only) for
the 97 tickers already collected for PEAD (earnings_cache/, used here only
as a stable list of the universe's tickers, not for earnings data itself).

Fundamentals (balance sheet + income statement) no longer go through this
script or through Alpha Vantage at all -- they come from SEC EDGAR
(bot/fundamentals_data.py, SECEdgarFundamentalsClient), which has no daily
quota and completes the whole universe in minutes. See
collect_sec_fundamentals.py for that one-shot collection.

This script now has Alpha Vantage's full 25/day quota to itself for
sector data alone (1 call/ticker instead of competing with 2 more calls/
ticker for statements) -- ~97 calls / 25 per day = ~4 days for the whole
universe, down from the original combined estimate. Same self-limiting
pace as before: stops gracefully when Alpha Vantage's rate-limit response
is detected, resumable across days without re-fetching anything cached.
JPM is fetched first (not alphabetically) so a usable sample is available
the same day this is first run.
"""
import logging
from pathlib import Path

import config
from bot.fundamentals_data import SECEdgarFundamentalsClient
from bot.earnings_data import AlphaVantageRateLimitError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("drip_feed_sector")

TICKER_UNIVERSE_DIR = Path("earnings_cache")
PRIORITY = ["JPM"]  # fetched first so a demo sample is available same-day


def discover_tickers() -> list:
    return sorted(p.stem.replace("_earnings", "") for p in TICKER_UNIVERSE_DIR.glob("*_earnings.json"))


def main():
    all_tickers = discover_tickers()
    target = PRIORITY + [t for t in all_tickers if t not in PRIORITY]
    logger.info("Sector drip-feed target: %d tickers (Alpha Vantage OVERVIEW only -- "
               "fundamentals now come from SEC EDGAR, see collect_sec_fundamentals.py)", len(target))

    client = SECEdgarFundamentalsClient()
    fetched_today, already_cached, no_data_symbols = [], [], []
    quota_exhausted = False

    for t in target:
        ov_path = client._cache_path(t, "overview")
        if ov_path.exists():
            already_cached.append(t)
            continue
        try:
            sector_info = client.get_sector_industry(t, config.ALPHA_VANTAGE_API_KEY)
        except AlphaVantageRateLimitError:
            logger.warning("Daily quota exhausted, stopping for today (reached %s)", t)
            quota_exhausted = True
            break
        if not sector_info.get("sector"):
            logger.warning("%s: no usable sector data -- skipping, continuing batch", t)
            no_data_symbols.append(t)
            continue
        fetched_today.append(t)
        logger.info("%s: sector=%s industry=%s (%d symbols fetched so far today)", t, sector_info["sector"],
                   sector_info["industry"], len(fetched_today))

    remaining = [t for t in target if t not in fetched_today and t not in already_cached and t not in no_data_symbols]
    logger.info("Summary: %d fetched today, %d already cached, %d confirmed no-data, %d remaining for future days",
               len(fetched_today), len(already_cached), len(no_data_symbols), len(remaining))
    if no_data_symbols:
        logger.info("No-data symbols (won't be retried): %s", no_data_symbols)
    if remaining and quota_exhausted:
        rate = max(1, len(fetched_today))
        days_needed = -(-len(remaining) // rate)
        logger.info("At ~%d/day (today's observed rate): ~%d more day(s) needed", rate, days_needed)

    return target, fetched_today, already_cached, no_data_symbols, remaining


if __name__ == "__main__":
    main()
