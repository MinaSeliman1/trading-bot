"""
PEAD universe expansion drip-feed: objectively-drawn fresh tickers (same
rigor as round 2's independent-universe validation -- fixed seed, documented
pool, no manual cherry-picking), fetched at the free tier's 25/day pace.
Stops gracefully when the daily quota is exhausted (detected via Alpha
Vantage's rate-limit response) and reports progress so this can be resumed
on subsequent days without re-fetching anything already cached.
"""
import logging

import numpy as np

import config
from bot.earnings_data import AlphaVantageEarningsClient, AlphaVantageRateLimitError
from bot.sp500_pool import SP500_POOL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("drip_feed")

ALREADY_TESTED_PEAD = ["JPM", "BAC", "GS", "JNJ", "UNH", "PFE", "ABBV", "PG", "KO", "WMT",
                       "MCD", "CAT", "HON", "UPS", "XOM", "CVX", "VZ", "DIS", "HD", "COST",
                       "LIN", "DUK", "V", "AXP"]  # AXP already used as today's quota probe
RANDOM_SEED = 20260703  # same documented seed convention as round 2's independent validation draw
TARGET_BATCH_SIZE = 75  # aim to roughly quadruple the universe (23 -> ~98)


def main():
    remaining_pool = [t for t in SP500_POOL if t not in ALREADY_TESTED_PEAD]
    rng = np.random.default_rng(RANDOM_SEED)
    target = sorted(rng.choice(remaining_pool, size=min(TARGET_BATCH_SIZE, len(remaining_pool)), replace=False).tolist())
    logger.info("Fresh drip-feed target universe: %d tickers (seed=%d, pool=%d after excluding %d already tested)",
               len(target), RANDOM_SEED, len(remaining_pool), len(ALREADY_TESTED_PEAD))

    client = AlphaVantageEarningsClient(config.ALPHA_VANTAGE_API_KEY)
    fetched_today, already_cached, no_data_symbols = [], [], []
    quota_exhausted = False

    for t in target:
        cache_path = client._cache_path(t)
        if cache_path.exists():
            already_cached.append(t)
            continue
        try:
            df = client.get_quarterly_surprises(t, use_cache=True)
        except AlphaVantageRateLimitError:
            logger.warning("Daily quota exhausted, stopping for today (reached %s)", t)
            quota_exhausted = True
            break
        if df.empty:
            logger.warning("%s: no usable data (not a quota issue) -- skipping this symbol, continuing batch", t)
            no_data_symbols.append(t)
            continue
        fetched_today.append(t)
        logger.info("%s: %d quarters fetched (%d fetched so far today)", t, len(df), len(fetched_today))

    remaining = [t for t in target if t not in fetched_today and t not in already_cached and t not in no_data_symbols]
    logger.info("Summary: %d fetched today, %d already cached from before, %d confirmed no-data symbols, "
               "%d remaining for future days", len(fetched_today), len(already_cached), len(no_data_symbols),
               len(remaining))
    if no_data_symbols:
        logger.info("No-data symbols (won't be retried): %s", no_data_symbols)
    if remaining and quota_exhausted:
        days_needed = -(-len(remaining) // max(1, len(fetched_today)))
        logger.info("At ~%d/day (today's observed rate): ~%d more day(s) needed", len(fetched_today), days_needed)
    elif remaining:
        logger.info("Batch not yet complete but quota was NOT exhausted -- %d tickers had no data, "
                   "%d still to attempt (should complete on next run)", len(no_data_symbols), len(remaining))

    return target, fetched_today, already_cached, no_data_symbols, remaining


if __name__ == "__main__":
    main()
