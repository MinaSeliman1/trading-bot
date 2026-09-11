"""
One-shot SEC EDGAR fundamentals collection for the 97-ticker universe
(earnings_cache/, used only as the ticker list). Unlike the old Alpha
Vantage drip-feed, this has no daily quota -- expected to complete the
whole universe in minutes, throttled only by SEC's own ~10 req/sec
fair-use guidance (bot/fundamentals_data.py's SEC_MIN_SECONDS_BETWEEN_CALLS).
Resumable/idempotent: already-cached tickers (fundamentals_cache/{t}_secfacts.json)
are skipped without a new request.
"""
import logging
import time
from pathlib import Path

from bot.fundamentals_data import SECEdgarFundamentalsClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("collect_sec_fundamentals")

TICKER_UNIVERSE_DIR = Path("earnings_cache")


def discover_tickers() -> list:
    return sorted(p.stem.replace("_earnings", "") for p in TICKER_UNIVERSE_DIR.glob("*_earnings.json"))


def main():
    tickers = discover_tickers()
    logger.info("SEC EDGAR fundamentals collection target: %d tickers", len(tickers))

    client = SECEdgarFundamentalsClient()
    start = time.time()
    succeeded, failed, all_missing_concepts = [], [], {}

    for t in tickers:
        try:
            df, missing_concepts = client.get_quarterly_fundamentals(t)
        except Exception as e:
            logger.error("%s: fetch failed: %r", t, e)
            failed.append(t)
            continue
        if df.empty:
            logger.warning("%s: no usable data at all from SEC EDGAR", t)
            failed.append(t)
            continue
        succeeded.append(t)
        if missing_concepts:
            all_missing_concepts[t] = missing_concepts
        logger.info("%s: %d quarters, missing_concepts=%s", t, len(df), missing_concepts or "none")

    elapsed = time.time() - start
    logger.info("Done in %.1fs: %d succeeded, %d failed", elapsed, len(succeeded), len(failed))
    if failed:
        logger.info("Failed tickers (no CIK match or no usable data): %s", failed)
    if all_missing_concepts:
        logger.info("Tickers with at least one concept missing entirely: %d/%d", len(all_missing_concepts), len(succeeded))

    return {"succeeded": succeeded, "failed": failed, "missing_concepts": all_missing_concepts, "elapsed": elapsed}


if __name__ == "__main__":
    main()
