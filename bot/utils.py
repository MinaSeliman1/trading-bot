"""
Shared helpers: bar fetching, timeframe parsing, and a retry wrapper used to
absorb transient Alpaca API / network errors.
"""
import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca_trade_api.rest import APIError, TimeFrame, TimeFrameUnit

logger = logging.getLogger(__name__)

# Equities only trade ~6.5h of each 24h day, 5 of 7 days -- padding the
# lookback window keeps us from under-requesting history around weekends
# and holidays. Crypto trades 24/7 so it needs no padding.
_EQUITY_LOOKBACK_PADDING = 8
# Stay clear of the free-tier SIP embargo on very recent bars (and of the
# still-forming current bar in general).
_DATA_EMBARGO_MINUTES = 16

_TIMEFRAME_MAP = {
    "15Min": TimeFrame(15, TimeFrameUnit.Minute),
    "1Hour": TimeFrame.Hour,
    "4Hour": TimeFrame(4, TimeFrameUnit.Hour),
}

_TIMEFRAME_SECONDS = {
    "15Min": 15 * 60,
    "1Hour": 60 * 60,
    "4Hour": 4 * 60 * 60,
}


def parse_timeframe(tf_str: str) -> TimeFrame:
    try:
        return _TIMEFRAME_MAP[tf_str]
    except KeyError:
        raise ValueError(f"Unsupported timeframe: {tf_str!r}")


def timeframe_seconds(tf_str: str) -> int:
    try:
        return _TIMEFRAME_SECONDS[tf_str]
    except KeyError:
        raise ValueError(f"Unsupported timeframe: {tf_str!r}")


def retry_api_call(fn, *args, retries: int = 3, backoff: float = 2.0, **kwargs):
    """Retry an Alpaca REST call with exponential-ish backoff. Re-raises the last error."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except APIError as e:
            last_exc = e
            logger.warning("Alpaca API error (attempt %s/%s): %s", attempt, retries, e)
        except Exception as e:  # network drops, timeouts, DNS failures, etc.
            last_exc = e
            logger.warning("Connection error (attempt %s/%s): %s", attempt, retries, e)
        if attempt < retries:
            time.sleep(backoff * attempt)
    logger.error("All %s retries exhausted: %s", retries, last_exc)
    raise last_exc


def fetch_bars(api, instrument_cfg, limit: int = 250, feed: str = None) -> pd.DataFrame:
    """Fetch OHLCV bars for one instrument, oldest-first, as a plain DataFrame.

    Alpaca's get_bars/get_crypto_bars default to a narrow "recent" window when
    start/end are omitted -- for equities that can return nothing at all before
    today's session has opened. We always pass an explicit start/end sized to
    the requested bar count so history is available regardless of market hours.
    """
    tf = parse_timeframe(instrument_cfg.timeframe)
    is_crypto = instrument_cfg.asset_class == "crypto"

    end = datetime.now(timezone.utc) - timedelta(minutes=0 if is_crypto else _DATA_EMBARGO_MINUTES)
    lookback_seconds = timeframe_seconds(instrument_cfg.timeframe) * limit
    if not is_crypto:
        lookback_seconds *= _EQUITY_LOOKBACK_PADDING
    start = end - timedelta(seconds=lookback_seconds)

    start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Note: Alpaca's bars endpoint paginates *forward* from `start` when a
    # `limit` is given -- with our padded, historically-wide `start` that would
    # return the OLDEST bars in the window, not the most recent ones. So we
    # fetch the whole start/end range (no API-side limit) and take the tail
    # ourselves after sorting.
    if is_crypto:
        bar_set = retry_api_call(api.get_crypto_bars, instrument_cfg.symbol, tf,
                                  start=start_str, end=end_str)
    else:
        kwargs = {"start": start_str, "end": end_str, "adjustment": "raw"}
        if feed:
            kwargs["feed"] = feed
        bar_set = retry_api_call(api.get_bars, instrument_cfg.symbol, tf, **kwargs)

    df = bar_set.df
    if df.empty:
        return df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(instrument_cfg.symbol, level=0)
    return df.sort_index().tail(limit)


def required_bar_count(cfg) -> int:
    """Minimum lookback (in bars) a strategy needs, plus headroom for ATR."""
    from config import ATR_PERIOD

    needed = max(
        ATR_PERIOD,
        cfg.params.get("slow_ema", 0),
        cfg.params.get("sma_period", 0),
        cfg.params.get("lookback", 0),
    )
    return needed + 10
