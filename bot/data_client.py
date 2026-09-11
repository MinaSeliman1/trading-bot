"""
Historical/live market data via alpaca-py, with local Parquet caching and a
hard guarantee that indicator/signal code never sees an in-progress candle.

Cache layout: data_cache/{SYMBOL_SANITIZED}_{TIMEFRAME}.parquet, oldest-first,
deduplicated by timestamp. Re-running never re-downloads the full history --
only the gap between the cached tail and now is fetched.
"""
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

logger = logging.getLogger(__name__)

CACHE_DIR = Path(os.getenv("DATA_CACHE_DIR", "data_cache"))

_TIMEFRAME_MAP = {
    "1Min": TimeFrame(1, TimeFrameUnit.Minute),
    "15Min": TimeFrame(15, TimeFrameUnit.Minute),
    "1Hour": TimeFrame.Hour,
    "4Hour": TimeFrame(4, TimeFrameUnit.Hour),
    "1Day": TimeFrame.Day,
}

_TIMEFRAME_SECONDS = {
    "1Min": 60,
    "15Min": 15 * 60,
    "1Hour": 60 * 60,
    "4Hour": 4 * 60 * 60,
    "1Day": 24 * 60 * 60,
}

# Stay clear of any recent-data embargo and of the still-forming current bar.
DATA_EMBARGO_MINUTES = 16


def timeframe_seconds(tf_str: str) -> int:
    return _TIMEFRAME_SECONDS[tf_str]


def _cache_path(symbol: str, timeframe: str) -> Path:
    safe_symbol = symbol.replace("/", "")
    return CACHE_DIR / f"{safe_symbol}_{timeframe}.parquet"


def _dedupe_sort(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


class DataClient:
    def __init__(self, api_key: str, api_secret: str, data_feed: str = "iex"):
        self.stock_client = StockHistoricalDataClient(api_key, api_secret)
        self.crypto_client = CryptoHistoricalDataClient()
        self.feed = DataFeed.IEX if data_feed == "iex" else DataFeed.SIP
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # closed-candle enforcement
    # ------------------------------------------------------------------ #
    @staticmethod
    def drop_unclosed_candle(df: pd.DataFrame, timeframe: str, now: datetime = None) -> pd.DataFrame:
        """Drop any bar whose interval hasn't fully elapsed yet, so indicators
        never see an in-progress candle. Bar timestamps are interval starts."""
        if df.empty:
            return df
        now = now or datetime.now(timezone.utc)
        bar_end = df.index + pd.Timedelta(seconds=timeframe_seconds(timeframe))
        return df[bar_end <= now]

    # ------------------------------------------------------------------ #
    # raw fetch (single request, no caching)
    # ------------------------------------------------------------------ #
    def _fetch_raw(self, symbol: str, asset_class: str, timeframe: str,
                    start: datetime, end: datetime) -> pd.DataFrame:
        tf = _TIMEFRAME_MAP[timeframe]
        if asset_class == "crypto":
            req = CryptoBarsRequest(symbol_or_symbols=[symbol], timeframe=tf, start=start, end=end)
            bar_set = self.crypto_client.get_crypto_bars(req)
        else:
            req = StockBarsRequest(symbol_or_symbols=[symbol], timeframe=tf, start=start, end=end,
                                    adjustment=Adjustment.RAW, feed=self.feed)
            bar_set = self.stock_client.get_stock_bars(req)

        df = bar_set.df
        if df.empty:
            return df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)
        return _dedupe_sort(df)

    # ------------------------------------------------------------------ #
    # cached fetch
    # ------------------------------------------------------------------ #
    def get_bars(self, symbol: str, asset_class: str, timeframe: str,
                  start: datetime, end: datetime = None, use_cache: bool = True) -> pd.DataFrame:
        """Return closed-candle-only OHLCV bars for [start, end], oldest-first.
        Fills the cache incrementally: fetches whatever isn't already covered,
        both an older backfill gap (start < cached start) and a newer gap
        (end > cached end) -- a cache seeded by an earlier, narrower request
        must not silently starve a later, wider one."""
        end = end or (datetime.now(timezone.utc) - timedelta(minutes=DATA_EMBARGO_MINUTES))
        path = _cache_path(symbol, timeframe)
        start_ts = pd.Timestamp(start) if start.tzinfo else pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end) if end.tzinfo else pd.Timestamp(end, tz="UTC")

        cached = pd.DataFrame()
        if use_cache and path.exists():
            cached = pd.read_parquet(path)
            cached.index = pd.to_datetime(cached.index, utc=True)

        fresh_chunks = []
        if cached.empty:
            fresh_chunks.append(self._fetch_raw(symbol, asset_class, timeframe, start, end))
        else:
            cached_start, cached_end = cached.index.min(), cached.index.max()
            if start_ts < cached_start:
                fresh_chunks.append(self._fetch_raw(symbol, asset_class, timeframe, start, cached_start))
            if end_ts > cached_end:
                fresh_chunks.append(self._fetch_raw(symbol, asset_class, timeframe, cached_end + timedelta(seconds=1), end))

        combined = pd.concat([cached] + fresh_chunks) if fresh_chunks else cached
        combined = _dedupe_sort(combined)

        if use_cache and not combined.empty:
            path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(path)

        window = combined[(combined.index >= start_ts) & (combined.index <= end_ts)]
        window = self.drop_unclosed_candle(window, timeframe, now=datetime.now(timezone.utc))
        return window

    # ------------------------------------------------------------------ #
    # liquidity filter
    # ------------------------------------------------------------------ #
    @staticmethod
    def liquidity_ok(bars: pd.DataFrame, lookback: int = 20, min_volume_ratio: float = 0.5,
                       bid: float = None, ask: float = None, max_spread_pct: float = 0.01) -> bool:
        """True if the latest closed bar clears the volume floor and (when quote
        data is supplied) the bid-ask spread isn't blown out -- relevant mainly
        for equity pre/post-market where spreads widen."""
        if len(bars) < lookback + 1:
            return True  # not enough history to judge; don't block on it
        avg_volume = bars["volume"].iloc[-(lookback + 1):-1].mean()
        last_volume = bars["volume"].iloc[-1]
        if avg_volume > 0 and last_volume < min_volume_ratio * avg_volume:
            logger.info("Liquidity filter: volume %.0f < %.0f%% of %d-bar avg (%.0f)",
                        last_volume, min_volume_ratio * 100, lookback, avg_volume)
            return False
        if bid is not None and ask is not None and bid > 0:
            mid = (bid + ask) / 2
            spread_pct = (ask - bid) / mid if mid > 0 else 0
            if spread_pct > max_spread_pct:
                logger.info("Liquidity filter: spread %.4f%% > threshold %.4f%%", spread_pct * 100, max_spread_pct * 100)
                return False
        return True


def validate_cache_integrity(symbol: str, timeframe: str) -> dict:
    """Sanity-check a cached Parquet file: no duplicate timestamps, monotonic
    index, no obviously-missing bars during regular trading hours (equities)
    or excessive gaps (crypto)."""
    path = _cache_path(symbol, timeframe)
    if not path.exists():
        return {"exists": False}

    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    duplicates = int(df.index.duplicated().sum())
    monotonic = bool(df.index.is_monotonic_increasing)

    gaps = df.index.to_series().diff().dropna()
    expected = pd.Timedelta(seconds=timeframe_seconds(timeframe))
    # Large gaps are expected across weekends/holidays for equities; flag only
    # gaps that are neither ~1 bar nor a plausible multi-day market closure.
    suspicious_gaps = int((gaps > expected * 20).sum())

    return {
        "exists": True,
        "rows": len(df),
        "duplicates": duplicates,
        "monotonic": monotonic,
        "suspicious_gaps": suspicious_gaps,
        "start": df.index.min(),
        "end": df.index.max(),
    }
