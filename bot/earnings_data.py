"""
Alpha Vantage EARNINGS endpoint client: quarterly earnings surprises
(reported EPS vs analyst estimate) for a single symbol per call, cached to
disk so the free tier's 25 requests/day is never spent twice on the same
symbol. Verified against real data before this module was written --
see exploration_log_pead.md for the verification sample.
"""
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

CACHE_DIR = Path("earnings_cache")
BASE_URL = "https://www.alphavantage.co/query"
MIN_SECONDS_BETWEEN_CALLS = 15  # keeps us well under the 5/min free-tier limit
MIN_ABS_ESTIMATED_EPS = 0.05  # below this, surprise_pct = (actual-est)/est blows up meaninglessly


class AlphaVantageRateLimitError(Exception):
    """Raised specifically when Alpha Vantage's documented rate-limit/quota
    response is detected (an 'Information' or 'Note' key mentioning the
    daily/per-minute cap) -- distinct from a symbol simply having no usable
    data, so callers processing a batch can stop on quota exhaustion but
    keep going past an individual bad symbol."""


class AlphaVantageEarningsClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._last_call_time = 0.0
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, symbol: str) -> Path:
        return CACHE_DIR / f"{symbol.replace('/', '_')}_earnings.json"

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call_time
        if elapsed < MIN_SECONDS_BETWEEN_CALLS:
            time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)

    def get_quarterly_surprises(self, symbol: str, use_cache: bool = True) -> pd.DataFrame:
        """Returns a DataFrame indexed by reportedDate (the actual public
        announcement date) with columns: fiscal_date_ending, reported_eps,
        estimated_eps, surprise, surprise_pct, report_time."""
        cache_path = self._cache_path(symbol)
        if use_cache and cache_path.exists():
            with open(cache_path) as f:
                raw = json.load(f)
            return self._parse(raw, symbol)

        self._throttle()
        resp = requests.get(BASE_URL, params={"function": "EARNINGS", "symbol": symbol, "apikey": self.api_key},
                             timeout=30)
        self._last_call_time = time.time()
        resp.raise_for_status()
        raw = resp.json()

        rate_limit_text = raw.get("Information") or raw.get("Note")
        if rate_limit_text and ("rate limit" in rate_limit_text.lower() or "per day" in rate_limit_text.lower()
                                or "per minute" in rate_limit_text.lower()):
            logger.error("Alpha Vantage rate limit hit for %s: %s", symbol, rate_limit_text)
            raise AlphaVantageRateLimitError(rate_limit_text)

        if "Information" in raw or "Note" in raw or "Error Message" in raw or not raw.get("quarterlyEarnings"):
            # covers both other documented error shapes AND any malformed/empty
            # response (e.g. missing quarterlyEarnings with none of the above
            # keys) -- caching either would permanently and silently treat this
            # symbol as "no data" on every future run. Not a quota issue, so
            # callers should skip this symbol and continue, not stop the batch.
            logger.error("Alpha Vantage returned no usable data for %s: %s", symbol,
                         raw.get("Information") or raw.get("Note") or raw.get("Error Message") or
                         "response missing/empty quarterlyEarnings")
            return pd.DataFrame()

        with open(cache_path, "w") as f:
            json.dump(raw, f)
        return self._parse(raw, symbol)

    @staticmethod
    def _parse(raw: dict, symbol: str) -> pd.DataFrame:
        rows = raw.get("quarterlyEarnings", [])
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["reportedDate"] = pd.to_datetime(df["reportedDate"], utc=True)
        for col in ("reportedEPS", "estimatedEPS", "surprise", "surprisePercentage"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["reportedDate"]).set_index("reportedDate").sort_index()
        df = df.rename(columns={"fiscalDateEnding": "fiscal_date_ending", "reportedEPS": "reported_eps",
                                "estimatedEPS": "estimated_eps", "surprise": "surprise",
                                "surprisePercentage": "surprise_pct", "reportTime": "report_time"})
        df["symbol"] = symbol

        # Data hygiene, not strategy logic: surprise_pct = (actual-est)/est blows
        # up to meaningless magnitudes (seen up to +-7500% in this dataset) when
        # the consensus estimate is near zero -- a known artifact of percentage
        # surprise metrics, not a genuine outsized earnings beat. Null out
        # surprise_pct (not drop the row -- reported/estimated EPS are still
        # valid) so these quarters are excluded from any surprise_pct>=threshold
        # signal exactly like the already-handled NaN case for old pre-coverage
        # quarters, without touching any other logic.
        near_zero_estimate = df["estimated_eps"].abs() < MIN_ABS_ESTIMATED_EPS
        df.loc[near_zero_estimate, "surprise_pct"] = float("nan")

        # Same reportedDate can appear twice (seen for COST 2007-05-31, likely a
        # restated/delayed report) -- keep the row with a usable surprise_pct if
        # there's a choice, else the first, so a single event isn't double-counted.
        if df.index.duplicated().any():
            df = df.sort_values("surprise_pct", key=lambda s: s.isna()).groupby(level=0).head(1).sort_index()

        return df
