"""
Operational health checks: flag-and-alert only in this phase, no automatic
remediation. Repeated API errors, missing/incomplete candle data, and
local-vs-broker position mismatches are logged loudly so a human notices.
"""
import logging
from collections import deque
from datetime import datetime, timedelta, timezone

import pandas as pd

logger = logging.getLogger(__name__)


class HealthCheck:
    def __init__(self, api_error_threshold: int = 5, api_error_window_minutes: int = 10):
        self.api_error_threshold = api_error_threshold
        self.api_error_window = timedelta(minutes=api_error_window_minutes)
        self._api_error_times = deque()

    def record_api_error(self, context: str = "") -> bool:
        """Record an API error occurrence. Returns True if the error-rate
        threshold has been breached (flags, does not halt anything)."""
        now = datetime.now(timezone.utc)
        self._api_error_times.append(now)
        cutoff = now - self.api_error_window
        while self._api_error_times and self._api_error_times[0] < cutoff:
            self._api_error_times.popleft()

        breached = len(self._api_error_times) >= self.api_error_threshold
        if breached:
            logger.error(
                "HEALTH ALERT: %d API errors in the last %s minutes (threshold=%d). Context: %s",
                len(self._api_error_times), self.api_error_window.total_seconds() / 60,
                self.api_error_threshold, context,
            )
        return breached

    @staticmethod
    def check_candle_completeness(bars: pd.DataFrame, timeframe_seconds_: int, max_gap_multiplier: float = 3.0) -> list:
        """Flag intraday gaps larger than `max_gap_multiplier`x the expected bar
        interval. Returns a list of (gap_start, gap_end, gap_seconds) tuples."""
        if bars.empty or len(bars) < 2:
            return []
        diffs = bars.index.to_series().diff().dropna()
        threshold = pd.Timedelta(seconds=timeframe_seconds_ * max_gap_multiplier)
        flagged = []
        for ts, gap in diffs.items():
            if gap > threshold:
                gap_start = ts - gap
                flagged.append((gap_start, ts, gap.total_seconds()))
        if flagged:
            logger.warning("HEALTH ALERT: %d candle gap(s) detected exceeding %.0fs", len(flagged), threshold.total_seconds())
        return flagged

    @staticmethod
    def check_position_mismatch(local_positions: dict, broker_positions: list) -> list:
        """Compare local in-memory positions against broker-reported positions.
        Returns a list of mismatch descriptions (empty if none)."""
        broker_by_symbol = {p.symbol: p for p in broker_positions}
        mismatches = []

        for key, local_pos in local_positions.items():
            broker_pos = broker_by_symbol.get(local_pos.symbol)
            if broker_pos is None:
                mismatches.append(f"{key}: local position open ({local_pos.symbol}) but broker reports none")
                continue
            broker_qty = abs(float(broker_pos.qty))
            if abs(broker_qty - local_pos.qty) > 1e-6:
                mismatches.append(
                    f"{key}: qty mismatch -- local={local_pos.qty}, broker={broker_qty}"
                )

        local_symbols = {p.symbol for p in local_positions.values()}
        for symbol in broker_by_symbol:
            if symbol not in local_symbols:
                mismatches.append(f"{symbol}: broker reports an open position with no local record")

        if mismatches:
            logger.error("HEALTH ALERT: local/broker position mismatch: %s", mismatches)
        return mismatches
