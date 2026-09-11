"""
Strategy 3 - Trend Following (GLD, USO on 4-hour candles).

Entry: 50 EMA crosses above the 200 EMA -> long. Exit: 50 EMA crosses below
the 200 EMA (closes a long, or opens a short if flat). Trailing-stop
management (3x ATR) is handled by the portfolio/risk manager, not here.
"""
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class TrendFollowingStrategy:
    def __init__(self, fast_ema: int = 50, slow_ema: int = 200):
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema

    def evaluate(self, bars: pd.DataFrame, current_direction: Optional[str]) -> Optional[str]:
        """Returns "long", "short", "exit", or None."""
        if len(bars) < self.slow_ema + 1:
            return None

        closes = bars["close"]
        fast = closes.ewm(span=self.fast_ema, adjust=False).mean()
        slow = closes.ewm(span=self.slow_ema, adjust=False).mean()

        fast_prev, fast_now = fast.iloc[-2], fast.iloc[-1]
        slow_prev, slow_now = slow.iloc[-2], slow.iloc[-1]

        crossed_up = fast_prev <= slow_prev and fast_now > slow_now
        crossed_down = fast_prev >= slow_prev and fast_now < slow_now

        if current_direction is None:
            if crossed_up:
                return "long"
            if crossed_down:
                return "short"
            return None

        if current_direction == "long" and crossed_down:
            return "exit"
        if current_direction == "short" and crossed_up:
            return "exit"
        return None
