"""
Strategy B - Trend Following (GLD, USO), regime-confirmed.

Same 50/200 EMA crossover as Strategy A, but a crossover only produces a
signal when the regime detector agrees the market is TRENDING at that bar --
an EMA cross during a NEUTRAL/RANGING ADX reading is treated as noise.
"""
import logging
from typing import Optional

import pandas as pd

from bot.regime_detector import TRENDING

logger = logging.getLogger(__name__)


class TrendFollowingStrategyB:
    def __init__(self, fast_ema: int = 50, slow_ema: int = 200):
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema

    def evaluate(self, bars: pd.DataFrame, current_direction: Optional[str], regime: str) -> Optional[str]:
        if len(bars) < self.slow_ema + 1:
            return None

        closes = bars["close"]
        fast = closes.ewm(span=self.fast_ema, adjust=False).mean()
        slow = closes.ewm(span=self.slow_ema, adjust=False).mean()

        fast_prev, fast_now = fast.iloc[-2], fast.iloc[-1]
        slow_prev, slow_now = slow.iloc[-2], slow.iloc[-1]

        crossed_up = fast_prev <= slow_prev and fast_now > slow_now
        crossed_down = fast_prev >= slow_prev and fast_now < slow_now
        regime_confirms = regime == TRENDING

        if current_direction is None:
            if crossed_up and regime_confirms:
                return "long"
            if crossed_down and regime_confirms:
                return "short"
            return None

        if current_direction == "long" and crossed_down:
            return "exit"
        if current_direction == "short" and crossed_up:
            return "exit"
        return None
