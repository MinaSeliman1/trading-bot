"""
Strategy B - Momentum Breakout (BTC/USD).

Same 20-period high/low breakout + volume confirmation as Strategy A, but a
bearish breakout NEVER produces a short -- Alpaca does not support shorting
crypto. A bearish breakout while long triggers an exit-to-cash; while flat
it produces no action at all. This mirrors the hard block in
bot/order_guards.py, enforced here at the strategy level as the primary
control (the order guard is the safety net, not the primary gate).
"""
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class MomentumBreakoutStrategyB:
    def __init__(self, lookback: int = 20, volume_mult: float = 1.5):
        self.lookback = lookback
        self.volume_mult = volume_mult

    def evaluate(self, bars: pd.DataFrame, current_direction: Optional[str]) -> Optional[str]:
        """Returns "long", "exit", or None. Never "short" -- see module docstring."""
        if len(bars) < self.lookback + 1:
            return None

        prior = bars.iloc[:-1]
        last = bars.iloc[-1]

        highest = prior["high"].tail(self.lookback).max()
        lowest = prior["low"].tail(self.lookback).min()
        avg_volume = prior["volume"].tail(self.lookback).mean()

        if pd.isna(highest) or pd.isna(lowest) or pd.isna(avg_volume) or avg_volume == 0:
            return None

        volume_confirmed = last["volume"] >= self.volume_mult * avg_volume
        breakout_up = last["close"] > highest and volume_confirmed
        breakout_down = last["close"] < lowest and volume_confirmed

        if current_direction is None:
            return "long" if breakout_up else None

        if current_direction == "long" and breakout_down:
            return "exit"
        return None
