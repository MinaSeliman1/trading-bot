"""
Strategy 2 - Momentum Breakout (BTC/USD on 1-hour candles).

Entry: close breaks above the prior `lookback`-period high (long) or below
the prior `lookback`-period low (short), confirmed by volume at least
`volume_mult`x the `lookback`-period average volume. An opposite-direction
breakout exits (or reverses) an open position. Trailing-stop management
(2x ATR) is handled by the portfolio/risk manager, not here.
"""
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class MomentumBreakoutStrategy:
    def __init__(self, lookback: int = 20, volume_mult: float = 1.5):
        self.lookback = lookback
        self.volume_mult = volume_mult

    def evaluate(self, bars: pd.DataFrame, current_direction: Optional[str]) -> Optional[str]:
        """Returns "long", "short", "exit", or None."""
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
            if breakout_up:
                return "long"
            if breakout_down:
                return "short"
            return None

        if current_direction == "long" and breakout_down:
            return "exit"
        if current_direction == "short" and breakout_up:
            return "exit"
        return None
