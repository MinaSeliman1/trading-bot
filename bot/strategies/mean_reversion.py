"""
Strategy 1 - Mean Reversion (SPY, QQQ on 15-minute candles).

Entry: price beyond `std_dev_mult` standard deviations from the rolling SMA
fades back toward the mean -- long below the lower band, short above the
upper band. Exit: price reverts back to the SMA.
"""
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class MeanReversionStrategy:
    def __init__(self, sma_period: int = 20, std_dev_mult: float = 1.5):
        self.sma_period = sma_period
        self.std_dev_mult = std_dev_mult

    def evaluate(self, bars: pd.DataFrame, current_direction: Optional[str]) -> Optional[str]:
        """Returns "long", "short", "exit", or None."""
        if len(bars) < self.sma_period + 1:
            return None

        closes = bars["close"]
        sma = closes.rolling(self.sma_period).mean().iloc[-1]
        std = closes.rolling(self.sma_period).std().iloc[-1]
        price = closes.iloc[-1]

        if pd.isna(sma) or pd.isna(std) or std == 0:
            return None

        upper = sma + self.std_dev_mult * std
        lower = sma - self.std_dev_mult * std

        if current_direction is None:
            if price <= lower:
                return "long"
            if price >= upper:
                return "short"
            return None

        if current_direction == "long" and price >= sma:
            return "exit"
        if current_direction == "short" and price <= sma:
            return "exit"
        return None
