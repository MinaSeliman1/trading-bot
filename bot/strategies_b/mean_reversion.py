"""
Strategy B - Mean Reversion (SPY, QQQ), regime-gated.

Same SMA/std-dev math as Strategy A, but entries only fire when the
regime detector reports RANGING. An open position exits either when price
reverts to the mean OR when the regime flips to TRENDING (the premise for
mean reversion no longer holds once a trend takes over).
"""
import logging
from typing import Optional

import pandas as pd

from bot.regime_detector import RANGING, TRENDING

logger = logging.getLogger(__name__)


class MeanReversionStrategyB:
    def __init__(self, sma_period: int = 20, std_dev_mult: float = 1.5):
        self.sma_period = sma_period
        self.std_dev_mult = std_dev_mult

    def evaluate(self, bars: pd.DataFrame, current_direction: Optional[str], regime: str) -> Optional[str]:
        if len(bars) < self.sma_period + 1:
            return None

        closes = bars["close"]
        sma = closes.rolling(self.sma_period).mean().iloc[-1]
        std = closes.rolling(self.sma_period).std().iloc[-1]
        price = closes.iloc[-1]

        if pd.isna(sma) or pd.isna(std) or std == 0:
            return None

        if current_direction is not None and regime == TRENDING:
            return "exit"

        upper = sma + self.std_dev_mult * std
        lower = sma - self.std_dev_mult * std

        if current_direction is None:
            if regime != RANGING:
                return None
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
