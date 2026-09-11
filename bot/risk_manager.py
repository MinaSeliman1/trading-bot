"""
ATR calculation and ATR-based position sizing / stop-loss placement.

Core rule: size every position so that a 1-ATR adverse move costs exactly
`risk_per_trade_pct` of account equity, and place the hard stop exactly 1 ATR
from entry. Those two facts together guarantee the hard stop caps loss at
`risk_per_trade_pct` of equity on every single trade, regardless of how
volatile the instrument is.
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def calculate_atr(bars: pd.DataFrame, period: int = 14) -> float:
    """Wilder's ATR. `bars` must be oldest-first with high/low/close columns."""
    if bars.empty or len(bars) < period + 1:
        return float("nan")

    high, low, close = bars["high"], bars["low"], bars["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return float(atr.iloc[-1])


class RiskManager:
    def __init__(self, risk_per_trade_pct: float, hard_stop_pct: float):
        if abs(risk_per_trade_pct - hard_stop_pct) > 1e-9:
            logger.warning(
                "RISK_PER_TRADE_PCT (%.4f) != HARD_STOP_PCT (%.4f); the hard stop is "
                "always placed 1 ATR from entry, so it will cap loss at %.4f%% of "
                "equity, not %.4f%%.",
                risk_per_trade_pct, hard_stop_pct, risk_per_trade_pct * 100, hard_stop_pct * 100,
            )
        self.risk_per_trade_pct = risk_per_trade_pct
        self.hard_stop_pct = hard_stop_pct

    def position_size(self, equity: float, atr: float, price: float, fractionable: bool) -> float:
        if atr is None or atr != atr or atr <= 0 or price <= 0 or equity <= 0:
            return 0.0

        risk_amount = equity * self.risk_per_trade_pct
        qty = risk_amount / atr

        # Guardrail: never let a single position's notional exceed total equity,
        # even for extremely low-ATR (quiet) instruments.
        qty = min(qty, equity / price)

        qty = float(int(qty)) if not fractionable else round(qty, 6)
        return max(qty, 0.0)

    def hard_stop_price(self, entry_price: float, atr: float, direction: str) -> float:
        return entry_price - atr if direction == "long" else entry_price + atr

    def trailing_stop_price(self, extreme_price: float, atr: float, direction: str, trail_mult: float) -> float:
        if direction == "long":
            return extreme_price - trail_mult * atr
        return extreme_price + trail_mult * atr
