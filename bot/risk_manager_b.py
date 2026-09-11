"""
Strategy B sizing: portfolio-level volatility targeting layered on top of the
same ATR-risk-unit base sizing used by Strategy A, with the mandated caps.
"""
import logging

logger = logging.getLogger(__name__)

TARGET_VOL = 0.10  # annualized
MAX_GROSS_EXPOSURE = 1.0    # x equity
MAX_POSITION_PCT = 0.30     # x equity, per instrument
DRAWDOWN_FREEZE_PCT = 0.05
SCALE_CAP_CONSERVATIVE = 1.0   # early-rollout cap: never scale UP
SCALE_CAP_NORMAL = 1.5
SCALE_FLOOR = 0.25
VOL_PERCENTILE_FLOOR = 0.10     # below this historical percentile, don't scale above 1.0x


class VolTargetRiskManager:
    def __init__(self, target_vol: float = TARGET_VOL, max_gross_exposure: float = MAX_GROSS_EXPOSURE,
                 max_position_pct: float = MAX_POSITION_PCT, base_risk_pct: float = 0.01):
        self.target_vol = target_vol
        self.max_gross_exposure = max_gross_exposure
        self.max_position_pct = max_position_pct
        self.base_risk_pct = base_risk_pct

    def scaling_factor(self, realized_vol_20d_ewma: float, current_drawdown_pct: float,
                        instrument_vol_percentile: float = None, conservative_mode: bool = False) -> float:
        """target_vol / realized_vol, clamped by all mandatory caps."""
        if realized_vol_20d_ewma is None or realized_vol_20d_ewma != realized_vol_20d_ewma or realized_vol_20d_ewma <= 0:
            return 1.0

        raw = self.target_vol / realized_vol_20d_ewma
        cap = SCALE_CAP_CONSERVATIVE if conservative_mode else SCALE_CAP_NORMAL

        if current_drawdown_pct is not None and current_drawdown_pct > DRAWDOWN_FREEZE_PCT:
            cap = min(cap, 1.0)
        if instrument_vol_percentile is not None and instrument_vol_percentile < VOL_PERCENTILE_FLOOR:
            cap = min(cap, 1.0)

        scaled = min(raw, cap)
        scaled = max(scaled, SCALE_FLOOR)
        return scaled

    def hard_stop_price(self, entry_price: float, atr: float, direction: str) -> float:
        return entry_price - atr if direction == "long" else entry_price + atr

    def trailing_stop_price(self, extreme_price: float, atr: float, direction: str, trail_mult: float) -> float:
        if direction == "long":
            return extreme_price - trail_mult * atr
        return extreme_price + trail_mult * atr

    def position_size(self, equity: float, atr: float, price: float, scaling_factor: float,
                       existing_gross_notional: float, fractionable: bool = True) -> float:
        if atr is None or atr != atr or atr <= 0 or price <= 0 or equity <= 0:
            return 0.0

        risk_amount = equity * self.base_risk_pct * max(scaling_factor, 0.0)
        qty = risk_amount / atr

        max_instrument_notional = self.max_position_pct * equity
        if qty * price > max_instrument_notional:
            qty = max_instrument_notional / price

        max_gross_notional = self.max_gross_exposure * equity
        available_notional = max(0.0, max_gross_notional - existing_gross_notional)
        if qty * price > available_notional:
            qty = available_notional / price

        qty = float(int(qty)) if not fractionable else round(qty, 6)
        return max(qty, 0.0)
