"""Instrument specifications and cost models.

Every number here is an ASSUMPTION documented in README.md (section "Hypotheses
de coûts"). None of them was measured on live fills. They are parameters, not
facts: change them in one place and every result changes consistently.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    symbol: str
    kind: str                       # "etf" | "future"
    tick: float                     # minimum price increment
    point_value: float              # USD per 1.00 of price per unit (share = 1.0)
    comm_per_unit_side: float       # USD per share/contract per order
    comm_min_per_order: float = 0.0
    comm_max_pct_notional: float = 0.0   # cap as a fraction of order value (0 = none)
    margin_per_unit: float = 0.0    # USD initial margin per contract (futures only)

    @property
    def tick_value(self) -> float:
        return self.tick * self.point_value

    def commission(self, units: int, price: float) -> float:
        """Commission for ONE order (one side)."""
        if units <= 0:
            return 0.0
        c = units * self.comm_per_unit_side
        c = max(c, self.comm_min_per_order)
        if self.comm_max_pct_notional > 0:
            c = min(c, self.comm_max_pct_notional * units * price * self.point_value)
        return c

    def round_trip_commission_per_unit(self) -> float:
        """Smooth (no minimum) per-unit round-trip commission, used only for the
        first-pass position size; the exact check afterwards uses commission()."""
        return 2.0 * self.comm_per_unit_side


# --- Defaults -------------------------------------------------------------
# ETFs: IBKR-Pro-Fixed-like schedule ($0.005/share, $1.00 min/order, 1% cap).
#   Regulatory fees (SEC/FINRA TAF) are NOT modelled: they are fractions of a
#   cent per share and are dominated by the 1-tick slippage assumption.
# Futures: $2.25 per contract per side, all-in retail estimate (broker +
#   exchange + clearing + NFA). Margin: conservative exchange-style initial
#   margin; it only acts as an exposure cap (see sizing).
SPY = Instrument("SPY", "etf", 0.01, 1.0, 0.005, 1.00, 0.01)
QQQ = Instrument("QQQ", "etf", 0.01, 1.0, 0.005, 1.00, 0.01)
ES = Instrument("ES", "future", 0.25, 50.0, 2.25, 0.0, 0.0, 15_000.0)
NQ = Instrument("NQ", "future", 0.25, 20.0, 2.25, 0.0, 0.0, 25_000.0)

INSTRUMENTS = {i.symbol: i for i in (SPY, QQQ, ES, NQ)}
