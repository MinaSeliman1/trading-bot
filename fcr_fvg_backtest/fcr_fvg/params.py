"""Strategy and simulation parameters.

The MAIN configuration is the dataclass default. It is fixed a priori (from the
specification) and is never tuned on results. Sensitivity runs change ONE field
at a time and are reported as sensitivity tests, never as a selection.
"""
from __future__ import annotations

from dataclasses import dataclass

TZ = "America/New_York"
OPEN_MINUTE = 9 * 60 + 30      # 09:30
OR_BARS = 5                    # 09:30-09:35 built from five 1-minute bars


@dataclass(frozen=True)
class StrategyParams:
    # --- rules ---------------------------------------------------------
    stop_variant: str = "A"            # "A" retest-extreme stop, "B" FVG-boundary stop
    fvg_ttl_bars: int = 15             # retest must occur within 15 bars after FVG completion (ASSUMPTION)
    confirm_extra_bars: int = 2        # engulfing on the retest bar or the next 2 bars (ASSUMPTION)
    entry_cutoff_idx: int = 90         # minutes after 09:30; fill bar must start BEFORE 11:00
    rr: float = 3.0
    stop_buffer_ticks: int = 1
    # --- execution -----------------------------------------------------
    slippage_ticks: int = 1            # adverse, applied to EVERY execution (entry and exit)
    check_exit_on_entry_bar: bool = True
    # --- capital / sizing ----------------------------------------------
    initial_capital: float = 100_000.0
    risk_pct: float = 0.005
    max_leverage_etf: float = 4.0      # intraday notional cap (Reg-T day-trading buying power)
    max_margin_usage: float = 0.5      # fraction of equity usable as futures initial margin (exposure cap)
    # --- data-quality policy (decided before running, never tuned) -------
    max_missing_signal_window: int = 0     # missing 1-min bars tolerated in [09:30, 11:00)
    end_tolerance_min: int = 5             # last bar may be at most this early vs. official close

    def with_(self, **kw) -> "StrategyParams":
        from dataclasses import replace
        return replace(self, **kw)

    def __post_init__(self):
        if self.stop_variant not in ("A", "B"):
            raise ValueError("stop_variant must be 'A' or 'B'")
        if self.fvg_ttl_bars < 1 or self.confirm_extra_bars < 0:
            raise ValueError("invalid ttl/confirm window")
        if self.slippage_ticks < 0:
            raise ValueError("slippage_ticks must be >= 0")
