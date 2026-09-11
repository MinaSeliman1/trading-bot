"""
Final safety layer. Usable both live and inside the backtest simulator so a
Strategy B backtest reflects the same halts/cuts it would face in production.

- Portfolio drawdown > 10% from peak: halt all new trading, requires an
  explicit resume() (simulating manual review) -- never auto-clears.
- Realized portfolio vol > 2x target for 3 consecutive observations: cut all
  new position sizes by 50% until vol normalizes.
- API error-rate breach (from health_check): halts new orders only; existing
  positions keep being monitored by the caller.
- Local/broker position mismatch: halts immediately, no auto-reconciliation.
- Duplicate order for the same (symbol, direction) within a short window:
  rejected outright.
"""
import logging
import subprocess
import sys
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

DRAWDOWN_HALT_PCT = 0.10
VOL_BREACH_MULTIPLE = 2.0
VOL_BREACH_CONSECUTIVE_DAYS = 3
VOL_BREACH_SIZE_CUT = 0.5
DUPLICATE_ORDER_WINDOW_SECONDS = 60


class KillSwitch:
    def __init__(self):
        self.peak_equity = None
        self.halted_for_drawdown = False
        self.halted_for_mismatch = False
        self.halted_for_api_errors = False
        self._vol_breach_streak = 0
        self._vol_cut_active = False
        self._last_order_time: dict = {}

    # ------------------------------------------------------------------ #
    # drawdown halt
    # ------------------------------------------------------------------ #
    def update_drawdown(self, equity: float) -> bool:
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity
        drawdown = 0.0 if self.peak_equity in (None, 0) else (self.peak_equity - equity) / self.peak_equity
        if drawdown > DRAWDOWN_HALT_PCT and not self.halted_for_drawdown:
            self.halted_for_drawdown = True
            logger.error(
                "KILL SWITCH: drawdown %.2f%% exceeds %.0f%% -- halting all new trading. "
                "Manual review required to resume.", drawdown * 100, DRAWDOWN_HALT_PCT * 100,
            )
        return self.halted_for_drawdown

    def current_drawdown(self, equity: float) -> float:
        if self.peak_equity in (None, 0):
            return 0.0
        return max(0.0, (self.peak_equity - equity) / self.peak_equity)

    def resume_from_drawdown_halt(self) -> None:
        logger.warning("KILL SWITCH: drawdown halt manually cleared by operator")
        self.halted_for_drawdown = False

    # ------------------------------------------------------------------ #
    # volatility breach -> size cut (not a halt)
    # ------------------------------------------------------------------ #
    def update_volatility(self, realized_vol: float, target_vol: float) -> float:
        """Call once per period (e.g. once per trading day). Returns the size
        multiplier to apply to new positions (1.0 normally, 0.5 once 3
        consecutive breaches of 2x target vol have occurred)."""
        if realized_vol == realized_vol and target_vol > 0 and realized_vol > VOL_BREACH_MULTIPLE * target_vol:
            self._vol_breach_streak += 1
        else:
            self._vol_breach_streak = 0
            self._vol_cut_active = False

        if self._vol_breach_streak >= VOL_BREACH_CONSECUTIVE_DAYS and not self._vol_cut_active:
            self._vol_cut_active = True
            logger.warning(
                "KILL SWITCH: realized vol > %.1fx target for %d consecutive periods -- cutting new size by %.0f%%",
                VOL_BREACH_MULTIPLE, self._vol_breach_streak, (1 - VOL_BREACH_SIZE_CUT) * 100,
            )
        return VOL_BREACH_SIZE_CUT if self._vol_cut_active else 1.0

    # ------------------------------------------------------------------ #
    # API errors / position mismatch
    # ------------------------------------------------------------------ #
    def set_api_error_halt(self, breached: bool) -> None:
        if breached and not self.halted_for_api_errors:
            logger.error("KILL SWITCH: API error-rate threshold breached -- halting new orders")
        elif not breached and self.halted_for_api_errors:
            logger.info("KILL SWITCH: API error-rate back under threshold -- new orders re-enabled")
        self.halted_for_api_errors = breached

    def set_mismatch_halt(self, mismatches: list) -> None:
        if mismatches and not self.halted_for_mismatch:
            logger.error("KILL SWITCH: local/broker position mismatch -- halting immediately: %s", mismatches)
        self.halted_for_mismatch = bool(mismatches)

    def resume_from_mismatch_halt(self) -> None:
        logger.warning("KILL SWITCH: position-mismatch halt manually cleared by operator")
        self.halted_for_mismatch = False

    # ------------------------------------------------------------------ #
    # duplicate order guard
    # ------------------------------------------------------------------ #
    def check_duplicate_order(self, symbol: str, direction: str, now: datetime = None) -> bool:
        """Returns True if this order should be REJECTED as a duplicate."""
        now = now or datetime.now(timezone.utc)
        key = (symbol, direction)
        last = self._last_order_time.get(key)
        if last is not None and (now - last) < timedelta(seconds=DUPLICATE_ORDER_WINDOW_SECONDS):
            logger.error("KILL SWITCH: duplicate order rejected for %s %s within %ds window",
                         symbol, direction, DUPLICATE_ORDER_WINDOW_SECONDS)
            return True
        self._last_order_time[key] = now
        return False

    # ------------------------------------------------------------------ #
    # overall gate
    # ------------------------------------------------------------------ #
    def can_open_new_positions(self) -> bool:
        return not (self.halted_for_drawdown or self.halted_for_mismatch or self.halted_for_api_errors)


def run_startup_test_gate(test_path: str = "tests", python_executable: str = None) -> bool:
    """Run the test suite and refuse to start the trading loop if anything fails."""
    python_executable = python_executable or sys.executable
    try:
        result = subprocess.run(
            [python_executable, "-m", "pytest", test_path, "-q"],
            capture_output=True, text=True, timeout=600,
        )
    except Exception:
        logger.exception("Could not run startup test gate")
        return False

    if result.returncode != 0:
        logger.error("STARTUP TEST GATE FAILED -- refusing to start trading loop.\n%s", result.stdout[-4000:])
        return False
    logger.info("Startup test gate passed.")
    return True
