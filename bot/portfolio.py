"""
Portfolio: tracks open positions, submits/closes orders through Alpaca, and
logs every closed trade to trades.csv plus a running daily_pnl.csv.

Every open position is protected by a real broker-side stop order (not just
software monitoring), so the hard stop still applies even if the bot is
offline. As trailing stops ratchet in the caller's favor, the resting stop
order is canceled and replaced at the new price.
"""
import csv
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from bot.risk_manager import calculate_atr
from bot.utils import fetch_bars, retry_api_call

logger = logging.getLogger(__name__)

EASTERN = ZoneInfo("America/New_York")

# Minimum move (as a fraction of price, with a floor) before we bother
# canceling and replacing a resting stop order. Keeps trailing-stop updates
# from spamming cancel/replace on every tiny tick.
STOP_REPLACE_MIN_FRACTION = 0.0005
STOP_REPLACE_MIN_ABS = 0.01

# Buffer applied to crypto protective stops, which must be stop_limit orders
# (Alpaca crypto does not support plain stop-market). Keeps the limit
# marketable once the stop triggers.
CRYPTO_STOP_LIMIT_BUFFER = 0.005


@dataclass
class Position:
    symbol: str
    direction: str  # "long" or "short"
    qty: float
    entry_price: float
    entry_time: datetime
    atr_at_entry: float
    hard_stop: float
    trail_mult: Optional[float]
    extreme_price: float
    trail_stop: Optional[float] = None
    stop_order_id: Optional[str] = None
    stop_order_price: Optional[float] = None

    def current_stop(self) -> float:
        if self.trail_stop is None:
            return self.hard_stop
        if self.direction == "long":
            return max(self.hard_stop, self.trail_stop)
        return min(self.hard_stop, self.trail_stop)

    def update_trailing(self, price: float, atr: float, risk_manager) -> None:
        if self.trail_mult is None or atr is None or atr != atr or atr <= 0:
            return
        if self.direction == "long":
            self.extreme_price = max(self.extreme_price, price)
        else:
            self.extreme_price = min(self.extreme_price, price)

        candidate = risk_manager.trailing_stop_price(self.extreme_price, atr, self.direction, self.trail_mult)
        if self.trail_stop is None:
            self.trail_stop = candidate
        elif self.direction == "long":
            self.trail_stop = max(self.trail_stop, candidate)
        else:
            self.trail_stop = min(self.trail_stop, candidate)

    def stop_hit(self, price: float) -> bool:
        stop = self.current_stop()
        return price <= stop if self.direction == "long" else price >= stop

    def protective_side(self) -> str:
        return "sell" if self.direction == "long" else "buy"


class Portfolio:
    def __init__(self, api, risk_manager, trades_csv: str, daily_pnl_csv: str):
        self.api = api
        self.risk_manager = risk_manager
        self.trades_csv = trades_csv
        self.daily_pnl_csv = daily_pnl_csv
        self.positions: dict[str, Position] = {}
        self._ensure_csv_headers()

    # ------------------------------------------------------------------ #
    # setup / bookkeeping
    # ------------------------------------------------------------------ #
    def _ensure_csv_headers(self) -> None:
        if not os.path.exists(self.trades_csv):
            with open(self.trades_csv, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["timestamp", "instrument", "direction", "entry_price", "exit_price", "pnl", "position_size"]
                )
        if not os.path.exists(self.daily_pnl_csv):
            with open(self.daily_pnl_csv, "w", newline="") as f:
                csv.writer(f).writerow(["date", "realized_pnl", "trades_closed"])

    def reconcile(self, instruments_cfg: dict, atr_period: int) -> None:
        """Pull existing broker positions into local state on startup (e.g. after a restart),
        adopting any protective stop order already resting at the broker, or placing a fresh
        one if none is found."""
        try:
            broker_positions = retry_api_call(self.api.list_positions)
        except Exception:
            logger.exception("Could not fetch existing positions from Alpaca; starting with empty local state")
            return

        symbol_to_key = {cfg.symbol: key for key, cfg in instruments_cfg.items()}
        for bp in broker_positions:
            key = symbol_to_key.get(bp.symbol)
            if key is None or key in self.positions:
                continue

            qty = abs(float(bp.qty))
            direction = "long" if float(bp.qty) > 0 else "short"
            entry_price = float(bp.avg_entry_price)
            cfg = instruments_cfg[key]

            atr = float("nan")
            try:
                bars = fetch_bars(self.api, cfg, limit=atr_period + 10)
                if not bars.empty:
                    atr = calculate_atr(bars, atr_period)
            except Exception:
                logger.exception("Could not compute ATR while reconciling %s", key)

            if atr != atr:  # NaN fallback so a stop always exists
                atr = entry_price * 0.01

            hard_stop = self.risk_manager.hard_stop_price(entry_price, atr, direction)
            pos = Position(
                symbol=bp.symbol,
                direction=direction,
                qty=qty,
                entry_price=entry_price,
                entry_time=datetime.now(timezone.utc),
                atr_at_entry=atr,
                hard_stop=hard_stop,
                trail_mult=cfg.params.get("trailing_atr_mult"),
                extreme_price=entry_price,
            )
            self.positions[key] = pos
            logger.warning(
                "Reconciled existing broker position %s (%s qty=%s entry=%.4f) -> recomputed hard stop %.4f",
                key, direction, qty, entry_price, hard_stop,
            )

            existing = self._find_existing_stop_order(bp.symbol, pos.protective_side())
            if existing is not None:
                pos.stop_order_id = existing.id
                pos.stop_order_price = float(existing.stop_price)
                logger.info(
                    "Adopted existing protective stop order %s for %s at %.4f",
                    existing.id, key, pos.stop_order_price,
                )
            else:
                self.sync_protective_stop(key)

    # ------------------------------------------------------------------ #
    # account / queries
    # ------------------------------------------------------------------ #
    def get_equity(self) -> float:
        account = retry_api_call(self.api.get_account)
        return float(account.equity)

    def is_long(self, key: str) -> bool:
        pos = self.positions.get(key)
        return pos is not None and pos.direction == "long"

    def is_short(self, key: str) -> bool:
        pos = self.positions.get(key)
        return pos is not None and pos.direction == "short"

    def has_position(self, key: str) -> bool:
        return key in self.positions

    # ------------------------------------------------------------------ #
    # protective (broker-side) stop orders
    # ------------------------------------------------------------------ #
    def _find_existing_stop_order(self, symbol: str, protective_side: str):
        try:
            open_orders = retry_api_call(self.api.list_orders, status="open", symbols=[symbol])
        except Exception:
            logger.exception("Could not list open orders for %s while reconciling", symbol)
            return None
        for order in open_orders:
            if order.side == protective_side and order.type in ("stop", "stop_limit"):
                return order
        return None

    def _submit_stop_order(self, symbol: str, direction: str, qty: float, stop_price: float):
        side = "sell" if direction == "long" else "buy"
        is_crypto = "/" in symbol
        try:
            if is_crypto:
                # Alpaca crypto doesn't support plain stop-market orders; use a
                # stop-limit with a small buffer so it's marketable once triggered.
                buffer = stop_price * CRYPTO_STOP_LIMIT_BUFFER
                limit_price = stop_price - buffer if direction == "long" else stop_price + buffer
                return retry_api_call(
                    self.api.submit_order, symbol=symbol, qty=qty, side=side, type="stop_limit",
                    stop_price=round(stop_price, 2), limit_price=round(limit_price, 2), time_in_force="gtc",
                )
            return retry_api_call(
                self.api.submit_order, symbol=symbol, qty=qty, side=side, type="stop",
                stop_price=round(stop_price, 2), time_in_force="gtc",
            )
        except Exception:
            logger.exception("Failed to submit protective stop order for %s", symbol)
            return None

    def _cancel_stop_order(self, pos: Position) -> None:
        if pos.stop_order_id is None:
            return
        try:
            retry_api_call(self.api.cancel_order, pos.stop_order_id, retries=1)
        except Exception:
            logger.debug("Stop order %s already gone (filled/canceled)", pos.stop_order_id)
        pos.stop_order_id = None
        pos.stop_order_price = None

    def sync_protective_stop(self, key: str) -> None:
        """Ensure the resting broker-side stop order matches Position.current_stop().
        No-ops if the order is already close enough, to avoid cancel/replace churn."""
        pos = self.positions.get(key)
        if pos is None:
            return
        target_price = pos.current_stop()
        if pos.stop_order_price is not None:
            threshold = max(target_price * STOP_REPLACE_MIN_FRACTION, STOP_REPLACE_MIN_ABS)
            if abs(target_price - pos.stop_order_price) < threshold:
                return

        if pos.stop_order_id is not None:
            self._cancel_stop_order(pos)

        order = self._submit_stop_order(pos.symbol, pos.direction, pos.qty, target_price)
        if order is not None:
            pos.stop_order_id = order.id
            pos.stop_order_price = target_price
            logger.info("Protective stop for %s now resting at %.4f", key, target_price)

    def sync_stop_order(self, key: str) -> str:
        """Poll the broker-side stop order for a position. Returns "filled" if the
        broker already closed the position, "missing" if no valid stop is resting
        (caller should re-place one), or "active" otherwise."""
        pos = self.positions.get(key)
        if pos is None or pos.stop_order_id is None:
            return "missing"
        try:
            order = retry_api_call(self.api.get_order, pos.stop_order_id, retries=1)
        except Exception:
            logger.exception("Could not fetch stop order status for %s; leaving as-is", key)
            return "active"

        if order.status == "filled":
            filled_price = float(order.filled_avg_price) if order.filled_avg_price else pos.current_stop()
            self._finalize_close(key, filled_price, reason="broker_stop")
            return "filled"

        if order.status in ("canceled", "expired", "rejected"):
            logger.warning("Protective stop order %s for %s is %s; will re-place", pos.stop_order_id, key, order.status)
            pos.stop_order_id = None
            pos.stop_order_price = None
            return "missing"

        return "active"

    # ------------------------------------------------------------------ #
    # order execution
    # ------------------------------------------------------------------ #
    def _await_fill(self, order_id: str, fallback_price: float, attempts: int = 5, delay: float = 1.0) -> float:
        for _ in range(attempts):
            try:
                order = self.api.get_order(order_id)
                if order.status == "filled" and order.filled_avg_price:
                    return float(order.filled_avg_price)
            except Exception:
                logger.exception("Error polling order %s status", order_id)
            time.sleep(delay)
        logger.warning("Order %s not confirmed filled after %s attempts; using reference price", order_id, attempts)
        return fallback_price

    def open_position(self, key: str, symbol: str, direction: str, qty: float,
                       reference_price: float, atr: float, trail_mult: Optional[float]) -> bool:
        side = "buy" if direction == "long" else "sell"
        tif = "gtc" if "/" in symbol else "day"
        try:
            order = retry_api_call(self.api.submit_order, symbol=symbol, qty=qty, side=side,
                                    type="market", time_in_force=tif)
        except Exception:
            logger.exception("Failed to submit entry order for %s; not opening position", key)
            return False

        filled_price = self._await_fill(order.id, fallback_price=reference_price)
        hard_stop = self.risk_manager.hard_stop_price(filled_price, atr, direction)

        self.positions[key] = Position(
            symbol=symbol,
            direction=direction,
            qty=qty,
            entry_price=filled_price,
            entry_time=datetime.now(timezone.utc),
            atr_at_entry=atr,
            hard_stop=hard_stop,
            trail_mult=trail_mult,
            extreme_price=filled_price,
        )
        logger.info("Opened %s %s qty=%s entry=%.4f hard_stop=%.4f", direction.upper(), key, qty, filled_price, hard_stop)

        self.sync_protective_stop(key)
        if self.positions[key].stop_order_id is None:
            logger.error(
                "No broker-side stop could be placed for %s -- position is UNPROTECTED at the broker "
                "until check_stops() software-monitors and force-closes it.", key,
            )
        return True

    def close_position(self, key: str, reference_price: float, reason: str = "") -> None:
        pos = self.positions.get(key)
        if pos is None:
            return

        self._cancel_stop_order(pos)

        side = "sell" if pos.direction == "long" else "buy"
        tif = "gtc" if "/" in pos.symbol else "day"
        filled_price = reference_price
        try:
            order = retry_api_call(self.api.submit_order, symbol=pos.symbol, qty=pos.qty, side=side,
                                    type="market", time_in_force=tif)
            filled_price = self._await_fill(order.id, fallback_price=reference_price)
        except Exception:
            logger.exception("Failed to submit closing order for %s; logging at reference price", key)

        self._finalize_close(key, filled_price, reason)

    def _finalize_close(self, key: str, filled_price: float, reason: str) -> None:
        pos = self.positions.get(key)
        if pos is None:
            return

        if pos.direction == "long":
            pnl = (filled_price - pos.entry_price) * pos.qty
        else:
            pnl = (pos.entry_price - filled_price) * pos.qty

        logger.info(
            "Closed %s %s qty=%s entry=%.4f exit=%.4f pnl=%.2f reason=%s",
            pos.direction.upper(), key, pos.qty, pos.entry_price, filled_price, pnl, reason,
        )
        self._log_trade(key, pos.direction, pos.entry_price, filled_price, pnl, pos.qty)
        self._update_daily_pnl(pnl)
        del self.positions[key]

    # ------------------------------------------------------------------ #
    # CSV logging
    # ------------------------------------------------------------------ #
    def _log_trade(self, key: str, direction: str, entry_price: float, exit_price: float,
                    pnl: float, qty: float) -> None:
        with open(self.trades_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now(timezone.utc).isoformat(),
                key, direction, f"{entry_price:.4f}", f"{exit_price:.4f}", f"{pnl:.2f}", qty,
            ])

    def _update_daily_pnl(self, pnl_delta: float) -> None:
        date_key = datetime.now(EASTERN).date().isoformat()
        rows: dict[str, dict] = {}
        if os.path.exists(self.daily_pnl_csv):
            with open(self.daily_pnl_csv, newline="") as f:
                for row in csv.DictReader(f):
                    rows[row["date"]] = {
                        "realized_pnl": float(row["realized_pnl"]),
                        "trades_closed": int(row["trades_closed"]),
                    }

        entry = rows.get(date_key, {"realized_pnl": 0.0, "trades_closed": 0})
        entry["realized_pnl"] += pnl_delta
        entry["trades_closed"] += 1
        rows[date_key] = entry

        with open(self.daily_pnl_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "realized_pnl", "trades_closed"])
            for d in sorted(rows):
                r = rows[d]
                writer.writerow([d, f"{r['realized_pnl']:.2f}", r["trades_closed"]])
