"""
Order-level guards, checked before ANY short is ever submitted. Built ahead
of live order placement so the policy is enforced at the API-call boundary,
not just inside strategy logic.

Policy:
- Crypto (BTC/USD) is never shortable on Alpaca. A bearish signal on a crypto
  instrument must always resolve to an exit-to-cash instruction, never a
  short order -- this is a hard, unconditional block, not a query.
- Equities may only be shorted if the account has margin/shorting enabled
  AND the specific asset reports shortable=True via a live Alpaca query.
  Never approximate this with a hardcoded dollar threshold or asset list.
"""
import logging

logger = logging.getLogger(__name__)


class ShortNotAllowedError(Exception):
    pass


class OrderGuards:
    def __init__(self, trading_client):
        self.trading_client = trading_client

    def check_short_allowed(self, symbol: str, asset_class: str) -> tuple:
        """Returns (allowed: bool, reason: str). Never raises on a normal
        disallowed case -- callers decide what to do (e.g. force an exit)."""
        if asset_class == "crypto":
            return False, "crypto is not shortable/marginable on Alpaca"

        try:
            account = self.trading_client.get_account()
            if not bool(account.shorting_enabled):
                return False, "account shorting_enabled=False"

            asset = self.trading_client.get_asset(symbol)
            if not bool(asset.shortable):
                return False, f"{symbol} asset.shortable=False"
            if not bool(asset.marginable):
                return False, f"{symbol} asset.marginable=False"
        except Exception:
            logger.exception("Could not verify shortability for %s; blocking short as a fail-safe", symbol)
            return False, "shortability check failed (API error) -- blocked fail-safe"

        return True, "ok"

    def resolve_bearish_signal(self, symbol: str, asset_class: str, has_open_long: bool) -> str:
        """Translate a strategy's raw bearish signal into a safe order-level
        action: "short", or "exit" if shorting isn't permitted right now."""
        if asset_class == "crypto":
            if has_open_long:
                logger.info("Bearish signal on crypto %s -> exit-to-cash (shorting blocked)", symbol)
                return "exit"
            logger.info("Bearish signal on crypto %s with no open long -> no action (shorting blocked)", symbol)
            return "none"

        allowed, reason = self.check_short_allowed(symbol, asset_class)
        if allowed:
            return "short"
        logger.info("Bearish signal on %s -> %s (short blocked: %s)", symbol,
                    "exit" if has_open_long else "none", reason)
        return "exit" if has_open_long else "none"

    def submit_short_order(self, submit_fn, symbol: str, asset_class: str, **order_kwargs):
        """Hard gate in front of any short submission. Raises ShortNotAllowedError
        instead of calling `submit_fn` if the guard fails -- this is the
        API-call-level backstop referenced by the BTC policy above."""
        allowed, reason = self.check_short_allowed(symbol, asset_class)
        if not allowed:
            raise ShortNotAllowedError(f"Refusing to submit short order for {symbol}: {reason}")
        return submit_fn(symbol=symbol, **order_kwargs)
