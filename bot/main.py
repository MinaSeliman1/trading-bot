"""
Entry point: wires together strategies, risk management, and the portfolio,
then runs a continuous loop that checks stops every cycle and evaluates each
instrument's strategy once its own timeframe has elapsed.
"""
import logging
import signal as signal_module
import sys
import time

import alpaca_trade_api as tradeapi

import config
from bot.portfolio import Portfolio
from bot.risk_manager import RiskManager, calculate_atr
from bot.strategies.mean_reversion import MeanReversionStrategy
from bot.strategies.momentum_breakout import MomentumBreakoutStrategy
from bot.strategies.trend_following import TrendFollowingStrategy
from bot.utils import fetch_bars, required_bar_count, retry_api_call, timeframe_seconds

logger = logging.getLogger("trading_bot")

STRATEGY_FACTORIES = {
    "mean_reversion": lambda p: MeanReversionStrategy(
        sma_period=p.get("sma_period", 20), std_dev_mult=p.get("std_dev_mult", 1.5)
    ),
    "momentum_breakout": lambda p: MomentumBreakoutStrategy(
        lookback=p.get("lookback", 20), volume_mult=p.get("volume_mult", 1.5)
    ),
    "trend_following": lambda p: TrendFollowingStrategy(
        fast_ema=p.get("fast_ema", 50), slow_ema=p.get("slow_ema", 200)
    ),
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("bot.log")],
    )


class TradingBot:
    def __init__(self):
        # Strategies A (mean_reversion/momentum_breakout/trend_following, wired
        # below) and B were both invalidated by walk-forward backtesting -- see
        # comparison_report.md: out-of-sample Sharpe -3.46/-2.51, max drawdown
        # 97%/43%, and the edge does not survive any transaction-cost scenario
        # tested. Disabled until a viable strategy (see exploration_log_pead.md)
        # is wired into this entry point.
        raise RuntimeError(
            "Live trading is disabled: strategies A and B failed walk-forward "
            "validation (comparison_report.md). Do not remove this guard until "
            "a validated strategy replaces the wiring below."
        )

        if not config.ALPACA_API_KEY or not config.ALPACA_API_SECRET:
            raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET -- fill in .env first")

        self.api = tradeapi.REST(
            key_id=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_API_SECRET,
            base_url=config.ALPACA_BASE_URL,
        )
        self.risk_manager = RiskManager(config.RISK_PER_TRADE_PCT, config.HARD_STOP_PCT)
        self.portfolio = Portfolio(self.api, self.risk_manager, config.TRADES_CSV, config.DAILY_PNL_CSV)

        self.strategies = {
            key: STRATEGY_FACTORIES[cfg.strategy](cfg.params) for key, cfg in config.INSTRUMENTS.items()
        }
        self.last_signal_check = {key: None for key in config.INSTRUMENTS}

        self._running = True
        self._clock_cache = None
        self._clock_cache_time = 0.0

    def stop(self, *_args) -> None:
        logger.info("Shutdown signal received; stopping after this cycle")
        self._running = False

    # ------------------------------------------------------------------ #
    # market hours
    # ------------------------------------------------------------------ #
    def _market_open(self) -> bool:
        now = time.time()
        if self._clock_cache is None or now - self._clock_cache_time > 30:
            try:
                self._clock_cache = retry_api_call(self.api.get_clock)
                self._clock_cache_time = now
            except Exception:
                logger.exception("Failed to fetch market clock; treating equities market as closed")
                return False
        return bool(self._clock_cache.is_open)

    def _tradable_now(self, cfg) -> bool:
        return True if cfg.asset_class == "crypto" else self._market_open()

    # ------------------------------------------------------------------ #
    # correlation filter
    # ------------------------------------------------------------------ #
    def _correlation_blocks_long(self, key: str) -> bool:
        if key not in config.RISK_ON_SENSITIVE:
            return False
        if all(self.portfolio.is_long(sym) for sym in config.RISK_ON_EQUITY):
            logger.info(
                "Correlation filter: %s all long, blocking new long on %s",
                config.RISK_ON_EQUITY, key,
            )
            return True
        return False

    # ------------------------------------------------------------------ #
    # per-cycle work
    # ------------------------------------------------------------------ #
    def check_stops(self) -> None:
        """Every open position is primarily protected by a broker-side stop order
        (see Portfolio.sync_protective_stop). Each cycle we: (1) check whether that
        order has already filled, (2) recompute trailing stops and re-place the
        broker order if it moved, and (3) as a last-resort software fallback, force
        a market close if price has already gapped through the stop before the
        broker order caught up (e.g. a crypto stop-limit that got skipped over)."""
        for key in list(self.portfolio.positions.keys()):
            cfg = config.INSTRUMENTS[key]
            if not self._tradable_now(cfg):
                continue
            try:
                status = self.portfolio.sync_stop_order(key)
                if status == "filled":
                    continue

                bars = fetch_bars(self.api, cfg, limit=config.ATR_PERIOD + 10, feed=config.ALPACA_DATA_FEED)
                if bars.empty:
                    continue
                pos = self.portfolio.positions.get(key)
                if pos is None:
                    continue
                price = float(bars["close"].iloc[-1])
                atr = calculate_atr(bars, config.ATR_PERIOD)
                pos.update_trailing(price, atr, self.risk_manager)

                if pos.stop_hit(price):
                    logger.warning(
                        "Software stop-hit fallback for %s at %.4f (stop=%.4f); "
                        "broker-side stop order did not fill in time",
                        key, price, pos.current_stop(),
                    )
                    self.portfolio.close_position(key, price, reason="stop_fallback")
                    continue

                self.portfolio.sync_protective_stop(key)
            except Exception:
                logger.exception("Error checking stop for %s", key)

    def evaluate_instrument(self, key: str) -> None:
        cfg = config.INSTRUMENTS[key]
        strategy = self.strategies[key]
        try:
            bars = fetch_bars(self.api, cfg, limit=required_bar_count(cfg), feed=config.ALPACA_DATA_FEED)
            if bars.empty or len(bars) < config.ATR_PERIOD + 1:
                logger.warning("Not enough bar history for %s yet", key)
                return

            pos = self.portfolio.positions.get(key)
            current_direction = pos.direction if pos else None
            action = strategy.evaluate(bars, current_direction)
            if action is None:
                return

            price = float(bars["close"].iloc[-1])

            if action == "exit":
                logger.info("Signal EXIT for %s at %.4f", key, price)
                self.portfolio.close_position(key, price, reason="signal")
                return

            if pos is not None and pos.direction != action:
                logger.info("Signal flip for %s: %s -> %s", key, pos.direction, action)
                self.portfolio.close_position(key, price, reason="flip")
                pos = None

            if pos is not None:
                return  # already positioned in the desired direction

            if action == "long" and self._correlation_blocks_long(key):
                return

            atr = calculate_atr(bars, config.ATR_PERIOD)
            if atr != atr or atr <= 0:
                logger.warning("Invalid ATR for %s, skipping entry", key)
                return

            equity = self.portfolio.get_equity()
            qty = self.risk_manager.position_size(equity, atr, price, fractionable=cfg.fractionable)
            if qty <= 0:
                logger.warning("Computed zero position size for %s, skipping entry", key)
                return

            trail_mult = cfg.params.get("trailing_atr_mult")
            logger.info("Opening %s %s: qty=%s price=%.4f atr=%.4f", action.upper(), key, qty, price, atr)
            self.portfolio.open_position(key, cfg.symbol, action, qty, price, atr, trail_mult)

        except Exception:
            logger.exception("Error evaluating strategy for %s", key)

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        setup_logging()
        logger.info("Trading bot starting. Instruments: %s", list(config.INSTRUMENTS.keys()))
        signal_module.signal(signal_module.SIGINT, self.stop)
        signal_module.signal(signal_module.SIGTERM, self.stop)

        self.portfolio.reconcile(config.INSTRUMENTS, config.ATR_PERIOD)

        while self._running:
            cycle_start = time.time()
            try:
                self.check_stops()
                for key, cfg in config.INSTRUMENTS.items():
                    if not self._tradable_now(cfg):
                        continue
                    interval = timeframe_seconds(cfg.timeframe)
                    last = self.last_signal_check[key]
                    if last is None or cycle_start - last >= interval:
                        self.evaluate_instrument(key)
                        self.last_signal_check[key] = cycle_start
            except Exception:
                logger.exception("Unhandled error in main loop; continuing")

            elapsed = time.time() - cycle_start
            time.sleep(max(1.0, config.POLL_INTERVAL_SECONDS - elapsed))

        logger.info("Trading bot stopped.")


if __name__ == "__main__":
    TradingBot().run()
