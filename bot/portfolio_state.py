"""
Phase 1 portfolio state: read-only equity/position/P&L/volatility snapshot
via alpaca-py's TradingClient. No orders are placed from this module.
"""
import csv
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetPortfolioHistoryRequest

logger = logging.getLogger(__name__)


class PortfolioState:
    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        self.client = TradingClient(api_key, api_secret, paper=paper)

    def snapshot(self) -> dict:
        account = self.client.get_account()
        positions = self.client.get_all_positions()

        unrealized_pnl = sum(float(p.unrealized_pl) for p in positions)
        position_summaries = [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": p.side.value if hasattr(p.side, "value") else str(p.side),
                "avg_entry_price": float(p.avg_entry_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
            }
            for p in positions
        ]

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "equity": float(account.equity),
            "cash": float(account.cash),
            "unrealized_pnl": unrealized_pnl,
            "positions": position_summaries,
            "realized_vol_20d": self.rolling_realized_vol(lookback_days=20),
        }

    def rolling_realized_vol(self, lookback_days: int = 20) -> float:
        """Annualized realized volatility of daily portfolio equity returns."""
        try:
            req = GetPortfolioHistoryRequest(period=f"{lookback_days + 5}D", timeframe="1D")
            history = self.client.get_portfolio_history(req)
        except Exception:
            logger.exception("Could not fetch portfolio history for realized vol")
            return float("nan")

        equity = pd.Series(history.equity, dtype=float).dropna()
        if len(equity) < 3:
            return float("nan")
        returns = equity.pct_change().dropna().tail(lookback_days)
        if len(returns) < 2:
            return float("nan")
        return float(returns.std() * np.sqrt(252))


def append_daily_state(csv_path: str, equity: float, regimes: dict, realized_vol: float,
                        liquidity_triggers: list) -> None:
    """daily_state.csv: timestamp, equity, per-instrument regime, portfolio realized vol,
    liquidity filter triggers."""
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "equity", "regimes", "realized_vol_20d", "liquidity_triggers"])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            f"{equity:.2f}",
            ";".join(f"{k}={v}" for k, v in regimes.items()),
            f"{realized_vol:.6f}" if realized_vol == realized_vol else "",
            ";".join(liquidity_triggers),
        ])
