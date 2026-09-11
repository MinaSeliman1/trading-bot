"""
Central configuration: environment/credentials, instrument universe, and
risk-management constants. Nothing in here talks to the network.
"""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Alpaca connection
# ---------------------------------------------------------------------------
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex")  # "iex" (free tier) or "sip"
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")

# ---------------------------------------------------------------------------
# Loop / logging
# ---------------------------------------------------------------------------
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "30"))
TRADES_CSV = os.getenv("TRADES_CSV", "trades.csv")
DAILY_PNL_CSV = os.getenv("DAILY_PNL_CSV", "daily_pnl.csv")

# ---------------------------------------------------------------------------
# Risk management
# ---------------------------------------------------------------------------
# Position sizing rule: a 1-ATR adverse move costs exactly this fraction of equity.
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.01"))
# Hard stop-loss cap expressed as a fraction of equity (kept equal to the ATR risk
# fraction so the hard stop sits exactly 1 ATR away from entry -- see risk_manager.py).
HARD_STOP_PCT = float(os.getenv("HARD_STOP_PCT", "0.01"))
ATR_PERIOD = 14


@dataclass
class InstrumentConfig:
    symbol: str            # Alpaca trading/data symbol, e.g. "SPY" or "BTC/USD"
    asset_class: str       # "us_equity" or "crypto"
    strategy: str          # "mean_reversion" | "momentum_breakout" | "trend_following"
    timeframe: str         # "15Min" | "1Hour" | "4Hour"
    fractionable: bool = True
    params: dict = field(default_factory=dict)


INSTRUMENTS = {
    "SPY": InstrumentConfig(
        symbol="SPY",
        asset_class="us_equity",
        strategy="mean_reversion",
        timeframe="15Min",
        params={"sma_period": 20, "std_dev_mult": 1.5},
    ),
    "QQQ": InstrumentConfig(
        symbol="QQQ",
        asset_class="us_equity",
        strategy="mean_reversion",
        timeframe="15Min",
        params={"sma_period": 20, "std_dev_mult": 1.8},
    ),
    "BTCUSD": InstrumentConfig(
        symbol="BTC/USD",
        asset_class="crypto",
        strategy="momentum_breakout",
        timeframe="1Hour",
        params={"lookback": 20, "volume_mult": 1.5, "trailing_atr_mult": 2.0},
    ),
    "GLD": InstrumentConfig(
        symbol="GLD",
        asset_class="us_equity",
        strategy="trend_following",
        timeframe="4Hour",
        params={"fast_ema": 50, "slow_ema": 200, "trailing_atr_mult": 3.0},
    ),
    "USO": InstrumentConfig(
        symbol="USO",
        asset_class="us_equity",
        strategy="trend_following",
        timeframe="4Hour",
        params={"fast_ema": 50, "slow_ema": 200, "trailing_atr_mult": 3.0},
    ),
}

# Correlation filter: instruments whose *long* exposure is considered "risk-on
# equity beta". If every symbol in this list is already long, new longs on the
# symbols in RISK_ON_SENSITIVE are blocked (see bot/main.py).
RISK_ON_EQUITY = ["SPY", "QQQ"]
RISK_ON_SENSITIVE = ["BTCUSD"]
