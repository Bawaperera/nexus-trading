"""
NEXUS — Central Configuration
Reads settings from .env file (or environment variables).
Never hardcode API keys — always use this module.

Usage:
    from config import cfg
    print(cfg.SYMBOL)        # "btcusdt"
    print(cfg.ACCOUNT_SIZE)  # 1000.0
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()  # reads .env file if it exists


@dataclass
class NexusConfig:
    # Exchange
    BINANCE_API_KEY:    str   = os.getenv("BINANCE_API_KEY",    "")
    BINANCE_API_SECRET: str   = os.getenv("BINANCE_API_SECRET", "")

    # News
    CRYPTOPANIC_TOKEN:  str   = os.getenv("CRYPTOPANIC_TOKEN",  "")

    # Trading
    SYMBOL:             str   = os.getenv("TRADING_SYMBOL",     "btcusdt")
    TIMEFRAME:          str   = os.getenv("TRADING_TIMEFRAME",  "1h")
    MODE:               str   = os.getenv("TRADING_MODE",       "paper")
    ACCOUNT_SIZE:       float = float(os.getenv("ACCOUNT_SIZE", "1000"))
    RISK_PCT:           float = float(os.getenv("RISK_PCT",     "0.01"))
    MIN_CONFIDENCE:     float = float(os.getenv("MIN_CONFIDENCE","0.60"))

    def validate(self):
        """Check config is safe before live trading."""
        assert self.MODE in ("paper", "live"), f"Invalid mode: {self.MODE}"
        assert 0 < self.RISK_PCT <= 0.02,      "Risk must be between 0% and 2%"
        assert self.ACCOUNT_SIZE > 0,           "Account size must be positive"

        if self.MODE == "live":
            assert self.BINANCE_API_KEY,    "BINANCE_API_KEY required for live mode"
            assert self.BINANCE_API_SECRET, "BINANCE_API_SECRET required for live mode"
            print("⚠️  LIVE MODE ACTIVE — real money will be traded")
        else:
            print("✅ PAPER MODE — no real money at risk")


# Singleton — import and use anywhere
cfg = NexusConfig()
