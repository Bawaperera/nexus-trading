"""
NEXUS — Layer 1: Data Engine
Fetches OHLCV data from Binance (crypto) or yfinance (stocks).
Supports multiple timeframes and symbols.
"""

import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class DataEngine:
    """
    Unified data fetcher for crypto (via CCXT/Binance) and stocks (yfinance).
    
    Usage:
        engine = DataEngine(source="crypto")
        df = engine.fetch("BTC/USDT", timeframe="1h", days=365)
    """

    VALID_TIMEFRAMES = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "4h": 240, "1d": 1440
    }

    def __init__(self, source: str = "crypto", api_key: str = None, api_secret: str = None):
        """
        Args:
            source: "crypto" (Binance) or "stock" (yfinance)
            api_key: Binance API key (optional for public data)
            api_secret: Binance API secret (optional for public data)
        """
        self.source = source
        self.exchange = None

        if source == "crypto":
            # We use Binance — no API key needed for historical public data
            self.exchange = ccxt.binance({
                "apiKey": api_key or "",
                "secret": api_secret or "",
                "enableRateLimit": True,
                "options": {"defaultType": "spot"}
            })
            log.info("DataEngine initialized with Binance (CCXT)")
        else:
            log.info("DataEngine initialized with yfinance")

    def fetch(
        self,
        symbol: str,
        timeframe: str = "1h",
        days: int = 365,
        limit: int = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: e.g. "BTC/USDT" for crypto, "AAPL" for stocks
            timeframe: "1m", "5m", "15m", "30m", "1h", "4h", "1d"
            days: how many days of history
            limit: override number of candles (optional)

        Returns:
            DataFrame with columns: open, high, low, close, volume, symbol, timeframe
        """
        if timeframe not in self.VALID_TIMEFRAMES:
            raise ValueError(f"Invalid timeframe. Choose from {list(self.VALID_TIMEFRAMES.keys())}")

        if self.source == "crypto":
            return self._fetch_crypto(symbol, timeframe, days, limit)
        else:
            return self._fetch_stock(symbol, days)

    def _fetch_crypto(self, symbol: str, timeframe: str, days: int, limit: int) -> pd.DataFrame:
        """Fetch crypto OHLCV from Binance via CCXT."""
        try:
            # Calculate number of candles needed
            minutes_per_candle = self.VALID_TIMEFRAMES[timeframe]
            candles_needed = limit or int((days * 24 * 60) / minutes_per_candle)

            # Binance has a 1000-candle limit per request — paginate
            all_candles = []
            since = self.exchange.parse8601(
                (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
            )

            log.info(f"Fetching {candles_needed} candles for {symbol} [{timeframe}]...")

            while len(all_candles) < candles_needed:
                batch = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=since, limit=min(1000, candles_needed - len(all_candles))
                )
                if not batch:
                    break
                all_candles.extend(batch)
                since = batch[-1][0] + 1  # move cursor forward
                time.sleep(self.exchange.rateLimit / 1000)  # respect rate limits

                if len(batch) < 1000:
                    break  # no more data available

            df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp")
            df = df.astype(float)
            df["symbol"] = symbol
            df["timeframe"] = timeframe
            df = df.sort_index().drop_duplicates()

            log.info(f"Fetched {len(df)} candles for {symbol} | {df.index[0]} → {df.index[-1]}")
            return df

        except ccxt.NetworkError as e:
            log.error(f"Network error fetching {symbol}: {e}")
            raise
        except ccxt.ExchangeError as e:
            log.error(f"Exchange error fetching {symbol}: {e}")
            raise

    def _fetch_stock(self, ticker: str, days: int) -> pd.DataFrame:
        """Fetch stock OHLCV from yfinance."""
        try:
            end = datetime.now()
            start = end - timedelta(days=days)
            df = yf.download(ticker, start=start, end=end, progress=False)

            if df.empty:
                raise ValueError(f"No data returned for ticker {ticker}")

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.columns = [c.lower() for c in df.columns]
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.rename(columns={"adj close": "close"}) if "adj close" in df.columns else df
            df["symbol"] = ticker
            df["timeframe"] = "1d"
            df = df[["open", "high", "low", "close", "volume", "symbol", "timeframe"]]
            df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})

            log.info(f"Fetched {len(df)} bars for {ticker} | {df.index[0].date()} → {df.index[-1].date()}")
            return df

        except Exception as e:
            log.error(f"Error fetching stock {ticker}: {e}")
            raise

    def fetch_multiple(self, symbols: list, timeframe: str = "1h", days: int = 365) -> dict:
        """
        Fetch data for multiple symbols.

        Returns:
            dict of {symbol: DataFrame}
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.fetch(symbol, timeframe, days)
            except Exception as e:
                log.warning(f"Skipping {symbol}: {e}")
        return results

    def save(self, df: pd.DataFrame, path: str):
        """Save DataFrame to CSV."""
        df.to_csv(path)
        log.info(f"Saved {len(df)} rows to {path}")

    def load(self, path: str) -> pd.DataFrame:
        """Load DataFrame from CSV."""
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        log.info(f"Loaded {len(df)} rows from {path}")
        return df
