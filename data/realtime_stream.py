"""
NEXUS — Real-time Price Stream
Connects to Binance WebSocket and streams live OHLCV candles.
On every candle CLOSE, it triggers feature recomputation + model inference.

Binance streams used:
  - btcusdt@kline_1h  → 1-hour candles (main trading signal)
  - btcusdt@kline_1m  → 1-minute candles (entry timing)
  - btcusdt@trade     → raw tick data (for volume analysis)

No API key needed — these are all public streams.
"""

import asyncio
import json
import logging
from collections import deque
from datetime import datetime, timezone

import websockets
import pandas as pd

log = logging.getLogger(__name__)


class RealtimeStream:
    """
    Connects to Binance WebSocket and maintains a live rolling
    OHLCV buffer that the Feature Engineer reads from.

    Usage:
        stream = RealtimeStream(symbol="btcusdt", timeframe="1h", buffer_size=300)
        asyncio.run(stream.start(on_candle_close=my_callback))
    """

    BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws"

    def __init__(self, symbol: str = "btcusdt", timeframe: str = "1h", buffer_size: int = 300):
        """
        Args:
            symbol: Binance symbol in lowercase (e.g. "btcusdt", "ethusdt")
            timeframe: Candle interval ("1m","5m","15m","30m","1h","4h","1d")
            buffer_size: How many candles to keep in memory (300 = 12.5 days of 1h candles)
        """
        self.symbol    = symbol.lower()
        self.timeframe = timeframe
        self.stream_id = f"{self.symbol}@kline_{timeframe}"
        self.uri       = f"{self.BINANCE_WS_BASE}/{self.stream_id}"

        # Rolling buffer — oldest candles auto-drop
        self.buffer: deque = deque(maxlen=buffer_size)

        # Current live (unfinished) candle
        self.live_candle: dict = {}

        self._running = False
        log.info(f"RealtimeStream ready: {self.stream_id}")

    async def start(self, on_candle_close=None):
        """
        Connect and stream. Calls on_candle_close(df) every time a
        candle CLOSES (not on every tick — we don't overtrade).

        Args:
            on_candle_close: async callable receiving a pd.DataFrame
                             of the buffer after each closed candle.
        """
        self._running = True
        reconnect_delay = 1

        while self._running:
            try:
                log.info(f"Connecting to Binance WebSocket: {self.uri}")
                async with websockets.connect(
                    self.uri,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    reconnect_delay = 1  # reset on successful connect
                    log.info(f"Connected ✅ — streaming {self.stream_id}")

                    async for raw in ws:
                        if not self._running:
                            break
                        await self._handle_message(raw, on_candle_close)

            except websockets.ConnectionClosed as e:
                log.warning(f"WebSocket closed ({e}), reconnecting in {reconnect_delay}s...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)

            except Exception as e:
                log.error(f"Stream error: {e}, reconnecting in {reconnect_delay}s...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)

    def stop(self):
        self._running = False
        log.info("Stream stopped")

    async def _handle_message(self, raw: str, on_candle_close):
        """Parse a Binance kline WebSocket message."""
        try:
            msg = json.loads(raw)
            k   = msg["k"]

            # Build candle dict from Binance kline payload
            candle = {
                "timestamp": pd.Timestamp(k["t"], unit="ms", tz="UTC"),
                "open":      float(k["o"]),
                "high":      float(k["h"]),
                "low":       float(k["l"]),
                "close":     float(k["c"]),
                "volume":    float(k["v"]),
                "is_closed": k["x"],  # True = candle has fully closed
            }

            self.live_candle = candle

            if k["x"]:  # Candle CLOSED — this is when we act
                # Add to buffer
                self.buffer.append({k: v for k, v in candle.items() if k != "is_closed"})
                log.info(
                    f"Candle closed | {candle['timestamp']} | "
                    f"O:{candle['open']:.2f} H:{candle['high']:.2f} "
                    f"L:{candle['low']:.2f} C:{candle['close']:.2f} "
                    f"V:{candle['volume']:.1f}"
                )

                if on_candle_close and len(self.buffer) >= 50:
                    df = self.get_dataframe()
                    await on_candle_close(df)

        except (KeyError, json.JSONDecodeError) as e:
            log.warning(f"Message parse error: {e}")

    def get_dataframe(self) -> pd.DataFrame:
        """Return current buffer as a pandas DataFrame ready for FeatureEngineer."""
        if not self.buffer:
            return pd.DataFrame()

        df = pd.DataFrame(list(self.buffer))
        df = df.set_index("timestamp").sort_index()
        df["symbol"]    = self.symbol.upper().replace("USDT", "/USDT")
        df["timeframe"] = self.timeframe
        return df

    def get_live_price(self) -> float:
        """Return the current (unfinished) candle's close price."""
        return self.live_candle.get("close", 0.0)

    def get_live_candle(self) -> dict:
        """Return the current live (unfinished) candle."""
        return self.live_candle.copy()

    def buffer_size_actual(self) -> int:
        return len(self.buffer)


# ─── Multi-symbol stream (runs multiple symbols concurrently) ─────────────────

class MultiStream:
    """
    Runs multiple symbol streams concurrently.

    Usage:
        ms = MultiStream(["btcusdt", "ethusdt", "solusdt"], "1h")
        asyncio.run(ms.start(on_candle_close=callback))
    """

    def __init__(self, symbols: list, timeframe: str = "1h"):
        self.streams = {s: RealtimeStream(s, timeframe) for s in symbols}

    async def start(self, on_candle_close=None):
        """Run all streams concurrently."""
        tasks = [
            stream.start(on_candle_close=on_candle_close)
            for stream in self.streams.values()
        ]
        await asyncio.gather(*tasks)

    def get_live_prices(self) -> dict:
        """Return live price for each symbol."""
        return {
            symbol: stream.get_live_price()
            for symbol, stream in self.streams.items()
        }
