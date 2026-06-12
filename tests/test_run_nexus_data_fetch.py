import unittest
from unittest import mock

import pandas as pd

import run_nexus


class FetchLiveHourlyDataTests(unittest.TestCase):
    def test_uses_binance_when_available(self):
        fake_exchange = mock.Mock()
        fake_exchange.fetch_ohlcv.return_value = [
            [1710000000000, 1, 2, 0.5, 1.5, 100],
            [1710003600000, 1.5, 2.5, 1.0, 2.0, 110],
        ]

        with mock.patch("ccxt.binance", return_value=fake_exchange):
            df = run_nexus.fetch_live_hourly_data("BTC/USDT", limit=2)

        self.assertEqual(len(df), 2)
        self.assertIn("symbol", df.columns)
        self.assertIn("timeframe", df.columns)
        self.assertEqual(df["symbol"].iloc[-1], "BTC/USDT")
        self.assertEqual(df["timeframe"].iloc[-1], "1h")

    def test_falls_back_to_coinbase_when_binance_fails(self):
        fake_coinbase = mock.Mock()
        fake_coinbase.fetch_ohlcv.return_value = [
            [1710000000000, 10, 11, 9.5, 10.5, 1000],
            [1710003600000, 10.5, 11.5, 10.0, 11.0, 1200],
        ]

        with mock.patch("ccxt.binance", side_effect=Exception("451 restricted location")), mock.patch(
            "ccxt.coinbase", return_value=fake_coinbase
        ):
            df = run_nexus.fetch_live_hourly_data("BTC/USDT", limit=2)

        self.assertEqual(len(df), 2)
        self.assertAlmostEqual(float(df["close"].iloc[-1]), 11.0)
        self.assertEqual(df["symbol"].iloc[-1], "BTC/USDT")
        self.assertEqual(df["timeframe"].iloc[-1], "1h")

    def test_falls_back_to_yfinance_when_binance_fails(self):
        index = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
        yf_df = pd.DataFrame(
            {
                "Open": [1.0, 1.1, 1.2],
                "High": [1.1, 1.2, 1.3],
                "Low": [0.9, 1.0, 1.1],
                "Close": [1.05, 1.15, 1.25],
                "Volume": [10, 11, 12],
            },
            index=index,
        )

        with mock.patch("ccxt.binance", side_effect=Exception("451 restricted location")), mock.patch(
            "ccxt.coinbase", side_effect=Exception("coinbase unavailable")
        ), mock.patch("ccxt.kraken", side_effect=Exception("kraken unavailable")), mock.patch(
            "yfinance.download", return_value=yf_df
        ):
            df = run_nexus.fetch_live_hourly_data("BTC/USDT", limit=2)

        self.assertEqual(len(df), 2)
        self.assertAlmostEqual(float(df["close"].iloc[-1]), 1.25)
        self.assertEqual(df["symbol"].iloc[-1], "BTC/USDT")
        self.assertEqual(df["timeframe"].iloc[-1], "1h")


if __name__ == "__main__":
    unittest.main()
