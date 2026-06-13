"""
NEXUS v2 — Multi-Timeframe (MTF) Analyzer
==========================================
Analyzes BTC trend direction across 4 timeframes:
  Weekly → Daily → 4H → 1H

Professional traders check higher timeframes FIRST to understand
the big picture, then drop down for entry timing. This module does
that automatically and returns how many timeframes agree on direction.

Trend logic per timeframe:
  bullish  → EMA fast > EMA slow + price above long EMA + RSI > 55 + higher highs
  bearish  → EMA fast < EMA slow + price below long EMA + RSI < 45 + lower lows
  neutral  → mixed signals (score between -1 and +1)
"""

import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)


class MultiTimeframeAnalyzer:
    """
    Analyzes multiple timeframes resampled from 1H OHLCV data.

    Usage:
        analyzer = MultiTimeframeAnalyzer()
        features = analyzer.analyze(df_1h)
        if features["mtf_bull_count"] >= 3:
            print("Strong BUY alignment across timeframes")
    """

    def analyze(self, df_1h: pd.DataFrame) -> dict:
        """
        Analyze all timeframes from a 1H OHLCV dataframe.

        Args:
            df_1h: 1H OHLCV DataFrame with DatetimeIndex (UTC)

        Returns:
            dict with mtf_bull_count, mtf_bear_count, mtf_alignment,
            mtf_daily_rsi, and per-timeframe trend strings
        """
        timeframes = {
            "weekly": self._resample(df_1h, "W"),
            "daily":  self._resample(df_1h, "D"),
            "4h":     self._resample(df_1h, "4h"),
            "1h":     df_1h.copy(),
        }

        trends     = {}
        rsi_values = {}

        for tf_name, tf_df in timeframes.items():
            if len(tf_df) < 10:
                trends[tf_name]     = "neutral"
                rsi_values[tf_name] = 50.0
                continue

            trend, rsi = self._analyze_timeframe(tf_df, tf_name)
            trends[tf_name]     = trend
            rsi_values[tf_name] = rsi

        bull_count = sum(1 for t in trends.values() if t == "bullish")
        bear_count = sum(1 for t in trends.values() if t == "bearish")
        alignment  = (bull_count - bear_count) / 4.0

        log.info(
            f"MTF | W:{trends['weekly']} D:{trends['daily']} "
            f"4H:{trends['4h']} 1H:{trends['1h']} | "
            f"Bull {bull_count}/4 Bear {bear_count}/4 | "
            f"Alignment {alignment:+.2f}"
        )

        return {
            "mtf_bull_count":   bull_count,
            "mtf_bear_count":   bear_count,
            "mtf_alignment":    round(alignment, 3),
            "mtf_daily_rsi":    round(rsi_values.get("daily", 50.0), 2),
            "mtf_4h_rsi":       round(rsi_values.get("4h",    50.0), 2),
            "mtf_1h_rsi":       round(rsi_values.get("1h",    50.0), 2),
            "mtf_weekly_trend": trends.get("weekly", "neutral"),
            "mtf_daily_trend":  trends.get("daily",  "neutral"),
            "mtf_4h_trend":     trends.get("4h",     "neutral"),
            "mtf_1h_trend":     trends.get("1h",     "neutral"),
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _resample(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        try:
            return df.resample(rule).agg({
                "open": "first", "high": "max",
                "low":  "min",   "close": "last", "volume": "sum",
            }).dropna()
        except Exception as e:
            log.warning(f"Resample {rule} failed: {e}")
            return pd.DataFrame()

    def _analyze_timeframe(self, df: pd.DataFrame, tf_name: str) -> tuple:
        """
        Score trend direction for one timeframe.
        Returns (trend: str, rsi: float)
        """
        close = df["close"].astype(float)
        n     = len(close)

        # Adapt indicator periods to available data length
        fast_p = min(9,  max(3,  n // 4))
        slow_p = min(21, max(5,  n // 2))
        long_p = min(50, max(10, n - 1))
        rsi_p  = min(14, max(5,  n // 3))

        ema_fast = close.ewm(span=fast_p, adjust=False).mean()
        ema_slow = close.ewm(span=slow_p, adjust=False).mean()
        ema_long = close.ewm(span=long_p, adjust=False).mean()

        rsi_series  = self._calc_rsi(close, rsi_p)
        current_rsi = float(rsi_series.iloc[-1]) if len(rsi_series) > 0 else 50.0

        price    = float(close.iloc[-1])
        fast_val = float(ema_fast.iloc[-1])
        slow_val = float(ema_slow.iloc[-1])
        long_val = float(ema_long.iloc[-1])

        score = 0

        # Factor 1: EMA crossover (weight 2 — most important)
        if fast_val > slow_val:
            score += 2
        elif fast_val < slow_val:
            score -= 2

        # Factor 2: Price vs long-term EMA
        if price > long_val:
            score += 1
        elif price < long_val:
            score -= 1

        # Factor 3: RSI momentum
        if current_rsi > 55:
            score += 1
        elif current_rsi < 45:
            score -= 1

        # Factor 4: Higher highs/lows vs lower highs/lows
        if n >= 10:
            try:
                r_high = float(df["high"].iloc[-3:].max())
                o_high = float(df["high"].iloc[-10:-3].max())
                r_low  = float(df["low"].iloc[-3:].min())
                o_low  = float(df["low"].iloc[-10:-3].min())
                if r_high > o_high and r_low > o_low:
                    score += 1
                elif r_high < o_high and r_low < o_low:
                    score -= 1
            except Exception:
                pass

        # Max score = 5, min = -5
        if score >= 2:
            trend = "bullish"
        elif score <= -2:
            trend = "bearish"
        else:
            trend = "neutral"

        log.debug(f"  [{tf_name}] score={score:+d} trend={trend} RSI={current_rsi:.1f}")
        return trend, current_rsi

    def _calc_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        delta    = close.diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
        rs       = avg_gain / (avg_loss + 1e-9)
        return 100 - (100 / (1 + rs))
