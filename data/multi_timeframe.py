"""
NEXUS v2 — Multi-Timeframe (MTF) Analyzer
==========================================
Reads 4 timeframes to answer 4 different questions:

  Weekly  → "What is the long-term trend?" (background context)
  Daily   → "Is the overall trend bullish or bearish?"
  4H      → "Is a pattern setup forming?"
  15M     → "Is NOW the right moment to enter?" ← entry trigger

The 15M entry trigger looks for:
  1. Bullish/bearish engulfing candle (big green candle swallowing a red one)
  2. Volume spike (2× more trading than normal = strong conviction)
  3. Breaking the previous 15M structure (price making a new higher high)
  4. RSI in a favorable zone (not overbought if buying, not oversold if shorting)

Why 15M for entry?
  The Daily chart might say "bullish" for weeks.
  The 4H might show a setup for days.
  But you want to enter at the BEST moment — when buyers
  are actually stepping in RIGHT NOW. The 15M shows this.
"""

import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)


class MultiTimeframeAnalyzer:
    """
    Analyzes Weekly, Daily, 4H from 1H data, and 15M separately.

    Usage:
        analyzer = MultiTimeframeAnalyzer()
        features = analyzer.analyze(df_1h, df_15m)
        if features["mtf_bull_count"] >= 3 and features["mtf_15m_trigger"]:
            print("Strong setup with 15M entry confirmation")
    """

    def analyze(self, df_1h: pd.DataFrame, df_15m: pd.DataFrame = None) -> dict:
        """
        Analyze all timeframes.

        Args:
            df_1h:  1H OHLCV DataFrame (used to build weekly, daily, 4H)
            df_15m: 15M OHLCV DataFrame (fetched separately, for entry timing)

        Returns dict with all MTF features.
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

        # 15M entry trigger analysis
        entry_15m = self._analyze_15m_entry(df_15m) if df_15m is not None else self._no_15m()

        log.info(
            f"MTF | W:{trends['weekly']} D:{trends['daily']} "
            f"4H:{trends['4h']} 1H:{trends['1h']} | "
            f"Bull {bull_count}/4 Bear {bear_count}/4 | "
            f"15M trigger: {entry_15m['triggered']} (score {entry_15m['score']}/6)"
        )

        return {
            # Standard MTF
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
            # 15M entry trigger
            "mtf_15m_trigger":  entry_15m["triggered"],
            "mtf_15m_score":    entry_15m["score"],
            "mtf_15m_reasons":  entry_15m["reasons"],
            "mtf_15m_rsi":      entry_15m["rsi"],
        }

    # ── 15M entry trigger ──────────────────────────────────────────────────────

    def _analyze_15m_entry(self, df_15m: pd.DataFrame) -> dict:
        """
        Check if the 15M chart shows a good entry trigger.

        A trigger fires when 3+ of these are true:
          1. Bullish/bearish engulfing candle (strong reversal signal)
          2. Volume spike 1.5× average (real conviction, not just noise)
          3. RSI in the right zone (room to run)
          4. Breaking previous 15M structure high/low
          5. Recent momentum (last 3 candles trending in direction)

        Score 0-6. Trigger fires at score ≥ 3.
        """
        if df_15m is None or len(df_15m) < 20:
            return self._no_15m()

        close  = df_15m["close"].astype(float)
        open_  = df_15m["open"].astype(float)
        high   = df_15m["high"].astype(float)
        low    = df_15m["low"].astype(float)
        volume = df_15m["volume"].astype(float)

        last = df_15m.iloc[-1]
        prev = df_15m.iloc[-2]

        last_c, last_o = float(last["close"]), float(last["open"])
        prev_c, prev_o = float(prev["close"]), float(prev["open"])
        last_h, last_v = float(last["high"]), float(last["volume"])
        last_l         = float(last["low"])

        # RSI
        rsi_series  = self._calc_rsi(close, period=14)
        current_rsi = float(rsi_series.iloc[-1]) if len(rsi_series) > 0 else 50.0

        # Average volume
        avg_vol = float(volume.iloc[-20:].mean())
        vol_ratio = last_v / (avg_vol + 1e-9)

        # Previous structure
        prev_high_10 = float(high.iloc[-11:-1].max()) if len(high) >= 11 else last_h
        prev_low_10  = float(low.iloc[-11:-1].min())  if len(low)  >= 11 else last_l

        # Bullish signals
        b_engulf = (last_c > last_o and          # green candle
                    prev_c < prev_o and           # previous red
                    last_o <= prev_c and          # open ≤ prev close
                    last_c >= prev_o)             # close ≥ prev open

        b_vol    = vol_ratio >= 1.5
        b_rsi    = 30 <= current_rsi <= 60
        b_break  = last_c > prev_high_10
        b_mom    = float(close.iloc[-1]) > float(close.iloc[-4])  # rising last 3

        # Bearish signals
        s_engulf = (last_c < last_o and
                    prev_c > prev_o and
                    last_o >= prev_c and
                    last_c <= prev_o)

        s_vol    = vol_ratio >= 1.5
        s_rsi    = 40 <= current_rsi <= 70
        s_break  = last_c < prev_low_10
        s_mom    = float(close.iloc[-1]) < float(close.iloc[-4])

        # Score for each direction
        b_score = sum([b_engulf*2, b_vol, b_rsi, b_break*2, b_mom])
        s_score = sum([s_engulf*2, s_vol, s_rsi, s_break*2, s_mom])

        reasons_b = []
        if b_engulf: reasons_b.append("Bullish engulfing on 15M")
        if b_vol:    reasons_b.append(f"Volume {vol_ratio:.1f}× on 15M")
        if b_rsi:    reasons_b.append(f"RSI {current_rsi:.0f} — room to run on 15M")
        if b_break:  reasons_b.append("Breaking 15M structure high")
        if b_mom:    reasons_b.append("Rising momentum on 15M")

        reasons_s = []
        if s_engulf: reasons_s.append("Bearish engulfing on 15M")
        if s_vol:    reasons_s.append(f"Volume {vol_ratio:.1f}× on 15M")
        if s_rsi:    reasons_s.append(f"RSI {current_rsi:.0f} — room to fall on 15M")
        if s_break:  reasons_s.append("Breaking 15M structure low")
        if s_mom:    reasons_s.append("Falling momentum on 15M")

        # Return best direction
        if b_score >= s_score:
            return {
                "triggered": b_score >= 3,
                "direction": "bullish",
                "score":     b_score,
                "reasons":   reasons_b,
                "rsi":       round(current_rsi, 1),
                "vol_ratio": round(vol_ratio, 2),
            }
        else:
            return {
                "triggered": s_score >= 3,
                "direction": "bearish",
                "score":     s_score,
                "reasons":   reasons_s,
                "rsi":       round(current_rsi, 1),
                "vol_ratio": round(vol_ratio, 2),
            }

    def _no_15m(self) -> dict:
        return {
            "triggered": False, "direction": "neutral",
            "score": 0, "reasons": ["No 15M data available"],
            "rsi": 50.0, "vol_ratio": 1.0,
        }

    # ── Timeframe analysis ─────────────────────────────────────────────────────

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
        """Score trend direction. Returns (trend: str, rsi: float)."""
        close = df["close"].astype(float)
        n     = len(close)

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
        if fast_val > slow_val: score += 2
        elif fast_val < slow_val: score -= 2
        if price > long_val: score += 1
        elif price < long_val: score -= 1
        if current_rsi > 55: score += 1
        elif current_rsi < 45: score -= 1

        if n >= 10:
            try:
                r_high = float(df["high"].iloc[-3:].max())
                o_high = float(df["high"].iloc[-10:-3].max())
                r_low  = float(df["low"].iloc[-3:].min())
                o_low  = float(df["low"].iloc[-10:-3].min())
                if r_high > o_high and r_low > o_low: score += 1
                elif r_high < o_high and r_low < o_low: score -= 1
            except Exception:
                pass

        if score >= 2:   trend = "bullish"
        elif score <= -2: trend = "bearish"
        else:            trend = "neutral"

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
