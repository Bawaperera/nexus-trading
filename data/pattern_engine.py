"""
NEXUS v2 — Pattern Detection Engine
=====================================
Detects 10 chart patterns from OHLCV data, tracks their status,
and identifies retest opportunities for low-risk entries.

Patterns detected:
  Reversal:     Double Bottom, Double Top, Head & Shoulders
  Continuation: Bull Flag, Bear Flag, Ascending Triangle,
                Descending Triangle, Symmetrical Triangle
  Special:      Falling Wedge, Rising Wedge

Each pattern goes through these states:
  FORMING         → pattern structure being built (no trade yet)
  BREAKOUT        → price broke the key level (watch for retest)
  AWAITING_RETEST → price moved away, waiting for it to come back
  RETEST_CONFIRMED → price returned to the broken level → ENTER HERE

Why retest entry?
  Breaking resistance then waiting for price to come back to that level
  (now acting as support) gives a much lower-risk entry than chasing
  the breakout. Stop can be placed just below the retest zone.
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone

log = logging.getLogger(__name__)


@dataclass
class PatternSignal:
    """One detected chart pattern with all trade parameters."""

    name: str           # e.g. "double_bottom", "bull_flag"
    direction: str      # "bullish" or "bearish"
    timeframe: str      # "daily", "4h", "1h"
    status: str         # FORMING | BREAKOUT | AWAITING_RETEST | RETEST_CONFIRMED

    breakout_level: float    # price that confirms the pattern
    stop_loss: float         # hard stop placement
    target: float            # measured move target
    entry: float             # ideal entry price

    confidence: float        # 0.0–1.0 based on pattern quality
    pattern_height: float    # distance from key level to opposite extreme
    timeframe_bars: int      # how many bars the pattern spans

    detected_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    key_levels: dict = field(default_factory=dict)

    @property
    def signal_type(self) -> str:
        return "BUY" if self.direction == "bullish" else "SELL"

    @property
    def risk(self) -> float:
        return abs(self.entry - self.stop_loss)

    @property
    def reward(self) -> float:
        return abs(self.target - self.entry)

    @property
    def risk_reward(self) -> float:
        return round(self.reward / self.risk, 2) if self.risk > 0 else 0.0

    @property
    def is_ready(self) -> bool:
        """Pattern is ready to trade — retest confirmed."""
        return self.status == "RETEST_CONFIRMED"

    def is_tradeable(self, min_rr: float = 2.0, min_conf: float = 0.50) -> bool:
        return self.is_ready and self.risk_reward >= min_rr and self.confidence >= min_conf

    def summary(self) -> str:
        return (
            f"{self.name.replace('_',' ').title()} [{self.timeframe}] "
            f"| {self.direction.upper()} | {self.status} "
            f"| Conf: {self.confidence:.0%} | R:R {self.risk_reward}"
        )


class PatternEngine:
    """
    Scans BTC price data across 3 timeframes and detects 10 patterns.

    Usage:
        engine = PatternEngine()
        patterns = engine.scan_all(df_1h)
        tradeable = [p for p in patterns if p.is_tradeable()]
    """

    PIVOT_WINDOW   = 5      # bars each side for a valid swing high/low
    MIN_BARS       = 10     # minimum bars to form a valid pattern
    SIMILAR_PCT    = 0.03   # within 3% = "at the same level"
    RETEST_ZONE    = 0.008  # ±0.8% of breakout level = retest zone

    def scan_all(self, df_1h: pd.DataFrame) -> list:
        """
        Scan daily, 4H, and 1H charts for all patterns.
        Returns list of PatternSignal sorted by confidence (highest first).
        """
        all_patterns = []

        timeframes = {
            "daily": self._resample(df_1h, "D"),
            "4h":    self._resample(df_1h, "4h"),
            "1h":    df_1h.copy(),
        }

        for tf_name, tf_df in timeframes.items():
            if len(tf_df) < 30:
                continue
            tf_patterns = self._scan_timeframe(tf_df, tf_name)
            all_patterns.extend(tf_patterns)
            if tf_patterns:
                log.info(
                    f"  [{tf_name}] patterns: "
                    f"{', '.join(p.name for p in tf_patterns)}"
                )

        all_patterns.sort(key=lambda p: p.confidence, reverse=True)
        log.info(f"Total patterns found: {len(all_patterns)}")
        return all_patterns

    # ── Private: orchestration ────────────────────────────────────────────────

    def _scan_timeframe(self, df: pd.DataFrame, tf: str) -> list:
        detectors = [
            self._detect_double_bottom,
            self._detect_double_top,
            self._detect_head_shoulders,
            self._detect_ascending_triangle,
            self._detect_descending_triangle,
            self._detect_symmetrical_triangle,
            self._detect_bull_flag,
            self._detect_bear_flag,
            self._detect_falling_wedge,
            self._detect_rising_wedge,
        ]
        patterns = []
        for fn in detectors:
            try:
                p = fn(df, tf)
                if p is not None:
                    patterns.append(p)
            except Exception as e:
                log.debug(f"  {fn.__name__} on {tf}: {e}")
        return patterns

    # ── Private: helpers ──────────────────────────────────────────────────────

    def _resample(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        return df.resample(rule).agg(
            {"open": "first", "high": "max",
             "low": "min", "close": "last", "volume": "sum"}
        ).dropna()

    def _pivots(self, df: pd.DataFrame, w: int = None):
        """Return (index, price) lists of pivot highs and lows."""
        w = w or self.PIVOT_WINDOW
        h, l = df["high"].values, df["low"].values
        highs, lows = [], []
        for i in range(w, len(h) - w):
            if h[i] == max(h[i - w : i + w + 1]):
                highs.append((i, float(h[i])))
            if l[i] == min(l[i - w : i + w + 1]):
                lows.append((i, float(l[i])))
        return highs, lows

    def _similar(self, a: float, b: float) -> bool:
        return abs(a - b) / max(abs(a), abs(b), 1e-9) < self.SIMILAR_PCT

    def _atr(self, df: pd.DataFrame, periods: int = 14) -> float:
        n = min(periods, len(df))
        return float(df["high"].iloc[-n:].max() - df["low"].iloc[-n:].min()) / n

    def _retest_status(
        self, df: pd.DataFrame, level: float, direction: str
    ) -> str:
        """
        Given a breakout level and direction, return the pattern status.

        Logic:
          - If price has not yet broken the level → FORMING
          - If price broke and moved away → AWAITING_RETEST
          - If price came back within the retest zone → RETEST_CONFIRMED
          - If price just broke → BREAKOUT
        """
        close  = float(df["close"].iloc[-1])
        tol    = level * self.RETEST_ZONE
        recent = df.iloc[-15:]

        if direction == "bullish":
            if close < level - tol:
                return "FORMING"
            if close > level + tol * 2:
                recent_lows = recent["low"].values
                if any(abs(float(v) - level) <= tol * 2 for v in recent_lows):
                    return "RETEST_CONFIRMED"
                return "AWAITING_RETEST"
            return "BREAKOUT"

        else:  # bearish
            if close > level + tol:
                return "FORMING"
            if close < level - tol * 2:
                recent_highs = recent["high"].values
                if any(abs(float(v) - level) <= tol * 2 for v in recent_highs):
                    return "RETEST_CONFIRMED"
                return "AWAITING_RETEST"
            return "BREAKOUT"

    # ── Pattern detectors ──────────────────────────────────────────────────────

    def _detect_double_bottom(self, df: pd.DataFrame, tf: str) -> Optional[PatternSignal]:
        """
        Double Bottom — bullish reversal (W shape)
        Two troughs at roughly the same price.
        Neckline = peak between the two troughs.
        Trade: break above neckline → retest → BUY.
        Target: neckline + (neckline − trough)
        """
        _, lows = self._pivots(df)
        if len(lows) < 2:
            return None

        t1_i, t1_p = lows[-2]
        t2_i, t2_p = lows[-1]

        if t2_i - t1_i < self.MIN_BARS:
            return None
        if not self._similar(t1_p, t2_p):
            return None

        neckline = float(df["high"].iloc[t1_i : t2_i + 1].max())
        height   = neckline - min(t1_p, t2_p)

        if height / neckline < 0.02:
            return None

        atr_val  = self._atr(df)
        status   = self._retest_status(df, neckline, "bullish")
        conf     = 0.75 if abs(t1_p - t2_p) / t1_p < 0.01 else 0.60

        return PatternSignal(
            name="double_bottom", direction="bullish", timeframe=tf,
            status=status, breakout_level=neckline,
            stop_loss=min(t1_p, t2_p) - atr_val * 0.5,
            target=neckline + height,
            entry=neckline,
            confidence=conf, pattern_height=height,
            timeframe_bars=t2_i - t1_i,
            key_levels={"trough1": t1_p, "trough2": t2_p, "neckline": neckline},
        )

    def _detect_double_top(self, df: pd.DataFrame, tf: str) -> Optional[PatternSignal]:
        """
        Double Top — bearish reversal (M shape)
        Two peaks at roughly the same price.
        Trade: break below neckline → retest → SELL.
        Target: neckline − (peak − neckline)
        """
        highs, _ = self._pivots(df)
        if len(highs) < 2:
            return None

        h1_i, h1_p = highs[-2]
        h2_i, h2_p = highs[-1]

        if h2_i - h1_i < self.MIN_BARS:
            return None
        if not self._similar(h1_p, h2_p):
            return None

        neckline = float(df["low"].iloc[h1_i : h2_i + 1].min())
        height   = max(h1_p, h2_p) - neckline

        if height / neckline < 0.02:
            return None

        atr_val = self._atr(df)
        status  = self._retest_status(df, neckline, "bearish")
        conf    = 0.75 if abs(h1_p - h2_p) / h1_p < 0.01 else 0.60

        return PatternSignal(
            name="double_top", direction="bearish", timeframe=tf,
            status=status, breakout_level=neckline,
            stop_loss=max(h1_p, h2_p) + atr_val * 0.5,
            target=neckline - height,
            entry=neckline,
            confidence=conf, pattern_height=height,
            timeframe_bars=h2_i - h1_i,
            key_levels={"peak1": h1_p, "peak2": h2_p, "neckline": neckline},
        )

    def _detect_head_shoulders(self, df: pd.DataFrame, tf: str) -> Optional[PatternSignal]:
        """
        Head & Shoulders — bearish reversal
        Left shoulder + head (tallest peak) + right shoulder (≈ left).
        Neckline = line connecting the two troughs between shoulders and head.
        Trade: break below neckline → retest → SELL.
        Target: neckline − (head − neckline)
        """
        highs, _ = self._pivots(df)
        if len(highs) < 3:
            return None

        ls_i, ls_p = highs[-3]
        h_i,  h_p  = highs[-2]
        rs_i, rs_p = highs[-1]

        if not (h_p > ls_p and h_p > rs_p):
            return None
        if not self._similar(ls_p, rs_p):
            return None
        if (h_i - ls_i) < 5 or (rs_i - h_i) < 5:
            return None

        t1      = float(df["low"].iloc[ls_i : h_i + 1].min())
        t2      = float(df["low"].iloc[h_i  : rs_i + 1].min())
        neckline = (t1 + t2) / 2
        height   = h_p - neckline

        if height / neckline < 0.03:
            return None

        atr_val = self._atr(df)
        status  = self._retest_status(df, neckline, "bearish")

        return PatternSignal(
            name="head_and_shoulders", direction="bearish", timeframe=tf,
            status=status, breakout_level=neckline,
            stop_loss=rs_p + atr_val * 0.5,
            target=neckline - height,
            entry=neckline,
            confidence=0.75, pattern_height=height,
            timeframe_bars=rs_i - ls_i,
            key_levels={
                "left_shoulder": ls_p, "head": h_p,
                "right_shoulder": rs_p, "neckline": neckline,
            },
        )

    def _detect_ascending_triangle(self, df: pd.DataFrame, tf: str) -> Optional[PatternSignal]:
        """
        Ascending Triangle — bullish continuation
        Flat resistance top + rising lows (buyers getting more aggressive).
        Trade: break above flat resistance → retest → BUY.
        Target: resistance + triangle height.
        """
        n = min(40, len(df))
        if n < 20:
            return None

        recent = df.iloc[-n:]
        h_vals = recent["high"].values
        l_vals = recent["low"].values

        res = float(np.percentile(h_vals, 88))
        touches = sum(1 for h in h_vals if abs(h - res) / res < 0.005)
        if touches < 2:
            return None

        ls = float(np.polyfit(np.arange(n), l_vals, 1)[0]) / float(recent["close"].mean())
        if ls <= 0.0005:
            return None

        height = res - float(recent["low"].min())
        if height / res < 0.02:
            return None

        atr_val = self._atr(df)
        status  = self._retest_status(df, res, "bullish")

        return PatternSignal(
            name="ascending_triangle", direction="bullish", timeframe=tf,
            status=status, breakout_level=res,
            stop_loss=float(recent["low"].min()) - atr_val * 0.3,
            target=res + height,
            entry=res,
            confidence=0.65, pattern_height=height,
            timeframe_bars=n,
            key_levels={"resistance": res, "triangle_low": float(recent["low"].min())},
        )

    def _detect_descending_triangle(self, df: pd.DataFrame, tf: str) -> Optional[PatternSignal]:
        """
        Descending Triangle — bearish continuation
        Flat support bottom + declining highs (sellers getting more aggressive).
        Trade: break below flat support → retest → SELL.
        Target: support − triangle height.
        """
        n = min(40, len(df))
        if n < 20:
            return None

        recent = df.iloc[-n:]
        h_vals = recent["high"].values
        l_vals = recent["low"].values

        sup = float(np.percentile(l_vals, 12))
        touches = sum(1 for l in l_vals if abs(l - sup) / sup < 0.005)
        if touches < 2:
            return None

        hs = float(np.polyfit(np.arange(n), h_vals, 1)[0]) / float(recent["close"].mean())
        if hs >= -0.0005:
            return None

        height  = float(recent["high"].max()) - sup
        atr_val = self._atr(df)
        status  = self._retest_status(df, sup, "bearish")

        return PatternSignal(
            name="descending_triangle", direction="bearish", timeframe=tf,
            status=status, breakout_level=sup,
            stop_loss=float(recent["high"].max()) + atr_val * 0.3,
            target=sup - height,
            entry=sup,
            confidence=0.65, pattern_height=height,
            timeframe_bars=n,
            key_levels={"support": sup, "triangle_high": float(recent["high"].max())},
        )

    def _detect_symmetrical_triangle(self, df: pd.DataFrame, tf: str) -> Optional[PatternSignal]:
        """
        Symmetrical Triangle — neutral
        Both highs declining and lows rising (converging).
        Direction decided by which side breaks first.
        """
        n = min(40, len(df))
        if n < 20:
            return None

        recent = df.iloc[-n:]
        x = np.arange(n)
        avg = float(recent["close"].mean())

        hs = float(np.polyfit(x, recent["high"].values, 1)[0]) / avg
        ls = float(np.polyfit(x, recent["low"].values,  1)[0]) / avg

        if not (hs < -0.001 and ls > 0.001):
            return None

        apex_h = float(np.poly1d(np.polyfit(x, recent["high"].values, 1))(n))
        apex_l = float(np.poly1d(np.polyfit(x, recent["low"].values,  1))(n))
        height = float(recent["high"].max() - recent["low"].min())

        cur  = float(df["close"].iloc[-1])
        up   = cur > avg
        bkl  = apex_h if up else apex_l
        dir_ = "bullish" if up else "bearish"

        atr_val = self._atr(df)
        status  = self._retest_status(df, bkl, dir_)

        return PatternSignal(
            name="symmetrical_triangle", direction=dir_, timeframe=tf,
            status=status, breakout_level=bkl,
            stop_loss=apex_l - atr_val if up else apex_h + atr_val,
            target=bkl + height if up else bkl - height,
            entry=bkl,
            confidence=0.55, pattern_height=height,
            timeframe_bars=n,
            key_levels={"apex_high": apex_h, "apex_low": apex_l},
        )

    def _detect_bull_flag(self, df: pd.DataFrame, tf: str) -> Optional[PatternSignal]:
        """
        Bull Flag — bullish continuation
        Sharp upward move (flagpole) + downward-sloping consolidation channel (flag).
        Trade: break above upper channel → retest → BUY.
        Target: breakout + height of flagpole.
        """
        if len(df) < 25:
            return None

        recent = df.iloc[-25:]
        best_rise, pole_start, pole_end = 0.0, 0, 0

        for i in range(3, len(recent) - 5):
            for j in range(i + 3, min(i + 10, len(recent) - 3)):
                rise = float(recent["close"].iloc[j] / recent["close"].iloc[i] - 1)
                if rise > 0.04 and rise > best_rise:
                    best_rise, pole_start, pole_end = rise, i, j

        if best_rise < 0.04:
            return None

        flag = recent.iloc[pole_end:]
        if len(flag) < 3:
            return None

        x = np.arange(len(flag))
        avg = float(flag["close"].mean())
        fh = float(np.polyfit(x, flag["high"].values, 1)[0]) / avg
        fl = float(np.polyfit(x, flag["low"].values,  1)[0]) / avg

        if not (fh < -0.0003 and fl < -0.0003):
            return None

        upper_ch  = float(flag["high"].max())
        pole_ht   = float(recent["close"].iloc[pole_end] - recent["close"].iloc[pole_start])
        atr_val   = self._atr(df)
        status    = self._retest_status(df, upper_ch, "bullish")

        return PatternSignal(
            name="bull_flag", direction="bullish", timeframe=tf,
            status=status, breakout_level=upper_ch,
            stop_loss=float(flag["low"].min()) - atr_val * 0.3,
            target=upper_ch + pole_ht,
            entry=upper_ch,
            confidence=0.70, pattern_height=pole_ht,
            timeframe_bars=len(recent),
            key_levels={
                "pole_start_price": float(recent["close"].iloc[pole_start]),
                "pole_end_price":   float(recent["close"].iloc[pole_end]),
                "flag_upper":       upper_ch,
                "flag_lower":       float(flag["low"].min()),
            },
        )

    def _detect_bear_flag(self, df: pd.DataFrame, tf: str) -> Optional[PatternSignal]:
        """
        Bear Flag — bearish continuation
        Sharp downward move (pole) + upward-sloping consolidation (flag).
        Trade: break below lower channel → retest → SELL.
        Target: breakdown − height of pole.
        """
        if len(df) < 25:
            return None

        recent = df.iloc[-25:]
        best_drop, pole_start, pole_end = 0.0, 0, 0

        for i in range(3, len(recent) - 5):
            for j in range(i + 3, min(i + 10, len(recent) - 3)):
                drop = float(1 - recent["close"].iloc[j] / recent["close"].iloc[i])
                if drop > 0.04 and drop > best_drop:
                    best_drop, pole_start, pole_end = drop, i, j

        if best_drop < 0.04:
            return None

        flag = recent.iloc[pole_end:]
        if len(flag) < 3:
            return None

        x = np.arange(len(flag))
        avg = float(flag["close"].mean())
        fh = float(np.polyfit(x, flag["high"].values, 1)[0]) / avg
        fl = float(np.polyfit(x, flag["low"].values,  1)[0]) / avg

        if not (fh > 0.0003 and fl > 0.0003):
            return None

        lower_ch = float(flag["low"].min())
        pole_ht  = float(recent["close"].iloc[pole_start] - recent["close"].iloc[pole_end])
        atr_val  = self._atr(df)
        status   = self._retest_status(df, lower_ch, "bearish")

        return PatternSignal(
            name="bear_flag", direction="bearish", timeframe=tf,
            status=status, breakout_level=lower_ch,
            stop_loss=float(flag["high"].max()) + atr_val * 0.3,
            target=lower_ch - pole_ht,
            entry=lower_ch,
            confidence=0.70, pattern_height=pole_ht,
            timeframe_bars=len(recent),
            key_levels={
                "flag_upper": float(flag["high"].max()),
                "flag_lower": lower_ch,
            },
        )

    def _detect_falling_wedge(self, df: pd.DataFrame, tf: str) -> Optional[PatternSignal]:
        """
        Falling Wedge — bullish (reversal or continuation)
        Both highs and lows declining, but converging (highs fall faster).
        Trade: break above upper wedge line → retest → BUY.
        """
        n = min(40, len(df))
        if n < 20:
            return None

        recent = df.iloc[-n:]
        x = np.arange(n)
        avg = float(recent["close"].mean())

        hs = float(np.polyfit(x, recent["high"].values, 1)[0]) / avg
        ls = float(np.polyfit(x, recent["low"].values,  1)[0]) / avg

        if not (hs < -0.001 and ls < -0.001 and hs < ls):
            return None

        upper_w = float(np.poly1d(np.polyfit(x, recent["high"].values, 1))(n - 1))
        lower_w = float(np.poly1d(np.polyfit(x, recent["low"].values,  1))(n - 1))
        height  = float(recent["high"].max() - recent["low"].min())

        atr_val = height / n
        status  = self._retest_status(df, upper_w, "bullish")

        return PatternSignal(
            name="falling_wedge", direction="bullish", timeframe=tf,
            status=status, breakout_level=upper_w,
            stop_loss=lower_w - atr_val * 0.5,
            target=upper_w + height * 0.618,
            entry=upper_w,
            confidence=0.62, pattern_height=height,
            timeframe_bars=n,
            key_levels={"upper_wedge": upper_w, "lower_wedge": lower_w},
        )

    def _detect_rising_wedge(self, df: pd.DataFrame, tf: str) -> Optional[PatternSignal]:
        """
        Rising Wedge — bearish (often reversal)
        Both highs and lows rising, but converging (lows rise faster).
        Trade: break below lower wedge line → retest → SELL.
        """
        n = min(40, len(df))
        if n < 20:
            return None

        recent = df.iloc[-n:]
        x = np.arange(n)
        avg = float(recent["close"].mean())

        hs = float(np.polyfit(x, recent["high"].values, 1)[0]) / avg
        ls = float(np.polyfit(x, recent["low"].values,  1)[0]) / avg

        if not (hs > 0.001 and ls > 0.001 and ls > hs):
            return None

        upper_w = float(np.poly1d(np.polyfit(x, recent["high"].values, 1))(n - 1))
        lower_w = float(np.poly1d(np.polyfit(x, recent["low"].values,  1))(n - 1))
        height  = float(recent["high"].max() - recent["low"].min())

        atr_val = height / n
        status  = self._retest_status(df, lower_w, "bearish")

        return PatternSignal(
            name="rising_wedge", direction="bearish", timeframe=tf,
            status=status, breakout_level=lower_w,
            stop_loss=upper_w + atr_val * 0.5,
            target=lower_w - height * 0.618,
            entry=lower_w,
            confidence=0.62, pattern_height=height,
            timeframe_bars=n,
            key_levels={"upper_wedge": upper_w, "lower_wedge": lower_w},
        )
