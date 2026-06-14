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

Stop-loss logic (KEY to achieving R:R ≥ 2.0):
  BREAKOUT / FORMING      → stop at the FULL PATTERN low/high + ATR buffer
                            (wide stop = R:R ~1.0, usually blocked by MIN_RR)
  RETEST_CONFIRMED        → stop 1.5×ATR below the BREAKOUT LEVEL (now support)
                            (tight stop → R:R 4–8×, auto-trades qualify)

This is why the system waits for RETEST_CONFIRMED before trading:
the retest entry dramatically tightens the stop (price has returned to the
level that just acted as resistance, which now acts as support).
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

    name: str
    direction: str
    timeframe: str
    status: str

    breakout_level: float
    stop_loss: float
    target: float
    entry: float

    confidence: float
    pattern_height: float
    timeframe_bars: int

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

    PIVOT_WINDOW   = 5
    MIN_BARS       = 10
    SIMILAR_PCT    = 0.03
    RETEST_ZONE    = 0.008

    # ── Stop loss multipliers ──────────────────────────────────────────────────
    # RETEST: tight stop 1.5×ATR from breakout level → R:R 4-8×
    # BREAKOUT/FORMING: wide stop at pattern extremes → R:R ~1.0 (usually blocked)
    RETEST_ATR_MULT  = 1.5   # tight stop for retest entries
    PATTERN_ATR_MULT = 0.3   # buffer beyond pattern extreme for non-retest

    def scan_all(self, df_1h: pd.DataFrame) -> list:
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

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _resample(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        return df.resample(rule).agg(
            {"open": "first", "high": "max",
             "low": "min", "close": "last", "volume": "sum"}
        ).dropna()

    def _pivots(self, df: pd.DataFrame, w: int = None):
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

    def _retest_status(self, df: pd.DataFrame, level: float, direction: str) -> str:
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
        else:
            if close > level + tol:
                return "FORMING"
            if close < level - tol * 2:
                recent_highs = recent["high"].values
                if any(abs(float(v) - level) <= tol * 2 for v in recent_highs):
                    return "RETEST_CONFIRMED"
                return "AWAITING_RETEST"
            return "BREAKOUT"

    def _sl_bullish(self, status: str, level: float, pattern_low: float, atr: float) -> float:
        """
        Calculate stop-loss for a bullish pattern.

        RETEST_CONFIRMED → tight stop 1.5×ATR below broken level (now support).
                           Entry is near 'level' so risk is small. R:R = 4–8×.
        Everything else  → stop below the pattern's extreme low + buffer.
                           Wide stop → R:R ≈ 1.0 → blocked by MIN_RR filter.
        """
        if status == "RETEST_CONFIRMED":
            return level - atr * self.RETEST_ATR_MULT
        return pattern_low - atr * self.PATTERN_ATR_MULT

    def _sl_bearish(self, status: str, level: float, pattern_high: float, atr: float) -> float:
        """
        Calculate stop-loss for a bearish pattern.

        RETEST_CONFIRMED → tight stop 1.5×ATR above broken level (now resistance).
        Everything else  → stop above the pattern's extreme high + buffer.
        """
        if status == "RETEST_CONFIRMED":
            return level + atr * self.RETEST_ATR_MULT
        return pattern_high + atr * self.PATTERN_ATR_MULT

    # ── Pattern detectors ──────────────────────────────────────────────────────

    def _detect_double_bottom(self, df: pd.DataFrame, tf: str) -> Optional[PatternSignal]:
        """
        Double Bottom (W shape) — bullish reversal.
        Two troughs at roughly the same price with a peak between them.
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
        sl       = self._sl_bullish(status, neckline, min(t1_p, t2_p), atr_val)
        conf     = 0.75 if abs(t1_p - t2_p) / t1_p < 0.01 else 0.60

        return PatternSignal(
            name="double_bottom", direction="bullish", timeframe=tf,
            status=status, breakout_level=neckline,
            stop_loss=sl, target=neckline + height, entry=neckline,
            confidence=conf, pattern_height=height,
            timeframe_bars=t2_i - t1_i,
            key_levels={"trough1": t1_p, "trough2": t2_p, "neckline": neckline},
        )

    def _detect_double_top(self, df: pd.DataFrame, tf: str) -> Optional[PatternSignal]:
        """
        Double Top (M shape) — bearish reversal.
        Two peaks at roughly the same price with a trough between them.
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
        sl      = self._sl_bearish(status, neckline, max(h1_p, h2_p), atr_val)
        conf    = 0.75 if abs(h1_p - h2_p) / h1_p < 0.01 else 0.60

        return PatternSignal(
            name="double_top", direction="bearish", timeframe=tf,
            status=status, breakout_level=neckline,
            stop_loss=sl, target=neckline - height, entry=neckline,
            confidence=conf, pattern_height=height,
            timeframe_bars=h2_i - h1_i,
            key_levels={"peak1": h1_p, "peak2": h2_p, "neckline": neckline},
        )

    def _detect_head_shoulders(self, df: pd.DataFrame, tf: str) -> Optional[PatternSignal]:
        """
        Head & Shoulders — bearish reversal.
        Three peaks; the middle (head) is tallest; shoulders roughly equal.
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

        t1       = float(df["low"].iloc[ls_i : h_i + 1].min())
        t2       = float(df["low"].iloc[h_i  : rs_i + 1].min())
        neckline = (t1 + t2) / 2
        height   = h_p - neckline

        if height / neckline < 0.03:
            return None

        atr_val = self._atr(df)
        status  = self._retest_status(df, neckline, "bearish")
        sl      = self._sl_bearish(status, neckline, rs_p, atr_val)

        return PatternSignal(
            name="head_and_shoulders", direction="bearish", timeframe=tf,
            status=status, breakout_level=neckline,
            stop_loss=sl, target=neckline - height, entry=neckline,
            confidence=0.75, pattern_height=height,
            timeframe_bars=rs_i - ls_i,
            key_levels={
                "left_shoulder": ls_p, "head": h_p,
                "right_shoulder": rs_p, "neckline": neckline,
            },
        )

    def _detect_ascending_triangle(self, df: pd.DataFrame, tf: str) -> Optional[PatternSignal]:
        """
        Ascending Triangle — bullish continuation.
        Flat resistance top + rising lows (buyers getting more aggressive).

        Stop-loss (why R:R improves with RETEST_CONFIRMED):
          BREAKOUT:         stop at pattern's lowest low (~$60.7K) → risk $3.4K → R:R 1.0
          RETEST_CONFIRMED: stop 1.5×ATR below resistance (~$63.4K) → risk $0.6K → R:R 6.0
        """
        n = min(40, len(df))
        if n < 20:
            return None

        recent = df.iloc[-n:]
        h_vals = recent["high"].values
        l_vals = recent["low"].values

        res     = float(np.percentile(h_vals, 88))
        touches = sum(1 for h in h_vals if abs(h - res) / res < 0.005)
        if touches < 2:
            return None

        ls = float(np.polyfit(np.arange(n), l_vals, 1)[0]) / float(recent["close"].mean())
        if ls <= 0.0005:
            return None

        height  = res - float(recent["low"].min())
        if height / res < 0.02:
            return None

        atr_val = self._atr(df)
        status  = self._retest_status(df, res, "bullish")
        sl      = self._sl_bullish(status, res, float(recent["low"].min()), atr_val)

        return PatternSignal(
            name="ascending_triangle", direction="bullish", timeframe=tf,
            status=status, breakout_level=res,
            stop_loss=sl, target=res + height, entry=res,
            confidence=0.65, pattern_height=height,
            timeframe_bars=n,
            key_levels={"resistance": res, "triangle_low": float(recent["low"].min())},
        )

    def _detect_descending_triangle(self, df: pd.DataFrame, tf: str) -> Optional[PatternSignal]:
        """
        Descending Triangle — bearish continuation.
        Flat support bottom + declining highs.
        """
        n = min(40, len(df))
        if n < 20:
            return None

        recent = df.iloc[-n:]
        h_vals = recent["high"].values
        l_vals = recent["low"].values

        sup     = float(np.percentile(l_vals, 12))
        touches = sum(1 for l in l_vals if abs(l - sup) / sup < 0.005)
        if touches < 2:
            return None

        hs = float(np.polyfit(np.arange(n), h_vals, 1)[0]) / float(recent["close"].mean())
        if hs >= -0.0005:
            return None

        height  = float(recent["high"].max()) - sup
        atr_val = self._atr(df)
        status  = self._retest_status(df, sup, "bearish")
        sl      = self._sl_bearish(status, sup, float(recent["high"].max()), atr_val)

        return PatternSignal(
            name="descending_triangle", direction="bearish", timeframe=tf,
            status=status, breakout_level=sup,
            stop_loss=sl, target=sup - height, entry=sup,
            confidence=0.65, pattern_height=height,
            timeframe_bars=n,
            key_levels={"support": sup, "triangle_high": float(recent["high"].max())},
        )

    def _detect_symmetrical_triangle(self, df: pd.DataFrame, tf: str) -> Optional[PatternSignal]:
        """
        Symmetrical Triangle — neutral (direction decided at breakout).
        Both highs declining and lows rising.
        """
        n = min(40, len(df))
        if n < 20:
            return None

        recent = df.iloc[-n:]
        x   = np.arange(n)
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

        if dir_ == "bullish":
            sl = self._sl_bullish(status, bkl, apex_l, atr_val)
            tp = bkl + height
        else:
            sl = self._sl_bearish(status, bkl, apex_h, atr_val)
            tp = bkl - height

        return PatternSignal(
            name="symmetrical_triangle", direction=dir_, timeframe=tf,
            status=status, breakout_level=bkl,
            stop_loss=sl, target=tp, entry=bkl,
            confidence=0.55, pattern_height=height,
            timeframe_bars=n,
            key_levels={"apex_high": apex_h, "apex_low": apex_l},
        )

    def _detect_bull_flag(self, df: pd.DataFrame, tf: str) -> Optional[PatternSignal]:
        """
        Bull Flag — bullish continuation.
        Sharp upward move (flagpole) + downward-sloping channel (flag).
        Naturally achieves R:R 2+ when pole height > 2× flag depth.
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

        x   = np.arange(len(flag))
        avg = float(flag["close"].mean())
        fh  = float(np.polyfit(x, flag["high"].values, 1)[0]) / avg
        fl  = float(np.polyfit(x, flag["low"].values,  1)[0]) / avg

        if not (fh < -0.0003 and fl < -0.0003):
            return None

        upper_ch = float(flag["high"].max())
        pole_ht  = float(recent["close"].iloc[pole_end] - recent["close"].iloc[pole_start])
        atr_val  = self._atr(df)
        status   = self._retest_status(df, upper_ch, "bullish")
        sl       = self._sl_bullish(status, upper_ch, float(flag["low"].min()), atr_val)

        return PatternSignal(
            name="bull_flag", direction="bullish", timeframe=tf,
            status=status, breakout_level=upper_ch,
            stop_loss=sl, target=upper_ch + pole_ht, entry=upper_ch,
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
        Bear Flag — bearish continuation.
        Sharp downward move (pole) + upward-sloping channel (flag).
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

        x   = np.arange(len(flag))
        avg = float(flag["close"].mean())
        fh  = float(np.polyfit(x, flag["high"].values, 1)[0]) / avg
        fl  = float(np.polyfit(x, flag["low"].values,  1)[0]) / avg

        if not (fh > 0.0003 and fl > 0.0003):
            return None

        lower_ch = float(flag["low"].min())
        pole_ht  = float(recent["close"].iloc[pole_start] - recent["close"].iloc[pole_end])
        atr_val  = self._atr(df)
        status   = self._retest_status(df, lower_ch, "bearish")
        sl       = self._sl_bearish(status, lower_ch, float(flag["high"].max()), atr_val)

        return PatternSignal(
            name="bear_flag", direction="bearish", timeframe=tf,
            status=status, breakout_level=lower_ch,
            stop_loss=sl, target=lower_ch - pole_ht, entry=lower_ch,
            confidence=0.70, pattern_height=pole_ht,
            timeframe_bars=len(recent),
            key_levels={
                "flag_upper": float(flag["high"].max()),
                "flag_lower": lower_ch,
            },
        )

    def _detect_falling_wedge(self, df: pd.DataFrame, tf: str) -> Optional[PatternSignal]:
        """
        Falling Wedge — bullish.
        Both highs and lows declining but converging (highs fall faster).
        """
        n = min(40, len(df))
        if n < 20:
            return None

        recent = df.iloc[-n:]
        x   = np.arange(n)
        avg = float(recent["close"].mean())

        hs = float(np.polyfit(x, recent["high"].values, 1)[0]) / avg
        ls = float(np.polyfit(x, recent["low"].values,  1)[0]) / avg

        if not (hs < -0.001 and ls < -0.001 and hs < ls):
            return None

        upper_w = float(np.poly1d(np.polyfit(x, recent["high"].values, 1))(n - 1))
        lower_w = float(np.poly1d(np.polyfit(x, recent["low"].values,  1))(n - 1))
        height  = float(recent["high"].max() - recent["low"].min())

        atr_val = self._atr(df)
        status  = self._retest_status(df, upper_w, "bullish")
        sl      = self._sl_bullish(status, upper_w, lower_w, atr_val)

        return PatternSignal(
            name="falling_wedge", direction="bullish", timeframe=tf,
            status=status, breakout_level=upper_w,
            stop_loss=sl, target=upper_w + height * 0.618, entry=upper_w,
            confidence=0.62, pattern_height=height,
            timeframe_bars=n,
            key_levels={"upper_wedge": upper_w, "lower_wedge": lower_w},
        )

    def _detect_rising_wedge(self, df: pd.DataFrame, tf: str) -> Optional[PatternSignal]:
        """
        Rising Wedge — bearish.
        Both highs and lows rising but converging (lows rise faster).
        """
        n = min(40, len(df))
        if n < 20:
            return None

        recent = df.iloc[-n:]
        x   = np.arange(n)
        avg = float(recent["close"].mean())

        hs = float(np.polyfit(x, recent["high"].values, 1)[0]) / avg
        ls = float(np.polyfit(x, recent["low"].values,  1)[0]) / avg

        if not (hs > 0.001 and ls > 0.001 and ls > hs):
            return None

        upper_w = float(np.poly1d(np.polyfit(x, recent["high"].values, 1))(n - 1))
        lower_w = float(np.poly1d(np.polyfit(x, recent["low"].values,  1))(n - 1))
        height  = float(recent["high"].max() - recent["low"].min())

        atr_val = self._atr(df)
        status  = self._retest_status(df, lower_w, "bearish")
        sl      = self._sl_bearish(status, lower_w, upper_w, atr_val)

        return PatternSignal(
            name="rising_wedge", direction="bearish", timeframe=tf,
            status=status, breakout_level=lower_w,
            stop_loss=sl, target=lower_w - height * 0.618, entry=lower_w,
            confidence=0.62, pattern_height=height,
            timeframe_bars=n,
            key_levels={"upper_wedge": upper_w, "lower_wedge": lower_w},
        )
