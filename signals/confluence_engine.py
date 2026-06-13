"""
NEXUS v2 — Confluence Engine
==============================
Combines pattern signals with MTF alignment, indicators, volume,
and sentiment into a single score (0–100).

A signal only fires when:
  1. A valid pattern is detected (RETEST_CONFIRMED preferred)
  2. At least 3 independent confluences agree
  3. Score >= 60 (for signals) or >= 75 (for auto-trade)
  4. R:R >= 2.0

Scoring breakdown (max 100):
  Pattern quality:    0–25  (confidence × 25)
  MTF alignment:      0–20  (4 TF = 20, 3 TF = 15, 2 TF = 8)
  Retest confirmed:   0–20  (confirmed = 20, breakout = 5)
  Volume:             0–15  (2× = 15, 1.5× = 8)
  Indicator (RSI):    0–10  (favorable position = 10)
  Sentiment:          0–10  (aligned with signal = 10)

This score is stored in trade memory and used to weight future
pattern reliability (patterns with high-score signals that win
get higher future confidence weights).
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class ConfluenceEngine:
    """
    Scores a pattern signal against all available market data.

    Usage:
        engine = ConfluenceEngine()
        result = engine.score(pattern, mtf_features, df_1h, sentiment_score, fg_value)
        if result["tradeable"]:
            execute_trade(result)
    """

    SIGNAL_THRESHOLD   = 60   # minimum score to send a Telegram alert
    AUTO_TRADE_THRESHOLD = 75  # minimum score to auto-place an order
    MIN_RR             = 2.0   # minimum risk:reward ratio
    AVG_VOLUME_PERIODS = 20    # periods for average volume calculation

    def score(
        self,
        pattern,                # PatternSignal
        mtf_features: dict,     # from MultiTimeframeAnalyzer
        df_1h: pd.DataFrame,    # 1H OHLCV for volume check
        sentiment_score: float, # -1.0 to +1.0
        fg_value: int,          # 0-100 Fear & Greed
        trade_memory: list = None,  # past trades for pattern reliability
    ) -> dict:
        """
        Score a pattern signal across all confluence factors.

        Returns:
            dict with score, reasons, signal level, and trade parameters.
        """
        total   = 0
        reasons = []

        # ── 1. Pattern quality (0–25) ──────────────────────────────────────────
        p_score = round(pattern.confidence * 25)
        total  += p_score
        reasons.append(f"Pattern confidence {pattern.confidence:.0%}: +{p_score}")

        # ── 2. Multi-timeframe alignment (0–20) ───────────────────────────────
        direction = pattern.direction
        bull_c = mtf_features.get("mtf_bull_count", 0)
        bear_c = mtf_features.get("mtf_bear_count", 0)
        aligned = bull_c if direction == "bullish" else bear_c

        if aligned >= 4:
            tf_score = 20
        elif aligned >= 3:
            tf_score = 15
        elif aligned >= 2:
            tf_score = 8
        else:
            tf_score = 0

        total  += tf_score
        reasons.append(f"MTF {aligned}/4 aligned: +{tf_score}")

        # ── 3. Retest status (0–20) ────────────────────────────────────────────
        retest_scores = {
            "RETEST_CONFIRMED": 20,
            "AWAITING_RETEST":  8,
            "BREAKOUT":         5,
            "FORMING":          0,
        }
        rt_score = retest_scores.get(pattern.status, 0)
        total   += rt_score
        reasons.append(f"Status {pattern.status}: +{rt_score}")

        # ── 4. Volume confirmation (0–15) ──────────────────────────────────────
        vol_score = 0
        try:
            avg_vol     = float(df_1h["volume"].iloc[-self.AVG_VOLUME_PERIODS:].mean())
            recent_vol  = float(df_1h["volume"].iloc[-3:].mean())
            vol_ratio   = recent_vol / (avg_vol + 1e-9)

            if vol_ratio >= 2.0:
                vol_score = 15
            elif vol_ratio >= 1.5:
                vol_score = 8
            elif vol_ratio >= 1.2:
                vol_score = 4

            reasons.append(f"Volume {vol_ratio:.1f}× avg: +{vol_score}")
        except Exception:
            reasons.append("Volume data unavailable: +0")

        total += vol_score

        # ── 5. RSI indicator (0–10) ────────────────────────────────────────────
        rsi_val   = mtf_features.get("mtf_daily_rsi", 50.0)
        rsi_score = 0

        if direction == "bullish":
            if rsi_val < 40:
                rsi_score = 10   # oversold + bullish pattern = excellent
            elif rsi_val < 55:
                rsi_score = 7    # neutral, room to run
            elif rsi_val < 65:
                rsi_score = 3    # getting hot
            else:
                rsi_score = 0    # overbought, risky to buy
        else:
            if rsi_val > 60:
                rsi_score = 10
            elif rsi_val > 45:
                rsi_score = 7
            elif rsi_val > 35:
                rsi_score = 3
            else:
                rsi_score = 0

        total   += rsi_score
        reasons.append(f"RSI {rsi_val:.0f}: +{rsi_score}")

        # ── 6. Sentiment (0–10) ────────────────────────────────────────────────
        sent_score = 0
        if direction == "bullish":
            if sentiment_score > 0.1:
                sent_score = 10
            elif sentiment_score > -0.2:
                sent_score = 6   # neutral is fine for bullish
            else:
                sent_score = 0   # very bearish news blocks
        else:
            if sentiment_score < -0.1:
                sent_score = 10
            elif sentiment_score < 0.2:
                sent_score = 6
            else:
                sent_score = 0

        total   += sent_score
        reasons.append(f"Sentiment {sentiment_score:+.2f}: +{sent_score}")

        # ── Pattern memory bonus/penalty ──────────────────────────────────────
        memory_adj = 0
        if trade_memory:
            recent_pattern_trades = [
                t for t in trade_memory[-50:]
                if t.get("pattern_name") == pattern.name
                and t.get("outcome") in ("WIN", "LOSS")
            ]
            if len(recent_pattern_trades) >= 5:
                wins    = sum(1 for t in recent_pattern_trades if t["outcome"] == "WIN")
                wr      = wins / len(recent_pattern_trades)
                # Patterns with >60% recent win rate get a bonus
                if wr >= 0.65:
                    memory_adj = 5
                elif wr >= 0.50:
                    memory_adj = 0
                else:
                    memory_adj = -5

                total   += memory_adj
                reasons.append(
                    f"Pattern memory ({wr:.0%} recent WR, "
                    f"{len(recent_pattern_trades)} trades): {memory_adj:+d}"
                )

        # ── Final score and decision ──────────────────────────────────────────
        total = max(0, min(100, total))

        # Count independent confluences (pattern + each positive factor)
        n_confluences = sum([
            1 if tf_score >= 8    else 0,  # MTF
            1 if rt_score >= 5    else 0,  # retest
            1 if vol_score >= 4   else 0,  # volume
            1 if rsi_score >= 3   else 0,  # RSI
            1 if sent_score >= 6  else 0,  # sentiment
        ])

        tradeable_signal   = total >= self.SIGNAL_THRESHOLD and n_confluences >= 3
        tradeable_autotrade = total >= self.AUTO_TRADE_THRESHOLD and n_confluences >= 3

        # For auto-trade: also require R:R >= 2.0
        if pattern.risk_reward < self.MIN_RR:
            tradeable_autotrade = False
            reasons.append(f"R:R {pattern.risk_reward} < {self.MIN_RR} min: auto-trade blocked")

        # Spot vs futures decision based on score
        trade_type = self._decide_trade_type(total, direction)

        log.info(
            f"Confluence score: {total}/100 | {n_confluences} confluences | "
            f"{'SIGNAL' if tradeable_signal else 'SKIP'} | "
            f"Auto: {'YES' if tradeable_autotrade else 'NO'}"
        )

        return {
            "score":              total,
            "n_confluences":      n_confluences,
            "reasons":            reasons,
            "tradeable_signal":   tradeable_signal,
            "tradeable_autotrade": tradeable_autotrade,
            "trade_type":         trade_type,
            "pattern":            pattern,
            "rsi_val":            rsi_val,
            "vol_ratio":          vol_ratio if vol_score > 0 else 1.0,
            "sentiment_score":    sentiment_score,
            "fg_value":           fg_value,
        }

    def _decide_trade_type(self, score: int, direction: str) -> dict:
        """
        Decide whether to use spot, futures, or both based on score.
        Higher score = more conviction = use futures with leverage.

        Returns dict: {spot: bool, futures: bool, leverage: int, size_pct: float}
        """
        if score >= 85:
            return {"spot": True, "futures": True, "leverage": 3, "size_pct": 0.015}
        elif score >= 75:
            return {"spot": True, "futures": True, "leverage": 2, "size_pct": 0.010}
        elif score >= 60:
            return {"spot": True, "futures": False, "leverage": 1, "size_pct": 0.010}
        else:
            return {"spot": False, "futures": False, "leverage": 0, "size_pct": 0.0}
