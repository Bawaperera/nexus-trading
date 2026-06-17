"""
NEXUS v2 — Confluence Engine
==============================
Combines pattern signals with MTF alignment, indicators, volume,
sentiment, and on-chain derivatives into a single score (0–100).

A signal only fires when:
  1. A valid pattern is detected
  2. At least 3 independent confluences agree
  3. Score >= 60 (Telegram signal) or >= 60 (auto-trade)
  4. R:R >= 2.0 (only achievable with RETEST_CONFIRMED tight stop)

Scoring breakdown (max 100):
  Pattern quality:    0–25  (confidence × 25)
  MTF alignment:      0–20  (4 TF = 20, 3 TF = 15, 2 TF = 8)
  Retest confirmed:   0–20  (confirmed = 20, breakout = 5)
  Volume:             0–15  (2× = 15, 1.5× = 8)
  RSI:                0–10  (favorable RSI = 10)
  Sentiment:          0–10  (aligned with signal = 10)
  On-chain adj:      -5–+5  (funding rate + L/S ratio bias)

Memory bonus/penalty: patterns with >65% recent win rate get +5,
patterns with <50% recent win rate get -5.
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
        result = engine.score(
            pattern, mtf_features, df_1h,
            sentiment_score, fg_value,
            trade_memory=[], onchain_adj=0
        )
        if result["tradeable_signal"]:
            executor.execute(result)
    """

    SIGNAL_THRESHOLD     = 60
    AUTO_TRADE_THRESHOLD = 60
    MIN_RR               = 2.0
    AVG_VOLUME_PERIODS   = 20

    def score(
        self,
        pattern,
        mtf_features:    dict,
        df_1h:           pd.DataFrame,
        sentiment_score: float,
        fg_value:        int,
        trade_memory:    list = None,
        onchain_adj:     int  = 0,
    ) -> dict:
        total   = 0
        reasons = []

        # ── 1. Pattern quality (0–25) ──────────────────────────────────────
        p_score = round(pattern.confidence * 25)
        total  += p_score
        reasons.append(f"+{p_score} Pattern confidence {pattern.confidence:.0%}")

        # ── 2. Multi-timeframe alignment (0–20) ───────────────────────────
        direction = pattern.direction
        bull_c    = int(mtf_features.get("mtf_bull_count", 0))
        bear_c    = int(mtf_features.get("mtf_bear_count", 0))
        aligned   = bull_c if direction == "bullish" else bear_c

        if   aligned >= 4: tf_score = 20
        elif aligned >= 3: tf_score = 15
        elif aligned >= 2: tf_score = 8
        else:              tf_score = 0

        total  += tf_score
        reasons.append(f"+{tf_score} MTF {aligned}/4 timeframes aligned")

        # ── 3. Retest status (0–20) ────────────────────────────────────────
        retest_scores = {
            "RETEST_CONFIRMED": 20,
            "AWAITING_RETEST":  8,
            "BREAKOUT":         5,
            "FORMING":          0,
        }
        rt_score = retest_scores.get(pattern.status, 0)
        total   += rt_score
        reasons.append(f"+{rt_score} Status: {pattern.status}")

        # ── 4. Volume confirmation (0–15) ──────────────────────────────────
        vol_ratio = 1.0
        vol_score = 0
        try:
            avg_vol    = float(df_1h["volume"].iloc[-self.AVG_VOLUME_PERIODS:].mean())
            recent_vol = float(df_1h["volume"].iloc[-3:].mean())
            vol_ratio  = recent_vol / avg_vol if avg_vol > 0 else 1.0

            if   vol_ratio >= 2.0: vol_score = 15
            elif vol_ratio >= 1.5: vol_score = 8
            elif vol_ratio >= 1.2: vol_score = 4

            reasons.append(f"+{vol_score} Volume {vol_ratio:.1f}× average")
        except Exception as e:
            log.debug(f"Volume check failed: {e}")
            reasons.append("+0 Volume data unavailable")

        total += vol_score

        # ── 5. RSI indicator (0–10) ────────────────────────────────────────
        rsi_val   = float(mtf_features.get("mtf_daily_rsi", 50.0))
        rsi_score = 0

        if direction == "bullish":
            if   rsi_val < 40: rsi_score = 10
            elif rsi_val < 55: rsi_score = 7
            elif rsi_val < 65: rsi_score = 3
        else:
            if   rsi_val > 60: rsi_score = 10
            elif rsi_val > 45: rsi_score = 7
            elif rsi_val > 35: rsi_score = 3

        total   += rsi_score
        label    = "oversold/overbought" if rsi_score == 10 else "neutral" if rsi_score >= 3 else "risky"
        reasons.append(f"+{rsi_score} RSI {rsi_val:.0f} ({label})")

        # ── 6. Sentiment (0–10) ────────────────────────────────────────────
        sent_score = 0
        if direction == "bullish":
            if   sentiment_score > 0.1:  sent_score = 10
            elif sentiment_score > -0.2: sent_score = 6
        else:
            if   sentiment_score < -0.1: sent_score = 10
            elif sentiment_score < 0.2:  sent_score = 6

        total   += sent_score
        reasons.append(f"+{sent_score} Sentiment {sentiment_score:+.2f}")

        # ── 7. On-chain derivatives adjustment (-5 to +5) ─────────────────
        onchain_adj = max(-5, min(5, int(onchain_adj)))
        if onchain_adj != 0:
            direction_word = "bullish" if onchain_adj > 0 else "bearish"
            reasons.append(
                f"{'+' if onchain_adj > 0 else ''}{onchain_adj} "
                f"On-chain bias ({direction_word}: funding/L/S signal)"
            )
        total += onchain_adj

        # ── Memory bonus/penalty ───────────────────────────────────────────
        memory_adj = 0
        if trade_memory:
            past = [
                t for t in trade_memory[-50:]
                if t.get("pattern_name") == pattern.name
                and t.get("outcome") in ("WIN", "LOSS")
            ]
            if len(past) >= 5:
                wins = sum(1 for t in past if t["outcome"] == "WIN")
                wr   = wins / len(past)
                if wr >= 0.65:
                    memory_adj = 5
                    reasons.append(f"+5 Pattern memory {wr:.0%} WR ({len(past)} trades)")
                elif wr < 0.50:
                    memory_adj = -5
                    reasons.append(f"-5 Pattern memory {wr:.0%} WR ({len(past)} trades)")

        total = max(0, min(100, total + memory_adj))

        # ── Count independent confluences ──────────────────────────────────
        n_confluences = sum([
            1 if tf_score  >= 8  else 0,
            1 if rt_score  >= 5  else 0,
            1 if vol_score >= 4  else 0,
            1 if rsi_score >= 3  else 0,
            1 if sent_score >= 6 else 0,
        ])

        tradeable_signal    = (total >= self.SIGNAL_THRESHOLD    and n_confluences >= 3)
        tradeable_autotrade = (total >= self.AUTO_TRADE_THRESHOLD and n_confluences >= 3)

        if pattern.risk_reward < self.MIN_RR:
            tradeable_autotrade = False
            reasons.append(
                f"R:R {pattern.risk_reward:.1f} < {self.MIN_RR} min: auto-trade blocked "
                f"(unlocks at RETEST_CONFIRMED)"
            )

        trade_type = self._decide_trade_type(total)

        log.info(
            f"Confluence [{pattern.name} {pattern.timeframe}] "
            f"score={total}/100 | {n_confluences} confluences | "
            f"R:R={pattern.risk_reward} | onchain_adj={onchain_adj:+d} | "
            f"{'SIGNAL' if tradeable_signal else 'skip'} | "
            f"{'AUTO-TRADE' if tradeable_autotrade else ''}"
        )

        return {
            "score":               total,
            "n_confluences":       n_confluences,
            "reasons":             reasons,
            "tradeable_signal":    tradeable_signal,
            "tradeable_autotrade": tradeable_autotrade,
            "trade_type":          trade_type,
            "pattern":             pattern,
            "rsi_val":             rsi_val,
            "vol_ratio":           vol_ratio,
            "sentiment_score":     sentiment_score,
            "fg_value":            fg_value,
            # Component breakdown for dashboard display
            "component_scores": {
                "pattern":   p_score,
                "mtf":       tf_score,
                "retest":    rt_score,
                "volume":    vol_score,
                "rsi":       rsi_score,
                "sentiment": sent_score,
                "onchain":   onchain_adj,
                "memory":    memory_adj,
            },
        }

    def _decide_trade_type(self, score: int) -> dict:
        """
        Decide spot vs futures and leverage based on signal score.
        Score 85+  → Spot + Futures 3×
        Score 75+  → Spot + Futures 2×
        Score 60+  → Spot only
        Score < 60 → No trade
        """
        if   score >= 85:
            return {"spot": True, "futures": True,  "leverage": 3, "size_pct": 0.015}
        elif score >= 75:
            return {"spot": True, "futures": True,  "leverage": 2, "size_pct": 0.010}
        elif score >= 60:
            return {"spot": True, "futures": False, "leverage": 1, "size_pct": 0.010}
        else:
            return {"spot": False,"futures": False, "leverage": 0, "size_pct": 0.0}
