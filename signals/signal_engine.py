"""
NEXUS — Signal Engine
Combines ML model predictions + real-time sentiment → final trade signal.

Signal logic:
  80% weight → XGBoost model prediction (price direction + confidence)
  20% weight → Sentiment score (news + fear/greed)

Both must agree (or one must be very strong) before generating a signal.
This prevents trading against the news — a common beginner mistake.
"""

import logging
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    timestamp: datetime
    symbol: str
    action: str          # "BUY", "SELL", "HOLD"
    confidence: float    # 0.0 to 1.0
    model_score: float   # raw model probability
    sentiment_score: float
    composite_score: float
    reasoning: list      # list of reasons for the signal
    entry_price: float
    atr: float


class SignalEngine:
    """
    Takes raw model output + sentiment report and generates final signals.

    Usage:
        engine = SignalEngine(symbol="BTC/USDT")
        signal = engine.generate(
            model_proba={"BUY": 0.72, "SELL": 0.18, "HOLD": 0.10},
            sentiment_score=-0.25,
            entry_price=65000,
            atr=800
        )
    """

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        model_weight: float = 0.80,
        sentiment_weight: float = 0.20,
        min_confidence: float = 0.58,    # Must be 58%+ confident to trade
        sentiment_veto_threshold: float = -0.5,  # Strong news can block a trade
    ):
        self.symbol                  = symbol
        self.model_weight            = model_weight
        self.sentiment_weight        = sentiment_weight
        self.min_confidence          = min_confidence
        self.sentiment_veto_threshold = sentiment_veto_threshold

        self.signal_history = []
        log.info(f"SignalEngine ready | {symbol} | Model: {model_weight:.0%} / Sentiment: {sentiment_weight:.0%}")

    def generate(
        self,
        model_proba: dict,   # {"BUY": 0.72, "SELL": 0.18, "HOLD": 0.10}
        sentiment_score: float,
        entry_price: float,
        atr: float,
        current_regime: int = 0,   # 1=uptrend, -1=downtrend, 0=sideways
        fear_greed: int = 50,      # 0-100
    ) -> TradeSignal:
        """
        Generate a trade signal.

        Args:
            model_proba: Dict with BUY/SELL/HOLD probabilities (must sum to ~1.0)
            sentiment_score: From NewsCollector (-1.0 to +1.0)
            entry_price: Current market price
            atr: Current ATR value
            current_regime: Market structure regime
            fear_greed: Fear & Greed Index value (0=extreme fear, 100=extreme greed)

        Returns:
            TradeSignal with action, confidence, and reasoning
        """
        reasoning = []

        # ── Step 1: Read model probabilities ──────────────────────────────
        buy_prob  = model_proba.get("BUY",  0.33)
        sell_prob = model_proba.get("SELL", 0.33)
        hold_prob = model_proba.get("HOLD", 0.34)

        # Model direction: positive = bullish signal, negative = bearish
        raw_model_direction = buy_prob - sell_prob  # -1.0 to +1.0

        # ── Step 2: Sentiment contribution ────────────────────────────────
        reasoning.append(f"Model: BUY {buy_prob:.0%} / SELL {sell_prob:.0%} / HOLD {hold_prob:.0%}")
        reasoning.append(f"Sentiment: {sentiment_score:+.3f} | Fear/Greed: {fear_greed}")

        # ── Step 3: Sentiment VETO (hard block) ───────────────────────────
        # If model says BUY but news is extremely bearish → HOLD (don't fight the news)
        if raw_model_direction > 0.2 and sentiment_score < self.sentiment_veto_threshold:
            reasoning.append(f"⚠️ Sentiment VETO: news too bearish ({sentiment_score:.2f}) to take BUY")
            return self._make_signal("HOLD", 0.0, raw_model_direction, sentiment_score,
                                     entry_price, atr, reasoning)

        if raw_model_direction < -0.2 and sentiment_score > abs(self.sentiment_veto_threshold):
            reasoning.append(f"⚠️ Sentiment VETO: news too bullish ({sentiment_score:.2f}) to take SELL")
            return self._make_signal("HOLD", 0.0, raw_model_direction, sentiment_score,
                                     entry_price, atr, reasoning)

        # ── Step 4: Composite score ────────────────────────────────────────
        composite = (self.model_weight * raw_model_direction) + \
                    (self.sentiment_weight * sentiment_score)
        composite = max(-1.0, min(1.0, composite))

        # ── Step 5: Market regime filter ──────────────────────────────────
        if current_regime == 1:  # Uptrend
            reasoning.append("✅ Market regime: UPTREND — BUY signals favoured")
            if composite < 0:
                composite *= 0.7  # Reduce strength of SELL signals in uptrend
        elif current_regime == -1:  # Downtrend
            reasoning.append("⚠️ Market regime: DOWNTREND — SELL signals favoured")
            if composite > 0:
                composite *= 0.7  # Reduce strength of BUY signals in downtrend

        # ── Step 6: Fear & Greed bonus/penalty ────────────────────────────
        # Classic contrarian: extreme fear = buy opportunity, extreme greed = caution
        if fear_greed <= 20:  # Extreme Fear
            reasoning.append(f"🔥 Extreme Fear ({fear_greed}) — contrarian BUY zone")
            if composite > 0:
                composite = min(1.0, composite * 1.15)  # Slight boost to BUY
        elif fear_greed >= 80:  # Extreme Greed
            reasoning.append(f"🎯 Extreme Greed ({fear_greed}) — contrarian caution zone")
            if composite > 0:
                composite *= 0.85  # Dampen BUY signals in greed zone

        # ── Step 7: Final signal determination ───────────────────────────
        confidence = abs(composite)

        if composite > 0.15 and confidence >= self.min_confidence:
            action = "BUY"
            reasoning.append(f"✅ BUY signal | Composite: {composite:+.3f} | Confidence: {confidence:.0%}")
        elif composite < -0.15 and confidence >= self.min_confidence:
            action = "SELL"
            reasoning.append(f"🔴 SELL signal | Composite: {composite:+.3f} | Confidence: {confidence:.0%}")
        else:
            action = "HOLD"
            reasoning.append(f"⏸️ HOLD | Signal too weak ({confidence:.0%} < {self.min_confidence:.0%} threshold)")

        signal = self._make_signal(action, confidence, composite, sentiment_score,
                                   entry_price, atr, reasoning)
        self.signal_history.append(signal)

        self._log_signal(signal)
        return signal

    def _make_signal(self, action, confidence, composite, sentiment,
                     entry_price, atr, reasoning) -> TradeSignal:
        return TradeSignal(
            timestamp        = datetime.now(tz=timezone.utc),
            symbol           = self.symbol,
            action           = action,
            confidence       = round(confidence, 4),
            model_score      = round(composite, 4),
            sentiment_score  = round(sentiment, 4),
            composite_score  = round(composite, 4),
            reasoning        = reasoning,
            entry_price      = entry_price,
            atr              = atr,
        )

    def _log_signal(self, signal: TradeSignal):
        icon = {"BUY": "✅", "SELL": "🔴", "HOLD": "⏸️"}[signal.action]
        log.info(
            f"{icon} SIGNAL [{signal.symbol}] {signal.action} | "
            f"Confidence: {signal.confidence:.0%} | "
            f"Price: ${signal.entry_price:,.2f} | "
            f"Sentiment: {signal.sentiment_score:+.3f}"
        )
        for r in signal.reasoning:
            log.debug(f"   {r}")
