"""
NEXUS — Main Orchestrator
The brain that ties all 6 layers together into a real-time trading loop.

Real-time loop (every candle close):
  1. RealtimeStream fires → new closed candle
  2. FeatureEngineer recomputes 102 features from rolling buffer
  3. ML Model (XGBoost) runs inference → BUY/SELL/HOLD probabilities
  4. NewsCollector provides latest sentiment score
  5. SignalEngine combines model + sentiment → final signal
  6. RiskEngine validates + calculates position size
  7. PaperTrader (or live) executes the order

Mode: PAPER (safe, fake money) by default.
      Set mode="live" only after 30+ profitable paper trading days.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from data.realtime_stream import RealtimeStream
from data.news_collector import NewsCollector
from data.feature_engineer import FeatureEngineer
from signals.signal_engine import SignalEngine
from risk.risk_engine import RiskEngine

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/nexus.log"),
    ]
)


class NEXUSOrchestrator:
    """
    The main NEXUS trading loop.

    Usage:
        nexus = NEXUSOrchestrator(
            symbol="btcusdt",
            timeframe="1h",
            account_size=1000,
            mode="paper"
        )
        asyncio.run(nexus.run())
    """

    def __init__(
        self,
        symbol: str = "btcusdt",
        timeframe: str = "1h",
        account_size: float = 1000,
        mode: str = "paper",  # "paper" or "live"
        risk_pct: float = 0.01,
        ml_model=None,        # Pass trained XGBoost model (Phase 2)
        cryptopanic_token: str = None,
        binance_api_key: str = None,
        binance_api_secret: str = None,
    ):
        self.symbol    = symbol
        self.timeframe = timeframe
        self.mode      = mode

        display_symbol = symbol.upper().replace("USDT", "/USDT")

        log.info(f"")
        log.info(f"{'='*55}")
        log.info(f"  NEXUS AI TRADING SYSTEM — Starting Up")
        log.info(f"  Symbol: {display_symbol} | TF: {timeframe} | Mode: {mode.upper()}")
        log.info(f"  Account: ${account_size:,.2f} | Risk/trade: {risk_pct*100:.1f}%")
        log.info(f"{'='*55}")

        # ── Layer 1: Real-time price stream ───────────────────────────────
        self.stream = RealtimeStream(
            symbol=symbol,
            timeframe=timeframe,
            buffer_size=300,
        )

        # ── Layer 2: Feature engineering ──────────────────────────────────
        self.feature_engineer = FeatureEngineer(target_horizon=1)

        # ── Layer 2b: News & sentiment ────────────────────────────────────
        self.news_collector = NewsCollector(
            cryptopanic_token=cryptopanic_token,
            cache_duration_minutes=15,
        )

        # ── Layer 3: ML Model ─────────────────────────────────────────────
        # If no model passed yet (Phase 1), use a rule-based fallback
        self.ml_model = ml_model  # Set to trained XGBoost after Phase 2

        # ── Layer 4: Signal engine ────────────────────────────────────────
        self.signal_engine = SignalEngine(
            symbol=display_symbol,
            model_weight=0.80,
            sentiment_weight=0.20,
        )

        # ── Layer 4b: Risk engine ─────────────────────────────────────────
        self.risk_engine = RiskEngine(
            account_size=account_size,
            risk_pct=risk_pct,
            rr_ratio=2.0,
            max_positions=3,
        )

        # ── State tracking ────────────────────────────────────────────────
        self.candles_processed = 0
        self.signals_generated = 0
        self.trades_taken      = 0

        log.info("All 6 layers initialized. Waiting for first candle close...")

    async def run(self):
        """Start the real-time trading loop."""
        log.info(f"Connecting to Binance WebSocket [{self.symbol}@kline_{self.timeframe}]...")

        # Pre-fetch sentiment on startup (don't wait for candle)
        log.info("Pre-loading sentiment data...")
        try:
            report = self.news_collector.get_sentiment_report()
            log.info(f"Initial sentiment: {report.sentiment_label} ({report.composite_score:+.3f})")
        except Exception as e:
            log.warning(f"Sentiment pre-load failed: {e}")

        # Start the WebSocket stream
        await self.stream.start(on_candle_close=self._on_candle_close)

    async def _on_candle_close(self, raw_df):
        """
        Called automatically by RealtimeStream on every candle close.
        This is the core real-time trading loop.
        """
        self.candles_processed += 1
        candle_time = raw_df.index[-1]
        close_price = float(raw_df["close"].iloc[-1])

        log.info(f"\n{'─'*50}")
        log.info(f"CANDLE #{self.candles_processed} CLOSED | {candle_time} | Price: ${close_price:,.2f}")

        # ── Step 1: Engineer features ──────────────────────────────────────
        try:
            features_df = self.feature_engineer.build(raw_df)
            if features_df.empty or len(features_df) < 10:
                log.warning("Not enough data for feature engineering, waiting...")
                return
        except Exception as e:
            log.error(f"Feature engineering failed: {e}")
            return

        latest = features_df.iloc[-1]
        atr    = float(latest.get("atr_14", close_price * 0.01))
        regime = int(latest.get("regime", 0))

        # ── Step 2: ML Model inference ────────────────────────────────────
        model_proba = self._run_model(features_df)

        # ── Step 3: Get latest sentiment ──────────────────────────────────
        sentiment_report = self.news_collector.get_sentiment_report()

        # ── Step 4: Generate signal ───────────────────────────────────────
        signal = self.signal_engine.generate(
            model_proba     = model_proba,
            sentiment_score = sentiment_report.composite_score,
            entry_price     = close_price,
            atr             = atr,
            current_regime  = regime,
            fear_greed      = sentiment_report.fear_greed_value,
        )
        self.signals_generated += 1

        # ── Step 5: Risk check & execution ───────────────────────────────
        if signal.action in ("BUY", "SELL"):
            direction = 1 if signal.action == "BUY" else -1

            order = self.risk_engine.calculate_order(
                symbol      = signal.symbol,
                signal      = direction,
                confidence  = signal.confidence,
                entry_price = close_price,
                atr         = atr,
            )

            if order["approved"]:
                self.trades_taken += 1
                await self._execute(order, signal, sentiment_report)
            else:
                log.info(f"⛔ Order rejected by risk engine: {order['reject_reason']}")

        # ── Step 6: Status summary ────────────────────────────────────────
        stats = self.risk_engine.get_stats()
        account = self.risk_engine.account_size
        log.info(
            f"STATUS | Account: ${account:,.2f} | "
            f"Trades: {self.trades_taken} | "
            f"Win rate: {stats.get('win_rate', 0):.0%} | "
            f"Profit factor: {stats.get('profit_factor', 0):.2f}"
        )

    async def _execute(self, order, signal, sentiment_report):
        """Execute (paper or live) and log to journal."""
        if self.mode == "paper":
            log.info(
                f"\n📋 PAPER TRADE #{self.trades_taken}\n"
                f"   Action:    {order['direction']}\n"
                f"   Symbol:    {order['symbol']}\n"
                f"   Entry:     ${order['entry_price']:,.2f}\n"
                f"   Stop Loss: ${order['stop_loss']:,.2f} (-{order['sl_pct']:.2f}%)\n"
                f"   Take Prof: ${order['take_profit']:,.2f}\n"
                f"   Size:      ${order['position_size_usd']:,.2f}\n"
                f"   Risk:      ${order['risk_amount_usd']:.2f} ({order['effective_risk_pct']:.2f}%)\n"
                f"   R:R:       {order['reward_risk_ratio']:.1f}:1\n"
                f"   Confidence:{signal.confidence:.0%}\n"
                f"   Sentiment: {sentiment_report.sentiment_label} ({sentiment_report.composite_score:+.3f})\n"
                f"   Reasoning:\n" +
                "\n".join(f"     {r}" for r in signal.reasoning)
            )
            self._log_journal(order, signal, sentiment_report)
        else:
            # Live trading via Binance API
            # Implemented in Phase 4 after 30+ profitable paper days
            log.warning("LIVE mode not yet enabled — use paper mode first")

    def _run_model(self, features_df) -> dict:
        """
        Run the ML model on the latest features.

        If no model is loaded (Phase 1), uses a simple rule-based fallback.
        Replace self.ml_model with the trained XGBoost after Phase 2.
        """
        if self.ml_model is not None:
            # Phase 2+: Real ML model
            feature_cols = self.feature_engineer._get_feature_cols(features_df)
            X_latest     = features_df[feature_cols].iloc[[-1]]
            proba        = self.ml_model.predict_proba(X_latest)[0]
            classes      = self.ml_model.classes_

            proba_dict = {str(int(cls)): float(p) for cls, p in zip(classes, proba)}
            return {
                "BUY":  proba_dict.get("1",  0.33),
                "SELL": proba_dict.get("-1", 0.33),
                "HOLD": proba_dict.get("0",  0.34),
            }
        else:
            # Phase 1 fallback: rule-based signal using technical indicators
            return self._rule_based_signal(features_df)

    def _rule_based_signal(self, features_df) -> dict:
        """
        Simple rule-based signal as Phase 1 placeholder.
        Will be replaced by XGBoost in Phase 2.
        Rules: RSI + MACD + EMA cross confirmation
        """
        latest = features_df.iloc[-1]

        score = 0
        total = 0

        rsi = latest.get("rsi_14", 50)
        if rsi < 35: score += 1   # oversold
        elif rsi > 65: score -= 1 # overbought
        total += 1

        macd_hist = latest.get("macd_hist", 0)
        if macd_hist > 0: score += 1
        elif macd_hist < 0: score -= 1
        total += 1

        ema_cross = latest.get("ema_9_21_cross", 0)
        if ema_cross > 0: score += 1
        elif ema_cross < 0: score -= 1
        total += 1

        regime = int(latest.get("regime", 0))
        if regime == 1: score += 0.5
        elif regime == -1: score -= 0.5
        total += 0.5

        normalized = score / total if total > 0 else 0

        if normalized > 0.3:
            buy_p, sell_p, hold_p = 0.60, 0.15, 0.25
        elif normalized < -0.3:
            buy_p, sell_p, hold_p = 0.15, 0.60, 0.25
        else:
            buy_p, sell_p, hold_p = 0.25, 0.25, 0.50

        return {"BUY": buy_p, "SELL": sell_p, "HOLD": hold_p}

    def _log_journal(self, order, signal, sentiment):
        """Append trade to CSV journal for performance tracking."""
        import csv
        os.makedirs("logs", exist_ok=True)
        row = {
            "timestamp":   signal.timestamp.isoformat(),
            "symbol":      order["symbol"],
            "action":      order["direction"],
            "entry_price": order["entry_price"],
            "stop_loss":   order["stop_loss"],
            "take_profit": order["take_profit"],
            "size_usd":    order["position_size_usd"],
            "risk_usd":    order["risk_amount_usd"],
            "confidence":  signal.confidence,
            "sentiment":   sentiment.composite_score,
            "fear_greed":  sentiment.fear_greed_value,
            "mode":        self.mode,
        }
        path = "logs/trade_journal.csv"
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    nexus = NEXUSOrchestrator(
        symbol       = "btcusdt",
        timeframe    = "1h",
        account_size = 1000,
        mode         = "paper",  # ← Always start with paper
        risk_pct     = 0.01,
        # cryptopanic_token = "your_free_token_here",  # optional
        # binance_api_key   = "your_key",              # Phase 4 only
        # binance_api_secret= "your_secret",           # Phase 4 only
    )
    asyncio.run(nexus.run())
