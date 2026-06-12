"""
NEXUS — Hourly Runner
Designed to run inside GitHub Actions every hour (completely free).

What it does in ~60 seconds:
  1. Fetches 500 hours of live BTC/USDT data (Binance public API — no key needed)
  2. Engineers 102 features
  3. Trains XGBoost on all historical data (~5 seconds)
  4. Fetches live news sentiment + Fear & Greed Index
  5. Generates signal (BUY / SELL / HOLD)
  6. Appends result to logs/signals_log.csv
  7. Prints full report
  8. Sends Telegram alert if BUY or SELL (optional)
"""

import sys, os, logging, warnings, json, csv
from datetime import datetime, timezone

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ─── Config from environment ──────────────────────────────────────────────────
SYMBOL           = os.getenv("TRADING_SYMBOL",    "BTC/USDT")
RISK_PCT         = float(os.getenv("RISK_PCT",    "0.01"))
ACCOUNT_SIZE     = float(os.getenv("ACCOUNT_SIZE","1000"))
MIN_CONFIDENCE   = float(os.getenv("MIN_CONFIDENCE", "0.58"))
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID",   "")


def fetch_live_hourly_data(symbol: str = "BTC/USDT", limit: int = 500):
    """Fetch recent hourly candles with a fallback when Binance is unavailable."""
    import ccxt
    import pandas as pd

    def _fetch_from_exchange(exchange_id: str, market_symbol: str):
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({"enableRateLimit": True})
        ohlcv = exchange.fetch_ohlcv(market_symbol, "1h", limit=limit)
        out = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        out["timestamp"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True)
        out = out.set_index("timestamp").astype(float)
        return out

    try:
        log.info("Fetching live BTC/USDT hourly data from Binance...")
        df = _fetch_from_exchange("binance", symbol)
        source = "Binance"
    except Exception as e:
        import yfinance as yf

        usd_symbol = symbol.replace("USDT", "USD")
        fallback_exchanges = [("coinbase", "Coinbase"), ("kraken", "Kraken")]
        df = None
        source = None

        log.warning(f"Binance data fetch failed ({e}). Trying fallback exchanges...")
        for exchange_id, exchange_name in fallback_exchanges:
            try:
                df = _fetch_from_exchange(exchange_id, usd_symbol)
                source = exchange_name
                break
            except Exception as exchange_error:
                log.warning(f"{exchange_name} data fetch failed ({exchange_error}).")

        if df is None:
            log.warning("All exchange fallbacks failed. Falling back to yfinance...")
            ticker = symbol.replace("/USDT", "-USD").replace("/", "-")
            yf_df = yf.download(ticker, period="60d", interval="1h", progress=False)
            if yf_df.empty:
                raise RuntimeError("Fallback data fetch failed: yfinance returned no rows")
            if isinstance(yf_df.columns, pd.MultiIndex):
                yf_df.columns = yf_df.columns.get_level_values(0)
            yf_df.columns = [c.lower() for c in yf_df.columns]
            yf_df.index = pd.to_datetime(yf_df.index, utc=True)
            df = yf_df[["open", "high", "low", "close", "volume"]].astype(float).tail(limit)
            source = "yfinance"

    df["symbol"] = symbol
    df["timeframe"] = "1h"
    log.info(f"Loaded {len(df)} candles from {source}")
    return df


def main():
    log.info("=" * 55)
    log.info("  NEXUS Hourly Run — " + datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    log.info("=" * 55)

    # ── 1. Fetch live hourly BTC data ─────────────────────────────────────────
    df = fetch_live_hourly_data(symbol=SYMBOL, limit=500)

    current_price = float(df["close"].iloc[-1])
    log.info(f"Current BTC price: ${current_price:,.2f} | {len(df)} candles loaded")

    # ── 2. Feature engineering ────────────────────────────────────────────────
    log.info("Engineering features...")
    from data.feature_engineer import FeatureEngineer
    fe      = FeatureEngineer(target_horizon=1, target_pct=0.003)
    feat_df = fe.build(df)
    X, y    = fe.get_targets(feat_df)
    log.info(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")

    # ── 2b. Clean inf/NaN values ──────────────────────────────────────────────
    # Some features (VWAP, AD-line pct_change) can produce inf when volume = 0
    # on certain candles. XGBoost hard-crashes on inf values — must clean first.
    import numpy as np
    inf_rows = np.isinf(X.values).any(axis=1)
    nan_rows = X.isna().any(axis=1)
    bad_rows = inf_rows | nan_rows
    if bad_rows.sum() > 0:
        log.info(f"Cleaning {bad_rows.sum()} rows with inf/NaN (VWAP/volume edge cases)...")
        X = X[~bad_rows]
        y = y[~bad_rows]
    log.info(f"Clean feature matrix: {X.shape[0]} rows × {X.shape[1]} features")

    # ── 3. Train XGBoost ──────────────────────────────────────────────────────
    log.info("Training XGBoost model...")
    from models.model_trainer import ModelTrainer
    trainer = ModelTrainer(n_splits=3, model_dir="models")   # 3 folds = faster
    model   = trainer.train_final(X, y)
    log.info("Model trained ✅")

    # ── 4. Get model prediction ───────────────────────────────────────────────
    pred        = trainer.predict(model, X)
    model_proba = {"BUY": pred["BUY"], "SELL": pred["SELL"], "HOLD": pred["HOLD"]}
    log.info(f"Model: BUY {pred['BUY']:.0%} | SELL {pred['SELL']:.0%} | HOLD {pred['HOLD']:.0%}")

    # ── 5. Live sentiment ─────────────────────────────────────────────────────
    log.info("Fetching live news sentiment...")
    from data.news_collector import NewsCollector
    collector      = NewsCollector()
    sentiment      = collector.get_sentiment_report()
    log.info(f"Sentiment: {sentiment.composite_score:+.3f} ({sentiment.sentiment_label})")
    log.info(f"Fear & Greed: {sentiment.fear_greed_value} — {sentiment.fear_greed_label}")

    # ── 6. Signal generation ──────────────────────────────────────────────────
    latest = feat_df.iloc[-1]
    atr     = float(latest.get("atr_14", current_price * 0.01))
    regime  = int(latest.get("regime", 0))

    from signals.signal_engine import SignalEngine
    engine = SignalEngine(
        symbol             = SYMBOL,
        model_weight       = 0.80,
        sentiment_weight   = 0.20,
        min_confidence     = MIN_CONFIDENCE,
    )
    signal = engine.generate(
        model_proba     = model_proba,
        sentiment_score = sentiment.composite_score,
        entry_price     = current_price,
        atr             = atr,
        current_regime  = regime,
        fear_greed      = sentiment.fear_greed_value,
    )

    # ── 7. Risk check ─────────────────────────────────────────────────────────
    order = None
    if signal.action in ("BUY", "SELL"):
        from risk.risk_engine import RiskEngine
        risk  = RiskEngine(account_size=ACCOUNT_SIZE, risk_pct=RISK_PCT)
        order = risk.calculate_order(
            symbol      = SYMBOL,
            signal      = 1 if signal.action == "BUY" else -1,
            confidence  = signal.confidence,
            entry_price = current_price,
            atr         = atr,
        )

    # ── 8. Log result ─────────────────────────────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    row = {
        "timestamp":     datetime.now(tz=timezone.utc).isoformat(),
        "price":         round(current_price, 2),
        "action":        signal.action,
        "confidence":    round(signal.confidence * 100, 1),
        "model_buy":     round(pred["BUY"] * 100, 1),
        "model_sell":    round(pred["SELL"] * 100, 1),
        "sentiment":     round(sentiment.composite_score, 4),
        "fear_greed":    sentiment.fear_greed_value,
        "fg_label":      sentiment.fear_greed_label,
        "regime":        regime,
        "atr":           round(atr, 2),
        "stop_loss":     round(order["stop_loss"], 2) if order and order.get("approved") else "",
        "take_profit":   round(order["take_profit"], 2) if order and order.get("approved") else "",
        "position_usd":  round(order["position_size_usd"], 2) if order and order.get("approved") else "",
        "reasoning":     " | ".join(signal.reasoning[:3]),
    }

    log_path   = "logs/signals_log.csv"
    write_head = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_head:
            writer.writeheader()
        writer.writerow(row)

    # ── 9. Print report ───────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 55)
    log.info(f"  SIGNAL: {signal.action}  |  Confidence: {signal.confidence:.0%}")
    log.info(f"  Price:  ${current_price:,.2f}  |  ATR: ${atr:,.2f}")
    if order and order.get("approved"):
        log.info(f"  Stop Loss:   ${order['stop_loss']:,.2f}")
        log.info(f"  Take Profit: ${order['take_profit']:,.2f}")
        log.info(f"  Position:    ${order['position_size_usd']:,.2f}")
    log.info(f"  Sentiment:  {sentiment.composite_score:+.3f} ({sentiment.sentiment_label})")
    log.info(f"  Reasoning:  {signal.reasoning[0]}")
    log.info("=" * 55)

    # ── 10. Telegram alert (if BUY or SELL) ───────────────────────────────────
    if signal.action in ("BUY", "SELL") and TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        send_telegram(signal, order, sentiment, current_price)


def send_telegram(signal, order, sentiment, price):
    """Send a Telegram notification when there's a trade signal."""
    import requests

    icon = "🟢" if signal.action == "BUY" else "🔴"
    sl   = f"${order['stop_loss']:,.2f}"   if order and order.get("approved") else "n/a"
    tp   = f"${order['take_profit']:,.2f}" if order and order.get("approved") else "n/a"
    size = f"${order['position_size_usd']:,.2f}" if order and order.get("approved") else "n/a"

    msg = (
        f"{icon} *NEXUS {signal.action} SIGNAL*\n\n"
        f"💰 Price:      `${price:,.2f}`\n"
        f"🎯 Confidence: `{signal.confidence:.0%}`\n"
        f"🛑 Stop Loss:  `{sl}`\n"
        f"✅ Take Profit:`{tp}`\n"
        f"📊 Size:       `{size}`\n\n"
        f"📰 Sentiment: `{sentiment.composite_score:+.3f}` ({sentiment.sentiment_label})\n"
        f"😨 Fear/Greed: `{sentiment.fear_greed_value}` ({sentiment.fear_greed_label})\n\n"
        f"_{signal.reasoning[0]}_"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       msg,
            "parse_mode": "Markdown",
        }, timeout=10)
        log.info("Telegram alert sent ✅")
    except Exception as e:
        log.warning(f"Telegram failed: {e}")


if __name__ == "__main__":
    main()