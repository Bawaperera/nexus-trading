"""
NEXUS — Hourly Runner
Runs inside GitHub Actions every hour (completely free).

What it does:
  1. Fetches 2 years of live BTC hourly data via yfinance (no API key, no geo-block)
  2. Engineers 102 features
  3. Cleans inf/NaN values (VWAP edge case fix)
  4. Trains XGBoost
  5. Fetches live news sentiment + Fear & Greed
  6. Generates BUY / SELL / HOLD signal
  7. Writes signal to logs/signals_log.csv
  8. Writes visual summary to GitHub Actions run page
  9. Sends Telegram alert (optional)
"""

import sys, os, logging, warnings, csv
from datetime import datetime, timezone

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SYMBOL           = os.getenv("TRADING_SYMBOL",    "BTC/USDT")
RISK_PCT         = float(os.getenv("RISK_PCT",    "0.01"))
ACCOUNT_SIZE     = float(os.getenv("ACCOUNT_SIZE","1000"))
MIN_CONFIDENCE   = float(os.getenv("MIN_CONFIDENCE", "0.58"))
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID",   "")


def main():
    log.info("=" * 55)
    log.info("  NEXUS Hourly Run — " + datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    log.info("=" * 55)

    # ── 1. Fetch live BTC hourly data via yfinance ────────────────────────────
    # Binance blocks GitHub Actions US servers (HTTP 451).
    # yfinance uses Yahoo Finance — no geo-blocks, no API key needed.
    log.info("Fetching live BTC/USDT hourly data via yfinance...")
    import yfinance as yf
    import pandas as pd
    import numpy as np

    raw = yf.download("BTC-USD", period="2y", interval="1h", progress=False, auto_adjust=True)

    if raw.empty:
        raise RuntimeError("yfinance returned no data")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.columns = [c.lower() for c in raw.columns]
    raw.index   = pd.to_datetime(raw.index, utc=True)

    df = raw[["open", "high", "low", "close", "volume"]].dropna().copy()
    df["symbol"]    = "BTC/USDT"
    df["timeframe"] = "1h"

    current_price = float(df["close"].iloc[-1])
    log.info(f"Current BTC price: ${current_price:,.2f} | {len(df)} candles loaded")

    # ── 2. Feature engineering ────────────────────────────────────────────────
    log.info("Engineering features...")
    from data.feature_engineer import FeatureEngineer
    fe      = FeatureEngineer(target_horizon=1, target_pct=0.003)
    feat_df = fe.build(df)
    X, y    = fe.get_targets(feat_df)
    log.info(f"Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")

    # ── 2b. Clean inf/NaN (VWAP produces inf when volume=0 on some candles) ───
    inf_rows = np.isinf(X.values).any(axis=1)
    nan_rows = X.isna().any(axis=1)
    bad_rows = inf_rows | nan_rows
    if bad_rows.sum() > 0:
        log.info(f"Cleaning {bad_rows.sum()} rows with inf/NaN...")
        X = X[~bad_rows]
        y = y[~bad_rows]
    log.info(f"Clean feature matrix: {X.shape[0]} rows x {X.shape[1]} features")

    # ── 3. Train XGBoost ──────────────────────────────────────────────────────
    log.info("Training XGBoost model...")
    from models.model_trainer import ModelTrainer
    trainer = ModelTrainer(n_splits=3, model_dir="models")
    model   = trainer.train_final(X, y)
    log.info("Model trained")

    # ── 4. Model prediction ───────────────────────────────────────────────────
    pred        = trainer.predict(model, X)
    model_proba = {"BUY": pred["BUY"], "SELL": pred["SELL"], "HOLD": pred["HOLD"]}
    log.info(f"Model: BUY {pred['BUY']:.0%} | SELL {pred['SELL']:.0%} | HOLD {pred['HOLD']:.0%}")

    # ── 5. Live sentiment ─────────────────────────────────────────────────────
    log.info("Fetching live news sentiment...")
    from data.news_collector import NewsCollector
    sentiment = NewsCollector().get_sentiment_report()
    log.info(f"Sentiment: {sentiment.composite_score:+.3f} ({sentiment.sentiment_label})")
    log.info(f"Fear & Greed: {sentiment.fear_greed_value} -- {sentiment.fear_greed_label}")

    # ── 6. Signal generation ──────────────────────────────────────────────────
    latest = feat_df.iloc[-1]
    atr    = float(latest.get("atr_14", current_price * 0.01))
    regime = int(latest.get("regime", 0))

    from signals.signal_engine import SignalEngine
    signal = SignalEngine(
        symbol           = SYMBOL,
        model_weight     = 0.80,
        sentiment_weight = 0.20,
        min_confidence   = MIN_CONFIDENCE,
    ).generate(
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
        order = RiskEngine(account_size=ACCOUNT_SIZE, risk_pct=RISK_PCT).calculate_order(
            symbol      = SYMBOL,
            signal      = 1 if signal.action == "BUY" else -1,
            confidence  = signal.confidence,
            entry_price = current_price,
            atr         = atr,
        )

    # ── 8. Write CSV log ──────────────────────────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    row = {
        "timestamp":   datetime.now(tz=timezone.utc).isoformat(),
        "price":       round(current_price, 2),
        "action":      signal.action,
        "confidence":  round(signal.confidence * 100, 1),
        "model_buy":   round(pred["BUY"] * 100, 1),
        "model_sell":  round(pred["SELL"] * 100, 1),
        "sentiment":   round(sentiment.composite_score, 4),
        "fear_greed":  sentiment.fear_greed_value,
        "fg_label":    sentiment.fear_greed_label,
        "regime":      regime,
        "atr":         round(atr, 2),
        "stop_loss":   round(order["stop_loss"], 2) if order and order.get("approved") else "",
        "take_profit": round(order["take_profit"], 2) if order and order.get("approved") else "",
        "pos_usd":     round(order["position_size_usd"], 2) if order and order.get("approved") else "",
        "reasoning":   " | ".join(signal.reasoning[:2]),
    }

    log_path   = "logs/signals_log.csv"
    write_head = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_head:
            writer.writeheader()
        writer.writerow(row)

    # ── 9. Print summary ──────────────────────────────────────────────────────
    icon = "BUY" if signal.action == "BUY" else ("SELL" if signal.action == "SELL" else "HOLD")
    log.info("")
    log.info("=" * 55)
    log.info(f"  SIGNAL : {signal.action}")
    log.info(f"  Conf   : {signal.confidence:.0%}")
    log.info(f"  Price  : ${current_price:,.2f}  |  ATR: ${atr:,.2f}")
    if order and order.get("approved"):
        log.info(f"  SL     : ${order['stop_loss']:,.2f}")
        log.info(f"  TP     : ${order['take_profit']:,.2f}")
        log.info(f"  Size   : ${order['position_size_usd']:,.2f}")
    log.info(f"  Sent.  : {sentiment.composite_score:+.3f} ({sentiment.sentiment_label})")
    log.info(f"  F&G    : {sentiment.fear_greed_value} ({sentiment.fear_greed_label})")
    log.info(f"  Reason : {signal.reasoning[0]}")
    log.info("=" * 55)

    # ── 10. GitHub Actions Step Summary (visible on the Actions run page) ─────
    # This writes a visual card directly to the workflow run summary tab.
    # Go to: GitHub repo → Actions → click the run → see "NEXUS Signal" card.
    summary_path = os.getenv("GITHUB_STEP_SUMMARY", "")
    if summary_path:
        s_icon = "🟢" if signal.action == "BUY" else ("🔴" if signal.action == "SELL" else "⏸️")
        sl_str = f"${order['stop_loss']:,.2f}"   if order and order.get("approved") else "—"
        tp_str = f"${order['take_profit']:,.2f}" if order and order.get("approved") else "—"
        sz_str = f"${order['position_size_usd']:,.2f}" if order and order.get("approved") else "—"
        with open(summary_path, "a") as f:
            f.write(f"## {s_icon} NEXUS — {signal.action}\n\n")
            f.write(f"**{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}**\n\n")
            f.write(f"| | |\n|---|---|\n")
            f.write(f"| Signal | **{signal.action}** |\n")
            f.write(f"| Confidence | {signal.confidence:.0%} |\n")
            f.write(f"| BTC Price | ${current_price:,.2f} |\n")
            f.write(f"| Model BUY | {pred['BUY']:.0%} |\n")
            f.write(f"| Model SELL | {pred['SELL']:.0%} |\n")
            f.write(f"| Stop Loss | {sl_str} |\n")
            f.write(f"| Take Profit | {tp_str} |\n")
            f.write(f"| Position Size | {sz_str} |\n")
            f.write(f"| Sentiment | {sentiment.composite_score:+.3f} ({sentiment.sentiment_label}) |\n")
            f.write(f"| Fear & Greed | {sentiment.fear_greed_value} ({sentiment.fear_greed_label}) |\n")
            f.write(f"| ATR | ${atr:,.2f} |\n\n")
            f.write(f"> {signal.reasoning[0]}\n")

    # ── 11. Telegram alert ────────────────────────────────────────────────────
    if signal.action in ("BUY", "SELL") and TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        send_telegram(signal, order, sentiment, current_price)


def send_telegram(signal, order, sentiment, price):
    import requests
    sl   = f"${order['stop_loss']:,.2f}"   if order and order.get("approved") else "n/a"
    tp   = f"${order['take_profit']:,.2f}" if order and order.get("approved") else "n/a"
    size = f"${order['position_size_usd']:,.2f}" if order and order.get("approved") else "n/a"
    msg  = (
        f"*NEXUS {signal.action} SIGNAL*\n\n"
        f"Price: ${price:,.2f}\n"
        f"Confidence: {signal.confidence:.0%}\n"
        f"Stop Loss: {sl}\n"
        f"Take Profit: {tp}\n"
        f"Size: {size}\n\n"
        f"Sentiment: {sentiment.composite_score:+.3f} ({sentiment.sentiment_label})\n"
        f"Fear/Greed: {sentiment.fear_greed_value} ({sentiment.fear_greed_label})\n\n"
        f"{signal.reasoning[0]}"
    )
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
        log.info("Telegram alert sent")
    except Exception as e:
        log.warning(f"Telegram failed: {e}")


if __name__ == "__main__":
    main()
