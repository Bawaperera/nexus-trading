"""
NEXUS — Hourly Runner with Prediction Validation
Runs inside GitHub Actions every hour (completely free).

Every hour it:
  1. Validates all past BUY/SELL signals — did price hit TP or SL?
  2. Fetches live BTC hourly data via yfinance
  3. Engineers 102 features, cleans inf/NaN
  4. Trains XGBoost, generates signal
  5. Logs signal to CSV
  6. Shows accuracy dashboard on GitHub Actions run page
  7. Sends Telegram alert on BUY/SELL
"""

import os, sys, csv, logging, warnings
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

SIGNALS_CSV    = "logs/signals_log.csv"
VALIDATION_CSV = "logs/validation_log.csv"


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════════

def validate_past_signals():
    """
    For every past BUY/SELL signal with a stop-loss and take-profit:
      - Fetch hourly price data from signal time to now
      - Check if price hit TP (WIN) or SL (LOSS) first
      - After 20 hours with no resolution, use current price direction

    Returns accuracy stats dict.
    """
    import pandas as pd
    import numpy as np
    import yfinance as yf

    os.makedirs("logs", exist_ok=True)

    if not os.path.exists(SIGNALS_CSV):
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "pending": 0, "new_resolved": 0, "all_rows": []}

    # Load past signals
    signals = pd.read_csv(SIGNALS_CSV)
    signals = signals[
        signals["action"].isin(["BUY", "SELL"]) &
        (signals["stop_loss"].astype(str) != "") &
        (signals["stop_loss"].notna())
    ].copy()

    if signals.empty:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "pending": 0, "new_resolved": 0, "all_rows": []}

    # Load already validated
    already_validated = set()
    existing_rows = []
    if os.path.exists(VALIDATION_CSV):
        vdf = pd.read_csv(VALIDATION_CSV)
        already_validated = set(vdf["signal_timestamp"].astype(str).tolist())
        existing_rows = vdf.to_dict("records")

    now = datetime.now(tz=timezone.utc)
    new_rows = []

    for _, row in signals.iterrows():
        ts_str = str(row["timestamp"])

        # Skip already validated
        if ts_str in already_validated:
            continue

        # Parse timestamp
        try:
            ts = pd.Timestamp(ts_str)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
        except Exception:
            continue

        # Need at least 2 hours before validating
        if (now - ts).total_seconds() < 7200:
            continue

        # Fetch hourly BTC data from signal time onward
        try:
            start_date = ts.strftime("%Y-%m-%d")
            raw = yf.download("BTC-USD", start=start_date, interval="1h",
                              progress=False, auto_adjust=True)
            if raw.empty:
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw.columns = [c.lower() for c in raw.columns]
            raw.index   = pd.to_datetime(raw.index, utc=True)
            raw         = raw[raw.index >= ts]

            if len(raw) < 2:
                continue

        except Exception as e:
            log.warning(f"Validation fetch failed for {ts_str}: {e}")
            continue

        entry  = float(row["price"])
        sl     = float(row["stop_loss"])
        tp     = float(row["take_profit"])
        action = row["action"]

        outcome          = "PENDING"
        resolution_time  = None
        resolution_price = None
        bars_held        = 0

        # Scan candle by candle
        for i, (idx, bar) in enumerate(raw.iterrows()):
            high = float(bar["high"])
            low  = float(bar["low"])

            if action == "BUY":
                if low <= sl and high >= tp:   # Both in same candle → SL wins (conservative)
                    outcome, resolution_price = "LOSS", sl
                elif low <= sl:
                    outcome, resolution_price = "LOSS", sl
                elif high >= tp:
                    outcome, resolution_price = "WIN", tp
            else:  # SELL
                if high >= sl and low <= tp:
                    outcome, resolution_price = "LOSS", sl
                elif high >= sl:
                    outcome, resolution_price = "LOSS", sl
                elif low <= tp:
                    outcome, resolution_price = "WIN", tp

            if outcome != "PENDING":
                resolution_time = idx
                bars_held = i + 1
                break

        # After 20 bars with no TP/SL hit → use directional accuracy
        if outcome == "PENDING" and len(raw) >= 20:
            current = float(raw["close"].iloc[-1])
            if action == "BUY":
                outcome = "WIN" if current > entry else "LOSS"
            else:
                outcome = "WIN" if current < entry else "LOSS"
            resolution_time  = raw.index[-1]
            resolution_price = current
            bars_held        = len(raw)

        if outcome != "PENDING":
            new_rows.append({
                "signal_timestamp": ts_str,
                "action":           action,
                "entry_price":      round(entry, 2),
                "stop_loss":        round(sl, 2),
                "take_profit":      round(tp, 2),
                "confidence":       row.get("confidence", ""),
                "sentiment":        row.get("sentiment", ""),
                "outcome":          outcome,
                "resolved_at":      str(resolution_time)[:19] if resolution_time else "",
                "resolution_price": round(float(resolution_price), 2) if resolution_price else "",
                "bars_held":        bars_held,
                "pct_from_entry":   round((float(resolution_price) - entry) / entry * 100, 3) if resolution_price else "",
            })
            log.info(f"Validated: {action} @ ${entry:,.0f} → {outcome} (signal: {ts_str[:16]})")

    # Save updated validation log
    if new_rows:
        all_rows = existing_rows + new_rows
        fieldnames = list(new_rows[0].keys())
        with open(VALIDATION_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        log.info(f"Validated {len(new_rows)} new signal(s)")

    # Calculate accuracy stats
    all_validated = existing_rows + new_rows
    wins   = sum(1 for r in all_validated if r["outcome"] == "WIN")
    losses = sum(1 for r in all_validated if r["outcome"] == "LOSS")
    total  = wins + losses
    pending_count = len(signals) - total

    return {
        "total":        total,
        "wins":         wins,
        "losses":       losses,
        "win_rate":     round(wins / total * 100, 1) if total > 0 else 0.0,
        "pending":      max(0, pending_count),
        "new_resolved": len(new_rows),
        "all_rows":     all_validated,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SIGNAL LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 55)
    log.info("  NEXUS Hourly Run — " + datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    log.info("=" * 55)

    os.makedirs("logs", exist_ok=True)

    # ── Step 0: Validate past predictions ────────────────────────────────────
    log.info("Validating past signals...")
    accuracy = validate_past_signals()
    log.info(
        f"Accuracy so far: {accuracy['wins']}W / {accuracy['losses']}L "
        f"({accuracy['win_rate']}% win rate) | "
        f"{accuracy['pending']} pending | "
        f"{accuracy['new_resolved']} newly resolved"
    )

    # ── Step 1: Fetch live BTC data via yfinance ──────────────────────────────
    import yfinance as yf
    import pandas as pd
    import numpy as np

    log.info("Fetching live BTC/USDT hourly data via yfinance...")
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
    log.info(f"Current BTC price: ${current_price:,.2f} | {len(df)} candles")

    # ── Step 2: Feature engineering ──────────────────────────────────────────
    from data.feature_engineer import FeatureEngineer
    fe      = FeatureEngineer(target_horizon=1, target_pct=0.003)
    feat_df = fe.build(df)
    X, y    = fe.get_targets(feat_df)

    # ── Step 2b: Clean inf/NaN ────────────────────────────────────────────────
    bad = np.isinf(X.values).any(axis=1) | X.isna().any(axis=1)
    if bad.sum() > 0:
        log.info(f"Cleaning {bad.sum()} inf/NaN rows...")
        X, y = X[~bad], y[~bad]
    log.info(f"Feature matrix: {X.shape[0]} x {X.shape[1]}")

    # ── Step 3: Train XGBoost ─────────────────────────────────────────────────
    from models.model_trainer import ModelTrainer
    trainer = ModelTrainer(n_splits=3, model_dir="models")
    model   = trainer.train_final(X, y)

    pred        = trainer.predict(model, X)
    model_proba = {"BUY": pred["BUY"], "SELL": pred["SELL"], "HOLD": pred["HOLD"]}
    log.info(f"Model: BUY {pred['BUY']:.0%} | SELL {pred['SELL']:.0%} | HOLD {pred['HOLD']:.0%}")

    # ── Step 4: Live sentiment ────────────────────────────────────────────────
    from data.news_collector import NewsCollector
    sentiment = NewsCollector().get_sentiment_report()
    log.info(f"Sentiment: {sentiment.composite_score:+.3f} ({sentiment.sentiment_label})")

    # ── Step 5: Signal ────────────────────────────────────────────────────────
    latest = feat_df.iloc[-1]
    atr    = float(latest.get("atr_14", current_price * 0.01))
    regime = int(latest.get("regime", 0))

    from signals.signal_engine import SignalEngine
    signal = SignalEngine(
        symbol=SYMBOL, model_weight=0.80, sentiment_weight=0.20,
        min_confidence=MIN_CONFIDENCE,
    ).generate(
        model_proba=model_proba, sentiment_score=sentiment.composite_score,
        entry_price=current_price, atr=atr, current_regime=regime,
        fear_greed=sentiment.fear_greed_value,
    )

    # ── Step 6: Risk check ────────────────────────────────────────────────────
    order = None
    if signal.action in ("BUY", "SELL"):
        from risk.risk_engine import RiskEngine
        order = RiskEngine(account_size=ACCOUNT_SIZE, risk_pct=RISK_PCT).calculate_order(
            symbol=SYMBOL,
            signal=1 if signal.action == "BUY" else -1,
            confidence=signal.confidence,
            entry_price=current_price,
            atr=atr,
        )

    # ── Step 7: Log to CSV ────────────────────────────────────────────────────
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
    write_head = not os.path.exists(SIGNALS_CSV)
    with open(SIGNALS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_head:
            w.writeheader()
        w.writerow(row)

    # ── Step 8: Console report ────────────────────────────────────────────────
    log.info("")
    log.info("=" * 55)
    log.info(f"  SIGNAL : {signal.action}  |  Confidence: {signal.confidence:.0%}")
    log.info(f"  Price  : ${current_price:,.2f}  |  ATR: ${atr:,.2f}")
    if order and order.get("approved"):
        log.info(f"  SL     : ${order['stop_loss']:,.2f}")
        log.info(f"  TP     : ${order['take_profit']:,.2f}")
        log.info(f"  Size   : ${order['position_size_usd']:,.2f}")
    log.info(f"  Sent.  : {sentiment.composite_score:+.3f} ({sentiment.sentiment_label})")
    log.info(f"  F&G    : {sentiment.fear_greed_value} ({sentiment.fear_greed_label})")
    log.info(f"  ACCURACY: {accuracy['wins']}W {accuracy['losses']}L {accuracy['win_rate']}% win rate")
    log.info("=" * 55)

    # ── Step 9: GitHub Actions step summary ──────────────────────────────────
    summary_path = os.getenv("GITHUB_STEP_SUMMARY", "")
    if summary_path:
        _write_summary(summary_path, signal, pred, order, sentiment,
                       current_price, atr, accuracy)

    # ── Step 10: Telegram alert ───────────────────────────────────────────────
    if signal.action in ("BUY", "SELL") and TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        _send_telegram(signal, order, sentiment, current_price, accuracy)


def _write_summary(path, signal, pred, order, sentiment, price, atr, accuracy):
    """Write a visual dashboard to the GitHub Actions run summary page."""
    now    = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    s_icon = "🟢" if signal.action == "BUY" else ("🔴" if signal.action == "SELL" else "⏸️")
    sl_str = f"${order['stop_loss']:,.2f}"        if order and order.get("approved") else "—"
    tp_str = f"${order['take_profit']:,.2f}"      if order and order.get("approved") else "—"
    sz_str = f"${order['position_size_usd']:,.2f}" if order and order.get("approved") else "—"

    # Accuracy colour
    wr = accuracy["win_rate"]
    acc_icon = "🟢" if wr >= 55 else ("🟡" if wr >= 45 else ("🔴" if accuracy["total"] > 0 else "⏳"))

    lines = [
        f"## {s_icon} NEXUS Signal — {now}",
        "",
        "### Current Signal",
        "| | |",
        "|---|---|",
        f"| Signal | **{signal.action}** |",
        f"| Confidence | {signal.confidence:.0%} |",
        f"| BTC Price | ${price:,.2f} |",
        f"| Model BUY | {pred['BUY']:.0%} |",
        f"| Model SELL | {pred['SELL']:.0%} |",
        f"| Stop Loss | {sl_str} |",
        f"| Take Profit | {tp_str} |",
        f"| Position Size | {sz_str} |",
        f"| Sentiment | {sentiment.composite_score:+.3f} ({sentiment.sentiment_label}) |",
        f"| Fear & Greed | {sentiment.fear_greed_value} ({sentiment.fear_greed_label}) |",
        f"| ATR | ${atr:,.2f} |",
        "",
        f"> {signal.reasoning[0]}",
        "",
        "---",
        "",
        f"### {acc_icon} Prediction Accuracy (live validation)",
        "| | |",
        "|---|---|",
        f"| Total validated signals | {accuracy['total']} |",
        f"| Wins (TP hit) | {accuracy['wins']} |",
        f"| Losses (SL hit) | {accuracy['losses']} |",
        f"| Win rate | **{accuracy['win_rate']}%** |",
        f"| Pending (not yet resolved) | {accuracy['pending']} |",
        "",
    ]

    # Go-live readiness indicator
    if accuracy["total"] >= 10:
        if wr >= 55:
            lines.append("> 🟢 **Win rate above 55% — on track for live trading after 30 days**")
        elif wr >= 45:
            lines.append("> 🟡 **Win rate 45-55% — marginal. Keep monitoring.**")
        else:
            lines.append("> 🔴 **Win rate below 45% — model needs improvement before live trading**")
    else:
        needed = 10 - accuracy["total"]
        lines.append(f"> ⏳ **Need {needed} more validated signals before accuracy is meaningful**")

    lines += ["", "---", ""]

    # Recent validated signals table
    all_rows = accuracy.get("all_rows", [])
    if all_rows:
        recent = all_rows[-10:]  # last 10
        lines += [
            "### Recent Validated Signals",
            "| Time | Action | Entry | Outcome | Bars |",
            "|---|---|---|---|---|",
        ]
        for r in reversed(recent):
            o_icon = "✅" if r["outcome"] == "WIN" else "❌"
            ts_short = str(r["signal_timestamp"])[:16]
            lines.append(
                f"| {ts_short} | {r['action']} | ${float(r['entry_price']):,.0f} "
                f"| {o_icon} {r['outcome']} | {r['bars_held']} |"
            )

    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")


def _send_telegram(signal, order, sentiment, price, accuracy):
    import requests
    sl   = f"${order['stop_loss']:,.2f}"        if order and order.get("approved") else "n/a"
    tp   = f"${order['take_profit']:,.2f}"      if order and order.get("approved") else "n/a"
    size = f"${order['position_size_usd']:,.2f}" if order and order.get("approved") else "n/a"
    acc  = f"{accuracy['win_rate']}% ({accuracy['wins']}W/{accuracy['losses']}L)" if accuracy["total"] > 0 else "building..."

    msg = (
        f"*NEXUS {signal.action} SIGNAL*\n\n"
        f"Price: ${price:,.2f}\n"
        f"Confidence: {signal.confidence:.0%}\n"
        f"Stop Loss: {sl}\n"
        f"Take Profit: {tp}\n"
        f"Size: {size}\n\n"
        f"Sentiment: {sentiment.composite_score:+.3f} ({sentiment.sentiment_label})\n"
        f"Fear/Greed: {sentiment.fear_greed_value} ({sentiment.fear_greed_label})\n\n"
        f"Model accuracy so far: {acc}\n\n"
        f"{signal.reasoning[0]}"
    )
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
        log.info("Telegram sent")
    except Exception as e:
        log.warning(f"Telegram failed: {e}")


if __name__ == "__main__":
    main()
