"""
NEXUS v2 — Professional Trading Signal System
===============================================
Full pipeline every hour:

  1. Load trade memory → validate open positions
  2. Fetch BTC 1H data (2 years) + 15M data (60 days)
  3. Detect chart patterns on Daily, 4H, 1H
  4. Run MTF analysis: Weekly → Daily → 4H → 15M entry
  5. Fetch news sentiment (8 RSS feeds: crypto + macro)
  5b. Collect on-chain intelligence (derivatives + Reddit + calendar + CoinGecko)
  6. Score each pattern with full confluence engine
  7. If score >= 60 AND 3+ confluences: Telegram + paper trade
  8. Save active_patterns.json + market_intel.json
  9. Write GitHub Actions step summary

Trading framework (Day Trader style):
  Daily  → confirm the overall trend direction
  4H     → find the pattern setup
  15M    → confirm the entry moment (engulfing candle + volume)
  → ENTER when all three agree
"""

import os, sys, csv, logging, warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))

SYMBOL           = "BTC/USDT"
ACCOUNT_SIZE     = float(os.getenv("ACCOUNT_SIZE",     "1000"))
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN",     "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID",       "")
BINANCE_KEY      = os.getenv("BINANCE_API_KEY",         "")
BINANCE_SECRET   = os.getenv("BINANCE_API_SECRET",      "")
PAPER_MODE       = os.getenv("TRADING_MODE", "paper").lower() == "paper"

SIGNAL_LOG  = "logs/signals_v2.csv"
MEMORY_PATH = "logs/trade_memory.csv"


def main():
    log.info("=" * 60)
    log.info(f"  NEXUS v2 — {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    log.info(f"  Mode: {'PAPER' if PAPER_MODE else 'LIVE'}")
    log.info("=" * 60)

    os.makedirs("logs", exist_ok=True)

    # ── Step 0: Load trade memory ─────────────────────────────────────────────
    from execution.trade_executor import TradeExecutor
    executor = TradeExecutor(
        api_key      = BINANCE_KEY,
        api_secret   = BINANCE_SECRET,
        account_size = ACCOUNT_SIZE,
        paper_mode   = PAPER_MODE,
    )
    trade_memory = executor.load_memory()
    open_trades  = [t for t in trade_memory if t.get("outcome") == "OPEN"]
    log.info(f"Trade memory: {len(trade_memory)} total | {len(open_trades)} open")

    # ── Step 1: Validate open positions ──────────────────────────────────────
    _validate_open_trades(executor, open_trades)

    # ── Step 2: Fetch price data ──────────────────────────────────────────────
    import yfinance as yf
    import pandas as pd
    import numpy as np

    log.info("Fetching BTC/USDT 1H data (2 years)...")
    raw = yf.download("BTC-USD", period="2y", interval="1h",
                      progress=False, auto_adjust=True)
    if raw.empty:
        raise RuntimeError("yfinance returned no 1H data")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]
    raw.index   = pd.to_datetime(raw.index, utc=True)
    df = raw[["open", "high", "low", "close", "volume"]].dropna().copy()
    df["symbol"]    = "BTC/USDT"
    df["timeframe"] = "1h"

    current_price = float(df["close"].iloc[-1])
    log.info(f"BTC price: ${current_price:,.2f} | {len(df)} 1H candles loaded")

    # ── Fetch 15M data ────────────────────────────────────────────────────────
    df_15m = None
    try:
        log.info("Fetching BTC/USDT 15M data (60 days)...")
        raw_15m = yf.download("BTC-USD", period="60d", interval="15m",
                              progress=False, auto_adjust=True)
        if not raw_15m.empty:
            if isinstance(raw_15m.columns, pd.MultiIndex):
                raw_15m.columns = raw_15m.columns.get_level_values(0)
            raw_15m.columns = [c.lower() for c in raw_15m.columns]
            raw_15m.index   = pd.to_datetime(raw_15m.index, utc=True)
            df_15m = raw_15m[["open", "high", "low", "close", "volume"]].dropna().copy()
            log.info(f"15M data: {len(df_15m)} candles loaded")
        else:
            log.warning("15M data empty — using 1H only")
    except Exception as e:
        log.warning(f"15M fetch failed: {e} — continuing without 15M")

    # ── Step 3: Pattern detection ─────────────────────────────────────────────
    log.info("Scanning for chart patterns (Daily, 4H, 1H)...")
    from data.pattern_engine import PatternEngine
    patterns = PatternEngine().scan_all(df)

    if not patterns:
        log.info("No patterns detected this run")
        _write_summary_no_patterns(current_price, trade_memory)
        return

    log.info(f"Patterns found: {len(patterns)}")
    for p in patterns:
        log.info(f"  → {p.summary()}")

    # ── Step 4: MTF analysis ──────────────────────────────────────────────────
    log.info("Running MTF analysis (Daily → 4H → 15M entry)...")
    mtf_features = {}
    try:
        from data.multi_timeframe import MultiTimeframeAnalyzer
        mtf_features = MultiTimeframeAnalyzer().analyze(df, df_15m)
        log.info(
            f"MTF: Daily={mtf_features.get('mtf_daily_trend')} | "
            f"4H={mtf_features.get('mtf_4h_trend')} | "
            f"Bull {mtf_features.get('mtf_bull_count',0)}/4 | "
            f"15M trigger: {mtf_features.get('mtf_15m_trigger',False)}"
        )
        for r in mtf_features.get("mtf_15m_reasons", []):
            log.info(f"  15M: {r}")
    except Exception as e:
        log.warning(f"MTF failed: {e}")

    # ── Step 5: News sentiment (crypto + macro RSS feeds) ─────────────────────
    log.info("Fetching news sentiment (8 RSS feeds: crypto + macro)...")
    sentiment_score = 0.0
    fg_value        = 50
    fg_label        = "Neutral"
    try:
        from data.news_collector import NewsCollector
        sentiment       = NewsCollector().get_sentiment_report()
        sentiment_score = sentiment.composite_score
        fg_value        = sentiment.fear_greed_value
        fg_label        = sentiment.fear_greed_label
        log.info(
            f"Sentiment: {sentiment_score:+.3f} | F&G: {fg_value} ({fg_label}) | "
            f"Crypto news: {sentiment.news_score:+.3f} | "
            f"Macro news: {sentiment.macro_score:+.3f}"
        )
    except Exception as e:
        log.warning(f"Sentiment failed: {e}")

    # ── Step 5b: On-chain market intelligence ─────────────────────────────────
    log.info("Collecting on-chain intelligence (derivatives + Reddit + calendar + CoinGecko)...")
    onchain_adj  = 0
    intel_report = {}
    try:
        from data.onchain_collector import OnChainCollector
        collector    = OnChainCollector()
        intel_report = collector.collect()
        collector.save(intel_report)
        onchain_adj  = intel_report.get("onchain_score_adj", 0)

        deriv = intel_report.get("derivatives", {})
        log.info(
            f"On-chain: Funding {deriv.get('funding_rate',0)*100:+.4f}% | "
            f"L/S {deriv.get('long_short_ratio',1):.2f} | "
            f"Score adj: {onchain_adj:+d} | "
            f"Intel: {intel_report.get('signal_label','?')}"
        )
        for w in intel_report.get("warnings", []):
            log.warning(w)
    except Exception as e:
        log.warning(f"On-chain intelligence failed: {e}")

    # ── Step 6: Score each pattern ────────────────────────────────────────────
    log.info("Scoring confluences...")
    from signals.confluence_engine import ConfluenceEngine
    engine  = ConfluenceEngine()
    results = []

    for pattern in patterns:
        result = engine.score(
            pattern         = pattern,
            mtf_features    = mtf_features,
            df_1h           = df,
            sentiment_score = sentiment_score,
            fg_value        = fg_value,
            trade_memory    = trade_memory,
            onchain_adj     = onchain_adj,
        )
        results.append(result)
        log.info(
            f"  {pattern.name} [{pattern.timeframe}]: "
            f"score={result['score']} | "
            f"{'SIGNAL' if result['tradeable_signal'] else 'skip'} | "
            f"{'AUTO-TRADE' if result['tradeable_autotrade'] else ''}"
        )

    results.sort(key=lambda r: r["score"], reverse=True)
    top_results = results[:3]

    # ── Step 7: Execute and alert ─────────────────────────────────────────────
    executed_trades = []

    for result in top_results:
        if not result["tradeable_signal"]:
            continue

        exec_result = executor.execute(result)
        _send_telegram(result, exec_result, sentiment_score, fg_value, fg_label, current_price, mtf_features, intel_report)
        _log_signal(result, exec_result, sentiment_score, fg_value)

        if exec_result.get("executed"):
            executed_trades.append(exec_result)

    # ── Step 8: Save active patterns + market intel ───────────────────────────
    _save_active_patterns(results, mtf_features, sentiment_score, fg_value, current_price)

    # ── Step 9: GitHub Actions summary ───────────────────────────────────────
    summary_path = os.getenv("GITHUB_STEP_SUMMARY", "")
    if summary_path:
        _write_summary(
            summary_path, top_results, current_price,
            mtf_features, sentiment_score, fg_value, fg_label,
            trade_memory, executed_trades, intel_report,
        )

    signals_fired = sum(1 for r in results if r["tradeable_signal"])
    log.info(f"Run complete | Signals: {signals_fired} | Auto-trades: {len(executed_trades)}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _validate_open_trades(executor, open_trades: list):
    if not open_trades:
        return
    try:
        import yfinance as yf
        import pandas as pd
        recent = yf.download("BTC-USD", period="5d", interval="1h",
                             progress=False, auto_adjust=True)
        if recent.empty:
            return
        if isinstance(recent.columns, pd.MultiIndex):
            recent.columns = recent.columns.get_level_values(0)
        recent.columns = [c.lower() for c in recent.columns]
        recent.index   = pd.to_datetime(recent.index, utc=True)
        for trade in open_trades:
            try:
                opened_at = pd.Timestamp(trade["timestamp"])
                if opened_at.tzinfo is None:
                    opened_at = opened_at.tz_localize("UTC")
                since = recent[recent.index >= opened_at]
                if len(since) < 1:
                    continue
                sl   = float(trade["stop_loss"])
                tp   = float(trade["take_profit"])
                dir_ = trade["direction"]
                for _, bar in since.iterrows():
                    h, l = float(bar["high"]), float(bar["low"])
                    if dir_ == "bullish":
                        if l <= sl:
                            executor.update_outcome(trade["signal_id"], "LOSS", sl); break
                        if h >= tp:
                            executor.update_outcome(trade["signal_id"], "WIN",  tp); break
                    else:
                        if h >= sl:
                            executor.update_outcome(trade["signal_id"], "LOSS", sl); break
                        if l <= tp:
                            executor.update_outcome(trade["signal_id"], "WIN",  tp); break
            except Exception as e:
                log.debug(f"Validation failed for {trade.get('signal_id')}: {e}")
    except Exception as e:
        log.warning(f"Trade validation failed: {e}")


def _save_active_patterns(results, mtf, sent, fg, price):
    """
    Save all tradeable patterns to active_patterns.json.
    Includes component_scores and reasons so the dashboard
    can show the full score breakdown for each signal.
    """
    import json
    patterns_to_save = []
    for r in results:
        if r.get("tradeable_signal"):
            p = r["pattern"]
            patterns_to_save.append({
                "name":               p.name,
                "direction":          p.direction,
                "timeframe":          p.timeframe,
                "status":             p.status,
                "entry":              round(float(p.entry), 2),
                "stop_loss":          round(float(p.stop_loss), 2),
                "target":             round(float(p.target), 2),
                "risk_reward":        float(p.risk_reward),
                "confidence":         round(float(p.confidence), 2),
                "score":              r["score"],
                "n_confluences":      r["n_confluences"],
                "tradeable_signal":   r["tradeable_signal"],
                "tradeable_autotrade":r["tradeable_autotrade"],
                "already_traded":     False,
                # Dashboard needs these for score breakdown bars
                "component_scores":   r.get("component_scores", {}),
                "reasons":            r.get("reasons", []),
            })
    data = {
        "scan_time":       datetime.now(tz=timezone.utc).isoformat(),
        "btc_price":       round(float(price), 2),
        "daily_trend":     mtf.get("mtf_daily_trend", "neutral"),
        "4h_trend":        mtf.get("mtf_4h_trend",    "neutral"),
        "weekly_trend":    mtf.get("mtf_weekly_trend", "neutral"),
        "bull_count":      mtf.get("mtf_bull_count", 0),
        "fg_value":        fg,
        "sentiment_score": round(float(sent), 4),
        "patterns":        patterns_to_save,
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/active_patterns.json", "w") as f:
        json.dump(data, f, indent=2)
    log.info(f"Saved {len(patterns_to_save)} active pattern(s) to active_patterns.json")


def _send_telegram(result, exec_result, sent, fg, fg_label, price, mtf, intel):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    import requests

    pattern    = result["pattern"]
    trade_type = result["trade_type"]
    executed   = exec_result.get("executed", False)
    dir_icon   = "BUY" if pattern.direction == "bullish" else "SELL"
    status     = "AUTO-TRADE PLACED ✅" if executed else "SIGNAL — manual entry"

    confluences = "\n".join(f"   {r}" for r in result["reasons"][:6])

    trigger_15m  = mtf.get("mtf_15m_trigger", False)
    reasons_15m  = mtf.get("mtf_15m_reasons", [])
    entry_line   = ""
    if trigger_15m:
        entry_line = "\n15M entry: " + ", ".join(reasons_15m[:2])
    elif reasons_15m:
        entry_line = "\n15M watching: " + reasons_15m[0]

    # On-chain warning
    warnings = intel.get("warnings", [])
    warn_line = ("\n⚠️ " + warnings[0]) if warnings else ""

    spot_line = f"\nSPOT {dir_icon}: ${pattern.entry:,.2f}" if trade_type["spot"] else ""
    fut_line  = f"\nFUTURES {trade_type['leverage']}x" if trade_type["futures"] else ""

    msg = (
        f"*NEXUS v2 — {dir_icon} SIGNAL*\n"
        f"_{status}_\n\n"
        f"*{pattern.name.replace('_',' ').title()}* [{pattern.timeframe}] {pattern.status}\n"
        f"Score: {result['score']}/100 | Confluences: {result['n_confluences']}/5\n\n"
        f"*Entry:*  ${pattern.entry:,.2f}\n"
        f"*Stop:*   ${pattern.stop_loss:,.2f}\n"
        f"*Target:* ${pattern.target:,.2f}\n"
        f"*R:R:*    1:{pattern.risk_reward:.1f}\n"
        f"{spot_line}{fut_line}{entry_line}{warn_line}\n\n"
        f"*Breakdown:*\n{confluences}\n\n"
        f"Daily: {mtf.get('mtf_daily_trend','?')} | 4H: {mtf.get('mtf_4h_trend','?')}\n"
        f"Sentiment: {sent:+.2f} | F&G: {fg} ({fg_label})\n"
        f"BTC: ${price:,.2f}"
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


def _log_signal(result, exec_result, sentiment, fg):
    pattern = result["pattern"]
    row = {
        "timestamp":   datetime.now(tz=timezone.utc).isoformat(),
        "pattern":     pattern.name,
        "direction":   pattern.direction,
        "timeframe":   pattern.timeframe,
        "status":      pattern.status,
        "score":       result["score"],
        "confluences": result["n_confluences"],
        "entry":       round(pattern.entry, 2),
        "stop_loss":   round(pattern.stop_loss, 2),
        "take_profit": round(pattern.target, 2),
        "risk_reward": pattern.risk_reward,
        "auto_traded": exec_result.get("executed", False),
        "sentiment":   round(sentiment, 4),
        "fear_greed":  fg,
    }
    write_head = not os.path.exists(SIGNAL_LOG)
    with open(SIGNAL_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_head:
            w.writeheader()
        w.writerow(row)


def _write_summary(path, results, price, mtf, sent, fg, fg_label, memory, executed, intel):
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    deriv = intel.get("derivatives", {})
    lines = [
        f"## NEXUS v2 — {now}",
        f"**BTC:** ${price:,.2f} | Daily: {mtf.get('mtf_daily_trend','?')} | 4H: {mtf.get('mtf_4h_trend','?')} | 15M: {mtf.get('mtf_15m_trigger',False)}",
        f"**Funding:** {deriv.get('funding_rate',0)*100:+.4f}% | **L/S:** {deriv.get('long_short_ratio',1):.2f} | **Intel:** {intel.get('signal_label','?')}",
        "",
    ]
    if results and results[0]["tradeable_signal"]:
        p  = results[0]["pattern"]
        cs = results[0].get("component_scores", {})
        lines += [
            f"### {'BUY' if p.direction=='bullish' else 'SELL'} — {p.name.replace('_',' ').title()} [{p.timeframe}]",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Score | {results[0]['score']}/100 |",
            f"| Confluences | {results[0]['n_confluences']}/5 |",
            f"| Status | {p.status} |",
            f"| Entry | ${p.entry:,.2f} |",
            f"| Stop Loss | ${p.stop_loss:,.2f} |",
            f"| Take Profit | ${p.target:,.2f} |",
            f"| R:R | 1:{p.risk_reward:.1f} |",
            f"| Pattern | {cs.get('pattern',0)}/25 |",
            f"| MTF | {cs.get('mtf',0)}/20 |",
            f"| Retest | {cs.get('retest',0)}/20 |",
            f"| Volume | {cs.get('volume',0)}/15 |",
            f"| RSI | {cs.get('rsi',0)}/10 |",
            f"| Sentiment | {cs.get('sentiment',0)}/10 |",
            f"| On-chain adj | {cs.get('onchain',0):+d} |",
            "",
            "**Reasons:**",
        ]
        for r in results[0]["reasons"]:
            lines.append(f"- {r}")
    else:
        lines.append("### No tradeable signals this run")
        if results:
            lines.append(f"Best score: {results[0]['score']}/100 — below 60 threshold")

    warns = intel.get("warnings", [])
    if warns:
        lines += ["", "**⚠️ Market Warnings:**"]
        for w in warns:
            lines.append(f"- {w}")

    wins   = sum(1 for t in memory if t.get("outcome") == "WIN")
    losses = sum(1 for t in memory if t.get("outcome") == "LOSS")
    opens  = sum(1 for t in memory if t.get("outcome") == "OPEN")
    total  = wins + losses
    wr     = f"{wins/total*100:.1f}%" if total > 0 else "—"

    lines += [
        "",
        "---",
        "### Trade memory",
        f"**Total:** {len(memory)} | **Open:** {opens} | **Win rate:** {wr} ({wins}W / {losses}L)",
        f"**Sentiment:** {sent:+.2f} | **F&G:** {fg} ({fg_label})",
    ]

    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")


def _write_summary_no_patterns(price, memory):
    path = os.getenv("GITHUB_STEP_SUMMARY", "")
    if not path:
        return
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with open(path, "a") as f:
        f.write(f"## NEXUS v2 — {now}\n")
        f.write(f"**BTC:** ${price:,.2f} | No patterns detected this run\n\n")
        wins   = sum(1 for t in memory if t.get("outcome") == "WIN")
        losses = sum(1 for t in memory if t.get("outcome") == "LOSS")
        f.write(f"Trade history: {wins}W / {losses}L\n")


if __name__ == "__main__":
    main()
