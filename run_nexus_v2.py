"""
NEXUS v2 — Professional Trading Signal System
===============================================
Full pipeline every hour:

  1. Load trade memory → validate open positions
  2. Fetch BTC 1H data (2 years) + 15M data (60 days)
  3. Detect chart patterns on Daily, 4H, 1H
  4. Run MTF analysis: Weekly → Daily → 4H → 15M entry
  5. Score each pattern with full confluence engine
  6. If score >= 60 AND 3+ confluences: Telegram + paper trade
  7. Log everything to trade_memory.csv
  8. Write GitHub Actions step summary

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
    df["symbol"] = "BTC/USDT"
    df["timeframe"] = "1h"

    current_price = float(df["close"].iloc[-1])
    log.info(f"BTC price: ${current_price:,.2f} | {len(df)} 1H candles loaded")

    # ── Fetch 15M data (60 days, for entry timing) ─────────────────────────────
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
            log.warning("15M data empty — using 1H only for entry timing")
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

    # ── Step 4: MTF analysis (now includes 15M) ───────────────────────────────
    log.info("Running MTF analysis (Daily → 4H → 15M entry)...")
    try:
        from data.multi_timeframe import MultiTimeframeAnalyzer
        mtf_features = MultiTimeframeAnalyzer().analyze(df, df_15m)
        log.info(
            f"MTF: Daily={mtf_features.get('mtf_daily_trend')} | "
            f"4H={mtf_features.get('mtf_4h_trend')} | "
            f"Bull {mtf_features.get('mtf_bull_count',0)}/4 | "
            f"15M trigger: {mtf_features.get('mtf_15m_trigger',False)}"
        )
        if mtf_features.get("mtf_15m_reasons"):
            for r in mtf_features["mtf_15m_reasons"]:
                log.info(f"  15M: {r}")
    except Exception as e:
        log.warning(f"MTF failed: {e}")
        mtf_features = {}

    # ── Step 5: News sentiment ────────────────────────────────────────────────
    log.info("Fetching news sentiment...")
    try:
        from data.news_collector import NewsCollector
        sentiment       = NewsCollector().get_sentiment_report()
        sentiment_score = sentiment.composite_score
        fg_value        = sentiment.fear_greed_value
        fg_label        = sentiment.fear_greed_label
        log.info(f"Sentiment: {sentiment_score:+.3f} | F&G: {fg_value} ({fg_label})")
    except Exception as e:
        log.warning(f"Sentiment failed: {e}")
        sentiment_score, fg_value, fg_label = 0.0, 50, "Neutral"

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
        _send_telegram(result, exec_result, sentiment_score, fg_value, fg_label, current_price, mtf_features)
        _log_signal(result, exec_result, sentiment_score, fg_value)

        if exec_result.get("executed"):
            executed_trades.append(exec_result)

    # ── Step 8: GitHub Actions summary ───────────────────────────────────────
    summary_path = os.getenv("GITHUB_STEP_SUMMARY", "")
    if summary_path:
        _write_summary(
            summary_path, top_results, current_price,
            mtf_features, sentiment_score, fg_value, fg_label,
            trade_memory, executed_trades,
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
                            executor.update_outcome(trade["signal_id"], "LOSS", sl)
                            break
                        if h >= tp:
                            executor.update_outcome(trade["signal_id"], "WIN", tp)
                            break
                    else:
                        if h >= sl:
                            executor.update_outcome(trade["signal_id"], "LOSS", sl)
                            break
                        if l <= tp:
                            executor.update_outcome(trade["signal_id"], "WIN", tp)
                            break
            except Exception as e:
                log.debug(f"Validation failed for {trade.get('signal_id')}: {e}")
    except Exception as e:
        log.warning(f"Trade validation failed: {e}")


def _send_telegram(result, exec_result, sent, fg, fg_label, price, mtf):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    import requests

    pattern    = result["pattern"]
    trade_type = result["trade_type"]
    executed   = exec_result.get("executed", False)

    dir_icon = "BUY" if pattern.direction == "bullish" else "SELL"
    status   = "AUTO-TRADE PLACED" if executed else "SIGNAL — manual entry"

    confluences = "\n".join(
        f"   {'OK' if '+' in r else '--'} {r}"
        for r in result["reasons"]
    )

    # 15M entry status
    trigger_15m = mtf.get("mtf_15m_trigger", False)
    reasons_15m = mtf.get("mtf_15m_reasons", [])
    entry_line  = ""
    if trigger_15m:
        entry_line = "\n15M entry: " + ", ".join(reasons_15m[:2])
    elif reasons_15m:
        entry_line = "\n15M entry: " + reasons_15m[0] + " (watching)"

    spot_line = ""
    fut_line  = ""
    if trade_type["spot"]:
        spot_line = f"\nSPOT {dir_icon}: entry ${pattern.entry:,.2f}"
    if trade_type["futures"]:
        lev = trade_type["leverage"]
        fut_line = f"\nFUTURES {'LONG' if pattern.direction=='bullish' else 'SHORT'} {lev}x"

    msg = (
        f"*NEXUS v2 — {dir_icon} SIGNAL*\n"
        f"_{status}_\n\n"
        f"*Pattern:* {pattern.name.replace('_',' ').title()}\n"
        f"*Timeframe:* {pattern.timeframe} | Status: {pattern.status}\n"
        f"*Score:* {result['score']}/100 | Confluences: {result['n_confluences']}/5\n\n"
        f"*Entry:*     ${pattern.entry:,.2f}\n"
        f"*Stop Loss:* ${pattern.stop_loss:,.2f}\n"
        f"*Target:*    ${pattern.target:,.2f}\n"
        f"*R:R:*       1:{pattern.risk_reward:.1f}\n"
        f"{spot_line}{fut_line}{entry_line}\n\n"
        f"*Confluences:*\n{confluences}\n\n"
        f"Daily: {mtf.get('mtf_daily_trend','?')} | 4H: {mtf.get('mtf_4h_trend','?')}\n"
        f"Sentiment: {sent:+.2f} | F&G: {fg} ({fg_label})\n"
        f"BTC: ${price:,.2f}"
    )

    try:
        import requests
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


def _write_summary(path, results, price, mtf, sent, fg, fg_label, memory, executed):
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"## NEXUS v2 — {now}",
        f"**BTC:** ${price:,.2f} | Daily: {mtf.get('mtf_daily_trend','?')} | 4H: {mtf.get('mtf_4h_trend','?')} | 15M trigger: {mtf.get('mtf_15m_trigger',False)}",
        "",
    ]
    if results and results[0]["tradeable_signal"]:
        p = results[0]["pattern"]
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
            "",
            "**Reasons:**",
        ]
        for r in results[0]["reasons"]:
            lines.append(f"- {r}")
        if mtf.get("mtf_15m_reasons"):
            lines.append(f"- 15M: " + ", ".join(mtf["mtf_15m_reasons"][:2]))
    else:
        lines.append("### No tradeable signals this run")
        if results:
            lines.append(f"Best score: {results[0]['score']}/100 — below threshold")

    wins  = sum(1 for t in memory if t.get("outcome") == "WIN")
    losses= sum(1 for t in memory if t.get("outcome") == "LOSS")
    opens = sum(1 for t in memory if t.get("outcome") == "OPEN")
    total = wins + losses
    wr    = f"{wins/total*100:.1f}%" if total > 0 else "—"

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
