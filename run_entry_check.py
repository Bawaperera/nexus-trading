"""
NEXUS v2 — 15-Minute Entry Trigger Checker
===========================================
Lightweight companion to run_nexus_v2.py. Runs every 15 min.

Problem it solves:
  Main hourly scan finds a bear flag at 2:00pm.
  The 15M entry trigger fires at 2:23pm.
  Without this script: you miss it until 3:00pm.
  With this script:    you get the Telegram within 15 minutes.

What it does:
  1. Read active_patterns.json (written each hour by run_nexus_v2.py)
  2. Fetch only 5 days of 15M BTC data (~500KB, takes <1 second)
  3. Check the 15M entry trigger (engulfing candle, volume, structure break)
  4. If triggered AND matches an active pattern: send Telegram alert

What it does NOT do:
  - Re-run the full pattern scan (that's the hourly job)
  - Re-fetch news/sentiment (already scored hourly)
  - Reinstall heavy packages (uses minimal deps)
"""

import os, json, csv, sys, logging, warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(__file__))

TELEGRAM_TOKEN        = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID      = os.getenv("TELEGRAM_CHAT_ID", "")
ACTIVE_PATTERNS_PATH  = "logs/active_patterns.json"
SIGNAL_LOG            = "logs/signals_v2.csv"
MAX_PATTERN_AGE_HOURS = 2   # stop checking patterns older than 2 hours


def main():
    now_utc = datetime.now(tz=timezone.utc)
    log.info("=" * 50)
    log.info(f"  NEXUS Entry Check — {now_utc.strftime('%Y-%m-%d %H:%M UTC')}")
    log.info("=" * 50)

    # ── Step 1: Load active patterns saved by hourly run ──────────────────
    if not os.path.exists(ACTIVE_PATTERNS_PATH):
        log.info("No active_patterns.json yet — wait for the hourly scan to run once")
        return

    with open(ACTIVE_PATTERNS_PATH) as f:
        data = json.load(f)

    scan_time = datetime.fromisoformat(data["scan_time"])
    age_hours = (now_utc - scan_time).total_seconds() / 3600
    if age_hours > MAX_PATTERN_AGE_HOURS:
        log.info(f"Patterns are {age_hours:.1f}h old (max {MAX_PATTERN_AGE_HOURS}h) — skipping")
        return

    tradeable = [
        p for p in data.get("patterns", [])
        if p.get("tradeable_signal") and not p.get("already_traded")
    ]

    if not tradeable:
        log.info("No active tradeable patterns to watch")
        _write_summary(None, None, data, 0)
        return

    log.info(f"Watching {len(tradeable)} pattern(s): {', '.join(p['name'] for p in tradeable)}")

    # ── Step 2: Fetch only 5 days of 15M data (very fast) ─────────────────
    import yfinance as yf
    import pandas as pd

    log.info("Fetching 5 days of 15M BTC data...")
    raw = yf.download("BTC-USD", period="5d", interval="15m",
                      progress=False, auto_adjust=True)
    if raw.empty:
        log.warning("15M fetch returned no data")
        return

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]
    raw.index   = pd.to_datetime(raw.index, utc=True)
    df_15m = raw[["open","high","low","close","volume"]].dropna().copy()

    btc_price = float(df_15m["close"].iloc[-1])
    log.info(f"BTC: ${btc_price:,.2f} | {len(df_15m)} 15M candles loaded")

    # ── Step 3: Check 15M entry trigger ───────────────────────────────────
    from data.multi_timeframe import MultiTimeframeAnalyzer
    entry = MultiTimeframeAnalyzer()._analyze_15m_entry(df_15m)

    trigger_word = "FIRED ✓" if entry["triggered"] else "not fired"
    log.info(
        f"15M trigger: {trigger_word} | "
        f"Score: {entry['score']}/6 (need 3+) | "
        f"Direction: {entry['direction']}"
    )
    for reason in entry.get("reasons", []):
        log.info(f"  {reason}")

    _write_summary(entry, btc_price, data, len(tradeable))

    if not entry["triggered"]:
        log.info("Nothing to do — check again in 15 minutes")
        return

    # ── Step 4: Find patterns that match the 15M direction ─────────────────
    aligned = [p for p in tradeable if p["direction"] == entry["direction"]]

    if not aligned:
        log.info(
            f"15M trigger direction ({entry['direction']}) doesn't match "
            f"any active pattern. Active: {[p['direction'] for p in tradeable]}"
        )
        return

    log.info(f"15M ENTRY TRIGGER CONFIRMED — {len(aligned)} pattern(s) align!")

    # ── Step 5: Send Telegram alert for each aligned pattern ───────────────
    for pattern in aligned:
        _send_telegram(pattern, entry, btc_price, data)
        _log_signal(pattern, entry)
        log.info(f"  Alert sent → {pattern['name']} [{pattern['timeframe']}] | R:R {pattern['risk_reward']}")

    # ── Step 6: Mark as alerted so we don't spam every 15 min ─────────────
    for p in data["patterns"]:
        for aligned_p in aligned:
            if p["name"] == aligned_p["name"] and p["timeframe"] == aligned_p["timeframe"]:
                p["already_traded"] = True
    data["last_entry_check"] = now_utc.isoformat()

    with open(ACTIVE_PATTERNS_PATH, "w") as f:
        json.dump(data, f, indent=2)

    log.info("Patterns marked as alerted — will reset on next hourly scan")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _send_telegram(pattern: dict, entry: dict, price: float, data: dict):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram credentials not set — skipping alert")
        return

    import requests

    side     = "BUY" if pattern["direction"] == "bullish" else "SELL"
    pat_name = pattern["name"].replace("_", " ").title()
    reasons  = "\n".join(f"   • {r}" for r in entry.get("reasons", []))

    msg = (
        f"*NEXUS v2 — 15M ENTRY* 🎯\n\n"
        f"*{side}* — {pat_name} [{pattern['timeframe'].upper()}]\n"
        f"Score: {pattern['score']}/100 | R:R: 1:{pattern['risk_reward']}\n\n"
        f"*Entry:*     ${pattern['entry']:,.2f}\n"
        f"*Stop Loss:* ${pattern['stop_loss']:,.2f}\n"
        f"*Target:*    ${pattern['target']:,.2f}\n\n"
        f"*15M confirmation ({entry['score']}/6):*\n{reasons}\n\n"
        f"Daily: {data.get('daily_trend','?')} | "
        f"4H: {data.get('4h_trend','?')} | "
        f"F&G: {data.get('fg_value','?')}\n"
        f"BTC now: ${price:,.2f}"
    )

    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
        if r.json().get("ok"):
            log.info("Telegram sent")
        else:
            log.warning(f"Telegram error: {r.json().get('description')}")
    except Exception as e:
        log.warning(f"Telegram failed: {e}")


def _log_signal(pattern: dict, entry: dict):
    """Log the entry trigger to signals_v2.csv."""
    row = {
        "timestamp":   datetime.now(tz=timezone.utc).isoformat(),
        "pattern":     pattern["name"],
        "direction":   pattern["direction"],
        "timeframe":   pattern["timeframe"],
        "status":      pattern["status"] + "+15M",
        "score":       pattern["score"],
        "confluences": pattern.get("n_confluences", 0),
        "entry":       pattern["entry"],
        "stop_loss":   pattern["stop_loss"],
        "take_profit": pattern["target"],
        "risk_reward": pattern["risk_reward"],
        "auto_traded": False,
        "sentiment":   0,
        "fear_greed":  0,
    }
    os.makedirs("logs", exist_ok=True)
    write_head = not os.path.exists(SIGNAL_LOG)
    with open(SIGNAL_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_head:
            w.writeheader()
        w.writerow(row)


def _write_summary(entry, price, data, n_watching):
    """Write to GitHub Actions run summary page."""
    path = os.getenv("GITHUB_STEP_SUMMARY", "")
    if not path:
        return
    now = datetime.now(tz=timezone.utc).strftime("%H:%M UTC")
    with open(path, "a") as f:
        if entry and entry.get("triggered"):
            f.write(f"### 🎯 NEXUS 15M Entry Trigger — {now}\n\n")
            f.write(f"**BTC:** ${price:,.2f} | **15M trigger FIRED** (score {entry['score']}/6)\n\n")
            f.write("**Signals:**\n")
            for r in entry.get("reasons", []):
                f.write(f"- {r}\n")
            f.write("\n")
        else:
            score = entry["score"] if entry else 0
            f.write(f"### NEXUS Entry Check — {now}\n\n")
            p = f"${price:,.2f}" if price else "—"
            f.write(f"**BTC:** {p} | 15M trigger: no (score {score}/6, need 3+) | Watching {n_watching} pattern(s)\n\n")


if __name__ == "__main__":
    main()
