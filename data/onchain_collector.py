"""
NEXUS v2 — Market Intelligence Collector
=========================================
Gathers on-chain, derivatives, macro, and social data.

Free sources — zero API keys required:
  1. Binance Futures public API
     - Funding rate  (positive = longs paying → bearish squeeze risk)
     - Open interest (OI rising + price rising = trend confirmation)
     - Long/Short ratio (>1.3 = crowded longs → squeeze risk)
     - Top trader position ratio (smart money vs retail)
  2. Reddit r/CryptoCurrency
     - Sentiment scored from hot post titles (weighted by upvotes)
  3. Forex Factory Economic Calendar
     - Upcoming high-impact USD events (CPI, FOMC, NFP, GDP)
     - Risk flag: event within 2h → reduce auto-trade confidence
  4. CoinGecko Trending
     - Top 7 trending coins (free API, no auth)
     - BTC in trending = retail interest = mildly bullish

Output: logs/market_intel.json — read by dashboard + confluence engine
"""

import json
import logging
import os
import requests
from datetime import datetime, timezone, timedelta

log = logging.getLogger(__name__)

BINANCE_FUTURES = "https://fapi.binance.com"
HEADERS = {"User-Agent": "NEXUS-Trading-Research-Bot/2.0"}

BULLISH_WORDS = {
    "bullish", "moon", "rally", "pump", "breakout", "surge", "ath",
    "green", "gains", "recover", "bounce", "buy", "accumulate",
    "adoption", "institutional", "approved", "etf", "bull",
}
BEARISH_WORDS = {
    "bearish", "dump", "crash", "drop", "fall", "fear", "red",
    "loss", "bleeding", "down", "bear", "sell", "ban", "hack",
    "regulation", "lawsuit", "fraud", "collapse", "liquidation",
}


class OnChainCollector:
    """
    Collects market intelligence from multiple free public APIs.

    Usage:
        intel = OnChainCollector().collect()
        OnChainCollector().save(intel)
        print(intel["composite_signal"])   # -1.0 to +1.0
    """

    def collect(self, symbol: str = "BTCUSDT") -> dict:
        log.info("Collecting market intelligence from all sources...")

        derivatives = self._get_derivatives(symbol)
        macro       = self._get_macro_calendar()
        social      = self._get_reddit_sentiment()
        trending    = self._get_coingecko_trending()

        # ── Composite signal: -1 (bearish) to +1 (bullish) ────────────────
        scores = []

        # Funding rate: extreme positive = overleveraged longs = squeeze risk
        fr = derivatives.get("funding_rate", 0.0)
        if   fr >  0.001:   scores.append(-0.6)
        elif fr >  0.0003:  scores.append(-0.2)
        elif fr < -0.001:   scores.append(0.6)
        elif fr < -0.0003:  scores.append(0.2)
        else:               scores.append(0.0)

        # L/S ratio: >1.3 = crowded longs = squeeze risk
        ls = derivatives.get("long_short_ratio", 1.0)
        if   ls > 1.5:   scores.append(-0.4)
        elif ls > 1.2:   scores.append(-0.1)
        elif ls < 0.75:  scores.append(0.4)
        elif ls < 0.85:  scores.append(0.1)
        else:            scores.append(0.0)

        # OI change: rising = trend confirmation
        oi_chg = derivatives.get("oi_change_pct", 0.0)
        scores.append(max(-0.3, min(0.3, oi_chg / 100 * 3)))

        # Reddit (contrarian: extreme greed = caution)
        reddit_sent = social.get("sentiment", 0.0)
        if abs(reddit_sent) > 0.5:
            scores.append(-reddit_sent * 0.2)
        else:
            scores.append(reddit_sent * 0.2)

        # CoinGecko: BTC trending = mildly bullish retail interest
        if trending.get("btc_in_trending"):
            scores.append(0.1)

        composite = sum(scores) / max(len(scores), 1)
        composite = max(-1.0, min(1.0, composite))

        # ── Warnings ──────────────────────────────────────────────────────
        warnings = []
        if fr > 0.001:
            warnings.append(f"⚠️ Extreme positive funding ({fr*100:.4f}%) — longs overpaying, squeeze risk")
        elif fr < -0.001:
            warnings.append(f"⚠️ Extreme negative funding ({fr*100:.4f}%) — shorts overpaying, squeeze risk")
        if ls > 1.5:
            warnings.append(f"⚠️ Crowded longs ({ls:.2f}x) — long squeeze possible")
        elif ls < 0.7:
            warnings.append(f"⚠️ Crowded shorts ({ls:.2f}x) — short squeeze possible")

        macro_risk = macro.get("risk_level", "low")
        if macro_risk == "high":
            hours = macro.get("hours_until_next", 99)
            warnings.append(f"⚠️ High-impact macro event in {hours:.1f}h — auto-trade confidence reduced")

        # ── Score adjustment for confluence engine (-5 to +5) ──────────────
        onchain_score_adj = 0
        if fr < -0.0003:    onchain_score_adj += 3
        elif fr > 0.0003:   onchain_score_adj -= 3
        if ls < 0.85:       onchain_score_adj += 2
        elif ls > 1.3:      onchain_score_adj -= 2
        if macro_risk == "high":
            onchain_score_adj = 0  # neutralize during macro events
        onchain_score_adj = max(-5, min(5, onchain_score_adj))

        report = {
            "timestamp":           datetime.now(tz=timezone.utc).isoformat(),
            "composite_signal":    round(composite, 4),
            "signal_label":        self._label(composite),
            "onchain_score_adj":   onchain_score_adj,
            "warnings":            warnings,
            "derivatives":         derivatives,
            "macro":               macro,
            "social":              social,
            "trending":            trending,
        }

        log.info(
            f"Market Intel: {report['signal_label']} ({composite:+.3f}) | "
            f"Funding: {fr*100:+.4f}% | L/S: {ls:.2f} | "
            f"Macro risk: {macro_risk} | Reddit: {social.get('label', '?')} | "
            f"Trending: {', '.join(trending.get('trending_coins', [])[:3])}"
        )
        for w in warnings:
            log.warning(w)

        return report

    # ── Binance Futures ────────────────────────────────────────────────────

    def _get_derivatives(self, symbol: str) -> dict:
        result = {
            "funding_rate":      0.0,
            "funding_signal":    "Neutral",
            "open_interest_btc": 0.0,
            "oi_change_pct":     0.0,
            "long_short_ratio":  1.0,
            "long_pct":          50.0,
            "short_pct":         50.0,
            "top_trader_ls":     1.0,
        }

        # Funding rate
        try:
            r = requests.get(
                f"{BINANCE_FUTURES}/fapi/v1/fundingRate",
                params={"symbol": symbol, "limit": 1},
                timeout=5, headers=HEADERS,
            )
            data = r.json()
            if data:
                result["funding_rate"] = float(data[0]["fundingRate"])
        except Exception as e:
            log.debug(f"Funding rate: {e}")

        # Open interest history (for % change)
        try:
            r = requests.get(
                f"{BINANCE_FUTURES}/futures/data/openInterestHist",
                params={"symbol": symbol, "period": "1h", "limit": 2},
                timeout=5, headers=HEADERS,
            )
            hist = r.json()
            if isinstance(hist, list) and len(hist) >= 2:
                oi_now  = float(hist[-1]["sumOpenInterest"])
                oi_prev = float(hist[-2]["sumOpenInterest"])
                result["open_interest_btc"] = oi_now
                result["oi_change_pct"] = round((oi_now - oi_prev) / max(oi_prev, 1) * 100, 3)
        except Exception as e:
            log.debug(f"OI history: {e}")

        # Global Long/Short ratio
        try:
            r = requests.get(
                f"{BINANCE_FUTURES}/futures/data/globalLongShortAccountRatio",
                params={"symbol": symbol, "period": "1h", "limit": 1},
                timeout=5, headers=HEADERS,
            )
            data = r.json()
            if isinstance(data, list) and data:
                result["long_short_ratio"] = float(data[0]["longShortRatio"])
                result["long_pct"]         = round(float(data[0]["longAccount"])  * 100, 2)
                result["short_pct"]        = round(float(data[0]["shortAccount"]) * 100, 2)
        except Exception as e:
            log.debug(f"L/S ratio: {e}")

        # Top trader position ratio
        try:
            r = requests.get(
                f"{BINANCE_FUTURES}/futures/data/topLongShortPositionRatio",
                params={"symbol": symbol, "period": "1h", "limit": 1},
                timeout=5, headers=HEADERS,
            )
            data = r.json()
            if isinstance(data, list) and data:
                result["top_trader_ls"] = float(data[0]["longShortRatio"])
        except Exception as e:
            log.debug(f"Top trader ratio: {e}")

        # Funding signal label
        fr = result["funding_rate"]
        if   fr >  0.001:   result["funding_signal"] = "⚠️ Extreme — longs at risk"
        elif fr >  0.0003:  result["funding_signal"] = "Positive — mild bearish pressure"
        elif fr < -0.001:   result["funding_signal"] = "⚠️ Extreme — shorts at risk"
        elif fr < -0.0003:  result["funding_signal"] = "Negative — mild bullish pressure"
        else:               result["funding_signal"] = "Neutral — balanced"

        return result

    # ── Forex Factory Economic Calendar ───────────────────────────────────

    def _get_macro_calendar(self) -> dict:
        try:
            r = requests.get(
                "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
                timeout=8, headers=HEADERS,
            )
            events = r.json()
            now    = datetime.now(tz=timezone.utc)
            window = now + timedelta(hours=24)

            upcoming = []
            for ev in events:
                if ev.get("country") != "USD":
                    continue
                if ev.get("impact") not in ("High", "Medium"):
                    continue
                try:
                    dt_str = ev.get("date", "")
                    if not dt_str:
                        continue
                    ev_dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                    if ev_dt.tzinfo is None:
                        ev_dt = ev_dt.replace(tzinfo=timezone.utc)
                    if now <= ev_dt <= window:
                        hours = (ev_dt - now).total_seconds() / 3600
                        upcoming.append({
                            "title":       ev.get("title", ""),
                            "time_utc":    ev_dt.strftime("%H:%M UTC"),
                            "impact":      ev.get("impact", "Medium"),
                            "hours_until": round(hours, 1),
                            "forecast":    ev.get("forecast", "—"),
                            "previous":    ev.get("previous", "—"),
                        })
                except Exception:
                    continue

            upcoming.sort(key=lambda e: e["hours_until"])
            hours_until = upcoming[0]["hours_until"] if upcoming else 999

            if any(e["hours_until"] < 1 and e["impact"] == "High" for e in upcoming):
                risk = "high"
            elif any(e["impact"] == "High" for e in upcoming):
                risk = "medium"
            elif upcoming:
                risk = "low"
            else:
                risk = "clear"

            return {
                "upcoming_events":  upcoming[:6],
                "risk_level":       risk,
                "hours_until_next": hours_until,
                "event_count_24h":  len(upcoming),
            }

        except Exception as e:
            log.warning(f"Macro calendar fetch failed: {e}")
            return {
                "upcoming_events":  [],
                "risk_level":       "unknown",
                "hours_until_next": 999,
                "event_count_24h":  0,
            }

    # ── Reddit Sentiment ───────────────────────────────────────────────────

    def _get_reddit_sentiment(self) -> dict:
        try:
            r = requests.get(
                "https://www.reddit.com/r/CryptoCurrency/hot.json?limit=25",
                headers={"User-Agent": "NEXUS-Trading-Research/2.0 (educational project)"},
                timeout=8,
            )
            posts = r.json().get("data", {}).get("children", [])

            bull, bear = 0.0, 0.0
            titles = []

            for post in posts[:20]:
                pd_ = post.get("data", {})
                title  = pd_.get("title", "").lower()
                ups    = int(pd_.get("ups", 0))
                weight = min(3.0, ups / 500 + 1.0)
                titles.append(pd_.get("title", "")[:90])

                for w in BULLISH_WORDS:
                    if w in title:
                        bull += weight
                for w in BEARISH_WORDS:
                    if w in title:
                        bear += weight
                if any(neg in title for neg in ["not", "no longer", "never", "isn't"]):
                    bull = max(0, bull - weight * 0.5)

            total     = bull + bear
            sentiment = (bull - bear) / total if total > 0 else 0.0
            sentiment = round(max(-1.0, min(1.0, sentiment)), 3)

            label = "Very Bullish" if sentiment > 0.4 else \
                    "Bullish"      if sentiment > 0.1 else \
                    "Very Bearish" if sentiment < -0.4 else \
                    "Bearish"      if sentiment < -0.1 else "Neutral"

            return {
                "sentiment":  sentiment,
                "label":      label,
                "bull_score": round(bull, 1),
                "bear_score": round(bear, 1),
                "post_count": len(posts),
                "top_titles": titles[:5],
            }

        except Exception as e:
            log.warning(f"Reddit sentiment failed: {e}")
            return {
                "sentiment":  0.0,
                "label":      "Neutral",
                "bull_score": 0,
                "bear_score": 0,
                "post_count": 0,
                "top_titles": [],
            }

    # ── CoinGecko Trending ─────────────────────────────────────────────────

    def _get_coingecko_trending(self) -> dict:
        """
        CoinGecko trending coins — free, no auth required.
        BTC appearing in trending = strong retail interest = mildly bullish.
        Many altcoins trending without BTC = risk-on without bitcoin = caution.
        """
        try:
            r = requests.get(
                "https://api.coingecko.com/api/v3/search/trending",
                timeout=5, headers=HEADERS,
            )
            data  = r.json()
            coins = [c["item"]["name"] for c in data.get("coins", [])[:7]]
            slugs = [c["item"].get("id", "").lower() for c in data.get("coins", [])[:7]]

            btc_trending = any(
                "bitcoin" in s or s == "btc"
                for s in slugs + [c.lower() for c in coins]
            )

            return {
                "trending_coins":   coins,
                "btc_in_trending":  btc_trending,
                "trend_count":      len(coins),
            }

        except Exception as e:
            log.debug(f"CoinGecko trending: {e}")
            return {
                "trending_coins":  [],
                "btc_in_trending": False,
                "trend_count":     0,
            }

    # ── Helpers ────────────────────────────────────────────────────────────

    def _label(self, score: float) -> str:
        if   score >  0.4: return "Strong Bullish"
        elif score >  0.1: return "Bullish"
        elif score < -0.4: return "Strong Bearish"
        elif score < -0.1: return "Bearish"
        return "Neutral"

    def save(self, report: dict, path: str = "logs/market_intel.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        log.info(f"Market intel saved → {path}")
