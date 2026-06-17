"""
NEXUS — Real-time News & Sentiment Collector
Pulls crypto and macro news from multiple FREE sources and scores market sentiment.

Sources (all free, no API key needed):
  Crypto News:
    1. CoinDesk RSS         → top crypto journalism
    2. CoinTelegraph RSS    → market news and analysis
    3. Decrypt RSS          → crypto culture and tech
    4. The Block RSS        → institutional-grade reporting
    5. Blockworks RSS       → research-focused crypto coverage
    6. The Defiant RSS      → DeFi and on-chain news
  Macro News:
    7. Federal Reserve RSS  → rate decisions, FOMC statements, policy press releases
    8. Investing.com RSS    → CPI, NFP, GDP, macro economic news

  + Fear & Greed Index     → alternative.me (market sentiment 0-100)
  + CryptoPanic API        → (optional) crypto-specific news

Sentiment scoring:
  - Keyword-based NLP on headlines (fast, no GPU needed)
  - Macro-aware: Fed/CPI/inflation keywords weighted for crypto impact
  - Outputs score -1.0 (extreme fear/bearish) to +1.0 (extreme greed/bullish)
"""

import feedparser
import requests
import logging
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

log = logging.getLogger(__name__)


# ─── Sentiment keywords ───────────────────────────────────────────────────────

BULLISH_KEYWORDS = {
    # Strong crypto positive
    "surge", "soar", "rally", "breakout", "bull", "bullish", "moon", "ath",
    "all-time high", "record", "milestone", "institutional", "adoption",
    "approval", "etf approved", "upgrade", "outperform", "massive gains",
    "bitcoin etf", "spot etf", "etf inflows",
    # Macro positive for crypto (rate cuts = risk-on = crypto up)
    "rate cut", "dovish", "pause", "pivot", "stimulus", "easy money",
    "soft landing", "lower rates", "cut rates", "accommodation",
    # Mild positive
    "rise", "gain", "up", "growth", "positive", "recovery", "rebound",
    "support", "buy", "accumulate", "strong", "momentum", "confidence",
    "optimistic", "partnership", "launch", "integration", "expansion",
}

BEARISH_KEYWORDS = {
    # Strong crypto negative
    "crash", "collapse", "plunge", "dump", "bear", "bearish", "hack",
    "exploit", "scam", "fraud", "ban", "crackdown", "bankrupt", "liquidation",
    "all-time low", "regulatory action", "sec", "lawsuit", "jail",
    # Macro negative for crypto (rate hikes = risk-off = crypto down)
    "rate hike", "hawkish", "tighten", "tightening", "inflation",
    "recession", "stagflation", "higher rates", "hike rates", "aggressive fed",
    "cpi higher", "hot inflation", "unemployment rises", "gdp contraction",
    # Mild negative
    "fall", "drop", "decline", "down", "loss", "fear", "concern", "risk",
    "warning", "caution", "sell", "resistance", "correction", "retreat",
    "uncertainty", "delay", "cancel", "suspend",
}

STRONG_KEYWORDS = {
    "crash", "collapse", "plunge", "hack", "exploit", "fraud", "ban",
    "surge", "soar", "ath", "all-time high", "etf approved", "breakout",
    "rate cut", "rate hike", "recession", "inflation",
}


@dataclass
class NewsItem:
    title:            str
    source:           str
    url:              str
    published:        datetime
    sentiment_score:  float
    sentiment_label:  str
    keywords_matched: list = field(default_factory=list)
    is_macro:         bool = False


@dataclass
class SentimentReport:
    timestamp:        datetime
    composite_score:  float
    sentiment_label:  str
    fear_greed_value: int
    fear_greed_label: str
    news_score:       float
    macro_score:      float
    news_count:       int
    macro_count:      int
    top_headlines:    list
    signal_weight:    float


class NewsCollector:
    """
    Collects real-time crypto + macro news and generates a composite
    sentiment score that feeds into the NEXUS Confluence Engine.

    Usage:
        collector = NewsCollector()
        report = collector.get_sentiment_report()
        print(report.composite_score)   # -0.4 = bearish
    """

    # ── Confirmed working RSS feeds (tested June 2026) ────────────────────────
    CRYPTO_FEEDS = {
        "CoinDesk":      "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "CoinTelegraph": "https://cointelegraph.com/rss",
        "Decrypt":       "https://decrypt.co/feed",
        "The Block":     "https://www.theblock.co/rss.xml",
        "Blockworks":    "https://blockworks.co/feed",
        "The Defiant":   "https://thedefiant.io/api/feed",
    }

    MACRO_FEEDS = {
        # Federal Reserve: press releases cover rate decisions, FOMC statements
        "Federal Reserve": "https://www.federalreserve.gov/feeds/press_all.xml",
        # Investing.com economy: covers CPI, NFP, GDP, unemployment reports
        "Investing.com":   "https://www.investing.com/rss/news_25.rss",
    }

    FEAR_GREED_URL  = "https://api.alternative.me/fng/?limit=1"
    CRYPTOPANIC_URL = "https://cryptopanic.com/api/free/v2/posts/?public=true&currencies=BTC"

    def __init__(
        self,
        cryptopanic_token:       str = None,
        cache_duration_minutes:  int = 15,
        max_article_age_hours:   int = 6,
    ):
        self.cryptopanic_token   = cryptopanic_token
        self.cache_duration      = timedelta(minutes=cache_duration_minutes)
        self.max_article_age     = timedelta(hours=max_article_age_hours)

        self._last_fetch:    Optional[datetime]       = None
        self._cached_report: Optional[SentimentReport] = None
        self.history:        deque                    = deque(maxlen=50)

        all_feeds = len(self.CRYPTO_FEEDS) + len(self.MACRO_FEEDS)
        log.info(
            f"NewsCollector ready | {all_feeds} feeds "
            f"({len(self.CRYPTO_FEEDS)} crypto + {len(self.MACRO_FEEDS)} macro) | "
            f"{'CryptoPanic ON' if cryptopanic_token else 'CryptoPanic OFF'}"
        )

    # ─── Public API ───────────────────────────────────────────────────────────

    def get_sentiment_report(self, force_refresh: bool = False) -> SentimentReport:
        if not force_refresh and self._is_cache_valid():
            return self._cached_report

        log.info("Fetching fresh news and sentiment data...")

        crypto_items = self._fetch_feeds(self.CRYPTO_FEEDS, is_macro=False)
        macro_items  = self._fetch_feeds(self.MACRO_FEEDS,  is_macro=True)
        all_items    = crypto_items + macro_items

        fear_greed   = self._fetch_fear_greed()
        crypto_score = self._aggregate_sentiment(crypto_items)
        macro_score  = self._aggregate_sentiment(macro_items)

        # Weight: 60% crypto news + 25% macro news + 15% fear/greed
        fg_normalized = (fear_greed["value"] - 50) / 50
        composite = (
            0.60 * crypto_score +
            0.25 * macro_score  +
            0.15 * fg_normalized
        )
        composite = max(-1.0, min(1.0, composite))

        signal_weight = min(1.0, len(all_items) / 20) * (1 + abs(fg_normalized) * 0.3)
        signal_weight = max(0.3, min(1.0, signal_weight))

        report = SentimentReport(
            timestamp        = datetime.now(tz=timezone.utc),
            composite_score  = round(composite, 4),
            sentiment_label  = self._score_to_label(composite),
            fear_greed_value = fear_greed["value"],
            fear_greed_label = fear_greed["label"],
            news_score       = round(crypto_score, 4),
            macro_score      = round(macro_score, 4),
            news_count       = len(crypto_items),
            macro_count      = len(macro_items),
            top_headlines    = [n.title for n in all_items[:6]],
            signal_weight    = round(signal_weight, 3),
        )

        self._cached_report = report
        self._last_fetch    = datetime.now(tz=timezone.utc)
        self.history.append(report)

        self._log_report(report)
        return report

    def get_sentiment_score(self) -> float:
        return self.get_sentiment_report().composite_score

    # ─── Data fetchers ────────────────────────────────────────────────────────

    def _fetch_feeds(self, feeds: dict, is_macro: bool) -> list:
        all_items = []
        cutoff    = datetime.now(tz=timezone.utc) - self.max_article_age

        for source, url in feeds.items():
            try:
                feed  = feedparser.parse(url)
                count = 0
                for entry in feed.entries[:20]:
                    published = self._parse_date(entry)
                    if published and published < cutoff:
                        continue
                    item = self._score_headline(
                        entry.title, source,
                        getattr(entry, "link", ""),
                        published, is_macro
                    )
                    all_items.append(item)
                    count += 1
                log.debug(f"{source}: {count} recent articles")
            except Exception as e:
                log.warning(f"RSS fetch failed ({source}): {e}")

        if not is_macro and self.cryptopanic_token:
            try:
                cp_items = self._fetch_cryptopanic()
                all_items.extend(cp_items)
                log.debug(f"CryptoPanic: {len(cp_items)} articles")
            except Exception as e:
                log.warning(f"CryptoPanic fetch failed: {e}")

        all_items.sort(
            key=lambda x: x.published or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True
        )
        log.info(f"{'Macro' if is_macro else 'Crypto'} articles: {len(all_items)}")
        return all_items

    def _fetch_fear_greed(self) -> dict:
        try:
            resp = requests.get(self.FEAR_GREED_URL, timeout=5)
            data = resp.json()["data"][0]
            result = {
                "value": int(data["value"]),
                "label": data["value_classification"],
            }
            log.info(f"Fear & Greed: {result['value']} — {result['label']}")
            return result
        except Exception as e:
            log.warning(f"Fear & Greed fetch failed: {e}")
            return {"value": 50, "label": "Neutral"}

    def _fetch_cryptopanic(self) -> list:
        url    = f"{self.CRYPTOPANIC_URL}&auth_token={self.cryptopanic_token}"
        resp   = requests.get(url, timeout=8)
        data   = resp.json()
        items  = []
        cutoff = datetime.now(tz=timezone.utc) - self.max_article_age
        for post in data.get("results", []):
            published = datetime.fromisoformat(post["published_at"].replace("Z", "+00:00"))
            if published < cutoff:
                continue
            item = self._score_headline(
                post["title"], "CryptoPanic", post["url"], published, is_macro=False
            )
            items.append(item)
        return items

    # ─── Sentiment scoring ────────────────────────────────────────────────────

    def _score_headline(
        self, title: str, source: str, url: str,
        published: datetime, is_macro: bool = False
    ) -> NewsItem:
        title_lower = title.lower()
        bull_score  = 0
        bear_score  = 0
        matched     = []

        for kw in BULLISH_KEYWORDS:
            if kw in title_lower:
                weight = 2 if kw in STRONG_KEYWORDS else 1
                bull_score += weight
                matched.append(f"+{kw}")

        for kw in BEARISH_KEYWORDS:
            if kw in title_lower:
                weight = 2 if kw in STRONG_KEYWORDS else 1
                bear_score += weight
                matched.append(f"-{kw}")

        total = bull_score + bear_score
        if total == 0:
            score = 0.0
            label = "NEUTRAL"
        else:
            score = (bull_score - bear_score) / total
            label = "BULLISH" if score > 0.1 else ("BEARISH" if score < -0.1 else "NEUTRAL")

        return NewsItem(
            title            = title,
            source           = source,
            url              = url,
            published        = published,
            sentiment_score  = round(score, 3),
            sentiment_label  = label,
            keywords_matched = matched,
            is_macro         = is_macro,
        )

    def _aggregate_sentiment(self, items: list) -> float:
        if not items:
            return 0.0
        now          = datetime.now(tz=timezone.utc)
        total_weight = 0
        weighted_sum = 0
        for item in items:
            if item.published:
                age_hours   = (now - item.published).total_seconds() / 3600
                time_weight = max(0.1, 2.718 ** (-age_hours / 3))
            else:
                time_weight = 0.5
            weighted_sum  += item.sentiment_score * time_weight
            total_weight  += time_weight
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _is_cache_valid(self) -> bool:
        if not self._last_fetch or not self._cached_report:
            return False
        return (datetime.now(tz=timezone.utc) - self._last_fetch) < self.cache_duration

    def _parse_date(self, entry) -> Optional[datetime]:
        try:
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        except Exception:
            pass
        return datetime.now(tz=timezone.utc)

    @staticmethod
    def _score_to_label(score: float) -> str:
        if score >  0.3: return "STRONG BULLISH"
        if score >  0.1: return "BULLISH"
        if score < -0.3: return "STRONG BEARISH"
        if score < -0.1: return "BEARISH"
        return "NEUTRAL"

    def _log_report(self, report: SentimentReport):
        log.info(
            f"SENTIMENT | Composite: {report.composite_score:+.3f} [{report.sentiment_label}] | "
            f"Crypto: {report.news_score:+.3f} ({report.news_count} articles) | "
            f"Macro: {report.macro_score:+.3f} ({report.macro_count} articles) | "
            f"F&G: {report.fear_greed_value} ({report.fear_greed_label})"
        )
        if report.top_headlines:
            log.info("Top headlines:")
            for h in report.top_headlines[:4]:
                log.info(f"  • {h[:90]}")
