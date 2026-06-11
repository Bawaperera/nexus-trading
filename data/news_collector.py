"""
NEXUS — Real-time News & Sentiment Collector
Pulls crypto news from multiple FREE sources and scores market sentiment.

Sources (all free, no paid API needed to start):
  1. CoinDesk RSS        → top crypto journalism
  2. CoinTelegraph RSS   → market news and analysis
  3. Fear & Greed Index  → alternative.me (market-wide sentiment score)
  4. CryptoPanic API     → (optional) crypto-specific news with source voting

Sentiment scoring:
  - Keyword-based NLP on headlines (fast, no GPU needed)
  - Outputs score from -1.0 (extreme fear/bearish) to +1.0 (extreme greed/bullish)
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


# ─── Sentiment keywords (hand-curated for crypto) ────────────────────────────

BULLISH_KEYWORDS = {
    # Strong positive (weight 2)
    "surge", "soar", "rally", "breakout", "bull", "bullish", "moon", "ath",
    "all-time high", "record", "milestone", "institutional", "adoption",
    "approval", "etf approved", "upgrade", "outperform", "massive gains",
    # Mild positive (weight 1)
    "rise", "gain", "up", "growth", "positive", "recovery", "rebound",
    "support", "buy", "accumulate", "strong", "momentum", "confidence",
    "optimistic", "partnership", "launch", "integration", "expansion",
}

BEARISH_KEYWORDS = {
    # Strong negative (weight 2)
    "crash", "collapse", "plunge", "dump", "bear", "bearish", "hack",
    "exploit", "scam", "fraud", "ban", "crackdown", "bankrupt", "liquidation",
    "all-time low", "regulatory action", "sec", "lawsuit", "jail",
    # Mild negative (weight 1)
    "fall", "drop", "decline", "down", "loss", "fear", "concern", "risk",
    "warning", "caution", "sell", "resistance", "correction", "retreat",
    "uncertainty", "delay", "cancel", "suspend",
}

# Weight overrides for strong keywords
STRONG_KEYWORDS = {
    "crash", "collapse", "plunge", "hack", "exploit", "fraud", "ban",
    "surge", "soar", "ath", "all-time high", "etf approved", "breakout",
}


@dataclass
class NewsItem:
    title: str
    source: str
    url: str
    published: datetime
    sentiment_score: float  # -1.0 to +1.0
    sentiment_label: str    # BULLISH / BEARISH / NEUTRAL
    keywords_matched: list = field(default_factory=list)


@dataclass
class SentimentReport:
    timestamp: datetime
    composite_score: float       # -1.0 to +1.0
    sentiment_label: str
    fear_greed_value: int        # 0-100
    fear_greed_label: str
    news_score: float
    news_count: int
    top_headlines: list
    signal_weight: float         # how much to trust this sentiment (0.0-1.0)


class NewsCollector:
    """
    Collects real-time crypto news and generates a sentiment score
    that feeds into the NEXUS Signal Engine.

    Usage:
        collector = NewsCollector()
        report = collector.get_sentiment_report()
        print(report.composite_score)  # -0.4 = bearish
    """

    RSS_FEEDS = {
        "CoinDesk":        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "CoinTelegraph":   "https://cointelegraph.com/rss",
        "Decrypt":         "https://decrypt.co/feed",
        "The Block":       "https://www.theblock.co/rss.xml",
    }

    FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"
    CRYPTOPANIC_URL = "https://cryptopanic.com/api/free/v2/posts/?public=true&currencies=BTC"

    def __init__(
        self,
        cryptopanic_token: str = None,
        cache_duration_minutes: int = 15,
        max_article_age_hours: int = 6,
    ):
        """
        Args:
            cryptopanic_token: Optional free token from cryptopanic.com
                               Sign up free at https://cryptopanic.com/developers/api/
            cache_duration_minutes: How long to cache results before re-fetching
            max_article_age_hours: Ignore articles older than this
        """
        self.cryptopanic_token    = cryptopanic_token
        self.cache_duration       = timedelta(minutes=cache_duration_minutes)
        self.max_article_age      = timedelta(hours=max_article_age_hours)

        # Cache
        self._last_fetch:   Optional[datetime] = None
        self._cached_report: Optional[SentimentReport] = None

        # History buffer (last 50 reports for trend analysis)
        self.history: deque = deque(maxlen=50)

        log.info(f"NewsCollector ready | {len(self.RSS_FEEDS)} RSS feeds | "
                 f"{'CryptoPanic enabled' if cryptopanic_token else 'CryptoPanic disabled (no token)'}")

    # ─── Public API ───────────────────────────────────────────────────────────

    def get_sentiment_report(self, force_refresh: bool = False) -> SentimentReport:
        """
        Get the current sentiment report.
        Uses cache unless force_refresh=True or cache expired.
        """
        if not force_refresh and self._is_cache_valid():
            return self._cached_report

        log.info("Fetching fresh news and sentiment data...")

        news_items   = self._fetch_all_news()
        fear_greed   = self._fetch_fear_greed()
        news_score   = self._aggregate_news_sentiment(news_items)

        # Composite score: 70% news sentiment + 30% fear/greed
        fg_normalized = (fear_greed["value"] - 50) / 50  # convert 0-100 to -1 to +1
        composite     = (0.70 * news_score) + (0.30 * fg_normalized)
        composite     = max(-1.0, min(1.0, composite))

        # Signal weight: how reliable is this sentiment reading?
        # Higher if more articles + fear/greed extreme
        signal_weight = min(1.0, len(news_items) / 15) * (1 + abs(fg_normalized) * 0.3)
        signal_weight = max(0.3, min(1.0, signal_weight))

        report = SentimentReport(
            timestamp         = datetime.now(tz=timezone.utc),
            composite_score   = round(composite, 4),
            sentiment_label   = self._score_to_label(composite),
            fear_greed_value  = fear_greed["value"],
            fear_greed_label  = fear_greed["label"],
            news_score        = round(news_score, 4),
            news_count        = len(news_items),
            top_headlines     = [n.title for n in news_items[:5]],
            signal_weight     = round(signal_weight, 3),
        )

        self._cached_report = report
        self._last_fetch    = datetime.now(tz=timezone.utc)
        self.history.append(report)

        self._log_report(report)
        return report

    def get_sentiment_score(self) -> float:
        """Quick access — returns just the composite score (-1.0 to +1.0)."""
        return self.get_sentiment_report().composite_score

    def get_sentiment_trend(self) -> float:
        """
        Returns the direction of sentiment change over recent reports.
        Positive = improving sentiment, Negative = deteriorating.
        """
        if len(self.history) < 3:
            return 0.0
        recent = [r.composite_score for r in list(self.history)[-5:]]
        return recent[-1] - recent[0]

    # ─── Data fetchers ────────────────────────────────────────────────────────

    def _fetch_all_news(self) -> list:
        """Fetch and score news from all RSS feeds + CryptoPanic."""
        all_items = []
        cutoff    = datetime.now(tz=timezone.utc) - self.max_article_age

        for source, url in self.RSS_FEEDS.items():
            try:
                feed  = feedparser.parse(url)
                count = 0
                for entry in feed.entries[:20]:
                    published = self._parse_date(entry)
                    if published and published < cutoff:
                        continue

                    item = self._score_headline(entry.title, source, entry.link, published)
                    all_items.append(item)
                    count += 1

                log.debug(f"{source}: {count} recent articles")

            except Exception as e:
                log.warning(f"RSS fetch failed ({source}): {e}")

        # CryptoPanic (optional — better crypto-specific news)
        if self.cryptopanic_token:
            try:
                cp_items = self._fetch_cryptopanic()
                all_items.extend(cp_items)
                log.debug(f"CryptoPanic: {len(cp_items)} articles")
            except Exception as e:
                log.warning(f"CryptoPanic fetch failed: {e}")

        # Sort by recency
        all_items.sort(key=lambda x: x.published or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        log.info(f"Total news articles collected: {len(all_items)}")
        return all_items

    def _fetch_fear_greed(self) -> dict:
        """Fetch the Crypto Fear & Greed Index."""
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
            return {"value": 50, "label": "Neutral"}  # neutral fallback

    def _fetch_cryptopanic(self) -> list:
        """Fetch news from CryptoPanic API (requires free token)."""
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
                post["title"], "CryptoPanic", post["url"], published
            )
            items.append(item)
        return items

    # ─── Sentiment scoring ────────────────────────────────────────────────────

    def _score_headline(self, title: str, source: str, url: str, published) -> NewsItem:
        """Score a headline using keyword matching."""
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
            title             = title,
            source            = source,
            url               = url,
            published         = published,
            sentiment_score   = round(score, 3),
            sentiment_label   = label,
            keywords_matched  = matched,
        )

    def _aggregate_news_sentiment(self, items: list) -> float:
        """
        Aggregate individual article scores into one news sentiment score.
        Recent articles weighted higher (recency bias).
        """
        if not items:
            return 0.0

        now    = datetime.now(tz=timezone.utc)
        total_weight = 0
        weighted_sum = 0

        for item in items:
            if item.published:
                age_hours = (now - item.published).total_seconds() / 3600
                # Exponential decay — 1-hour-old article gets weight 1.0, 6-hour gets ~0.37
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
            f"SENTIMENT REPORT | Score: {report.composite_score:+.3f} [{report.sentiment_label}] | "
            f"Fear/Greed: {report.fear_greed_value} ({report.fear_greed_label}) | "
            f"News: {report.news_count} articles | Weight: {report.signal_weight:.2f}"
        )
        if report.top_headlines:
            log.info("Top headlines:")
            for h in report.top_headlines[:3]:
                log.info(f"  • {h[:80]}")
