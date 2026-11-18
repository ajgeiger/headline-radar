"""Incremental loader for a small set of news RSS feeds."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from time import struct_time

import feedparser

from utils.Article import Article
from utils.RssFeed import RssFeed
from utils.Database import utc_now
from utils.LoadResult import LoadResult
from utils.NewsLoader import NewsLoader
from utils.text import blank_to_none, strip_html

logger = logging.getLogger("headline_radar")


def struct_time_to_utc(value: struct_time | None) -> datetime | None:
    """Convert a feedparser time tuple to a UTC datetime."""
    if value is None:
        return None
    return datetime(*value[:6], tzinfo=timezone.utc)


class RssLoader(NewsLoader):
    """Fetch the current items from each configured RSS feed.

    RSS feeds only expose a short rolling window, so every run downloads the
    latest feed documents. Idempotent inserts (by URL hash) prevent duplicates.
    A high-water mark is stored as the latest published_at seen, for logging
    and later incremental checks, but it is not used to drop items.
    """

    source_type = "rss"

    def load_new(self) -> LoadResult:
        fetched_at = utc_now()
        articles: list[Article] = []
        for feed in self.config.rss_feeds:
            articles.extend(self._fetch_feed(feed, fetched_at))

        inserted, duplicates = self.database.insert_articles(articles)
        published_times = [a.published_at for a in articles if a.published_at is not None]
        previous = self.database.get_watermark(self.source_key)
        if published_times:
            latest = max(published_times)
            high_water = latest if previous is None else max(previous, latest)
        else:
            high_water = previous or fetched_at

        logger.info(
            "RSS fetched=%s inserted=%s duplicates=%s feeds=%s",
            len(articles),
            inserted,
            duplicates,
            len(self.config.rss_feeds),
        )
        return LoadResult(len(articles), inserted, duplicates, high_water)

    def _fetch_feed(self, feed: RssFeed, fetched_at: datetime) -> list[Article]:
        try:
            response = self.http.get(feed.url)
        except Exception as exc:
            logger.error("Failed to fetch RSS feed %s (%s): %s", feed.name, feed.url, exc)
            return []

        parsed = feedparser.parse(response.content)
        if parsed.bozo and not parsed.entries:
            logger.error(
                "Could not parse RSS feed %s: %s",
                feed.name,
                getattr(parsed, "bozo_exception", "unknown parse error"),
            )
            return []

        articles: list[Article] = []
        seen_urls: set[str] = set()
        for entry in parsed.entries:
            article = self._entry_to_article(entry, feed, fetched_at)
            if article is None or article.url in seen_urls:
                continue
            seen_urls.add(article.url)
            articles.append(article)

        logger.info("RSS %s: %s items", feed.name, len(articles))
        return articles

    def _entry_to_article(
        self,
        entry: feedparser.FeedParserDict,
        feed: RssFeed,
        fetched_at: datetime,
    ) -> Article | None:
        url = blank_to_none(entry.get("link") or entry.get("id"))
        if url is None:
            logger.warning("Skipping %s entry with no URL: %s", feed.name, entry.get("title"))
            return None

        raw_description = (
            entry.get("summary")
            or entry.get("description")
            or _first_content(entry)
        )
        language = blank_to_none(entry.get("language"))
        if language is None:
            language = blank_to_none(getattr(entry, "language", None))

        published = struct_time_to_utc(
            entry.get("published_parsed") or entry.get("updated_parsed")
        )

        return Article(
            url=url,
            title=blank_to_none(entry.get("title")),
            description=strip_html(raw_description) if raw_description else None,
            source=feed.name,
            source_type=self.source_type,
            published_at=published,
            language=language,
            fetched_at=fetched_at,
        )


def _first_content(entry: feedparser.FeedParserDict) -> str | None:
    contents = entry.get("content") or []
    if not contents:
        return None
    first = contents[0]
    if isinstance(first, dict):
        return first.get("value")
    return None
