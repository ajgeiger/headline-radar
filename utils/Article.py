"""Canonical article record used by every news source."""

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256


def article_id_from_url(url: str) -> str:
    """Return a stable id for an article: SHA-256 of the trimmed URL."""
    normalized = url.strip()
    return sha256(normalized.encode("utf-8")).hexdigest()


@dataclass
class Article:
    """One news item ready to be stored in the database."""

    url: str
    title: str | None
    description: str | None
    source: str
    source_type: str
    published_at: datetime | None
    language: str | None
    fetched_at: datetime

    @property
    def id(self) -> str:
        return article_id_from_url(self.url)
