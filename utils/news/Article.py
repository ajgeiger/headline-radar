"""One article record, shared by every news source."""

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256


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
        """Stable id for the article: SHA-256 of the trimmed URL."""
        return sha256(self.url.strip().encode("utf-8")).hexdigest()
