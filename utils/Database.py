"""SQLite database for articles, load-run logs, and high-water marks."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from utils.Article import Article


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY,
    title TEXT,
    description TEXT,
    url TEXT NOT NULL,
    source TEXT,
    source_type TEXT NOT NULL,
    published_at TEXT,
    fetched_at TEXT NOT NULL,
    language TEXT
);

CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles (published_at);
CREATE INDEX IF NOT EXISTS idx_articles_source_type ON articles (source_type);
CREATE INDEX IF NOT EXISTS idx_articles_url ON articles (url);

CREATE TABLE IF NOT EXISTS load_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    records_fetched INTEGER DEFAULT 0,
    records_inserted INTEGER DEFAULT 0,
    records_duplicate INTEGER DEFAULT 0,
    high_water_mark TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS watermarks (
    source_key TEXT PRIMARY KEY,
    high_water_mark TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _to_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _from_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def utc_now() -> datetime:
    """Current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


class Database:
    """Thin SQLite helper. One file, standard library only."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self.initialize()

    def initialize(self) -> None:
        """Create tables and indexes if they do not already exist."""
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    def insert_articles(self, articles: list[Article]) -> tuple[int, int]:
        """Insert articles, skipping ids that are already present.

        Returns (inserted_count, duplicate_count). Rows with a blank title or
        description are still stored; blanks are saved as NULL.
        """
        sql = """
            INSERT OR IGNORE INTO articles (
                id, title, description, url, source, source_type,
                published_at, fetched_at, language
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        inserted = 0
        duplicates = 0
        for article in articles:
            cursor = self._conn.execute(
                sql,
                (
                    article.id,
                    article.title,
                    article.description,
                    article.url,
                    article.source,
                    article.source_type,
                    _to_iso(article.published_at),
                    _to_iso(article.fetched_at),
                    article.language,
                ),
            )
            if cursor.rowcount == 1:
                inserted += 1
            else:
                duplicates += 1
        self._conn.commit()
        return inserted, duplicates

    def get_watermark(self, source_key: str) -> datetime | None:
        """Return the last successful high-water mark for a source, if any."""
        row = self._conn.execute(
            "SELECT high_water_mark FROM watermarks WHERE source_key = ?",
            (source_key,),
        ).fetchone()
        if row is None:
            return None
        return _from_iso(row["high_water_mark"])

    def set_watermark(self, source_key: str, mark: datetime) -> None:
        """Create or update the high-water mark for a source."""
        now = _to_iso(utc_now())
        self._conn.execute(
            """
            INSERT INTO watermarks (source_key, high_water_mark, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(source_key) DO UPDATE SET
                high_water_mark = excluded.high_water_mark,
                updated_at = excluded.updated_at
            """,
            (source_key, _to_iso(mark), now),
        )
        self._conn.commit()

    def start_load_run(self, source_type: str) -> int:
        """Insert a running load-run row and return its id."""
        cursor = self._conn.execute(
            """
            INSERT INTO load_runs (source_type, started_at, status)
            VALUES (?, ?, 'running')
            """,
            (source_type, _to_iso(utc_now())),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def finish_load_run(
        self,
        run_id: int,
        status: str,
        records_fetched: int = 0,
        records_inserted: int = 0,
        records_duplicate: int = 0,
        high_water_mark: datetime | None = None,
        error: str | None = None,
    ) -> None:
        """Mark a load run as finished, failed, or otherwise complete."""
        self._conn.execute(
            """
            UPDATE load_runs
               SET finished_at = ?,
                   status = ?,
                   records_fetched = ?,
                   records_inserted = ?,
                   records_duplicate = ?,
                   high_water_mark = ?,
                   error = ?
             WHERE id = ?
            """,
            (
                _to_iso(utc_now()),
                status,
                records_fetched,
                records_inserted,
                records_duplicate,
                _to_iso(high_water_mark),
                error,
                run_id,
            ),
        )
        self._conn.commit()

    def count_articles(self, source_type: str | None = None) -> int:
        """Count stored articles, optionally for one source type."""
        if source_type is None:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM articles").fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM articles WHERE source_type = ?",
                (source_type,),
            ).fetchone()
        return int(row["n"])

    def counts_by_source_type(self) -> dict[str, int]:
        """Return article counts grouped by source_type."""
        rows = self._conn.execute(
            """
            SELECT source_type, COUNT(*) AS n
              FROM articles
             GROUP BY source_type
             ORDER BY source_type
            """
        ).fetchall()
        return {row["source_type"]: int(row["n"]) for row in rows}

    def close(self) -> None:
        self._conn.close()
