"""Incremental loader for GDELT 2.0 15-minute GKG files."""

from __future__ import annotations

import logging
import re
import zipfile
from datetime import datetime, timedelta, timezone
from io import BytesIO

from utils.Article import Article
from utils.Database import utc_now
from utils.HttpError import HttpError
from de.FileNotReadyError import FileNotReadyError
from utils.LoadResult import LoadResult
from utils.NewsLoader import NewsLoader
from utils.text import blank_to_none, strip_html

logger = logging.getLogger("headline_radar")

GDELT_STAMP_FORMAT = "%Y%m%d%H%M%S"
GKG_SOURCE_NAME = 3
GKG_DOCUMENT_URL = 4
GKG_DATE = 1
GKG_TRANSLATION_INFO = 25
GKG_EXTRAS = 26
TITLE_RE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.IGNORECASE | re.DOTALL)
PUBLISHED_RE = re.compile(
    r"<PAGE_PRECISEPUBTIMESTAMP>(.*?)</PAGE_PRECISEPUBTIMESTAMP>",
    re.IGNORECASE | re.DOTALL,
)
FILE_STAMP_RE = re.compile(r"(\d{14})\.gkg\.csv\.zip", re.IGNORECASE)


def parse_gdelt_stamp(value: str | None) -> datetime | None:
    """Parse a GDELT YYYYMMDDHHMMSS timestamp as UTC."""
    if not value:
        return None
    text = value.strip()
    try:
        return datetime.strptime(text, GDELT_STAMP_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        logger.warning("Could not parse GDELT timestamp: %s", value)
        return None


def format_gdelt_stamp(value: datetime) -> str:
    """Format a UTC datetime as a GDELT file stamp."""
    return value.astimezone(timezone.utc).strftime(GDELT_STAMP_FORMAT)


def floor_to_15_minutes(value: datetime) -> datetime:
    """Round a datetime down to the previous 15-minute GDELT file slot."""
    value = value.astimezone(timezone.utc).replace(second=0, microsecond=0)
    minute = (value.minute // 15) * 15
    return value.replace(minute=minute)


def iter_gkg_stamps(start_exclusive: datetime, end_inclusive: datetime) -> list[datetime]:
    """Return 15-minute file stamps after start_exclusive through end_inclusive."""
    stamp = floor_to_15_minutes(start_exclusive)
    if stamp <= start_exclusive:
        stamp += timedelta(minutes=15)
    end = floor_to_15_minutes(end_inclusive)
    stamps: list[datetime] = []
    while stamp <= end:
        stamps.append(stamp)
        stamp += timedelta(minutes=15)
    return stamps


class GdeltLoader(NewsLoader):
    """Load new article metadata from GDELT 2.0 15-minute GKG files.

    Each file contains source domain, article URL, a DATE slot, optional
    precise publish time, and a title in the Extras ``PAGE_TITLE`` field.
    Descriptions are not included in GKG, so they are stored as NULL.

    The first run walks a backfill window (default: 6 hours). Later runs
    start after the stored high-water mark so only newer files are pulled.
    HTTP URLs from lastupdate.txt are fetched over HTTPS instead, because
    the HTTP file endpoint is unreliable.
    """

    source_type = "gdelt"

    def __init__(
        self,
        database,
        http,
        config,
        backfill_hours: int | None = None,
    ) -> None:
        super().__init__(database, http, config)
        self.backfill_hours = (
            backfill_hours
            if backfill_hours is not None
            else config.default_backfill_hours
        )

    def load_new(self) -> LoadResult:
        now = utc_now().replace(microsecond=0)
        watermark = self.database.get_watermark(self.source_key)
        if watermark is not None:
            start = watermark
            logger.info("GDELT watermark is %s; loading files after that", start)
        else:
            start = now - timedelta(hours=self.backfill_hours)
            logger.info(
                "No GDELT watermark; backfilling %s hours from %s",
                self.backfill_hours,
                start,
            )

        latest_listed = self._latest_listed_stamp()
        end = latest_listed or floor_to_15_minutes(now)
        stamps = iter_gkg_stamps(start, end)
        if not stamps:
            logger.info("No new GDELT GKG files to load")
            return LoadResult(0, 0, 0, watermark)

        fetched = 0
        inserted = 0
        duplicates = 0
        high_water = watermark
        grace = timedelta(minutes=self.config.gdelt_recent_grace_minutes)

        for stamp in stamps:
            try:
                articles = self._fetch_file(stamp)
            except FileNotReadyError:
                age = now - stamp
                if age <= grace:
                    logger.info(
                        "GDELT file %s is not available yet; will retry next run",
                        format_gdelt_stamp(stamp),
                    )
                    break
                logger.warning(
                    "GDELT file %s is missing and older than the grace period; skipping",
                    format_gdelt_stamp(stamp),
                )
                high_water = stamp
                self.database.set_watermark(self.source_key, high_water)
                continue

            batch_inserted, batch_duplicates = self.database.insert_articles(articles)
            fetched += len(articles)
            inserted += batch_inserted
            duplicates += batch_duplicates
            high_water = stamp
            self.database.set_watermark(self.source_key, high_water)
            logger.info(
                "GDELT file %s: fetched=%s inserted=%s duplicates=%s",
                format_gdelt_stamp(stamp),
                len(articles),
                batch_inserted,
                batch_duplicates,
            )

        return LoadResult(fetched, inserted, duplicates, high_water)

    def _latest_listed_stamp(self) -> datetime | None:
        """Read lastupdate.txt and return the latest GKG file stamp, if possible."""
        try:
            response = self.http.get(self.config.gdelt_lastupdate_url)
        except Exception as exc:
            logger.warning("Could not read GDELT lastupdate.txt: %s", exc)
            return None

        latest: datetime | None = None
        for line in response.text.splitlines():
            match = FILE_STAMP_RE.search(line)
            if match is None:
                continue
            parsed = parse_gdelt_stamp(match.group(1))
            if parsed is not None and (latest is None or parsed > latest):
                latest = parsed
        if latest is None:
            logger.warning("No GKG file stamp found in lastupdate.txt")
        else:
            logger.info("GDELT lastupdate lists GKG file %s", format_gdelt_stamp(latest))
        return latest

    def _fetch_file(self, stamp: datetime) -> list[Article]:
        url = f"{self.config.gdelt_file_base_url}/{format_gdelt_stamp(stamp)}.gkg.csv.zip"
        try:
            response = self.http.get(url)
        except HttpError as exc:
            if exc.status_code == 404:
                raise FileNotReadyError(url) from exc
            raise

        try:
            archive = zipfile.ZipFile(BytesIO(response.content))
        except zipfile.BadZipFile as exc:
            raise RuntimeError(f"GDELT file was not a valid zip: {url}") from exc

        names = archive.namelist()
        if not names:
            raise RuntimeError(f"GDELT zip was empty: {url}")

        fetched_at = utc_now()
        articles: list[Article] = []
        seen_urls: set[str] = set()
        with archive.open(names[0]) as handle:
            for raw_line in handle:
                article = self._row_to_article(
                    raw_line.decode("utf-8", "replace"),
                    file_stamp=stamp,
                    fetched_at=fetched_at,
                )
                if article is None or article.url in seen_urls:
                    continue
                seen_urls.add(article.url)
                articles.append(article)
        return articles

    def _row_to_article(
        self,
        line: str,
        file_stamp: datetime,
        fetched_at: datetime,
    ) -> Article | None:
        fields = line.rstrip("\n").split("\t")
        if len(fields) <= GKG_DOCUMENT_URL:
            return None
        url = blank_to_none(fields[GKG_DOCUMENT_URL])
        if url is None:
            return None

        extras = fields[GKG_EXTRAS] if len(fields) > GKG_EXTRAS else ""
        title_match = TITLE_RE.search(extras)
        raw_title = title_match.group(1) if title_match else None
        title = strip_html(raw_title) if raw_title else None

        published_at = None
        published_match = PUBLISHED_RE.search(extras)
        if published_match:
            published_at = parse_gdelt_stamp(published_match.group(1))
        if published_at is None and len(fields) > GKG_DATE:
            published_at = parse_gdelt_stamp(fields[GKG_DATE]) or file_stamp

        source = "gdelt"
        if len(fields) > GKG_SOURCE_NAME:
            source = blank_to_none(fields[GKG_SOURCE_NAME]) or "gdelt"

        language = "English"
        if len(fields) > GKG_TRANSLATION_INFO:
            translation = blank_to_none(fields[GKG_TRANSLATION_INFO])
            if translation:
                language = translation

        return Article(
            url=url,
            title=title,
            description=None,
            source=source,
            source_type=self.source_type,
            published_at=published_at,
            language=language,
            fetched_at=fetched_at,
        )

