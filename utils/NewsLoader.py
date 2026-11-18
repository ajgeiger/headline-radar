"""Base class for incremental news loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod

from utils.Config import Config
from utils.Database import Database
from utils.HttpClient import HttpClient
from utils.LoadResult import LoadResult


class NewsLoader(ABC):
    """Shared run/log/watermark flow around a source-specific fetch."""

    source_type: str

    def __init__(self, database: Database, http: HttpClient, config: Config) -> None:
        self.database = database
        self.http = http
        self.config = config

    @property
    def source_key(self) -> str:
        """Key used in the watermarks table. Override if a loader has sub-sources."""
        return self.source_type

    def run(self) -> LoadResult:
        """Fetch new items, insert them, and record the load run.

        Articles are committed as they are inserted. The high-water mark is
        only advanced after a successful fetch so a retry can resume cleanly.
        """
        run_id = self.database.start_load_run(self.source_type)
        try:
            result = self.load_new()
        except Exception as exc:
            self.database.finish_load_run(
                run_id,
                status="failed",
                error=str(exc),
            )
            raise

        status = "success" if result.ok else "failed"
        self.database.finish_load_run(
            run_id,
            status=status,
            records_fetched=result.records_fetched,
            records_inserted=result.records_inserted,
            records_duplicate=result.records_duplicate,
            high_water_mark=result.high_water_mark,
            error=result.error,
        )
        if result.ok and result.high_water_mark is not None:
            self.database.set_watermark(self.source_key, result.high_water_mark)
        return result

    @abstractmethod
    def load_new(self) -> LoadResult:
        """Pull new records for this source and insert them."""
