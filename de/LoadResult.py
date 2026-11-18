"""Summary of a single load run."""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class LoadResult:
    """Summary of a single load run."""

    records_fetched: int
    records_inserted: int
    records_duplicate: int
    high_water_mark: datetime | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None
