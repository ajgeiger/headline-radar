"""Project configuration as a plain class."""

from dataclasses import dataclass, field
from pathlib import Path

from utils.RssFeed import RssFeed, default_rss_feeds


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    """Paths, HTTP settings, GDELT options, and the RSS feed list."""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    database_path: Path = PROJECT_ROOT / "data" / "headline_radar.db"
    user_agent: str = "HeadlineRadar/0.1 (portfolio project; educational use)"
    http_timeout_seconds: int = 30
    http_max_retries: int = 5
    # 15-minute GKG files. HTTP URLs in lastupdate.txt are rewritten to HTTPS
    # because the HTTP endpoint often returns 500.
    gdelt_lastupdate_url: str = "https://data.gdeltproject.org/gdeltv2/lastupdate.txt"
    gdelt_file_base_url: str = "https://data.gdeltproject.org/gdeltv2"
    gdelt_min_interval_seconds: float = 0.3
    gdelt_retry_after_seconds: float = 20.0
    gdelt_file_timeout_seconds: int = 90
    # Recent 404s are often just not published yet; older missing slots are skipped.
    gdelt_recent_grace_minutes: int = 45
    default_backfill_hours: int = 6
    rss_feeds: list[RssFeed] = field(default_factory=default_rss_feeds)
