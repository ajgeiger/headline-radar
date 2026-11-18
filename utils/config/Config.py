"""Project settings in one plain class."""

from dataclasses import dataclass, field
from pathlib import Path

from utils.news import RssFeed

# utils/config/Config.py -> project root is three folders up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Feeds that were reachable when the project was set up.
DEFAULT_RSS_FEEDS = [
    RssFeed("BBC News", "https://feeds.bbci.co.uk/news/rss.xml"),
    RssFeed("NPR News", "https://feeds.npr.org/1001/rss.xml"),
    RssFeed("CNBC Top Stories", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    RssFeed("Guardian Business", "https://www.theguardian.com/uk/business/rss"),
    RssFeed("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines"),
    RssFeed("NYT World", "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"),
    RssFeed("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
]


@dataclass
class Config:
    """Database path, HTTP settings, GDELT options, and the RSS feed list."""

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
    rss_feeds: list[RssFeed] = field(default_factory=lambda: list(DEFAULT_RSS_FEEDS))
