"""A single RSS or Atom feed to poll."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RssFeed:
    """A single RSS or Atom feed to poll."""

    name: str
    url: str


def default_rss_feeds() -> list[RssFeed]:
    """Feeds confirmed reachable when this project was set up."""
    return [
        RssFeed("BBC News", "https://feeds.bbci.co.uk/news/rss.xml"),
        RssFeed("NPR News", "https://feeds.npr.org/1001/rss.xml"),
        RssFeed("CNBC Top Stories", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
        RssFeed("Guardian Business", "https://www.theguardian.com/uk/business/rss"),
        RssFeed("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines"),
        RssFeed("NYT World", "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"),
        RssFeed("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
    ]
