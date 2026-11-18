"""One RSS or Atom feed to poll."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RssFeed:
    """Display name and URL of one RSS or Atom feed."""

    name: str
    url: str
