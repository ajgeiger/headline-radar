"""Small text helpers used by the loaders."""

from html import unescape

from utils.TextExtractor import TextExtractor


def strip_html(value: str | None) -> str | None:
    """Return plain text from a short HTML snippet, or None if the input is empty."""
    if value is None:
        return None
    parser = TextExtractor()
    parser.feed(value)
    text = unescape(" ".join(parser.parts))
    text = " ".join(text.split())
    return text or None


def blank_to_none(value: str | None) -> str | None:
    """Treat missing or whitespace-only strings as NULL."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None
