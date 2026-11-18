"""Small text cleanup helpers used by the loaders."""

from html import unescape

from utils.text.HtmlTextExtractor import HtmlTextExtractor


class TextCleaner:
    """Turn raw feed values into clean text, or None when nothing is left."""

    @staticmethod
    def strip_html(value: str | None) -> str | None:
        """Return plain text from a short HTML snippet, or None if it is empty."""
        if value is None:
            return None
        parser = HtmlTextExtractor()
        parser.feed(value)
        text = unescape(" ".join(parser.parts))
        text = " ".join(text.split())
        return text or None

    @staticmethod
    def blank_to_none(value: str | None) -> str | None:
        """Trim a string and return None if it is missing or only whitespace."""
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None
