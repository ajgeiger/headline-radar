"""Collect the visible text from a fragment of HTML."""

from html.parser import HTMLParser


class HtmlTextExtractor(HTMLParser):
    """Keep the text between tags and drop the tags themselves."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        """Called by HTMLParser for every piece of text it finds."""
        self.parts.append(data)
