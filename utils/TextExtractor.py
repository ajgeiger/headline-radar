"""Collect visible text from a fragment of HTML."""

from html.parser import HTMLParser


class TextExtractor(HTMLParser):
    """Collect visible text from a fragment of HTML."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)
