"""Build an interactive HTML page for browsing the topics of a fitted TopicModel."""

import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from umap import UMAP


class TopicExplorer:
    """Make a single HTML file with a document map, topic sizes, and a topic table.

    The map places every unique text in 2D (UMAP of its embedding) and colors
    it by topic. Hovering shows the text. The bar chart shows how many texts
    each topic has, with its top words on hover. The table lists every topic
    with its keywords and example texts.
    """

    OUTLIER_COLOR = "#C4C4C4"
    MAX_HOVER_CHARACTERS = 200
    MAX_WORDS_IN_TABLE = 10

    def __init__(self, topic_model, colored_topics: int = 20, random_state: int = 42):
        """
        Args:
            topic_model: A fitted TopicModel.
            colored_topics: The largest topics get their own color and legend entry.
                Smaller topics share one gray "Other topics" group.
            random_state: Seed for the 2D UMAP layout.
        """
        self.model = topic_model
        self.colored_topics = colored_topics
        self.random_state = random_state

    def save(self, path: str | Path, texts: list[str], topic_ids) -> Path:
        """Write the explorer page to path and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.build_page(texts, topic_ids), encoding="utf-8")
        return path

    def build_page(self, texts: list[str], topic_ids) -> str:
        """Return the full HTML page as a string. Plotly is embedded, so it works offline."""
        figure = self.build_figure(texts, topic_ids)
        chart = figure.to_html(include_plotlyjs=True, full_html=False)
        return (
            "<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n"
            "<title>Topic explorer</title>\n"
            "<style>\n"
            "body { font-family: Arial, sans-serif; margin: 24px; color: #111827; }\n"
            "table { border-collapse: collapse; width: 100%; font-size: 14px; }\n"
            "th, td { border-bottom: 1px solid #E5E7EB; padding: 6px 8px; text-align: left; vertical-align: top; }\n"
            "th { background: #F3F4F6; }\n"
            "</style>\n</head>\n<body>\n"
            "<h1>Topic explorer</h1>\n"
            f"{self._summary_html(topic_ids)}\n{chart}\n"
            "<h2>Topics</h2>\n"
            f"{self._topic_table_html()}\n"
            "</body>\n</html>\n"
        )

    def build_figure(self, texts: list[str], topic_ids) -> go.Figure:
        """Return a figure with the document map on the left and topic sizes on the right."""
        documents = self._unique_documents(texts, topic_ids)
        documents[["x", "y"]] = self._layout_2d(documents["text"].tolist())

        figure = make_subplots(
            rows=1, cols=2, column_widths=[0.68, 0.32], horizontal_spacing=0.08,
            subplot_titles=("Document map (each dot is a unique text)", "Topic sizes"),
        )
        self._add_document_map(figure, documents)
        self._add_topic_sizes(figure)
        figure.update_layout(template="plotly_white", height=800, legend=dict(font=dict(size=10)))
        figure.update_xaxes(showticklabels=False, row=1, col=1)
        figure.update_yaxes(showticklabels=False, row=1, col=1)
        return figure

    def _unique_documents(self, texts: list[str], topic_ids) -> pd.DataFrame:
        """One row per unique text, so repeated headlines are drawn once."""
        documents = pd.DataFrame({
            "text": self.model.clean_texts(texts),
            "topic_id": np.asarray(topic_ids, dtype=int),
        })
        documents = documents[documents["topic_id"] != self.model.BLANK_TOPIC]
        return documents.drop_duplicates("text").reset_index(drop=True)

    def _layout_2d(self, texts: list[str]) -> np.ndarray:
        """Embed the texts and squeeze the embeddings down to 2D with UMAP."""
        embeddings = self.model.embed(texts)
        reducer = UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric="cosine",
                       random_state=self.random_state, n_jobs=1, compatibility_layout=True)
        return reducer.fit_transform(embeddings)

    def _group_name(self, topic_id: int, colored: set[int]) -> str:
        """Legend group for a topic: its own name, "Other topics", or outliers."""
        if topic_id == self.model.OUTLIER_TOPIC:
            return "Outliers (no topic)"
        if topic_id in colored:
            return f"{topic_id}: {self.model.topic_name(topic_id)}"
        return "Other topics"

    def _add_document_map(self, figure: go.Figure, documents: pd.DataFrame) -> None:
        """Draw one scatter trace per legend group. Click the legend to hide a group."""
        real_topics = [topic for topic in self.model.topic_ids_ if topic >= 0]
        colored = set(real_topics[: self.colored_topics])
        documents["group"] = [self._group_name(topic, colored) for topic in documents["topic_id"]]
        documents["hover"] = [self._hover_text(text, topic)
                              for text, topic in zip(documents["text"], documents["topic_id"])]

        # Draw gray groups first so colored topics sit on top.
        gray_groups = ["Outliers (no topic)", "Other topics"]
        colored_groups = [self._group_name(topic, colored) for topic in sorted(colored)]
        for group in gray_groups + colored_groups:
            points = documents[documents["group"] == group]
            if points.empty:
                continue
            marker = dict(size=4, opacity=0.7)
            if group in gray_groups:
                marker.update(color=self.OUTLIER_COLOR, opacity=0.35)
            figure.add_trace(go.Scattergl(
                x=points["x"], y=points["y"], mode="markers", name=group, marker=marker,
                text=points["hover"], hovertemplate="%{text}<extra></extra>",
            ), row=1, col=1)

    def _hover_text(self, text: str, topic_id: int) -> str:
        """Topic name on the first line, the (shortened) text below it."""
        if len(text) > self.MAX_HOVER_CHARACTERS:
            text = text[: self.MAX_HOVER_CHARACTERS].rstrip() + "..."
        return f"<b>Topic {topic_id}: {html.escape(self.model.topic_name(topic_id))}</b><br>{html.escape(text)}"

    def _add_topic_sizes(self, figure: go.Figure) -> None:
        """Horizontal bars, largest topic at the top. Outliers are left out."""
        info = self.model.get_topic_info()
        info = info[info["topic_id"] >= 0].head(self.colored_topics).iloc[::-1]
        hover = [
            f"<b>{html.escape(name)}</b><br>{count:,} texts<br>{html.escape(', '.join(words[:8]))}"
            for name, count, words in zip(info["topic_name"], info["document_count"], info["top_words"])
        ]
        figure.add_trace(go.Bar(
            x=info["document_count"], y=info["topic_name"], orientation="h",
            marker=dict(color="#1D4ED8"), customdata=hover, hovertemplate="%{customdata}<extra></extra>",
            showlegend=False,
        ), row=1, col=2)
        figure.update_yaxes(tickfont=dict(size=10), row=1, col=2)
        figure.update_xaxes(title_text="Number of texts", row=1, col=2)

    def _summary_html(self, topic_ids) -> str:
        """One line with the total, topic count, and outlier share."""
        topic_ids = np.asarray(topic_ids, dtype=int)
        topic_count = sum(1 for topic in self.model.topic_ids_ if topic >= 0)
        outlier_share = float(np.mean(topic_ids == self.model.OUTLIER_TOPIC)) if len(topic_ids) else 0.0
        return (
            f"<p>{len(topic_ids):,} texts, {topic_count} topics, "
            f"{outlier_share:.0%} outliers (topic -1).</p>"
        )

    def _topic_table_html(self) -> str:
        """HTML table with every topic, its size, top words, and example texts."""
        rows = []
        for _, topic in self.model.get_topic_info().iterrows():
            words = ", ".join(topic["top_words"][: self.MAX_WORDS_IN_TABLE])
            examples = "<br>".join(html.escape(text) for text in topic["example_documents"])
            rows.append(
                f"<tr><td>{topic['topic_id']}</td><td>{topic['document_count']:,}</td>"
                f"<td>{html.escape(topic['topic_name'])}</td><td>{html.escape(words)}</td>"
                f"<td>{examples}</td></tr>"
            )
        header = "<tr><th>Topic</th><th>Texts</th><th>Name</th><th>Top words</th><th>Examples</th></tr>"
        return f"<table>\n{header}\n" + "\n".join(rows) + "\n</table>"
