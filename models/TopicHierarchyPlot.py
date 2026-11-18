"""Draw a dendrogram that shows which topics are most alike."""

import numpy as np
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity


class TopicHierarchyPlot:
    """Build a Plotly dendrogram from a fitted TopicModel.

    Topics are compared by the cosine distance between their c-TF-IDF
    keyword vectors and merged with Ward linkage. The keywords shown at a
    merge point are the top words of the merged topics, weighted by size.
    They are a rough guide for review, not a refit.
    """

    LINE_COLOR = "#6B7280"
    LEAF_COLOR = "#1D4ED8"
    MERGE_COLOR = "#111827"
    ROW_SPACING = 10
    MERGE_KEYWORDS = 3

    def __init__(self, topic_model):
        self.model = topic_model

    def build(self, top_n_topics: int = 30, labeled_merges: int = 0) -> go.Figure:
        """Return the figure for the top_n_topics largest topics."""
        self.topics = self._largest_topics(top_n_topics)
        self.rows = [self.model.topic_ids_.index(topic) for topic in self.topics]
        self.merge_tree = self._link_topics()
        self.leaf_order = dendrogram(self.merge_tree, no_plot=True)

        figure = go.Figure()
        self._add_branches(figure)
        self._add_leaves(figure)
        self._add_merge_points(figure, labeled_merges)
        self._style(figure)
        return figure

    def _largest_topics(self, top_n_topics: int) -> list[int]:
        """Pick the largest real topics (no outliers) and sort them by id."""
        sizes = self.model.topic_sizes_
        real_topics = [topic for topic in self.model.topic_ids_ if topic >= 0]
        largest = sorted(real_topics, key=lambda topic: -sizes[topic])[:top_n_topics]
        if len(largest) < 2:
            raise ValueError("At least two topics are needed to draw a hierarchy.")
        return sorted(largest)

    def _link_topics(self) -> np.ndarray:
        """Cluster the topics by keyword distance with Ward linkage."""
        distances = 1 - cosine_similarity(self.model.c_tf_idf_[self.rows])
        distances = np.clip(distances, 0, 2)
        np.fill_diagonal(distances, 0)
        return linkage(squareform(distances, checks=False), method="ward", optimal_ordering=True)

    def _leaf_position(self, leaf: int) -> float:
        """Vertical position of a leaf, matching scipy's dendrogram layout."""
        order = self.leaf_order["leaves"].index(leaf)
        return self.ROW_SPACING / 2 + self.ROW_SPACING * order

    def _leaf_text(self, leaf: int) -> str:
        topic = self.topics[leaf]
        size = self.model.topic_sizes_[topic]
        return f"{topic}: {self.model.topic_name(topic)}  (n={size:,})"

    def _add_branches(self, figure: go.Figure) -> None:
        """Draw the lines of the tree."""
        for x_values, y_values in zip(self.leaf_order["dcoord"], self.leaf_order["icoord"]):
            figure.add_trace(go.Scatter(
                x=x_values, y=y_values, mode="lines",
                line=dict(color=self.LINE_COLOR, width=2),
                hoverinfo="skip", showlegend=False,
            ))

    def _add_leaves(self, figure: go.Figure) -> None:
        """Draw one hoverable dot per topic."""
        leaves = range(len(self.topics))
        hover = [
            f"Topic {self.topics[leaf]}<br>{self.model.topic_name(self.topics[leaf])}"
            f"<br>Documents: {self.model.topic_sizes_[self.topics[leaf]]:,}"
            for leaf in leaves
        ]
        figure.add_trace(go.Scatter(
            x=[0] * len(self.topics), y=[self._leaf_position(leaf) for leaf in leaves],
            mode="markers", marker=dict(size=5, color=self.LEAF_COLOR),
            text=hover, hovertemplate="%{text}<extra></extra>", showlegend=False,
        ))

    def _add_merge_points(self, figure: go.Figure, labeled_merges: int) -> None:
        """Draw a dot at every merge, with the combined keywords on hover.

        The last labeled_merges merges (the ones nearest the root) also show
        their keywords as text on the chart.
        """
        topic_count = len(self.topics)
        members = {leaf: [leaf] for leaf in range(topic_count)}
        centers = {leaf: self._leaf_position(leaf) for leaf in range(topic_count)}
        x_values, y_values, hover, labels = [], [], [], []

        for step, (left, right, distance, _) in enumerate(self.merge_tree):
            node = topic_count + step
            members[node] = members[int(left)] + members[int(right)]
            centers[node] = (centers[int(left)] + centers[int(right)]) / 2

            merged_topics = [self.topics[leaf] for leaf in members[node]]
            keywords, document_count = self._merged_keywords(merged_topics)
            is_labeled = step >= topic_count - 1 - labeled_merges

            x_values.append(float(distance))
            y_values.append(centers[node])
            hover.append(f"{keywords}<br>{len(merged_topics)} topics, {document_count:,} documents")
            labels.append(keywords if labeled_merges and is_labeled else "")

        figure.add_trace(go.Scatter(
            x=x_values, y=y_values,
            mode="markers+text" if labeled_merges else "markers",
            marker=dict(size=7, color=self.MERGE_COLOR),
            text=labels, textposition="top right",
            hovertext=hover, hovertemplate="%{hovertext}<extra></extra>", showlegend=False,
        ))

    def _merged_keywords(self, topics: list[int]) -> tuple[str, int]:
        """Return the top keywords of a group of topics and its total size."""
        sizes = np.array([self.model.topic_sizes_[topic] for topic in topics])
        rows = [self.model.topic_ids_.index(topic) for topic in topics]
        scores = self.model.c_tf_idf_[rows].multiply(sizes[:, None]).sum(axis=0)
        scores = np.asarray(scores).ravel()

        vocabulary = self.model.vectorizer.get_feature_names_out()
        best = np.argsort(-scores, kind="stable")[: self.MERGE_KEYWORDS]
        keywords = ", ".join(vocabulary[i] for i in best if scores[i] > 0)
        return keywords or "No descriptive terms", int(sizes.sum())

    def _style(self, figure: go.Figure) -> None:
        """Put topic names on the right and the root of the tree on the left."""
        topic_count = len(self.topics)
        tick_positions = [self.ROW_SPACING / 2 + self.ROW_SPACING * order for order in range(topic_count)]
        tick_labels = [self._leaf_text(leaf) for leaf in self.leaf_order["leaves"]]

        figure.update_layout(
            title="Topic similarity hierarchy",
            template="plotly_white",
            width=1050,
            height=max(450, 28 * topic_count),
            showlegend=False,
            margin=dict(l=170, r=380, t=65, b=50),
            xaxis=dict(title="Topic distance", autorange="reversed", zeroline=False),
            yaxis=dict(
                tickmode="array", tickvals=tick_positions, ticktext=tick_labels,
                side="right", automargin=True, range=[0, self.ROW_SPACING * topic_count],
            ),
        )
