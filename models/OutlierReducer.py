"""Reassign outlier texts to their nearest coherent topic using embedding centroids."""

import numpy as np
from sklearn.preprocessing import normalize


class OutlierReducer:
    """Assign outlier embeddings (-1) to the nearest topic centroid above a threshold.

    Topic centroids are built from assigned embeddings (topic_id >= 0) and L2-normalized
    so dot products equal cosine similarities. Blank texts (-2) are never changed, and
    outliers whose best centroid similarity falls below min_similarity remain -1.
    """

    OUTLIER_TOPIC = -1
    BLANK_TOPIC = -2
    DEFAULT_CHUNK_SIZE = 5000

    def __init__(self, min_similarity: float = 0.45, chunk_size: int = DEFAULT_CHUNK_SIZE):
        if not -1.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be between -1.0 and 1.0.")
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1.")

        self.min_similarity = float(min_similarity)
        self.chunk_size = int(chunk_size)
        self.topic_ids_: list[int] = []
        self.centroids_: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray, topic_ids: np.ndarray) -> "OutlierReducer":
        """Compute one L2-normalized centroid per real topic (topic_id >= 0)."""
        embeddings = np.asarray(embeddings, dtype=np.float32)
        topics = np.asarray(topic_ids, dtype=np.int32)
        self._check_inputs(embeddings, topics)

        real_topics = sorted({int(topic) for topic in topics if topic >= 0})
        if not real_topics:
            self.topic_ids_ = []
            self.centroids_ = np.empty((0, embeddings.shape[1]), dtype=np.float32)
            return self

        unit_embeddings = normalize(embeddings, norm="l2", axis=1).astype(np.float32, copy=False)
        topic_to_row = {topic: row for row, topic in enumerate(real_topics)}
        centroids = np.zeros((len(real_topics), embeddings.shape[1]), dtype=np.float32)

        assigned_mask = topics >= 0
        assigned_rows = np.array([topic_to_row[int(topic)] for topic in topics[assigned_mask]], dtype=np.int32)
        np.add.at(centroids, assigned_rows, unit_embeddings[assigned_mask])

        self.topic_ids_ = real_topics
        self.centroids_ = normalize(centroids, norm="l2", axis=1).astype(np.float32, copy=False)
        return self

    def reduce(
        self,
        embeddings: np.ndarray,
        topic_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reassign -1 rows to the best matching centroid at or above min_similarity.

        Returns (updated_topics, reassigned_mask, best_similarities).
        """
        if self.centroids_ is None:
            raise ValueError("Call fit before reduce.")

        embeddings = np.asarray(embeddings, dtype=np.float32)
        updated_topics = np.asarray(topic_ids, dtype=np.int32).copy()
        self._check_inputs(embeddings, updated_topics)

        reassigned_mask = np.zeros(len(updated_topics), dtype=bool)
        best_similarities = np.zeros(len(updated_topics), dtype=np.float32)

        outlier_mask = updated_topics == self.OUTLIER_TOPIC
        if not outlier_mask.any() or len(self.topic_ids_) == 0:
            return updated_topics, reassigned_mask, best_similarities

        outlier_embeddings = normalize(embeddings[outlier_mask], norm="l2", axis=1).astype(np.float32, copy=False)
        best_indices, max_similarities = self._nearest_centroids(outlier_embeddings)

        accepted = max_similarities >= self.min_similarity
        topic_lookup = np.asarray(self.topic_ids_, dtype=np.int32)
        matched_topics = np.where(accepted, topic_lookup[best_indices], self.OUTLIER_TOPIC)

        updated_topics[outlier_mask] = matched_topics
        reassigned_mask[outlier_mask] = accepted
        best_similarities[outlier_mask] = max_similarities
        return updated_topics, reassigned_mask, best_similarities

    def remap_topics(self, topic_mapping: dict[int, int]) -> None:
        """Update stored topic ids after topics are renumbered."""
        if self.centroids_ is None or not self.topic_ids_:
            return

        new_ids = [int(topic_mapping[topic]) for topic in self.topic_ids_]
        order = np.argsort(new_ids, kind="stable")
        self.topic_ids_ = [new_ids[index] for index in order]
        self.centroids_ = self.centroids_[order]

    def _nearest_centroids(self, outlier_embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (centroid_index, cosine_similarity) for each outlier row."""
        total = len(outlier_embeddings)
        best_indices = np.empty(total, dtype=np.int32)
        best_similarities = np.empty(total, dtype=np.float32)
        centroids_t = self.centroids_.T

        for start in range(0, total, self.chunk_size):
            end = start + self.chunk_size
            similarities = outlier_embeddings[start:end] @ centroids_t
            indices = np.argmax(similarities, axis=1)
            best_indices[start:end] = indices
            best_similarities[start:end] = similarities[np.arange(len(indices)), indices]

        return best_indices, best_similarities

    @staticmethod
    def _check_inputs(embeddings: np.ndarray, topic_ids: np.ndarray) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array.")
        if topic_ids.ndim != 1:
            raise ValueError("topic_ids must be a 1D array.")
        if embeddings.shape[0] != topic_ids.shape[0]:
            raise ValueError("embeddings and topic_ids must have the same number of rows.")
