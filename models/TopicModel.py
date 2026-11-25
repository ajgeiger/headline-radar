"""Find topics in any collection of short texts.

How it works:
    1. Embed each unique text with TextEmbedder.
    2. Reduce the embeddings to a few dimensions with UMAP.
    3. Group nearby texts into clusters with HDBSCAN.
    4. Give identical texts the same topic, then drop topics that became too small.
    5. Optionally reassign remaining outliers (-1) to the nearest topic centroid.
    6. Number topics by size, so topic 0 is the largest.
    7. Describe each topic with c-TF-IDF keywords and a few example texts.

Topic ids:
    -1 means the text did not fit any topic (an outlier).
    -2 means the text was blank. Blank texts are never sent to the model.
"""

import re
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from hdbscan import HDBSCAN, approximate_predict
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from umap import UMAP

from models.OutlierReducer import OutlierReducer
from models.TextEmbedder import TextEmbedder
from models.TopicHierarchyPlot import TopicHierarchyPlot


class TopicModel:
    """Embed texts, cluster them, and describe each cluster as a topic.

    Identical texts (after trimming spaces) always get the same topic id,
    so results can be joined back to the source data by text.
    """

    OUTLIER_TOPIC = -1
    BLANK_TOPIC = -2

    MAX_EXAMPLE_CANDIDATES = 500
    EXAMPLES_PER_TOPIC = 3
    WORDS_IN_TOPIC_NAME = 4

    EMBEDDER_FOLDER = "embedder"
    MODEL_FILE = "topic_model.joblib"

    def __init__(
        self,
        min_topic_size: int = 20,
        top_n_words: int = 10,
        batch_size: int = 64,
        random_state: int = 42,
        min_samples: int | None = 5,
        cluster_selection_method: str = "eom",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        umap_n_neighbors: int = 15,
        cluster_selection_epsilon: float = 0.0,
        outlier_min_similarity: float | None = 0.45,
        *,
        embedder=None,
        reducer=None,
        clusterer=None,
        vectorizer=None,
        outlier_reducer=None,
    ):
        """Set up the model. Any of the building blocks can be passed in.

        Args:
            min_topic_size: Smallest number of texts a topic may have.
            top_n_words: Number of keywords kept per topic.
            batch_size: Number of texts embedded at once.
            random_state: Seed for UMAP and for sampling example texts.
            min_samples: HDBSCAN min_samples. None uses min_topic_size.
            cluster_selection_method: HDBSCAN method, "eom" or "leaf".
            embedding_model_name: Hugging Face model used by the default embedder.
            umap_n_neighbors: Number of neighbors used by the default UMAP reducer.
            cluster_selection_epsilon: HDBSCAN cluster_selection_epsilon.
            outlier_min_similarity: Cosine similarity threshold for OutlierReducer, or None to skip.
            embedder: Object with encode(texts, batch_size, show_progress, as_numpy) and save(folder).
            reducer: Object with fit_transform and transform, such as UMAP.
            clusterer: HDBSCAN model. It needs prediction_data=True to label new texts.
            vectorizer: scikit-learn CountVectorizer used for topic keywords.
            outlier_reducer: Optional OutlierReducer instance used after clustering.
        """
        self._check_settings(
            min_topic_size,
            top_n_words,
            batch_size,
            min_samples,
            cluster_selection_method,
            umap_n_neighbors,
            cluster_selection_epsilon,
            outlier_min_similarity,
        )

        self.min_topic_size = min_topic_size
        self.top_n_words = top_n_words
        self.batch_size = batch_size
        self.random_state = random_state
        self.min_samples = min_samples
        self.cluster_selection_method = cluster_selection_method
        self.embedding_model_name = embedding_model_name
        self.umap_n_neighbors = umap_n_neighbors
        self.cluster_selection_epsilon = float(cluster_selection_epsilon)
        self.outlier_min_similarity = None if outlier_min_similarity is None else float(outlier_min_similarity)

        self.embedder = embedder if embedder is not None else TextEmbedder(embedding_model_name)
        self.reducer = reducer if reducer is not None else self._default_reducer()
        self.clusterer = clusterer if clusterer is not None else self._default_clusterer()
        self.vectorizer = vectorizer if vectorizer is not None else self._default_vectorizer()
        self.outlier_reducer = outlier_reducer

        # Filled in by fit.
        self.document_topics_ = None
        self.c_tf_idf_ = None
        self.topic_labels_ = {}
        self.reassigned_outliers_ = None
        self.reassigned_counts_ = {}
        self.reassigned_texts_ = set()
        self.outlier_reducer_ = None

    @staticmethod
    def _check_settings(
        min_topic_size,
        top_n_words,
        batch_size,
        min_samples,
        cluster_selection_method,
        umap_n_neighbors,
        cluster_selection_epsilon,
        outlier_min_similarity,
    ) -> None:
        """Fail early on settings that cannot work."""
        if min_topic_size < 2:
            raise ValueError("min_topic_size must be at least 2.")
        if top_n_words < 1:
            raise ValueError("top_n_words must be at least 1.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if min_samples is not None and min_samples < 1:
            raise ValueError("min_samples must be at least 1.")
        if cluster_selection_method not in {"eom", "leaf"}:
            raise ValueError("cluster_selection_method must be 'eom' or 'leaf'.")
        if umap_n_neighbors < 2:
            raise ValueError("umap_n_neighbors must be at least 2.")
        if cluster_selection_epsilon < 0:
            raise ValueError("cluster_selection_epsilon must be non-negative.")
        if outlier_min_similarity is not None and not -1.0 <= outlier_min_similarity <= 1.0:
            raise ValueError("outlier_min_similarity must be between -1.0 and 1.0.")

    def _default_reducer(self) -> UMAP:
        """UMAP settings that work well for sentence embeddings."""
        return UMAP(
            n_neighbors=self.umap_n_neighbors,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=self.random_state,
            low_memory=True,
            n_jobs=1,
            # Keep the classic UMAP layout. Newer releases plan to change the default.
            compatibility_layout=True,
        )

    def _default_clusterer(self) -> HDBSCAN:
        """HDBSCAN settings. prediction_data=True lets transform label new texts."""
        return HDBSCAN(
            min_cluster_size=self.min_topic_size,
            min_samples=self.min_samples,
            metric="euclidean",
            cluster_selection_method=self.cluster_selection_method,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            prediction_data=True,
        )

    @staticmethod
    def _default_vectorizer() -> CountVectorizer:
        """Count single words and two-word phrases, skipping English stop words."""
        return CountVectorizer(stop_words="english", ngram_range=(1, 2))

    @staticmethod
    def clean_texts(texts: Iterable) -> pd.Series:
        """Turn any input into a Series of trimmed strings. Missing values become ""."""
        return (
            pd.Series(list(texts), dtype="object")
            .fillna("")
            .astype(str)
            .str.strip()
            .reset_index(drop=True)
        )

    def fit(self, texts: Iterable) -> "TopicModel":
        """Learn topics from the texts and return the fitted model."""
        self.fit_transform(texts)
        return self

    def fit_transform(self, texts: Iterable) -> np.ndarray:
        """Learn topics and return one topic id per input text.

        Every text must be nonblank. Give blank texts topic -2 yourself before
        or after calling this method.
        """
        texts = self.clean_texts(texts)
        self._check_fit_input(texts)
        self.topic_labels_ = {}

        # text_codes[i] is the position of texts[i] inside unique_texts.
        text_codes, unique_texts = pd.factorize(texts, sort=False)
        embeddings = self.embed(unique_texts.tolist(), show_progress=True)

        # Reduce every row, duplicates included, so repeated texts keep their weight.
        reduced = self.reducer.fit_transform(embeddings[text_codes])
        cluster_ids = np.asarray(self.clusterer.fit_predict(reduced), dtype=np.int32)
        if cluster_ids.shape != (len(texts),):
            raise ValueError("The clusterer did not return one label per text.")

        cluster_ids_after_vote = self._vote_on_duplicates(text_codes, cluster_ids, len(unique_texts))
        self.raw_cluster_sizes_ = self._count_values(cluster_ids)
        self.voted_cluster_sizes_ = self._count_values(cluster_ids_after_vote)

        kept_cluster_ids = self._drop_small_clusters(cluster_ids_after_vote)
        initial_cluster_to_topic = self._number_topics_by_size(kept_cluster_ids)
        initial_topics = np.array(
            [initial_cluster_to_topic.get(int(cluster), self.OUTLIER_TOPIC) for cluster in kept_cluster_ids],
            dtype=np.int32,
        )

        self._assign_final_topics(
            embeddings=embeddings,
            text_codes=text_codes,
            initial_cluster_to_topic=initial_cluster_to_topic,
            initial_topics=initial_topics,
        )

        self._remember_known_texts(text_codes, unique_texts)
        self.topic_sizes_ = self._count_values(self.document_topics_)
        self._describe_topics(texts)
        return self.document_topics_.copy()

    def _assign_final_topics(
        self,
        embeddings: np.ndarray,
        text_codes: np.ndarray,
        initial_cluster_to_topic: dict[int, int],
        initial_topics: np.ndarray,
    ) -> None:
        """Apply optional outlier reduction and renumber topics by final size."""
        self.reassigned_outliers_ = np.zeros(len(initial_topics), dtype=bool)
        self.reassigned_counts_ = {}
        self.outlier_reducer_ = None

        use_reducer = self.outlier_reducer is not None or self.outlier_min_similarity is not None
        if not use_reducer or not initial_cluster_to_topic:
            self.cluster_to_topic_ = initial_cluster_to_topic
            self.document_topics_ = initial_topics
            return

        first_row_per_text = np.unique(text_codes, return_index=True)[1]
        unique_initial_topics = initial_topics[first_row_per_text]

        active_reducer = (
            self.outlier_reducer
            if self.outlier_reducer is not None
            else OutlierReducer(min_similarity=self.outlier_min_similarity)
        )
        active_reducer.fit(embeddings, unique_initial_topics)
        reduced_unique_topics, reassigned_unique, _ = active_reducer.reduce(embeddings, unique_initial_topics)

        reduced_topics = reduced_unique_topics[text_codes]
        self.reassigned_outliers_ = reassigned_unique[text_codes]

        old_to_new = self._renumber_mapping_by_size(reduced_topics)
        self.cluster_to_topic_ = {
            cluster: old_to_new[topic] for cluster, topic in initial_cluster_to_topic.items()
        }
        self.document_topics_ = np.array(
            [old_to_new[int(topic)] if topic >= 0 else self.OUTLIER_TOPIC for topic in reduced_topics],
            dtype=np.int32,
        )
        active_reducer.remap_topics(old_to_new)
        self.outlier_reducer_ = active_reducer
        if self.reassigned_outliers_.any():
            self.reassigned_counts_ = self._count_values(self.document_topics_[self.reassigned_outliers_])

    def _check_fit_input(self, texts: pd.Series) -> None:
        """Fit needs enough nonblank texts to form at least one topic."""
        if texts.empty or texts.eq("").any():
            raise ValueError("Fit on nonblank texts only. Give blank texts topic -2.")
        if len(texts) < self.min_topic_size:
            raise ValueError(f"At least {self.min_topic_size} nonblank texts are required.")

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed texts with the model's embedder. Returns one row per text."""
        embeddings = self.embedder.encode(
            texts, batch_size=self.batch_size, show_progress=show_progress, as_numpy=True
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
            raise ValueError("The embedder returned an unexpected shape.")
        return embeddings

    @staticmethod
    def _count_values(values: np.ndarray) -> dict[int, int]:
        """Return {value: number of times it appears}."""
        return {int(value): int(count) for value, count in pd.Series(values).value_counts().items()}

    @staticmethod
    def _vote_on_duplicates(text_codes: np.ndarray, cluster_ids: np.ndarray, unique_count: int) -> np.ndarray:
        """Give every copy of a text the cluster that most of its copies received.

        Ties go to the lowest cluster id. Returns one cluster id per input row.
        """
        votes = (
            pd.DataFrame({"text_code": text_codes, "cluster_id": cluster_ids})
            .groupby(["text_code", "cluster_id"], sort=False)
            .size()
            .reset_index(name="votes")
            .sort_values(["text_code", "votes", "cluster_id"], ascending=[True, False, True])
            .drop_duplicates("text_code")
        )
        winner_per_text = np.full(unique_count, TopicModel.OUTLIER_TOPIC, dtype=np.int32)
        winner_per_text[votes["text_code"].to_numpy()] = votes["cluster_id"].to_numpy()
        return winner_per_text[text_codes]

    def _drop_small_clusters(self, cluster_ids: np.ndarray) -> np.ndarray:
        """Move clusters below min_topic_size to the outlier topic.

        HDBSCAN already enforces the size limit, but the duplicate vote can
        shrink a cluster afterwards, so the limit is applied again here.
        """
        cluster_ids = cluster_ids.copy()
        sizes = pd.Series(cluster_ids[cluster_ids >= 0]).value_counts()
        small_clusters = sizes.index[sizes < self.min_topic_size].to_numpy()
        is_small = np.isin(cluster_ids, small_clusters)

        self.pruned_document_count_ = int(is_small.sum())
        cluster_ids[is_small] = self.OUTLIER_TOPIC
        return cluster_ids

    @staticmethod
    def _number_topics_by_size(cluster_ids: np.ndarray) -> dict[int, int]:
        """Map each kept cluster id to a topic id: 0 for the largest, 1 for the next, and so on."""
        sizes = pd.Series(cluster_ids[cluster_ids >= 0]).value_counts()
        largest_first = sorted(sizes.index, key=lambda cluster: (-int(sizes[cluster]), int(cluster)))
        return {int(cluster): topic for topic, cluster in enumerate(largest_first)}

    @staticmethod
    def _renumber_mapping_by_size(topic_ids: np.ndarray) -> dict[int, int]:
        """Map existing topic ids (>= 0) to 0..N-1 sorted by descending document count."""
        sizes = pd.Series(topic_ids[topic_ids >= 0]).value_counts()
        largest_first = sorted(sizes.index, key=lambda topic: (-int(sizes[topic]), int(topic)))
        return {int(old_topic): new_topic for new_topic, old_topic in enumerate(largest_first)}

    def _remember_known_texts(self, text_codes: np.ndarray, unique_texts: pd.Index) -> None:
        """Store {text: topic id} and reassigned text set from the fit data."""
        first_row_per_text = np.unique(text_codes, return_index=True)[1]
        topics = self.document_topics_[first_row_per_text]
        reassigned = self.reassigned_outliers_[first_row_per_text]
        unique_list = unique_texts.tolist()
        self.known_text_topics_ = dict(zip(unique_list, topics.tolist()))
        self.reassigned_texts_ = {text for text, flag in zip(unique_list, reassigned) if bool(flag)}

    @staticmethod
    def _clean_for_keywords(text: str) -> str:
        """Replace punctuation with spaces so "rate/cut" counts as "rate cut".

        This only affects keywords. Embeddings always use the original text.
        """
        return re.sub(r"[^\w]+", " ", text).strip()

    def _describe_topics(self, texts: pd.Series) -> None:
        """Compute keywords and example texts for every topic, outliers included."""
        topic_documents = (
            pd.DataFrame({"topic_id": self.document_topics_, "text": texts})
            .groupby("topic_id", sort=True)["text"]
            .agg(" ".join)
        )
        self.topic_ids_ = [int(topic) for topic in topic_documents.index]
        cleaned = [self._clean_for_keywords(document) for document in topic_documents.tolist()]

        word_counts = self._fit_word_counts(cleaned)
        self.c_tf_idf_ = self._compute_c_tf_idf(word_counts)

        self.topic_words_ = {}
        self.example_documents_ = {}
        for row, topic in enumerate(self.topic_ids_):
            topic_vector = self.c_tf_idf_.getrow(row)
            self.topic_words_[topic] = self._top_words(topic_vector)
            self.example_documents_[topic] = self._pick_examples(topic, texts, topic_vector)

    def _fit_word_counts(self, topic_documents: list[str]) -> sparse.csr_matrix:
        """Count words per topic. Fall back to a looser vectorizer if nothing is left.

        Stop-word removal can leave an empty vocabulary on very short texts.
        In that case keep every word. If there are no words at all, use a
        single placeholder term so the rest of the pipeline still works.
        """
        try:
            return self.vectorizer.fit_transform(topic_documents).tocsr()
        except ValueError as error:
            if "empty vocabulary" not in str(error).lower():
                raise

        if any(topic_documents):
            self.vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")
            return self.vectorizer.fit_transform(topic_documents).tocsr()

        self.vectorizer = CountVectorizer(vocabulary={"no_descriptive_terms": 0})
        return self.vectorizer.transform(topic_documents).tocsr()

    def _compute_c_tf_idf(self, word_counts: sparse.csr_matrix) -> sparse.csr_matrix:
        """Class-based TF-IDF: treat each topic as one long document.

        tf  = share of the topic's words that are this word
        idf = log(1 + average words per topic / total count of this word)
        """
        word_totals = np.asarray(word_counts.sum(axis=0)).ravel()
        average_words_per_topic = float(np.asarray(word_counts.sum(axis=1)).mean())
        self.idf_weights_ = np.log1p(average_words_per_topic / np.maximum(word_totals, 1))

        c_tf_idf = self._weight_by_idf(word_counts).tocsr()
        c_tf_idf.eliminate_zeros()
        return c_tf_idf

    def _weight_by_idf(self, word_counts: sparse.spmatrix) -> sparse.spmatrix:
        """Turn raw counts into term frequencies and multiply by the stored idf."""
        term_frequency = normalize(word_counts.astype(np.float64), norm="l1", axis=1)
        return term_frequency @ sparse.diags(self.idf_weights_, format="csr")

    def _top_words(self, topic_vector: sparse.csr_matrix) -> list[tuple[str, float]]:
        """Return the top_n_words (word, score) pairs for one topic, best first."""
        vocabulary = self.vectorizer.get_feature_names_out()
        best = np.argsort(-topic_vector.data, kind="stable")[: self.top_n_words]
        return [(str(vocabulary[topic_vector.indices[i]]), float(topic_vector.data[i])) for i in best]

    def _pick_examples(self, topic: int, texts: pd.Series, topic_vector: sparse.csr_matrix) -> list[str]:
        """Return the texts whose keywords look most like the topic's keywords.

        Up to MAX_EXAMPLE_CANDIDATES unique texts are sampled to keep this fast.
        """
        candidates = texts[self.document_topics_ == topic].drop_duplicates()
        if len(candidates) > self.MAX_EXAMPLE_CANDIDATES:
            candidates = candidates.sample(self.MAX_EXAMPLE_CANDIDATES, random_state=self.random_state)
        candidates = candidates.tolist()

        cleaned = [self._clean_for_keywords(text) for text in candidates]
        candidate_vectors = self._weight_by_idf(self.vectorizer.transform(cleaned))
        similarity = cosine_similarity(candidate_vectors, topic_vector).ravel()
        best = np.argsort(-similarity, kind="stable")[: self.EXAMPLES_PER_TOPIC]
        return [candidates[i] for i in best]

    def get_topic_words(self, topic_id: int) -> list[tuple[str, float]]:
        """Return (word, score) pairs for one topic. Unknown ids return []."""
        self._check_is_fitted()
        return self.topic_words_.get(int(topic_id), [])

    def get_example_documents(self) -> dict[int, list[str]]:
        """Return {topic id: example texts}."""
        self._check_is_fitted()
        return self.example_documents_

    def set_topic_labels(self, labels: dict[int, str]) -> None:
        """Give topics short human-readable names, for example {0: "Interest rates"}."""
        self._check_is_fitted()
        unknown_ids = sorted(set(labels) - set(self.topic_ids_))
        if unknown_ids:
            raise ValueError(f"These topic ids do not exist: {unknown_ids}")
        if any(not isinstance(name, str) or not name.strip() for name in labels.values()):
            raise ValueError("Every label must be a nonblank string.")
        self.topic_labels_.update({int(topic): name.strip() for topic, name in labels.items()})

    def topic_name(self, topic_id: int) -> str:
        """Return the custom label if one was set, otherwise the top keywords."""
        if topic_id == self.OUTLIER_TOPIC:
            return "Outliers (no topic)"
        if topic_id in self.topic_labels_:
            return self.topic_labels_[topic_id]
        top_words = [word for word, _ in self.topic_words_.get(topic_id, [])[: self.WORDS_IN_TOPIC_NAME]]
        return ", ".join(top_words) or "No descriptive terms"

    def get_topic_info(self) -> pd.DataFrame:
        """One row per topic with its size, name, keywords, and example texts.

        needs_review is True for a real topic that ended up with no keywords.
        """
        self._check_is_fitted()
        reassigned_counts = getattr(self, "reassigned_counts_", {})
        rows = []
        for topic in self.topic_ids_:
            words = [word for word, _ in self.get_topic_words(topic)]
            rows.append({
                "topic_id": topic,
                "document_count": self.topic_sizes_[topic],
                "reassigned_count": reassigned_counts.get(topic, 0),
                "topic_name": f"{topic}: {self.topic_name(topic)}",
                "needs_review": topic >= 0 and not words,
                "top_words": words,
                "example_documents": self.example_documents_.get(topic, []),
            })
        return pd.DataFrame(rows)

    def get_fit_diagnostics(self) -> pd.DataFrame:
        """Compare cluster sizes from HDBSCAN, after the duplicate vote, and after outlier reduction.

        kept is False for outliers and for clusters dropped as too small.
        topic_id is the final topic for kept clusters, otherwise -1.
        """
        self._check_is_fitted()
        cluster_ids = sorted(set(self.raw_cluster_sizes_) | set(self.voted_cluster_sizes_))
        reassigned_counts = getattr(self, "reassigned_counts_", {})
        rows = []
        for cluster in cluster_ids:
            kept = cluster in self.cluster_to_topic_
            topic_id = self.cluster_to_topic_.get(cluster, self.OUTLIER_TOPIC)
            if cluster == self.OUTLIER_TOPIC:
                reassigned_count = 0
                final_count = self.topic_sizes_.get(self.OUTLIER_TOPIC, 0)
            elif kept:
                reassigned_count = reassigned_counts.get(topic_id, 0)
                final_count = self.topic_sizes_.get(topic_id, 0)
            else:
                reassigned_count = 0
                final_count = 0

            rows.append({
                "cluster_id": cluster,
                "hdbscan_count": self.raw_cluster_sizes_.get(cluster, 0),
                "count_after_vote": self.voted_cluster_sizes_.get(cluster, 0),
                "kept": kept,
                "topic_id": topic_id,
                "reassigned_count": reassigned_count,
                "final_count": final_count,
            })
        return pd.DataFrame(rows)

    def transform(self, texts: Iterable) -> np.ndarray:
        """Return one topic id per text.

        Blank texts get -2. Texts seen during fit get their fitted topic.
        New texts are embedded and labeled with HDBSCAN approximate_predict
        followed by OutlierReducer when fitted.
        """
        return self.transform_with_status(texts)["topic_id"].to_numpy()

    def transform_with_status(self, texts: Iterable) -> pd.DataFrame:
        """Like transform, but also say how each topic was found.

        Columns: text, topic_id, status, is_reassigned_outlier. status is
        "blank", "seen" (the text was in the fit data), or "new" (predicted).
        """
        self._check_is_fitted()
        texts = self.clean_texts(texts)

        is_blank = texts.eq("")
        is_seen = ~is_blank & texts.isin(self.known_text_topics_.keys())
        is_new = ~is_blank & ~is_seen

        topic_ids = np.full(len(texts), self.BLANK_TOPIC, dtype=np.int32)
        reassigned_flags = np.zeros(len(texts), dtype=bool)
        reassigned_texts = getattr(self, "reassigned_texts_", set())

        if is_seen.any():
            seen_mask = is_seen.to_numpy()
            topic_ids[seen_mask] = texts[is_seen].map(self.known_text_topics_).to_numpy(dtype=np.int32)
            reassigned_flags[seen_mask] = texts[is_seen].isin(reassigned_texts).to_numpy(dtype=bool)

        if is_new.any():
            new_mask = is_new.to_numpy()
            new_topics, new_reassigned = self._predict_new_texts_with_flags(texts[is_new])
            topic_ids[new_mask] = new_topics
            reassigned_flags[new_mask] = new_reassigned

        status = np.select([is_blank, is_seen], ["blank", "seen"], default="new")
        return pd.DataFrame({
            "text": texts,
            "topic_id": topic_ids,
            "status": status,
            "is_reassigned_outlier": reassigned_flags,
        })

    def _predict_new_texts(self, texts: pd.Series) -> np.ndarray:
        """Embed texts the model has not seen and place them in the nearest topic or -1."""
        topics, _ = self._predict_new_texts_with_flags(texts)
        return topics

    def _predict_new_texts_with_flags(self, texts: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Predict new texts with HDBSCAN approximate_predict and optional OutlierReducer."""
        text_codes, unique_texts = pd.factorize(texts, sort=False)
        embeddings = self.embed(unique_texts.tolist())
        reduced = self.reducer.transform(embeddings)
        try:
            cluster_ids, _ = approximate_predict(self.clusterer, reduced)
        except (AttributeError, ValueError) as error:
            raise ValueError("Labeling new texts needs an HDBSCAN model with prediction_data=True.") from error

        topics = np.array(
            [self.cluster_to_topic_.get(int(cluster), self.OUTLIER_TOPIC) for cluster in cluster_ids],
            dtype=np.int32,
        )
        reassigned = np.zeros(len(unique_texts), dtype=bool)
        outlier_reducer = getattr(self, "outlier_reducer_", None)
        if outlier_reducer is not None:
            topics, reassigned, _ = outlier_reducer.reduce(embeddings, topics)

        return topics[text_codes], reassigned[text_codes]

    def visualize_hierarchy(self, top_n_topics: int = 30, labeled_merges: int = 0):
        """Return a Plotly dendrogram of how similar the largest topics are.

        labeled_merges is how many of the top merge points show keyword text.
        """
        self._check_is_fitted()
        return TopicHierarchyPlot(self).build(top_n_topics=top_n_topics, labeled_merges=labeled_merges)

    def save(self, folder: str | Path) -> None:
        """Save the fitted model and its embedder to a folder."""
        self._check_is_fitted()
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        self.embedder.save(folder / self.EMBEDDER_FOLDER)
        state = {name: value for name, value in self.__dict__.items() if name != "embedder"}
        joblib.dump(state, folder / self.MODEL_FILE, compress=3)

    @classmethod
    def load(cls, folder: str | Path, device: str | None = None) -> "TopicModel":
        """Load a model saved with save.

        joblib uses pickle, so only load folders you created yourself.
        """
        folder = Path(folder)
        model = cls.__new__(cls)
        model.__dict__.update(joblib.load(folder / cls.MODEL_FILE))
        model.embedder = TextEmbedder.load(folder / cls.EMBEDDER_FOLDER, device=device)
        model._check_is_fitted()
        return model

    def _check_is_fitted(self) -> None:
        if self.document_topics_ is None or self.c_tf_idf_ is None:
            raise ValueError("Call fit or fit_transform first.")
