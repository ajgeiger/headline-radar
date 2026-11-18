"""Add topic ids to a pandas DataFrame by matching on text."""

import pandas as pd

from models.TopicModel import TopicModel


class TopicJoiner:
    """Join a table of (text, topic_id) pairs onto any pandas DataFrame.

    Safety rules:
        - Each text appears once in the lookup, so the join never adds rows.
        - Blank or missing text always gets topic -2.
        - The same text with two different topic ids raises an error.
        - Any nonblank text missing from the lookup raises an error. That
          means the model was fit on older data and should be refit.
    """

    KEY_COLUMN = "__topic_join_key"

    def __init__(self, text_column: str = "text", topic_column: str = "topic_id"):
        self.text_column = text_column
        self.topic_column = topic_column

    def join(self, rows: pd.DataFrame, topics: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of rows with a new topic_column, in the same row order.

        Args:
            rows: DataFrame that has text_column.
            topics: DataFrame with text_column and topic_column, for example
                the fit texts next to the ids from fit_transform.
        """
        self._check_columns(rows, topics)
        lookup = self._build_lookup(topics)
        key_column = self._free_column_name(rows)

        keyed_rows = rows.assign(**{key_column: self._join_keys(rows[self.text_column])})
        keyed_lookup = lookup.rename(columns={self.text_column: key_column})

        # validate="many_to_one" makes pandas raise if a key is repeated in the lookup.
        labeled = keyed_rows.merge(keyed_lookup, on=key_column, how="left", validate="many_to_one")
        labeled.index = rows.index
        labeled = labeled.drop(columns=key_column)

        is_blank = TopicModel.clean_texts(rows[self.text_column]).eq("").to_numpy()
        labeled.loc[is_blank, self.topic_column] = TopicModel.BLANK_TOPIC

        self._raise_if_unmatched(labeled)
        labeled[self.topic_column] = labeled[self.topic_column].astype(int)
        return labeled

    def _check_columns(self, rows: pd.DataFrame, topics: pd.DataFrame) -> None:
        if self.text_column not in rows.columns:
            raise ValueError(f"The rows need a '{self.text_column}' column.")
        if self.topic_column in rows.columns:
            raise ValueError(f"The rows already have a '{self.topic_column}' column.")
        missing = {self.text_column, self.topic_column} - set(topics.columns)
        if missing:
            raise ValueError(f"The topics table is missing columns: {sorted(missing)}")

    @staticmethod
    def _join_keys(texts: pd.Series) -> pd.Series:
        """Exact text as a string. Missing values stay missing so they never match."""
        return texts.map(lambda text: None if pd.isna(text) else str(text))

    def _build_lookup(self, topics: pd.DataFrame) -> pd.DataFrame:
        """Check the topics table and return one row per exact text."""
        lookup = topics[[self.text_column, self.topic_column]].copy()
        cleaned_texts = TopicModel.clean_texts(lookup[self.text_column]).to_numpy()
        lookup[self.topic_column] = self._check_topic_ids(lookup[self.topic_column], cleaned_texts)

        conflicts = lookup.groupby(cleaned_texts)[self.topic_column].nunique()
        if conflicts.gt(1).any():
            raise ValueError("The same text has more than one topic id.")

        lookup[self.text_column] = self._join_keys(lookup[self.text_column])
        lookup = lookup[lookup[self.text_column].notna()]
        return lookup.drop_duplicates(self.text_column)

    @staticmethod
    def _check_topic_ids(topic_ids: pd.Series, cleaned_texts) -> pd.Series:
        """Topic ids must be whole numbers >= -2, and -2 must mean blank text."""
        topic_ids = pd.to_numeric(topic_ids, errors="raise")
        if topic_ids.isna().any() or (topic_ids < TopicModel.BLANK_TOPIC).any():
            raise ValueError("Every topic id must be a whole number of -2 or more.")
        if (topic_ids != topic_ids.astype(int)).any():
            raise ValueError("Every topic id must be a whole number of -2 or more.")

        is_blank = cleaned_texts == ""
        is_blank_topic = topic_ids.eq(TopicModel.BLANK_TOPIC).to_numpy()
        if (is_blank != is_blank_topic).any():
            raise ValueError("Topic -2 must be used for blank texts and only for blank texts.")
        return topic_ids.astype(int)

    def _free_column_name(self, rows: pd.DataFrame) -> str:
        """Pick a temporary column name that does not clash with existing columns."""
        name = self.KEY_COLUMN
        while name in rows.columns:
            name += "_"
        return name

    def _raise_if_unmatched(self, labeled: pd.DataFrame) -> None:
        if labeled[self.topic_column].isna().any():
            raise ValueError("Some texts were not in the data the model was fit on. Refit the model.")
