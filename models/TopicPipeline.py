"""Run the full topic model on the articles in the database and save the results."""

from pathlib import Path

import pandas as pd

from models.TextEmbedder import TextEmbedder
from models.TopicExplorer import TopicExplorer
from models.TopicModel import TopicModel
from utils.storage import Database


class TopicPipeline:
    """Load articles, fit a TopicModel, and write tables and charts to an output folder.

    Files written to output_folder:
        document_topics.csv     one row per article with its topic
        topic_info.csv          one row per topic with size, keywords, and examples
        fit_diagnostics.csv     cluster sizes before and after the duplicate vote
        topic_hierarchy.html    interactive dendrogram of topic similarity
        topic_hierarchy.png     the same dendrogram as an image
        topic_explorer.html     document map, topic sizes, and topic table
    """

    def __init__(
        self,
        database_path: str | Path = "data/headline_radar.db",
        output_folder: str | Path = "outputs",
        model_folder: str | Path = "data/topic_model",
        min_topic_size: int = 30,
        hierarchy_topics: int = 30,
    ):
        self.database_path = Path(database_path)
        self.output_folder = Path(output_folder)
        self.model_folder = Path(model_folder)
        self.min_topic_size = min_topic_size
        self.hierarchy_topics = hierarchy_topics

    def run(self) -> TopicModel:
        """Run every step and return the fitted model."""
        self.output_folder.mkdir(parents=True, exist_ok=True)

        articles = self.load_articles()
        print(f"Loaded {len(articles):,} articles from {self.database_path}")

        model = TopicModel(min_topic_size=self.min_topic_size, embedder=TextEmbedder())
        articles["topic_id"] = model.fit_transform(articles["text"])
        articles["topic_name"] = [model.topic_name(topic) for topic in articles["topic_id"]]

        model.save(self.model_folder)
        print(f"Saved the model to {self.model_folder}")

        self.write_tables(model, articles)
        self.write_hierarchy(model)
        self.write_explorer(model, articles)
        self.print_summary(model, articles)
        return model

    def load_articles(self) -> pd.DataFrame:
        """Read the articles and build one text per article from its title and description."""
        database = Database(self.database_path)
        try:
            articles = pd.DataFrame(database.read_articles())
        finally:
            database.close()

        if articles.empty:
            raise ValueError("The database has no articles. Run python -m de.load_news first.")
        articles["text"] = [
            self.combine_title_and_description(title, description)
            for title, description in zip(articles["title"], articles["description"])
        ]
        return articles

    @staticmethod
    def combine_title_and_description(title: str, description: str | None) -> str:
        """Join the title and description into one sentence-like text."""
        title = title.strip()
        description = (description or "").strip()
        if not description:
            return title
        separator = " " if title.endswith((".", "!", "?")) else ". "
        return f"{title}{separator}{description}"

    def write_tables(self, model: TopicModel, articles: pd.DataFrame) -> None:
        """Save the per-article topics, the topic summary, and the fit diagnostics as CSV."""
        columns = ["id", "published_at", "source", "source_type", "text", "topic_id", "topic_name"]
        articles[columns].to_csv(self.output_folder / "document_topics.csv", index=False)

        topic_info = model.get_topic_info()
        topic_info["top_words"] = topic_info["top_words"].str.join(", ")
        topic_info["example_documents"] = topic_info["example_documents"].str.join(" | ")
        topic_info.to_csv(self.output_folder / "topic_info.csv", index=False)

        model.get_fit_diagnostics().to_csv(self.output_folder / "fit_diagnostics.csv", index=False)

    def write_hierarchy(self, model: TopicModel) -> None:
        """Save the topic dendrogram as HTML and PNG. The PNG needs kaleido and Chrome."""
        figure = model.visualize_hierarchy(top_n_topics=self.hierarchy_topics)
        figure.write_html(self.output_folder / "topic_hierarchy.html", include_plotlyjs=True)
        figure.write_image(self.output_folder / "topic_hierarchy.png", scale=2)

    def write_explorer(self, model: TopicModel, articles: pd.DataFrame) -> None:
        """Save the interactive topic explorer page."""
        explorer = TopicExplorer(model)
        explorer.save(self.output_folder / "topic_explorer.html", articles["text"].tolist(), articles["topic_id"])

    def print_summary(self, model: TopicModel, articles: pd.DataFrame) -> None:
        topic_count = sum(1 for topic in model.topic_ids_ if topic >= 0)
        outlier_share = (articles["topic_id"] == TopicModel.OUTLIER_TOPIC).mean()
        print(f"Found {topic_count} topics. {outlier_share:.0%} of articles are outliers (topic -1).")
        print(f"Results are in {self.output_folder}")
