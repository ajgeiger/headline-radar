"""Entry point: python -m models.run_topic_model"""

import argparse
from pathlib import Path

from models.TopicPipeline import TopicPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit the topic model on stored articles and save the results.")
    parser.add_argument("--database", type=Path, default=Path("data/headline_radar.db"),
                        help="SQLite file with the articles table (default: data/headline_radar.db).")
    parser.add_argument("--output-folder", type=Path, default=Path("outputs"),
                        help="Where tables and charts are written (default: outputs).")
    parser.add_argument("--model-folder", type=Path, default=Path("data/topic_model"),
                        help="Where the fitted model is saved (default: data/topic_model).")
    parser.add_argument("--min-topic-size", type=int, default=30,
                        help="Smallest number of articles a topic may have (default: 30).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = TopicPipeline(
        database_path=args.database,
        output_folder=args.output_folder,
        model_folder=args.model_folder,
        min_topic_size=args.min_topic_size,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
