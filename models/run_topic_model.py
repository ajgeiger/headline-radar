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
    parser.add_argument("--min-topic-size", type=int, default=20,
                        help="Smallest number of articles a topic may have (default: 20).")
    parser.add_argument("--min-samples", type=int, default=5,
                        help="HDBSCAN min_samples density parameter (default: 5).")
    parser.add_argument("--cluster-selection-method", choices=["eom", "leaf"], default="eom",
                        help="HDBSCAN cluster selection method (default: eom).")
    parser.add_argument("--umap-n-neighbors", type=int, default=15,
                        help="UMAP n_neighbors for dimensionality reduction (default: 15).")
    parser.add_argument("--outlier-min-similarity", type=float, default=0.45,
                        help="Minimum cosine similarity to reassign an outlier (default: 0.45).")
    parser.add_argument("--no-outlier-reduction", action="store_true",
                        help="Disable post-fit outlier reassignment.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outlier_min_similarity = None if args.no_outlier_reduction else args.outlier_min_similarity
    pipeline = TopicPipeline(
        database_path=args.database,
        output_folder=args.output_folder,
        model_folder=args.model_folder,
        min_topic_size=args.min_topic_size,
        min_samples=args.min_samples,
        cluster_selection_method=args.cluster_selection_method,
        umap_n_neighbors=args.umap_n_neighbors,
        outlier_min_similarity=outlier_min_similarity,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
