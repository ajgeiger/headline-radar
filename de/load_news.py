"""Entry point: python -m de.load_news --source all|gdelt|rss"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from de.GdeltLoader import GdeltLoader
from de.RssLoader import RssLoader
from utils.Config import Config
from utils.Database import Database
from utils.HttpClient import HttpClient
from utils.logging_setup import setup_logging
from utils.LoadResult import LoadResult


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load new news articles from GDELT and RSS into the local database."
    )
    parser.add_argument(
        "--source",
        choices=("all", "gdelt", "rss"),
        default="all",
        help="Which source to load (default: all).",
    )
    parser.add_argument(
        "--backfill-hours",
        type=int,
        default=None,
        help=(
            "Hours of GDELT history to request when no watermark exists "
            f"(default: {Config.default_backfill_hours}). "
            "Ignored on later runs once a watermark is stored."
        ),
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="Optional path to the SQLite file (default: data/headline_radar.db).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = setup_logging(args.log_level)
    config = Config()
    if args.database is not None:
        config.database_path = Path(args.database)

    database = Database(config.database_path)
    results: dict[str, LoadResult] = {}
    exit_code = 0

    try:
        if args.source in {"all", "gdelt"}:
            gdelt_http = HttpClient(
                user_agent=config.user_agent,
                timeout_seconds=config.gdelt_file_timeout_seconds,
                max_retries=config.http_max_retries,
                min_interval_seconds=config.gdelt_min_interval_seconds,
                retry_after_seconds=config.gdelt_retry_after_seconds,
            )
            loader = GdeltLoader(
                database,
                gdelt_http,
                config,
                backfill_hours=args.backfill_hours,
            )
            logger.info("Starting GDELT load")
            try:
                results["gdelt"] = loader.run()
            except Exception as exc:
                exit_code = 1
                logger.exception("GDELT load failed: %s", exc)
                results["gdelt"] = LoadResult(0, 0, 0, None, error=str(exc))

        if args.source in {"all", "rss"}:
            rss_http = HttpClient(
                user_agent=config.user_agent,
                timeout_seconds=config.http_timeout_seconds,
                max_retries=config.http_max_retries,
                min_interval_seconds=0.2,
                retry_after_seconds=15.0,
            )
            loader = RssLoader(database, rss_http, config)
            logger.info("Starting RSS load")
            try:
                results["rss"] = loader.run()
            except Exception as exc:
                exit_code = 1
                logger.exception("RSS load failed: %s", exc)
                results["rss"] = LoadResult(0, 0, 0, None, error=str(exc))
    finally:
        counts = database.counts_by_source_type()
        total = database.count_articles()
        database.close()

    print()
    print("Load summary")
    print("------------")
    for name, result in results.items():
        status = "ok" if result.ok else f"failed: {result.error}"
        print(
            f"  {name}: fetched={result.records_fetched} "
            f"inserted={result.records_inserted} "
            f"duplicates={result.records_duplicate} "
            f"watermark={result.high_water_mark} "
            f"[{status}]"
        )
    print("Database totals")
    for source_type, count in counts.items():
        print(f"  {source_type}: {count}")
    print(f"  all: {total}")
    print(f"  file: {config.database_path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
