# Headline Radar: topic analysis of live news

Headline Radar collects live news headlines and short descriptions into a local SQLite database. This first step is the data-engineering layer only: an incremental loader for GDELT 2.0 and a small set of RSS feeds.

Sources:

- **GDELT 2.0 15-minute GKG files** — article title, URL, source domain, publish time, and language, read from the rolling lastupdate list and the corresponding 15-minute zip files. The first run walks a backfill window; later runs start from a stored high-water mark so only new files are pulled.
- **RSS feeds** — headline plus description/summary from a handful of general and financial outlets that were checked and responding when this project was set up.

Loads are idempotent: each article is identified by a SHA-256 hash of its URL, and re-running the loader never inserts a duplicate row. Blank descriptions (and blank titles) are stored as NULL; those rows are kept.

## Why this project

Headline Radar is an end-to-end text analytics project: it ingests live news, stores it in a clean, deduplicated database, and (in upcoming steps) finds and tracks topics with a custom BERTopic-style pipeline built from scratch. It is designed to be generic, so the same code can be pointed at any stream of short texts.

## Roadmap

- [x] Step 1: Incremental data engineering for GDELT and RSS into SQLite
- [x] Sentence embeddings with a custom `TextEmbedder` (MiniLM, mean pooling, L2 normalisation)
- [ ] Topic discovery: UMAP, HDBSCAN, and c-TF-IDF keywords with representative examples
- [ ] Scoring new headlines against fitted topics (blanks and outliers kept separate)
- [ ] Dashboard for topic trends over time

## Setup

The project runs on Windows, macOS, and Linux. Use Python 3.11 or later.

```text
cd headline-radar
python -m venv .venv
```

Activate the virtual environment:

- Windows: `.venv\Scripts\activate`
- macOS / Linux: `source .venv/bin/activate`

```text
pip install -r requirements.txt
```

## How to run

From the project root:

```text
python -m de.load_news
```

Useful options:

```text
python -m de.load_news --source all
python -m de.load_news --source gdelt --backfill-hours 6
python -m de.load_news --source rss
python -m de.load_news --database data/headline_radar.db
```

- `--source` is `all` (default), `gdelt`, or `rss`.
- `--backfill-hours` is used for GDELT only when no high-water mark has been stored yet. Later runs start from the watermark and pull only new data.
- The SQLite file is created automatically at `data/headline_radar.db`.

Re-run the same command at any time. Already-stored URLs are skipped and counted as duplicates.

## Project layout

Each class is PascalCase and lives in a file of the same name (for example, `TextEmbedder` is in `models/TextEmbedder.py`). Entry points and function-only modules keep snake_case names, including `python -m de.load_news`.

```text
headline-radar/
  de/                 Incremental loaders and the command-line entry point
    load_news.py      python -m de.load_news
    GdeltLoader.py
    RssLoader.py
    FileNotReadyError.py
  utils/              Reusable classes (Database, HttpClient, Config, NewsLoader, ...)
  models/             Topic-model classes (TextEmbedder, ...)
  data/               Local SQLite file (ignored by git)
  requirements.txt
```

## Database

`data/headline_radar.db` has three tables:

- `articles` — one row per URL (`id` is the SHA-256 of the URL). Columns: `title`, `description`, `url`, `source`, `source_type` (`gdelt` or `rss`), `published_at` (UTC), `fetched_at` (UTC), `language`.
- `load_runs` — a log of each load: counts fetched, inserted, and skipped as duplicates, plus the high-water mark and any error.
- `watermarks` — the last successful high-water mark per source, so the next GDELT run only requests new time windows.

## Source notes

- GDELT data comes from the 15-minute GKG files advertised by `https://data.gdeltproject.org/gdeltv2/lastupdate.txt`. Titles are read from the GKG `PAGE_TITLE` extras field; a precise publish time is used when `PAGE_PRECISEPUBTIMESTAMP` is present, otherwise the 15-minute file time. GKG does not include article descriptions, so those are stored as NULL. Files are downloaded over HTTPS; the HTTP URLs in lastupdate.txt often return 500. The very newest file listed in lastupdate can 404 for a short time and is retried on the next run. The GDELT DOC 2.0 API also exposes titles, but it rate-limits to one request every five seconds and, during setup, frequently returned HTTP 429 or an empty JSON object, so it is not used here.
- RSS feeds included here: BBC News, NPR News, CNBC Top Stories, Guardian Business, MarketWatch, NYT World, and Al Jazeera. Feeds that did not respond during setup (Yahoo Finance 404, Politico 403, CNN SSL error) were left out.
