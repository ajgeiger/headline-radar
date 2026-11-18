# Headline Radar: topic analysis of live news

Headline Radar collects live news headlines and short descriptions into a local SQLite database. This first step is the data-engineering layer only: an incremental loader for GDELT 2.0 and a small set of RSS feeds.

Sources:

- **GDELT 2.0 15-minute GKG files**: article title, URL, source domain, publish time, and language, read from the rolling lastupdate list and the corresponding 15-minute zip files. The first run walks a backfill window; later runs start from a stored high-water mark so only new files are pulled.
- **RSS feeds**: headline plus description/summary from a handful of general and financial outlets that were checked and responding when this project was set up.

Loads are idempotent: each article is identified by a SHA-256 hash of its URL, and re-running the loader never inserts a duplicate row. Blank descriptions (and blank titles) are stored as NULL; those rows are kept.

## Why this project

Headline Radar is an end-to-end text analytics project: it ingests live news, stores it in a clean, deduplicated database, and finds topics with a custom BERTopic-style pipeline built from scratch. It is designed to be generic, so the same code can be pointed at any stream of short texts.

## Roadmap

- [x] Step 1: Incremental data engineering for GDELT and RSS into SQLite
- [x] Sentence embeddings with a custom `TextEmbedder` (MiniLM, mean pooling, L2 normalisation)
- [x] Topic discovery with a custom `TopicModel` (UMAP, HDBSCAN, c-TF-IDF keywords, example texts)
- [x] Labeling new headlines against fitted topics (blanks and outliers kept separate)
- [x] `TopicJoiner` to add topic ids to any table without duplicating rows
- [ ] Scheduled refits and topic tracking in the database
- [ ] Dashboard for topic trends over time

## Topic model

`models/TopicModel.py` is a small BERTopic-style model written from scratch. It works on any list of short texts, not only news.

1. Embed each unique text with `TextEmbedder`.
2. Reduce the embeddings to 5 dimensions with UMAP.
3. Cluster with HDBSCAN.
4. Give identical texts the same topic, then move any topic smaller than `min_topic_size` to the outliers.
5. Number topics by size, so topic 0 is the largest.
6. Describe each topic with c-TF-IDF keywords and three example texts.

Topic ids:

- `-1` means the text did not fit any topic (an outlier).
- `-2` means the text was blank. Blank texts never reach the model.

```python
from models import TopicModel

model = TopicModel(min_topic_size=30)
topic_ids = model.fit_transform(texts)        # texts must be nonblank
print(model.get_topic_info())                 # topic_id, document_count, topic_name, top_words, ...
print(model.get_fit_diagnostics())            # cluster sizes before and after the duplicate vote

model.transform_with_status(["Fed signals another rate cut", ""])
# topic_id per text, with status "seen", "new", or "blank"

model.visualize_hierarchy(top_n_topics=25).show()
model.save("data/topic_model")
model = TopicModel.load("data/topic_model")
```

Texts that were in the fit data get their fitted topic back. New texts are placed with HDBSCAN `approximate_predict`. The model only knows the data it was fit on, so refit it as new headlines arrive.

### Run it end to end

```text
python -m models.run_topic_model
python -m models.run_topic_model --min-topic-size 20 --output-folder outputs
```

`TopicPipeline` reads the articles, fits the model, saves it to `data/topic_model/`, and writes these files to `outputs/`:

- `document_topics.csv`: one row per article with its topic
- `topic_info.csv`: one row per topic with size, top words, and examples
- `fit_diagnostics.csv`: cluster sizes before and after the duplicate vote
- `topic_hierarchy.html` and `topic_hierarchy.png`: dendrogram of topic similarity
- `topic_explorer.html`: 2D map of every text colored by topic, topic sizes, and a topic table

The PNG export uses kaleido, which needs Google Chrome. If Chrome is not installed, run `kaleido_get_chrome` once.

### Notebook

`experiments/topic_model.ipynb` walks through the same flow one step at a time: load texts, embed, fit, inspect topics, set labels, plot, and label new texts. Its charts go to `outputs/notebook/`.

`TopicJoiner` adds a `topic_id` column to a pandas DataFrame by exact text match. It never adds rows, gives blank text `-2`, and raises an error if two topic ids are given for the same text or if a text is missing from the lookup (a sign the model needs a refit).

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

Each class is PascalCase and lives in a file of the same name (for example, `TextEmbedder` is in `models/TextEmbedder.py`). Entry points and function-only modules keep snake_case names, including `python -m de.load_news`. Each `utils/` subfolder exports its classes, so imports read `from utils.storage import Database`.

```text
headline-radar/
  de/                 Incremental loaders and the command-line entry point
    load_news.py      python -m de.load_news
    NewsLoader.py     Base class: run, log, and advance the watermark
    GdeltLoader.py
    RssLoader.py
    LoadResult.py
    FileNotReadyError.py
  utils/              Shared building blocks, grouped by job
    config/           Config, setup_logging
    http/             HttpClient, HttpError
    news/             Article, RssFeed
    storage/          Database
    text/             TextCleaner, HtmlTextExtractor
  models/             Embedding and topic-model classes
    TextEmbedder.py   MiniLM sentence embeddings
    TopicModel.py     UMAP + HDBSCAN + c-TF-IDF topic model
    TopicHierarchyPlot.py
    TopicJoiner.py    Adds topic ids to a pandas DataFrame
    TopicExplorer.py  Interactive HTML page for browsing topics
    TopicPipeline.py  Fit on the database and write every output
    run_topic_model.py  python -m models.run_topic_model
  experiments/        Step-by-step notebook (topic_model.ipynb)
  outputs/            Tables and charts from the pipeline (ignored by git)
  data/               Local SQLite file (ignored by git)
  requirements.txt
```

## Database

`data/headline_radar.db` has three tables:

- `articles`: one row per URL (`id` is the SHA-256 of the URL). Columns: `title`, `description`, `url`, `source`, `source_type` (`gdelt` or `rss`), `published_at` (UTC), `fetched_at` (UTC), `language`.
- `load_runs`: a log of each load: counts fetched, inserted, and skipped as duplicates, plus the high-water mark and any error.
- `watermarks`: the last successful high-water mark per source, so the next GDELT run only requests new time windows.

## Source notes

- GDELT data comes from the 15-minute GKG files advertised by `https://data.gdeltproject.org/gdeltv2/lastupdate.txt`. Titles are read from the GKG `PAGE_TITLE` extras field; a precise publish time is used when `PAGE_PRECISEPUBTIMESTAMP` is present, otherwise the 15-minute file time. GKG does not include article descriptions, so those are stored as NULL. Files are downloaded over HTTPS; the HTTP URLs in lastupdate.txt often return 500. The very newest file listed in lastupdate can 404 for a short time and is retried on the next run. The GDELT DOC 2.0 API also exposes titles, but it rate-limits to one request every five seconds and, during setup, frequently returned HTTP 429 or an empty JSON object, so it is not used here.
- RSS feeds included here: BBC News, NPR News, CNBC Top Stories, Guardian Business, MarketWatch, NYT World, and Al Jazeera. Feeds that did not respond during setup (Yahoo Finance 404, Politico 403, CNN SSL error) were left out.
