# Preprocessing

Turns the raw data at the UrzaLLM repo root (article and YouTube exports in `jsons/`, book
EPUBs in `books/`) into one clean, chunked corpus ready for embedding, plus a report of
everything that was changed or removed.

The books are read from the EPUBs, not from `jsons/ankap.json` / `anag.json`: that export
merged sections that share an EPUB file (e.g. all AMEN economics objections became one
section) and split words at formatting boundaries. See `epub_books.py`.

## Setup (once)

The virtual environment lives at the repo root, next to this folder, in `.venv`
(see the root `requirements.txt`).

```powershell
cd UrzaLLM
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Run

```powershell
cd UrzaLLM\Preprocessing
..\.venv\Scripts\python run.py
```

It takes about 10 seconds. Then open `output\report.html`.

Options:

| Flag | Effect |
|---|---|
| `--data PATH` | Read the raw JSON files from another folder (default `../jsons`). |
| `--out PATH` | Write results to another folder (default `output/`). |
| `--no-parquet` | Write JSONL only. |
| `--no-report` | Skip the report. |

## Output (`output/`)

| File | Contents |
|---|---|
| `chunks.parquet` / `.jsonl` | **Input for embedding.** One row per chunk: `text`, `embed_text` (text with a context line such as `Kniha Anarchokapitalismus › 10. Zdravotnictví`), and metadata for the map (`source`, `kind`, `year`, `title`, `url`, `series`, `chapter`, `tags`, `guests`, `flags`, …). |
| `documents.parquet` / `.jsonl` | Every raw record, including excluded ones, with `status`, `exclude_reason`, `kind`, `flags` and cleaned text. |
| `article_video_links.jsonl` | Every article-to-video match decision, with score and days apart. |
| `stats.json` | All numbers from the report, machine-readable. |
| `report.html` | Self-contained report (charts embedded), easy to share. |
| `report.md` + `figures/` | The same report as Markdown. |

## What the pipeline does

1. **Videos** (`videos.py`): removes the donation block, donation lines, URLs and crypto
   addresses; reads guest names from the `– Name; bio` lines, then removes those lines;
   drops generic tags, merges tag synonyms; builds the text from description + chapter
   titles + tags.
2. **Articles** (`articles.py`): rejoins paragraphs the scraper split at inline links;
   removes video-player lines; matches text-less posts to their video (`linking.py`) and
   excludes them as duplicates (unmatched ones are kept, with their title as text);
   removes duplicate texts; labels short event posts as `announcement`.
3. **Books** (`epub_books.py`, `books.py`): parses each EPUB along its table of contents
   (one section per TOC entry, with its full path such as
   `dodatek › AMEN › Ekonomie › Kartely`); drops front matter, back matter (partner pages,
   sponsors) and headings that only group subsections; writes AMEN objections as
   `Námitka: … / Odpověď: …` and labels them `book_faq`.
4. **Chunking** (`chunking.py`): short documents stay whole; longer ones are split at
   paragraph breaks. Sizes are set per source in `config.CHUNKING`: books about 150 words
   (max 250), because book text is dense and a smaller chunk usually holds one argument;
   articles and videos about 300 words (max 450), so most stay whole.
5. **Report** (`report.py`).

Nothing is deleted silently. Excluded records stay in `documents.*` with a reason.

## Choosing what to embed

`kind` tells you what a chunk is. For coverage you'll usually want to drop announcements
(example uses pandas, which isn't in `requirements.txt`; `pip install pandas` if needed):

```python
import pandas as pd
chunks = pd.read_parquet("output/chunks.parquet")
evidence = chunks[chunks.kind != "announcement"]
```

| kind | Meaning |
|---|---|
| `article` | Blog post with real text |
| `announcement` | Short event/stream/cancellation notice |
| `video_post` | Post whose video wasn't found; title only |
| `book_section`, `book_story`, `book_faq` | Book chapters, opening stories, AMEN sections |
| `video` | YouTube video (description, chapters, tags) |

## Tuning

All thresholds, patterns, tag synonyms and the book structure rules are in `config.py`.
Change them and rerun. The report's sections 4 (matching) and 5 (cleaning) show the
borderline cases worth checking after a change.
