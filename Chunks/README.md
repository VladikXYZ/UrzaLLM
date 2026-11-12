# Chunks: aligned Czech and English chunks

Chunks the preprocessed data in `preprocessed/` (Czech originals and English translations) with one simple rule
per source, and embeds the Czech chunks.

| Source  | Chunk                                  | Count |
|---------|----------------------------------------|-------|
| article | the whole article text                 | 956   |
| video   | the whole video description            | 978   |
| book    | one paragraph (text split at blank lines) | 976 (Anarchokapitalismus and Anarchoagorismus) |

Run from the repo root:

```bash
python Chunks/chunk.py       # writes Chunks/output/chunks_cs.* and chunks_en.*
python Chunks/embed_cs.py    # embeds the Czech chunks (bge-m3; GPU if available, otherwise CPU)
```

## Outputs (`Chunks/output/`)

- `chunks_cs.parquet` / `.jsonl` and `chunks_en.parquet` / `.jsonl`: one row per chunk, with the same `chunk_id` in
  the same row of both files. Columns:
  - `chunk_id`, `source`, `doc_id`, `lang`, `title`, `text`, `n_words`;
  - books: `book`, `chapter`, `section`, `para`;
  - articles and videos: `date`, `year`, `url`, and `video_url` (the article's linked video, if any).
- `embeddings_cs.npy`: float32 [2910, 1024], L2-normalised, from the Czech chunk text only (no title or source
  added). Row i belongs to line i of `chunk_ids_cs.txt`. The run's details are in `manifest_cs.json`.

To show a Czech search hit in English, look up its `chunk_id` in `chunks_en.parquet`.

## Notes

- `chunk.py` stops if Czech and English don't line up (count of items, sections or paragraphs).
- Video dates come from `jsons/channel_metadata.json`, matched by URL.
- Kept as they are, following the rules: 22 very short chunks (three words or fewer), such as hashtag-only articles,
  link-only video descriptions, book headings like “Vzácnost” or one line of dialogue; and 7 articles with the
  same text as another article.
