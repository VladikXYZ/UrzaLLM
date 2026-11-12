# Chunks: the final corpus, Czech and English, and the final reports

Chunks the preprocessed data in `preprocessed/` (Czech originals and English translations) with one simple rule
per source, embeds the Czech chunks in three variants, and builds every report into `FinalReports/`.

| Source  | Chunk | Count |
|---------|-------|-------|
| article | the whole article text | 956 |
| video   | the whole video description, without the recurring boilerplate (see below) | 978 |
| book    | one paragraph (text split at blank lines) | 976 (Anarchokapitalismus and Anarchoagorismus) |

**Video boilerplate.** Lines that recur in 20 or more descriptions, or that match the donation/signature patterns
(crypto addresses, IBAN and account numbers, donation pleas, separators, Urza's signature line, the closing thanks),
are removed from every description, separately in each language: 102 lines, 34% of all video words. Guest bios stay.
No description becomes empty, so every video is still one chunk.

## Run (from the repo root)

```bash
python Chunks/chunk.py                                # chunks_cs.*, chunks_en.*, chunks.parquet
python Chunks/embed_final.py --variant withsource     # GPU if available; repeat for nosource and textonly
python Chunks/build_reports.py                        # analysis + every report → FinalReports/
```

## Outputs (`Chunks/output/`)

- `chunks_cs.parquet` / `.jsonl` and `chunks_en.parquet` / `.jsonl`: one row per chunk, with the same `chunk_id` in
  the same row of both files. Columns: `chunk_id`, `source`, `doc_id`, `lang`, `title`, `text`, `n_words`; books
  also `book`, `chapter`, `section`, `para`; articles and videos `date`, `year`, `url`, `video_url`.
- `chunks.parquet`: the Czech chunks in the format of the earlier pipeline (`Preprocessing/output/chunks.parquet`),
  with kind, tags, series and guests taken from the earlier documents by URL and the three embedding texts:
  `embed_text` (`withsource`: context line naming the source), `embed_text_nosource` (the same line without the
  source word) and `embed_text_textonly` (the chunk text only).
- `results_<variant>/`: `embeddings.npy` (float32 [2910, 1024], L2-normalised, bge-m3, nothing truncated),
  `chunk_ids.txt`, `manifest.json`; `Embedding/analyze.py` adds `embedding_report.md`,
  `embeddings_source_centred.npy` and `chunks_embedded.parquet`.
- `projection/`: working output of the report scripts (copied to `FinalReports/`).

The scripts in `Embedding/` and `Projection/` read this corpus when `URZA_DATASET=final` is set (see `dataset.py`);
without it they read the earlier pipeline as before. To show a Czech search hit in English, look up its `chunk_id`
in `chunks_en.parquet`.

## Notes

- `chunk.py` stops if Czech and English don't line up (number of items, sections or paragraphs).
- Video dates come from `jsons/channel_metadata.json`, matched by URL.
- 21 chunks repeat another chunk's text word for word; `embed_final.py` embeds each distinct text once.
- Kept as they are, following the rules: 22 very short chunks (three words or fewer), such as hashtag-only articles,
  book headings like “Vzácnost” or single lines of dialogue.
