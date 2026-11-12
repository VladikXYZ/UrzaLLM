# Search the map

`search_map.py` is a small local web app. Write a topic or a question, and it highlights the top-k
most relevant passages on the two maps from `projection_compare_textonly.html` (raw and
source-centred), lists them with their full text, and marks where the question itself lands.

```bash
python Projection/search_map.py          # from the repo root, then open http://127.0.0.1:8765
```

It starts in a few seconds.

## First time on a new machine

Needs Python 3.10–3.12, git, and about 5 GB of free disk space (PyTorch plus the 2.3 GB model).
Everything else the search uses (chunks, embeddings, map coordinates) is already in the repo.

```bash
git clone git@github.com:VladikXYZ/UrzaLLM.git      # or: git pull, if you already have it
cd UrzaLLM
python -m venv .venv-search
.venv-search\Scripts\activate                      # Mac/Linux: source .venv-search/bin/activate
pip install -r Embedding/requirements.txt          # torch, sentence-transformers, pyarrow, numpy
python Projection/search_map.py --online           # first run only: downloads BAAI/bge-m3 (~2.3 GB)
```

Then open http://127.0.0.1:8765. From then on, start it without `--online`: activate the
environment, run `python Projection/search_map.py`, and stop it with Ctrl+C. A GPU isn't needed:
on a CPU a search takes about a second instead of a fraction of one. Options: `--port 9000`, `--device cpu` (no GPU; a query then takes
about a second), `--host 0.0.0.0` (let others on your network open it), and `--online` (see below).

## What it does

- **Scoring.** The query is embedded with the same model as the chunks (`BAAI/bge-m3`) and compared
  with the raw `textonly` vectors by cosine similarity, as for any search query (see
  `Reports/README.md`). For a question from `Questions/political_space.tsv`, the top 10 is
  identical to the one built from that question's stored embedding.
- **The question's position.** UMAP isn't re-fitted (and `umap-learn` doesn't import with
  NumPy 2 on this machine). Instead, the question is placed at the similarity-weighted
  average position of its 10 best passages on each map, with weights `exp((s − s_best) / 0.02)`,
  so the closest passages dominate. This is how UMAP itself initialises a new point. If the
  best passages sit in two distant regions, the marker lands between them; the lines to the
  hits show when that happens.
- **Auto-zoom to hits** (on by default) frames the question and the central 70% of the hits after
  each search. Hits outside the frame are counted in the corner and can be found by following
  their lines. Turn it off to keep the whole map after a search.
- **Zooming by hand.** Each map zooms on its own:
  - scroll over it to zoom in or out at the cursor;
  - drag it to move around (a drag doesn't count as a click);
  - double-click it to frame the hits again (the whole map when there is no search);
  - or use the buttons above it: **+**, **−**, **Fit hits** and **Whole map**.

  The corner shows the zoom factor and how many hits are outside the view.
- **Filters.** The source boxes and "Include announcements" limit which chunks can be hits;
  announcements are excluded by default.
- **Sharing a search.** The URL keeps the query (`?q=…&k=…`), so a link reopens the same search
  as long as the server is running.

Hover a dot or a result to link the two; click a dot or a result to expand its full text.

## Model download

The script loads `bge-m3` from the local Hugging Face cache **offline**. Otherwise `transformers`
starts a background download of a 2.3 GB safetensors copy whenever the model is cached only as
`pytorch_model.bin`. On a machine that doesn't have the model yet, run it once with `--online`.

## Inputs

`Preprocessing/output/chunks.parquet`, `Embedding/results_textonly/{embeddings.npy, chunk_ids.txt}`
and `Projection/output/coords_textonly.parquet`. The script refuses to start if they don't hold
the same chunks in the same order. After re-chunking or re-embedding, rerun
`Projection/project_compare.py --variant textonly` to rebuild the coordinates.
