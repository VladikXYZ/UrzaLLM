# Search the map

`search_map.py` is a small local web app. Write a topic or a question, and it highlights the top-k
most relevant passages on two maps (raw and source-centred embeddings), lists them with their full
text, and marks where the question itself lands.

It searches 60,604 chunks:

- Vladimir's corpus `real_data/chunked.json` (2,911 chunks: 975 from the two books, 956 stoky.urza.cz
  articles, 980 YouTube video descriptions, each in Czech and English);
- Urza's speech from all 980 videos, `real_data/transcripts_urza.json` (57,693 chunks made by
  `Transcription/chunk_videos.py`, translated into English; embedded by `Transcription/embed_chunks.py`,
  which writes Czech vectors to `embeddings_out_rollo_cs/` and English ones to `embeddings_out_rollo_en/`).
  Each has the question it answers and a "watch at this moment" link to the video.

The search runs on the embeddings of the Czech texts (`real_data/embeddings_out_cs/` for the corpus,
`embeddings_out_rollo_cs/` for the transcripts, same chunk order as the JSON files). Both models are
multilingual, so a query can be in English or Czech. Texts are shown in English where a translation
exists, with a **Czech original** fold-out. `--corpus cs` leaves out the transcripts.

Two embedding models are available (`--model`):

| | `bge_m3` (default) | `embeddinggemma_2` |
|---|---|---|
| Model | `BAAI/bge-m3`, 1024 dims | `google/embeddinggemma-2`, 768 dims |
| Query | embedded as is | prefixed with `task: search result \| query: ` (`--no-query-prompt` turns it off); the chunks were embedded without a prompt |
| Needs | sentence-transformers 3+ | sentence-transformers ≥ 6.1, Pillow, torchvision (see below) |

On a few test queries neither was clearly better: bge-m3 found the right Czech passages for English
questions more often, EmbeddingGemma did better on some Czech questions but leaned towards the
English-language videos for English ones.

The source of a chunk (book / article / video / transcript) comes from `type` and, for articles vs
video descriptions, from the URL (older `chunked.json` files had those two types swapped).

The two maps are UMAP layouts made by `Projection/real_data_coords.py` and kept in git
(`real_data/projection/coords_<model>_{cs,all}.parquet`), because `umap-learn` doesn't import in the
main Python here. After new embeddings, recompute them in a throwaway environment (a short
`UV_CACHE_DIR` avoids Windows path-length errors):

```bash
uv run --no-project --python 3.12 --with umap-learn --with pyarrow python Projection/real_data_coords.py
uv run --no-project --python 3.12 --with umap-learn --with pyarrow python Projection/real_data_coords.py --model embeddinggemma_2
```

```bash
python Projection/search_map.py                                   # bge-m3, http://127.0.0.1:8765
python Projection/search_map.py --model embeddinggemma_2 --port 8766
```

It starts in under a minute (most of it loading the 60k chunks).

## First time on a new machine

Needs Python 3.10–3.12, git, and about 5 GB of free disk space (PyTorch plus the 2.3 GB model).
Everything else the search uses (chunks, embeddings, map coordinates) is already in the repo.

```bash
git clone git@github.com:VladikXYZ/UrzaLLM.git      # or: git pull, if you already have it
cd UrzaLLM
python -m venv .venv-search
.venv-search\Scripts\activate                      # Mac/Linux: source .venv-search/bin/activate
pip install -r requirements.txt                    # torch, sentence-transformers, pyarrow, numpy, …
pip install -U "sentence-transformers>=6.1" pillow torchvision   # only for EmbeddingGemma
python Projection/search_map.py --online           # first run only: downloads BAAI/bge-m3 (~2.3 GB)
python Projection/search_map.py --model embeddinggemma_2 --online   # first run only for Gemma
```

If the Gemma download is refused (Gemma models are often gated on Hugging Face), accept the licence
on the model page and log in with `hf auth login`, then run it again.

Then open http://127.0.0.1:8765. From then on, start it without `--online`: activate the
environment, run `python Projection/search_map.py`, and stop it with Ctrl+C. A GPU isn't needed:
on a CPU a search takes about a second instead of a fraction of one.

Options: `--model embeddinggemma_2`, `--no-query-prompt`, `--corpus cs`, `--port 9000`,
`--device cpu`, `--host 0.0.0.0` (let others on your network open it), and `--online` (see below).

## What it does

- **Scoring.** The query is embedded with the same model as the chunks (`--model`) and compared
  with the raw vectors by cosine similarity.
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
- **Filters.** The source boxes limit which chunks can be hits.
- **Sharing a search.** The URL keeps the query (`?q=…&k=…`), so a link reopens the same search
  as long as the server is running.

Hover a dot or a result to link the two. Click a dot to read its chunk in the panel under the maps
(with the question it answers and a link to that moment of the video for transcripts); click a
result to expand its full text.

## Model download

The script loads the model from the local Hugging Face cache **offline**. Otherwise `transformers`
starts a background download of a 2.3 GB safetensors copy whenever bge-m3 is cached only as
`pytorch_model.bin`. On a machine that doesn't have the model yet, run it once with `--online`.

## Inputs

| | |
|---|---|
| Chunks | `real_data/chunked.json`, then `real_data/transcripts_urza.json` |
| Embeddings | `real_data/embeddings_out_cs/<model>_vectors.npy`, then `embeddings_out_rollo_cs/<model>_vectors.npy` (both of `text_cs`) |
| Map | `real_data/projection/coords_<model>_all.parquet` (`_cs` with `--corpus cs`) |
| Dates | `real_data/yt_full.json`, `real_data/arts_full.json` (matched by URL) |

The vectors are matched to the chunks by position. The script refuses to start if the embeddings or
the map don't have one row per chunk in the same order; after re-chunking or re-embedding, rerun
`Projection/real_data_coords.py`.
