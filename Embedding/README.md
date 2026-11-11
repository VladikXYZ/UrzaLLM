# Embedding

Turns the preprocessed chunks (`Preprocessing/output/chunks.parquet`) into vectors with
the open model [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (1,024 dimensions,
strong on Czech). Everything runs locally; the model (~2.3 GB) is downloaded from
Hugging Face on first use.

## Variants

The text in front of each chunk (the context line) comes in two versions; see
`variants.py`.

| Variant | Context line | Results |
|---|---|---|
| `withsource` | `Kniha Anarchokapitalismus › 10. Zdravotnictví`, `Článek „…“ (2017)`, `Video „…“ (2026; hosté: …)` | `results_withsource/` (done 2026-11-11) |
| `nosource` | the same without the word naming the source | `results_nosource/` |

The `nosource` run tests how much that word makes chunks cluster by source.

## Run (on a machine with a GPU)

From the repo root:

```bash
python -m venv .venv-embed
source .venv-embed/bin/activate            # Windows: .venv-embed\Scripts\activate
pip install -r Embedding/requirements.txt
python Embedding/embed.py --variant nosource
python Embedding/analyze.py --variant nosource
```

`embed.py` prints a self-test (two related Czech sentences must score higher than an
unrelated one), the device it uses, and progress with the time left. On an NVIDIA GPU
it takes about 2 minutes; on a CPU 30–90 minutes. If it stops, run it again: finished
batches are kept in `Embedding/.partial/`.

`analyze.py` needs only numpy and pyarrow, so it can also run on any other machine
after the results are pulled.

Useful options for `embed.py`: `--batch-size 4` (out of memory), `--device cpu`,
`--dry-run` (random vectors, to test the setup without the model; writes to
`Embedding/.dryrun_<variant>/`).

Then commit and push the results:

```bash
git add Embedding/results_nosource
git commit -m "Add nosource embeddings"
git push
```

## Output (`results_<variant>/`)

| File | In git | Contents |
|---|---|---|
| `embeddings.npy` | yes | float32 [N, 1024], L2-normalised; row *i* is chunk *i* of `chunk_ids.txt` |
| `chunk_ids.txt` | yes | chunk id per row |
| `manifest.json` | yes | variant, model and versions, device, timing, `text_fingerprint` (hash of exactly what was embedded) |
| `embedding_report.md` | yes | checks, similarity distribution, same-source clustering, near-duplicates, example neighbours |
| `chunks_embedded.parquet` | no (rebuilt by `analyze.py`) | all chunk metadata + an `embedding` column |
| `embeddings_source_centred.npy` | no (rebuilt by `analyze.py`) | each source's average vector subtracted, re-normalised |

`analyze.py` refuses results whose `text_fingerprint` doesn't match the current
`chunks.parquet`: if the preprocessing changes, re-embed.
