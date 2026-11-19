"""Embed the transcript chunks (chunk_videos.py) with the search models, in Czech and in English.

    .venv-search\\Scripts\\python.exe Transcription/embed_chunks.py [--chunks FILE] [--langs cs en] [--models bge_m3 embeddinggemma_2]

Reads   real_data/transcripts_urza.json                  (default; any file in the same format works)
Writes  embeddings_out_rollo_<lang>/<model>_vectors.npy  one unit vector per chunk, same order as the JSON
        embeddings_out_rollo_<lang>/metadata_index.json  the embedded texts, for checking the alignment

The text of a chunk is the question it answers plus the chunk itself (context_<lang> + " " + text_<lang>),
as in Vladimir's embedding.py (which joined them without the space). Documents get no prompt, like the
corpus vectors in real_data/embeddings_out_cs/, so both live in one index; search_map.py adds the query
prompt. EmbeddingGemma-2 needs sentence-transformers >= 6.1 (hence .venv-search).
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"bge_m3": "BAAI/bge-m3", "embeddinggemma_2": "google/embeddinggemma-2"}


def texts(chunks, lang):
    return [" ".join(t for t in (c.get(f"context_{lang}", "").strip(), c[f"text_{lang}"].strip()) if t)
            for c in chunks]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", type=Path, default=ROOT / "real_data" / "transcripts_urza.json")
    ap.add_argument("--langs", nargs="+", choices=["cs", "en"], default=["cs", "en"])
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    ap.add_argument("--out-prefix", default=str(ROOT / "embeddings_out_rollo"))
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    by_lang = {lang: texts(chunks, lang) for lang in args.langs}
    for lang, t in by_lang.items():
        out = Path(f"{args.out_prefix}_{lang}")
        out.mkdir(exist_ok=True)
        (out / "metadata_index.json").write_text(json.dumps(t, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{lang}: {len(t):,} chunks, {sum(len(x.split()) for x in t):,} words -> {out}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name in args.models:
        model = SentenceTransformer(MODELS[name], device=device, model_kwargs={"dtype": torch.bfloat16})
        for lang, t in by_lang.items():
            t0 = time.time()
            v = model.encode(t, batch_size=args.batch_size, normalize_embeddings=True, convert_to_numpy=True,
                             show_progress_bar=True).astype(np.float32)
            path = Path(f"{args.out_prefix}_{lang}") / f"{name}_vectors.npy"
            tmp = path.with_name(path.stem + ".tmp.npy")
            np.save(tmp, v)
            tmp.replace(path)
            print(f"{name} {lang}: {v.shape} in {time.time() - t0:.0f}s -> {path}", flush=True)
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
