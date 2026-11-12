"""Embed the Czech chunks with BAAI/bge-m3 (same model as Embedding/embed.py).

    python Chunks/chunk.py        # first: writes Chunks/output/chunks_cs.parquet
    python Chunks/embed_cs.py     # from the repo root; uses the GPU if there is one, otherwise the CPU

Embeds each chunk's text exactly as chunked (no title or source added), L2-normalised.
Writes Chunks/output/embeddings_cs.npy  float32 [N, 1024], row i = chunk_ids_cs.txt line i
       Chunks/output/chunk_ids_cs.txt, Chunks/output/manifest_cs.json
The English chunks (chunks_en.parquet) share the chunk_id, so a Czech hit can be shown in English.
"""
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
MODEL = "BAAI/bge-m3"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import torch
    from sentence_transformers import SentenceTransformer
    rows = pq.read_table(OUT / "chunks_cs.parquet", columns=["chunk_id", "text"]).to_pylist()
    texts = [r["text"] for r in rows]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(rows)} chunks, device {device}; loading {MODEL} …", flush=True)
    t0 = time.time()
    model = SentenceTransformer(MODEL, device=device)
    emb = model.encode(texts, batch_size=16 if device == "cuda" else 8, normalize_embeddings=True,
                       show_progress_bar=True).astype(np.float32)
    np.save(OUT / "embeddings_cs.npy", emb)
    (OUT / "chunk_ids_cs.txt").write_text("\n".join(r["chunk_id"] for r in rows) + "\n", encoding="utf-8", newline="\n")
    fp = hashlib.sha256("".join(f"{r['chunk_id']}\t{r['text']}\n" for r in rows).encode("utf-8")).hexdigest()
    (OUT / "manifest_cs.json").write_text(json.dumps({
        "model": MODEL, "lang": "cs", "count": len(rows), "dim": int(emb.shape[1]), "device": device,
        "embedded": "chunk text only", "normalized": True, "text_fingerprint": fp,
        "seconds": round(time.time() - t0, 1), "created": datetime.now().isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
    }, indent=2), encoding="utf-8")
    print(f"Done in {time.time() - t0:.0f} s → {OUT / 'embeddings_cs.npy'}", flush=True)


if __name__ == "__main__":
    main()
