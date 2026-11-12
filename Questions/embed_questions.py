"""Embed the political space (topics, questions, concepts) with BAAI/bge-m3.

    python Questions/embed_questions.py            # from the repo root; CPU is fine (~326 short texts)

Reads   Questions/political_space.tsv   (id, layer, major, sub, text, description)
Writes  Questions/output/embeddings.npy  float32 [N, 1024], L2-normalised, row i = ids[i]
        Questions/output/ids.txt, Questions/output/manifest.json

What is embedded: a topic or concept as "name: description" (the name alone is too short to carry
meaning), a question as the question itself. Same model as Embedding/embed.py, so the vectors are
directly comparable with the chunk embeddings.
"""
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "political_space.tsv"
OUT = HERE / "output"
MODEL = "BAAI/bge-m3"


def load():
    rows = list(csv.DictReader(open(SRC, encoding="utf-8"), delimiter="\t"))
    for r in rows:
        r["embed_text"] = (r["text"] if r["layer"] == "question" or not r["description"]
                           else f"{r['text']}: {r['description']}")
    return rows


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import torch
    from sentence_transformers import SentenceTransformer
    rows = load()
    texts = [r["embed_text"] for r in rows]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(rows)} items, device {device}; loading {MODEL} …")
    t0 = time.time()
    model = SentenceTransformer(MODEL, device=device)
    emb = model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    OUT.mkdir(exist_ok=True)
    np.save(OUT / "embeddings.npy", emb)
    (OUT / "ids.txt").write_text("\n".join(r["id"] for r in rows) + "\n", encoding="utf-8", newline="\n")
    fp = hashlib.sha256("".join(f"{r['id']}\t{r['embed_text']}\n" for r in rows).encode("utf-8")).hexdigest()
    (OUT / "manifest.json").write_text(json.dumps({
        "model": MODEL, "count": len(rows), "dim": int(emb.shape[1]), "device": device,
        "text_fingerprint": fp, "seconds": round(time.time() - t0, 1),
        "created": datetime.now().isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
    }, indent=2), encoding="utf-8")
    print(f"Done in {time.time() - t0:.0f} s → {OUT}")


if __name__ == "__main__":
    main()
