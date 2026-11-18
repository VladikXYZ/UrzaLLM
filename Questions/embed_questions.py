"""Embed the political space (topics, questions, concepts) with BAAI/bge-m3.

    python Questions/embed_questions.py            # from the repo root; CPU is fine (~326 short texts)
    python Questions/embed_questions.py --model google/embeddinggemma-300m --name embeddinggemma_2
        # another model: writes Questions/output_<name>/ (read by the real_data reports with URZA_MODEL=<name>).
        # Encode the items exactly as the chunks of real_data/embeddings_out_cs were encoded (same model, same
        # prompt or none), or the two sets of vectors are not comparable.

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
    import argparse
    global MODEL, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL, help="Hugging Face id (default BAAI/bge-m3)")
    ap.add_argument("--name", help="folder suffix: writes Questions/output_<name>/ (default: Questions/output/)")
    ap.add_argument("--prompt-name", help="sentence-transformers prompt_name, if the chunks were encoded with one")
    a = ap.parse_args()
    MODEL, OUT = a.model, (HERE / f"output_{a.name}" if a.name else OUT)
    sys.stdout.reconfigure(encoding="utf-8")
    import torch
    from sentence_transformers import SentenceTransformer
    rows = load()
    texts = [r["embed_text"] for r in rows]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(rows)} items, device {device}; loading {MODEL} …")
    t0 = time.time()
    model = SentenceTransformer(MODEL, device=device)
    kw = {"prompt_name": a.prompt_name} if a.prompt_name else {}
    emb = model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True, **kw).astype(np.float32)
    OUT.mkdir(exist_ok=True)
    np.save(OUT / "embeddings.npy", emb)
    (OUT / "ids.txt").write_text("\n".join(r["id"] for r in rows) + "\n", encoding="utf-8", newline="\n")
    fp = hashlib.sha256("".join(f"{r['id']}\t{r['embed_text']}\n" for r in rows).encode("utf-8")).hexdigest()
    (OUT / "manifest.json").write_text(json.dumps({
        "model": MODEL, "prompt_name": a.prompt_name, "count": len(rows), "dim": int(emb.shape[1]), "device": device,
        "text_fingerprint": fp, "seconds": round(time.time() - t0, 1),
        "created": datetime.now().isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
    }, indent=2), encoding="utf-8")
    print(f"Done in {time.time() - t0:.0f} s → {OUT}")


if __name__ == "__main__":
    main()
