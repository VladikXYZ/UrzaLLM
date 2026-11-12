"""Embed one variant of the final chunks, reusing every vector whose exact text was embedded before.

    python Chunks/embed_final.py --variant textonly       # from the repo root; withsource | nosource | textonly
    python Chunks/embed_final.py --seed-from-cs           # once: add Chunks/embed_cs.py's vectors to the textonly cache

Reads the variant's text from Chunks/output/chunks.parquet (see Embedding/variants.py), looks each text up in a
cache keyed by its SHA-256 (Chunks/output/embed_cache/<variant>/), embeds only the missing ones with BAAI/bge-m3 (no
truncation: up to 8,192 tokens), and writes Chunks/output/results_<variant>/{embeddings.npy, chunk_ids.txt,
manifest.json} in the same format as Embedding/embed.py, so Embedding/analyze.py and Projection/ read it.
Same model and settings for every text, so a cached vector is identical to a fresh one.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ["URZA_DATASET"] = "final"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "Embedding"))
from variants import VARIANTS, fingerprint, load_texts, results_dir  # noqa: E402

MODEL = "BAAI/bge-m3"
CACHE = HERE / "output" / "embed_cache"


def h(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_cache(variant):
    d = CACHE / variant
    if variant == "withsource" and not (d / "keys.txt").exists() and (CACHE / "keys.txt").exists():
        d = CACHE                                  # first withsource run wrote to the top folder
    if not (d / "keys.txt").exists():
        return {}
    keys = (d / "keys.txt").read_text(encoding="utf-8").split()
    vecs = np.load(d / "vectors.npy")
    return {k: vecs[i] for i, k in enumerate(keys)}


def save_cache(cache, variant):
    d = CACHE / variant                            # one folder per variant: runs can overlap safely
    d.mkdir(parents=True, exist_ok=True)
    keys = list(cache)
    np.save(d / "vectors.npy", np.stack([cache[k] for k in keys]).astype(np.float32))
    (d / "keys.txt").write_text("\n".join(keys) + "\n", encoding="utf-8")


def seed_from_cs(cache):
    """Vectors from Chunks/embed_cs.py, for the texts it embedded (snapshot taken when it started)."""
    import pyarrow.parquet as pq
    out = HERE / "output"
    ids = (out / "chunk_ids_cs.txt").read_text(encoding="utf-8").split()
    emb = np.load(out / "embeddings_cs.npy")
    snap = {r["chunk_id"]: r["text"] for r in pq.read_table(out / "_embed_cs_input.parquet",
                                                             columns=["chunk_id", "text"]).to_pylist()}
    for i, cid in enumerate(ids):
        cache[h(snap[cid])] = emb[i]
    print(f"Seeded the cache with {len(ids)} vectors from embed_cs.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--seed-from-cs", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.seed_from_cs:
        cache = load_cache("textonly")
        seed_from_cs(cache)
        save_cache(cache, "textonly")
    if not args.variant:
        return
    cache = load_cache(args.variant)
    ids, _, texts = load_texts(args.variant)
    keys = [h(t) for t in texts]
    todo = sorted({k: t for k, t in zip(keys, texts) if k not in cache}.items(), key=lambda kt: -len(kt[1]))
    print(f"Variant {args.variant}: {len(ids):,} chunks, {len(ids) - sum(k not in cache for k in keys):,} cached, "
          f"{len(todo):,} to embed", flush=True)
    t0 = time.time()
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL, device=device)
    tok = model.tokenizer
    if todo:
        bs, step = (16 if device == "cuda" else 8), 200        # save the cache every 200 texts (resumable)
        for i in range(0, len(todo), step):
            part = todo[i:i + step]
            vec = model.encode([t for _, t in part], batch_size=bs, normalize_embeddings=True)
            for (k, _), v in zip(part, vec):
                cache[k] = v.astype(np.float32)
            save_cache(cache, args.variant)
            el = time.time() - t0
            done = i + len(part)
            print(f"  {done:>5}/{len(todo)} embedded  {el / 60:5.1f} min, ~{el / done * (len(todo) - done) / 60:5.1f} min left",
                  flush=True)
    emb = np.stack([cache[k] for k in keys]).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1)
    assert np.isfinite(emb).all() and np.allclose(norms, 1, atol=1e-3), "embeddings not finite / normalised"
    tl = [len(tok(t, add_special_tokens=True)["input_ids"]) for t in texts]
    out = results_dir(args.variant)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", emb)
    (out / "chunk_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8", newline="\n")
    (out / "manifest.json").write_text(json.dumps({
        "variant": args.variant, "text_column": VARIANTS[args.variant], "text_fingerprint": fingerprint(ids, texts),
        "model": MODEL, "dim": int(emb.shape[1]), "count": len(ids), "normalized": True,
        "max_length_tokens": model.max_seq_length, "truncated_chunks": sum(t > model.max_seq_length for t in tl),
        "token_length_max": max(tl), "token_length_median": sorted(tl)[len(tl) // 2], "device": device,
        "dry_run": False, "seconds": round(time.time() - t0, 1), "newly_embedded": len(todo),
        "created": datetime.now().isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "torch": torch.__version__}, "platform": platform.platform(),
    }, indent=2), encoding="utf-8")
    print(f"Done: {out}", flush=True)


if __name__ == "__main__":
    main()
