"""Embed the UrzaLLM text chunks with BAAI/bge-m3.

    python embed.py              # embed everything, then pack results into a zip
    python embed.py --dry-run    # check the data and output files without the model

Input:  data/chunks.parquet  (columns: chunk_id, source, embed_text)
Output: output/embeddings.npy   float32 [N, 1024], L2-normalised, row i = chunk_ids[i]
        output/chunk_ids.txt    one chunk_id per line, same order as the rows
        output/manifest.json    model, versions, device, timings, checksums
        results_<date>.zip      the three files above, ready to send back

The run can be stopped and restarted: finished batches are saved in
output/partial/ and skipped on the next run.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "chunks.parquet"
OUT = HERE / "output"
PARTIAL = OUT / "partial"

DEFAULT_MODEL = "BAAI/bge-m3"
DIM = 1024


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_chunks():
    t = pq.read_table(DATA, columns=["chunk_id", "source", "embed_text"]).to_pylist()
    ids = [r["chunk_id"] for r in t]
    if len(set(ids)) != len(ids):
        sys.exit("ERROR: duplicate chunk_id in data/chunks.parquet")
    if any(not (r["embed_text"] or "").strip() for r in t):
        sys.exit("ERROR: empty embed_text in data/chunks.parquet")
    return t


def pack_results(manifest: dict) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    prefix = "DRYRUN_not_real" if manifest["dry_run"] else "results"
    zpath = HERE / f"{prefix}_{stamp}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("embeddings.npy", "chunk_ids.txt", "manifest.json"):
            z.write(OUT / name, name)
    return zpath


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--batch-size", type=int, default=0, help="default: 32 on GPU, 8 on CPU")
    ap.add_argument("--max-length", type=int, default=1024, help="tokens; longer texts are truncated")
    ap.add_argument("--dry-run", action="store_true", help="check data and outputs with random vectors")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not DATA.exists():
        sys.exit(f"ERROR: {DATA} not found. Unzip the whole package and run from its folder.")
    rows = load_chunks()
    n = len(rows)
    print(f"Loaded {n:,} chunks from {DATA.name}")
    OUT.mkdir(exist_ok=True)
    PARTIAL.mkdir(exist_ok=True)
    t0 = time.time()

    if args.dry_run:
        device, versions, model_rev = "none (dry run)", {}, None
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((n, DIM)).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        token_lengths = []
    else:
        import torch
        import sentence_transformers
        import transformers
        from sentence_transformers import SentenceTransformer

        device = pick_device(args.device)
        bs = args.batch_size or (32 if device == "cuda" else 16 if device == "mps" else 8)
        print(f"Device: {device}, batch size {bs}. Loading {args.model} "
              "(first run downloads ~2.3 GB) …")
        model = SentenceTransformer(args.model, device=device)
        model.max_seq_length = args.max_length
        dim = model.get_sentence_embedding_dimension()
        versions = {"python": platform.python_version(), "torch": torch.__version__,
                    "sentence_transformers": sentence_transformers.__version__,
                    "transformers": transformers.__version__, "numpy": np.__version__}
        model_rev = getattr(getattr(model[0], "auto_model", None), "config", None)
        model_rev = getattr(model_rev, "_commit_hash", None)

        # Self-test: similar Czech sentences should score higher than unrelated ones.
        test = model.encode(["Stát by neměl regulovat drogy.",
                             "Drogová prohibice státu škodí lidem.",
                             "Dnes je hezké počasí na procházku."], normalize_embeddings=True)
        close, far = float(test[0] @ test[1]), float(test[0] @ test[2])
        print(f"Self-test: related {close:.2f} vs unrelated {far:.2f} "
              f"{'OK' if close > far else 'UNEXPECTED – check the install'}")

        tok = model.tokenizer
        token_lengths = [len(tok(r["embed_text"], add_special_tokens=True)["input_ids"]) for r in rows]

        # Longest first: memory problems show up immediately, and batches are evenly sized.
        order = sorted(range(n), key=lambda i: -token_lengths[i])
        # Partial results are only reused for the same data, model and batch size.
        run_dir = PARTIAL / f"{sha256(DATA)[:12]}_{args.model.replace('/', '_')}_{args.max_length}_bs{bs}"
        run_dir.mkdir(parents=True, exist_ok=True)
        batches = [order[i:i + bs] for i in range(0, n, bs)]
        emb = np.zeros((n, dim), dtype=np.float32)
        done = 0
        for b, idx in enumerate(batches):
            part = run_dir / f"batch_{b:05d}.npy"
            if part.exists():
                emb[idx] = np.load(part)
            else:
                vec = model.encode([rows[i]["embed_text"] for i in idx], batch_size=bs,
                                   normalize_embeddings=True, convert_to_numpy=True,
                                   show_progress_bar=False)
                np.save(part, vec.astype(np.float32))
                emb[idx] = vec
            done += len(idx)
            if b % 10 == 0 or done == n:
                el = time.time() - t0
                eta = el / done * (n - done)
                print(f"  {done:>5}/{n} chunks  {el / 60:5.1f} min elapsed, ~{eta / 60:5.1f} min left", flush=True)

    norms = np.linalg.norm(emb, axis=1)
    if not np.all(np.isfinite(emb)) or abs(norms.mean() - 1) > 1e-3:
        sys.exit("ERROR: embeddings are not finite / not normalised")

    np.save(OUT / "embeddings.npy", emb)
    (OUT / "chunk_ids.txt").write_text("\n".join(r["chunk_id"] for r in rows) + "\n", encoding="utf-8")
    manifest = {
        "model": args.model,
        "model_revision": model_rev,
        "dim": int(emb.shape[1]),
        "count": n,
        "normalized": True,
        "max_length_tokens": args.max_length,
        "truncated_chunks": sum(1 for t in token_lengths if t > args.max_length),
        "token_length_max": max(token_lengths) if token_lengths else None,
        "token_length_median": sorted(token_lengths)[n // 2] if token_lengths else None,
        "device": device,
        "dry_run": args.dry_run,
        "seconds": round(time.time() - t0, 1),
        "created": datetime.now().isoformat(timespec="seconds"),
        "input_sha256": sha256(DATA),
        "embeddings_sha256": sha256(OUT / "embeddings.npy"),
        "versions": versions,
        "platform": platform.platform(),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    zpath = pack_results(manifest)
    print(f"\nDone in {manifest['seconds'] / 60:.1f} min. "
          f"{manifest['truncated_chunks']} chunks were longer than {args.max_length} tokens (truncated).")
    if args.dry_run:
        print("DRY RUN: the vectors are random. Run without --dry-run for real embeddings.")
    if not args.dry_run:
        print(f"Send this file back: {zpath}")


if __name__ == "__main__":
    main()
