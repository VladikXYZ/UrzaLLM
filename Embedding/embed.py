"""Embed the preprocessed chunks with BAAI/bge-m3 (needs PyTorch; a GPU makes it fast).

    python Embedding/embed.py --variant nosource      # from the repo root
    python Embedding/embed.py --variant withsource
    python Embedding/embed.py --variant nosource --dry-run   # random vectors, no model

Reads   Preprocessing/output/chunks.parquet (column per variant, see variants.py)
Writes  Embedding/results_<variant>/embeddings.npy   float32 [N, 1024], L2-normalised
        Embedding/results_<variant>/chunk_ids.txt    row i = chunk_ids[i]
        Embedding/results_<variant>/manifest.json    model, versions, device, timing, text fingerprint
Then run Embedding/analyze.py --variant <variant> to check the vectors and write the report.

An interrupted run resumes: finished batches are kept in Embedding/.partial/.
"""
import argparse
import json
import platform
import shutil
import sys
import time
from datetime import datetime

import numpy as np

from variants import HERE, VARIANTS, fingerprint, load_texts, results_dir

DEFAULT_MODEL = "BAAI/bge-m3"
PARTIAL = HERE / ".partial"


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--batch-size", type=int, default=0, help="default: 32 on GPU, 8 on CPU")
    ap.add_argument("--max-length", type=int, default=1024, help="tokens; longer texts are truncated")
    ap.add_argument("--dry-run", action="store_true", help="random vectors, to test the plumbing")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ids, _, texts = load_texts(args.variant)
    n = len(ids)
    if len(set(ids)) != n or any(not t.strip() for t in texts):
        sys.exit("ERROR: duplicate chunk ids or empty texts in chunks.parquet")
    fp = fingerprint(ids, texts)
    out = results_dir(args.variant) if not args.dry_run else HERE / f".dryrun_{args.variant}"
    print(f"Variant {args.variant}: {n:,} chunks (text fingerprint {fp[:12]}) → {out.relative_to(HERE.parent)}")
    t0 = time.time()

    if args.dry_run:
        device, versions, model_rev, token_lengths = "none (dry run)", {}, None, []
        emb = np.random.default_rng(0).standard_normal((n, 1024)).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    else:
        import torch
        import sentence_transformers
        import transformers
        from sentence_transformers import SentenceTransformer

        device = pick_device(args.device)
        bs = args.batch_size or (32 if device == "cuda" else 16 if device == "mps" else 8)
        print(f"Device: {device}, batch size {bs}. Loading {args.model} (first run downloads ~2.3 GB) …")
        model = SentenceTransformer(args.model, device=device)
        model.max_seq_length = args.max_length
        dim = model.get_sentence_embedding_dimension()
        versions = {"python": platform.python_version(), "torch": torch.__version__,
                    "sentence_transformers": sentence_transformers.__version__,
                    "transformers": transformers.__version__, "numpy": np.__version__}
        cfg = getattr(getattr(model[0], "auto_model", None), "config", None)
        model_rev = getattr(cfg, "_commit_hash", None)

        # Self-test: related Czech sentences should score higher than unrelated ones.
        test = model.encode(["Stát by neměl regulovat drogy.", "Drogová prohibice státu škodí lidem.",
                             "Dnes je hezké počasí na procházku."], normalize_embeddings=True)
        close, far = float(test[0] @ test[1]), float(test[0] @ test[2])
        print(f"Self-test: related {close:.2f} vs unrelated {far:.2f} "
              f"{'OK' if close > far else 'UNEXPECTED - check the install'}")

        tok = model.tokenizer
        token_lengths = [len(tok(t, add_special_tokens=True)["input_ids"]) for t in texts]
        order = sorted(range(n), key=lambda i: -token_lengths[i])   # longest first
        run_dir = PARTIAL / f"{fp[:12]}_{args.model.replace('/', '_')}_{args.max_length}_bs{bs}"
        run_dir.mkdir(parents=True, exist_ok=True)
        batches = [order[i:i + bs] for i in range(0, n, bs)]
        emb = np.zeros((n, dim), dtype=np.float32)
        done = 0
        for b, idx in enumerate(batches):
            part = run_dir / f"batch_{b:05d}.npy"
            if part.exists():
                emb[idx] = np.load(part)
            else:
                vec = model.encode([texts[i] for i in idx], batch_size=bs, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False)
                np.save(part, vec.astype(np.float32))
                emb[idx] = vec
            done += len(idx)
            if b % 10 == 0 or done == n:
                el = time.time() - t0
                print(f"  {done:>5}/{n} chunks  {el / 60:5.1f} min elapsed, "
                      f"~{el / done * (n - done) / 60:5.1f} min left", flush=True)

    norms = np.linalg.norm(emb, axis=1)
    if not np.all(np.isfinite(emb)) or abs(norms.mean() - 1) > 1e-3:
        sys.exit("ERROR: embeddings are not finite / not normalised")

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", emb)
    (out / "chunk_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8", newline="\n")
    manifest = {
        "variant": args.variant,
        "text_column": VARIANTS[args.variant],
        "text_fingerprint": fp,
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
        "versions": versions,
        "platform": platform.platform(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not args.dry_run:
        shutil.rmtree(PARTIAL, ignore_errors=True)
    print(f"\nDone in {manifest['seconds'] / 60:.1f} min; {manifest['truncated_chunks']} chunks truncated.")
    print(f"Next: python Embedding/analyze.py --variant {args.variant}"
          + (" --dry-run" if args.dry_run else ""))


if __name__ == "__main__":
    main()
