"""Check a returned results zip and join the vectors with the chunk metadata.

    ..\\.venv\\Scripts\\python import_results.py ..\\results_20261111_2033.zip

Checks: input checksum matches data/chunks.parquet, embeddings checksum, shape,
chunk ids and order, finite and normalised vectors. Then writes
    results/embeddings.npy, results/chunk_ids.txt, results/manifest.json
    results/chunks_embedded.parquet   all chunk metadata + an `embedding` column
    results/embedding_report.md       checks and quality sanity tests
"""
import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "chunks.parquet"
FULL = HERE.parent / "Preprocessing" / "output" / "chunks.parquet"
RES = HERE / "results"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main(zpath: Path):
    sys.stdout.reconfigure(encoding="utf-8")
    z = zipfile.ZipFile(zpath)
    man = json.loads(z.read("manifest.json"))
    emb_bytes = z.read("embeddings.npy")
    ids = z.read("chunk_ids.txt").decode("utf-8").split()
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, ok, detail))
        print(f"  [{'OK' if ok else 'FAIL'}] {name} {detail}")

    print(f"Checking {zpath.name}")
    check("not a dry run", not man.get("dry_run"))
    check("input checksum matches data/chunks.parquet", man["input_sha256"] == sha(DATA.read_bytes()))
    check("embeddings checksum", man["embeddings_sha256"] == sha(emb_bytes))
    RES.mkdir(exist_ok=True)
    (RES / "embeddings.npy").write_bytes(emb_bytes)
    emb = np.load(RES / "embeddings.npy")
    data_ids = pq.read_table(DATA, columns=["chunk_id"]).column("chunk_id").to_pylist()
    check("shape", emb.shape == (len(data_ids), man["dim"]), str(emb.shape))
    check("chunk ids and order match", ids == data_ids)
    check("finite", bool(np.isfinite(emb).all()))
    norms = np.linalg.norm(emb, axis=1)
    check("normalised", bool(np.allclose(norms, 1, atol=1e-3)), f"(norm {norms.min():.4f}–{norms.max():.4f})")
    check("no truncated chunks", man["truncated_chunks"] == 0, f"(longest {man['token_length_max']} tokens)")
    if not all(ok for _, ok, _ in checks):
        sys.exit("Some checks failed; results not imported.")

    (RES / "chunk_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    (RES / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")

    # Join with the full chunk metadata from preprocessing.
    full = pq.read_table(FULL).to_pylist()
    by_id = {r["chunk_id"]: r for r in full}
    missing = [i for i in ids if i not in by_id]
    if missing:
        sys.exit(f"{len(missing)} chunk ids are not in {FULL} (preprocessing rerun since packaging?)")
    rows = [dict(by_id[i], embedding=emb[k].tolist()) for k, i in enumerate(ids)]
    pq.write_table(pa.Table.from_pylist(rows), RES / "chunks_embedded.parquet", compression="zstd")

    report(emb, rows, man, checks)
    print(f"Imported to {RES}")


def report(emb, rows, man, checks):
    n = len(rows)
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1)
    src = np.array([r["source"] for r in rows])
    doc = np.array([r["doc_id"] for r in rows])
    title = [r["title"] for r in rows]
    L = ["# Embedding report", "",
         f"Model **{man['model']}** on **{man['device']}** ({man['platform']}), {man['seconds']} s, "
         f"{man['count']:,} chunks × {man['dim']} dimensions. Created {man['created']}.", "",
         "## Checks", "", "| Check | Result | Detail |", "|---|---|---|"]
    L += [f"| {c} | {'OK' if ok else 'FAIL'} | {d} |" for c, ok, d in checks]

    off = sim[np.triu_indices(n, 1)]
    q = np.percentile(off, [1, 25, 50, 75, 99])
    L += ["", "## Similarity distribution", "",
          "Cosine similarity between all pairs of different chunks:", "",
          "| 1% | 25% | median | 75% | 99% |", "|---:|---:|---:|---:|---:|",
          "| " + " | ".join(f"{v:.3f}" for v in q) + " |"]

    # Do neighbours cluster by source (writing style) rather than topic?
    k = 10
    nn = np.argsort(-sim, axis=1)[:, :k]
    L += ["", "## Do chunks cluster by source?", "",
          f"For each chunk, the share of its {k} nearest neighbours that come from the same source, "
          "compared with the share expected if neighbours were picked at random. A large gap means the "
          "embedding partly encodes *what kind of text* it is (book prose vs. video blurb), not only the topic.", "",
          "| Source | Chunks | Same-source neighbours | Expected at random |", "|---|---:|---:|---:|"]
    stats = {}
    for s in ("article", "book", "video"):
        m = src == s
        same = float((src[nn[m]] == s).mean())
        exp = (m.sum() - 1) / (n - 1)
        stats[s] = (same, exp)
        L.append(f"| {s} | {m.sum():,} | {same:.0%} | {exp:.0%} |")
    # Remedy without re-embedding: subtract each source's mean vector.
    cen = emb.copy()
    for s in ("article", "book", "video"):
        cen[src == s] -= cen[src == s].mean(0)
    cen /= np.linalg.norm(cen, axis=1, keepdims=True)
    np.save(RES / "embeddings_source_centred.npy", cen.astype(np.float32))
    csim = cen @ cen.T
    np.fill_diagonal(csim, -1)
    cnn = np.argsort(-csim, axis=1)[:, :k]
    L += ["", "**Source-centred vectors** (`embeddings_source_centred.npy`): each source's average vector is "
          "subtracted, then vectors are re-normalised. This removes the shared “book-ness” / “video-blurb-ness” "
          "and lets topics match across sources. Use them for the 2D map; for scoring a question against "
          "chunks, the raw vectors are the ones the model was trained for.", "",
          "| Source | Same-source neighbours, raw | Centred | Expected at random |", "|---|---:|---:|---:|"]
    for s in ("article", "book", "video"):
        m = src == s
        L.append(f"| {s} | {stats[s][0]:.0%} | {float((src[cnn[m]] == s).mean()):.0%} | {stats[s][1]:.0%} |")
    L += ["", "Part of the remaining clustering likely comes from the context line at the start of each "
          "chunk (“Kniha …”, “Video „…“”, “Článek …”), which names the source outright. Re-embedding "
          "without the source word would test that."]
    sim_doc = float((doc[nn[:, 0]] == doc).mean())
    L += ["", f"Nearest neighbour is another chunk of the same document for {sim_doc:.0%} of chunks."]

    # Near-duplicates across documents
    iu = np.argwhere(np.triu(sim, 1) > 0.95)
    pairs = [(i, j) for i, j in iu if doc[i] != doc[j]]
    L += ["", "## Near-duplicates (similarity > 0.95, different documents)", "", f"{len(pairs)} pairs."]
    if pairs:
        L += ["", "| Similarity | Chunk A | Chunk B |", "|---:|---|---|"]
        for i, j in sorted(pairs, key=lambda p: -sim[p])[:15]:
            L.append(f"| {sim[i, j]:.3f} | {title[i]} ({src[i]}) | {title[j]} ({src[j]}) |")

    # Example neighbourhoods: pick seeds by title
    seeds = ["Kartely", "10. Zdravotnictví", "Drogy", "Bitcoin", "Demokracie", "Školství"]
    L += ["", "## Example neighbourhoods", "",
          "Five nearest chunks from *other* documents, for a few seed chunks. They should share the topic.", ""]
    for sd in seeds:
        cand = [i for i, t in enumerate(title) if sd.casefold() in t.casefold()]
        if not cand:
            continue
        i = cand[0]
        order = [j for j in np.argsort(-sim[i]) if doc[j] != doc[i]][:5]
        L += [f"**{title[i]}** ({src[i]})", ""]
        L += [f"- {sim[i, j]:.3f} · {src[j]} · {title[j]}" for j in order]
        L.append("")
    (RES / "embedding_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[L.index("## Do chunks cluster by source?"):]))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
