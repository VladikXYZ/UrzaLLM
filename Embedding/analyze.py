"""Check a variant's embeddings and write the analysis files and report.

    python Embedding/analyze.py --variant nosource      # from the repo root

Reads   Embedding/results_<variant>/{embeddings.npy, chunk_ids.txt, manifest.json}
        Preprocessing/output/chunks.parquet
Checks  the embedded text is exactly the current chunks (text fingerprint), ids and order,
        shape, finite and normalised vectors, no truncation.
Writes  Embedding/results_<variant>/chunks_embedded.parquet      chunk metadata + `embedding` column
        Embedding/results_<variant>/embeddings_source_centred.npy each source's mean subtracted
        Embedding/results_<variant>/embedding_report.md          checks and quality sanity tests
"""
import argparse
import json
import sys
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from variants import CHUNKS, HERE, VARIANTS, fingerprint, load_texts, results_dir
from dataset import ENGLISH, nm, to_display  # noqa: E402  (variants puts the repo root on sys.path)


def main():
    global RES
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--dry-run", action="store_true", help="analyse the dry-run output instead")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    RES = results_dir(args.variant) if not args.dry_run else HERE / f".dryrun_{args.variant}"
    if not (RES / "manifest.json").exists():
        sys.exit(f"{RES} has no results. Run: python Embedding/embed.py --variant {args.variant}")
    man = json.loads((RES / "manifest.json").read_text(encoding="utf-8"))
    emb = np.load(RES / "embeddings.npy")
    ids = (RES / "chunk_ids.txt").read_text(encoding="utf-8").split()
    cur_ids, _, texts = load_texts(args.variant)
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, ok, detail))
        print(f"  [{'OK' if ok else 'FAIL'}] {name} {detail}")

    print(f"Checking {RES.relative_to(HERE.parent)}")
    if not args.dry_run:
        check("not a dry run", not man.get("dry_run"))
    check("variant matches", man.get("variant") == args.variant, f"({man.get('variant')})")
    check("embedded text = current chunks.parquet", man.get("text_fingerprint") == fingerprint(cur_ids, texts),
          "" if man.get("text_fingerprint") == fingerprint(cur_ids, texts)
          else "(preprocessing changed since embedding: re-embed)")
    check("chunk ids and order match", ids == cur_ids)
    check("shape", emb.shape == (len(cur_ids), man["dim"]), str(emb.shape))
    check("finite", bool(np.isfinite(emb).all()))
    norms = np.linalg.norm(emb, axis=1)
    check("normalised", bool(np.allclose(norms, 1, atol=1e-3)), f"(norm {norms.min():.4f}-{norms.max():.4f})")
    check("no truncated chunks", man["truncated_chunks"] == 0, f"(longest {man['token_length_max']} tokens)")
    if not all(ok for _, ok, _ in checks):
        sys.exit("Some checks failed.")

    full = pq.read_table(CHUNKS).to_pylist()
    rows = [dict(r, embedding=emb[k].tolist()) for k, r in enumerate(full)]
    pq.write_table(pa.Table.from_pylist(rows), RES / "chunks_embedded.parquet", compression="zstd")
    to_display(rows)                       # report shows English text/titles for the final corpus
    report(emb, rows, man, checks)
    print(f"Wrote analysis to {RES}")


def report(emb, rows, man, checks):
    n = len(rows)
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1)
    src = np.array([r["source"] for r in rows])
    doc = np.array([r["doc_id"] for r in rows])
    title = [r["title"] for r in rows]
    L = [f"# Embedding report: variant `{man.get('variant')}`", "",
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
          "and helps topics match across sources. Use them for the 2D map; for scoring a question against "
          "chunks, the raw vectors are the ones the model was trained for.", "",
          "| Source | Same-source neighbours, raw | Centred | Expected at random |", "|---|---:|---:|---:|"]
    for s in ("article", "book", "video"):
        m = src == s
        L.append(f"| {s} | {stats[s][0]:.0%} | {float((src[cnn[m]] == s).mean()):.0%} | {stats[s][1]:.0%} |")
    variant = man.get("variant")
    if variant == "withsource":
        L += ["", "Part of the remaining clustering likely comes from the context line at the start of each "
              "chunk (“Kniha …”, “Video „…“”, “Článek …” or “Oznámení …”), which names the source outright; the `nosource` "
              "and `textonly` variants test that."]
    elif variant == "nosource":
        L += ["", "Here the context line no longer names the source, but still carries a chapter path or a "
              "title and year that differ by source; the `textonly` variant removes it entirely."]
    else:
        L += ["", "Nothing is put in front of the chunks here, so the remaining clustering comes from the texts "
              "themselves: how each source writes (book prose, article, video description), their length and "
              "their typical subjects."]
    same_doc = doc[nn[:, 0]] == doc
    multi = [s for s in ("article", "book", "video") if len(set(doc[src == s])) < (src == s).sum()]
    L += ["", f"Nearest neighbour is another chunk of the same document for {float(same_doc.mean()):.0%} of chunks"
          + (". " + ("Only " if len(multi) < 3 else "All sources (")
             + " and ".join([", ".join(f"{s}s" for s in multi[:-1]), f"{multi[-1]}s"] if len(multi) > 1 else [f"{multi[0]}s"])
             + ("" if len(multi) < 3 else ")") + " have documents split into several chunks; per source, the share is "
             + ", ".join(f"{float(same_doc[src == s].mean()):.0%} for {s}s" for s in multi) + "."
             if multi else " (no document is split into several chunks).")]

    # Near-duplicates across documents
    iu = np.argwhere(np.triu(sim, 1) > 0.95)
    pairs = [(i, j) for i, j in iu if doc[i] != doc[j]]
    L += ["", "## Near-duplicates (similarity > 0.95, different documents)", "", f"{len(pairs)} pairs."]
    if pairs:
        L += ["", "| Similarity | Chunk A | Chunk B |", "|---:|---|---|"]
        for i, j in sorted(pairs, key=lambda p: -sim[p])[:15]:
            L.append(f"| {sim[i, j]:.3f} | {title[i]} ({src[i]}) | {title[j]} ({src[j]}) |")

    # Example neighbourhoods: pick seeds by title
    seeds = ([nm(x) for x in ("10. Zdravotnictví", "Drogy")] + ["Cartels", "Bitcoin", "Democracy", "Education"]
             if ENGLISH else ["Kartely", "10. Zdravotnictví", "Drogy", "Bitcoin", "Demokracie", "Školství"])
    found = [sd for sd in seeds if any(sd.casefold() in t.casefold() for t in title)]
    L += ["", "## Example neighbourhoods", "",
          "Five nearest chunks from *other* documents, for the first chunk whose title contains each of: "
          + ", ".join(f"“{sd}”" for sd in found) + ". Ideally they share its topic; announcements and episodes "
          "of a recurring series often match other items of the same series instead.", ""]
    for sd in found:
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
    main()
