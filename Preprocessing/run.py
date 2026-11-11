"""UrzaLLM preprocessing pipeline.

Reads the raw JSON exports in jsons/ (repo root), cleans and links them, splits
them into chunks and writes the corpus plus a report to Preprocessing/output.

    ..\\.venv\\Scripts\\python run.py            (from the Preprocessing folder)
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import config as C
from articles import process_articles
from books import process_books
from chunking import chunk_documents
from textutils import quantiles
from videos import process_videos


def load(name: str):
    with open(C.DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_parquet(path: Path, rows: list[dict]):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(rows), path)


def build_stats(docs, chunks, raw_counts, src_stats, elapsed):
    by = Counter((d["source"], d["status"]) for d in docs)
    reasons = Counter((d["source"], d["exclude_reason"]) for d in docs if d["status"] == "excluded")
    kinds = Counter((d["source"], d["kind"]) for d in docs if d["status"] == "included")
    flags = Counter((d["source"], f) for d in docs if d["status"] == "included" for f in d.get("flags", []))
    inc = [d for d in docs if d["status"] == "included"]
    sources = ["article", "book", "video"]
    return {
        "run_seconds": round(elapsed, 2),
        "raw_records": raw_counts,
        "documents": {
            s: {
                "raw": raw_counts[s],
                "included": by[(s, "included")],
                "excluded": by[(s, "excluded")],
                "words_raw": sum(d["raw_words"] for d in docs if d["source"] == s),
                "words_included": sum(d["words"] for d in inc if d["source"] == s),
            } for s in sources
        },
        "exclusions": [{"source": s, "reason": r, "count": c} for (s, r), c in sorted(reasons.items())],
        "kinds": [{"source": s, "kind": k, "count": c} for (s, k), c in sorted(kinds.items())],
        "flags": [{"source": s, "flag": f, "count": c} for (s, f), c in sorted(flags.items())],
        "chunks": {
            "total": len(chunks),
            "by_source": dict(Counter(c["source"] for c in chunks)),
            "by_kind": dict(Counter(c["kind"] for c in chunks)),
            "words_total": sum(c["words"] for c in chunks),
            "words_quantiles": {s: quantiles([c["words"] for c in chunks if c["source"] == s]) for s in sources},
            "docs_split": {s: sum(1 for d in inc if d["source"] == s and
                                  sum(1 for c in chunks if c["doc_id"] == d["doc_id"]) > 1) for s in sources},
        },
        "per_year": {
            str(y): {s: {"docs": sum(1 for d in inc if d["source"] == s and d.get("year") == y),
                         "words": sum(d["words"] for d in inc if d["source"] == s and d.get("year") == y)}
                     for s in ("article", "video")}
            for y in sorted({d["year"] for d in inc if d.get("year")})
        },
        "video_hours_per_year": {
            str(y): round(sum(d["duration_min"] for d in inc if d["source"] == "video" and d["year"] == y) / 60, 1)
            for y in sorted({d["year"] for d in inc if d["source"] == "video" and d.get("year")})
        },
        "sources": src_stats,
        "config": {k: getattr(C, k) for k in dir(C)
                   if k.isupper() and isinstance(getattr(C, k), (int, float, str, list, dict))
                   and k not in ("BOOKS",)},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=C.DATA_DIR, help="folder with the raw JSON files")
    ap.add_argument("--out", type=Path, default=C.OUTPUT_DIR, help="output folder")
    ap.add_argument("--no-parquet", action="store_true", help="write JSONL only")
    ap.add_argument("--no-report", action="store_true", help="skip the HTML/Markdown report")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    C.DATA_DIR = args.data
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"Reading {C.DATA_DIR}")
    raw_videos = load(C.VIDEOS_FILE)
    raw_articles = load(C.ARTICLES_FILE)

    videos, vstats = process_videos(raw_videos)
    print(f"  videos:   {len(videos)} processed")
    articles, astats = process_articles(raw_articles, videos)
    print(f"  articles: {len(articles)} processed")
    books, bstats = process_books()
    print(f"  books:    {len(books)} sections processed")

    docs = articles + books + videos
    chunks = chunk_documents(docs)
    print(f"  chunks:   {len(chunks)}")

    raw_counts = {"article": len(raw_articles), "video": len(raw_videos), "book": len(books)}
    link_log = astats.pop("link_log")
    stats = build_stats(docs, chunks, raw_counts,
                        {"articles": astats, "videos": vstats, "books": bstats},
                        time.time() - t0)

    write_jsonl(out / "documents.jsonl", docs)
    write_jsonl(out / "chunks.jsonl", chunks)
    write_jsonl(out / "article_video_links.jsonl", link_log)
    if not args.no_parquet:
        write_parquet(out / "documents.parquet", docs)
        write_parquet(out / "chunks.parquet", chunks)
    if not args.no_report:
        from report import build_report
        md, html = build_report(docs, chunks, link_log, stats, out)   # may add figures to stats
        print(f"Report: {html}\n        {md}")
    with open(out / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Done in {time.time() - t0:.1f}s → {out}")


if __name__ == "__main__":
    main()
