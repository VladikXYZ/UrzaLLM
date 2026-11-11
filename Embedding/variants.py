"""Embedding input variants and shared helpers.

withsource  context line names the source: "Kniha …", "Článek „…“", "Video „…“" (first run, 2026-11-11)
nosource    the same text without that word, to test how much it makes chunks cluster by source
"""
import hashlib
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
CHUNKS = HERE.parent / "Preprocessing" / "output" / "chunks.parquet"

VARIANTS = {
    "withsource": "embed_text",
    "nosource": "embed_text_nosource",
}


def results_dir(variant: str) -> Path:
    return HERE / f"results_{variant}"


def load_texts(variant: str) -> tuple[list[str], list[str], list[str]]:
    """(chunk_ids, sources, texts) for a variant, in chunks.parquet order."""
    if not CHUNKS.exists():
        raise SystemExit(f"{CHUNKS} not found. Run Preprocessing/run.py first (or pull it from the repo).")
    t = pq.read_table(CHUNKS, columns=["chunk_id", "source", VARIANTS[variant]])
    return (t.column(0).to_pylist(), t.column(1).to_pylist(), t.column(2).to_pylist())


def fingerprint(ids: list[str], texts: list[str]) -> str:
    """Hash of exactly what was embedded (ids, order and text), independent of file format."""
    h = hashlib.sha256()
    for i, t in zip(ids, texts):
        h.update(i.encode("utf-8") + b"\t" + t.encode("utf-8") + b"\n")
    return h.hexdigest()
