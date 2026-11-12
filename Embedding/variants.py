"""Embedding input variants and shared helpers.

withsource  context line names the source: "Kniha …", "Článek „…“", "Video „…“" (first run, 2026-11-11)
nosource    the same text without that word, to test how much it makes chunks cluster by source
textonly    only the chunk's own text, no context line at all (no chapter path, title, year or guests),
            so nothing outside the text can group chunks
"""
import hashlib
import sys
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dataset import CHUNKS, results_dir  # noqa: E402,F401  (URZA_DATASET switch)

VARIANTS = {
    "withsource": "embed_text",
    "nosource": "embed_text_nosource",
    "textonly": "embed_text_textonly",
}


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
