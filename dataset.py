"""Which corpus the embedding and projection scripts read, and where they write.

Default (no environment variable): the original pipeline
    chunks      Preprocessing/output/chunks.parquet
    embeddings  Embedding/results_<variant>/
    reports     Projection/output/

URZA_DATASET=final: the final chunks made from preprocessed/ (Chunks/chunk.py)
    chunks      Chunks/output/chunks.parquet
    embeddings  Chunks/output/results_<variant>/
    reports     Chunks/output/projection/   (copied to FinalReports/ by Chunks/build_reports.py)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FINAL = os.environ.get("URZA_DATASET", "").strip().lower() == "final"

CHUNKS = (ROOT / "Chunks" / "output" / "chunks.parquet") if FINAL else (ROOT / "Preprocessing" / "output" / "chunks.parquet")
OUT_DIR = (ROOT / "Chunks" / "output" / "projection") if FINAL else (ROOT / "Projection" / "output")


def results_dir(variant: str) -> Path:
    return (ROOT / "Chunks" / "output" / f"results_{variant}") if FINAL else (ROOT / "Embedding" / f"results_{variant}")


# ---- Display language. The final reports are for an English audience: the final chunks carry the English twin of
# each chunk (text_en, title_en, chapter_en, subchapter_en), shown instead of the Czech; everything computed (the
# embeddings, the embedded texts) stays Czech. The earlier pipeline has no English, so it shows Czech as before.
ENGLISH = FINAL
DISPLAY = ("text", "title", "chapter", "subchapter")
_NAMES = None


def read_chunks(path=None, columns=None):
    """Rows of a chunks parquet (default: CHUNKS) as dicts, with the display columns in the display language."""
    import pyarrow.parquet as pq
    path = path or CHUNKS
    have = set(pq.read_schema(path).names)
    en = ENGLISH and "text_en" in have
    cols = None if columns is None else list(dict.fromkeys(
        columns + ([f"{c}_en" for c in DISPLAY if c in columns and f"{c}_en" in have] if en else [])))
    return to_display(pq.read_table(path, columns=cols).to_pylist())


def to_display(rows):
    """Swap the display columns of chunk rows (dicts) to the display language, in place; returns rows."""
    if ENGLISH:
        for r in rows:
            for c in DISPLAY:
                if c in r and f"{c}_en" in r and (r[f"{c}_en"] is not None or c in ("chapter", "subchapter")):
                    r[c] = r[f"{c}_en"]
    return rows


def nm(name):
    """A book section named in Czech in the code (e.g. "Tři příběhy"), in the display language."""
    global _NAMES
    if not ENGLISH:
        return name
    if _NAMES is None:
        import json
        p = ROOT / "Chunks" / "output" / "names_en.json"
        _NAMES = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return _NAMES.get(name, name)
