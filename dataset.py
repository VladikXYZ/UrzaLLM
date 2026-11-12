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
