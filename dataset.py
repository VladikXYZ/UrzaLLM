"""Which corpus the embedding and projection scripts read, and where they write.

Default (no environment variable): the original pipeline
    chunks      Preprocessing/output/chunks.parquet
    embeddings  Embedding/results_<variant>/
    reports     Projection/output/

URZA_DATASET=final: the final chunks made from preprocessed/ (Chunks/chunk.py)
    chunks      Chunks/output/chunks.parquet
    embeddings  Chunks/output/results_<variant>/
    reports     Chunks/output/projection/   (copied to FinalReports/ by Chunks/build_reports.py)

URZA_DATASET=real: real_data/ (the friend's cleaned chunks and embeddings, Chunks/real_data.py), one model at a time
chosen by URZA_MODEL (bge_m3, the default, or embeddinggemma_2); only the textonly variant exists
    chunks      Chunks/output_real/chunks.parquet
    embeddings  Chunks/output_real/<model>/results_textonly/
    reports     Chunks/output_real/<model>/projection/   (copied to RealReports/<model>/ by Chunks/build_real_reports.py)
"""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_DS = os.environ.get("URZA_DATASET", "").strip().lower()
FINAL = _DS == "final"
REAL = _DS == "real"
MODEL = os.environ.get("URZA_MODEL", "bge_m3").strip() if REAL else "bge_m3"
MODELS = {  # real_data model folder name -> (label shown in reports, Hugging Face id, dimension)
    "bge_m3": ("bge-m3", "BAAI/bge-m3", 1024),
    "embeddinggemma_2": ("EmbeddingGemma", "google/embeddinggemma-2", 768),
}
MODEL_LABEL, MODEL_ID, DIM = MODELS[MODEL]
REAL_DIR = ROOT / "Chunks" / "output_real"

if REAL:
    CHUNKS = REAL_DIR / "chunks.parquet"
    OUT_DIR = REAL_DIR / MODEL / "projection"
elif FINAL:
    CHUNKS = ROOT / "Chunks" / "output" / "chunks.parquet"
    OUT_DIR = ROOT / "Chunks" / "output" / "projection"
else:
    CHUNKS = ROOT / "Preprocessing" / "output" / "chunks.parquet"
    OUT_DIR = ROOT / "Projection" / "output"
# political-space items embedded with the same model (Questions/embed_questions.py; other models: Questions/output_<model>/)
QUESTIONS_DIR = ROOT / "Questions" / ("output" if MODEL == "bge_m3" else f"output_{MODEL}")


def results_dir(variant: str) -> Path:
    if REAL:
        return REAL_DIR / MODEL / f"results_{variant}"
    return (ROOT / "Chunks" / "output" / f"results_{variant}") if FINAL else (ROOT / "Embedding" / f"results_{variant}")


# ---- Display language. The final reports are for an English audience: the final chunks carry the English twin of
# each chunk (text_en, title_en, chapter_en, subchapter_en), shown instead of the Czech; everything computed (the
# embeddings, the embedded texts) stays Czech. The earlier pipeline has no English, so it shows Czech as before.
ENGLISH = FINAL or REAL
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
    if REAL:
        for r in rows:
            for c in ("chapter", "subchapter", "title"):     # only book chapters/titles carry "10. "-style numbers
                if isinstance(r.get(c), str):
                    r[c] = unnumber(r[c])
    return rows


# ---- real_data reports show book chapters without their numbering ("10. Healthcare" -> "Healthcare"). Where two
# chapters would then share a name (both books have "Education"), the book name is added in brackets.
_NUM = re.compile(r"^\d+\.\s+")
_UNNUM = None


def unnumber(name):
    global _UNNUM
    if not REAL or not isinstance(name, str) or not _NUM.match(name):
        return name
    if _UNNUM is None:
        import pyarrow.parquet as pq
        t = pq.read_table(CHUNKS, columns=["source", "book", "chapter", "chapter_en"]).to_pylist()
        seen = {}
        for r in t:
            if r["source"] == "book":
                for n in {r["chapter"], r["chapter_en"]} - {None}:
                    if _NUM.match(n):
                        seen.setdefault(_NUM.sub("", n), set()).add((n, r["book"]))
        _UNNUM = {}
        for plain, origs in seen.items():
            books = {b for _, b in origs}
            for n, b in origs:
                _UNNUM[n] = f"{plain} ({b})" if len(books) > 1 else plain
    return _UNNUM.get(name, _NUM.sub("", name))


def nm(name):
    """A book section named in Czech in the code (e.g. "Tři příběhy"), in the display language."""
    global _NAMES
    if not ENGLISH:
        return name
    if _NAMES is None:
        import json
        p = ROOT / "Chunks" / "output" / "names_en.json"
        _NAMES = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return unnumber(_NAMES.get(name, name))
