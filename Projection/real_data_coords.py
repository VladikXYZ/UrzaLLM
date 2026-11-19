"""2D map coordinates for the real_data corpus (Vladimir's chunks and embeddings), for search_map.py.

    python Projection/real_data_coords.py            # needs umap-learn (see below)

Reads   real_data/chunked.json                         2,911 chunks, Czech + English
        real_data/embeddings_out{,_en}/bge_m3_vectors.npy  bge-m3 vectors of text_cs / text_en
Writes  real_data/projection/coords_bge_m3_{cs,en}.parquet  chunk_id, source, x/y raw, x/y centred

Same method as Projection/project_compare.py: UMAP (cosine, 15 neighbours, min_dist 0.1, seed 42) on the
raw vectors, and on source-centred vectors (each source's mean vector subtracted, then renormalised, as
Embedding/analyze.py does); the centred map is rotated onto the raw one (orthogonal Procrustes).

umap-learn does not import in the main Python on this machine (old pynndescent with NumPy 2), so run it
in a throwaway environment, e.g.:
    uv run --no-project --with umap-learn --with pyarrow python Projection/real_data_coords.py
"""
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "real_data"
OUT = DATA / "projection"
SOURCES = ["article", "book", "video"]
UMAP_PARAMS = dict(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
EMB = {"cs": DATA / "embeddings_out" / "bge_m3_vectors.npy", "en": DATA / "embeddings_out_en" / "bge_m3_vectors.npy"}


def source_of(chunk: dict) -> str:
    """book / video / article. Taken from the URL: in chunked.json the `type` of YouTube descriptions is
    "article" and of stoky.urza.cz articles "video" (swapped in chunking.py)."""
    if chunk["type"] in ("ancap", "anag"):
        return "book"
    return "video" if "youtube.com" in chunk["src"] or "youtu.be" in chunk["src"] else "article"


def chunk_ids(chunks) -> list[str]:
    return [f"{source_of(c)}-{i:04d}" for i, c in enumerate(chunks)]


def align(ref, Y):
    """Rotate/flip Y onto ref (orthogonal Procrustes) so the two maps are oriented alike."""
    a, b = ref - ref.mean(0), Y - Y.mean(0)
    u, _, vt = np.linalg.svd(b.T @ a)
    return b @ (u @ vt)


def main():
    import umap
    chunks = json.loads((DATA / "chunked.json").read_text(encoding="utf-8"))
    src = np.array([source_of(c) for c in chunks])
    ids = chunk_ids(chunks)
    OUT.mkdir(exist_ok=True)
    for lang, path in EMB.items():
        raw = np.load(path).astype(np.float32)
        if len(raw) != len(chunks):
            raise SystemExit(f"{path}: {len(raw)} vectors for {len(chunks)} chunks")
        raw /= np.linalg.norm(raw, axis=1, keepdims=True)
        cen = raw.copy()
        for s in SOURCES:
            cen[src == s] -= cen[src == s].mean(0)
        cen /= np.linalg.norm(cen, axis=1, keepdims=True)
        print(f"{lang}: UMAP on raw vectors …", flush=True)
        p_raw = umap.UMAP(**UMAP_PARAMS).fit_transform(raw)
        print(f"{lang}: UMAP on source-centred vectors …", flush=True)
        p_cen = align(p_raw, umap.UMAP(**UMAP_PARAMS).fit_transform(cen))
        out = OUT / f"coords_bge_m3_{lang}.parquet"
        pq.write_table(pa.table({"chunk_id": ids, "source": src.tolist(),
                                 "x_raw": p_raw[:, 0], "y_raw": p_raw[:, 1],
                                 "x_centred": p_cen[:, 0], "y_centred": p_cen[:, 1]}), out)
        print(f"  wrote {out.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
