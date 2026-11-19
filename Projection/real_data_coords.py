"""2D map coordinates for search_map.py: Vladimir's corpus, and the same plus the video transcript chunks.

    uv run --no-project --with umap-learn --with pyarrow python Projection/real_data_coords.py [--model embeddinggemma_2]

Reads   real_data/chunked.json                               2,911 chunks (books, articles, video descriptions)
        real_data/embeddings_out_cs/bge_m3_vectors.npy       their bge-m3 vectors (of text_cs)
        real_data/transcripts_urza.json                      57,693 chunks of Urza's speech (Transcription/)
        embeddings_out_rollo_cs/bge_m3_vectors.npy           their bge-m3 vectors (of text_cs, same order)
Writes  real_data/projection/coords_<model>_cs.parquet       the 2,911 chunks only
        real_data/projection/coords_<model>_all.parquet      all 60,604 together
        (chunk_id, source, x/y raw, x/y centred); <model> = bge_m3 (default) or embeddinggemma_2 (--model), the
        other vector file in the same folders

Same method as Projection/project_compare.py: UMAP (cosine, 15 neighbours, min_dist 0.1, seed 42) on the
raw vectors, and on source-centred vectors (each source's mean vector subtracted, then renormalised, as
Embedding/analyze.py does); the centred map is rotated onto the raw one (orthogonal Procrustes).

umap-learn does not import in the main Python on this machine (old pynndescent with NumPy 2), hence uv run.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "real_data"
OUT = DATA / "projection"
SOURCES = ["article", "book", "video", "transcript"]
UMAP_PARAMS = dict(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
TRANSCRIPTS = DATA / "transcripts_urza.json"
# the vector files of each embedding model: (Vladimir's chunks, the transcript chunks), same chunk order
MODELS = {"bge_m3": "bge_m3_vectors.npy", "embeddinggemma_2": "embeddinggemma_2_vectors.npy"}


def emb_paths(model="bge_m3"):
    f = MODELS[model]
    return DATA / "embeddings_out_cs" / f, ROOT / "embeddings_out_rollo_cs" / f


EMB_CS, EMB_TRANSCRIPTS = emb_paths("bge_m3")


def coords_path(model, corpus):
    """bge-m3 keeps its original file names (coords_bge_m3_all.parquet)."""
    return OUT / f"coords_{model}_{corpus}.parquet"


def source_of(chunk: dict) -> str:
    """book / video / article / transcript. Books: type "book" with the book in src (current chunked.json) or
    type "ancap"/"anag" (older). Videos vs articles from the URL, since older chunked.json files had those two
    types swapped."""
    if chunk["type"] in ("transcript", "transcription"):
        return "transcript"
    if chunk["type"] in ("book", "ancap", "anag"):
        return "book"
    return "video" if "youtube.com" in chunk["src"] or "youtu.be" in chunk["src"] else "article"


def chunk_ids(chunks) -> list[str]:
    return [c["id"] if c.get("type") == "transcript" and c.get("id") else f"{source_of(c)}-{i:04d}"
            for i, c in enumerate(chunks)]


def load(with_transcripts: bool, model="bge_m3"):
    """(chunks, unit vectors), Vladimir's chunks first, then the transcript chunks."""
    emb_cs, emb_tr = emb_paths(model)
    chunks = json.loads((DATA / "chunked.json").read_text(encoding="utf-8"))
    vec = [np.load(emb_cs).astype(np.float32)]
    if len(vec[0]) != len(chunks):
        raise SystemExit(f"{emb_cs}: {len(vec[0])} vectors for {len(chunks)} chunks")
    if with_transcripts:
        tr = json.loads(TRANSCRIPTS.read_text(encoding="utf-8"))
        tv = np.load(emb_tr).astype(np.float32)
        if len(tv) != len(tr):
            raise SystemExit(f"{emb_tr}: {len(tv)} vectors for {len(tr)} transcript chunks")
        chunks, vec = chunks + tr, vec + [tv]
    v = np.concatenate(vec)
    return chunks, v / np.linalg.norm(v, axis=1, keepdims=True)


def align(ref, Y):
    """Rotate/flip Y onto ref (orthogonal Procrustes) so the two maps are oriented alike."""
    a, b = ref - ref.mean(0), Y - Y.mean(0)
    u, _, vt = np.linalg.svd(b.T @ a)
    return b @ (u @ vt)


def main():
    import argparse
    import pyarrow as pa
    import pyarrow.parquet as pq
    import umap
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="bge_m3")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for name, with_tr in (("cs", False), ("all", True)):
        chunks, raw = load(with_tr, args.model)
        src = np.array([source_of(c) for c in chunks])
        cen = raw.copy()
        for s in SOURCES:
            if (src == s).any():
                cen[src == s] -= cen[src == s].mean(0)
        cen /= np.linalg.norm(cen, axis=1, keepdims=True)
        print(f"{name}: {len(chunks):,} chunks; UMAP on raw vectors …", flush=True)
        p_raw = umap.UMAP(**UMAP_PARAMS).fit_transform(raw)
        print(f"{name}: UMAP on source-centred vectors …", flush=True)
        p_cen = align(p_raw, umap.UMAP(**UMAP_PARAMS).fit_transform(cen))
        out = coords_path(args.model, name)
        pq.write_table(pa.table({"chunk_id": chunk_ids(chunks), "source": src.tolist(),
                                 "x_raw": p_raw[:, 0], "y_raw": p_raw[:, 1],
                                 "x_centred": p_cen[:, 0], "y_centred": p_cen[:, 1]}), out)
        print(f"  wrote {out.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
