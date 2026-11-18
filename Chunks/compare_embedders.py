"""One report: how the two embedding models of real_data/ lay out the same chunks (bge-m3 vs EmbeddingGemma).

    python Chunks/compare_embedders.py          # from the repo root, about 2 minutes on a CPU

Reads   real_data/chunked.json, real_data/embeddings_out_cs/{bge_m3,embeddinggemma_2}_vectors.npy (+ metadata_index.json)
Writes  RealReports/embedder_comparison.html (self-contained)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RD = ROOT / "real_data"
OUT = ROOT / "RealReports" / "embedder_comparison.html"
MODELS = [("bge_m3", "bge-m3", "BAAI/bge-m3"), ("embeddinggemma_2", "EmbeddingGemma", "google/embeddinggemma-300m")]
SOURCES = ["article", "book", "video"]
K = 10
UMAP_PARAMS = dict(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)


def knn(X, k=K):
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    return np.argpartition(-S, k, axis=1)[:, :k]


def same_share(nn, lab, mask=None):
    """Share of each chunk's k neighbours with the same label, averaged per label (only rows in mask)."""
    hit = (lab[nn] == lab[:, None]).mean(1)
    rows = np.ones(len(lab), bool) if mask is None else mask
    return {s: float(hit[rows & (lab == s)].mean()) for s in np.unique(lab[rows])}


def centre(X, lab):
    Y = X.copy()
    for s in np.unique(lab):
        Y[lab == s] -= Y[lab == s].mean(0)
    return Y / np.linalg.norm(Y, axis=1, keepdims=True)


def procrustes(ref, P):
    """Rotate/reflect/scale P onto ref (both centred), so the two maps can be compared by eye."""
    a, b = ref - ref.mean(0), P - P.mean(0)
    U, s, Vt = np.linalg.svd(b.T @ a)
    R = U @ Vt
    return (b @ R) * (s.sum() / (b ** 2).sum())


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import umap
    from scipy.stats import spearmanr
    ch = json.loads((RD / "chunked.json").read_text(encoding="utf-8"))
    meta = json.loads((RD / "embeddings_out_cs" / "metadata_index.json").read_text(encoding="utf-8"))
    assert len(meta) == len(ch) and all(m.strip() == c["text_cs"].strip() for m, c in zip(meta, ch))
    src = np.array([c["type"] for c in ch])
    book = np.array([c["src"] if c["type"] == "book" else "" for c in ch])
    chap = np.array([f"{c['src']}|{c['title_cs']}" if c["type"] == "book" else "" for c in ch])
    isbook = src == "book"
    rng = np.random.default_rng(0)
    pairs = rng.integers(0, len(ch), size=(200_000, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]

    E, st, maps, nns = {}, {}, {}, {}
    for key, label, hf in MODELS:
        X = np.load(RD / "embeddings_out_cs" / f"{key}_vectors.npy").astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        E[key] = X
        C = centre(X, src)
        nn_raw, nn_cen = knn(X), knn(C)
        nns[key] = nn_raw
        Xb = X[isbook]
        nn_b = knn(Xb)
        cos = (X[pairs[:, 0]] * X[pairs[:, 1]]).sum(1)
        ev = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
        ev /= ev.sum()
        cum = np.cumsum(ev)
        between = sum((src == s).sum() * np.sum((X[src == s].mean(0) - X.mean(0)) ** 2) for s in SOURCES) \
            / np.sum((X - X.mean(0)) ** 2)
        st[key] = {
            "label": label, "hf": hf, "dim": int(X.shape[1]),
            "same_raw": same_share(nn_raw, src), "same_cen": same_share(nn_cen, src),
            "chapter_same": float((chap[isbook][nn_b] == chap[isbook][:, None]).mean()),
            "book_same": float((book[isbook][nn_b] == book[isbook][:, None]).mean()),
            "cos_mean": float(cos.mean()), "cos_p5": float(np.percentile(cos, 5)), "cos_p95": float(np.percentile(cos, 95)),
            "pc1": float(ev[0]), "pcs50": int(np.searchsorted(cum, 0.5) + 1), "pcs80": int(np.searchsorted(cum, 0.8) + 1),
            "between": float(between),
            "hist": np.histogram(cos, bins=40, range=(-0.1, 0.9))[0].tolist(),
        }
        print(label, "UMAP raw / centred …", flush=True)
        maps[key] = {"raw": umap.UMAP(**UMAP_PARAMS).fit_transform(X), "cen": umap.UMAP(**UMAP_PARAMS).fit_transform(C)}
    a, b = MODELS[0][0], MODELS[1][0]
    for v in ("raw", "cen"):           # gemma's maps turned to match bge-m3's, then both scaled to [-1, 1]
        maps[b][v] = procrustes(maps[a][v], maps[b][v])
        maps[a][v] = maps[a][v] - maps[a][v].mean(0)
        for k in (a, b):
            maps[k][v] = maps[k][v] / np.abs(maps[k][v]).max()

    ov = np.array([len(set(x) & set(y)) / K for x, y in zip(nns[a], nns[b])])
    rho = float(spearmanr((E[a][pairs[:, 0]] * E[a][pairs[:, 1]]).sum(1), (E[b][pairs[:, 0]] * E[b][pairs[:, 1]]).sum(1))[0])
    # random-pair baseline for neighbour overlap: K of n-1 at random
    cross = {"overlap": float(ov.mean()), "overlap_src": {s: float(ov[src == s].mean()) for s in SOURCES},
             "overlap_zero": float((ov == 0).mean()), "overlap_half": float((ov >= 0.5).mean()), "rho": rho,
             "random": K / (len(ch) - 1)}
    # neighbour sets per chunk on the full vectors, for the hover
    rnd = {s: float((src == s).mean()) for s in SOURCES}
    for name, lab in (("book_same", book[isbook]), ("chapter_same", chap[isbook])):
        _, c = np.unique(lab, return_counts=True)      # chance that a random other paragraph shares the label
        rnd[name] = float((c * (c - 1)).sum() / (len(lab) * (len(lab) - 1)))
    pts = {"src": [SOURCES.index(s) for s in src], "ov": [round(float(x), 1) for x in ov],
           "title": [c["title_en"] or "" for c in ch], "sub": [c["subtitle_en"] or "" for c in ch],
           "text": [(c["text_en"] or "")[:260] for c in ch],
           "book": [c["src"] if c["type"] == "book" else "" for c in ch],
           "xy": {k: {v: np.round(maps[k][v], 4).tolist() for v in ("raw", "cen")} for k, _, _ in MODELS},
           "nn": {k: nns[k][:, :5].tolist() for k, _, _ in MODELS}}
    html = PAGE
    for k, v in {"__ST__": json.dumps(st), "__CROSS__": json.dumps(cross), "__RND__": json.dumps(rnd),
                 "__PTS__": json.dumps(pts, ensure_ascii=False, separators=(",", ":"))}.items():
        html = html.replace(k, v)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print("Wrote", OUT, f"{OUT.stat().st_size / 1e6:.1f} MB")
    print(json.dumps({k: {x: y for x, y in v.items() if x != "hist"} for k, v in st.items()}, indent=1))
    print(json.dumps(cross, indent=1))


PAGE = open(Path(__file__).with_name("compare_embedders.html"), encoding="utf-8").read()

if __name__ == "__main__":
    main()
