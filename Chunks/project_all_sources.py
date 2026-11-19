"""One map of everything: articles, book paragraphs, video descriptions and Urza's spoken words (video transcript
chunks), raw vectors (left) and each source's average removed (right), for bge-m3 and EmbeddingGemma (switch).

    python Chunks/project_all_sources.py        # from the repo root; ~10-20 minutes (4 UMAPs of ~94k points)

Reads   real_data/chunked.json + embeddings_out_cs/<model>_vectors.npy          the written chunks
        embeddings_out_new_cs/trans_chunks.json + <model>_vectors.npy           the video transcript chunks
Writes  RealReports/projection_all_sources.html (self-contained; template Chunks/project_all_sources.html). The page
        has window.setModel(key) so Chunks/presentation_real.py can switch it from outside.
        Chunks/output_real/all_sources_cache_<model>.npz (the two maps per model, reused unless --recompute)
EmbeddingGemma's maps are rotated and mirrored to line up with bge-m3's (Procrustes); their shape is unchanged.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Projection"))
from labels import LABEL_JS  # noqa: E402  (shared label placer: every label shown, leader lines when moved)

NUM = re.compile(r"^\s*\d+\.\s*")
ROOT = Path(__file__).resolve().parent.parent
RD = ROOT / "real_data"
OUT = ROOT / "RealReports" / "projection_all_sources.html"
NEW = ROOT / "embeddings_out_new_cs"
CACHE = ROOT / "Chunks" / "output_real" / "all_sources_cache_{}.npz"
MODELS = [("bge_m3", "bge-m3"), ("embeddinggemma_2", "EmbeddingGemma")]
SOURCES = ["article", "book", "video", "transcription"]          # colour order of the earlier reports + transcripts
UMAP_PARAMS = dict(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42, low_memory=True)


def source_of(r):
    if r["type"] == "transcription":
        return "transcription"
    if r["type"] in ("book", "ancap", "anag"):
        return "book"
    return "video" if "youtu" in (r["src"] or "") else "article"


def load():
    ch = json.loads((RD / "chunked.json").read_text(encoding="utf-8"))
    tr = json.loads((NEW / "trans_chunks.json").read_text(encoding="utf-8"))
    return ch + tr


def vectors(key, n):
    X = np.concatenate([np.load(RD / "embeddings_out_cs" / f"{key}_vectors.npy"),
                        np.load(NEW / f"{key}_vectors.npy")]).astype(np.float32)
    assert len(X) == n, f"{key}: {len(X)} vectors for {n} chunks"
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def centre(X, lab):
    Y = X.copy()
    for s in range(len(SOURCES)):
        Y[lab == s] -= Y[lab == s].mean(0)
    return Y / np.linalg.norm(Y, axis=1, keepdims=True)


def procrustes(ref, P):
    a, b = ref - ref.mean(0), P - P.mean(0)
    U, s, Vt = np.linalg.svd(b.T @ a)
    return (b @ (U @ Vt)) * (s.sum() / (b ** 2).sum())


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    rows = load()
    lab = np.array([SOURCES.index(source_of(r)) for r in rows])
    res = {}
    for key, label in MODELS:
        cache = Path(str(CACHE).format(key))
        if cache.exists() and not args.recompute and len(np.load(cache)["lab"]) == len(rows):
            z = np.load(cache)
            res[f"{key}_raw"], res[f"{key}_cen"] = z["raw"], z["cen"]
            continue
        import umap
        X = vectors(key, len(rows))
        for v, Y in (("raw", X), ("cen", centre(X, lab))):
            t0 = time.time()
            res[f"{key}_{v}"] = umap.UMAP(**UMAP_PARAMS).fit_transform(Y).astype(np.float32)
            print(f"{label} UMAP {v}: {time.time() - t0:.0f} s", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, lab=lab, raw=res[f"{key}_raw"], cen=res[f"{key}_cen"])
    a = MODELS[0][0]
    for key, _ in MODELS:                                  # every map turned to line up with bge-m3's raw map
        for v in ("raw", "cen"):
            if (key, v) != (a, "raw"):
                res[f"{key}_{v}"] = procrustes(res[f"{a}_raw"], res[f"{key}_{v}"])
    for k in res:
        res[k] = res[k] - res[k].mean(0)
        res[k] = res[k] / np.abs(res[k]).max()

    # hover text: titles once per document, a short English snippet per chunk (transcripts: what Urza says)
    titles, urls, tix, doc = [], [], {}, []
    for r in rows:
        t = r["title_en"] or r["title_cs"] or ""
        if source_of(r) == "book" and r.get("subtitle_en"):
            t += " · " + r["subtitle_en"]
        u = r["src"] if (r["src"] or "").startswith("http") else ""
        if (t, u) not in tix:
            tix[t, u] = len(titles)
            titles.append(t)
            urls.append(u)
        doc.append(tix[t, u])
    snip = [(r["text_en"] or r["text_cs"] or "")[:110] for r in rows]
    tm = [int(float(r["time"])) if r["type"] == "transcription" else -1 for r in rows]
    counts = {s: int((lab == i).sum()) for i, s in enumerate(SOURCES)}
    n_vid = len({r["src"] for r in rows if r["type"] == "transcription"})
    xy = {k: np.round(v * 10000).astype(int).ravel().tolist() for k, v in res.items()}
    data = {"lab": lab.tolist(), "doc": doc, "titles": titles, "urls": urls, "snip": snip, "tm": tm, "xy": xy}
    chapters, cix = [], {}              # book chapters (numbering removed; one name in both books = one chapter)
    for i, r in enumerate(rows):
        if source_of(r) != "book":
            continue
        k = NUM.sub("", r["title_cs"] or "").strip()
        if k not in cix:
            cix[k] = len(chapters)
            chapters.append({"label": NUM.sub("", (r["title_en"] or k).split(" › ")[0]).strip(), "members": []})
        chapters[cix[k]]["members"].append(i)
    meta = {"counts": counts, "n_vid": n_vid, "chapters": chapters, "models": [[k, l] for k, l in MODELS]}
    html = Path(__file__).with_suffix(".html").read_text(encoding="utf-8")
    html = html.replace("__LABEL_JS__", LABEL_JS).replace("__META__", json.dumps(meta, ensure_ascii=False)).replace(
        "__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print("Wrote", OUT, f"{OUT.stat().st_size / 1e6:.1f} MB", json.dumps(counts))


if __name__ == "__main__":
    main()
