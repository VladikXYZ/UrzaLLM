"""One report: all four sources (articles, books, video descriptions, video transcripts) on one map, raw vs
source-centred, for both embedding models of real_data/ (bge-m3, EmbeddingGemma).

    python Chunks/project_all_sources.py        # from the repo root; ~10-20 minutes on a CPU (4 UMAPs of ~94k points)

Reads   real_data/chunked.json + embeddings_out_cs/, real_data/trans_chunks.json + embeddings_out_trans_cs/
Writes  RealReports/projection_all_sources.html (self-contained; template Chunks/project_all_sources.html)
        Chunks/output_real/all_sources_cache.npz (maps and neighbour stats, reused unless --recompute)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import re

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Projection"))
from labels import LABEL_JS  # noqa: E402  (shared label placer: every label shown, leader lines when moved)

NUM = re.compile(r"^\s*\d+\.\s*")

ROOT = Path(__file__).resolve().parent.parent
RD = ROOT / "real_data"
OUT = ROOT / "RealReports" / "projection_all_sources.html"
CACHE = ROOT / "Chunks" / "output_real" / "all_sources_cache.npz"
MODELS = [("bge_m3", "bge-m3", "BAAI/bge-m3"), ("embeddinggemma_2", "EmbeddingGemma", "google/embeddinggemma-300m")]
SOURCES = ["article", "book", "video", "transcription"]          # colour order of the earlier reports + transcripts
UMAP_PARAMS = dict(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42, low_memory=True)


def load():
    ch = json.loads((RD / "chunked.json").read_text(encoding="utf-8"))
    tr = json.loads((RD / "trans_chunks.json").read_text(encoding="utf-8"))
    for f, rows in (("embeddings_out_cs", ch), ("embeddings_out_trans_cs", tr)):
        meta = json.loads((RD / f / "metadata_index.json").read_text(encoding="utf-8"))
        assert len(meta) == len(rows) and all(m.strip() == r["text_cs"].strip() for m, r in zip(meta, rows)), f
    return ch + tr


def vectors(key):
    X = np.concatenate([np.load(RD / "embeddings_out_cs" / f"{key}_vectors.npy"),
                        np.load(RD / "embeddings_out_trans_cs" / f"{key}_vectors.npy")]).astype(np.float32)
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


def compute(rows):
    import umap
    lab = np.array([SOURCES.index(r["type"]) for r in rows])
    res, stats = {}, {}
    for key, label, _ in MODELS:
        X = vectors(key)
        assert len(X) == len(rows)
        C = centre(X, lab)
        stats[key] = {}
        for v, Y in (("raw", X), ("cen", C)):
            t0 = time.time()
            res[f"{key}_{v}"] = umap.UMAP(**UMAP_PARAMS).fit_transform(Y).astype(np.float32)
            print(f"{label} {v}: UMAP {time.time() - t0:.0f} s", flush=True)
    return lab, res, stats


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    rows = load()
    if CACHE.exists() and not args.recompute:
        z = np.load(CACHE, allow_pickle=False)
        lab, stats = z["lab"], json.loads(str(z["stats"]))
        res = {k: z[k] for k in z.files if k not in ("lab", "stats")}
        assert len(lab) == len(rows), "cache is for other data; run with --recompute"
    else:
        lab, res, stats = compute(rows)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE, lab=lab, stats=json.dumps(stats), **res)

    a, b = MODELS[0][0], MODELS[1][0]
    for v in ("raw", "cen"):            # turn EmbeddingGemma's maps to match bge-m3's; scale both to [-1, 1]
        res[f"{b}_{v}"] = procrustes(res[f"{a}_{v}"], res[f"{b}_{v}"])
        res[f"{a}_{v}"] = res[f"{a}_{v}"] - res[f"{a}_{v}"].mean(0)
        for k in (a, b):
            res[f"{k}_{v}"] = res[f"{k}_{v}"] / np.abs(res[f"{k}_{v}"]).max()

    # hover text: titles once per document, a short English (transcripts: Czech) snippet per chunk
    titles, tix = [], {}
    doc = []
    for r in rows:
        t = r["title_en"] or r["title_cs"] or ""
        if r["type"] == "book" and r.get("subtitle_en"):
            t += " · " + r["subtitle_en"]
        if t not in tix:
            tix[t] = len(titles); titles.append(t)
        doc.append(tix[t])
    snip = [((r["text_cs"] if r["type"] == "transcription" else r["text_en"]) or "")[:110] for r in rows]
    tm = [int(float(r["time"])) if r["type"] == "transcription" else -1 for r in rows]
    counts = {s: int((lab == i).sum()) for i, s in enumerate(SOURCES)}
    n_vid = len({r["src"] for r in rows if r["type"] == "transcription"})
    xy = {f"{k}_{v}": np.round(res[f"{k}_{v}"] * 10000).astype(int).ravel().tolist()
          for k, _, _ in MODELS for v in ("raw", "cen")}
    data = {"lab": lab.tolist(), "doc": doc, "titles": titles, "snip": snip, "tm": tm, "xy": xy}
    # book chapters (headings without numbering; the same name in both books is one chapter), for the map labels
    chapters, cix = [], {}
    for i, r in enumerate(rows):
        if r["type"] != "book":
            continue
        k = NUM.sub("", r["title_cs"] or "").strip()
        if k not in cix:
            cix[k] = len(chapters)
            chapters.append({"label": NUM.sub("", (r["title_en"] or k).split(" › ")[0]).strip(), "members": []})
        chapters[cix[k]]["members"].append(i)
    meta = {"models": {k: {"label": l, "hf": h} for k, l, h in MODELS}, "stats": stats, "counts": counts,
            "rnd": {s: c / len(rows) for s, c in counts.items()}, "n_vid": n_vid, "chapters": chapters}
    html = (Path(__file__).with_suffix(".html")).read_text(encoding="utf-8")
    html = html.replace("__LABEL_JS__", LABEL_JS)
    html = html.replace("__CONCL__", "[]").replace("__META__", json.dumps(meta, ensure_ascii=False)).replace(
        "__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print("Wrote", OUT, f"{OUT.stat().st_size / 1e6:.1f} MB")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
