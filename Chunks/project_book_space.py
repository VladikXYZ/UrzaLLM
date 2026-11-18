"""One report: the book topics as a space, with articles, video descriptions and transcripts projected into it.

    python Chunks/project_book_space.py          # from the repo root; ~10 minutes on a CPU

Concepts: every chapter title and every subtitle of the two books (real_data/chunked.json), with the numbering
removed ("10. Zdravotnictví" -> "Zdravotnictví"); the same name in both books is one concept. A concept's vector is
the average of the book paragraphs under that heading (a title includes its subsections), so it stands for what
the book says there, and it exists for both models without embedding anything new.

The space: every chunk is described by its cosine similarity to each concept (one number per concept), z-scored per
concept on the book paragraphs and centred per chunk (so only which topics a chunk leans to counts, not how
"bookish" it is overall). The map is UMAP fitted on the book paragraphs alone; articles, descriptions and
transcripts are then placed into it with umap.transform, which does not change it. Each concept sits at the average
position of its paragraphs. "Source-centred": each source's average vector is subtracted before everything above.

Reads   real_data/chunked.json + embeddings_out_cs/, real_data/trans_chunks.json + embeddings_out_trans_cs/
Writes  RealReports/projection_book_space.html (template Chunks/project_book_space.html)
        Chunks/output_real/book_space_cache.npz (reused unless --recompute)
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Projection"))
from labels import LABEL_JS  # noqa: E402

RD = ROOT / "real_data"
OUT = ROOT / "RealReports" / "projection_book_space.html"
CACHE = ROOT / "Chunks" / "output_real" / "book_space_cache.npz"
MODELS = [("bge_m3", "bge-m3", "BAAI/bge-m3"), ("embeddinggemma_2", "EmbeddingGemma", "google/embeddinggemma-300m")]
SOURCES = ["article", "book", "video", "transcription"]          # colour order of the earlier reports
UMAP_PARAMS = dict(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
NUM = re.compile(r"^\s*\d+\.\s*")


def clean(s):
    return NUM.sub("", (s or "").split(" › ")[0]).strip()


def load():
    ch = json.loads((RD / "chunked.json").read_text(encoding="utf-8"))
    tr = json.loads((RD / "trans_chunks.json").read_text(encoding="utf-8"))
    for f, rows in (("embeddings_out_cs", ch), ("embeddings_out_trans_cs", tr)):
        meta = json.loads((RD / f / "metadata_index.json").read_text(encoding="utf-8"))
        assert len(meta) == len(rows) and all(m.strip() == r["text_cs"].strip() for m, r in zip(meta, rows)), f
    return ch + tr


def concepts(rows):
    """[{key, label, kind (title|subtitle), books, members}] in book order."""
    out, ix = [], {}
    for i, r in enumerate(rows):
        if r["type"] != "book":
            continue
        for kind, cs, en in (("title", r["title_cs"], r["title_en"]), ("subtitle", r["subtitle_cs"], r["subtitle_en"])):
            k = clean(cs)
            if not k:
                continue
            if k not in ix:
                ix[k] = len(out)
                out.append({"key": k, "label": clean(en) or k, "kind": kind, "books": [], "members": []})
            c = out[ix[k]]
            c["members"].append(i)
            if r["src"] not in c["books"]:
                c["books"].append(r["src"])
    return out


def vectors(key):
    X = np.concatenate([np.load(RD / "embeddings_out_cs" / f"{key}_vectors.npy"),
                        np.load(RD / "embeddings_out_trans_cs" / f"{key}_vectors.npy")]).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def procrustes(ref, P):
    a, b = ref - ref.mean(0), P - P.mean(0)
    U, s, Vt = np.linalg.svd(b.T @ a)
    return (b @ (U @ Vt)) * (s.sum() / (b ** 2).sum()), (ref.mean(0), P.mean(0), U @ Vt, s.sum() / (b ** 2).sum())


def compute(rows, con):
    import umap
    lab = np.array([SOURCES.index(r["type"]) for r in rows])
    book = np.flatnonzero(lab == 1)
    other = np.flatnonzero(lab != 1)
    res = {}
    for key, label, _ in MODELS:
        X = vectors(key)
        for v in ("raw", "cen"):
            t0 = time.time()
            Y = X
            if v == "cen":
                Y = X.copy()
                for s in range(len(SOURCES)):
                    Y[lab == s] -= Y[lab == s].mean(0)
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            C = np.stack([Y[c["members"]].mean(0) for c in con])
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            S = Y @ C.T                                            # cosine to every concept, N x K
            mu, sd = S[book].mean(0), S[book].std(0) + 1e-6
            P = (S - mu) / sd
            P -= P.mean(1, keepdims=True)
            m = umap.UMAP(**UMAP_PARAMS).fit(P[book])
            xy = np.zeros((len(rows), 2), np.float32)
            xy[book] = m.embedding_
            for a in range(0, len(other), 20000):
                xy[other[a:a + 20000]] = m.transform(P[other[a:a + 20000]])
            res[f"{key}_{v}_xy"] = xy
            res[f"{key}_{v}_near"] = S.argmax(1).astype(np.int16)    # the concept each chunk is most similar to
            print(f"{label} {v}: {time.time() - t0:.0f} s", flush=True)
    return lab, res


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    rows = load()
    con = concepts(rows)
    print(len(con), "concepts:", sum(c["kind"] == "title" for c in con), "titles,",
          sum(c["kind"] == "subtitle" for c in con), "subtitles", flush=True)
    if CACHE.exists() and not args.recompute:
        z = np.load(CACHE)
        lab, res = z["lab"], {k: z[k] for k in z.files if k != "lab"}
        assert len(lab) == len(rows), "cache is for other data; run with --recompute"
    else:
        lab, res = compute(rows, con)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE, lab=lab, **res)

    a, b = MODELS[0][0], MODELS[1][0]
    book = lab == 1
    for v in ("raw", "cen"):            # align EmbeddingGemma's map to bge-m3's on the book paragraphs, scale to [-1, 1]
        ra, rb = res[f"{a}_{v}_xy"], res[f"{b}_{v}_xy"]
        _, (ma, mb, R, sc) = procrustes(ra[book], rb[book])
        res[f"{b}_{v}_xy"] = (rb - mb) @ R * sc
        res[f"{a}_{v}_xy"] = ra - ma
        span = max(np.abs(res[f"{k}_{v}_xy"][book]).max() for k in (a, b))
        for k in (a, b):
            res[f"{k}_{v}_xy"] = res[f"{k}_{v}_xy"] / span

    K = len(con)
    n_src = {s: int((lab == i).sum()) for i, s in enumerate(SOURCES)}
    shares = {}                          # per model/view: share of each source's chunks most similar to each concept
    for k, _, _ in MODELS:
        for v in ("raw", "cen"):
            near = res[f"{k}_{v}_near"]
            shares[f"{k}_{v}"] = {s: (np.bincount(near[lab == i], minlength=K) / max(1, (lab == i).sum())).round(5).tolist()
                                  for i, s in enumerate(SOURCES)}
    titles, tix, doc = [], {}, []
    for r in rows:
        t = r["title_en"] or r["title_cs"] or ""
        if r["type"] == "book" and r.get("subtitle_en"):
            t += " · " + r["subtitle_en"]
        if t not in tix:
            tix[t] = len(titles); titles.append(t)
        doc.append(tix[t])
    data = {"lab": lab.tolist(), "doc": doc, "titles": titles,
            "snip": [((r["text_cs"] if r["type"] == "transcription" else r["text_en"]) or "")[:100] for r in rows],
            "tm": [int(float(r["time"])) if r["type"] == "transcription" else -1 for r in rows],
            "xy": {f"{k}_{v}": np.round(res[f"{k}_{v}_xy"] * 10000).astype(int).ravel().tolist()
                   for k, _, _ in MODELS for v in ("raw", "cen")},
            "near": {f"{k}_{v}": res[f"{k}_{v}_near"].tolist() for k, _, _ in MODELS for v in ("raw", "cen")}}
    meta = {"models": {k: {"label": l, "hf": h} for k, l, h in MODELS}, "counts": n_src,
            "n_vid": len({r["src"] for r in rows if r["type"] == "transcription"}),
            "concepts": [{"label": c["label"], "cs": c["key"], "kind": c["kind"], "books": c["books"], "members": c["members"],
                          "n": len(c["members"])} for c in con], "shares": shares}
    html = Path(__file__).with_suffix(".html").read_text(encoding="utf-8")
    html = (html.replace("__LABEL_JS__", LABEL_JS).replace("__META__", json.dumps(meta, ensure_ascii=False))
            .replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print("Wrote", OUT, f"{OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
