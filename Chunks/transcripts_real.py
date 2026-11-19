"""Urza in the political space with his spoken words (the video transcript chunks) added, per embedding model.

    python Chunks/transcripts_real.py [--models bge_m3 embeddinggemma_2]   # from the repo root, after Chunks/real_data.py

Reads   Chunks/output_real/chunks.parquet + <model>/results_textonly/embeddings.npy    the 2,911 written chunks
        embeddings_out_new_cs/trans_chunks.json + <model>_vectors.npy                  91,499 transcript chunks
        Questions/political_space_en.tsv + Questions/output[_<model>]/embeddings.npy   the 326 political items
Writes  Chunks/output_real/<model>/transcripts_results.json   numbers and rows for Chunks/presentation_real.py:
          depth of each political item (similarity of Urza's best passage) from the written texts alone and with
          his speech, per policy area; best-covered questions and blind spots; a control with the speech sampled
          down to the size of the written texts

Thresholds: bge-m3 uses those of Projection/political_space.py (0.60 on topic, 0.55 relevant). Another model has its
own similarity scale, so its thresholds are the values that just as large a share of all question-passage pairs
reaches as with bge-m3 (quantile matching).
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault("URZA_DATASET", "real")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from dataset import CHUNKS, MODELS, read_chunks  # noqa: E402

SOURCES = ["article", "book", "video", "transcript"]
NEW = ROOT / "embeddings_out_new_cs"
TRANSCRIPTS = NEW / "trans_chunks.json"
STRONG, SOME = 0.60, 0.55          # bge-m3 depth tiers of Projection/political_space.py


def unit(A):
    return A / np.linalg.norm(A, axis=-1, keepdims=True)


def timed_url(c):
    t = int(float(c.get("time") or 0))
    return c["src"] + ("&" if "?" in c["src"] else "?") + f"t={t}s"


def load_rows():
    rows = read_chunks(CHUNKS, ["chunk_id", "doc_id", "source", "kind", "year", "title", "url", "book", "chapter",
                                "subchapter", "text"])
    tr = json.loads(TRANSCRIPTS.read_text(encoding="utf-8"))
    return rows, tr


def items():
    it = list(csv.DictReader(open(ROOT / "Questions" / "political_space_en.tsv", encoding="utf-8"), delimiter="\t"))
    return it


def sims(model, rows, tr, it):
    """Similarity of every political item to every kept chunk (announcements left out), and the chunk indices."""
    res = HERE / "output_real" / model / "results_textonly"
    E = np.load(res / "embeddings.npy").astype(np.float32)
    assert (res / "chunk_ids.txt").read_text(encoding="utf-8").split() == [r["chunk_id"] for r in rows], \
        "run Chunks/real_data.py first"
    T = np.load(NEW / f"{model}_vectors.npy").astype(np.float32)
    assert len(T) == len(tr), f"{len(T)} vectors for {len(tr)} transcript chunks"
    qdir = ROOT / "Questions" / ("output" if model == "bge_m3" else f"output_{model}")
    assert (qdir / "ids.txt").read_text(encoding="utf-8").split() == [q["id"] for q in it]
    QE = unit(np.load(qdir / "embeddings.npy").astype(np.float32))
    X = unit(np.concatenate([E, T]))
    kind = np.array([r["kind"] for r in rows] + ["transcript"] * len(tr))
    A = np.where(kind != "announcement")[0]
    return QE @ X[A].T, A


def politics(model, rows, tr, it, bge_S=None):
    lab = np.array([SOURCES.index(r["source"]) for r in rows] + [3] * len(tr))
    S, A = sims(model, rows, tr, it)
    layer = np.array([q["layer"] for q in it])
    major = np.array([q["major"] for q in it])
    q_idx = np.where(layer == "question")[0]
    if bge_S is None:
        strong, some = STRONG, SOME
    else:                                              # same share of question-passage pairs as bge-m3
        ref, own = bge_S[q_idx].ravel(), S[q_idx].ravel()
        strong, some = (float(np.quantile(own, np.mean(ref < t))) for t in (STRONG, SOME))
    w_in_a = lab[A] != 3
    best_w = S[:, w_in_a].max(1)
    best_a, best_j = S.max(1), A[S.argmax(1)]

    def passage(j):
        if j < len(rows):
            r = rows[j]
            return {"src": r["source"], "title": r["title"], "url": r.get("url") or "", "text": (r["text"] or "")[:200]}
        c = tr[j - len(rows)]
        return {"src": "transcript", "title": c["title_en"] or c["title_cs"], "url": timed_url(c),
                "text": (c["text_en"] or c["text_cs"])[:200]}

    def qrow(i):
        return {"q": it[i]["text"], "area": it[i]["major"], "w": round(float(best_w[i]), 3),
                "a": round(float(best_a[i]), 3), "p": passage(int(best_j[i]))}
    areas = []
    for m in dict.fromkeys(q["major"] for q in it if q["layer"] != "concept"):
        its = np.where((major == m) & (layer != "concept"))[0]
        areas.append({"name": m, "w": float(best_w[its].mean()), "a": float(best_a[its].mean())})
    by_a = sorted(q_idx, key=lambda i: -best_a[i])
    won = Counter(SOURCES[lab[j]] for j in best_j[q_idx])
    # control for size: the speech alone, sampled down to as many chunks as the written texts (20 draws)
    n_w = int(w_in_a.sum())
    tcols = np.where(~w_in_a)[0]
    rng = np.random.default_rng(0)
    ctrl = [float(np.mean(S[np.ix_(q_idx, rng.choice(tcols, n_w, replace=False))].max(1) >= strong)) for _ in range(20)]
    out = {"n_q": int(len(q_idx)), "n_items": len(it), "n_written": n_w, "strong": strong, "some": some,
           "q_strong_speech_same_size": [float(np.mean(ctrl)), float(np.min(ctrl)), float(np.max(ctrl))],
           "q_strong": [float(np.mean(best_w[q_idx] >= strong)), float(np.mean(best_a[q_idx] >= strong))],
           "q_some": [float(np.mean(best_w[q_idx] >= some)), float(np.mean(best_a[q_idx] >= some))],
           "best_from": dict(won), "areas": areas,
           "top": [qrow(i) for i in by_a[:5]], "bottom": [qrow(i) for i in by_a[::-1][:5]]}
    return out, S


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    args = ap.parse_args()
    rows, tr, it = *load_rows(), items()
    print(f"{len(rows):,} written + {len(tr):,} transcript chunks", flush=True)
    bge_S = None
    if any(m != "bge_m3" for m in args.models):
        _, bge_S = politics("bge_m3", rows, tr, it)
    for model in args.models:
        pol, _ = politics(model, rows, tr, it, None if model == "bge_m3" else bge_S)
        print(f"{model}: thresholds {pol['strong']:.3f}/{pol['some']:.3f}; questions on topic {pol['q_strong']}, "
              f"speech at written size {pol['q_strong_speech_same_size'][0]:.2f}; best from {pol['best_from']}", flush=True)
        out = {"counts": {s: (sum(r["source"] == s for r in rows) if s != "transcript" else len(tr)) for s in SOURCES},
               "n_videos": len({c["src"] for c in tr}), "politics": pol}
        path = HERE / "output_real" / model / "transcripts_results.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
