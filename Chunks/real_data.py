"""Turn real_data/ (the friend's cleaned chunks and their embeddings) into the inputs the report scripts read.

    python Chunks/real_data.py            # from the repo root

Reads   real_data/chunked.json (2,911 chunks: articles, video descriptions, book paragraphs; CS + EN)
        real_data/embeddings_out_cs/<model>_vectors.npy + metadata_index.json (embedded Czech text, row order = chunked.json)
        real_data/yt_full.json (metadata of videos the earlier corpus did not have)
        Chunks/output/chunks.parquet (the earlier final chunks, for metadata: ids, dates, tags, book chapters)
Writes  Chunks/output_real/chunks.parquet        (the pipeline schema, in real_data order)
        Chunks/output_real/<model>/results_textonly/{embeddings.npy, chunk_ids.txt, manifest.json}

The real_data embeddings are of the chunk text alone, so they are the "textonly" variant. Articles and book paragraphs
have the same Czech text as the earlier final chunks and take their metadata by text; videos take it by URL (their
descriptions were cleaned again), and videos the earlier corpus lacked get it from yt_full.json. English text and
titles come from real_data (retranslated); book chapter names keep the earlier English (names_en.json).
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
RD = ROOT / "real_data"
OUT = ROOT / "Chunks" / "output_real"
MODELS = {"bge_m3": "BAAI/bge-m3", "embeddinggemma_2": "google/embeddinggemma-300m"}


def main():
    ch = json.loads((RD / "chunked.json").read_text(encoding="utf-8"))
    old_t = pq.read_table(ROOT / "Chunks" / "output" / "chunks.parquet")
    schema, old = old_t.schema, old_t.to_pylist()
    by_text, by_url = defaultdict(list), {}
    for o in old:
        if o["source"] == "video":
            by_url[o["url"]] = o
        else:
            by_text[(o["source"], o["text"].strip())].append(o)
    yt = {v["url"]: v for v in json.loads((RD / "yt_full.json").read_text(encoding="utf-8"))}
    n_vid = max(int(o["chunk_id"].split("-")[1]) for o in old if o["source"] == "video")

    rows, how = [], Counter()
    for c in ch:
        src, text = c["type"], c["text_cs"].strip()
        if src == "video" and c["src"] in by_url:
            r = dict(by_url[c["src"]]); how["video by url"] += 1
        elif src == "video":
            v = yt[c["src"]]; n_vid += 1; how["video new (yt_full)"] += 1
            d = v["date"]
            r = {k: None for k in schema.names}
            r.update(chunk_id=f"vid-{n_vid:04d}", doc_id=f"vid-{n_vid:04d}", source="video", kind="video", position=0,
                     n_chunks=1, title=c["title_cs"], date=f"{d[:4]}-{d[4:6]}-{d[6:8]}", year=int(d[:4]), url=v["url"],
                     tags=v.get("tags") or [], guests=[], flags=[],
                     duration_min=round(v["duration"] / 60, 1) if v.get("duration") else None)
        else:
            q = by_text.get((src, text))
            if not q:
                sys.exit(f"no earlier chunk with this {src} text: {text[:80]}")
            r = dict(q.pop(0)); how[f"{src} by text"] += 1
        r["text"] = r["embed_text"] = r["embed_text_nosource"] = r["embed_text_textonly"] = c["text_cs"]
        r["words"] = len(c["text_cs"].split())
        r["text_en"] = c["text_en"]
        if src != "book":
            r["title"], r["title_en"] = c["title_cs"], c["title_en"]
        rows.append(r)
    ids = [r["chunk_id"] for r in rows]
    assert len(set(ids)) == len(ids), "duplicate chunk ids"
    print("chunks:", len(rows), dict(how), "| earlier chunks not used:", len(old) - sum(how.values()) + how["video new (yt_full)"])

    OUT.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), OUT / "chunks.parquet")
    meta = json.loads((RD / "embeddings_out_cs" / "metadata_index.json").read_text(encoding="utf-8"))
    assert len(meta) == len(ch) and all(m.strip() == c["text_cs"].strip() for m, c in zip(meta, ch)), \
        "metadata_index.json is not chunked.json's text in order"
    h = hashlib.sha256()
    for i, r in zip(ids, rows):
        h.update(i.encode("utf-8") + b"\t" + r["embed_text_textonly"].encode("utf-8") + b"\n")
    from transformers import AutoTokenizer      # token lengths, with bge-m3's tokenizer (EmbeddingGemma's is gated)
    tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    tl = [len(x) for x in tok([r["embed_text_textonly"] for r in rows], add_special_tokens=True)["input_ids"]]
    MAXLEN = {"bge_m3": 8192, "embeddinggemma_2": 2048}
    for model, hf in MODELS.items():
        f = RD / "embeddings_out_cs" / f"{model}_vectors.npy"
        if not f.exists():
            print("missing", f); continue
        X = np.load(f).astype(np.float32)
        assert X.shape[0] == len(rows) and not np.isnan(X).any()
        X /= np.linalg.norm(X, axis=1, keepdims=True)      # stored norms are 0.996-1.004; make them exactly 1
        d = OUT / model / "results_textonly"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "embeddings.npy", X)
        (d / "chunk_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
        (d / "manifest.json").write_text(json.dumps({
            "variant": "textonly", "text_column": "embed_text_textonly", "text_fingerprint": h.hexdigest(),
            "model": hf, "dim": int(X.shape[1]), "count": len(ids), "normalized": True,
            "max_length_tokens": MAXLEN[model], "truncated_chunks": sum(t > MAXLEN[model] for t in tl),
            "token_length_max": max(tl), "token_length_median": int(np.median(tl)),
            "token_lengths_from": "BAAI/bge-m3 tokenizer" + ("" if model == "bge_m3" else " (an estimate for this model)"),
            "dry_run": False, "device": "GPU (friend's machine)", "platform": "real_data/embeddings_out_cs", "seconds": None,
            "source": f"real_data/embeddings_out_cs/{model}_vectors.npy (embedded on the friend's GPU)",
            "created": datetime.now().isoformat(timespec="seconds")}, indent=2), encoding="utf-8")
        print(f"{model}: {X.shape} -> {d}")


if __name__ == "__main__":
    main()
