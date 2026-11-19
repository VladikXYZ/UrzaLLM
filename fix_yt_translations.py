"""Re-pair the YouTube translations in yt_full.json with the right videos (and fix chunked.json with them).

    python fix_yt_translations.py          # needs sentence-transformers + scipy and BAAI/bge-m3 (cached)

What went wrong: yt_scraper.py runs with 2 parallel workers and saves its checkpoint as list(data.values()),
i.e. in the order the downloads finished, so every run (a resume, a new video) rewrites
mat_czech/yt_metadata_nice.json in a different order. The translation input (yt_tr.json) was made from one
order; merging_mat.py later zipped the translations with a newer order of yt_metadata_nice.json by position
(the translations carry no video id). Result: most videos got another video's title_en / desc_en
(e.g. "Křest Domu Svobodného přístavu" had the title and description of the Grünwald interview).

The fix: the translations themselves are fine, only attached to the wrong videos. Every Czech title +
description and every English title + description is embedded with bge-m3 (multilingual: a text and its
translation land close together), and the best one-to-one pairing is found (Hungarian algorithm on the
mean of title and description similarity). On 18 Nov 2026: 974 of 980 pairs were each other's best match
from both sides; the other 6 were parts of one series (Streamaraton hours 1-12 / 13-17, Digitální
(R)evoluce #9/#11/#12) and checked by hand to be right.

Writes materials/yt_full.json and real_data/yt_full.json (same content) and updates title_en / text_en of
the YouTube-description chunks in real_data/chunked.json (matched by URL). The English embeddings of those
chunks (real_data/embeddings_out_en/) were made from the wrong texts and need re-running embedding.py.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
FILES = [ROOT / "materials" / "yt_full.json", ROOT / "real_data" / "yt_full.json"]
CHUNKED = ROOT / "real_data" / "chunked.json"


def main():
    from scipy.optimize import linear_sum_assignment
    from sentence_transformers import SentenceTransformer
    y = json.loads(FILES[0].read_text(encoding="utf-8"))
    m = SentenceTransformer("BAAI/bge-m3")

    def enc(xs):
        return m.encode([(x or "")[:1500] for x in xs], batch_size=32, normalize_embeddings=True, show_progress_bar=False)

    S = 0.5 * (enc([r["title_cs"] for r in y]) @ enc([r["title_en"] for r in y]).T) \
        + 0.5 * (enc([r["desc_cs"] for r in y]) @ enc([r["desc_en"] for r in y]).T)
    rows, cols = linear_sum_assignment(-S)
    mutual = sum(1 for i, j in zip(rows, cols) if S[i].argmax() == j and S[:, j].argmax() == i)
    print(f"{len(y)} videos: {mutual} pairs are each other's best match; "
          f"{sum(i != j for i, j in zip(rows, cols))} translations moved")
    en = [(r["title_en"], r["desc_en"]) for r in y]
    for i, j in zip(rows, cols):
        y[i]["title_en"], y[i]["desc_en"] = en[j]
    for f in FILES:
        f.write_text(json.dumps(y, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {f.relative_to(ROOT)}")

    if CHUNKED.exists():
        by_url = {r["url"]: r for r in y}
        ch = json.loads(CHUNKED.read_text(encoding="utf-8"))
        n = 0
        for c in ch:
            r = by_url.get(c.get("src"))
            if r and "youtube.com" in c["src"]:
                c["title_en"], c["text_en"] = r["title_en"], r["desc_en"]
                n += 1
        CHUNKED.write_text(json.dumps(ch, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"updated {n} YouTube-description chunks in {CHUNKED.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
