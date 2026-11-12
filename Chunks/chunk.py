"""Chunk the preprocessed Czech and English data into aligned chunk files.

    python Chunks/chunk.py            # from the repo root

Rules
  article  the article text is one chunk
  video    the video description is one chunk, without the recurring boilerplate (donation details, crypto
           addresses, signature, separators: any line found in 20 or more descriptions), removed per language
  book     every paragraph (text split at blank lines) is one chunk

Reads (repo root)
  preprocessed/arts.json, yt.json                  Czech originals with metadata (date, url, linked video)
  preprocessed/arts_tr.json, yt_tr.json            Czech [title, text] pairs, same order as the originals
  preprocessed/arts_en.json, yt_en.json            English translations of those pairs, same order
  preprocessed/ancap_tr.json, anag_tr.json         Czech book sections (chapter, name, text)
  preprocessed/ancap_en.json, anag_en.json         English book sections, same order and paragraphs
  jsons/channel_metadata.json                      video upload dates (joined by URL)

Writes Chunks/output/chunks_cs.{parquet,jsonl} and chunks_en.{parquet,jsonl}
  Row i of the Czech file and row i of the English file are the same chunk with the same chunk_id;
  the script stops if the two languages don't line up.
Also writes Chunks/output/chunks.parquet: the Czech chunks in the same format as Preprocessing/output/chunks.parquet
  (kind, series, tags, guests, views and duration taken from the earlier pipeline's documents, matched by URL;
  book chapters stored the same way), with the three embedding texts used throughout the project:
    embed_text            context line naming the source + text ("Kniha … › chapter", "Článek „…“ (year)", "Video „…“")
    embed_text_nosource   the same context line without the word naming the source
    embed_text_textonly   the chunk text only
  This is what Embedding/ and Projection/ read when run with URZA_DATASET=final.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PRE = ROOT / "preprocessed"
OUT = HERE / "output"
BOOKS = [("AC", "Anarchokapitalismus", "ancap"), ("AG", "Anarchoagorismus", "anag")]
BOILERPLATE_MIN = 20   # a line found in this many video descriptions or more is boilerplate, not content
DODATEK = ("AMEN: analýza mnohých etatistických námitek", "Další oblasti státního vlivu")


def book_place(chapter, name):
    """(kind, chapter, subchapter) in the earlier pipeline's convention, so chapter groups stay the same."""
    if chapter.startswith(DODATEK):
        kind = "book_faq" if chapter.startswith("AMEN") else "book_section"
        return kind, "dodatek", f"{chapter} › {name}"
    if chapter == "Tři příběhy":
        return "book_story", chapter, name
    if chapter.startswith("AMEN"):
        return "book_faq", chapter, name
    if chapter:
        return "book_section", chapter, name
    return "book_section", name, ""


def pipeline_rows(cs_rows):
    """Czech chunks in the format of Preprocessing/output/chunks.parquet, with the three embedding texts."""
    sys.path.insert(0, str(ROOT / "Preprocessing"))
    from chunking import header  # noqa: E402
    docs = {d["url"]: d for d in pq.read_table(ROOT / "Preprocessing" / "output" / "documents.parquet").to_pylist()
            if d.get("url")}
    vmeta = {}
    for r in pq.read_table(ROOT / "Preprocessing" / "output" / "chunks.parquet",
                           columns=["source", "url", "views", "duration_min"]).to_pylist():
        if r["source"] == "video":
            vmeta.setdefault(r["url"], r)
    out = []
    per_doc = {}
    for r in cs_rows:
        per_doc[r["doc_id"]] = per_doc.get(r["doc_id"], 0) + 1
    for r in cs_rows:
        d = docs.get(r["url"], {})
        if r["source"] == "book":
            kind, chapter, sub = book_place(r["chapter"], r["section"])
        else:
            kind = d.get("kind") or r["source"]
            if kind == "video_post":          # articles the earlier pipeline set aside for their video; here they count as text
                kind = "article"
            chapter, sub = None, None
        doc = {"source": r["source"], "kind": kind, "title": r["title"], "year": r["year"] or None,
               "book": r["book"] or None, "chapter": chapter, "subchapter": sub or None,
               "guests": list(d.get("guests") or [])}
        vm = vmeta.get(r["url"], {})
        out.append({
            "chunk_id": r["chunk_id"], "doc_id": r["doc_id"], "source": r["source"], "kind": kind,
            "position": r["para"], "n_chunks": per_doc[r["doc_id"]], "title": r["title"],
            "date": r["date"] or None, "year": r["year"] or None, "url": r["url"] or None,
            "series": d.get("series"), "book": r["book"] or None, "chapter": chapter, "subchapter": sub or None,
            "tags": list(d.get("tags") or []), "guests": doc["guests"], "flags": [],
            "views": vm.get("views"), "duration_min": vm.get("duration_min"),
            "words": r["n_words"], "text": r["text"],
            "embed_text": f"{header(doc)}\n\n{r['text']}",
            "embed_text_nosource": f"{header(doc, with_source=False)}\n\n{r['text']}",
            "embed_text_textonly": r["text"],
        })
    return out


def load(path):
    return json.load(open(path, encoding="utf-8"))


def words(text):
    return len(text.split())


def chunk_rows():
    """Yield (meta, czech_text, english_text) per chunk."""
    upload = {v["webpage_url"]: v.get("upload_date") or "" for v in load(ROOT / "jsons" / "channel_metadata.json")}

    arts, arts_cs, arts_en = load(PRE / "arts.json"), load(PRE / "arts_tr.json"), load(PRE / "arts_en.json")
    assert len(arts) == len(arts_cs) == len(arts_en), "articles: Czech and English files differ in length"
    for i, (a, cs, en) in enumerate(zip(arts, arts_cs, arts_en)):
        assert a["text"] == cs[1], f"article {i}: arts.json and arts_tr.json disagree"
        date = a["date"][:10]
        yield ({"chunk_id": f"art-{i:04d}", "source": "article", "doc_id": f"art-{i:04d}", "book": "", "chapter": "",
                "section": "", "para": 0, "date": date, "year": int(date[:4]) if date else 0, "url": a["url"],
                "video_url": a.get("video") or ""}, (cs[0], cs[1]), (en[0], en[1]))

    yt, yt_cs, yt_en = load(PRE / "yt.json"), load(PRE / "yt_tr.json"), load(PRE / "yt_en.json")
    assert len(yt) == len(yt_cs) == len(yt_en), "videos: Czech and English files differ in length"
    for i, (v, cs, en) in enumerate(zip(yt, yt_cs, yt_en)):
        assert v["desc"] == cs[1], f"video {i}: yt.json and yt_tr.json disagree"
        d = upload.get(v["url"], "")
        date = f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 else ""
        yield ({"chunk_id": f"vid-{i:04d}", "source": "video", "doc_id": f"vid-{i:04d}", "book": "", "chapter": "",
                "section": "", "para": 0, "date": date, "year": int(date[:4]) if date else 0, "url": v["url"],
                "video_url": v["url"]}, (cs[0], cs[1]), (en[0], en[1]))

    for code, book, stem in BOOKS:
        secs_cs, secs_en = load(PRE / f"{stem}_tr.json"), load(PRE / f"{stem}_en.json")
        assert len(secs_cs) == len(secs_en), f"{book}: Czech and English differ in number of sections"
        for s, (cs, en) in enumerate(zip(secs_cs, secs_en)):
            pc, pe = cs["text"].split("\n\n"), en["text"].split("\n\n")
            assert len(pc) == len(pe), f"{book} section {s}: {len(pc)} Czech vs {len(pe)} English paragraphs"
            for p, (tc, te) in enumerate(zip(pc, pe)):
                cid = f"{code}-{s:03d}-{p:03d}"
                yield ({"chunk_id": cid, "source": "book", "doc_id": f"{code}-{s:03d}", "book": book,
                        "chapter": cs["chapter"], "section": cs["name"], "para": p, "date": "", "year": 0,
                        "url": "", "video_url": ""},
                       (cs["name"], tc), (en["name"], te))


BOILERPLATE_RE = re.compile("|".join([
    r"^(BTC|LTC|ETH|XMR|BCH|DOGE|LN)\s*:",                                 # crypto addresses, Lightning
    r"\b(bc1q|ltc1q)[0-9a-z]{20,}",
    r"^(IBAN|Číslo účtu|Account number|Bank account)\b", r"variabilní symbol|variable symbol",
    r"^Pravidelná podpora|^Regular (support|donations?)",
    r"^(Pokud se vám|Jestli se vám|Chcete-li podpořit|Chcete-li mě podpořit)",       # donation pleas
    r"^(If you (liked|enjoyed|like|want to support|would like to support))",
    r"^Děkujeme všem diskutujícím|^We thank all the (discussants|participants)",
    r"^[-–—=_*]{3,}$",                                                     # separators
    r"^[–-]\s*Urza \(www\.urza\.cz\)",                                     # Urza's signature line
]), re.IGNORECASE)


def strip_video_boilerplate(rows):
    """Remove boilerplate lines from video descriptions, per language, in place: lines that recur in at least
    BOILERPLATE_MIN descriptions, and lines matching BOILERPLATE_RE (donation pleas, crypto addresses, bank
    details, separators, Urza's signature), which change over the years and so recur less. Guest bios stay:
    they describe who is in the video. Returns the set of removed lines."""
    lines = Counter(ln.strip() for r in rows if r["source"] == "video"
                    for ln in set(r["text"].split("\n")) if ln.strip())
    common = {ln for ln, c in lines.items() if c >= BOILERPLATE_MIN or BOILERPLATE_RE.search(ln)}
    for r in rows:
        if r["source"] != "video":
            continue
        kept = [ln for ln in r["text"].split("\n") if ln.strip() not in common]
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
        assert text, f"{r['chunk_id']}: nothing left after removing boilerplate"
        r["text"], r["n_words"] = text, words(text)
    return common


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cs_rows, en_rows = [], []
    for meta, (t_cs, x_cs), (t_en, x_en) in chunk_rows():
        cs_rows.append({**meta, "lang": "cs", "title": t_cs, "text": x_cs, "n_words": words(x_cs)})
        en_rows.append({**meta, "lang": "en", "title": t_en, "text": x_en, "n_words": words(x_en)})
    for lang, rows in (("cs", cs_rows), ("en", en_rows)):
        removed = strip_video_boilerplate(rows)
        print(f"{lang}: removed {len(removed)} recurring lines from video descriptions")
    ids = [r["chunk_id"] for r in cs_rows]
    assert len(set(ids)) == len(ids), "duplicate chunk ids"
    empty = [r["chunk_id"] for r in cs_rows if not r["text"].strip()]
    assert not empty, f"empty Czech chunks: {empty[:5]}"

    OUT.mkdir(parents=True, exist_ok=True)
    for lang, rows in (("cs", cs_rows), ("en", en_rows)):
        pq.write_table(pa.Table.from_pylist(rows), OUT / f"chunks_{lang}.parquet")
        with open(OUT / f"chunks_{lang}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pq.write_table(pa.Table.from_pylist(pipeline_rows(cs_rows)), OUT / "chunks.parquet")

    for src in ("article", "video", "book"):
        w = sorted(r["n_words"] for r in cs_rows if r["source"] == src)
        print(f"{src:8s} {len(w):5d} chunks   words: median {w[len(w) // 2]}, min {w[0]}, max {w[-1]}")
    print(f"total {len(cs_rows)} chunks per language → {OUT}")


if __name__ == "__main__":
    main()
