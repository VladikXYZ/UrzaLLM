"""Split documents into embedding-sized chunks with a context header.

Chunk sizes are set per source in config.CHUNKING.
"""
import config as C
from textutils import paragraphs, sentences, words

KIND_LABEL = {
    "article": "Článek",
    "announcement": "Oznámení",
    "video_post": "Video",
    "video": "Video",
}


def _units(text: str, p: dict) -> list[str]:
    """Paragraphs, with over-long paragraphs broken into sentence groups."""
    out = []
    for para in paragraphs(text):
        if words(para) <= p["max"]:
            out.append(para)
            continue
        group = []
        for s in sentences(para):
            if group and words(" ".join(group + [s])) > p["target"]:
                out.append(" ".join(group))
                group = []
            group.append(s)
        if group:
            out.append(" ".join(group))
    return out


def split_text(text: str, p: dict) -> list[str]:
    """Split text into chunks using settings p (target, max, min_tail, overlap)."""
    if words(text) <= p["max"]:
        return [text]
    chunks: list[list[str]] = []
    cur: list[str] = []
    cur_w = 0
    overlap_len = 0          # words at the start of `cur` that repeat the previous chunk
    for u in _units(text, p):
        uw = words(u)
        if cur and cur_w + uw > p["target"]:
            if cur_w + uw <= p["max"]:
                cur.append(u)
                chunks.append(cur)
                cur, cur_w, overlap_len = [], 0, 0
                continue
            chunks.append(cur)
            # Overlap: carry the previous chunk's last sentence if it is short.
            last = sentences(cur[-1])[-1] if cur else ""
            cur = [last] if last and words(last) <= p["overlap"] else []
            cur_w = overlap_len = words(" ".join(cur))
        cur.append(u)
        cur_w += uw
    if cur:
        tail = cur[1:] if overlap_len else cur
        tail_w = cur_w - overlap_len
        prev_w = words(" ".join(chunks[-1])) if chunks else 0
        if chunks and tail_w < p["min_tail"] and prev_w + tail_w <= 1.2 * p["max"]:
            chunks[-1].extend(tail)
        else:
            chunks.append(cur)
    return ["\n\n".join(c) for c in chunks]


def header(doc: dict, with_source: bool = True) -> str:
    """Context line put in front of each chunk before embedding.

    with_source=True:  "Kniha Anarchokapitalismus › 10. Zdravotnictví", "Článek „…“ (2017)", "Video „…“ (…)"
    with_source=False: the same without the word naming the source
                       ("Anarchokapitalismus › 10. Zdravotnictví", "„…“ (2017)").
    """
    if doc["source"] == "book":
        parts = [f"Kniha {doc['book']}" if with_source else doc["book"], doc.get("chapter")]
        if doc.get("subchapter"):
            parts.append(doc["subchapter"])
        return " › ".join(p for p in parts if p)
    label = KIND_LABEL.get(doc["kind"], doc["kind"])
    extra = [str(doc["year"])] if doc.get("year") else []
    if doc.get("guests"):
        extra.append("hosté: " + ", ".join(doc["guests"]))
    line = f"„{doc['title']}“" + (f" ({'; '.join(extra)})" if extra else "")
    return f"{label} {line}" if with_source else line


def chunk_documents(docs: list[dict]) -> list[dict]:
    chunks = []
    for d in docs:
        if d["status"] != "included":
            continue
        texts = split_text(d["text"], C.CHUNKING[d["source"]])
        h = header(d)
        h_ns = header(d, with_source=False)
        for i, t in enumerate(texts):
            chunks.append({
                "chunk_id": f"{d['doc_id']}#{i:02d}",
                "doc_id": d["doc_id"],
                "source": d["source"],
                "kind": d["kind"],
                "position": i,
                "n_chunks": len(texts),
                "title": d["title"],
                "date": d.get("date"),
                "year": d.get("year"),
                "url": d.get("url"),
                "series": d.get("series"),
                "book": d.get("book"),
                "chapter": d.get("chapter"),
                "subchapter": d.get("subchapter"),
                "tags": d.get("tags", []),
                "guests": d.get("guests", []),
                "flags": d.get("flags", []),
                "views": d.get("views"),
                "duration_min": d.get("duration_min"),
                "words": words(t),
                "text": t,
                "embed_text": f"{h}\n\n{t}",
                # Same text without the word naming the source (Kniha/Článek/Oznámení/Video),
                # to test how much that word makes chunks cluster by source.
                "embed_text_nosource": f"{h_ns}\n\n{t}",
            })
    return chunks
