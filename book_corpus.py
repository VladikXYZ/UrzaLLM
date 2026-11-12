"""Topic-based embedding corpus over the parsed books, with semantic search.

Two levels, both embedded with BAAI/bge-m3 (multilingual, strong on Czech, 1024-dim,
up to 8192 tokens; the same model as the rest of the project):

* section  - one vector per topic section from epub_parser.py (the whole section,
             prefixed with its heading path). Answers "which topics cover X?".
* chunk    - sections split on paragraph boundaries into ~1200-character passages,
             each prefixed with its section's heading path. Answers "where exactly
             is X discussed?". Every chunk keeps a pointer to its section.

Usage:
    python epub_parser.py                              # data/*.epub -> jsons/*.json
    python book_corpus.py build                        # jsons/{anag,ankap}.json -> corpus/
    python book_corpus.py search "kdo bude stavět silnice"
    python book_corpus.py search "monopoly a kartely" --level section -k 5
    python book_corpus.py search "školství" --book ankap

From Python:
    from book_corpus import BookCorpus
    corpus = BookCorpus.load()
    for hit in corpus.search("daně jsou krádež", k=5):
        print(hit["score"], hit["book_title"], hit["name"])
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

MODEL_NAME = "BAAI/bge-m3"
BOOKS = ["anag", "ankap"]
JSON_DIR = Path("jsons")
EPUB_DIR = Path("data")
OUT_DIR = Path("corpus")

CHUNK_TARGET = 1200  # characters; ~300 bge-m3 tokens of Czech
CHUNK_MAX = 1800
CHUNK_MIN = 300  # a shorter tail is merged into the previous chunk
SECTION_MAX_TOKENS = 8192

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[„\"(A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ0-9])")


# ---------------------------------------------------------------- chunking

def split_long(paragraph: str, limit: int) -> list[str]:
    """Split a paragraph longer than `limit` at sentence boundaries."""
    if len(paragraph) <= limit:
        return [paragraph]
    out, cur = [], ""
    for sent in _SENTENCE_END.split(paragraph):
        if cur and len(cur) + 1 + len(sent) > limit:
            out.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        out.append(cur)
    return out


def chunk_section(text: str) -> list[str]:
    """Pack paragraphs into ~CHUNK_TARGET-char chunks without crossing CHUNK_MAX."""
    paras = [p for raw in text.split("\n\n") for p in split_long(raw.strip(), CHUNK_MAX) if p]
    chunks, cur = [], []
    for p in paras:
        size = sum(len(x) + 2 for x in cur)
        if cur and (size >= CHUNK_TARGET or size + len(p) > CHUNK_MAX):
            chunks.append("\n\n".join(cur))
            cur = []
        cur.append(p)
    if cur:
        tail = "\n\n".join(cur)
        if chunks and len(tail) < CHUNK_MIN and len(chunks[-1]) + len(tail) <= CHUNK_MAX + CHUNK_MIN:
            chunks[-1] += "\n\n" + tail
        else:
            chunks.append(tail)
    return chunks


def book_titles() -> dict[str, str]:
    from epub_parser import book_title
    import zipfile

    titles = {}
    for book in BOOKS:
        epub = EPUB_DIR / f"{book}.epub"
        if epub.exists():
            with zipfile.ZipFile(epub) as z:
                titles[book] = book_title(z) or book
    return titles


def make_records(books=BOOKS) -> tuple[list[dict], list[dict]]:
    titles = book_titles()
    sections, chunks = [], []
    for book in books:
        data = json.loads((JSON_DIR / f"{book}.json").read_text(encoding="utf-8"))
        for i, sec in enumerate(data):
            sid = f"{book}:{i:03d}"
            chapter = sec.get("chapter", "")
            sections.append({
                "id": sid, "book": book, "book_title": titles.get(book, book),
                "chapter": chapter, "name": sec["name"], "text": sec["text"], "chars": len(sec["text"]),
            })
            for j, ch in enumerate(chunk_section(sec["text"])):
                chunks.append({
                    "id": f"{sid}:{j:02d}", "section_id": sid, "book": book,
                    "book_title": titles.get(book, book), "chapter": chapter, "name": sec["name"],
                    "text": ch, "chars": len(ch),
                })
    return sections, chunks


def heading_path(rec: dict) -> str:
    return " › ".join(p for p in (rec.get("chapter"), rec["name"]) if p)


def embed_input(rec: dict) -> str:
    # The heading path is the strongest topic signal we have, so it goes in front
    # (for the anag aphorisms the name is the aphorism itself).
    return f"{heading_path(rec)}\n\n{rec['text']}"


# ---------------------------------------------------------------- model

def load_model(device: str | None = None):
    import torch
    from sentence_transformers import SentenceTransformer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(MODEL_NAME, device=device)
    if device == "cuda":
        model.half()
    model.max_seq_length = SECTION_MAX_TOKENS
    return model


def encode(model, texts: list[str], batch_size: int) -> np.ndarray:
    # Longest first: batches have similar lengths, so less padding and OOM shows up early.
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    vecs = model.encode([texts[i] for i in order], batch_size=batch_size,
                        normalize_embeddings=True, show_progress_bar=True,
                        convert_to_numpy=True)
    out = np.empty_like(vecs, dtype=np.float32)
    out[order] = vecs
    return out


def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build(args):
    sections, chunks = make_records(args.books)
    print(f"{len(sections)} sections, {len(chunks)} chunks "
          f"(chunk chars median {int(np.median([c['chars'] for c in chunks]))}, "
          f"max {max(c['chars'] for c in chunks)})")

    t0 = time.time()
    model = load_model(args.device)
    print(f"model {MODEL_NAME} on {model.device}")
    tok = model.tokenizer
    sec_tokens = [len(tok(embed_input(s))["input_ids"]) for s in sections]
    truncated = sum(t > SECTION_MAX_TOKENS for t in sec_tokens)

    print("embedding chunks...")
    chunk_vecs = encode(model, [embed_input(c) for c in chunks], args.batch_size)
    print("embedding sections...")
    section_vecs = encode(model, [embed_input(s) for s in sections], max(1, args.batch_size // 8))

    OUT_DIR.mkdir(exist_ok=True)
    write_jsonl(OUT_DIR / "sections.jsonl", sections)
    write_jsonl(OUT_DIR / "chunks.jsonl", chunks)
    np.save(OUT_DIR / "section_embeddings.npy", section_vecs)
    np.save(OUT_DIR / "chunk_embeddings.npy", chunk_vecs)
    manifest = {
        "model": MODEL_NAME, "dim": int(chunk_vecs.shape[1]), "normalized": True,
        "books": args.books, "sections": len(sections), "chunks": len(chunks),
        "chunk_target_chars": CHUNK_TARGET, "chunk_max_chars": CHUNK_MAX,
        "section_tokens_max": max(sec_tokens), "sections_truncated": truncated,
        "embed_input": "chapter › name + blank line + text", "device": str(model.device),
        "seconds": round(time.time() - t0, 1), "created": datetime.now().isoformat(timespec="seconds"),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------- search

class BookCorpus:
    def __init__(self, sections, chunks, section_vecs, chunk_vecs, model=None):
        self.sections, self.chunks = sections, chunks
        self.section_vecs, self.chunk_vecs = section_vecs, chunk_vecs
        self._model = model

    @classmethod
    def load(cls, directory: Path = OUT_DIR, model=None) -> "BookCorpus":
        return cls(read_jsonl(directory / "sections.jsonl"), read_jsonl(directory / "chunks.jsonl"),
                   np.load(directory / "section_embeddings.npy"), np.load(directory / "chunk_embeddings.npy"),
                   model)

    @property
    def model(self):
        if self._model is None:
            self._model = load_model()
        return self._model

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)

    def search(self, query: str, k: int = 10, level: str = "chunk", book: str | None = None) -> list[dict]:
        """Cosine-similarity search. level='chunk' returns passages, 'section' returns topics."""
        records, vecs = (self.chunks, self.chunk_vecs) if level == "chunk" else (self.sections, self.section_vecs)
        scores = vecs @ self.embed_query(query)
        if book:
            scores = np.where([r["book"] == book for r in records], scores, -np.inf)
        top = np.argsort(-scores)[:k]
        return [{**records[i], "score": float(scores[i])} for i in top if np.isfinite(scores[i])]


def snippet(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[:width].rsplit(" ", 1)[0] + " …"


def search_cli(args):
    corpus = BookCorpus.load()
    for q in args.queries:
        hits = corpus.search(q, k=args.k, level=args.level, book=args.book)
        print(f"\n=== {q!r}  ({args.level}s)")
        for rank, h in enumerate(hits, 1):
            print(f"{rank:2d}. {h['score']:.3f}  [{h['book_title']}] {heading_path(h)}")
            print(f"      {snippet(h['text'], args.width)}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="chunk + embed jsons/<book>.json into corpus/")
    b.add_argument("--books", nargs="+", default=BOOKS)
    b.add_argument("--batch-size", type=int, default=16)
    b.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    b.set_defaults(func=build)

    s = sub.add_parser("search", help="semantic search over the corpus")
    s.add_argument("queries", nargs="+")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--level", choices=["chunk", "section"], default="chunk")
    s.add_argument("--book", choices=BOOKS, default=None)
    s.add_argument("--width", type=int, default=220, help="snippet length")
    s.set_defaults(func=search_cli)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
