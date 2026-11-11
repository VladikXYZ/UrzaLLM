"""Turn the book EPUBs into documents: one per table-of-contents section."""
import re
from collections import Counter

import config as C
from epub_books import parse_epub
from textutils import words

NUM_RE = re.compile(r"^(\d+|XX)(\.*)\s+(.+)$")


def _starts(label: str, prefix: str) -> bool:
    return label.casefold().startswith(prefix.casefold())


def clean_label(label: str, top_level: bool) -> str:
    """'1 Svoboda' → '1. Svoboda'; subsection '2.. Nemožnost …' → 'Nemožnost …'."""
    m = NUM_RE.match(label)
    if not m:
        return label
    if top_level and len(m.group(2)) <= 1:
        return f"{m.group(1)}. {m.group(3)}"
    return m.group(3)


def _merge_tiny(sections: list[dict]) -> list[dict]:
    """Merge near-empty sections (an aphorism with only its attribution) into the
    next section with the same parent, so one commentary page isn't split."""
    out = []
    carry = None
    for s in sections:
        tiny = 0 < words(s["text"]) < C.BOOK_MERGE_MAX_WORDS   # judged before merging
        if carry and carry["path"][:-1] == s["path"][:-1]:
            s = dict(s)
            s["merged_labels"] = carry["merged_labels"] + [s["label"]]
            s["text"] = "\n\n".join(t for t in (carry["text"], s["text"]) if t)
            carry = None
        elif carry:
            out.append(carry)
            carry = None
        s.setdefault("merged_labels", [s["label"]])
        if tiny:
            # Keep the aphorism itself as text: „Agora. Anarchie. Akce!“ – SEK3
            s = dict(s)
            own = s["text"].rsplit("\n\n", 1)[-1]
            prev = s["text"][: -len(own)].rstrip() if own != s["text"] else ""
            s["text"] = "\n\n".join(t for t in (prev, f"„{s['label']}“ {own}".strip()) if t)
            carry = s
            continue
        out.append(s)
    if carry:
        out.append(carry)
    return out


def process_book(key: str, cfg: dict) -> tuple[list[dict], dict]:
    sections = parse_epub(C.BOOKS_DIR / cfg["epub"])
    tops = [s["path"][0] for s in sections]
    start = next(i for i, t in enumerate(tops) if _starts(t, cfg["content_from"]))
    end = next((i for i, t in enumerate(tops) if _starts(t, cfg["content_until"])), len(sections))

    in_content = [s for i, s in enumerate(sections) if start <= i < end]
    merged = _merge_tiny(in_content)
    stats = Counter(sections_parsed=len(sections), merged_sections=len(in_content) - len(merged))

    ordered = ([(s, "front_matter") for s in sections[:start]] + [(s, None) for s in merged]
               + [(s, "back_matter") for s in sections[end:]])
    docs = []
    for i, (s, reason) in enumerate(ordered):
        path = [clean_label(p, depth == 0) for depth, p in enumerate(s["path"])]
        labels = s.get("merged_labels", [s["label"]])
        title = " / ".join(clean_label(l, len(s["path"]) == 1) for l in labels)
        kind = next((k for prefix, k in C.BOOK_KINDS if any(_starts(p, prefix) for p in s["path"])),
                    "book_section")
        n_words = words(s["text"])
        if reason is None and n_words == 0:
            reason = "heading_only"
        sub = path[1:-1] + [title] if len(path) > 1 else []
        doc = {
            "doc_id": f"book-{key}-{i:03d}",
            "source": "book",
            "kind": kind,
            "status": "excluded" if reason else "included",
            "exclude_reason": reason,
            "title": title,
            "date": None,
            "year": None,
            "url": None,
            "text": s["text"],
            "raw_words": n_words,
            "words": n_words,
            "series": None,
            "tags": [],
            "guests": [],
            "book": cfg["title"],
            "book_key": key,
            "section_index": i,
            "section_path": path[:-1] + [title],
            "depth": len(path) - 1,
            "chapter": path[0] if len(path) > 1 else title,
            "subchapter": " › ".join(sub) if sub else None,
            "epub_files": s["files"],
            "flags": ["merged_sections"] if len(labels) > 1 else [],
        }
        stats[doc["status"]] += 1
        docs.append(doc)
    return docs, dict(stats)


def process_books() -> tuple[list[dict], dict]:
    all_docs, stats = [], {}
    for key, cfg in C.BOOKS.items():
        d, s = process_book(key, cfg)
        all_docs += d
        stats[key] = s
    return all_docs, stats
