"""Clean YouTube video metadata into documents."""
import re
from collections import Counter

import config as C
from textutils import URL_RE, normalize, words

DONATION_RES = [re.compile(p, re.IGNORECASE) for p in C.DONATION_LINE_PATTERNS]
CRYPTO_RE = re.compile(r"\b(?:bc1|ltc1|0x)[0-9a-zA-Z]{20,}\b")
BIO_RE = re.compile(r"^\s*[–—-]\s+(.+)$")
DASH_LINE_RE = re.compile(r"^\s*[-–—=_]{4,}\s*$")


def _is_donation(line: str) -> bool:
    return any(r.search(line) for r in DONATION_RES) or bool(CRYPTO_RE.search(line))


def _bio_name(line: str):
    """Return the person's name if the line is a '– Name; bio' line, else None."""
    m = BIO_RE.match(line)
    if not m or not (";" in line or "(www." in line):
        return None
    name = re.split(r"[;(]", m.group(1))[0].strip()
    return name if 0 < len(name.split()) <= 6 else None


def clean_description(desc: str) -> tuple[str, list[str], bool, dict]:
    """Remove donation block, bio lines, URLs. Returns (text, guests, urza_bio, stats)."""
    desc = normalize(desc).replace("\r", "")
    lines = desc.split("\n")
    stats = Counter()

    # 1) Drop everything above a dashed separator when that part is donation boilerplate.
    for i, line in enumerate(lines):
        if DASH_LINE_RE.match(line):
            head = lines[:i]
            if any(_is_donation(h) for h in head):
                stats["donation_block_removed"] += 1
                lines = lines[i + 1:]
            break

    kept, guests, urza = [], [], False
    for line in lines:
        s = line.strip()
        if not s:
            kept.append("")
            continue
        if DASH_LINE_RE.match(s):
            continue
        if _is_donation(s):
            stats["donation_lines_removed"] += 1
            continue
        name = _bio_name(s)
        if name:
            stats["bio_lines_removed"] += 1
            if "urza" in name.casefold():
                urza = True
            elif name not in guests:
                guests.append(name)
            continue
        s = URL_RE.sub("", s).strip(" :–-")
        if s:
            kept.append(s)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    # Collapse single newlines inside paragraphs to keep paragraph structure simple.
    paras = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    return "\n\n".join(p for p in paras if p), guests, urza, stats


def build_tag_vocabulary(videos: list[dict]) -> tuple[set, dict]:
    """Decide which tags are generic (dropped) and the display form of each tag."""
    df = Counter()
    casing = Counter()
    for v in videos:
        seen = set()
        for t in v.get("tags") or []:
            t = normalize(t)
            key = C.TAG_SYNONYMS.get(t.casefold(), t.casefold())
            casing[(key, t if t.casefold() == key else key)] += 1
            seen.add(key)
        df.update(seen)
    n = max(len(videos), 1)
    generic = {t.casefold() for t in C.GENERIC_TAGS}
    generic |= {k for k, c in df.items() if c / n > C.GENERIC_TAG_SHARE}
    display = {}
    for (key, form), c in casing.most_common():
        display.setdefault(key, form)
    return generic, display


def series_of(title: str) -> str | None:
    m = re.match(r"^([^:]{3,60}):\s", title)
    return m.group(1).strip() if m else None


def process_videos(raw: list[dict]) -> tuple[list[dict], dict]:
    generic, display = build_tag_vocabulary(raw)
    stats = Counter()
    raw_tag_set, kept_tag_set = set(), set()
    synonym_hits = Counter()
    docs = []
    for v in raw:
        title = normalize(v.get("title") or "")
        desc_raw = v.get("description") or ""
        desc, guests, urza, s = clean_description(desc_raw)
        stats.update(s)
        stats["description_words_raw"] += words(desc_raw)
        stats["description_words_clean"] += words(desc)

        tags, seen = [], set()
        for t in v.get("tags") or []:
            t = normalize(t)
            raw_tag_set.add(t)
            key = t.casefold()
            if key in C.TAG_SYNONYMS:
                synonym_hits[f"{t} → {C.TAG_SYNONYMS[key]}"] += 1
                key = C.TAG_SYNONYMS[key]
            if key in generic or key in seen:
                continue
            seen.add(key)
            tags.append(display.get(key, t))
        kept_tag_set.update(tags)

        chapters = [normalize(c.get("title", "")) for c in (v.get("chapters") or [])]
        parts = []
        if desc:
            parts.append(desc)
        if chapters:
            parts.append("Kapitoly: " + "; ".join(c for c in chapters if c))
        if tags:
            parts.append("Témata: " + ", ".join(tags))
        text = "\n\n".join(parts) if parts else title

        ud = str(v.get("upload_date") or "")
        date = f"{ud[:4]}-{ud[4:6]}-{ud[6:8]}" if len(ud) == 8 else None
        dur = v.get("duration") or 0
        flags = []
        if dur < C.SHORT_VIDEO_SECONDS:
            flags.append("short")
        if words(desc) < C.THIN_DESCRIPTION_WORDS:
            flags.append("thin_description")
        if guests:
            flags.append("has_guests")
        if v.get("was_live") or v.get("live_status") == "was_live":
            flags.append("was_live")
        if chapters:
            flags.append("has_chapters")

        docs.append({
            "doc_id": f"vid-{v['id']}",
            "source": "video",
            "kind": "video",
            "status": "included",
            "exclude_reason": None,
            "title": title,
            "date": date,
            "year": int(date[:4]) if date else None,
            "url": v.get("webpage_url") or f"https://www.youtube.com/watch?v={v['id']}",
            "text": text,
            "raw_words": words(desc_raw) + words(title),
            "words": words(text),
            "series": series_of(title),
            "tags": tags,
            "guests": guests,
            "urza_in_bio": urza,
            "views": v.get("view_count"),
            "likes": v.get("like_count"),
            "comments": v.get("comment_count"),
            "duration_min": round(dur / 60, 1),
            "language": v.get("language"),
            "chapters": chapters,
            "flags": flags,
        })
    stats["raw_distinct_tags"] = len(raw_tag_set)
    stats["kept_distinct_tags"] = len(kept_tag_set)
    stats["generic_tags_dropped"] = sorted(generic)
    stats["synonym_merges"] = dict(synonym_hits.most_common())
    return docs, dict(stats)
