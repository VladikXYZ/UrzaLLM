"""Clean stoky.urza.cz posts into documents."""
import hashlib
import re
from collections import Counter

import config as C
from linking import VideoIndex
from textutils import join_paragraphs, normalize, repair_split_lines, words

YOUTUBE_ID_RE = re.compile(
    r"(?:youtube(?:-nocookie)?\.com/(?:embed/|live/|shorts/|watch\?(?:.*&)?v=)|youtu\.be/)([\w-]{11})")
CUE_RES = [re.compile(p, re.IGNORECASE) for p in C.ANNOUNCEMENT_CUES]
EMBED = {normalize(l) for l in C.VIDEO_EMBED_LINES}


def clean_text(text: str) -> tuple[str, int, int]:
    """Return (clean text, embed lines removed, line joins)."""
    lines = [normalize(l) for l in (text or "").replace("\r", "").split("\n")]
    removed_embed = sum(1 for l in lines if l in EMBED)
    lines = [l for l in lines if l and l not in EMBED]
    paras, joins = repair_split_lines(lines)
    return join_paragraphs(paras), removed_embed, joins


def _series_counts(raw):
    c = Counter()
    for a in raw:
        m = re.match(r"^([^:]{3,60}):\s", a["title"])
        if m:
            c[m.group(1).strip()] += 1
    return c


def process_articles(raw: list[dict], videos: list[dict]) -> tuple[list[dict], dict]:
    index = VideoIndex(videos)
    channel_ids = {v["doc_id"] for v in videos}
    stats_has_video_field = any("video" in a for a in raw)
    series_counts = _series_counts(raw)
    stats = Counter()
    link_log = []           # every link decision, for the report
    seen_hash = {}
    docs = []

    # Oldest first so the earliest copy of a duplicate is kept.
    order = sorted(range(len(raw)), key=lambda i: (raw[i]["date"], i))
    for n, i in enumerate(order, 1):
        a = raw[i]
        title = normalize(a["title"])
        date = a["date"][:10] if a.get("date") else None
        text, embed_removed, joins = clean_text(a.get("text", ""))
        stats["embed_lines_removed"] += embed_removed
        stats["line_joins"] += joins
        if joins:
            stats["articles_with_line_joins"] += 1
        m = re.match(r"^([^:]{3,60}):\s", title)
        series = m.group(1).strip() if m and series_counts[m.group(1).strip()] >= C.SERIES_MIN_COUNT else None

        doc = {
            "doc_id": f"art-{n:04d}",
            "source": "article",
            "kind": "article",
            "status": "included",
            "exclude_reason": None,
            "title": title,
            "date": date,
            "year": int(date[:4]) if date else None,
            "url": a.get("url"),
            "text": text,
            "raw_words": words(a.get("text", "")),
            "words": words(text),
            "series": series,
            "tags": [],
            "guests": [],
            "linked_video_id": None,
            "link_score": None,
            "link_method": None,
            "duplicate_of": None,
            "flags": [],
        }

        video = (a.get("video") or "").strip()
        doc["video"] = video
        yt = YOUTUBE_ID_RE.search(video)
        if video:
            host = ("youtube_on_channel" if yt and f"vid-{yt.group(1)}" in channel_ids
                    else "youtube_other" if yt else "other_video")
            stats[f"video_links_{host}"] += 1

        link = None
        if yt and f"vid-{yt.group(1)}" in channel_ids:
            link = (f"vid-{yt.group(1)}", 1.0, "video_link", None)
        elif doc["words"] < C.ARTICLE_MIN_WORDS or not video:
            link = index.match(title, date)
        if link:
            doc["linked_video_id"], doc["link_score"], doc["link_method"], _ = link

        if video and doc["words"] < C.ARTICLE_MIN_WORDS:
            # No text, only a video: the post is a wrapper around that video, so
            # it is rejected (the video, if it is on the channel, is kept as its
            # own document). Posts with any text are kept (flag "contains_video").
            doc["kind"] = "video_post"
            doc["status"] = "excluded"
            doc["exclude_reason"] = "video_only"
            link_log.append({
                "doc_id": doc["doc_id"], "title": title, "date": date,
                "matched": bool(link), "video_id": link[0] if link else None,
                "score": link[1] if link else None, "method": link[2] if link else None,
                "days": link[3] if link else None, "best_candidate": None, "video": video,
            })
        elif doc["words"] < C.ARTICLE_MIN_WORDS:
            # A post with no real text: a video announcement.
            doc["kind"] = "video_post"
            if link:
                doc["status"] = "excluded"
                doc["exclude_reason"] = "video_post_linked_to_video"
            elif not re.search(r"[^\W\d_]{2,}", title):
                doc["status"] = "excluded"
                doc["exclude_reason"] = "no_text_no_title"
            else:
                doc["text"] = title
                doc["words"] = words(title)
                doc["flags"].append("title_only")
            link_log.append({
                "doc_id": doc["doc_id"], "title": title, "date": date,
                "matched": bool(link),
                "video_id": link[0] if link else None,
                "score": link[1] if link else None,
                "method": link[2] if link else None,
                "days": link[3] if link else None,
                "best_candidate": None if link else index.best_candidate(title, date),
                "video": "",
            })
        else:
            # Only an exact copy of the raw text field counts as a duplicate.
            h = hashlib.sha1((a.get("text") or "").encode("utf-8")).hexdigest()
            if h in seen_hash:
                doc["status"] = "excluded"
                doc["exclude_reason"] = "duplicate_text"
                doc["duplicate_of"] = seen_hash[h]
            else:
                seen_hash[h] = doc["doc_id"]
                if doc["words"] <= C.ANNOUNCEMENT_MAX_WORDS and any(r.search(text) for r in CUE_RES):
                    doc["kind"] = "announcement"
            if video:
                doc["flags"].append("contains_video")
            elif link:
                doc["flags"].append("accompanies_video")
        docs.append(doc)

    stats = dict(stats)
    stats["has_video_field"] = stats_has_video_field
    stats["link_log"] = link_log
    return docs, stats
