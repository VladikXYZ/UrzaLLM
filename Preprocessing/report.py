"""Build the preprocessing report (Markdown + self-contained HTML) and its figures."""
import base64
import html
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config as C  # noqa: E402

# Validated categorical palette, fixed order: article, book, video.
COLORS = {"article": "#2a78d6", "book": "#eb6834", "video": "#1baf7a"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SOURCES = ["article", "book", "video"]
SOURCE_LABEL = {"article": "Articles", "book": "Books", "video": "Videos"}

REASON_TEXT = {
    "video_only": "Post with no text whose body embeds or links a video (field `video` from the scraper); the video itself is kept.",
    "video_post_linked_to_video": "Post with no text and no video link, matched to a channel video by title; the video itself is kept.",
    "duplicate_text": "Exactly the same raw text as an earlier article (earliest copy kept); posts whose text differs in any way are kept.",
    "no_text_no_title": "No text and a title without words (e.g. “????”).",
    "front_matter": "Before the main text: blurb, about the author, title page, dedication and epigraph, acknowledgements, contents, ISBN.",
    "back_matter": "After the main text: partner organisations and adverts (ankap “Nakloněni svobodě”, “Jiná cesta…”), sponsors, colophon.",
    "heading_only": "A heading that only groups subsections (e.g. “AMEN › Ekonomie”); its subsections are kept.",
}
KIND_TEXT = {
    "article": "Blog post with real text.",
    "announcement": "Short post (≤ %d words) announcing an event, stream, cancellation, etc." % C.ANNOUNCEMENT_MAX_WORDS,
    "video_post": "Post with no text whose video was not found on the channel; its title stands in for it.",
    "book_section": "Book chapter or subchapter.",
    "book_story": "Narrative stories at the start of Anarchokapitalismus.",
    "book_faq": "AMEN sections: objections and answers (ankap) or aphorisms with commentary (anag).",
    "video": "YouTube video: title + cleaned description + chapter titles + tags.",
}
FLAG_TEXT = {
    "title_only": "Text is just the title.",
    "accompanies_video": "Article with text and no video link whose title matches a channel video (kept).",
    "contains_video": "Article with text whose body also embeds or links a video (kept; see section 4).",
    "short": f"Video shorter than {C.SHORT_VIDEO_SECONDS // 60} minutes (clip or trailer).",
    "thin_description": f"Cleaned description under {C.THIN_DESCRIPTION_WORDS} words.",
    "has_guests": "Description lists guests besides Urza (interview, debate, guest lecture).",
    "was_live": "Archived livestream.",
    "has_chapters": "Video has YouTube chapters (their titles are included).",
    "merged_sections": "Book section made of several aphorisms that share one commentary page.",
}


# --- tiny document model rendered to both Markdown and HTML -----------------
class Doc:
    def __init__(self):
        self.blocks = []

    def h(self, level, text):
        self.blocks.append(("h", level, text))

    def p(self, text):
        self.blocks.append(("p", text))

    def ul(self, items):
        self.blocks.append(("ul", items))

    def table(self, headers, rows, align=None):
        self.blocks.append(("table", headers, rows, align))

    def img(self, path, alt):
        self.blocks.append(("img", path, alt))

    def md(self, fig_rel) -> str:
        out = []
        for b in self.blocks:
            if b[0] == "h":
                out.append("#" * b[1] + " " + b[2])
            elif b[0] == "p":
                out.append(b[1])
            elif b[0] == "ul":
                out.append("\n".join(f"- {i}" for i in b[1]))
            elif b[0] == "table":
                _, hd, rows, align = b
                al = align or ["l"] * len(hd)
                out.append("| " + " | ".join(hd) + " |")
                out.append("|" + "|".join("---:" if a == "r" else "---" for a in al) + "|")
                for r in rows:
                    out.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in r) + " |")
            elif b[0] == "img":
                out.append(f"![{b[2]}]({fig_rel}/{Path(b[1]).name})")
            out.append("")
        return "\n".join(out)

    def html(self, title) -> str:
        def inline(t):
            # allow **bold** and `code` in plain-text blocks
            t = html.escape(str(t))
            import re
            t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
            return re.sub(r"`(.+?)`", r"<code>\1</code>", t)

        body = []
        toc = []
        for b in self.blocks:
            if b[0] == "h":
                if b[1] == 1:
                    body.append(f"<h1>{inline(b[2])}</h1>")
                else:
                    anchor = f"s{len(toc)}"
                    if b[1] == 2:
                        toc.append((anchor, b[2]))
                    body.append(f'<h{b[1]} id="{anchor}">{inline(b[2])}</h{b[1]}>')
            elif b[0] == "p":
                body.append(f"<p>{inline(b[1])}</p>")
            elif b[0] == "ul":
                body.append("<ul>" + "".join(f"<li>{inline(i)}</li>" for i in b[1]) + "</ul>")
            elif b[0] == "table":
                _, hd, rows, align = b
                al = align or ["l"] * len(hd)
                th = "".join(f'<th class="{a}">{inline(h)}</th>' for h, a in zip(hd, al))
                trs = "".join(
                    "<tr>" + "".join(f'<td class="{a}">{inline(c)}</td>' for c, a in zip(r, al)) + "</tr>"
                    for r in rows)
                body.append(f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>')
            elif b[0] == "img":
                data = base64.b64encode(Path(b[1]).read_bytes()).decode()
                body.append(f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(b[2])}">'
                            f"<figcaption>{inline(b[2])}</figcaption></figure>")
        nav = "".join(f'<li><a href="#{a}">{html.escape(t)}</a></li>' for a, t in toc)
        return HTML_TEMPLATE.format(title=html.escape(title), nav=nav, body="\n".join(body))


HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --bg:#f9f9f7; --card:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --line:#e4e3df; --accent:#2a78d6; --code:#f0efec; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0d0d0d; --card:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --line:#383835; --accent:#3987e5; --code:#262624; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }}
.wrap {{ display:grid; grid-template-columns:220px minmax(0,1fr); gap:32px; max-width:1180px; margin:0 auto; padding:32px 16px; }}
nav {{ position:sticky; top:24px; align-self:start; font-size:13px; }}
nav ul {{ list-style:none; padding:0; margin:8px 0 0; }} nav li {{ margin:4px 0; }}
nav a {{ color:var(--ink2); text-decoration:none; }} nav a:hover {{ color:var(--accent); }}
main {{ min-width:0; }}
h1 {{ font-size:28px; margin:0 0 4px; }}
h2 {{ font-size:20px; margin:40px 0 12px; padding-top:12px; border-top:1px solid var(--line); }}
h3 {{ font-size:16px; margin:24px 0 8px; }}
p, li {{ color:var(--ink); }} ul {{ padding-left:20px; }}
code {{ background:var(--code); padding:1px 5px; border-radius:4px; font-size:13px; }}
.tw {{ overflow-x:auto; margin:8px 0 16px; border:1px solid var(--line); border-radius:8px; background:var(--card); }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; font-variant-numeric:tabular-nums; }}
th, td {{ padding:6px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
th {{ color:var(--ink2); font-weight:600; background:var(--card); }}
tr:last-child td {{ border-bottom:0; }}
td.r, th.r {{ text-align:right; white-space:nowrap; }}
figure {{ margin:12px 0 20px; background:#fcfcfb; border:1px solid var(--line); border-radius:8px; padding:8px; }}
figure img {{ width:100%; height:auto; display:block; }}
figcaption {{ color:#52514e; font-size:13px; padding:6px 4px 0; }}
@media (max-width: 820px) {{ .wrap {{ grid-template-columns:1fr; }} nav {{ position:static; }} }}
</style></head>
<body><div class="wrap"><nav><strong>Contents</strong><ul>{nav}</ul></nav><main>
{body}
</main></div></body></html>"""


# --- figures -----------------------------------------------------------------
def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_per_year(stats, path):
    years = sorted(stats["per_year"])
    x = list(range(len(years)))
    w = 0.38
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, facecolor=SURFACE)
    panels = [
        ("Documents kept per year", "docs", None),
        ("Words kept per year (text that will be embedded)", "words", None),
        ("Hours of video per year (not yet transcribed)", None, "hours"),
    ]
    for ax, (title, key, special) in zip(axes, panels):
        _style(ax)
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
        if special == "hours":
            vals = [stats["video_hours_per_year"].get(y, 0) for y in years]
            ax.bar(x, vals, width=0.6, color=COLORS["video"], edgecolor=SURFACE, linewidth=2)
            for xi, v in zip(x, vals):
                if v:
                    ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=INK2)
        else:
            for off, s in ((-w / 2, "article"), (w / 2, "video")):
                vals = [stats["per_year"][y][s][key] for y in years]
                ax.bar([xi + off for xi in x], vals, width=w, color=COLORS[s],
                       edgecolor=SURFACE, linewidth=2, label=SOURCE_LABEL[s])
            if key == "docs":
                ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK2, ncol=2)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axes[-1].set_xticks(x, years)
    fig.text(0.01, 0.005, "Books are undated and not shown.", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_chunk_lengths(chunks, path):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=False, facecolor=SURFACE)
    top = max(p["max"] for p in C.CHUNKING.values())
    bins = list(range(0, int(1.2 * top) + 25, 25))
    for ax, s in zip(axes, SOURCES):
        _style(ax)
        vals = [c["words"] for c in chunks if c["source"] == s]
        target = C.CHUNKING[s]["target"]
        ax.hist(vals, bins=bins, color=COLORS[s], edgecolor=SURFACE, linewidth=1.5)
        ax.axvline(target, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
        ax.text(target + 6, ax.get_ylim()[1] * 0.92, f"target {target}", fontsize=8, color=INK2)
        ax.set_title(f"{SOURCE_LABEL[s]} ({len(vals):,} chunks)", loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("words per chunk", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# --- report ------------------------------------------------------------------
def _n(v):
    return f"{v:,}" if isinstance(v, int) else ("–" if v is None else str(v))


def _pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "–"


def build_report(docs, chunks, link_log, stats, out: Path):
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    f_year = figdir / "per_year.png"
    f_chunks = figdir / "chunk_lengths.png"
    fig_per_year(stats, f_year)
    fig_chunk_lengths(chunks, f_chunks)

    by_id = {d["doc_id"]: d for d in docs}
    inc = [d for d in docs if d["status"] == "included"]
    S = stats["documents"]
    vs = stats["sources"]["videos"]
    asrc = stats["sources"]["articles"]
    R = Doc()

    R.h(1, "UrzaLLM – preprocessing report")
    R.p(f"Generated {datetime.now():%Y-%m-%d %H:%M} from `{C.DATA_DIR}` in {stats['run_seconds']} s. "
        "Every raw record appears in `documents.jsonl` with a `status`; nothing is silently dropped.")

    # 1 Summary
    R.h(2, "1. Summary")
    tot_raw = sum(S[s]["raw"] for s in SOURCES)
    tot_inc = sum(S[s]["included"] for s in SOURCES)
    ch = stats["chunks"]
    R.p(f"**{tot_raw:,}** raw records went in; **{tot_inc:,}** documents were kept and split into "
        f"**{ch['total']:,}** chunks ready for embedding. The chunks hold {ch['words_total']:,} words, slightly "
        "more than the documents, because a short sentence is repeated where a long document is split.")
    R.table(["Source", "Raw records¹", "Kept", "Excluded", "Words kept", "Chunks"],
            [[SOURCE_LABEL[s], _n(S[s]["raw"]), _n(S[s]["included"]), _n(S[s]["excluded"]),
              _n(S[s]["words_included"]), _n(ch["by_source"].get(s, 0))] for s in SOURCES]
            + [["**Total**", _n(tot_raw), _n(tot_inc), _n(tot_raw - tot_inc),
                _n(sum(S[s]["words_included"] for s in SOURCES)), _n(ch["total"])]],
            ["l", "r", "r", "r", "r", "r"])
    R.p("¹ For books: sections in the table of contents (after merging aphorisms that share a page).")
    hours = sum(d["duration_min"] for d in inc if d["source"] == "video") / 60
    R.p("Things to keep in mind when reading coverage results:")
    R.ul([
        f"**Videos are represented by their descriptions only.** {_n(S['video']['included'])} videos "
        f"({hours:,.0f} hours of speech) contribute {_n(S['video']['words_included'])} words. "
        "Their content is much richer than what is embedded here.",
        "**Text output drops after 2020** while video output grows (see section 8). "
        "A timeline of text alone would wrongly suggest Urza went quiet.",
        f"**{_n(sum(1 for d in inc if d['flags'] and 'has_guests' in d['flags']))} videos have guests.** "
        "Their descriptions summarise a conversation, not only Urza's views.",
        "**Announcements are kept but labelled** (`kind`), so the "
        "embedding and coverage steps can filter them out.",
    ])

    # 2 Exclusions
    R.h(2, "2. What was removed and why")
    ex_rows = []
    for e in stats["exclusions"]:
        ex = [d["title"] for d in docs if d["source"] == e["source"] and d["exclude_reason"] == e["reason"]][:3]
        ex_rows.append([SOURCE_LABEL[e["source"]], f"`{e['reason']}`", _n(e["count"]),
                        REASON_TEXT.get(e["reason"], ""), "; ".join(ex)])
    R.table(["Source", "Reason", "Count", "Meaning", "Examples"], ex_rows, ["l", "l", "r", "l", "l"])
    dups = [d for d in docs if d["exclude_reason"] == "duplicate_text"]
    if dups:
        R.h(3, "Duplicate articles")
        R.table(["Removed", "Date", "Same text as", "Date"],
                [[d["title"], d["date"], by_id[d["duplicate_of"]]["title"], by_id[d["duplicate_of"]]["date"]]
                 for d in dups])

    # 3 Kinds and flags
    R.h(2, "3. What was kept")
    R.p("Each kept document has a `kind`. Filter on it to choose what counts as evidence of coverage.")
    kc = Counter((d["source"], d["kind"]) for d in inc)
    ckc = Counter(c["kind"] for c in chunks)
    R.table(["Source", "Kind", "Documents", "Chunks", "Words", "Meaning"],
            [[SOURCE_LABEL[s], f"`{k}`", _n(n), _n(ckc[k]),
              _n(sum(d["words"] for d in inc if d["source"] == s and d["kind"] == k)), KIND_TEXT.get(k, "")]
             for (s, k), n in sorted(kc.items())], ["l", "l", "r", "r", "r", "l"])
    R.h(3, "Flags")
    R.p("Flags add detail without changing the kind. A document can have several.")
    R.table(["Source", "Flag", "Documents", "Meaning"],
            [[SOURCE_LABEL[f["source"]], f"`{f['flag']}`", _n(f["count"]), FLAG_TEXT.get(f["flag"], "")]
             for f in stats["flags"]], ["l", "l", "r", "l"])

    # 4 Linking
    R.h(2, "4. Articles and videos")
    arts = [d for d in docs if d["source"] == "article"]
    vl = [d for d in arts if d.get("video")]
    if asrc.get("has_video_field"):
        R.h(3, "Articles that contain a video")
        R.p("The scraper records the first video found in each post's body (an embedded YouTube player, a "
            "`<video>` player, or a link to a video host or video file) in the `video` field; links in reader "
            "comments are ignored. **A post with a video and no text is rejected** (`video_only`): it is a "
            "wrapper around the video, and when the video is on the channel, that video is kept as its own "
            "document. **A post with any text is kept**, flagged `contains_video`, with `linked_video_id` set "
            "when the video is on the channel.")

        def row(label, sel):
            g = [d for d in vl if sel(d)]
            rej = sum(1 for d in g if d["status"] == "excluded")
            return [label, _n(len(g)), _n(rej), _n(len(g) - rej)]
        on = lambda d: d["link_method"] == "video_link"
        R.table(["Video in the post", "Posts", "Rejected (no text)", "Kept (has text)"], [
            row("YouTube video on Urza's channel", on),
            row("YouTube video elsewhere", lambda d: "youtu" in d["video"] and not on(d)),
            row("Other video (own player, nostr, …)", lambda d: "youtu" not in d["video"]),
            row("**Total**", lambda d: True),
        ], ["l", "r", "r", "r"])
        # How much of the channel do the article posts point to?
        vids = [d for d in docs if d["source"] == "video"]
        by_link = {d["linked_video_id"] for d in arts if d.get("link_method") == "video_link"}
        by_title = {d["linked_video_id"] for d in arts if d.get("link_method") in ("exact", "fuzzy")} - by_link
        covered = by_link | by_title
        n = len(vids)
        R.h(3, f"How many of the {n:,} channel videos have an article")
        R.p("A video link matches a channel video when its YouTube ID equals the video's ID (the `v=` in "
            "`webpage_url` / `original_url`); these matches are certain. Title matching (below) is only used "
            "for posts without a video link.")
        R.table(["Channel videos", "Count", "Share"], [
            ["Linked by a video link in an article (certain)", _n(len(by_link)), _pct(len(by_link), n)],
            ["Linked only by title match", _n(len(by_title)), _pct(len(by_title), n)],
            ["**Covered by any article**", _n(len(covered)), _pct(len(covered), n)],
            ["No article points to them", _n(n - len(covered)), _pct(n - len(covered), n)],
        ], ["l", "r", "r"])
        multi = Counter(d["linked_video_id"] for d in arts if d.get("link_method") == "video_link")
        multi = sum(1 for c in multi.values() if c > 1)
        if multi:
            R.p(f"{multi} videos are linked from more than one article (e.g. an announcement and a later post).")
        nocov = Counter(d["year"] for d in vids if d["doc_id"] not in covered)
        R.p("Videos without an article, by year: " + ", ".join(f"{y}: {c}" for y, c in sorted(nocov.items())) + ".")
        stats["video_coverage"] = {"videos": n, "by_video_link": len(by_link), "by_title_only": len(by_title),
                                   "covered": len(covered), "uncovered": n - len(covered)}
        kept = [d for d in vl if d["status"] == "included"]
        if kept:
            short = sum(1 for d in kept if d["words"] < 60)
            R.p(f"Kept posts with a video hold {sum(d['words'] for d in kept):,} words; {short} of them have "
                f"under 60 words (mostly a short intro to the video). The longest:")
            R.table(["Title", "Date", "Kind", "Words", "Video"],
                    [[d["title"], d["date"], d["kind"], _n(d["words"]), d["video"]]
                     for d in sorted(kept, key=lambda d: -d["words"])[:12]], ["l", "l", "l", "r", "l"])
    else:
        R.p("The article export has no `video` field yet (rerun `stoky_scraper.py`), so posts are "
            "matched to videos by title only.")
    R.h(3, "Matching posts without a video link by title")
    R.p("Posts that have no text and no video link are matched to a video by title. Titles are compared after "
        "removing series prefixes (“Kanál Svobodného přístavu:”) "
        "and phrases like “V 19 hodin živě”. A fuzzy match needs more similarity the further apart the dates are: "
        + ", ".join(f"≥ {s:.2f} within {d} days" if d is not None else f"≥ {s:.2f} at any date"
                    for d, s in C.LINK_TIERS) + ".")
    link_log = [x for x in link_log if not x.get("video")]
    m = Counter(x["method"] if x["matched"] else "unmatched" for x in link_log)
    texted_links = sum(1 for d in docs if d["source"] == "article" and "accompanies_video" in d["flags"])
    R.table(["Result", "Posts", "What happens"], [
        ["Exact title match", _n(m["exact"]), "Post excluded; the video is kept."],
        ["Fuzzy title match", _n(m["fuzzy"]), "Post excluded; the video is kept."],
        ["No match within 30 days", _n(m["unmatched"]), "Kept as `video_post` with its title as text (or excluded if the title has no words)."],
        ["Articles *with* text that match a video", _n(texted_links), "Kept; flagged `accompanies_video` and given `linked_video_id`."],
    ], ["l", "r", "l"])
    far = [x for x in link_log if x["method"] == "exact" and (x["days"] or 0) > 30]
    if far:
        R.p(f"{len(far)} exact matches are more than 30 days apart (reuploads or recurring titles): "
            + "; ".join(f"“{x['title']}” ({x['days']} days)" for x in far) + ".")
    weak = sorted([x for x in link_log if x["method"] == "fuzzy"], key=lambda x: x["score"])[:15]
    if weak:
        R.h(4, "Weakest accepted matches (check these)")
        R.table(["Score", "Days apart", "Post title", "Matched video"],
                [[f"{x['score']:.2f}", _n(x["days"]), x["title"], by_id[x["video_id"]]["title"]] for x in weak],
                ["r", "r", "l", "l"])
    near = sorted([x for x in link_log if not x["matched"] and x["best_candidate"][1] >= 0.5],
                  key=lambda x: -x["best_candidate"][1])[:15]
    if near:
        R.h(4, "Closest rejected candidates")
        R.p("Unmatched posts and the most similar video title. A high score with many days apart is usually a "
            "different video with a similar name.")
        R.table(["Score", "Days apart", "Post title", "Closest video"],
                [[f"{x['best_candidate'][1]:.2f}", _n(x["best_candidate"][2]), x["title"],
                  by_id[x["best_candidate"][0]]["title"]] for x in near], ["r", "r", "l", "l"])
    if not weak and not near and asrc.get("has_video_field"):
        R.p("Every post without text has a video link, so title matching is only needed for the articles "
            "with text in the last row.")

    # 5 Cleaning
    R.h(2, "5. Cleaning details")
    R.h(3, "Articles")
    R.ul([
        f"The scraper split paragraphs at every inline link (“v / Paralelné Polis Košice / ; máte-li…”). "
        f"**{_n(asrc.get('line_joins', 0))}** broken lines were rejoined in "
        f"**{_n(asrc.get('articles_with_line_joins', 0))}** articles.",
        f"**{_n(asrc.get('embed_lines_removed', 0))}** video-player lines (“Váš prohlížeč nepodporuje přehrávání videa…”) removed.",
        f"Series names taken from title prefixes used at least {C.SERIES_MIN_COUNT}×, e.g. "
        + ", ".join(f"{k} ({v})" for k, v in Counter(d["series"] for d in inc if d["source"] == "article" and d["series"]).most_common(6)) + ".",
    ])
    R.h(3, "Video descriptions")
    raw_w, clean_w = vs["description_words_raw"], vs["description_words_clean"]
    R.ul([
        f"Description words: **{raw_w:,} → {clean_w:,}** ({_pct(raw_w - clean_w, raw_w)} was boilerplate).",
        f"Donation block above the `--------` line removed in **{_n(vs.get('donation_block_removed', 0))}** videos; "
        f"another **{_n(vs.get('donation_lines_removed', 0))}** donation lines removed elsewhere.",
        f"**{_n(vs.get('bio_lines_removed', 0))}** bio lines (“– Name; role; …”) removed. Guest names were saved first.",
        "URLs and crypto addresses removed.",
    ])
    guests = Counter(g for d in inc if d["source"] == "video" for g in d["guests"])
    R.p(f"**Guests:** {len(guests):,} distinct people. Most frequent:")
    R.table(["Guest", "Videos"], [[g, n] for g, n in guests.most_common(15)], ["l", "r"])
    R.h(3, "Tags")
    R.ul([
        f"Distinct tags: **{vs['raw_distinct_tags']:,} → {vs['kept_distinct_tags']:,}**.",
        "Dropped as too generic (on most videos or in the list in `config.py`): "
        + ", ".join(f"`{t}`" for t in vs["generic_tags_dropped"]) + ".",
        "Merged synonyms: " + "; ".join(f"{k} ({v}×)" for k, v in vs["synonym_merges"].items()) + ".",
    ])
    tags = Counter(t for d in inc if d["source"] == "video" for t in d["tags"])
    R.p("Most common tags after cleaning:")
    top = tags.most_common(30)
    half = (len(top) + 1) // 2
    R.table(["Tag", "Videos", "Tag", "Videos"],
            [[top[i][0], top[i][1]] + (list(top[i + half]) if i + half < len(top) else ["", ""])
             for i in range(half)], ["l", "r", "l", "r"])

    # 6 Books
    R.h(2, "6. Books")
    R.p("Books are parsed directly from the EPUB files in `books/`, following each book's own table of "
        "contents: every TOC entry becomes a section, and its place in the TOC tree gives its path "
        "(e.g. dodatek › AMEN › Ekonomie › Kartely). The earlier JSON export (`jsons/ankap.json`, "
        "`anag.json`) is no longer used, because its parser gave each EPUB file a single title, and it had "
        "these problems:")
    R.ul([
        "**Sections that share a file were merged under the wrong name.** All six economics objections of "
        "*Anarchokapitalismus*' AMEN section (dumping, cartels, monopolies, …) were one section called “AMEN”; "
        "“Drogy” was hidden inside “dodatek”; the nine subsections of *Anarchoagorismus* “19 Praxe” "
        "(Ježek bez klece, Svobodný přístav, Kryptoanarchie, Agora, …) were one 5,400-word block.",
        "**AMEN categories were mistaken for objections.** “Historie”, “Polemika”, “Přirozené právo” etc. are "
        "groups; the first objection of each group was filed under the group's name.",
        "**Words were split at formatting boundaries** (“pod nikatelé” for “podnikatelé”), because a space "
        "was inserted wherever inline tags met. Text is now taken with the book's own spacing.",
        "**A comparison table** (properties of Bitcoin, Monero, metals and fiat money) came out as a word "
        "salad and was filed as a section called “Vzácnost”. It is now rendered as readable lines inside "
        "“5. Peníze › Kryptopeníze”.",
        "**Adverts and partner pages** (“Nakloněni svobodě”, “Jiná cesta…”) were treated as AMEN objections. "
        "They are now back matter.",
    ])
    R.p("AMEN objections are written as “Námitka: … / Odpověď: …”, which suits question matching. Aphorisms "
        "that share one commentary page are merged into one section (flag `merged_sections`).")
    for key, cfg in C.BOOKS.items():
        bd = [d for d in docs if d.get("book_key") == key]
        bi = [d for d in bd if d["status"] == "included"]
        bs = stats["sources"]["books"][key]
        R.h(3, cfg["title"])
        R.p(f"{bs['sections_parsed']} TOC sections parsed ({bs.get('merged_sections', 0)} merged), "
            f"**{len(bi)} kept** ({sum(d['words'] for d in bi):,} words), {len(bd) - len(bi)} excluded "
            f"(see section 2). Kept sections:")
        R.table(["Kind", "Chapter", "Subsection", "Words", "Chunks"],
                [[d["kind"].replace("book_", ""), d["chapter"], d["subchapter"] or "", _n(d["words"]),
                  _n(sum(1 for c in chunks if c["doc_id"] == d["doc_id"]))] for d in bi],
                ["l", "l", "l", "r", "r"])

    # 7 Chunks
    R.h(2, "7. Chunks")
    R.p("Documents up to the maximum length stay whole. Longer ones are split at paragraph breaks "
        "(or sentences, for very long paragraphs) into chunks near the target length. The last sentence of a "
        "chunk is repeated at the start of the next one when it is short. Each chunk's `embed_text` starts "
        "with a context line, e.g. “Kniha Anarchokapitalismus › 10. Zdravotnictví” or "
        "“Video „…“ (2021; hosté: …)”.")
    R.p("**Books get smaller chunks than articles and videos.** Book text is dense: a 300-word passage often "
        "covers several arguments, and its embedding becomes an average of them that matches no specific "
        "question well. At about 150 words, a book chunk usually holds one argument, about the size of a "
        "typical article or video description, so chunks from all sources are comparable on the map. "
        "Because books now produce more chunks per page, coverage should count distinct documents or the "
        "best match, not raw chunk hits.")
    R.table(["Source", "Target", "Max", "Merge tail under", "Overlap ≤"],
            [[SOURCE_LABEL[s], C.CHUNKING[s]["target"], C.CHUNKING[s]["max"], C.CHUNKING[s]["min_tail"],
              C.CHUNKING[s]["overlap"]] for s in SOURCES], ["l", "r", "r", "r", "r"])
    q = ch["words_quantiles"]
    R.table(["Source", "Chunks", "Docs split", "Min", "Median", "95th pct", "Max"],
            [[SOURCE_LABEL[s], _n(ch["by_source"].get(s, 0)), _n(ch["docs_split"][s]),
              _n(q[s]["p0"]), _n(q[s]["p50"]), _n(q[s]["p95"]), _n(q[s]["p100"])] for s in SOURCES],
            ["l", "r", "r", "r", "r", "r", "r"])
    R.img(f_chunks, "Chunk length in words by source. Dashed line: target length, used only when a document "
                    "longer than the maximum is split; shorter documents are never padded or merged, so most "
                    "articles and video descriptions sit below it. Very short article chunks are announcements. "
                    "A video chunk is the title (in the context line) plus the cleaned description, chapter "
                    "titles and tags.")

    # 8 Time
    R.h(2, "8. Coverage over time")
    R.img(f_year, "Kept documents, words and video hours per year.")
    years = sorted(stats["per_year"])
    R.table(["Year", "Articles", "Article words", "Videos", "Video words", "Video hours"],
            [[y, _n(stats["per_year"][y]["article"]["docs"]), _n(stats["per_year"][y]["article"]["words"]),
              _n(stats["per_year"][y]["video"]["docs"]), _n(stats["per_year"][y]["video"]["words"]),
              stats["video_hours_per_year"].get(y, 0)] for y in years], ["l", "r", "r", "r", "r", "r"])

    # 9 Files
    R.h(2, "9. Output files")
    R.table(["File", "Rows", "Contents"], [
        ["`chunks.parquet` / `chunks.jsonl`", _n(len(chunks)), "One row per chunk: `text`, `embed_text` (with context line) and metadata for the map. **Input for embedding.**"],
        ["`documents.parquet` / `documents.jsonl`", _n(len(docs)), "Every raw record with `status`, `exclude_reason`, `kind`, `flags`, cleaned text and metadata."],
        ["`article_video_links.jsonl`", _n(len(link_log)), "Every post-to-video match decision with score and days apart."],
        ["`stats.json`", "–", "All numbers in this report, machine-readable."],
        ["`report.html` / `report.md`", "–", "This report. `figures/` holds the charts."],
    ], ["l", "r", "l"])

    # 10 Settings
    R.h(2, "10. Settings used")
    R.p("Change these in `config.py` and rerun.")
    keys = ["CHUNKING", "ARTICLE_MIN_WORDS", "ANNOUNCEMENT_MAX_WORDS", "LINK_TIERS", "SHORT_VIDEO_SECONDS",
            "THIN_DESCRIPTION_WORDS", "GENERIC_TAG_SHARE", "SERIES_MIN_COUNT"]
    R.table(["Setting", "Value"], [[f"`{k}`", json.dumps(getattr(C, k), ensure_ascii=False)] for k in keys])

    md_path, html_path = out / "report.md", out / "report.html"
    md_path.write_text(R.md("figures"), encoding="utf-8")
    html_path.write_text(R.html("Preprocessing report"), encoding="utf-8")
    return md_path, html_path
