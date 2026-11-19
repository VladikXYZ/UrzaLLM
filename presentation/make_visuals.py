"""Draw the five slide visuals (16:9 PNG, 1920x1080) for the project presentation.

    python presentation/make_visuals.py          # in PowerShell (slide 5 needs torch + bge-m3 for the query)

Colours follow the semantic map (Projection/search_map.py): articles blue, books orange, videos aqua.
Numbers come from real_data/ and the transcription outputs (see the comments next to them).
"""
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "visuals"
# Style of the deck "UrzaBOT part 1": black slides, yellow #FFD100 titles/accents, grey body text, Roboto.
INK, INK2, MUTED, LINE, SOFT, BG = "#ffffff", "#c4c7c5", "#939393", "#3a3a3a", "#161616", "#000000"
ACCENT = "#ffd100"
# source colours: the dark-surface steps of the validated categorical palette (slots 1-3), as on the map
ARTICLE, BOOK, VIDEO, TRANSCRIPT = "#3987e5", "#d95926", "#199e70", ACCENT
plt.rcParams.update({"font.family": "Roboto", "font.size": 14, "text.color": INK})


def canvas():
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 90)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, fc=SOFT, ec="none", r=2.0, lw=1.5, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))


def arrow(ax, a, b, color=MUTED, lw=2.2, z=2):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=22, color=color, lw=lw, zorder=z,
                                 shrinkA=0, shrinkB=0))


def icon(ax, kind, cx, cy, color, r=4.6):
    """A simple glyph in a coloured disc: book, article or video."""
    ax.add_patch(Circle((cx, cy), r, fc=color, ec="none", zorder=3))
    s = r * 0.5
    if kind == "book":
        ax.add_patch(Rectangle((cx - s, cy - s * 0.8), 2 * s, 1.6 * s, fc="white", ec="none", zorder=4))
        ax.plot([cx, cx], [cy - s * 0.8, cy + s * 0.8], color=color, lw=2.4, zorder=5)
        for k in (-0.35, 0, 0.35):
            ax.plot([cx - s * 0.75, cx - s * 0.15], [cy + k * s] * 2, color=color, lw=1.4, zorder=5)
            ax.plot([cx + s * 0.15, cx + s * 0.75], [cy + k * s] * 2, color=color, lw=1.4, zorder=5)
    elif kind == "article":
        ax.add_patch(Polygon([(cx - s * 0.7, cy - s), (cx + s * 0.7, cy - s), (cx + s * 0.7, cy + s * 0.55),
                              (cx + s * 0.25, cy + s), (cx - s * 0.7, cy + s)], fc="white", ec="none", zorder=4))
        for k in (0.45, 0.1, -0.25, -0.6):
            ax.plot([cx - s * 0.45, cx + s * 0.45], [cy + k * s] * 2, color=color, lw=1.4, zorder=5)
    else:
        box(ax, cx - s, cy - s * 0.65, 2 * s, 1.3 * s, fc="white", r=0.6, z=4)
        ax.add_patch(Polygon([(cx - s * 0.3, cy - s * 0.4), (cx - s * 0.3, cy + s * 0.4), (cx + s * 0.45, cy)],
                             fc=color, ec="none", zorder=5))


def save(fig, name):
    """Cropped to the content: each slide has its own title, so the images carry none."""
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, facecolor=BG)
    plt.close(fig)
    im = Image.open(OUT / name).convert("RGB")
    ink = np.asarray(im).max(axis=2) > 24             # anything visibly brighter than the black background
    ys, xs = np.where(ink)
    pad = 36
    im.crop((max(0, xs.min() - pad), max(0, ys.min() - pad),
             min(im.width, xs.max() + pad), min(im.height, ys.max() + pad))).save(OUT / name)
    print("wrote", OUT / name)


# ------------------------------------------------------------------ 1. goal: sources + pipeline
def slide1():
    fig, ax = canvas()
    sources = [("book", BOOK, "2 books", "Anarchokapitalismus,\nAnarchoagorismus · ~99k words"),
               ("article", ARTICLE, "1,953 articles", "stoky.urza.cz\n2015 – 2026"),
               ("video", VIDEO, "980 videos", "YouTube · 1,543 hours\n2016 – 2026")]
    steps = ["Scrape", "Preprocess", "Embed", "RAG\n(retrieve)", "LLM", "Answer\nas Urza"]
    w, gap, y = 20.5, 5.4, 18
    x0 = (160 - (len(steps) * w + (len(steps) - 1) * gap)) / 2
    for k, (kind, col, head, sub) in enumerate(sources):
        x = 9 + k * 49
        box(ax, x, 52, 45, 22, fc=BG, ec=LINE)
        icon(ax, kind, x + 8, 63, col)
        ax.text(x + 15.5, 67.2, head, fontsize=21, weight="bold", va="center")
        ax.text(x + 15.5, 59.2, sub, fontsize=13, color=INK2, va="center", linespacing=1.5)
        # every source goes into the start of the pipeline
        ax.add_patch(FancyArrowPatch((x + 22.5, 51.5), (x0 + w / 2 + (k - 1) * 3, y + 17.6), arrowstyle="-|>",
                                     mutation_scale=20, color=MUTED, lw=2, connectionstyle="arc3,rad=0.0", zorder=2))
    for k, s in enumerate(steps):
        x = x0 + k * (w + gap)
        last = k == len(steps) - 1
        box(ax, x, y, w, 17, fc=ACCENT if last else SOFT, ec="none" if last else LINE)
        ax.text(x + w / 2, y + 8.5, s, ha="center", va="center", fontsize=17, weight="bold",
                color="#000000" if last else INK, linespacing=1.15)
        if k:
            arrow(ax, (x - gap + 0.6, y + 8.5), (x - 0.6, y + 8.5))
    ax.text(80, 8, "A question is matched to the most relevant passages he wrote or said,"
                    " and the LLM answers from those sources only.", ha="center", fontsize=15, color=INK2)
    save(fig, "slide1_goal.png")


# ------------------------------------------------------------------ 2. scraping: three columns
def slide2():
    fig, ax = canvas()
    cols = [("book", BOOK, "Books",
             ["2 EPUB files\nAnarchokapitalismus,\nAnarchoagorismus",
              "epub_parser.py\nsplit at headings (h1–h4);\ncover, contents, supporter\nlists dropped",
              "113 sections\n~99k words"]),
            ("article", ARTICLE, "Articles",
             ["stoky.urza.cz\nUrza's author page",
              "requests + BeautifulSoup\ntitle, date, text,\nlinked video",
              "1,953 articles\n2015 – 2026"]),
            ("video", VIDEO, "YouTube",
             ["Kanál Svobodného\npřístavu",
              "yt-dlp\n2 parallel workers,\ncheckpoints (resumable)",
              "980 videos\nmetadata + 1,543 h\nof audio"])]
    rows = ["SOURCE", "TOOL", "RESULT"]
    for c, (kind, col, name, cells) in enumerate(cols):
        x = 12 + c * 48
        icon(ax, kind, x + 6, 72.5, col, r=4.2)
        ax.text(x + 12.5, 72.5, name, fontsize=22, weight="bold", va="center")
        for r_, (label, txt) in enumerate(zip(rows, cells)):
            y = 50 - r_ * 21
            last = r_ == 2
            box(ax, x, y, 40, 16, fc=BG if not last else SOFT, ec=col if last else LINE, lw=2.2 if last else 1.5)
            ax.text(x + 2.2, y + 13.6, label, fontsize=10.5, color=MUTED, weight="bold", va="center")
            ax.text(x + 20, y + 6.6, txt, ha="center", va="center", fontsize=14.5 if not last else 16,
                    weight="bold" if last else "normal", linespacing=1.3)
            if r_:
                arrow(ax, (x + 20, y + 21), (x + 20, y + 16.6))
    save(fig, "slide2_scraping.png")


# ------------------------------------------------------------------ 3. preprocessing: text -> chunk card
def slide3():
    c = json.loads((ROOT / "real_data" / "chunked.json").read_text(encoding="utf-8"))
    book, art = c[523], c[2127]                       # Anarchokapitalismus › Drogy;  article "Drogy a zbraně"
    fig, ax = canvas()

    def snippet(x, y, kind, col, head, text):
        box(ax, x, y, 52, 25, fc=BG, ec=LINE)
        icon(ax, kind, x + 5.5, y + 19.5, col, r=3.4)
        ax.text(x + 10.5, y + 19.5, head, fontsize=14, weight="bold", va="center")
        ax.text(x + 3, y + 13.5, textwrap.fill(text, 62), fontsize=11.5, color=INK2, va="top", linespacing=1.35)

    snippet(6, 49, "book", BOOK, "Book · Anarchokapitalismus › Drogy", " ".join(book["text_cs"].split()[:44]) + " …")
    snippet(6, 18, "article", ARTICLE, f"Article · {art['title_cs']}", " ".join(art["text_cs"].split()[:44]) + " …")
    arrow(ax, (59, 61.5), (72, 52))
    arrow(ax, (59, 30.5), (72, 38))
    # the chunk card (the book chunk's real fields)
    x, y, w, h = 74, 18, 80, 56
    box(ax, x, y, w, h, fc=SOFT, ec=BOOK, lw=2.5)
    ax.text(x + 3, y + h - 4.5, "chunk", fontsize=12, color=MUTED, weight="bold")
    fields = [("type", book["type"] + "   (book: Anarchokapitalismus)"),
              ("title_cs / title_en", f"{book['title_cs']}  /  {book['title_en']}"),
              ("subtitle_cs / _en", f"{book['subtitle_cs']}  /  {book['subtitle_en']}"),
              ("text_cs", " ".join(book["text_cs"].split()[:26]) + " …"),
              ("text_en", " ".join(book["text_en"].split()[:26]) + " …"),
              ("src", "book (no URL)   ·   articles and videos: their URL")]
    yy = y + h - 10
    for k, v in fields:
        ax.text(x + 3, yy, k, fontsize=12, color=INK2, weight="bold", va="top", family="Consolas")
        wrapped = textwrap.fill(v, 50)
        ax.text(x + 25, yy, wrapped, fontsize=12.5, va="top", linespacing=1.3)
        yy -= 5.2 + 2.75 * wrapped.count("\n")
    ax.text(80, 5, "975 book paragraphs + 956 articles + 980 video descriptions = 2,911 chunks"
                   "   ·   videos: speech-to-text, only Urza's own words", ha="center", fontsize=14, color=INK2)
    save(fig, "slide3_preprocessing.png")


# ------------------------------------------------------------------ 4. final data: words per source
def slide4():
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
    # top: the written corpus by source (words, from real_data/chunked.json)
    ax1 = fig.add_axes([0.25, 0.56, 0.62, 0.26])
    names = ["Books", "Articles", "Video descriptions"]
    vals = [98_830, 164_453, 126_922]
    chunks = [975, 956, 980]
    cols = [BOOK, ARTICLE, VIDEO]
    ys = np.arange(3)[::-1]
    ax1.barh(ys, vals, height=0.55, color=cols)
    for y, v, n in zip(ys, vals, chunks):
        ax1.text(v + 2500, y, f"{v / 1000:.0f}k words · {n:,} chunks", va="center", fontsize=15, color=INK)
    ax1.set_yticks(ys, names, fontsize=16)
    ax1.set_xlim(0, 230_000)
    ax1.set_title("Written sources: 2,911 chunks, ~390k words", loc="left", fontsize=17, color=INK2, pad=10)
    # bottom: written vs transcripts on one shared scale (Urza's speech only, from video_chunks.jsonl)
    ax2 = fig.add_axes([0.25, 0.12, 0.62, 0.26])
    v2 = [390_205, 9_522_299]
    ax2.barh([1, 0], v2, height=0.55, color=[MUTED, TRANSCRIPT])
    ax2.text(v2[0] + 120_000, 1, "~390k words", va="center", fontsize=15)
    ax2.text(v2[1] - 120_000, 0, "9.5 M words · 57,693 chunks  (≈ 24×)", va="center", ha="right", fontsize=15,
             color="#000000", weight="bold")
    ax2.set_yticks([1, 0], ["All written sources", "Urza's speech in videos\n(872 h of 1,543 h)"], fontsize=16)
    ax2.set_xlim(0, 10_000_000)
    ax2.set_title("Same scale: written sources vs. video transcripts", loc="left", fontsize=17, color=INK2, pad=10)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        ax.tick_params(axis="y", colors=INK)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(LINE)
        ax.tick_params(axis="x", colors=MUTED, labelsize=12)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v / 1e6:.0f} M" if v >= 1e6 else (f"{v / 1e3:.0f}k" if v else "0")))
        ax.grid(axis="x", color="#262626", lw=1)
        ax.set_axisbelow(True)
    save(fig, "slide4_final_data.png")


# ------------------------------------------------------------------ 5. embedding: the semantic map + a question
def slide5(query="Should the state regulate drugs?"):
    import os
    import pyarrow.parquet as pq
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer
    sys.path.insert(0, str(ROOT / "Projection"))
    from real_data_coords import source_of
    chunks = json.loads((ROOT / "real_data" / "chunked.json").read_text(encoding="utf-8"))
    coords = pq.read_table(ROOT / "real_data" / "projection" / "coords_bge_m3_cs.parquet").to_pydict()
    P = np.c_[coords["x_raw"], coords["y_raw"]]
    src = np.array([source_of(c) for c in chunks])
    E = np.load(ROOT / "real_data" / "embeddings_out" / "bge_m3_vectors.npy").astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    q = SentenceTransformer("BAAI/bge-m3").encode([query], normalize_embeddings=True)[0]
    s = E @ q
    top = np.argsort(-s)[:10]
    w = np.exp((s[top] - s[top].max()) / 0.02)
    qpos = (w[:, None] * P[top]).sum(0) / w.sum()          # as search_map.py places a question
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
    ax = fig.add_axes([0.03, 0.04, 0.6, 0.86])
    for name, col, lab in (("article", ARTICLE, "Articles"), ("book", BOOK, "Books"), ("video", VIDEO, "Video descriptions")):
        m = src == name
        ax.scatter(P[m, 0], P[m, 1], s=9, color=col, alpha=0.55, lw=0, label=f"{lab} ({m.sum():,})")
    ax.set_facecolor(BG)
    for i in top:
        ax.plot([qpos[0], P[i, 0]], [qpos[1], P[i, 1]], color=ACCENT, lw=0.9, alpha=0.5, zorder=3)
    ax.scatter(P[top, 0], P[top, 1], s=70, color=[{"article": ARTICLE, "book": BOOK, "video": VIDEO}[x] for x in src[top]],
               edgecolor=BG, lw=1.5, zorder=4)
    ax.scatter([qpos[0]], [qpos[1]], marker="*", s=700, color=ACCENT, edgecolor=BG, lw=1.5, zorder=5)
    ax.annotate(f"“{query}”", qpos, xytext=(18, 18), textcoords="offset points", fontsize=15, weight="bold",
                color=ACCENT, bbox=dict(boxstyle="round,pad=0.35", fc=SOFT, ec=LINE), zorder=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    leg = ax.legend(loc="lower left", frameon=False, fontsize=13, markerscale=2.2, labelcolor=INK)
    for h in leg.legend_handles:           # solid in the legend, though the map dots are translucent
        h.set_alpha(1)
    fig.text(0.03, 0.93, "2,911 chunks embedded with BGE-M3, projected to 2D (UMAP)", fontsize=15, color=INK2)
    # right: the top hits
    fig.text(0.66, 0.84, "Nearest chunks to the question", fontsize=17, color=INK2, weight="bold")
    y = 0.79
    label = {"article": "Article", "book": "Book", "video": "Video"}
    for r, i in enumerate(top[:6], 1):
        c = chunks[i]
        title = c["title_en"] if not c["subtitle_en"] else f"{c['title_en']} › {c['subtitle_en']}"
        col = {"article": ARTICLE, "book": BOOK, "video": VIDEO}[src[i]]
        fig.text(0.666, y + 0.004, "●", fontsize=17, color=col, va="top", family="DejaVu Sans")   # Roboto has no ●
        fig.text(0.686, y, f"#{r}  {label[src[i]]} · similarity {s[i]:.2f}", fontsize=12.5, color=INK2, va="top")
        fig.text(0.686, y - 0.03, textwrap.fill(title, 44), fontsize=13.5, va="top", weight="bold")
        y -= 0.085 + 0.028 * textwrap.fill(title, 44).count("\n")
    save(fig, "slide5_embedding.png")


if __name__ == "__main__":
    only = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    for k in only:
        {"1": slide1, "2": slide2, "3": slide3, "4": slide4, "5": slide5}[k]()
