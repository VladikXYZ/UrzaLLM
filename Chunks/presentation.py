"""One self-contained HTML presentation built from the final reports.

    python Chunks/presentation.py        # from the repo root, after Chunks/build_reports.py

Reads the reports in FinalReports/ (their interactive maps are embedded as isolated frames, showing only the
needed section of each report), plus the final chunks, embeddings and raw data for the few new numbers.
Writes FinalReports/presentation.html
"""
import html as H
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

os.environ["URZA_DATASET"] = "final"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REP = ROOT / "FinalReports"
WORK = HERE / "output" / "projection"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "Projection"))
DEC = json.JSONDecoder()
SRC_LABEL = {"article": "Articles", "book": "Books", "video": "Videos"}
VARS = [("withsource", "With source"), ("nosource", "No source"), ("textonly", "Text only")]


def esc(s):
    return H.escape(str(s), quote=True)


def pct(v):
    return f"{v:.0%}"


def read(name):
    return (REP / name).read_text(encoding="utf-8")


def consts(page):
    """JSON constants defined in a report's script (const X=[…] / {…})."""
    js = "".join(re.findall(r"<script>(.*?)</script>", page, flags=re.S))
    out = {}
    for m in re.finditer(r"(?:\b(?:const|let|var)\s+|,\s*)([A-Za-z_]\w*)\s*=\s*(?=[\[{])", js):
        if m.group(1) in out:
            continue
        try:
            out[m.group(1)] = DEC.raw_decode(js[m.end():])[0]
        except ValueError:
            pass
    return out


def section(page, title):
    i = page.find(title)
    j = page.find('<h2 class="sec">', i + 10)
    return page[i:j if j > 0 else len(page)]


def kpis(page):
    return [(re.sub("<[^>]+>", "", a), re.sub("<[^>]+>", "", b))
            for a, b in re.findall(r'<div class="kpi"><b>(.*?)</b><span>(.*?)</span></div>', page)]


# ---------------------------------------------------------------- embedded report pages
FRAME_CSS = """<style>
.secwrap{display:none}.secwrap.on{display:block}.secwrap.on>h2.sec{display:none}
html.nocap .secwrap p.cap{display:none}
.wrap{padding:2px 2px 6px!important;max-width:none!important}
body{background:transparent!important}
</style>"""
FRAME_JS = """<script>(function(){var f=window.frameElement,k=((f&&f.dataset.keep)||"").split("|");
document.documentElement.classList.toggle("nocap",!(f&&f.dataset.caps));
document.querySelectorAll(".secwrap").forEach(function(s){s.classList.toggle("on",k.some(function(t){return t&&s.dataset.s.indexOf(t)===0}))});
})();</script>"""


def frame_page(name):
    """The report with every section wrapped; the frame's data-keep attribute decides which are shown."""
    page = read(name)
    w = page.find('<div class="wrap">') + len('<div class="wrap">')
    page = page[:w] + '<div class="secwrap" data-s="intro">' + page[w:]
    page = re.sub(r'<h2 class="sec">(.*?)</h2>',
                  lambda m: f'</div><div class="secwrap" data-s="{esc(re.sub("<[^>]+>", "", m.group(1)))}">'
                            f'<h2 class="sec">{m.group(1)}</h2>', page)
    sc = page.rfind("<script>")
    tip = page.rfind('<div id="tip"', 0, sc)
    close = page.rfind("</div>", 0, tip if tip > 0 else sc)
    page = page[:close] + "</div>" + page[close:]
    sc = page.rfind("<script>")
    page = page[:sc] + FRAME_JS + page[sc:]
    return page.replace("</head>", FRAME_CSS + "</head>", 1)


def frame(page_key, keep, caps=False, h=600):
    return (f'<iframe class="emb" data-page="{page_key}" data-keep="{esc(keep)}"{" data-caps=1" if caps else ""} '
            f'style="height:{h}px" title="{esc(keep.split("|")[0])}"></iframe>')


def switch(group, options, panes):
    btns = "".join(f'<button data-v="{v}"{" class=on" if i == 0 else ""}>{esc(lab)}</button>'
                   for i, (v, lab) in enumerate(options))
    ps = "".join(f'<div class="vpane" data-group="{group}" data-v="{v}"{"" if i == 0 else " hidden"}>{panes[v]}</div>'
                 for i, (v, _) in enumerate(options))
    return f'<div class="vswitch" data-group="{group}"><span>Embedding variant</span><div class="seg">{btns}</div></div>{ps}'


# ---------------------------------------------------------------- data for the sections
def chunk_data():
    from chunk import chunk_rows, strip_video_boilerplate, words
    from report import hist_svg, years_svg
    cs = pq.read_table(HERE / "output" / "chunks_cs.parquet").to_pylist()
    en = pq.read_table(HERE / "output" / "chunks_en.parquet").to_pylist()
    L = lambda p: json.load(open(p, encoding="utf-8"))
    raw = L(ROOT / "jsons" / "urza_articles.json")
    yl = {v["webpage_url"] for v in L(ROOT / "jsons" / "channel_metadata.json")}
    dup = sum(1 for a in raw if (a.get("video") or "") in yl and not (a.get("text") or "").strip())
    linked = sum(1 for a in raw if (a.get("video") or "") in yl and (a.get("text") or "").strip())
    front = ("O knize", "O autorovi", "Věnování", "Poděkování", "Obsah")
    fm = {b: sum(len(x["text"].split()) for x in L(ROOT / "jsons" / f"{b}.json") if x["name"] in front)
          for b in ("ankap", "anag")}
    vrows = [{"source": "video", "text": cs_[1], "chunk_id": m["chunk_id"], "n_words": words(cs_[1])}
             for m, cs_, _ in chunk_rows() if m["source"] == "video"]
    raw_w = sum(r["n_words"] for r in vrows)
    nlines = len(strip_video_boilerplate(vrows))
    kept_w = sum(r["n_words"] for r in vrows)
    med = {s: float(np.median([r["n_words"] for r in cs if r["source"] == s])) for s in ("article", "video", "book")}
    mx = {s: max(r["n_words"] for r in cs if r["source"] == s) for s in ("article", "video", "book")}
    ratio = np.array([b["n_words"] / max(a["n_words"], 1) for a, b in zip(cs, en)])
    return {"n": len(cs), "cnt": Counter(r["source"] for r in cs), "raw": len(raw), "dup": dup, "linked": linked,
            "fm": fm, "nlines": nlines, "cut": 1 - kept_w / raw_w, "med": med, "max": mx,
            "rmed": float(np.median(ratio)), "rlo": float(np.percentile(ratio, 5)), "rhi": float(np.percentile(ratio, 95)),
            "same": sum(1 for a, b in zip(cs, en) if a["text"] == b["text"]),
            "hist": hist_svg(cs), "years": years_svg(cs)}


def pc1_check():
    """PC1's source effect with only the overall mean removed (as in the PCA report) vs each source's mean removed."""
    from sklearn.decomposition import PCA
    src = np.array([r["source"] for r in pq.read_table(HERE / "output" / "chunks.parquet", columns=["source"]).to_pylist()])
    E = np.load(HERE / "output" / "results_textonly" / "embeddings.npy")

    def eta(s):
        m = s.mean()
        return float(sum((src == k).sum() * (s[src == k].mean() - m) ** 2 for k in set(src)) / ((s - m) ** 2).sum())
    Xg = E - E.mean(0)
    Xs = E.copy()
    for k in set(src):
        Xs[src == k] -= Xs[src == k].mean(0)
    out = {}
    for name, X in (("global", Xg), ("source", Xs)):
        p = PCA(3, random_state=42).fit(X)
        out[name] = {"evr1": float(p.explained_variance_ratio_[0]), "eta1": eta(p.transform(X)[:, 0])}
    out["between"] = float(sum((src == k).sum() * ((E[src == k].mean(0) - E.mean(0)) ** 2).sum() for k in set(src))
                           / (Xg ** 2).sum())
    return out


def tight_rows(variant):
    page = read(f"book_map_{variant}.html")
    t = section(page, "How tight is each chapter")
    rows = []
    for m in re.finditer(r'<tr data-g="\d+"><td><span class="sw" style="background:var\(--b(\d)\)"></span>(.*?)</td>'
                         r'<td>(\d+)</td><td>([\d.]+)</td><td>([\d.]+)</td><td>(\d+)%</td></tr>', t):
        rows.append({"book": int(m.group(1)), "name": H.unescape(m.group(2)), "n": int(m.group(3)),
                     "u": float(m.group(4)), "t": float(m.group(5)), "nb": int(m.group(6))})
    return rows


def tight_svg(data):
    """Strip plot: each chapter's spread vs random (log scale), per variant, UMAP and t-SNE side by side."""
    W, rowh, top, pl, pr = 980, 62, 30, 112, 14
    pw = (W - pl - pr - 30) / 2
    lo, hi = np.log10(0.01), np.log10(2)
    X = lambda v, x0: x0 + (np.log10(min(max(v, 0.01), 2)) - lo) / (hi - lo) * pw
    labels = {"nosource": "No source", "textonly": "Text only"}
    el = []
    for pi, (key, name) in enumerate((("u", "UMAP"), ("t", "t-SNE"))):
        x0 = pl + pi * (pw + 30)
        el.append(f'<text x="{x0}" y="14" class="lab">{name}</text>')
        for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2):
            x = X(t, x0)
            el.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top - 6}" y2="{top + rowh * 2}" class="grid"/>'
                      f'<text x="{x:.1f}" y="{top + rowh * 2 + 14}" text-anchor="middle" class="ax">{t:g}</text>')
        for t, txt in ((0.2, "compact"), (1, "random")):
            x = X(t, x0)
            el.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top - 6}" y2="{top + rowh * 2}" class="ref"/>'
                      f'<text x="{x + 3:.1f}" y="{top - 8}" class="ax">{txt}</text>')
        for ri, (v, rows) in enumerate(data.items()):
            yc = top + ri * rowh + rowh / 2
            if pi == 0:
                el.append(f'<text x="{pl - 10}" y="{yc + 4:.1f}" text-anchor="end" class="lab">{labels[v]}</text>')
            bins = Counter()
            for r in sorted(rows, key=lambda r: r[key]):
                x = X(r[key], x0)
                b = int(x // 7)
                k = bins[b]
                bins[b] += 1
                y = yc + ((k + 1) // 2) * 7 * (1 if k % 2 else -1)
                el.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="var(--b{r["book"]})" fill-opacity=".85">'
                          f'<title>{esc(r["name"])}: {r[key]:.2f} ({labels[v]}, {name})</title></circle>')
            med = float(np.median([r[key] for r in rows]))
            xm = X(med, x0)
            el.append(f'<line x1="{xm:.1f}" x2="{xm:.1f}" y1="{yc - 22:.1f}" y2="{yc + 22:.1f}" class="med"/>'
                      f'<text x="{xm:.1f}" y="{yc - 25:.1f}" text-anchor="middle" class="ax">median {med:.2f}</text>')
            el.append(f'<line x1="{x0}" x2="{x0 + pw}" y1="{top + (ri + 1) * rowh:.1f}" y2="{top + (ri + 1) * rowh:.1f}" class="grid"/>')
    hgt = top + rowh * 2 + 22
    return (f'<svg viewBox="0 0 {W} {hgt}" role="img" aria-label="Spread of each chapter relative to random, '
            f'per variant" style="width:100%;height:auto">{"".join(el)}</svg>')


def table_rows(html_table, n=None, cls=None):
    head = re.search(r"<thead>.*?</thead>", html_table, flags=re.S).group(0)
    rows = re.findall(r"<tr>(?:(?!</tr>).)*</tr>", html_table.split("<tbody>", 1)[1], flags=re.S)
    if cls:
        rows = [r for r in rows if f"class='tag {cls}'" in r]
    return head, rows[:n] if n else rows


# ---------------------------------------------------------------- sections
def build():
    sys.stdout.reconfigure(encoding="utf-8")
    pages = {}
    for v, _ in VARS:
        pages[f"pc_{v}"] = frame_page(f"projection_compare_{v}.html")
    for v in ("nosource", "textonly"):
        pages[f"bm_{v}"] = frame_page(f"book_map_{v}.html")
    for k, f in (("pca", "all_pca_ica_textonly.html"), ("trails", "book_trails_textonly.html"),
                 ("topic", "topic_distribution_textonly.html"), ("pol", "political_space_textonly.html")):
        pages[k] = frame_page(f)

    S = []
    # ---- 0. title
    cd = chunk_data()
    S.append(("intro", "UrzaLLM: mapping Urza's ideas", f"""
<p class="lead">Urza's articles, video descriptions and two books, embedded with a multilingual language model, to see
what he writes about, how his ideas connect, and how much of politics in general he covers.</p>
<div class="kpis">
 <div class="kpi"><b>{cd['n']:,}</b><span>chunks of text, in Czech and English</span></div>
 <div class="kpi"><b>{cd['cnt']['article']:,} / {cd['cnt']['video']:,} / {cd['cnt']['book']:,}</b><span>articles / videos / book paragraphs</span></div>
 <div class="kpi"><b>bge-m3</b><span>embedding model; 1,024 dimensions; computed from the Czech originals</span></div>
 <div class="kpi"><b>3</b><span>embedding variants: with source, no source, text only</span></div>
</div>
<ol class="agenda">
 <li><a href="#data">Data and chunking</a></li><li><a href="#maps">Raw vs source-centred maps</a></li>
 <li><a href="#pca">Linear structure: PCA and ICA</a></li><li><a href="#books">The book idea map</a></li>
 <li><a href="#trails">Trails between book topics</a></li><li><a href="#topics">Topic distribution</a></li>
 <li><a href="#politics">Urza in the political space</a></li>
</ol>
<p class="cap">All texts are shown in English translation; every computation uses the Czech originals. Use the menu
or the ← → keys to move between sections. Interactive maps: hover for details, the switches swap variants.</p>"""))

    # ---- 1. data
    c = cd["cnt"]
    S.append(("data", "Data and chunking", f"""
<div class="kpis">
 <div class="kpi"><b>{cd['n']:,}</b><span>chunks per language</span></div>
 <div class="kpi"><b>{c['article']:,} / {c['video']:,} / {c['book']:,}</b><span>article / video / book chunks</span></div>
 <div class="kpi"><b>0</b><span>chunks where Czech and English don't line up</span></div>
</div>
<h3>From raw data to chunks</h3>
<ol>
 <li><strong>Articles and videos without duplicates.</strong> Of the {cd['raw']:,} posts on Urza's blog, {cd['dup']:,} only
 announce a video and have no text of their own; they would duplicate the video, which is represented by its own
 description, so they are left out. Kept are the {c['article']:,} posts with text ({cd['linked']} of them link a video but
 add their own text). All {c['video']:,} videos of the YouTube channel are included.</li>
 <li><strong>Books cleaned.</strong> Both books, <em>Anarchokapitalismus</em> and <em>Anarchoagorismus</em>, were parsed into
 sections; their front matter was removed: about the book, about the author, dedication, acknowledgements and contents
 ({cd['fm']['ankap']:,} + {cd['fm']['anag']:,} words). Prefaces, epilogues and the appendices are content and stay.</li>
 <li><strong>Video descriptions cleaned.</strong> {cd['nlines']} recurring lines (donation requests, crypto addresses,
 bank details, Urza's signature, separators) were removed from every description: {pct(cd['cut'])} of all video words.
 Guest introductions stay, since they say who the video is with.</li>
 <li><strong>Translated.</strong> Every item also exists in English, item by item, section by section.</li>
</ol>
<h3>Final chunking</h3>
<div class="tw"><table><thead><tr><th>Source</th><th>One chunk is</th><th class=r>Chunks</th><th class=r>Median words</th><th class=r>Longest</th></tr></thead><tbody>
<tr><td><span class=sw style="background:var(--s0)"></span>Articles</td><td>the whole article text</td><td class=r>{c['article']:,}</td><td class=r>{cd['med']['article']:.0f}</td><td class=r>{cd['max']['article']}</td></tr>
<tr><td><span class=sw style="background:var(--s2)"></span>Videos</td><td>the whole (cleaned) description</td><td class=r>{c['video']:,}</td><td class=r>{cd['med']['video']:.0f}</td><td class=r>{cd['max']['video']}</td></tr>
<tr><td><span class=sw style="background:var(--s1)"></span>Books</td><td>one paragraph</td><td class=r>{c['book']:,}</td><td class=r>{cd['med']['book']:.0f}</td><td class=r>{cd['max']['book']}</td></tr>
</tbody></table></div>
<p class="cap">Each chunk has the same ID in Czech and English (for example <code>art-0000</code>,
<code>vid-0544</code>, <code>AC-033-009</code> = book, section, paragraph). Every chunk fits easily within what the
model reads at once (about 8,000 tokens), so nothing is split further or truncated. Word counts are Czech.</p>
<h3>How long the chunks are</h3>
<div class="panel">{cd['hist']}</div>
<p class="cap"><strong>How to read it:</strong> share of each source's chunks by number of words. <strong>What it
shows:</strong> articles (median {cd['med']['article']:.0f} words) and video descriptions (median {cd['med']['video']:.0f})
range from a hashtag to several hundred words; book paragraphs are shorter and more even (median
{cd['med']['book']:.0f}). The shortest book chunks are lines of dialogue in the opening stories and headings.</p>
<h3>Articles and videos over time</h3>
<div class="panel">{cd['years']}</div>
<p class="cap"><strong>How to read it:</strong> article (blue) and video (green) chunks per year; one chunk is one
article or one video. <strong>What it shows:</strong> Urza's output moves from writing to video over the years; the
books have no year here.</p>
<h3>Czech and English</h3>
<p class="cap">An English chunk has a median of {cd['rmed']:.2f} × the words of its Czech original (90% between
{cd['rlo']:.2f} and {cd['rhi']:.2f}), as expected for a translation; {cd['same']} chunks are identical in both languages
(names, links, texts written in English). Only the Czech chunks are embedded; the presentation shows each chunk's
English twin, found by its ID.</p>"""))

    # ---- 2. projection compare
    R = {v: json.load(open(WORK / f"rates_{v}.json")) for v, _ in VARS}
    rows = []
    for v, lab in VARS:
        cells = "".join(f"<td class=r>{pct(R[v]['raw_1024d'][s])} → <b>{pct(R[v]['centred_1024d'][s])}</b></td>"
                        for s in ("article", "book", "video"))
        rows.append(f"<tr><td>{lab}</td>{cells}</tr>")
    drop = {v: {s: R[v]['raw_1024d'][s] - R[v]['centred_1024d'][s] for s in ("article", "book", "video")} for v, _ in VARS}
    cen_t = R["textonly"]["centred_1024d"]
    bm = {v: json.load(open(WORK / f"book_stats_{v}.json")) for v in ("nosource", "textonly")}
    panes = {v: frame(f"pc_{v}", "The two maps", h=760) for v, _ in VARS}
    S.append(("maps", "Raw vs source-centred maps", f"""
<p class="lead">Does the embedding place Urza's texts by <em>topic</em>, or by <em>kind of text</em> (book prose,
blog post, video description)? A coverage map needs the first.</p>
<h3>What was done</h3>
<ol>
 <li><strong>Three embedding variants</strong> of every chunk: <em>with source</em> (a context line in front naming
 the source, e.g. “Kniha Anarchokapitalismus › 10. Zdravotnictví”, “Článek „…“ (2017)”), <em>no source</em> (the same
 line without the word naming the source) and <em>text only</em> (nothing in front).</li>
 <li><strong>Source-centring:</strong> for each source, the average of its vectors is subtracted from each of them.
 What all books (or all videos) share, such as style, length and context line, is removed; what makes one chunk
 differ from another, its topic, stays.</li>
 <li><strong>Maps:</strong> both versions reduced to 2D with UMAP, which keeps neighbours together; the right map is
 rotated to line up with the left. Only who sits next to whom is meaningful, not distances between far clusters.</li>
 <li><strong>Measure:</strong> how often a chunk's 10 nearest neighbours come from the same source.</li>
</ol>
<h3>The maps: raw (left) and source-centred (right)</h3>
{switch("pc", VARS, panes)}
<p class="cap"><strong>How to read it:</strong> each dot is one chunk, coloured by source (blue articles, orange books,
green videos). Hover a dot to see the chunk; it is circled on the other map too. The boxes hide sources, the search
highlights titles (try <em>drug</em>, <em>school</em> or <em>bitcoin</em>). The switch swaps the embedding variant.</p>
<h3>Does centring help?</h3>
<div class="tw"><table><thead><tr><th>Variant</th><th class=r>Articles</th><th class=r>Books</th><th class=r>Videos</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="cap"><strong>How to read it:</strong> share of a chunk's 10 nearest neighbours (full 1,024-dim vectors) that
come from its own source, raw → <b>source-centred</b>. Lower means sources mix more.</p>
<h3>Conclusions</h3>
<ul>
 <li><strong>Centring reduces grouping by source, most where the source is most visible.</strong> Videos drop by
 {drop['withsource']['video'] * 100:.0f} points with the source line, {drop['nosource']['video'] * 100:.0f} without the
 source word and {drop['textonly']['video'] * 100:.0f} on text alone; books by {drop['withsource']['book'] * 100:.0f},
 {drop['nosource']['book'] * 100:.0f} and {drop['textonly']['book'] * 100:.0f}. Articles change little
 ({min(d['article'] for d in drop.values()) * -100:+.0f} to {max(d['article'] for d in drop.values()) * -100:+.0f}
 points): they already sit in the middle of the corpus.</li>
 <li><strong>The context line is the biggest source signal.</strong> Book chunks have
 {pct(R['withsource']['raw_1024d']['book'])} book neighbours with the line naming the source, and
 {pct(R['textonly']['raw_1024d']['book'])} on text alone. Text only is the honest basis for the maps.</li>
 <li><strong>What remains is genuine.</strong> After centring and without any context line, {pct(min(cen_t.values()))}–{pct(max(cen_t.values()))}
 of neighbours still share the source: the books, the blog and the videos really do discuss partly different subjects.
 Inside the books, chunks group by chapter on content alone ({pct(bm['textonly']['rates']['chapter'][0])} same-chapter
 neighbours against {pct(bm['textonly']['rates']['chapter'][1])} by chance): regions of the map correspond to ideas.</li>
</ul>"""))

    # ---- 3. PCA / ICA
    pp = read("all_pca_ica_textonly.html")
    psec = section(pp, "PCA: how much")
    svgs = re.findall(r"<svg.*?</svg>", psec, flags=re.S)
    ptab = psec[psec.find("<table"):psec.find("</table>") + 8]
    phead, prow = table_rows(ptab, 3)
    isec = section(pp, "ICA: themes")
    itab = isec[isec.find("<table"):isec.find("</table>") + 8]
    ihead, ith = table_rows(itab, 5, "th")
    _, ifo = table_rows(itab, 5, "fo")
    n_th = len(table_rows(itab, None, "th")[1])
    n_fo = len(table_rows(itab, None, "fo")[1])
    lv = {int(a): int(b) for a, b in re.findall(r"(\d+)%: (\d+) PCs", svgs[1])}
    evr1 = re.search(r"PC1 explains ([\d.]+%)", psec)
    chk = pc1_check()
    S.append(("pca", "Linear structure: PCA and ICA", f"""
<p class="lead">All {cd['n']:,} chunks (text-only embeddings) projected onto straight axes. Unlike UMAP, every axis
is one fixed direction in the 1,024-dimensional space, so it can be read and measured.</p>
<h3>What was done</h3>
<ol>
 <li><strong>Centring:</strong> the average vector of all chunks is subtracted (one average for all, not per
 source).</li>
 <li><strong>PCA</strong> finds the directions of largest variation, in order; <strong>ICA</strong> (20 components)
 finds directions whose scores are as independent and long-tailed as possible, so each tends to pick out one theme or
 one kind of text.</li>
 <li><strong>Reading an axis:</strong> the chunks at its two ends, their mix of sources, the video tags and book
 chapters over-represented among them. <strong>η²</strong> = share of an axis' variance explained by the source alone
 (0 = none, 1 = the axis only tells articles, books and videos apart).</li>
</ol>
<h3>How much does each direction explain?</h3>
<div class="panel">{svgs[0]}</div>
<p class="cap"><strong>How to read it:</strong> share of all variation captured by each of the first 20 principal
components. PC1 explains {evr1.group(1) if evr1 else 'about 6%'}; no direction dominates.</p>
<div class="panel" style="margin-top:12px">{svgs[1]}</div>
<p class="cap"><strong>How to read it:</strong> variance explained by the first k components together. Half the
variation needs {lv.get(50, '?')} components, 90% needs {lv.get(90, '?')}: the meaning of the texts is spread over
many directions, so no flat map can show all of it.</p>
<h3>The top three directions</h3>
<div class="tw"><table>{phead}<tbody>{''.join(prow)}</tbody></table></div>
<p class="cap"><strong>How to read it:</strong> each axis with its share of variance, source η² and the texts at its
two ends (colour = source). PC1 runs from book chapters to video conference panels and is partly the kind of text
(η² {chk['global']['eta1']:.2f}); PC2 and PC3 hardly depend on the source and are topical.</p>
<h3>PCA map</h3>
{frame("pca", "PCA map", h=760)}
<p class="cap"><strong>How to read it:</strong> every chunk placed by its scores on two principal components (choose
them above the map); hover a dot for its text. On PC1 × PC2 the sources are layered along PC1; from PC3 on they mix.</p>
<h3>ICA: themes and formats (top 5 of each)</h3>
<div class="tw"><table>{ihead}<tbody>{''.join(ith)}{''.join(ifo)}</tbody></table></div>
<p class="cap"><strong>How to read it:</strong> {n_th} of the 20 components are <em>themes</em> (their 30 top chunks
come from several sources), {n_fo} are <em>formats or series</em> (at least 80% from one source: a recurring kind of
post, a video series, or one book chapter); the five most focused of each are shown. <strong>Kurtosis</strong>
measures how long-tailed a component's scores are: a high value means a few chunks score extremely high and the rest
near zero, so the component is sharply focused on one subject; a low value means it is spread over many texts.</p>
<h3>ICA map</h3>
{frame("pca", "ICA map", h=760)}
<p class="cap"><strong>How to read it:</strong> chunks placed by two ICA components (selectable). Because themes
cut across sources, colours mix far more here than on the PCA map.</p>
<h3>Conclusions</h3>
<ul>
 <li><strong>No dominant direction:</strong> PC1 covers {evr1.group(1) if evr1 else 'about 6%'}, and half the
 variation needs {lv.get(50, '?')} components.</li>
 <li><strong>Is the strongest direction really the kind of text? Only because this PCA keeps the differences between
 sources.</strong> Here only one overall average was removed, so the gap between the average article, book and video
 stays in the data: {pct(chk['between'])} of all variation, and PC1 picks it up (η² {chk['global']['eta1']:.2f}).
 Checked separately: with each source's own average removed, as for the maps, PC1 explains
 {pct(chk['source']['evr1'])} and no axis separates the sources any more (η² {chk['source']['eta1']:.2f}). The “kind
 of text” is a shift between sources, not a structure within the topics.</li>
 <li><strong>Beneath it, the axes are topics:</strong> from PC2 on the source explains little, and ICA finds
 {n_th} themes shared by books, articles and videos (crypto, the Austrian school, the book campaign, drugs, …).</li>
 <li><strong>The rest are formats:</strong> {n_fo} ICA components each capture one recurring kind of post, series or
 book chapter, useful to spot template texts.</li>
</ul>"""))

    # ---- 4. book map
    trs = {v: tight_rows(v) for v in ("nosource", "textonly")}
    rho = {v: float(re.search(r"rank correlation of the two columns in the table below\): <strong>([\d.]+)",
                              read(f"book_map_{v}.html")).group(1)) for v in trs}   # as computed by the book map report
    lab_share = ((bm['nosource']['rates']['chapter'][0] - bm['textonly']['rates']['chapter'][0])
                 / (bm['nosource']['rates']['chapter'][0] - bm['nosource']['rates']['chapter'][1]))
    bvars = [("nosource", "Chapter path in front (no source)"), ("textonly", "Text only")]
    panes = {v: frame(f"bm_{v}", "The map|A second projection", h=1500) for v, _ in bvars}
    S.append(("books", "The book idea map", f"""
<p class="lead">Do regions of the map correspond to ideas? The books are the cleanest test: their chapters are topic
labels written by the author.</p>
<h3>What was done</h3>
<ol>
 <li><strong>Data:</strong> the {cd['cnt']['book']} book paragraphs with their embeddings, in two variants: with the
 chapter path in front of each chunk (<em>no source</em>: “Anarchokapitalismus › 10. Zdravotnictví”) and from the text
 alone (<em>text only</em>).</li>
 <li><strong>Chapter groups:</strong> each paragraph belongs to one of 56 groups: a chapter, or for the appendix of
 <em>Anarchokapitalismus</em> its topic or AMEN category; the aphorisms form one group.</li>
 <li><strong>Maps:</strong> UMAP and, as a check, t-SNE, on the book chunks only; each chapter's name sits at the
 average position of its chunks. A far-away island (the opening stories) is moved closer for display; measurements use
 the original layout.</li>
 <li><strong>Measures:</strong> who are each chunk's 10 nearest neighbours, and how tightly each chapter sits on the
 map (its spread relative to a random set of the same size: 1 = random, below 0.2 = compact).</li>
</ol>
<h3>UMAP and t-SNE maps</h3>
{switch("bm", bvars, panes)}
<p class="cap"><strong>How to read it:</strong> each dot is a paragraph, coloured by book (blue
<em>Anarchokapitalismus</em>, orange <em>Anarchoagorismus</em>). Click a chapter in the list, or on the map, to
highlight it on both maps; hover for the text. The switch swaps the variant.</p>
<h3>How tight is each chapter?</h3>
<div class="panel">{tight_svg(trs)}</div>
<p class="cap"><strong>How to read it:</strong> each dot is one chapter group (colour = book), placed by its spread
relative to a random set (log scale; left = tighter); the vertical mark is the median. Hover a dot for the chapter.
<strong>What it shows:</strong> with the chapter path in front, almost every chapter is a tight island (median
{np.median([r['u'] for r in trs['nosource']]):.2f}); on text alone chapters spread more (median
{np.median([r['u'] for r in trs['textonly']]):.2f}) but most are still far tighter than random. t-SNE spreads everything
a little more, keeping the same order.</p>
<h3>Conclusions</h3>
<ul>
 <li><strong>Chapters are ideas on the map.</strong> On text alone, {pct(bm['textonly']['rates']['chapter'][0])} of a
 paragraph's nearest neighbours are from its own chapter, against {pct(bm['textonly']['rates']['chapter'][1])} by
 chance, and {bm['textonly']['tight']} of 56 chapters form compact regions.</li>
 <li><strong>The chapter label inflates this:</strong> with the path in front, same-chapter neighbours rise to
 {pct(bm['nosource']['rates']['chapter'][0])} and {bm['nosource']['tight']} chapters are compact; the label accounts
 for about {pct(lab_share)} of the above-chance grouping. The text-only map is the honest one.</li>
 <li><strong>Not an artefact of UMAP:</strong> t-SNE agrees on which chapters are compact and which are spread out
 (rank correlation {rho['textonly']:.2f} on text only, {rho['nosource']:.2f} with the path).</li>
 <li><strong>Related chapters are neighbours</strong>, also across the two books: the two education chapters sit
 together, and the AMEN objections on economics next to the chapter on central planning they answer.</li>
</ul>"""))

    # ---- 5. trails
    tp = consts(read("book_trails_textonly.html"))
    N, E = tp["N"], tp["E"]
    tree = sorted([e for e in E if e["tree"]], key=lambda e: -e["lift"])
    cross = sum(1 for e in tree if N[e["a"]]["book"] != N[e["b"]]["book"])
    trow = "".join(
        f"<tr><td>{esc(N[e['a']]['name'])} – {esc(N[e['b']]['name'])}</td><td class=r>{e['lift']:.1f}</td>"
        f"<td class=r>{e['links']}</td><td class=m>{esc(e['bridges'][0][2][:110])}… / {esc(e['bridges'][0][4][:110])}…</td></tr>"
        for e in tree[:10])
    hubs = sorted(N, key=lambda n: -n["deg"])[:4]
    S.append(("trails", "Trails between book topics", f"""
<p class="lead">On the book map, sections sit next to related sections; following those neighbours reads like a line
of argument. Here the neighbourhoods become a network of links between the {len(N)} book sections.</p>
<h3>What was done</h3>
<ol>
 <li><strong>Links from the text:</strong> for every book paragraph (text-only embedding), its 10 most similar
 paragraphs are taken; each neighbour from a different section counts as one link between the two sections.</li>
 <li><strong>Lift</strong> measures how strong a link is: the number of links between two sections divided by the
 number expected if neighbours were picked at random. Big sections would collect many links just by size; lift corrects
 for that. Lift 1 = chance, lift 10 = ten times more often than chance. Kept are links with lift ≥ 2 and at least 3
 neighbour pairs: {len(E)} links.</li>
 <li><strong>Paths:</strong> between two sections, the chain of strongest links (each step costs 1 / lift); for every
 step the most similar pair of passages across it is shown, the actual bridge between the topics.</li>
</ol>
<h3>The trail map</h3>
{frame("trails", "The trail map", h=1250)}
<p class="cap"><strong>How to read it:</strong> circles are sections (size = number of paragraphs, colour = book),
lines are links (thicker = higher lift). Pick two sections, or a suggested pair, to see the trail between them and
the passages that bridge each step.</p>
<h3>The backbone</h3>
<p class="cap" style="margin-top:0">The <strong>backbone</strong> is the maximum spanning tree of the network: the
{len(tree)} strongest links that still connect every section, without loops. It removes redundant links and keeps the
main lines of argument, so it is the skeleton the trails follow; {cross} of its links join the two books, the places
where both books discuss the same thing. The ten backbone links with the highest lift:</p>
<div class="tw"><table><thead><tr><th>Link</th><th class=r>Lift</th><th class=r>Neighbour pairs</th><th>Strongest bridge (the two passages)</th></tr></thead><tbody>{trow}</tbody></table></div>
<h3>Conclusions</h3>
<ul>
 <li><strong>The topics form connected trails</strong> that follow the books' arguments, for example
 2. Education → 3. Indoctrination → 9. Democracy, or Drugs → smugglers and dealers → Liberation Travel → 22. Love.</li>
 <li><strong>Broad sections are hubs:</strong> {', '.join(esc(n['name']) for n in hubs)} connect the most topics.</li>
 <li><strong>Only the opening stories stand apart</strong>: as narrative dialogue, they have no strong link to any
 section.</li>
</ul>"""))

    # ---- 6. topics
    tdp = read("topic_distribution_textonly.html")
    tk = kpis(tdp)
    D = consts(tdp)["D"]
    A_ = sum(d["a"] for d in D)
    V_ = sum(d["v"] for d in D)
    ak_a = sum(d["a"] for d in D if d["book"] == "Anarchokapitalismus") / A_
    ag_v = sum(d["v"] for d in D if d["book"] == "Anarchoagorismus") / V_
    top = sorted([d for d in D if not d["framing"]], key=lambda d: -d["share"])[:3]
    S.append(("topics", "Topic distribution", f"""
<p class="lead">The books are Urza's most systematic statement. Here they define the concepts, and the articles and
videos are sorted into them: which of his book topics does he return to?</p>
<h3>What was done</h3>
<ol>
 <li><strong>Concept space from the books:</strong> PCA on the book paragraphs only, keeping the directions that carry
 more than a random split would; each source's average is removed first so style doesn't decide.</li>
 <li><strong>Concept centres:</strong> the average of each book section's paragraphs. Every article and video chunk is
 assigned to the most similar centre (announcements left out).</li>
</ol>
<div class="kpis">
 <div class="kpi"><b>{esc(tk[0][0])}</b><span>of book paragraphs land back in their own section when taken out and
 re-assigned (chance 1.5%): the sections are distinct enough to sort texts into.</span></div>
 <div class="kpi"><b>{esc(tk[1][0])}</b><span>topical book sections that at least some articles or videos are sorted
 into: no book topic is left untouched.</span></div>
 <div class="kpi"><b>{esc(tk[2][0])}</b><span>median similarity of articles / videos to their section centre, against
 0.42 for the books' own paragraphs: they fit the book topics, but more loosely, since they also discuss current
 events the books don't.</span></div>
</div>
<h3>Distribution over the books' concepts</h3>
{frame("topic", "Distribution over", h=900)}
<p class="cap"><strong>How to read it:</strong> each bar is a book section: the share of article and video chunks
assigned to it, compared with the section's share of the books. Hover for example texts.</p>
<h3>Conclusions</h3>
<ul>
 <li><strong>Every book topic recurs</strong>, most of all {', '.join(esc(d['g']) for d in top)}.</li>
 <li><strong>The two books split the output:</strong> articles lean to <em>Anarchokapitalismus</em> topics
 ({pct(ak_a)}), videos to <em>Anarchoagorismus</em> ({pct(ag_v)}), in line with the shift over time towards the
 second book's themes.</li>
 <li><strong>The broad picture is robust; the exact ranking is not:</strong> it changes with the size of the concept
 space.</li>
</ul>"""))

    # ---- 7. politics
    psp = read("political_space_textonly.html")
    pk = kpis(psp)
    pc = consts(psp)
    AREAS = pc["AREAS"]
    by_vol = sorted(AREAS, key=lambda a: -a["vol"])[:3]
    by_dep = sorted(AREAS, key=lambda a: a["depth"])[:3]
    SRCN = ["article", "book", "video"]

    def qtab(rows):
        return "".join(
            f"<tr><td>{esc(r[2])}<div class=m>{esc(r[1])}</div></td><td class=r>{r[3]:.2f}</td><td class=r>{r[4]}</td>"
            f"<td><span class=sw style='background:var(--s{['article', 'book', 'video'].index(SRCN[r[5][0]])})'></span>"
            f"{esc(r[5][1])}{' (' + str(r[5][3]) + ')' if r[5][3] else ''}</td></tr>" for r in rows[:5])
    S.append(("politics", "Urza in the political space", f"""
<p class="lead">Everything so far measured Urza against himself. Here the reference is independent of him: a map of
politics in general, and how much of it he addresses.</p>
<div class="kpis">{''.join(f'<div class="kpi"><b>{esc(a)}</b><span>{esc(b)}</span></div>' for a, b in pk)}</div>
<p class="cap" style="margin-top:0">By volume he writes most about {', '.join(esc(a['name']) for a in by_vol)}; his
answers are least direct in {', '.join(esc(a['name']) for a in by_dep)}.</p>
<h3>What was done</h3>
<ul>
 <li><strong>The political space:</strong> 326 items built on the Comparative Agendas Project classification: 22 policy
 areas, their subtopics, concrete questions as Czech voters meet them, and concepts of political philosophy.</li>
 <li><strong>Same model</strong> as Urza's texts (bge-m3); topics and concepts embedded as “name: description”,
 questions as written.</li>
 <li><strong>Depth</strong> of an item: the similarity of Urza's single best-matching passage, as a search would rank it
 (≥ 0.55 relevant, ≥ 0.60 squarely on topic).</li>
 <li><strong>Volume</strong> of an area: how many of his chunks are closest to its subtopics (each source's average
 removed first; announcements left out).</li>
 <li><strong>Map:</strong> the political items laid out with UMAP (each item type's average removed so items group by
 subject, not wording); Urza's chunks are then added without changing it, in three ways (switch above the map).</li>
</ul>
<h3>The map of the political space</h3>
{frame("pol", "The map of the political space", h=1150)}
<p class="cap"><strong>How to read it:</strong> large dots are subtopics, small dots questions, squares concepts;
colour = depth of Urza's best passage (darker = covered more squarely). Small dots behind are Urza's chunks. Hover an
item for its best passage; the placement switch changes how his chunks are placed, not the political map.</p>
<div class="two">
 <div><h3>Best-covered questions</h3><div class="tw"><table><thead><tr><th>Question</th><th class=r>Depth</th><th class=r>Texts</th><th>Best passage</th></tr></thead><tbody>{qtab(pc['QTOP'])}</tbody></table></div></div>
 <div><h3>Blind spots</h3><div class="tw"><table><thead><tr><th>Question</th><th class=r>Depth</th><th class=r>Texts</th><th>Nearest passage</th></tr></thead><tbody>{qtab(pc['QBOT'])}</tbody></table></div></div>
</div>
<p class="cap"><strong>How to read it:</strong> depth = similarity of his best passage; texts = how many different
articles, book sections or videos reach 0.55. The best-covered questions are about what the state provides and
regulates (roads, wages, pensions, housing, school), where Urza has written directly; the blind spots are conventional
policy questions (energy, agriculture, foreign policy) that he answers, if at all, only through general arguments
about the state.</p>"""))

    nav = "".join(f'<a href="#{k}">{esc(t if k != "intro" else "Overview")}</a>' for k, t, _ in S)
    body = "".join(f'<section class="slide" id="{k}"><h2>{esc(t)}</h2>{c}</section>' for k, t, c in S)
    page = TEMPLATE.replace("__NAV__", nav).replace("__BODY__", body).replace(
        "__PAGES__", json.dumps(pages, ensure_ascii=False).replace("</", "<\\/"))
    out = REP / "presentation.html"
    out.write_text(page, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mapping Urza's ideas</title>
<style>
:root{--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--s0:#2a78d6;--s1:#eb6834;--s2:#1baf7a;--b0:#2a78d6;--b1:#eb6834;--acc:#2a78d6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835;--s0:#3987e5;--s1:#d95926;--s2:#199e70;--b0:#3987e5;--b1:#d95926;--acc:#5598e7}}
:root[data-theme="dark"]{--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835;--s0:#3987e5;--s1:#d95926;--s2:#199e70;--b0:#3987e5;--b1:#d95926;--acc:#5598e7}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:56px}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
nav{position:sticky;top:0;z-index:10;background:var(--bg);border-bottom:1px solid var(--line);display:flex;gap:2px;overflow-x:auto;padding:8px 16px;scrollbar-width:thin}
nav a{color:var(--ink2);text-decoration:none;padding:5px 10px;border-radius:7px;white-space:nowrap;font-size:13.5px}
nav a.on{background:var(--line);color:var(--ink);font-weight:600}
.slide{max-width:1180px;margin:0 auto;padding:40px 16px 56px;border-bottom:1px solid var(--line);min-height:calc(100vh - 48px)}
.slide h2{font-size:30px;line-height:1.2;margin:0 0 10px;letter-spacing:-.01em}
.slide h3{font-size:18px;margin:30px 0 8px}
p.lead{font-size:18px;color:var(--ink2);max-width:860px;margin:0 0 14px}
p.cap{color:var(--ink2);font-size:13.5px;max-width:940px;margin:8px 2px 0}
ol,ul{max-width:940px;padding-left:22px}li{margin:6px 0}
code{background:var(--line);padding:1px 5px;border-radius:4px;font-size:12.5px}a{color:var(--acc)}
.kpis{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 16px;flex:1 1 200px;max-width:360px}
.kpi b{display:block;font-size:24px;line-height:1.2}.kpi span{color:var(--ink2);font-size:13px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
table{border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;font-size:13px;width:100%}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{color:var(--ink2);font-weight:600}
td.r,th.r{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.tw{overflow-x:auto}.m{color:var(--ink2);font-size:12px}
.sw{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:6px;vertical-align:-1px}.sw.sm{width:8px;height:8px;margin-right:5px}
.mix{display:flex;gap:2px;width:120px;height:8px;border-radius:4px;overflow:hidden}.mixt{color:var(--ink2);font-size:11.5px;margin-top:3px;white-space:nowrap}
.tag{font-size:11px;border-radius:10px;padding:1px 7px;border:1px solid var(--line);color:var(--ink2);white-space:nowrap}.tag.th{color:var(--ink)}
svg .grid{stroke:var(--line);stroke-width:1}svg .base{stroke:var(--ink2);stroke-width:1}svg .ref{stroke:var(--ink2);stroke-width:1;stroke-dasharray:3 3}
svg .med{stroke:var(--ink);stroke-width:2}svg .ax{fill:var(--ink2);font-size:10px}svg .lab{fill:var(--ink);font-size:11.5px;font-weight:600}
iframe.emb{width:100%;border:0;display:block;background:transparent}
.vswitch{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:6px 0 10px;font-size:13.5px;color:var(--ink2)}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;flex-wrap:wrap}
.seg button{background:var(--card);color:var(--ink2);border:0;padding:6px 14px;cursor:pointer;font:inherit}
.seg button.on{background:var(--line);color:var(--ink);font-weight:600}
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}
ol.agenda{font-size:17px;columns:2;max-width:760px}ol.agenda li{break-inside:avoid}
@media (max-width:860px){.two{grid-template-columns:1fr}ol.agenda{columns:1}.slide h2{font-size:24px}}
</style></head><body>
<nav>__NAV__</nav>
__BODY__
<script>
const PAGES=__PAGES__;
function fit(f){try{const d=f.contentDocument,w=d&&d.querySelector(".wrap");if(w)f.style.height=Math.max(80,Math.ceil(w.getBoundingClientRect().height)+4)+"px"}catch(e){}}
function load(f){if(f.dataset.loaded)return;f.dataset.loaded=1;
 f.addEventListener("load",()=>{fit(f);try{new ResizeObserver(()=>fit(f)).observe(f.contentDocument.querySelector(".wrap"))}catch(e){}setTimeout(()=>fit(f),400)});
 f.srcdoc=PAGES[f.dataset.page]}
const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting&&e.target.offsetParent!==null){load(e.target);io.unobserve(e.target)}}),{rootMargin:"900px"});
document.querySelectorAll("iframe.emb").forEach(f=>io.observe(f));
/* fallback: load frames near the screen on scroll, and the rest one by one once the page is idle */
function near(){document.querySelectorAll("iframe.emb:not([data-loaded])").forEach(f=>{if(f.offsetParent===null)return;
 const r=f.getBoundingClientRect();if(r.top<innerHeight+900&&r.bottom>-900)load(f)})}
addEventListener("scroll",near,{passive:true});addEventListener("load",()=>{near();
 const rest=[...document.querySelectorAll("iframe.emb")].filter(f=>f.offsetParent!==null);let i=0;
 (function next(){while(i<rest.length&&rest[i].dataset.loaded)i++;if(i<rest.length){load(rest[i++]);setTimeout(next,1500)}})()});
document.querySelectorAll(".vswitch").forEach(sw=>{const g=sw.dataset.group;
 sw.querySelectorAll("button").forEach(b=>b.onclick=()=>{sw.querySelectorAll("button").forEach(x=>x.classList.toggle("on",x===b));
  document.querySelectorAll(`.vpane[data-group="${g}"]`).forEach(p=>{const on=p.dataset.v===b.dataset.v;p.hidden=!on;
   if(on)p.querySelectorAll("iframe.emb").forEach(f=>{if(!f.dataset.loaded)load(f);else{try{f.contentWindow.dispatchEvent(new Event("resize"))}catch(e){}fit(f)}})})})});
const slides=[...document.querySelectorAll(".slide")],links=[...document.querySelectorAll("nav a")];
let curId="";
function markNav(){let cur=slides[0];for(const s of slides)if(s.getBoundingClientRect().top<=90)cur=s;if(cur.id===curId)return;curId=cur.id;
 links.forEach(a=>a.classList.toggle("on",a.getAttribute("href")==="#"+cur.id));
 const a=links.find(a=>a.classList.contains("on"));if(a)a.parentNode.scrollTo({left:a.offsetLeft-40,behavior:"smooth"})}
addEventListener("scroll",markNav,{passive:true});markNav();
addEventListener("keydown",e=>{if(e.target.closest&&e.target.closest("input,select,textarea"))return;
 const i=slides.findIndex(s=>s.getBoundingClientRect().top>60);const cur=i<0?slides.length-1:Math.max(0,i-1);
 if(e.key==="ArrowRight"||e.key==="PageDown"){e.preventDefault();(slides[cur+1]||slides[cur]).scrollIntoView()}
 if(e.key==="ArrowLeft"||e.key==="PageUp"){e.preventDefault();const t=slides[cur].getBoundingClientRect().top<-5?slides[cur]:slides[cur-1];(t||slides[0]).scrollIntoView()}});
</script></body></html>"""


if __name__ == "__main__":
    build()
