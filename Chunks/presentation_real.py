"""One self-contained HTML presentation of the real_data reports, both embedding models side by side.

    python Chunks/presentation_real.py      # from the repo root, after Chunks/build_real_reports.py

Reads the reports in RealReports/<model>/ (their interactive maps are embedded as isolated frames showing only the
needed section), the real_data chunks and the two models' vectors for the few numbers computed here.
Writes RealReports/presentation.html. Every section has a model switch (bge-m3 / EmbeddingGemma); each model's maps,
tables and conclusions are in its own pane. (The FinalReports presentation is Chunks/presentation.py.)
"""
import html as H
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import presentation as P  # noqa: E402  (shared helpers and page template)

REP = ROOT / "RealReports"
REAL = HERE / "output_real"
MODELS = [("bge_m3", "bge-m3", 1024), ("embeddinggemma_2", "EmbeddingGemma", 768)]
LABEL = {k: l for k, l, _ in MODELS}
DIM = {k: d for k, _, d in MODELS}
SRCS = ("article", "book", "video")
esc, pct, consts, section, kpis, table_rows, frame = P.esc, P.pct, P.consts, P.section, P.kpis, P.table_rows, P.frame


def read(model, name):
    return (REP / model / name).read_text(encoding="utf-8")


def frame_page(model, name):
    P.REP = REP / model
    return P.frame_page(name)


def switch(group, panes):
    btns = "".join(f'<button data-v="{k}"{" class=on" if i == 0 else ""}>{esc(l)}</button>' for i, (k, l, _) in enumerate(MODELS))
    ps = "".join(f'<div class="vpane" data-group="{group}" data-v="{k}"{"" if i == 0 else " hidden"}>{panes[k]}</div>'
                 for i, (k, _, _) in enumerate(MODELS))
    return f'<div class="vswitch" data-group="{group}"><span>Embedding model</span><div class="seg">{btns}</div></div>{ps}'


def stats(model, name):
    return json.loads((REAL / model / "projection" / f"{name}_textonly.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- numbers computed here
def chunk_data():
    from report import hist_svg, years_svg
    rows = pq.read_table(REAL / "chunks.parquet", columns=["source", "words", "year"]).to_pylist()
    for r in rows:
        r["n_words"] = r["words"]
    ch = json.loads((ROOT / "real_data" / "chunked.json").read_text(encoding="utf-8"))
    ratio = np.array([len((c["text_en"] or "").split()) / max(len(c["text_cs"].split()), 1) for c in ch])
    L = lambda p: json.load(open(p, encoding="utf-8"))
    raw = L(ROOT / "jsons" / "urza_articles.json")
    yl = {v["webpage_url"] for v in L(ROOT / "jsons" / "channel_metadata.json")}
    dup = sum(1 for a in raw if (a.get("video") or "") in yl and not (a.get("text") or "").strip())
    front = ("O knize", "O autorovi", "Věnování", "Poděkování", "Obsah")
    fm = {b: sum(len(x["text"].split()) for x in L(ROOT / "jsons" / f"{b}.json") if x["name"] in front)
          for b in ("ankap", "anag")}
    # video boilerplate: the earlier corpus kept both the raw and the cleaned description of the same videos
    old = pq.read_table(HERE / "output" / "chunks_cs.parquet").to_pylist() if (HERE / "output" / "chunks_cs.parquet").exists() else []
    return {"n": len(rows), "cnt": Counter(r["source"] for r in rows), "raw": len(raw), "dup": dup, "fm": fm,
            "med": {s: float(np.median([r["n_words"] for r in rows if r["source"] == s])) for s in SRCS},
            "max": {s: max(r["n_words"] for r in rows if r["source"] == s) for s in SRCS},
            "rmed": float(np.median(ratio)), "rlo": float(np.percentile(ratio, 5)), "rhi": float(np.percentile(ratio, 95)),
            "same": sum(1 for c in ch if c["text_cs"] == c["text_en"]), "hist": hist_svg(rows), "years": years_svg(rows),
            "n_old": len(old)}


def pc1_check(model):
    """PC1's source effect with only the overall mean removed vs each source's own mean removed."""
    from sklearn.decomposition import PCA
    src = np.array(pq.read_table(REAL / "chunks.parquet", columns=["source"]).column(0).to_pylist())
    E = np.load(REAL / model / "results_textonly" / "embeddings.npy")

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


def tight_rows(model):
    t = section(read(model, "book_map.html"), "How tight is each chapter")
    return [{"book": int(m.group(1)), "name": H.unescape(m.group(2)), "n": int(m.group(3)), "u": float(m.group(4)),
             "t": float(m.group(5))}
            for m in re.finditer(r'<tr data-g="\d+"><td><span class="sw" style="background:var\(--b(\d)\)"></span>(.*?)</td>'
                                 r'<td>(\d+)</td><td>([\d.]+)</td><td>([\d.]+)</td><td>(\d+)%</td></tr>', t)]


def tight_svg(data):
    """Strip plot: each chapter's spread vs random (log scale), one row per model, UMAP and t-SNE side by side."""
    W, rowh, top, pl, pr = 980, 62, 30, 120, 14
    pw = (W - pl - pr - 30) / 2
    lo, hi = np.log10(0.01), np.log10(2)
    X = lambda v, x0: x0 + (np.log10(min(max(v, 0.01), 2)) - lo) / (hi - lo) * pw
    el, nrow = [], len(data)
    for pi, (key, name) in enumerate((("u", "UMAP"), ("t", "t-SNE"))):
        x0 = pl + pi * (pw + 30)
        el.append(f'<text x="{x0}" y="14" class="lab">{name}</text>')
        for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2):
            x = X(t, x0)
            el.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top - 6}" y2="{top + rowh * nrow}" class="grid"/>'
                      f'<text x="{x:.1f}" y="{top + rowh * nrow + 14}" text-anchor="middle" class="ax">{t:g}</text>')
        for t, txt in ((0.2, "compact"), (1, "random")):
            x = X(t, x0)
            el.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top - 6}" y2="{top + rowh * nrow}" class="ref"/>'
                      f'<text x="{x + 3:.1f}" y="{top - 8}" class="ax">{txt}</text>')
        for ri, (m, rows) in enumerate(data.items()):
            yc = top + ri * rowh + rowh / 2
            if pi == 0:
                el.append(f'<text x="{pl - 10}" y="{yc + 4:.1f}" text-anchor="end" class="lab">{LABEL[m]}</text>')
            bins = Counter()
            for r in sorted(rows, key=lambda r: r[key]):
                x = X(r[key], x0)
                b = int(x // 7)
                k = bins[b]
                bins[b] += 1
                y = yc + ((k + 1) // 2) * 7 * (1 if k % 2 else -1)
                el.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="var(--b{r["book"]})" fill-opacity=".85">'
                          f'<title>{esc(r["name"])}: {r[key]:.2f} ({LABEL[m]}, {name})</title></circle>')
            med = float(np.median([r[key] for r in rows]))
            xm = X(med, x0)
            el.append(f'<line x1="{xm:.1f}" x2="{xm:.1f}" y1="{yc - 22:.1f}" y2="{yc + 22:.1f}" class="med"/>'
                      f'<text x="{xm:.1f}" y="{yc - 25:.1f}" text-anchor="middle" class="ax">median {med:.2f}</text>')
            el.append(f'<line x1="{x0}" x2="{x0 + pw}" y1="{top + (ri + 1) * rowh:.1f}" y2="{top + (ri + 1) * rowh:.1f}" class="grid"/>')
    hgt = top + rowh * nrow + 22
    return (f'<svg viewBox="0 0 {W} {hgt}" role="img" aria-label="Spread of each chapter relative to random, per model" '
            f'style="width:100%;height:auto">{"".join(el)}</svg>')


def transcripts_info():
    """Counts for the transcripts section (its maps are the interactive RealReports/projection_all_sources.html)."""
    if not (REP / "projection_all_sources.html").exists() or not (REAL / "all_sources_cache.npz").exists():
        print("no projection_all_sources.html - run Chunks/project_all_sources.py; transcripts section left out")
        return None
    lab = np.load(REAL / "all_sources_cache.npz")["lab"]
    ch = json.loads((ROOT / "real_data" / "trans_chunks.json").read_text(encoding="utf-8"))
    n_tr = int((lab == 3).sum())
    return {"n": n_tr, "n_vid": len({c["src"] for c in ch}), "share": n_tr / len(lab), "ratio": n_tr / (len(lab) - n_tr)}


# ---------------------------------------------------------------- sections
def build():
    sys.stdout.reconfigure(encoding="utf-8")
    pages, S = {}, []
    have_pol = {m: (REP / m / "political_space.html").exists() for m, _, _ in MODELS}
    for m, _, _ in MODELS:
        for key, f in (("pc", "projection_compare.html"), ("bm", "book_map.html"), ("pca", "all_pca_ica.html"),
                       ("trails", "book_trails.html"), ("topic", "topic_distribution.html"), ("pol", "political_space.html")):
            if (REP / m / f).exists():
                pages[f"{key}_{m}"] = frame_page(m, f)
    cd = chunk_data()
    c = cd["cnt"]

    # ---- 0. title
    S.append(("intro", "UrzaLLM: mapping Urza's ideas", f"""
<p class="lead">Urza's articles, video descriptions and two books, embedded with two multilingual language models, to
see what he writes about, how his ideas connect, and how much of politics in general he covers.</p>
<div class="kpis">
 <div class="kpi"><b>{cd['n']:,}</b><span>chunks of text, in Czech and English</span></div>
 <div class="kpi"><b>{c['article']:,} / {c['video']:,} / {c['book']:,}</b><span>articles / videos / book paragraphs</span></div>
 <div class="kpi"><b>2 models</b><span>bge-m3 (1,024 dimensions) and EmbeddingGemma (768), both from the Czech text of each chunk</span></div>
</div>
<ol class="agenda">
 <li><a href="#data">Data and chunking</a></li><li><a href="#maps">Raw vs source-centred maps</a></li>
 <li><a href="#transcripts">All sources with the video transcripts</a></li>
 <li><a href="#pca">Linear structure: PCA and ICA</a></li><li><a href="#books">The book idea map</a></li>
 <li><a href="#trails">Trails between book topics</a></li><li><a href="#topics">Topic distribution</a></li>
 <li><a href="#politics">Urza in the political space</a></li>
</ol>
<p class="cap">All texts are shown in English translation; every computation uses the Czech originals. Use the menu
or the ← → keys to move between sections. Interactive maps: hover for details; the switch above each map swaps the
embedding model.</p>"""))

    # ---- 1. data
    S.append(("data", "Data and chunking", f"""
<div class="kpis">
 <div class="kpi"><b>{cd['n']:,}</b><span>chunks per language</span></div>
 <div class="kpi"><b>{c['article']:,} / {c['video']:,} / {c['book']:,}</b><span>article / video / book chunks</span></div>
 <div class="kpi"><b>×{cd['rmed']:.2f}</b><span>English words per Czech word, median over chunks (translations run longer)</span></div>
</div>
<h3>From raw data to chunks</h3>
<ol>
 <li><strong>Articles and videos without duplicates.</strong> Of the {cd['raw']:,} posts on Urza's blog, {cd['dup']:,} only
 announce a video and have no text of their own; they would duplicate the video, which is represented by its own
 description, so they are left out. Kept are the {c['article']:,} posts with text, and all {c['video']:,} videos of the
 YouTube channel.</li>
 <li><strong>Books cleaned.</strong> Both books, <em>Anarchokapitalismus</em> and <em>Anarchoagorismus</em>, were parsed into
 sections; their front matter was removed: about the book, about the author, dedication, acknowledgements and contents
 ({cd['fm']['ankap']:,} + {cd['fm']['anag']:,} words). Prefaces, epilogues and the appendices are content and stay.</li>
 <li><strong>Video descriptions cleaned.</strong> Lines that recur in many descriptions (donation requests, crypto
 addresses, bank details, Urza's signature, separators) were removed from every description, about a third of all
 video words. Guest introductions stay, since they say who the video is with.</li>
 <li><strong>Translated.</strong> Every chunk also exists in English, aligned item by item.</li>
</ol>
<h3>Chunking</h3>
<div class="tw"><table><thead><tr><th>Source</th><th>One chunk is</th><th class=r>Chunks</th><th class=r>Median words</th><th class=r>Longest</th></tr></thead><tbody>
<tr><td><span class=sw style="background:var(--s0)"></span>Articles</td><td>the whole article text</td><td class=r>{c['article']:,}</td><td class=r>{cd['med']['article']:.0f}</td><td class=r>{cd['max']['article']:,}</td></tr>
<tr><td><span class=sw style="background:var(--s2)"></span>Videos</td><td>the whole (cleaned) description</td><td class=r>{c['video']:,}</td><td class=r>{cd['med']['video']:.0f}</td><td class=r>{cd['max']['video']:,}</td></tr>
<tr><td><span class=sw style="background:var(--s1)"></span>Books</td><td>one paragraph</td><td class=r>{c['book']:,}</td><td class=r>{cd['med']['book']:.0f}</td><td class=r>{cd['max']['book']:,}</td></tr>
</tbody></table></div>
<p class="cap">Every chunk fits easily within what both models read at once (bge-m3 about 8,000 tokens, EmbeddingGemma
about 2,000; the longest chunk is about 1,100), so nothing is split further or truncated. Word counts are Czech.</p>
<h3>How long the chunks are</h3>
<div class="panel">{cd['hist']}</div>
<p class="cap"><strong>How to read it:</strong> share of each source's chunks by number of words. <strong>What it
shows:</strong> articles (median {cd['med']['article']:.0f} words) and video descriptions (median {cd['med']['video']:.0f})
range from a line to several hundred words; book paragraphs are shorter and more even (median
{cd['med']['book']:.0f}).</p>
<h3>Articles and videos over time</h3>
<div class="panel">{cd['years']}</div>
<p class="cap"><strong>How to read it:</strong> article (blue) and video (green) chunks per year; one chunk is one
article or one video. <strong>What it shows:</strong> Urza's output moves from writing to video over the years; the
books have no year here.</p>
<h3>Czech and English</h3>
<p class="cap">An English chunk has a median of {cd['rmed']:.2f} × the words of its Czech original (90% between
{cd['rlo']:.2f} and {cd['rhi']:.2f}), as expected for a translation; {cd['same']} chunks are identical in both languages
(names, links, texts written in English). Only the Czech chunks are embedded; the presentation shows each chunk's
English twin.</p>"""))

    # ---- 2. projection compare
    R = {m: stats(m, "rates") for m, _, _ in MODELS}
    bm = {m: stats(m, "book_stats") for m, _, _ in MODELS}
    rows = "".join(f"<tr><td>{LABEL[m]}</td>" + "".join(
        f"<td class=r>{pct(R[m]['raw_1024d'][s])} → <b>{pct(R[m]['centred_1024d'][s])}</b></td>" for s in SRCS) + "</tr>"
        for m, _, _ in MODELS)
    rnd = R[MODELS[0][0]]["random"]
    a, b = MODELS[0][0], MODELS[1][0]
    gap = lambda k: {s: R[b][k][s] - R[a][k][s] for s in SRCS}
    more = [s for s in SRCS if gap("raw_1024d")[s] > 0.02]
    vs = ", ".join("%s %s vs %s" % (P.SRC_LABEL[s].lower(), pct(R[b]["raw_1024d"][s]), pct(R[a]["raw_1024d"][s])) for s in more)
    cmp_txt = ((f"With raw vectors {LABEL[b]} keeps {' and '.join(P.SRC_LABEL[s].lower() for s in more)} more apart than "
                f"{LABEL[a]} ({vs}); " if more else "") + f"after centring the largest gap between the models is "
               f"{max(abs(v) for v in gap('centred_1024d').values()) * 100:.0f} points.")

    def maps_pane(m):
        r, cn = R[m]["raw_1024d"], R[m]["centred_1024d"]
        d = {s: r[s] - cn[s] for s in SRCS}
        big = max(SRCS, key=lambda s: d[s])
        return f"""{frame(f"pc_{m}", "The two maps", h=760)}
<h3>Conclusions ({LABEL[m]})</h3>
<ul>
 <li><strong>Centring reduces grouping by source where it is strongest, most for {P.SRC_LABEL[big].lower()}.</strong>
 Same-source neighbours, raw → centred: {'; '.join(f"{P.SRC_LABEL[s].lower()} {pct(r[s])} → {pct(cn[s])}" for s in sorted(SRCS, key=lambda s: -d[s]))}{
 '. ' + ' and '.join(P.SRC_LABEL[s] for s in SRCS if d[s] < 0.02).capitalize() + ' barely change' if any(d[s] < 0.02 for s in SRCS) else ''}.</li>
 <li><strong>What remains is genuine.</strong> After centring, {pct(min(cn.values()))}–{pct(max(cn.values()))} of
 neighbours still share the source (chance {pct(min(rnd.values()))}–{pct(max(rnd.values()))}): the books, the blog and
 the videos really do discuss partly different subjects.</li>
 <li><strong>Inside the books, chunks group by chapter</strong> on content alone ({pct(bm[m]['rates']['chapter'][0])}
 same-chapter neighbours against {pct(bm[m]['rates']['chapter'][1])} by chance): regions of the map correspond to ideas.</li>
</ul>"""
    S.append(("maps", "Raw vs source-centred maps", (f"""
<p class="lead">Does the embedding place Urza's texts by <em>topic</em>, or by <em>kind of text</em> (book prose,
blog post, video description)? A coverage map needs the first.</p>
<h3>What was done</h3>
<ol>
 <li><strong>Embedding:</strong> every chunk's own text, with nothing added in front, embedded by each model.</li>
 <li><strong>Source-centring:</strong> for each source, the average of its vectors is subtracted from each of them.
 What all books (or all videos) share, such as style and length, is removed; what makes one chunk differ from
 another, its topic, stays.</li>
 <li><strong>Maps:</strong> both versions reduced to 2D with UMAP, which keeps neighbours together; the right map is
 rotated to line up with the left. Only who sits next to whom is meaningful, not distances between far clusters.</li>
 <li><strong>Measure:</strong> how often a chunk's 10 nearest neighbours (full vectors) come from the same source.</li>
</ol>
<h3>The maps: raw (left) and source-centred (right)</h3>
{switch("pc", {m: maps_pane(m) for m, _, _ in MODELS})}
<p class="cap"><strong>How to read it:</strong> each dot is one chunk, coloured by source (blue articles, orange books,
green videos). Hover a dot to see the chunk; it is circled on the other map too. The boxes hide sources, the search
highlights titles (try <em>drug</em>, <em>school</em> or <em>bitcoin</em>).</p>
<h3>Does centring help? Both models</h3>
<div class="tw"><table><thead><tr><th>Model</th><th class=r>Articles</th><th class=r>Books</th><th class=r>Videos</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="cap"><strong>How to read it:</strong> share of a chunk's 10 nearest neighbours (full vectors) that come from
its own source, raw → <b>source-centred</b>; by chance {' / '.join(pct(rnd[s]) for s in SRCS)}. Lower means sources mix
more. __CMP__</p>""").replace("__CMP__", cmp_txt)))

    # ---- 2b. all sources with the video transcripts (a mention; the full interactive page is linked)
    tr = transcripts_info()
    if tr:
        P.REP = REP
        pages["allsrc"] = P.frame_page("projection_all_sources.html")
        S.append(("transcripts", "All sources with the video transcripts", f"""
<p class="lead">Beyond the video descriptions: the transcripts of what is actually said in {tr['n_vid']} videos,
{tr['n']:,} passages of up to 256 words, mapped together with everything above.</p>
{frame("allsrc", "The maps", h=900)}
<p class="cap"><strong>How to read it:</strong> left bge-m3, right EmbeddingGemma (turned to line up with the left);
the switch above the maps shows raw vectors or each source's average removed. Each dot is one chunk: articles blue,
book paragraphs orange, video descriptions green, transcript passages violet (drawn small and underneath; they are
{pct(tr['share'])} of all dots). Black dots with labels are the book chapters, each at the average position of its
paragraphs. Untick a source to hide it; hover a dot for its text (transcripts with the time in the video).</p>
<ul>
 <li><strong>Spoken and written Urza share one space:</strong> the transcripts fill the whole map, and the articles, book
 paragraphs and descriptions lie inside that cloud rather than in territories of their own.</li>
 <li><strong>Two regions belong to one kind of text:</strong> a cluster of articles and descriptions that announce events
 (talks, seminars, interviews), and a large transcript-only region of conversational back-and-forth from debates and
 livestreams, with no written counterpart.</li>
 <li><strong>Centring loosens the source grouping</strong>, as for the written sources alone; both models give the same
 layout.</li>
 <li><strong>Caveat:</strong> the transcripts include hosts and guests, not only Urza, and outnumber the other sources
 about {tr['ratio']:.0f} to 1; the other sections use the written sources only.</li>
</ul>"""))

    # ---- 3. PCA / ICA
    def pca_pane(m):
        pp = read(m, "all_pca_ica.html")
        psec = section(pp, "PCA: how much")
        svgs = re.findall(r"<svg.*?</svg>", psec, flags=re.S)
        ptab = psec[psec.find("<table"):psec.find("</table>") + 8]
        phead, prow = table_rows(ptab, 3)
        isec = section(pp, "ICA: themes")
        itab = isec[isec.find("<table"):isec.find("</table>") + 8]
        ihead, ith = table_rows(itab, 5, "th")
        _, ifo = table_rows(itab, 5, "fo")
        n_th, n_fo = len(table_rows(itab, None, "th")[1]), len(table_rows(itab, None, "fo")[1])
        lv = {int(a): int(b) for a, b in re.findall(r"(\d+)%: (\d+) PCs", svgs[1])}
        evr1 = re.search(r"PC1 explains ([\d.]+%)", psec)
        e1 = evr1.group(1) if evr1 else "?"
        chk = pc1_check(m)
        return f"""<h3>How much does each direction explain?</h3>
<div class="panel">{svgs[0]}</div>
<p class="cap"><strong>How to read it:</strong> share of all variation captured by each of the first 20 principal
components. PC1 explains {e1}; no direction dominates.</p>
<div class="panel" style="margin-top:12px">{svgs[1]}</div>
<p class="cap"><strong>How to read it:</strong> variance explained by the first k components together. Half the
variation needs {lv.get(50, '?')} components, 90% needs {lv.get(90, '?')}: the meaning of the texts is spread over
many directions, so no flat map can show all of it.</p>
<h3>The top three directions</h3>
<div class="tw"><table>{phead}<tbody>{''.join(prow)}</tbody></table></div>
<p class="cap"><strong>How to read it:</strong> each axis with its share of variance, source η² and the texts at its
two ends (colour = source). PC1 depends partly on the kind of text (η² {chk['global']['eta1']:.2f}).</p>
<h3>PCA map</h3>
{frame(f"pca_{m}", "PCA map", h=760)}
<p class="cap"><strong>How to read it:</strong> every chunk placed by its scores on two principal components (choose
them above the map); hover a dot for its text.</p>
<h3>ICA: themes and formats (top 5 of each)</h3>
<div class="tw"><table>{ihead}<tbody>{''.join(ith)}{''.join(ifo)}</tbody></table></div>
<p class="cap"><strong>How to read it:</strong> {n_th} of the 20 components are <em>themes</em> (their 30 top chunks
come from several sources), {n_fo} are <em>formats or series</em> (at least 80% from one source: a recurring kind of
post, a video series, or one book chapter); the five most focused of each are shown. <strong>Kurtosis</strong>
measures how long-tailed a component's scores are: high means a few chunks score extremely high and the rest near
zero, so the component is sharply focused; low means it is spread over many texts.</p>
<h3>ICA map</h3>
{frame(f"pca_{m}", "ICA map", h=760)}
<p class="cap"><strong>How to read it:</strong> chunks placed by two ICA components (selectable). Because themes
cut across sources, colours mix more here than on the PCA map.</p>
<h3>Conclusions ({LABEL[m]})</h3>
<ul>
 <li><strong>No dominant direction:</strong> PC1 covers {e1}, and half the variation needs {lv.get(50, '?')} components.</li>
 <li><strong>Is the strongest direction the kind of text? Only because this PCA keeps the differences between
 sources.</strong> Here one overall average was removed, so the gap between the average article, book and video stays
 in the data: {pct(chk['between'])} of all variation, and PC1 picks it up (η² {chk['global']['eta1']:.2f}). With each
 source's own average removed, as for the maps, PC1 explains {pct(chk['source']['evr1'])} and no axis separates the
 sources any more (η² {chk['source']['eta1']:.2f}). The “kind of text” is a shift between sources, not a structure
 within the topics.</li>
 <li><strong>{"Beneath it, most independent axes are topics" if n_th >= n_fo else "Beneath it, most independent axes are formats"}:</strong>
 ICA finds {n_th} themes shared by books, articles and videos, and {n_fo} formats (a recurring kind of post, series or
 chapter), useful to spot template texts.</li>
</ul>"""
    S.append(("pca", "Linear structure: PCA and ICA", f"""
<p class="lead">All {cd['n']:,} chunks projected onto straight axes. Unlike UMAP, every axis is one fixed direction in
the embedding space, so it can be read and measured.</p>
<h3>What was done</h3>
<ol>
 <li><strong>Centring:</strong> the average vector of all chunks is subtracted (one average for all, not per source).</li>
 <li><strong>PCA</strong> finds the directions of largest variation, in order; <strong>ICA</strong> (20 components)
 finds directions whose scores are as independent and long-tailed as possible, so each tends to pick out one theme or
 one kind of text.</li>
 <li><strong>Reading an axis:</strong> the chunks at its two ends, their mix of sources, the video tags and book
 chapters over-represented among them. <strong>η²</strong> = share of an axis' variance explained by the source alone
 (0 = none, 1 = the axis only tells articles, books and videos apart).</li>
</ol>
{switch("pca", {m: pca_pane(m) for m, _, _ in MODELS})}"""))

    # ---- 4. book map
    trs = {m: tight_rows(m) for m, _, _ in MODELS}
    rho = {m: float(re.search(r"rank correlation of the two columns in the table below\): <strong>([\d.]+)",
                              read(m, "book_map.html")).group(1)) for m, _, _ in MODELS}
    n_groups = len(trs[MODELS[0][0]])

    def book_pane(m):
        return f"""{frame(f"bm_{m}", "The map|A second projection", h=1500)}
<h3>Conclusions ({LABEL[m]})</h3>
<ul>
 <li><strong>Chapters are ideas on the map.</strong> {pct(bm[m]['rates']['chapter'][0])} of a paragraph's nearest
 neighbours are from its own chapter, against {pct(bm[m]['rates']['chapter'][1])} by chance, and {bm[m]['tight']} of
 {n_groups} chapters form compact regions (median spread {np.median([r['u'] for r in trs[m]]):.2f} of random).</li>
 <li><strong>Not an artefact of UMAP:</strong> t-SNE agrees on which chapters are compact and which are spread out
 (rank correlation {rho[m]:.2f}).</li>
</ul>"""
    S.append(("books", "The book idea map", f"""
<p class="lead">Do regions of the map correspond to ideas? The books are the cleanest test: their chapters are topic
labels written by the author, and the paragraphs were embedded without them.</p>
<h3>What was done</h3>
<ol>
 <li><strong>Data:</strong> the {c['book']} book paragraphs, each embedded from its own text alone.</li>
 <li><strong>Chapter groups:</strong> each paragraph belongs to one of {n_groups} groups: a chapter, or for the appendix of
 <em>Anarchokapitalismus</em> its topic or AMEN category; the aphorisms form one group.</li>
 <li><strong>Maps:</strong> UMAP and, as a check, t-SNE, on the book paragraphs only; each chapter's name sits at the
 average position of its paragraphs. A far-away island (the opening stories) is moved closer for display; measurements
 use the original layout.</li>
 <li><strong>Measures:</strong> each paragraph's 10 nearest neighbours, and how tightly each chapter sits on the
 map (its spread relative to a random set of the same size: 1 = random, below 0.2 = compact).</li>
</ol>
<h3>UMAP and t-SNE maps</h3>
{switch("bm", {m: book_pane(m) for m, _, _ in MODELS})}
<p class="cap"><strong>How to read it:</strong> each dot is a paragraph, coloured by book (blue
<em>Anarchokapitalismus</em>, orange <em>Anarchoagorismus</em>). Click a chapter in the list, or on the map, to
highlight it on both maps; hover for the text.</p>
<h3>How tight is each chapter? Both models</h3>
<div class="panel">{tight_svg(trs)}</div>
<p class="cap"><strong>How to read it:</strong> each dot is one chapter group (colour = book), placed by its spread
relative to a random set (log scale; left = tighter); the vertical mark is the median. Hover a dot for the chapter.
<strong>What it shows:</strong> in both models most chapters are far tighter than random (median
{', '.join(f"{np.median([r['u'] for r in trs[m]]):.2f} in {LABEL[m]}" for m, _, _ in MODELS)} on UMAP); t-SNE spreads
everything a little more, keeping the same order.</p>"""))

    # ---- 5. trails
    def trails_pane(m):
        tp = consts(read(m, "book_trails.html"))
        N, E = tp["N"], tp["E"]
        tree = sorted([e for e in E if e["tree"]], key=lambda e: -e["lift"])
        cross = sum(1 for e in tree if N[e["a"]]["book"] != N[e["b"]]["book"])
        trow = "".join(
            f"<tr><td>{esc(N[e['a']]['name'])} – {esc(N[e['b']]['name'])}</td><td class=r>{e['lift']:.1f}</td>"
            f"<td class=r>{e['links']}</td><td class=m>{esc(e['bridges'][0][2][:110])}… / {esc(e['bridges'][0][4][:110])}…</td></tr>"
            for e in tree[:10])
        hubs = sorted(N, key=lambda n: -n["deg"])[:4]
        alone = [n["name"] for n in N if n["deg"] == 0]
        return f"""{frame(f"trails_{m}", "The trail map", h=1250)}
<p class="cap"><strong>How to read it:</strong> circles are sections (size = number of paragraphs, colour = book),
lines are links (thicker = higher lift). Pick two sections, or a suggested pair, to see the trail between them and
the passages that bridge each step.</p>
<h3>The backbone ({LABEL[m]})</h3>
<p class="cap" style="margin-top:0">The <strong>backbone</strong> is the maximum spanning tree of the network: the
{len(tree)} strongest links that still connect the sections, without loops. It keeps the main lines of argument;
{cross} of its links join the two books, the places where both books discuss the same thing. The ten backbone links
with the highest lift:</p>
<div class="tw"><table><thead><tr><th>Link</th><th class=r>Lift</th><th class=r>Neighbour pairs</th><th>Strongest bridge (the two passages)</th></tr></thead><tbody>{trow}</tbody></table></div>
<h3>Conclusions ({LABEL[m]})</h3>
<ul>
 <li><strong>The topics form connected trails</strong>: {len(E)} links between {len(N)} sections, {cross} backbone links
 across the two books.</li>
 <li><strong>Broad sections are hubs:</strong> {', '.join(esc(n['name']) for n in hubs)} connect the most topics.</li>
 {f"<li><strong>Standing apart:</strong> {', '.join(esc(a) for a in alone)} {'has' if len(alone) == 1 else 'have'} no strong link to any other section.</li>" if alone else ""}
</ul>"""
    S.append(("trails", "Trails between book topics", f"""
<p class="lead">On the book map, sections sit next to related sections; following those neighbours reads like a line
of argument. Here the neighbourhoods become a network of links between the book sections.</p>
<h3>What was done</h3>
<ol>
 <li><strong>Links from the text:</strong> for every book paragraph, its 10 most similar paragraphs are taken; each
 neighbour from a different section counts as one link between the two sections.</li>
 <li><strong>Lift</strong> measures how strong a link is: the number of links between two sections divided by the
 number expected if neighbours were picked at random. Big sections would collect many links just by size; lift corrects
 for that. Lift 1 = chance, lift 10 = ten times more often than chance. Kept are links with lift ≥ 2 and at least 3
 neighbour pairs.</li>
 <li><strong>Paths:</strong> between two sections, the chain of strongest links (each step costs 1 / lift); for every
 step the most similar pair of passages across it is shown, the actual bridge between the topics.</li>
</ol>
<h3>The trail map</h3>
{switch("trails", {m: trails_pane(m) for m, _, _ in MODELS})}"""))

    # ---- 6. topics
    def topic_pane(m):
        tdp = read(m, "topic_distribution.html")
        tk = kpis(tdp)
        D = consts(tdp)["D"]
        A_, V_ = sum(d["a"] for d in D), sum(d["v"] for d in D)
        ak_a = sum(d["a"] for d in D if d["book"] == "Anarchokapitalismus") / A_
        ag_v = sum(d["v"] for d in D if d["book"] == "Anarchoagorismus") / V_
        top = sorted([d for d in D if not d["framing"]], key=lambda d: -d["share"])[:3]
        k = "".join(f'<div class="kpi"><b>{esc(a)}</b><span>{esc(b)}</span></div>' for a, b in tk[:3])
        return f"""<div class="kpis">{k}</div>
{frame(f"topic_{m}", "Distribution over", h=900)}
<p class="cap"><strong>How to read it:</strong> each bar is a book section: the share of article and video chunks
assigned to it, compared with the section's share of the books. Hover for example texts.</p>
<h3>Conclusions ({LABEL[m]})</h3>
<ul>
 <li><strong>The most frequent book topics</strong> in the articles and videos: {', '.join(esc(d['g']) for d in top)}.</li>
 <li><strong>The two books split the output:</strong> {pct(ak_a)} of article chunks land on <em>Anarchokapitalismus</em>
 topics, and {pct(ag_v)} of video chunks on <em>Anarchoagorismus</em> topics.</li>
 <li><strong>The broad picture is robust; the exact ranking is not:</strong> it changes with the size of the concept
 space, and between the two models.</li>
</ul>"""
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
{switch("topic", {m: topic_pane(m) for m, _, _ in MODELS})}"""))

    # ---- 7. politics
    def pol_pane(m):
        if not have_pol[m]:
            return (f'<p class="cap">No political-space results for {LABEL[m]} yet: the 326 political items have to be '
                    f'embedded with the same model first (see RealReports/README.md).</p>')
        psp = read(m, "political_space.html")
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
        return f"""<div class="kpis">{''.join(f'<div class="kpi"><b>{esc(a)}</b><span>{esc(b)}</span></div>' for a, b in pk)}</div>
<p class="cap" style="margin-top:0">By volume he writes most about {', '.join(esc(a['name']) for a in by_vol)}; his
answers are least direct in {', '.join(esc(a['name']) for a in by_dep)}.</p>
<h3>The map of the political space</h3>
{frame(f"pol_{m}", "The map of the political space", h=1150)}
<p class="cap"><strong>How to read it:</strong> large dots are subtopics, small dots questions, squares concepts;
colour = depth of Urza's best passage (darker = covered more squarely). Small dots behind are Urza's chunks. Hover an
item for its best passage; the placement switch changes how his chunks are placed, not the political map.</p>
<div class="two">
 <div><h3>Best-covered questions</h3><div class="tw"><table><thead><tr><th>Question</th><th class=r>Depth</th><th class=r>Texts</th><th>Best passage</th></tr></thead><tbody>{qtab(pc['QTOP'])}</tbody></table></div></div>
 <div><h3>Blind spots</h3><div class="tw"><table><thead><tr><th>Question</th><th class=r>Depth</th><th class=r>Texts</th><th>Nearest passage</th></tr></thead><tbody>{qtab(pc['QBOT'])}</tbody></table></div></div>
</div>
<p class="cap"><strong>How to read it:</strong> depth = similarity of his best passage; texts = how many different
articles, book sections or videos reach 0.55.</p>"""
    S.append(("politics", "Urza in the political space", f"""
<p class="lead">Everything so far measured Urza against himself. Here the reference is independent of him: a map of
politics in general, and how much of it he addresses.</p>
<h3>What was done</h3>
<ul>
 <li><strong>The political space:</strong> 326 items built on the Comparative Agendas Project classification: 22 policy
 areas, their subtopics, concrete questions as Czech voters meet them, and concepts of political philosophy.</li>
 <li><strong>Same model</strong> as Urza's texts; topics and concepts embedded as “name: description”, questions as
 written.</li>
 <li><strong>Depth</strong> of an item: the similarity of Urza's single best-matching passage, as a search would rank it
 (≥ 0.55 relevant, ≥ 0.60 squarely on topic; thresholds set for bge-m3).</li>
 <li><strong>Volume</strong> of an area: how many of his chunks are closest to its subtopics (each source's average
 removed first; announcements left out).</li>
 <li><strong>Map:</strong> the political items laid out with UMAP (each item type's average removed so items group by
 subject, not wording); Urza's chunks are then added without changing it, in three ways (switch above the map).</li>
</ul>
{switch("pol", {m: pol_pane(m) for m, _, _ in MODELS})}"""))

    nav = "".join(f'<a href="#{k}">{esc(t if k != "intro" else "Overview")}</a>' for k, t, _ in S)
    body = "".join(f'<section class="slide" id="{k}"><h2>{esc(t)}</h2>{c}</section>' for k, t, c in S)
    page = P.TEMPLATE.replace("__NAV__", nav).replace("__BODY__", body).replace(
        "__PAGES__", json.dumps(pages, ensure_ascii=False).replace("</", "<\\/"))
    out = REP / "presentation.html"
    out.write_text(page, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    build()
