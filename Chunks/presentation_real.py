"""The results half of the UrzaBOT presentation (part 1, data and method, is the slide deck), one self-contained page.

    python Chunks/real_data.py                         # written chunks + vectors -> Chunks/output_real/
    python Questions/embed_questions.py --model google/embeddinggemma-2 --name embeddinggemma_2   # once
    python Chunks/build_real_reports.py                # both models
    python Chunks/project_all_sources.py               # the map of all sources with the transcripts
    python Chunks/transcripts_real.py                  # political coverage with Urza's speech
    python Chunks/presentation_real.py                 # -> RealReports/presentation.html

bge-m3 and EmbeddingGemma, one switch in the top bar for the whole page. Sections: all sources on one map (with the
91,499 video transcript chunks of embeddings_out_new_cs/), linear structure (PCA/ICA, written texts), the book idea map
(UMAP and t-SNE; Anarchokapitalismus yellow, Anarchoagorismus blue), and Urza in the political space (written texts
alone vs with the transcripts). Interactive maps are the reports in RealReports/, embedded as frames showing one
section; all maps zoom (scroll), move (drag) and reset (double-click).
"""
import html as H
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import presentation as P  # noqa: E402  (shared helpers and page template)

MODELS = [("bge_m3", "bge-m3"), ("embeddinggemma_2", "EmbeddingGemma")]
LABEL = dict(MODELS)
REP = ROOT / "RealReports"
REAL = HERE / "output_real"
esc, pct, section, table_rows, frame = P.esc, P.pct, P.section, P.table_rows, P.frame
SL = {"article": "Articles", "book": "Book paragraphs", "video": "Video descriptions", "transcript": "Video transcripts"}
SC = {"article": "var(--s0)", "book": "var(--s1)", "video": "var(--s2)", "transcript": "var(--s3)"}
CSS = """<style>:root{--s3:#8b5cc8}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--s3:#a98ae6}}:root[data-theme="dark"]{--s3:#a98ae6}
.take{list-style:none;padding:0;margin:16px 0 0;display:grid;gap:10px;max-width:1000px}
.take li{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin:0}
.lg{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--ink2);margin:6px 0}
svg text.big{font-size:12px;fill:var(--ink)}a.src{color:inherit}
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin:14px 0}
.step{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;font-size:13.5px}
.step b{display:block;margin-bottom:4px}.step span{color:var(--ink2)}
.slide{min-height:0}#intro{padding-bottom:32px}
nav .vswitch{margin:0 0 0 auto;padding-left:16px;font-size:13px}
</style></head>"""
# the book map's colours: Anarchokapitalismus yellow, Anarchoagorismus blue
BOOK_CSS = """<style>:root{--b0:#ffd100;--b1:#2a78d6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--b0:#ffd100;--b1:#3987e5}}
:root[data-theme="dark"]{--b0:#ffd100;--b1:#3987e5}</style></head>"""


def read(model, name):
    return (REP / model / name).read_text(encoding="utf-8")


def frame_page(name, root):
    P.REP = root
    return P.frame_page(name)


def stats(model, name):
    return json.loads((REAL / model / "projection" / f"{name}_textonly.json").read_text(encoding="utf-8"))


def panes(content):
    """One pane per model; the switch in the top bar shows one of them (all sections switch together)."""
    return "".join(f'<div class="vpane" data-group="model" data-v="{k}"{"" if i == 0 else " hidden"}>{content[k]}</div>'
                   for i, (k, _) in enumerate(MODELS))


# readings of the PCA axis tables, written by hand: check them after the reports are rebuilt on new data
PCA_READ = {
    "bge_m3": ["<strong>PC1: kind of text</strong>: book prose ↔ conference and panel announcements.",
               "<strong>PC2: argument ↔ story</strong>: the books' theory chapters (Anarcho-capitalism, Conclusion, "
               "Liberation) ↔ the three opening stories.",
               "<strong>PC3: money ↔ education</strong>: cryptocurrency and Austrian economics ↔ free learning and the "
               "first day of school."],
    "embeddinggemma_2": ["<strong>PC1: kind of text</strong>: studio talks and event announcements ↔ the books' "
                         "economic theory (central planning, the price system).",
                         "<strong>PC2: story ↔ argument</strong>: the three opening stories ↔ anarchist theory "
                         "(Anarchist Revolution?, Theory and Practice).",
                         "<strong>PC3: money ↔ state mandates</strong>: money and banking ↔ blog commentary on "
                         "compulsory kindergarten, voting and bans."],
}


def kpis(items):
    return '<div class="kpis">' + "".join(f'<div class="kpi"><b>{a}</b><span>{b}</span></div>' for a, b in items) + "</div>"


def take(items):
    return '<ul class="take">' + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


# ---------------------------------------------------------------- charts
def depth_svg(areas, strong, some):
    """Per policy area: mean depth of Urza's best passage, written texts alone (hollow) and with his speech (filled)."""
    rows = sorted(areas, key=lambda a: -a["a"])
    W, pl, pr, rh, top = 980, 300, 30, 24, 26
    lo = np.floor((min(a["w"] for a in rows) - 0.01) * 20) / 20
    hi = max(a["a"] for a in rows) + 0.02
    X = lambda v: pl + (v - lo) / (hi - lo) * (W - pl - pr)
    el = []
    for t in np.arange(lo, hi, 0.05):
        el.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top - 4}" y2="{top + rh * len(rows)}" class="grid"/>'
                  f'<text x="{X(t):.1f}" y="{top + rh * len(rows) + 14}" text-anchor="middle" class="ax">{t:.2f}</text>')
    for t, lab in ((some, "relevant"), (strong, "on topic")):
        el.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top - 4}" y2="{top + rh * len(rows)}" class="ref"/>'
                  f'<text x="{X(t) + 4:.1f}" y="{top - 8}" class="ax">{lab} ≥ {t:.2f}</text>')
    for i, a in enumerate(rows):
        y = top + i * rh + rh / 2
        el.append(f'<text x="{pl - 10}" y="{y + 4:.1f}" text-anchor="end" class="big">{esc(a["name"])}</text>'
                  f'<line x1="{X(a["w"]):.1f}" x2="{X(a["a"]):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--s3)" stroke-width="2" opacity=".5"/>'
                  f'<circle cx="{X(a["w"]):.1f}" cy="{y:.1f}" r="5" fill="var(--card)" stroke="var(--ink2)" stroke-width="1.5">'
                  f'<title>{esc(a["name"])}: written {a["w"]:.2f}</title></circle>'
                  f'<circle cx="{X(a["a"]):.1f}" cy="{y:.1f}" r="5.5" fill="var(--s3)"><title>{esc(a["name"])}: with speech {a["a"]:.2f}</title></circle>')
    h = top + rh * len(rows) + 20
    return f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="Depth per policy area, written vs with speech" style="width:100%;height:auto">{"".join(el)}</svg>'


def qtable(rows):
    def src(p):
        t = esc(p["title"][:70])
        link = f'<a class="src" href="{esc(p["url"])}" target="_blank" rel="noopener">{t}</a>' if p["url"] else t
        return f'<span class="sw" style="background:{SC[p["src"]]}"></span>{link}'
    return ('<div class="tw"><table><thead><tr><th>Question</th><th class=r>Written</th><th class=r>+ speech</th>'
            '<th>Best passage</th></tr></thead><tbody>' + "".join(
                f'<tr><td>{esc(r["q"])}</td><td class=r>{r["w"]:.2f}</td><td class=r><b>{r["a"]:.2f}</b></td>'
                f'<td>{src(r["p"])}</td></tr>' for r in rows) + "</tbody></table></div>")


# ---------------------------------------------------------------- sections
def pca_pane(m):
    pp = read(m, "all_pca_ica.html")
    psec = section(pp, "PCA: how much")
    svgs = re.findall(r"<svg.*?</svg>", psec, flags=re.S)
    lv = {int(a): int(b) for a, b in re.findall(r"(\d+)%: (\d+) PCs", svgs[1])}
    e1 = (re.search(r"PC1 explains ([\d.]+%)", psec) or [None, "?"])[1]
    ptab = psec[psec.find("<table"):psec.find("</table>") + 8]
    phead, prow = table_rows(ptab, 6)                 # PC1-PC3, low and high end each
    dim = int(np.load(REAL / m / "results_textonly" / "embeddings.npy", mmap_mode="r").shape[1])
    isec = section(pp, "ICA: themes")
    itab = isec[isec.find("<table"):isec.find("</table>") + 8]
    ihead, ith = table_rows(itab, 6, "th")
    n_th, n_fo = len(table_rows(itab, None, "th")[1]), len(table_rows(itab, None, "fo")[1])
    return f"""{kpis([(f"{dim:,}", f"dimensions of the embedding space ({LABEL[m]})"), (e1, "of the variation on the strongest axis"),
       (str(lv.get(50, "?")), "axes for half of it"), (str(lv.get(90, "?")), "axes for 90%")])}
<div class="panel">{svgs[1]}</div>
<h3>What the first PCA axes mean</h3>
<div class="tw"><table>{phead}<tbody>{''.join(prow)}</tbody></table></div>
<p class="cap">Each axis read from the 30 chunks at each of its ends, as for ICA below; η² = how much of the axis is just the kind of text.</p>
<h3>ICA: the most focused themes</h3>
<div class="tw"><table>{ihead}<tbody>{''.join(ith)}</tbody></table></div>
{take(PCA_READ[m] + [
       "<strong>Meaning is spread over many directions</strong>: no flat map can show it all, so maps keep neighbours, not distances.",
       f"<strong>ICA separates themes from formats</strong>: {n_th} of 20 components are themes shared across sources, {n_fo} are formats or series of one source."])}"""


def book_pane(m):
    bm = stats(m, "book_stats")
    rate = bm["rates"]["chapter"]
    return f"""{kpis([(pct(rate[0]), f"of a paragraph's 10 nearest neighbours are from its own chapter (chance {pct(rate[1])})"),
       (f"{bm['tight']}", "chapters form compact regions")])}
{frame(f"bm_{m}", "The map|A second projection", h=1700)}
{take(["<strong>Chapters are ideas on the map</strong>: regions correspond to the author's own topics. Click a chapter to highlight it."])}"""


def pol_pane(m):
    pol = json.loads((REAL / m / "transcripts_results.json").read_text(encoding="utf-8"))["politics"]
    qs, ctrl, st = pol["q_strong"], pol["q_strong_speech_same_size"], pol["strong"]
    scale = ("set by reading examples" if m == "bge_m3" else
             "the values that as large a share of all question-passage pairs reaches as 0.60 and 0.55 with bge-m3: "
             "each model has its own similarity scale")
    return f"""<div class="steps">
 <div class="step"><b>1 · Questions</b><span>{pol['n_q']} concrete political questions as Czech voters meet them
 (“Should school attendance be compulsory?”), sorted into 22 policy areas of the Comparative Agendas Project, the
 standard classification of policy topics.</span></div>
 <div class="step"><b>2 · Search</b><span>Each question is embedded with {LABEL[m]} and compared with all of Urza's chunks,
 exactly like a RAG query.</span></div>
 <div class="step"><b>3 · Depth</b><span>The similarity of his single best passage. ≥ {st:.2f} = squarely on topic,
 ≥ {pol['some']:.2f} = relevant ({scale}).</span></div>
 <div class="step"><b>4 · Compare</b><span>Written texts alone vs with the transcripts; and the transcripts cut down to the
 size of the written texts, to tell content from sheer volume.</span></div>
</div>
{kpis([(f"{pct(qs[0])} → {pct(qs[1])}", f"questions with a passage on topic (≥ {st:.2f}): written alone → with the transcripts"),
       (pct(ctrl[0]), f"transcripts alone, sampled to the size of the written texts ({pol['n_written']:,} chunks)"),
       (f"{pol['best_from'].get('transcript', 0)} / {pol['n_q']}", "questions whose best passage is spoken")])}
<h3>Depth per policy area</h3>
<div class="lg"><span><span class="sw" style="background:var(--card);border:1.5px solid var(--ink2)"></span>written texts</span><span><span class="sw" style="background:var(--s3)"></span>with the transcripts</span></div>
<div class="panel">{depth_svg(pol["areas"], st, pol["some"])}</div>
<p class="cap">One row per policy area: the average depth over its subtopics and questions. The line shows what the
transcripts add.</p>
<div class="two">
 <div><h3>Best covered</h3>{qtable(pol["top"])}</div>
 <div><h3>Blind spots</h3>{qtable(pol["bottom"])}</div>
</div>
<p class="cap">Depth of the best passage from the written texts and with the transcripts; the title opens the passage
(spoken ones at that moment of the video).</p>
{take([f"<strong>The transcripts cover politics far more completely, mostly through their volume</strong>: cut to the size of the written texts they reach {pct(ctrl[0])} of the questions, the written texts {pct(qs[0])}.",
       "<strong>Blind spots</strong> are conventional policy-making questions he rarely addresses head-on."])}"""


# the model switch also switches the all-sources map inside its frame (it has window.setModel)
SYNC_JS = """<script>(function(){let cur=document.querySelector("nav .vswitch button.on").dataset.v;
const apply=()=>document.querySelectorAll('iframe.emb[data-page="allsrc"]').forEach(f=>{try{f.contentWindow.setModel&&f.contentWindow.setModel(cur)}catch(e){}});
document.querySelectorAll("nav .vswitch button").forEach(b=>b.addEventListener("click",()=>{cur=b.dataset.v;apply()}));
document.querySelectorAll('iframe.emb[data-page="allsrc"]').forEach(f=>f.addEventListener("load",apply));})();</script>"""


def build():
    sys.stdout.reconfigure(encoding="utf-8")
    cnt = json.loads((REAL / MODELS[0][0] / "transcripts_results.json").read_text(encoding="utf-8"))["counts"]
    n_written = cnt["article"] + cnt["book"] + cnt["video"]
    pages, S = {}, []
    for m, _ in MODELS:
        pages[f"bm_{m}"] = frame_page("book_map.html", REP / m).replace("</head>", BOOK_CSS, 1)
    pages["allsrc"] = frame_page("projection_all_sources.html", REP)

    S.append(("intro", "UrzaBOT: what the embeddings show", """
<p class="lead">Part 2: results. Everything Urza has written and said, embedded with bge-m3 and EmbeddingGemma
(switch at the top right).</p>
<ol class="agenda">
 <li><a href="#maps">Everything on one map</a></li><li><a href="#pca">Linear structure</a></li>
 <li><a href="#books">The book idea map</a></li><li><a href="#politics">Urza in the political space</a></li>
</ol>"""))
    S.append(("maps", "Everything on one map", f"""
<p class="lead">Each dot is one chunk. Right: each source's average vector removed, so style no longer groups them.</p>
{frame("allsrc", "The maps", h=860)}
{take(["<strong>Writing and speech share one space.</strong> The written texts lie inside the cloud of the transcripts, not in territories of their own."])}"""))
    S.append(("pca", "Linear structure", f"""
<p class="lead">The {n_written:,} written chunks on straight axes (PCA, ICA): no single direction explains much.</p>
{panes({m: pca_pane(m) for m, _ in MODELS})}"""))
    S.append(("books", "The book idea map", f"""
<p class="lead">Book paragraphs, embedded without their chapter names, coloured by book.</p>
{panes({m: book_pane(m) for m, _ in MODELS})}"""))
    S.append(("politics", "Urza in the political space", f"""
<p class="lead">So far Urza was measured against himself. Here the yardstick is independent of him: does he have
something squarely on topic for the questions of ordinary politics?</p>
{panes({m: pol_pane(m) for m, _ in MODELS})}"""))

    nav = "".join(f'<a href="#{k}">{esc(t if k != "intro" else "Overview")}</a>' for k, t, _ in S)
    nav += ('<div class="vswitch" data-group="model"><div class="seg">' + "".join(
        f'<button data-v="{k}"{" class=on" if i == 0 else ""}>{esc(l)}</button>' for i, (k, l) in enumerate(MODELS))
        + "</div></div>")
    body = "".join(f'<section class="slide" id="{k}"><h2>{esc(t)}</h2>{c}</section>' for k, t, c in S)
    page = P.TEMPLATE.replace("</head>", CSS, 1).replace("<title>Mapping Urza's ideas</title>", "<title>UrzaBOT results</title>")
    page = page.replace("__NAV__", nav).replace("__BODY__", body).replace(
        "__PAGES__", json.dumps(pages, ensure_ascii=False).replace("</", "<\\/"))
    page = page.replace("</body></html>", SYNC_JS + "</body></html>")
    out = REP / "presentation.html"
    out.write_text(page, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    build()
