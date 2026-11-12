"""2D projection of the embeddings: raw vs. source-centred, side by side.

    ..\\.venv\\Scripts\\python project_compare.py

Reads  Embedding/results_<variant>/{chunks_embedded.parquet, embeddings.npy, embeddings_source_centred.npy}
       (run Embedding/analyze.py --variant <variant> first)
Writes Projection/output/projection_compare_<variant>.html   interactive, self-contained
       Projection/output/projection_compare_<variant>.png    static version
       Projection/output/coords_<variant>.parquet            chunk_id + 2D coordinates, raw and centred
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
EMB_ROOT = HERE.parent / "Embedding"
OUT = HERE / "output"
SOURCES = ["article", "book", "video"]
COLORS = {"article": "#2a78d6", "book": "#eb6834", "video": "#1baf7a"}  # validated slots 1–3
UMAP_PARAMS = dict(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)


def umap2d(X):
    import umap
    return umap.UMAP(**UMAP_PARAMS).fit_transform(X)


def align(ref, Y):
    """Rotate/flip Y onto ref (orthogonal Procrustes) so the two maps are oriented alike."""
    a, b = ref - ref.mean(0), Y - Y.mean(0)
    u, _, vt = np.linalg.svd(b.T @ a)
    return b @ (u @ vt)


def same_source_rate(X, src, k=10):
    S = X @ X.T if X.shape[1] > 2 else -((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(S, -np.inf)
    nn = np.argsort(-S, axis=1)[:, :k]
    return {s: float((src[nn[src == s]] == s).mean()) for s in SOURCES}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="withsource", help="embedding variant (see Embedding/variants.py)")
    variant = ap.parse_args().variant
    EMB = EMB_ROOT / f"results_{variant}"
    if not (EMB / "embeddings_source_centred.npy").exists():
        sys.exit(f"Run first: python Embedding/analyze.py --variant {variant}")
    sys.stdout.reconfigure(encoding="utf-8")
    meta = pq.read_table(EMB / "chunks_embedded.parquet",
                         columns=["chunk_id", "doc_id", "source", "kind", "title", "year", "book",
                                  "chapter", "subchapter", "words"]).to_pylist()
    ids = (EMB / "chunk_ids.txt").read_text(encoding="utf-8").split()
    assert ids == [m["chunk_id"] for m in meta], "chunk order mismatch"
    raw = np.load(EMB / "embeddings.npy")
    cen = np.load(EMB / "embeddings_source_centred.npy")
    src = np.array([m["source"] for m in meta])

    print("UMAP on raw vectors …")
    p_raw = umap2d(raw)
    print("UMAP on source-centred vectors …")
    p_cen = align(p_raw, umap2d(cen))
    p_raw = p_raw - p_raw.mean(0)

    rates = {"raw_1024d": same_source_rate(raw, src), "centred_1024d": same_source_rate(cen, src),
             "raw_2d": same_source_rate(p_raw, src), "centred_2d": same_source_rate(p_cen, src),
             "random": {s: float(((src == s).sum() - 1) / (len(src) - 1)) for s in SOURCES}}
    for k, v in rates.items():
        print(f"  {k:14}", {s: f"{x:.0%}" for s, x in v.items()})

    OUT.mkdir(exist_ok=True)
    pq.write_table(pa.table({"chunk_id": ids,
                             "x_raw": p_raw[:, 0], "y_raw": p_raw[:, 1],
                             "x_centred": p_cen[:, 0], "y_centred": p_cen[:, 1]}), OUT / f"coords_{variant}.parquet")
    png(p_raw, p_cen, src, OUT / f"projection_compare_{variant}.png")
    (OUT / f"rates_{variant}.json").write_text(json.dumps(rates, indent=2), encoding="utf-8")
    html(meta, p_raw, p_cen, rates, OUT / f"projection_compare_{variant}.html", variant)
    print(f"Wrote {OUT}")


def png(a, b, src, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), facecolor="#fcfcfb")
    for ax, P, t in zip(axes, (a, b), ("Raw embeddings", "After subtracting each source's average")):
        ax.set_facecolor("#fcfcfb")
        for s in SOURCES:
            m = src == s
            ax.scatter(P[m, 0], P[m, 1], s=6, c=COLORS[s], alpha=0.65, linewidths=0, label=s)
        ax.set_title(t, loc="left", fontsize=12, color="#0b0b0b")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#e4e3df")
    axes[0].legend(frameon=False, markerscale=2.5, fontsize=10, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def html(meta, a, b, rates, path, variant):
    def norm(P):
        lo, hi = P.min(0), P.max(0)
        return (P - lo) / (hi - lo).max()
    a, b = norm(a), norm(b)
    pts = [[m["chunk_id"], SOURCES.index(m["source"]), m["kind"], m["title"], m["year"],
            " › ".join(x for x in (m["book"], m["chapter"], m["subchapter"]) if x) if m["source"] == "book" else "",
            round(float(a[i, 0]), 4), round(float(a[i, 1]), 4), round(float(b[i, 0]), 4), round(float(b[i, 1]), 4)]
           for i, m in enumerate(meta)]
    page = TEMPLATE.replace("__DATA__", json.dumps(pts, ensure_ascii=False).replace("</", "<\\/"))
    page = page.replace("__RATES__", json.dumps(rates))
    p = lambda v: f"{round(v * 100)}%"
    r, c, rnd = rates["raw_1024d"], rates["centred_1024d"], rates["random"]
    excess = lambda d: {s: d[s] - rnd[s] for s in SOURCES}
    er, ec = excess(r), excess(c)
    cut = {s: 1 - ec[s] / er[s] for s in SOURCES}
    worst = max(SOURCES, key=lambda s: er[s])
    label = {"article": "articles", "book": "book sections", "video": "video descriptions"}
    page = page.replace("__N__", f"{len(meta):,}")
    page = page.replace("__VARIANT__", variant)
    # How much does the context line matter? Compare all embedding variants that have been projected.
    order = ["withsource", "nosource", "textonly"]
    desc = {"withsource": "context line with the source word (“Kniha …”, “Článek „…“”, “Video „…“”)",
            "nosource": "context line without the source word",
            "textonly": "no context line, only the chunk's own text"}
    allr = {v: json.loads((OUT / f"rates_{v}.json").read_text(encoding="utf-8"))
            for v in order if v != variant and (OUT / f"rates_{v}.json").exists()}
    allr[variant] = rates
    avail = [v for v in order if v in allr]
    if len(avail) > 1:
        rows_html = "".join(
            f"<tr><td>{'<strong>' if v == variant else ''}<code>{v}</code>{'</strong>' if v == variant else ''}"
            f" – {desc.get(v, '')}</td>"
            + "".join(f"<td>{p(allr[v]['raw_1024d'][s])}</td>" for s in SOURCES)
            + "".join(f"<td>{p(allr[v]['centred_1024d'][s])}</td>" for s in SOURCES) + "</tr>" for v in avail)
        found = []
        if "withsource" in allr and "nosource" in allr:
            d = [round((allr["withsource"]["raw_1024d"][s] - allr["nosource"]["raw_1024d"][s]) * 100) for s in SOURCES]
            found.append(f"removing just the source word lowers the raw same-source share by {min(d)}–{max(d)} points")
        if "nosource" in allr and "textonly" in allr:
            d = [round((allr["nosource"]["raw_1024d"][s] - allr["textonly"]["raw_1024d"][s]) * 100) for s in SOURCES]
            dc = [round((allr["nosource"]["centred_1024d"][s] - allr["textonly"]["centred_1024d"][s]) * 100) for s in SOURCES]
            found.append(f"removing the rest of the context line (titles, chapter paths, years, guests) lowers it by "
                         f"another {min(d)}–{max(d)} points raw and {min(dc)}–{max(dc)} points after centring")
        label_txt = (
            "<strong>The context line adds to the clustering, the texts do the rest.</strong> The same chunks "
            "were embedded with different context lines (same-source share of the 10 nearest neighbours, full "
            "vectors; chance: " + " / ".join(p(rnd[s]) for s in SOURCES) + "):"
            "</li></ul><div class='tw'><table><thead><tr><th>Variant</th>"
            + "".join(f"<th>Raw, {label[s]}</th>" for s in SOURCES)
            + "".join(f"<th>Centred, {label[s]}</th>" for s in SOURCES)
            + f"</tr></thead><tbody>{rows_html}</tbody></table></div><ul class='concl'><li style='list-style:none'>"
            + ("; ".join(found).capitalize() + ". " if found else "")
            + "What remains even without any context line comes from the texts themselves: their style, "
              "length and the way each source writes about a topic.")
    else:
        label_txt = ("Part may be the context line at the start of each chunk (“Kniha …”, “Video „…“”, "
                     "“Článek …”); the <code>nosource</code> and <code>textonly</code> variants test that.")
    page = page.replace("__LABEL_BULLET__", label_txt)
    page = page.replace("__TABLE_READING__", (
        f"<strong>Reading it:</strong> a book chunk's 10 nearest neighbours are {p(r['book'])} books in the raw "
        f"vectors, against {p(rnd['book'])} expected by chance. After centring it is {p(c['book'])}. The 2D maps "
        f"show the same pattern ({p(rates['raw_2d']['book'])} → {p(rates['centred_2d']['book'])} for books), so "
        f"the maps reflect the full vectors faithfully."))
    page = page.replace("__C1__", (
        f"In the raw vectors every source has far more same-source neighbours than chance would give: "
        + ", ".join(f"{label[s]} {p(r[s])} vs {p(rnd[s])}" for s in SOURCES)
        + f". The effect is strongest for {label[worst]}."))
    page = page.replace("__C2__", (
        "Measured as the excess over chance, centring "
        + ", ".join((f"removes {p(cut[s])} of it for {label[s]}" if cut[s] >= 0.005 else
                     f"slightly increases it for {label[s]} ({p(r[s])} → {p(c[s])})" if cut[s] < -0.005 else
                     f"leaves it unchanged for {label[s]}") for s in SOURCES)
        + f" (full vectors). Same-source neighbours after centring: {p(c['article'])} / {p(c['book'])} / "
          f"{p(c['video'])} for articles / books / videos, still well above chance. On the map the change "
          f"is larger: the source continents disappear into one map organised by subject."))
    path.write_text(page, encoding="utf-8")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Embedding projection comparison</title>
<style>
:root{--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;
 --s0:#2a78d6;--s1:#eb6834;--s2:#1baf7a;--hl:#0b0b0b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835;
 --s0:#3987e5;--s1:#d95926;--s2:#199e70;--hl:#f0efec}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1400px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px} p.sub{color:var(--ink2);margin:0 0 16px;max-width:900px}
.controls{display:flex;flex-wrap:wrap;gap:16px;align-items:center;margin:0 0 12px}
.controls label{display:inline-flex;align-items:center;gap:6px;cursor:pointer;color:var(--ink)}
.sw{width:10px;height:10px;border-radius:50%;display:inline-block}
input[type=search]{background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:6px 10px;min-width:220px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.panel h2{font-size:15px;margin:0 0 2px}.panel .note{color:var(--ink2);font-size:12.5px;margin:0 0 8px}
canvas{width:100%;aspect-ratio:1/1;display:block;cursor:crosshair}
#tip{position:fixed;pointer-events:none;background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:8px;
 padding:8px 10px;font-size:12.5px;max-width:340px;box-shadow:0 4px 16px rgba(0,0,0,.12);display:none;z-index:9}
#tip .t{font-weight:600;margin-bottom:2px}#tip .m{color:var(--ink2)}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;margin-top:8px;background:var(--card);border:1px solid var(--line);border-radius:8px}
th,td{padding:6px 12px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600}tr:last-child td{border-bottom:0}td:first-child{white-space:nowrap}
.tw{overflow-x:auto}
h2.sec{font-size:17px;margin:28px 0 6px;padding-top:12px;border-top:1px solid var(--line)}
.steps,.concl{max-width:900px;padding-left:20px}.steps li,.concl li{margin:6px 0}
p.cap{color:var(--ink2);font-size:13px;margin:8px 2px 0;max-width:900px}
code{background:var(--line);padding:1px 5px;border-radius:4px;font-size:12.5px}
@media (max-width:820px){.grid{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<h1>Raw vs. source-centred embeddings <span style="font-weight:400;color:var(--ink2)">· variant <code>__VARIANT__</code></span></h1>
<p class="sub">Does the embedding place Urza's texts by <em>topic</em>, or by <em>kind of text</em> (book prose, blog
post, video description)? For a coverage map we need the first. This page compares the embeddings as they come
from the model with a corrected version.</p>

<h2 class="sec">What was done</h2>
<ol class="steps">
 <li><strong>Embedding.</strong> Each of the __N__ chunks (articles, book sections, video descriptions; see the
 preprocessing report) was turned into a 1,024-number vector with the model <code>BAAI/bge-m3</code>. Texts on
 similar subjects should get similar vectors (measured by cosine similarity).</li>
 <li><strong>Correction (“source-centring”).</strong> For each source separately, the average of all its vectors
 was computed and subtracted from each of its vectors, which were then rescaled to length 1. Whatever all books
 have in common (style, length, any shared context line) is removed; what makes one book chunk different from
 another (its topic) stays.</li>
 <li><strong>Projection.</strong> Both versions were reduced to 2D with UMAP (cosine distance, 15 neighbours,
 min_dist 0.1, fixed random seed), which keeps nearby texts nearby. The right map was rotated to line up with the
 left one, so the two are easy to compare. Distances between far-apart clusters on a UMAP map are not
 meaningful; only who sits next to whom is.</li>
 <li><strong>Measurement.</strong> For both versions, how often a chunk's nearest neighbours come from the same
 source (table below).</li>
</ol>

<h2 class="sec">The two maps</h2>
<p class="sub">Each dot is one chunk, coloured by source. Hover a dot to see the chunk; the same chunk is circled
on the other map. Use the boxes to hide sources and the search to highlight titles (try <em>drog</em>,
<em>školství</em> or <em>bitcoin</em>).</p>
<div class="controls" id="controls"></div>
<div class="grid">
 <div class="panel"><h2>Raw embeddings</h2><canvas id="c0"></canvas>
  <p class="cap"><strong>What you see:</strong> the map splits into two large “continents” by kind of text, not
  by subject: video descriptions (green) on one, articles (blue) and book sections (orange) on the other. Chunks
  on the same subject from different sources can end up far apart; search <em>drog</em> to see where the drug
  chunks land.</p></div>
 <div class="panel"><h2>After subtracting each source's average</h2><canvas id="c1"></canvas>
  <p class="cap"><strong>What you see:</strong> one connected map with colours mixed throughout. Small groups
  are organised by subject and contain several sources (search <em>drog</em> again: the matching chunks from
  books, articles and videos move much closer together). Books still form small local groups, because
  consecutive sections of a chapter really do discuss the same thing.</p></div>
</div>

<h2 class="sec">Same-source share of the 10 nearest neighbours</h2>
<p class="sub"><strong>How it is computed:</strong> for every chunk, take the 10 chunks most similar to it and
count how many come from the same source (article, book or video); then average over all chunks of that source.
<strong>Expected at random</strong> is what this share would be if similarity ignored the source entirely, i.e.
that source's share of all other chunks (books are only 23% of the corpus, so a book chunk would have about 2 of
10 book neighbours by chance). The gap between a column and “expected at random” measures how much the source,
rather than the topic, decides who is similar. “Full vectors” uses cosine similarity of the 1,024-number vectors
(what the coverage scoring will use); “2D map” uses distance on the maps above.</p>
<div class="tw"><table id="rates"></table></div>
<p class="cap">__TABLE_READING__</p>

<h2 class="sec">Conclusions</h2>
<ul class="concl">
 <li><strong>The raw embeddings mix topic with kind of text.</strong> __C1__</li>
 <li><strong>Source-centring fixes the big picture but only part of the close neighbourhoods.</strong> __C2__
 The reason: subtracting an average removes the shift that pushed each source into its own region (hence the
 single connected map), but the closest neighbours of a chunk are still mostly texts of the same kind.</li>
 <li><strong>Some of the remaining same-source grouping is genuine.</strong> Sections of one book chapter, a
 series of lecture announcements or the yearly Rothbard posts really are about the same thing. __LABEL_BULLET__</li>
 <li><strong>For the coverage map, use the centred vectors</strong>: the map is organised by subject, which is
 what “which political issues does Urza cover” needs. Expect sources to still appear as small same-colour
 patches inside each subject area. <strong>For scoring a question against chunks</strong>, keep the raw vectors:
 a question has no source, and the model was trained on raw vectors. Because neighbours are still partly
 same-source, coverage should be reported per source as well as overall.</li>
</ul>
</div><div id="tip"></div>
<script>
const P=__DATA__, R=__RATES__;
const SRC=["article","book","video"], LABEL={article:"Articles",book:"Books",video:"Videos"};
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let on=[true,true,true], hover=-1, query="";
const ctr=document.getElementById("controls");
SRC.forEach((s,i)=>{const l=document.createElement("label");
 l.innerHTML=`<input type="checkbox" checked><span class="sw" style="background:var(--s${i})"></span>${LABEL[s]} (${P.filter(p=>p[1]===i).length.toLocaleString()})`;
 l.querySelector("input").onchange=e=>{on[i]=e.target.checked;draw()};ctr.appendChild(l)});
const q=document.createElement("input");q.type="search";q.placeholder="Highlight titles containing…";
q.oninput=()=>{query=q.value.trim().toLowerCase();draw()};ctr.appendChild(q);
const cv=[0,1].map(i=>document.getElementById("c"+i)), tip=document.getElementById("tip");
const PAD=10;
function size(){cv.forEach(c=>{const r=c.getBoundingClientRect(),d=devicePixelRatio||1;c.width=r.width*d;c.height=r.width*d})}
function xy(c,p,k){const d=devicePixelRatio||1,w=c.width-2*PAD*d;return[PAD*d+p[6+2*k]*w,c.height-PAD*d-p[7+2*k]*w]}
function draw(){const col=SRC.map((_,i)=>css("--s"+i)),hl=css("--hl"),bg=css("--card");
 cv.forEach((c,k)=>{const g=c.getContext("2d"),d=devicePixelRatio||1;g.clearRect(0,0,c.width,c.height);
  const match=p=>query&&p[3].toLowerCase().includes(query);
  P.forEach(p=>{if(!on[p[1]])return;const[x,y]=xy(c,p,k);g.globalAlpha=query&&!match(p)?.12:.7;
   g.fillStyle=col[p[1]];g.beginPath();g.arc(x,y,2.6*d,0,7);g.fill()});
  g.globalAlpha=1;
  if(query)P.forEach(p=>{if(!on[p[1]]||!match(p))return;const[x,y]=xy(c,p,k);g.fillStyle=col[p[1]];
   g.strokeStyle=bg;g.lineWidth=1.5*d;g.beginPath();g.arc(x,y,4*d,0,7);g.fill();g.stroke()});
  if(hover>=0){const[x,y]=xy(c,P[hover],k);g.strokeStyle=bg;g.lineWidth=5*d;g.beginPath();g.arc(x,y,7*d,0,7);g.stroke();
   g.strokeStyle=hl;g.lineWidth=2*d;g.beginPath();g.arc(x,y,7*d,0,7);g.stroke()}})}
cv.forEach((c,k)=>{c.onmousemove=e=>{const r=c.getBoundingClientRect(),d=devicePixelRatio||1,mx=(e.clientX-r.left)*d,my=(e.clientY-r.top)*d;
  let best=-1,bd=(10*d)**2;P.forEach((p,i)=>{if(!on[p[1]])return;const[x,y]=xy(c,p,k),dd=(x-mx)**2+(y-my)**2;if(dd<bd){bd=dd;best=i}});
  if(best!==hover){hover=best;draw()}
  if(best<0){tip.style.display="none";return}const p=P[best];
  tip.innerHTML=`<div class="t">${esc(p[3])}</div><div class="m">${LABEL[SRC[p[1]]]} · ${p[2]}${p[4]?" · "+p[4]:""}</div>${p[5]?`<div class="m">${esc(p[5])}</div>`:""}<div class="m">${p[0]}</div>`;
  tip.style.display="block";const tw=tip.offsetWidth,th=tip.offsetHeight;
  tip.style.left=Math.min(e.clientX+14,innerWidth-tw-8)+"px";tip.style.top=Math.min(e.clientY+14,innerHeight-th-8)+"px"};
 c.onmouseleave=()=>{hover=-1;tip.style.display="none";draw()}});
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
const pct=v=>Math.round(v*100)+"%";
document.getElementById("rates").innerHTML="<thead><tr><th>Source</th><th>Raw, full vectors</th><th>Centred, full vectors</th><th>Raw, 2D map</th><th>Centred, 2D map</th><th>Expected at random</th></tr></thead><tbody>"+
 SRC.map(s=>`<tr><td><span class="sw" style="background:var(--s${SRC.indexOf(s)})"></span> ${LABEL[s]}</td><td>${pct(R.raw_1024d[s])}</td><td>${pct(R.centred_1024d[s])}</td><td>${pct(R.raw_2d[s])}</td><td>${pct(R.centred_2d[s])}</td><td>${pct(R.random[s])}</td></tr>`).join("")+"</tbody>";
addEventListener("resize",()=>{size();draw()});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change",draw);
size();draw();
</script></body></html>"""


if __name__ == "__main__":
    main()
