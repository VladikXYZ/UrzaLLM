"""Map of the two books only (raw embeddings, no source-centring), labelled by chapter.

    python Projection/book_map.py --variant nosource      # from the repo root

Question: do regions of the map correspond to ideas? The books are the best test,
because their chapters are human-made topic labels.

Reads  Preprocessing/output/chunks.parquet, Embedding/results_<variant>/embeddings.npy
Writes Projection/output/book_map_<variant>.html   interactive, self-contained report
       Projection/output/book_map_<variant>.png    static map
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "output"
BOOK_COLORS = {"Anarchokapitalismus": "#2a78d6", "Anarchoagorismus": "#eb6834"}  # validated slots 1–2
UMAP_PARAMS = dict(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
K = 10


def chapter_group(r) -> str:
    """The idea-level unit: a chapter, or for the appendix its topic / AMEN category."""
    ch, sub = r["chapter"], (r["subchapter"] or "").split(" › ")
    if ch == "dodatek":
        return f"AMEN: {sub[1]}" if sub[0].startswith("AMEN") and len(sub) > 1 else sub[-1]
    if ch.startswith("AMEN"):
        return "AMEN: aforismy"
    return ch


def neighbour_rate(nn, lab):
    return float((lab[nn] == lab[:, None]).mean())


def random_rate(lab):
    c, n = Counter(lab), len(lab)
    return sum(v * (v - 1) for v in c.values()) / (n * (n - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="nosource")
    variant = ap.parse_args().variant
    sys.stdout.reconfigure(encoding="utf-8")
    import umap

    rows = pq.read_table(ROOT / "Preprocessing" / "output" / "chunks.parquet",
                         columns=["chunk_id", "doc_id", "source", "book", "chapter", "subchapter",
                                  "title", "text"]).to_pylist()
    E = np.load(ROOT / "Embedding" / f"results_{variant}" / "embeddings.npy")
    ids = (ROOT / "Embedding" / f"results_{variant}" / "chunk_ids.txt").read_text(encoding="utf-8").split()
    assert ids == [r["chunk_id"] for r in rows], "embeddings don't match chunks.parquet"
    idx = [i for i, r in enumerate(rows) if r["source"] == "book"]
    B, X = [rows[i] for i in idx], E[idx]
    grp = np.array([chapter_group(r) for r in B])
    book = np.array([r["book"] for r in B])
    sec = np.array([r["doc_id"] for r in B])

    # Neighbours in the full 1,024-dim space
    S = X @ X.T
    np.fill_diagonal(S, -1)
    nn = np.argsort(-S, axis=1)[:, :K]
    other_sec = sec[nn] != sec[:, None]
    rates = {
        "section": (neighbour_rate(nn, sec), random_rate(sec)),
        "chapter": (neighbour_rate(nn, grp), random_rate(grp)),
        "book": (neighbour_rate(nn, book), random_rate(book)),
    }
    cross = float(((grp[nn] == grp[:, None]) & other_sec).sum() / other_sec.sum())

    print(f"UMAP on {len(B)} book chunks …")
    P = umap.UMAP(**UMAP_PARAMS).fit_transform(X)
    P = (P - P.min(0)) / (P.max(0) - P.min(0)).max()

    rng = np.random.default_rng(0)

    def spread(ix):
        Q = P[ix]
        return float(np.sqrt(((Q - Q.mean(0)) ** 2).sum(1)).mean())

    chapters = []
    for g, cnt in Counter(grp).items():
        ix = np.where(grp == g)[0]
        ratio = None
        if cnt >= 3:
            ratio = spread(ix) / np.mean([spread(rng.choice(len(P), cnt, replace=False)) for _ in range(50)])
        nn_same = float((grp[nn[ix]] == g).mean())
        chapters.append({"name": g, "book": B[ix[0]]["book"], "n": int(cnt), "ratio": ratio,
                         "nn_same": nn_same, "cx": float(P[ix, 0].mean()), "cy": float(P[ix, 1].mean()),
                         "first": int(min(ix))})
    chapters.sort(key=lambda c: c["first"])
    for k, c in enumerate(chapters):
        c["id"] = k
    gid = {c["name"]: c["id"] for c in chapters}

    OUT.mkdir(exist_ok=True)
    png(P, book, chapters, OUT / f"book_map_{variant}.png")
    pts = [[round(float(P[i, 0]), 4), round(float(P[i, 1]), 4), 0 if b["book"] == "Anarchokapitalismus" else 1,
            gid[grp[i]], " › ".join(x for x in (b["chapter"], b["subchapter"]) if x), b["text"][:240]]
           for i, b in enumerate(B)]
    html(pts, chapters, rates, cross, variant, len(B), OUT / f"book_map_{variant}.html")
    print("same-section / chapter / book neighbours:",
          {k: f"{v[0]:.0%} vs {v[1]:.0%}" for k, v in rates.items()}, f"cross-section same chapter {cross:.0%}")
    print(f"Wrote {OUT / f'book_map_{variant}.html'}")


def png(P, book, chapters, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 11), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for b, col in BOOK_COLORS.items():
        m = book == b
        ax.scatter(P[m, 0], P[m, 1], s=10, c=col, alpha=0.6, linewidths=0, label=b)
    for c in chapters:
        if c["n"] >= 5:
            ax.text(c["cx"], c["cy"], c["name"], fontsize=7, ha="center", va="center", color="#0b0b0b",
                    bbox=dict(boxstyle="round,pad=0.15", fc="#fcfcfb", ec="none", alpha=0.75))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#e4e3df")
    ax.legend(frameon=False, loc="upper left", markerscale=2)
    ax.set_title("Book chunks (raw embeddings), labelled at each chapter's centre", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def html(pts, chapters, rates, cross, variant, n, path):
    p = lambda v: f"{round(v * 100)}%"
    ratios = sorted(c["ratio"] for c in chapters if c["ratio"] is not None)
    med = ratios[len(ratios) // 2]
    tight = sum(1 for r in ratios if r < 0.2)
    loose = [c for c in sorted(chapters, key=lambda c: -(c["ratio"] or 0)) if (c["ratio"] or 0) >= 0.3]
    rows = sorted([c for c in chapters if c["ratio"] is not None], key=lambda c: c["ratio"])
    table = "".join(
        f'<tr data-g="{c["id"]}"><td><span class="sw" style="background:var(--b{0 if c["book"] == "Anarchokapitalismus" else 1})"></span>'
        f'{esc(c["name"])}</td><td>{c["n"]}</td><td>{c["ratio"]:.2f}</td><td>{p(c["nn_same"])}</td></tr>' for c in rows)
    fill = {
        "__N__": str(n), "__NCH__": str(len(chapters)), "__VARIANT__": variant,
        "__SEC__": p(rates["section"][0]), "__SECR__": p(rates["section"][1]),
        "__CH__": p(rates["chapter"][0]), "__CHR__": p(rates["chapter"][1]),
        "__BK__": p(rates["book"][0]), "__BKR__": p(rates["book"][1]),
        "__CROSS__": p(cross), "__MED__": f"{med:.2f}", "__MEDX__": f"{1 / med:.0f}",
        "__TIGHT__": str(tight), "__NRAT__": str(len(ratios)),
        "__LOOSE__": ", ".join(f"{esc(c['name'])} ({c['ratio']:.2f})" for c in loose),
        "__TABLE__": table,
        "__DATA__": json.dumps(pts, ensure_ascii=False).replace("</", "<\\/"),
        "__CHAPTERS__": json.dumps([{k: c[k] for k in ("id", "name", "n", "cx", "cy", "book")} for c in chapters],
                                   ensure_ascii=False).replace("</", "<\\/"),
    }
    page = TEMPLATE
    for k, v in fill.items():
        page = page.replace(k, v)
    path.write_text(page, encoding="utf-8")


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Book idea map</title>
<style>
:root{--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--b0:#2a78d6;--b1:#eb6834;--hl:#0b0b0b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835;--b0:#3987e5;--b1:#d95926;--hl:#f0efec}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1200px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}p.sub,p.cap{color:var(--ink2);max-width:900px}p.cap{font-size:13px;margin:8px 2px 0}
h2.sec{font-size:17px;margin:28px 0 6px;padding-top:12px;border-top:1px solid var(--line)}
ol,ul{max-width:900px;padding-left:20px}li{margin:6px 0}
code{background:var(--line);padding:1px 5px;border-radius:4px;font-size:12.5px}
.mapwrap{display:grid;grid-template-columns:minmax(0,1fr) 260px;gap:16px;align-items:start}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
canvas{width:100%;aspect-ratio:1/1;display:block;cursor:crosshair}
.side{max-height:820px;overflow:auto;font-size:12.5px}.side h3{font-size:13px;margin:0 0 6px}
.side div.c{padding:3px 6px;border-radius:5px;cursor:pointer;display:flex;gap:6px;align-items:center}
.side div.c:hover,.side div.c.on{background:var(--line)}.side .n{color:var(--ink2);margin-left:auto}
.controls{display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin:0 0 10px}
.controls label{display:inline-flex;gap:6px;align-items:center;cursor:pointer}
.sw{width:10px;height:10px;border-radius:50%;display:inline-block;flex:none}
#tip{position:fixed;pointer-events:none;background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:8px;
 padding:8px 10px;font-size:12.5px;max-width:380px;box-shadow:0 4px 16px rgba(0,0,0,.12);display:none;z-index:9}
#tip .t{font-weight:600}#tip .m{color:var(--ink2);margin-top:4px}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;background:var(--card);border:1px solid var(--line);border-radius:8px}
th,td{padding:5px 12px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600}tbody tr{cursor:pointer}tbody tr:hover{background:var(--line)}
.tw{overflow-x:auto;max-width:100%}.scroll{max-height:460px;overflow:auto;display:inline-block;border-radius:8px}
@media (max-width:860px){.mapwrap{grid-template-columns:1fr}.side{max-height:300px}}
</style></head><body><div class="wrap">
<h1>Book idea map <span style="font-weight:400;color:var(--ink2)">· raw embeddings, variant <code>__VARIANT__</code></span></h1>
<p class="sub">Do regions of the embedding map correspond to ideas? The two books are the cleanest test: their
chapters are topic labels written by the author. This report maps only the book chunks, with the raw
embeddings (no source-centring, which is unnecessary when all chunks come from one source).</p>

<h2 class="sec">What was done</h2>
<ol>
 <li><strong>Data.</strong> The __N__ book chunks (about 150 words each) from <em>Anarchokapitalismus</em> and
 <em>Anarchoagorismus</em>, with their bge-m3 embeddings (variant <code>__VARIANT__</code>: the context line is
 “Anarchokapitalismus › 10. Zdravotnictví” etc., without the word “Kniha”).</li>
 <li><strong>Chapter groups.</strong> Each chunk belongs to one of __NCH__ groups: a chapter, or in the appendix
 of <em>Anarchokapitalismus</em> its own topic (Drogy, Zbraně, Věda, Práce) or AMEN category (Ekonomie,
 Historie, …). All AMEN aphorisms of <em>Anarchoagorismus</em> form one group.</li>
 <li><strong>Map.</strong> UMAP to 2D (cosine distance, 15 neighbours, min_dist 0.1, fixed seed) on these chunks
 only. Each chapter's name is placed at the average position of its chunks.</li>
 <li><strong>Measurements.</strong> Who are each chunk's 10 nearest neighbours (full 1,024-dim vectors), and how
 tightly each chapter's chunks sit together on the map.</li>
</ol>

<h2 class="sec">The map</h2>
<p class="sub">Each dot is a book chunk, coloured by book. Hover a dot to read the start of the chunk. Click a
chapter in the list (or in the table below) to highlight its chunks; click again to clear.</p>
<div class="controls"><label><input type="checkbox" id="labels" checked> Chapter labels</label>
<span><span class="sw" style="background:var(--b0)"></span> Anarchokapitalismus</span>
<span><span class="sw" style="background:var(--b1)"></span> Anarchoagorismus</span></div>
<div class="mapwrap">
 <div class="panel"><canvas id="cv"></canvas></div>
 <div class="panel side"><h3>Chapters (book order)</h3><div id="list"></div></div>
</div>
<p class="cap"><strong>What you see:</strong> chapters form compact islands with their label on top:
“10. Zdravotnictví”, “Drogy”, “6. Hasiči”, “12. Silnice a dopravní pravidla”, “14. Soudnictví” each occupy
their own spot. <strong>Related chapters are neighbours:</strong> “AMEN: Peníze a bankovnictví” sits between
“5. Peníze” and the economics chapters; “AMEN: Ekonomie” next to “1. Vzácné zdroje” and “2. Proč selhává
centrální plánování?”; “AMEN: Přirozené právo” next to “3. Princip neagrese”; law and enforcement
(“14. Soudnictví”, “15. Vymáhání práva”) together. <strong>The two books occupy different halves</strong>
(<em>Anarchokapitalismus</em> at the bottom: institutions and economics; <em>Anarchoagorismus</em> at the
top: freedom, obedience, self-development). <strong>Where they cross, they share a subject:</strong>
<em>Anarchokapitalismus</em> “8. Školství” sits among the education chapters of <em>Anarchoagorismus</em>
(“2. Školství”, “3. Indoktrinace”); <em>Anarchoagorismus</em> “11. Zločin” sits by the law chapters, and its
“19. Praxe › Kryptopeníze” inside the money region. The three opening stories (“Tři příběhy”) are an island
far to the side: narrative fiction, unlike everything else.</p>

<h2 class="sec">Who are a chunk's nearest neighbours?</h2>
<p class="sub"><strong>How it is computed:</strong> for each book chunk, the 10 most similar book chunks (cosine
similarity of the full vectors), and the share of them that come from the same section, the same chapter group,
or the same book. <strong>Expected at random</strong> is the share if neighbours were picked blindly.</p>
<div class="tw"><table>
<thead><tr><th>Neighbour from the same…</th><th>Observed</th><th>Expected at random</th></tr></thead>
<tbody>
<tr><td>section (consecutive chunks of one text)</td><td>__SEC__</td><td>__SECR__</td></tr>
<tr><td>chapter group</td><td>__CH__</td><td>__CHR__</td></tr>
<tr><td>book</td><td>__BK__</td><td>__BKR__</td></tr>
</tbody></table></div>
<p class="cap"><strong>Reading it:</strong> __CH__ of a chunk's nearest neighbours are from its own chapter,
against __CHR__ by chance. Part of that is trivial: __SEC__ are from the very same section (neighbouring
paragraphs, which also share one overlapping sentence). Looking only at neighbours from <em>other</em>
sections, __CROSS__ are still from the same chapter group, i.e. different parts of a chapter find each other.
__BK__ of neighbours are from the same book (50% by chance): the two books differ in subject and tone.</p>

<h2 class="sec">How tight is each chapter?</h2>
<p class="sub"><strong>How it is computed:</strong> the average distance of a chapter's chunks from their centre
on the map, divided by the same for random sets of chunks of the same size (50 draws).
<strong>1.00</strong> = no more clustered than random; <strong>0.10</strong> = packed into a tenth of the
spread. “Neighbours in chapter” is the share of the chunks' 10 nearest neighbours from the same chapter.
Click a row to highlight it on the map.</p>
<div class="scroll"><table id="ctab"><thead><tr><th>Chapter group</th><th>Chunks</th><th>Spread vs random</th><th>Neighbours in chapter</th></tr></thead>
<tbody>__TABLE__</tbody></table></div>
<p class="cap"><strong>Reading it:</strong> the median chapter has a spread ratio of __MED__, i.e. its chunks are
about __MEDX__× more concentrated than random; __TIGHT__ of __NRAT__ groups are below 0.20. The loosest groups
are the broad ones: __LOOSE__. “19. Praxe” is nine different subsections (Ježek bez klece, Svobodný přístav,
Kryptoanarchie, Agora, …), and “XX. Boření mýtů” debunks myths from many areas, so they are spread because
their content is.</p>

<h2 class="sec">Conclusions</h2>
<ul>
 <li><strong>Yes: regions of the map correspond to ideas.</strong> In the books, almost every chapter occupies
 its own compact region, and chapters that discuss related subjects lie next to each other. This is the
 evidence that the embedding organises Urza's texts by topic, which the coverage map relies on.</li>
 <li><strong>Where a chapter is spread out, its content is mixed.</strong> The loose groups are exactly the ones
 that cover several subjects, not failures of the embedding.</li>
 <li><strong>Caveat: the chapter name is in the text.</strong> Every chunk starts with its chapter path
 (“Anarchokapitalismus › 10. Zdravotnictví”), which helps chunks of one chapter find each other. The
 cross-section figure (__CROSS__) and the fact that related chapters sit side by side suggest the content
 does most of the work, but the effect of the name is not measured separately here.</li>
 <li><strong>The books are a good “skeleton” for the full map.</strong> Their chapter labels can name the
 regions of the full map (articles and videos included), since the same topics recur across sources.</li>
</ul>
</div><div id="tip"></div>
<script>
const P=__DATA__, CH=__CHAPTERS__;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const cv=document.getElementById("cv"), tip=document.getElementById("tip"), PAD=14;
let sel=-1, hover=-1, showLabels=true;
const list=document.getElementById("list");
CH.forEach(c=>{const d=document.createElement("div");d.className="c";d.dataset.g=c.id;
 d.innerHTML=`<span class="sw" style="background:var(--b${c.book==="Anarchokapitalismus"?0:1})"></span>${esc(c.name)}<span class="n">${c.n}</span>`;
 d.onclick=()=>select(c.id);list.appendChild(d)});
document.querySelectorAll("#ctab tbody tr").forEach(tr=>tr.onclick=()=>{select(+tr.dataset.g);cv.scrollIntoView({behavior:"smooth",block:"center"})});
document.getElementById("labels").onchange=e=>{showLabels=e.target.checked;draw()};
function select(g){sel=sel===g?-1:g;document.querySelectorAll(".side div.c").forEach(d=>d.classList.toggle("on",+d.dataset.g===sel));draw()}
function size(){const r=cv.getBoundingClientRect(),d=devicePixelRatio||1;cv.width=r.width*d;cv.height=r.width*d}
function xy(x,y){const d=devicePixelRatio||1,w=cv.width-2*PAD*d;return[PAD*d+x*w,cv.height-PAD*d-y*w]}
function draw(){const g=cv.getContext("2d"),d=devicePixelRatio||1,col=[css("--b0"),css("--b1")],bg=css("--card"),ink=css("--ink");
 g.clearRect(0,0,cv.width,cv.height);
 P.forEach(p=>{const[x,y]=xy(p[0],p[1]);g.globalAlpha=sel<0?.7:(p[3]===sel?.95:.1);g.fillStyle=col[p[2]];
  g.beginPath();g.arc(x,y,(sel===p[3]?4:3)*d,0,7);g.fill()});
 g.globalAlpha=1;
 if(hover>=0){const[x,y]=xy(P[hover][0],P[hover][1]);g.strokeStyle=bg;g.lineWidth=4*d;g.beginPath();g.arc(x,y,7*d,0,7);g.stroke();
  g.strokeStyle=css("--hl");g.lineWidth=2*d;g.beginPath();g.arc(x,y,7*d,0,7);g.stroke()}
 if(!showLabels&&sel<0)return;
 g.font=`${11*d}px system-ui,sans-serif`;g.textAlign="center";g.textBaseline="middle";
 const boxes=[];
 const order=CH.filter(c=>sel<0?c.n>=3:c.id===sel).sort((a,b)=>(b.id===sel)-(a.id===sel)||b.n-a.n);
 order.forEach(c=>{let[x,y]=xy(c.cx,c.cy);const w=g.measureText(c.name).width+8*d,h=15*d;
  x=Math.min(Math.max(x,w/2+2*d),cv.width-w/2-2*d);y=Math.min(Math.max(y,h/2+2*d),cv.height-h/2-2*d);const b=[x-w/2,y-h/2,x+w/2,y+h/2];
  if(c.id!==sel&&boxes.some(o=>!(b[2]<o[0]||b[0]>o[2]||b[3]<o[1]||b[1]>o[3])))return;boxes.push(b);
  g.globalAlpha=.85;g.fillStyle=bg;g.fillRect(b[0],b[1],w,h);g.globalAlpha=1;g.fillStyle=ink;g.fillText(c.name,x,y)})}
cv.onmousemove=e=>{const r=cv.getBoundingClientRect(),d=devicePixelRatio||1,mx=(e.clientX-r.left)*d,my=(e.clientY-r.top)*d;
 let best=-1,bd=(9*d)**2;P.forEach((p,i)=>{const[x,y]=xy(p[0],p[1]),dd=(x-mx)**2+(y-my)**2;if(dd<bd){bd=dd;best=i}});
 if(best!==hover){hover=best;draw()}if(best<0){tip.style.display="none";return}const p=P[best];
 tip.innerHTML=`<div class="t">${esc(p[4])}</div><div class="m">${esc(p[5])}…</div>`;tip.style.display="block";
 const tw=tip.offsetWidth,th=tip.offsetHeight;tip.style.left=Math.min(e.clientX+14,innerWidth-tw-8)+"px";tip.style.top=Math.min(e.clientY+14,innerHeight-th-8)+"px"};
cv.onmouseleave=()=>{hover=-1;tip.style.display="none";draw()};
cv.onclick=()=>{if(hover>=0)select(P[hover][3])};
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
addEventListener("resize",()=>{size();draw()});matchMedia("(prefers-color-scheme: dark)").addEventListener("change",draw);
size();draw();
</script></body></html>"""


if __name__ == "__main__":
    main()
