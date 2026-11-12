"""Greedy nearest-topic links on the book map (textonly): every topic points to its single nearest topic.

    python Projection/book_nearest.py        # from the repo root

Topics are the book map's chapter groups. "Nearest" is measured in meaning: cosine similarity of the
topics' average vectors in the full 1,024-dim space (an alternative view uses distance on the map).
Following the arrows from any topic always ends in a mutual pair (two topics that choose each other);
all topics that lead to the same pair form a family.

Writes book_nearest_textonly.html (+ .png) to the dataset's report folder (see dataset.py).
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import LABEL_JS, place_labels  # noqa: E402
from book_map import UMAP_PARAMS, chapter_group, compact, esc  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import CHUNKS, ENGLISH, OUT_DIR, nm, read_chunks, results_dir  # noqa: E402
OUT = OUT_DIR
VARIANT = "textonly"


def unit(A):
    return A / np.linalg.norm(A, axis=-1, keepdims=True)


def families(nn):
    """Follow the arrows; topics ending in the same cycle (a mutual pair) form one family."""
    fam = {}
    for i in range(len(nn)):
        seen, v = [], i
        while v not in seen:
            seen.append(v)
            v = int(nn[v])
        fam[i] = tuple(sorted(seen[seen.index(v):]))
    keys = sorted(set(fam.values()), key=lambda k: -sum(1 for v in fam.values() if v == k))
    return [keys.index(fam[i]) for i in range(len(nn))], keys


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import umap
    rows = read_chunks(CHUNKS, ["chunk_id", "source", "book", "chapter", "subchapter"])
    E = np.load(results_dir(VARIANT) / "embeddings.npy")
    b = [i for i, r in enumerate(rows) if r["source"] == "book"]
    B, X = [rows[i] for i in b], E[b]
    grp = np.array([chapter_group(r) for r in B], dtype=object)
    topics = list(dict.fromkeys(grp))
    n = len(topics)

    # Same map as the book map report
    P = umap.UMAP(**UMAP_PARAMS).fit_transform(X)
    P = (P - P.min(0)) / (P.max(0) - P.min(0)).max()
    Pd, moved = compact(P, grp)
    pos = np.array([Pd[grp == t].mean(0) for t in topics])
    opos = np.array([P[grp == t].mean(0) for t in topics])         # original layout, for the map-distance view

    # Nearest topic in meaning
    C = unit(np.array([X[grp == t].mean(0) for t in topics]))
    S = C @ C.T
    np.fill_diagonal(S, -1)
    nn_m = S.argmax(1)
    # Nearest topic on the map (original 2D layout)
    D = np.sqrt(((opos[:, None] - opos[None]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    nn_p = D.argmin(1)

    fam_m, keys_m = families(nn_m)
    fam_p, keys_p = families(nn_p)
    nodes = [{"id": k, "name": t, "book": B[int(np.where(grp == t)[0][0])]["book"], "n": int((grp == t).sum()),
              "x": round(float(pos[k, 0]), 4), "y": round(float(pos[k, 1]), 4),
              "nm": int(nn_m[k]), "sm": round(float(S[k, nn_m[k]]), 3), "fm": fam_m[k],
              "np": int(nn_p[k]), "sp": round(float(S[k, nn_p[k]]), 3), "fp": fam_p[k],
              "second": [int(j) for j in np.argsort(-S[k])[1:3]]} for k, t in enumerate(topics)]
    pts = [[round(float(Pd[i, 0]), 4), round(float(Pd[i, 1]), 4), 0 if B[i]["book"] == "Anarchokapitalismus" else 1]
           for i in range(len(B))]
    fams = {"m": [[list(k), [i for i in range(n) if fam_m[i] == f]] for f, k in enumerate(keys_m)],
            "p": [[list(k), [i for i in range(n) if fam_p[i] == f]] for f, k in enumerate(keys_p)]}
    agree = float(np.mean(nn_m == nn_p))
    indeg = Counter(nn_m)

    OUT.mkdir(exist_ok=True)
    png(nodes, Pd, B, OUT / f"book_nearest_{VARIANT}.png")
    html(nodes, pts, fams, agree, indeg, moved, OUT / f"book_nearest_{VARIANT}.html")
    print(f"{n} topics; meaning: {len(keys_m)} families; map: {len(keys_p)} families; same nearest {agree:.0%}")
    print(f"Wrote {OUT / f'book_nearest_{VARIANT}.html'}")


def png(nodes, Pd, B, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"Anarchokapitalismus": "#2a78d6", "Anarchoagorismus": "#eb6834"}
    fig, ax = plt.subplots(figsize=(11, 11 * Pd[:, 1].max() / Pd[:, 0].max() + 0.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.scatter(Pd[:, 0], Pd[:, 1], s=5, c=[col[r["book"]] for r in B], alpha=0.15, linewidths=0)
    for nd in nodes:
        t = nodes[nd["nm"]]
        mutual = nodes[nd["nm"]]["nm"] == nd["id"]
        ax.annotate("", xy=(t["x"], t["y"]), xytext=(nd["x"], nd["y"]),
                    arrowprops=dict(arrowstyle="-|>", color="#0b0b0b", lw=2.2 if mutual else 1.1,
                                    alpha=0.85, shrinkA=6, shrinkB=7, mutation_scale=10))
    for nd in nodes:
        ax.scatter([nd["x"]], [nd["y"]], s=14 + 4 * nd["n"], c=col[nd["book"]], edgecolors="#fcfcfb", linewidths=1, zorder=3)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for spn in ax.spines.values():
        spn.set_color("#e4e3df")
    ax.set_title("Each book topic points to its nearest topic in meaning (thick = mutual pair)", loc="left", fontsize=11.5)
    place_labels(ax, [(nd["x"], nd["y"], nd["name"]) for nd in sorted(nodes, key=lambda nd: -nd["n"])], fontsize=6.5)
    fig.savefig(path, dpi=150, facecolor="#fcfcfb", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def html(nodes, pts, fams, agree, indeg, moved, path):
    j = lambda o: json.dumps(o, ensure_ascii=False).replace("</", "<\\/")
    q = lambda s: f"“{esc(s)}”"
    name = lambda i: nodes[i]["name"]
    pairs = [k for k, _ in fams["m"]]
    big = fams["m"][0]
    popular = [(i, c) for i, c in indeg.most_common(5)]
    cross = [nd for nd in nodes if nodes[nd["nm"]]["book"] != nd["book"]]
    # Largest family: which book most of it comes from
    bigbook, nbig = Counter(nodes[i]["book"] for i in big[1]).most_common(1)[0]
    # Kinds of mutual pair: chapter + AMEN section, across the two books, within one book
    amen = lambda i: name(i).startswith("AMEN")
    k_amen = [p for p in pairs if amen(p[0]) != amen(p[1])]
    k_x = [p for p in pairs if p not in k_amen and nodes[p[0]]["book"] != nodes[p[1]]["book"]]
    k_rest = len(pairs) - len(k_amen) - len(k_x)
    pl = lambda c, s, p: f"{c} {s if c == 1 else p}"
    kinds = [f"{pl(len(k_amen), 'pair joins', 'pairs join')} a chapter with an AMEN section"]
    if k_x:
        kinds.append(f"{pl(len(k_x), 'pair joins', 'pairs join')} the two books ("
                     + ", ".join(f"{q(name(a))} ↔ {q(name(c))}" for a, c in k_x[:2]) + ")")
    if k_rest:
        kinds.append(f"{pl(k_rest, 'pair joins', 'pairs join')} other sections of the same book")
    kinds = "; ".join(kinds)
    if agree < 0.5:
        agree_txt = (f"The two views pick the same nearest topic for only {agree:.0%} of topics: the topic closest "
                     "in meaning is usually not the one next door on the map.")
    else:
        agree_txt = (f"The two views pick the same nearest topic for {agree:.0%} of topics; where they differ, "
                     "the topic closest in meaning is not the one next door on the map.")
    fill = {"__LABELJS__": LABEL_JS,"__NODES__": j(nodes), "__PTS__": j(pts), "__FAMS__": j(fams),
            "__NT__": str(len(nodes)), "__NF__": str(len(fams["m"])), "__NFP__": str(len(fams["p"])),
            "__AGREETXT__": agree_txt, "__KINDS__": kinds,
            "__BIGBOOK__": f"{nbig} of them from <em>{esc(bigbook)}</em>",
            "__PAIRS__": "; ".join(f"{q(name(a))} ↔ {q(name(c))}" for a, c in pairs),
            "__BIG__": f"{len(big[1])} topics ending in {q(name(big[0][0]))} ↔ {q(name(big[0][1]))}",
            "__POPULAR__": ", ".join(f"{q(name(i))} ({c})" for i, c in popular),
            "__NCROSS__": str(len(cross)),
            "__CROSS__": ", ".join(f"{q(nd['name'])} → {q(name(nd['nm']))}" for nd in cross[:6]),
            "__MOVED__": "The opening stories were moved closer on the map, as in the book map report. " if moved else ""}
    page = TEMPLATE
    for k, v in fill.items():
        page = page.replace(k, v)
    path.write_text(page, encoding="utf-8")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nearest topics in the books</title>
<style>
:root{--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--b0:#2a78d6;--b1:#eb6834;--arrow:#0b0b0b;--hl:#eda100}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835;--b0:#3987e5;--b1:#d95926;--arrow:#f0efec;--hl:#c98500}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}p.sub,p.cap{color:var(--ink2);max-width:900px}p.cap{font-size:13px;margin:8px 2px 0}
h2.sec{font-size:17px;margin:28px 0 6px;padding-top:12px;border-top:1px solid var(--line)}
ol,ul{max-width:900px;padding-left:20px}li{margin:6px 0}code{background:var(--line);padding:1px 5px;border-radius:4px;font-size:12.5px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.mapwrap{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:16px;align-items:start}
canvas{width:100%;display:block;cursor:pointer}
.controls{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:0 0 10px;font-size:13px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:7px;overflow:hidden}
.seg button{background:var(--card);color:var(--ink2);border:0;padding:5px 11px;cursor:pointer;font:inherit}
.seg button.on{background:var(--line);color:var(--ink);font-weight:600}
.side{font-size:12.5px;max-height:820px;overflow:auto}.side h3{font-size:13px;margin:4px 0 6px}
.fam{padding:6px 8px;border-radius:6px;cursor:pointer;border-top:1px solid var(--line)}.fam:hover,.fam.on{background:var(--line)}
.fam b{font-size:12.5px}.fam .m{color:var(--ink2);font-size:12px}
.sw{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:4px;vertical-align:-1px}
table{border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:8px;font-size:12.5px;width:100%}
th,td{padding:5px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{color:var(--ink2);font-weight:600}
td.r{text-align:right;font-variant-numeric:tabular-nums}.tw{overflow-x:auto;max-height:520px;overflow-y:auto}
@media (max-width:960px){.mapwrap{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<h1>Nearest topics in the books <span style="font-weight:400;color:var(--ink2)">· book map, variant <code>textonly</code></span></h1>
<p class="sub">The simplest possible link structure: on the book map, every topic draws one arrow to the single
topic closest to it. Following the arrows, the __NT__ topics fall into families that end in a pair of topics
choosing each other.</p>

<h2 class="sec">What was done</h2>
<ol>
 <li><strong>Topics</strong> are the __NT__ chapter groups of the book map (chapters; the appendix and AMEN split into
 their topics), placed at the average position of their chunks on the same textonly UMAP map. __MOVED__</li>
 <li><strong>Nearest in meaning</strong> (default): each topic's chunks are averaged into one vector (full 1,024
 dimensions), and each topic points to the topic whose average vector is most similar (cosine).</li>
 <li><strong>Nearest on the map</strong> (alternative view): the topic whose centre is closest on the 2D map.</li>
 <li><strong>Families:</strong> following the arrows from any topic always ends in a mutual pair, two topics that
 are each other's nearest. All topics leading to the same pair form a family.</li>
</ol>

<h2 class="sec">The map</h2>
<div class="controls"><span>Nearest</span><div class="seg" id="mode"><button data-v="m" class="on">in meaning</button><button data-v="p">on the map</button></div>
<span><span class="sw" style="background:var(--b0)"></span><em>Anarchokapitalismus</em></span><span><span class="sw" style="background:var(--b1)"></span><em>Anarchoagorismus</em></span></div>
<div class="mapwrap">
 <div class="panel"><canvas id="cv"></canvas></div>
 <div class="panel side"><h3>Families (click to highlight)</h3><div id="fams"></div></div>
</div>
<p class="cap"><strong>How to read it:</strong> each arrow goes from a topic to its nearest topic; thick double
arrows are mutual pairs, where a family ends. Hover a topic for its nearest and next-nearest topics with
similarities; click a topic or a family to highlight it. In meaning, there are __NF__ families; on the map
__NFP__. __AGREETXT__ A topic's position is the average of chunks that can be spread out,
and UMAP arranges the picture as a whole rather than keeping every pair of close topics side by side. The
“in meaning” arrows are therefore the reliable ones; the map gives them a place to be drawn.</p>

<h2 class="sec">The families</h2>
<p class="cap"><strong>What it shows:</strong> the mutual pairs are __PAIRS__. The largest family has __BIG__
(__BIGBOOK__; its members are listed in the side panel). The most chosen topics (how many
others point to them) are __POPULAR__. __NCROSS__ topics point across to the other book, for example
__CROSS__.</p>
<div class="tw" style="margin-top:10px"><table id="tab"><thead><tr><th>Topic</th><th>Nearest topic</th><th class="r">Similarity</th><th>Next nearest</th></tr></thead><tbody></tbody></table></div>

<h2 class="sec">Conclusions</h2>
<ul>
 <li><strong>One greedy step already reveals the books' structure:</strong> the families correspond to the
 books' main themes (economics, law and justice, the state, school, personal freedom).</li>
 <li><strong>The mutual pairs are the cores of those themes:</strong> of the __NF__ pairs, __KINDS__.</li>
 <li><strong>Limitation:</strong> each topic keeps only one link, so a topic close to several themes appears in
 just one family. The next-nearest topics in the table show the alternatives; the trails report keeps all
 strong links.</li>
</ul>
</div><div id="tip" style="position:fixed;pointer-events:none;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:12.5px;max-width:340px;box-shadow:0 4px 16px rgba(0,0,0,.12);display:none;z-index:9"></div>
<script>
__LABELJS__
const N=__NODES__, PT=__PTS__, F=__FAMS__, PAD=18;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const cv=document.getElementById("cv"),tip=document.getElementById("tip");let mode="m",selF=-1,selN=-1,hover=-1;
const nn=n=>mode==="m"?n.nm:n.np, fam=n=>mode==="m"?n.fm:n.fp, sim=n=>mode==="m"?n.sm:n.sp;
document.querySelectorAll("#mode button").forEach(b=>b.onclick=()=>{document.querySelectorAll("#mode button").forEach(x=>x.classList.toggle("on",x===b));mode=b.dataset.v;selF=-1;selN=-1;renderFams();renderTab();draw()});
function renderFams(){document.getElementById("fams").innerHTML=F[mode].map(([pair,mem],f)=>`<div class="fam${f===selF?" on":""}" data-f="${f}"><b>${esc(N[pair[0]].name)} ↔ ${esc(N[pair[1]].name)}</b><div class="m">${mem.length} topics: ${mem.filter(i=>!pair.includes(i)).map(i=>esc(N[i].name)).join(", ")||"–"}</div></div>`).join("");
 document.querySelectorAll(".fam").forEach(d=>d.onclick=()=>{const f=+d.dataset.f;selF=selF===f?-1:f;selN=-1;renderFams();draw()})}
function renderTab(){document.querySelector("#tab tbody").innerHTML=N.map(n=>`<tr><td><span class="sw" style="background:var(--b${n.book==="Anarchokapitalismus"?0:1})"></span>${esc(n.name)}</td><td>${esc(N[nn(n)].name)}${N[nn(n)].nm===n.id&&mode==="m"||N[nn(n)].np===n.id&&mode==="p"?" ↔":""}</td><td class="r">${sim(n).toFixed(3)}</td><td style="color:var(--ink2)">${n.second.map(i=>esc(N[i].name)).join(", ")}</td></tr>`).join("")}
const W=Math.max(...PT.map(p=>p[0])),H=Math.max(...PT.map(p=>p[1]));
function size(){const r=cv.getBoundingClientRect(),d=devicePixelRatio||1,h=(r.width-2*PAD)*H/W+2*PAD;cv.style.height=h+"px";cv.width=r.width*d;cv.height=h*d}
function xy(x,y){const d=devicePixelRatio||1,k=(cv.width-2*PAD*d)/W;return[PAD*d+x*k,cv.height-PAD*d-y*k]}
const rad=n=>(3.5+Math.sqrt(n.n)*1.7)*(devicePixelRatio||1);
const active=n=>selF>=0?fam(n)===selF:selN>=0?(n.id===selN||nn(N[selN])===n.id||nn(n)===selN):true;
function arrow(g,x1,y1,x2,y2,r1,r2,w,col,d){const a=Math.atan2(y2-y1,x2-x1),sx=x1+Math.cos(a)*r1,sy=y1+Math.sin(a)*r1,ex=x2-Math.cos(a)*(r2+2*d),ey=y2-Math.sin(a)*(r2+2*d);
 g.strokeStyle=col;g.fillStyle=col;g.lineWidth=w;g.beginPath();g.moveTo(sx,sy);g.lineTo(ex,ey);g.stroke();const h=(6+w)*1;g.beginPath();g.moveTo(ex,ey);g.lineTo(ex-Math.cos(a-.4)*h,ey-Math.sin(a-.4)*h);g.lineTo(ex-Math.cos(a+.4)*h,ey-Math.sin(a+.4)*h);g.closePath();g.fill()}
function draw(){const g=cv.getContext("2d"),d=devicePixelRatio||1,col=[css("--b0"),css("--b1")],bg=css("--card"),foc=selF>=0||selN>=0;g.clearRect(0,0,cv.width,cv.height);
 PT.forEach(p=>{const[x,y]=xy(p[0],p[1]);g.globalAlpha=.12;g.fillStyle=col[p[2]];g.beginPath();g.arc(x,y,2*d,0,7);g.fill()});
 N.forEach(n=>{const t=N[nn(n)],mut=nn(t)===n.id,[x1,y1]=xy(n.x,n.y),[x2,y2]=xy(t.x,t.y),on=active(n)&&active(t);g.globalAlpha=foc&&!on?.1:.85;
  arrow(g,x1,y1,x2,y2,rad(n),rad(t),(mut?3:1.4)*d,on&&foc?css("--hl"):css("--arrow"),d)});
 N.forEach(n=>{const[x,y]=xy(n.x,n.y),on=active(n);g.globalAlpha=foc&&!on?.3:1;g.fillStyle=col[n.book==="Anarchokapitalismus"?0:1];g.strokeStyle=n.id===hover||n.id===selN?css("--hl"):bg;g.lineWidth=(n.id===hover||n.id===selN?3:1.5)*d;g.beginPath();g.arc(x,y,rad(n),0,7);g.fill();g.stroke()});g.globalAlpha=1;
 placeLabels(g,[...N].sort((a,b)=>(active(b)&&foc)-(active(a)&&foc)||b.n-a.n).map(n=>{const[x,y]=xy(n.x,n.y),on=!foc||active(n);
  return{x,y,text:n.name,r:rad(n),bold:foc&&active(n),color:on?css("--ink"):css("--ink2")}}),d,bg,css("--ink"),css("--ink2"))}
function hit(e){const r=cv.getBoundingClientRect(),d=devicePixelRatio||1,mx=(e.clientX-r.left)*d,my=(e.clientY-r.top)*d;let best=-1,bd=Infinity;
 N.forEach(n=>{const[x,y]=xy(n.x,n.y),dd=Math.hypot(x-mx,y-my);if(dd<rad(n)+6*d&&dd<bd){bd=dd;best=n.id}});return best}
cv.onmousemove=e=>{const h=hit(e);if(h!==hover){hover=h;draw()}if(h<0){tip.style.display="none";return}const n=N[h];
 tip.innerHTML=`<b>${esc(n.name)}</b><div style="color:var(--ink2)">nearest: ${esc(N[nn(n)].name)} (${sim(n).toFixed(3)})<br>next: ${n.second.map(i=>esc(N[i].name)).join(", ")}</div>`;tip.style.display="block";
 tip.style.left=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-8)+"px";tip.style.top=Math.min(e.clientY+14,innerHeight-tip.offsetHeight-8)+"px"};
cv.onmouseleave=()=>{hover=-1;tip.style.display="none";draw()};
cv.onclick=e=>{const h=hit(e);selF=-1;selN=selN===h?-1:h;renderFams();draw()};
addEventListener("resize",()=>{size();draw()});matchMedia("(prefers-color-scheme: dark)").addEventListener("change",draw);
renderFams();renderTab();size();draw();
</script></body></html>"""


if __name__ == "__main__":
    main()
