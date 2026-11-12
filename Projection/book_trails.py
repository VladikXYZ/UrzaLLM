"""Trails between the books' topics: a link graph of book sections, drawn over the book map (textonly).

    python Projection/book_trails.py        # from the repo root

Two sections are linked when their chunks are among each other's nearest neighbours (10 nearest, full
1,024-dim vectors) more often than chance would give. Link strength = lift = observed links / expected
links. The report shows the graph over the UMAP book map, its backbone (maximum spanning tree), and a
path explorer: the chain of strongest links between any two sections, with the passages that bridge
each step.

Writes Projection/output/book_trails_textonly.html (+ .png)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import LABEL_JS, place_labels  # noqa: E402
from book_map import UMAP_PARAMS, compact, esc  # noqa: E402
from timeline_map import region_of  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import CHUNKS, OUT_DIR, results_dir  # noqa: E402
OUT = OUT_DIR
VARIANT = "textonly"
K = 10                # nearest neighbours per chunk
MIN_LINKS = 3         # an edge needs at least this many neighbour links
MIN_LIFT = 2.0        # …and this many times more than chance
SUGGESTED = [("2. Školství", "9. Demokracie"), ("8. Školství", "8. Stát"), ("Drogy", "22. Láska"),
             ("5. Peníze", "16. Neposlušnost"), ("3. Princip neagrese", "15. Osvobození"),
             ("10. Zdravotnictví", "AMEN: Lidské jednání"), ("12. Silnice a dopravní pravidla", "13. Revoluce")]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import umap
    rows = pq.read_table(CHUNKS,
                         columns=["chunk_id", "source", "book", "chapter", "subchapter", "text"]).to_pylist()
    E = np.load(results_dir(VARIANT) / "embeddings.npy")
    b = [i for i, r in enumerate(rows) if r["source"] == "book"]
    B, X = [rows[i] for i in b], E[b]
    grp = np.array([region_of(r) for r in B], dtype=object)
    groups = list(dict.fromkeys(grp))                                   # book order
    gi = np.array([groups.index(g) for g in grp])
    n, N = len(groups), len(B)
    cnt = np.bincount(gi, minlength=n)

    # Neighbour links between sections
    S = X @ X.T
    np.fill_diagonal(S, -1)
    nn = np.argsort(-S, axis=1)[:, :K]
    L = np.zeros((n, n))
    pairs = {}
    for a in range(N):
        for c in nn[a]:
            if gi[a] != gi[c]:
                i, j = sorted((gi[a], gi[c]))
                L[i, j] += 1
                L[j, i] += 1
                pairs.setdefault((i, j), set()).add((min(a, c), max(a, c)))
    expected = np.outer(cnt, cnt) * 2 * K / (N - 1)
    lift = L / expected
    edges = []
    for (i, j), ps in pairs.items():
        if L[i, j] >= MIN_LINKS and lift[i, j] >= MIN_LIFT:
            br = sorted(ps, key=lambda p: -S[p[0], p[1]])[:2]          # strongest bridging passage pairs
            bridges = [[float(round(S[a, c], 3)), sec(B[a]), B[a]["text"][:180], sec(B[c]), B[c]["text"][:180]]
                       for a, c in br]
            edges.append({"a": int(i), "b": int(j), "lift": round(float(lift[i, j]), 2), "links": int(L[i, j]),
                          "bridges": bridges})

    # Backbone: maximum spanning tree on lift (Kruskal)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for e in sorted(edges, key=lambda e: -e["lift"]):
        ra, rb = find(e["a"]), find(e["b"])
        e["tree"] = ra != rb
        if ra != rb:
            parent[ra] = rb

    # Positions: the textonly book map (same UMAP settings as book_map.py), chapter centres
    P = umap.UMAP(**UMAP_PARAMS).fit_transform(X)
    P = (P - P.min(0)) / (P.max(0) - P.min(0)).max()
    Pd, moved = compact(P, grp)
    nodes = [{"id": k, "name": g, "book": B[int(np.where(grp == g)[0][0])]["book"], "n": int(cnt[k]),
              "x": float(Pd[gi == k, 0].mean()), "y": float(Pd[gi == k, 1].mean()),
              "deg": int(sum(1 for e in edges if k in (e["a"], e["b"])))} for k, g in enumerate(groups)]
    pts = [[round(float(Pd[i, 0]), 4), round(float(Pd[i, 1]), 4), 0 if B[i]["book"] == "Anarchokapitalismus" else 1]
           for i in range(N)]
    sugg = [[groups.index(a), groups.index(c)] for a, c in SUGGESTED if a in groups and c in groups]

    OUT.mkdir(exist_ok=True)
    png(nodes, edges, Pd, B, OUT / f"book_trails_{VARIANT}.png")
    html(nodes, edges, pts, sugg, moved, OUT / f"book_trails_{VARIANT}.html")
    tree = [e for e in edges if e["tree"]]
    print(f"{n} sections, {len(edges)} links (lift ≥ {MIN_LIFT}, ≥ {MIN_LINKS} neighbour links), backbone {len(tree)}")
    print("isolated:", [nd["name"] for nd in nodes if nd["deg"] == 0])
    print(f"Wrote {OUT / f'book_trails_{VARIANT}.html'}")


def sec(r):
    return " › ".join(x for x in (r["chapter"], r["subchapter"]) if x)


def png(nodes, edges, Pd, B, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"Anarchokapitalismus": "#2a78d6", "Anarchoagorismus": "#eb6834"}
    fig, ax = plt.subplots(figsize=(11, 11 * Pd[:, 1].max() / Pd[:, 0].max() + 0.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.scatter(Pd[:, 0], Pd[:, 1], s=5, c=[col[r["book"]] for r in B], alpha=0.15, linewidths=0)
    for e in edges:
        a, b = nodes[e["a"]], nodes[e["b"]]
        ax.plot([a["x"], b["x"]], [a["y"], b["y"]], color="#0b0b0b" if e["tree"] else "#9a9993",
                lw=0.6 + 0.12 * min(e["lift"], 20) if e["tree"] else 0.5, alpha=0.85 if e["tree"] else 0.35, zorder=1)
    for nd in nodes:
        ax.scatter([nd["x"]], [nd["y"]], s=12 + 4 * nd["n"], c=col[nd["book"]], edgecolors="#fcfcfb", linewidths=1, zorder=2)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for spn in ax.spines.values():
        spn.set_color("#e4e3df")
    ax.set_title("Trails between book sections: black = backbone (strongest links), grey = other links", loc="left", fontsize=11.5)
    place_labels(ax, [(nd["x"], nd["y"], nd["name"]) for nd in sorted(nodes, key=lambda nd: -nd["n"])], fontsize=6.5)
    fig.savefig(path, dpi=150, facecolor="#fcfcfb", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def html(nodes, edges, pts, sugg, moved, path):
    j = lambda o: json.dumps(o, ensure_ascii=False).replace("</", "<\\/")
    tree = sorted([e for e in edges if e["tree"]], key=lambda e: -e["lift"])
    q = lambda s: f"“{esc(s)}”"
    strongest = ", ".join(f"{q(nodes[e['a']]['name'])} – {q(nodes[e['b']]['name'])}" for e in tree[:5])
    hubs = sorted(nodes, key=lambda nd: -nd["deg"])[:5]
    cross = [e for e in tree if nodes[e["a"]]["book"] != nodes[e["b"]]["book"]]
    iso = [nd["name"] for nd in nodes if nd["deg"] == 0]
    fill = {"__LABELJS__": LABEL_JS,"__NODES__": j(nodes), "__EDGES__": j(edges), "__PTS__": j(pts), "__SUGG__": j(sugg),
            "__K__": str(K), "__MINL__": str(MIN_LINKS), "__MINLIFT__": f"{MIN_LIFT:g}",
            "__NN__": str(len(nodes)), "__NE__": str(len(edges)), "__NT__": str(len(tree)),
            "__STRONG__": strongest, "__HUBS__": ", ".join(f"{q(nd['name'])} ({nd['deg']} links)" for nd in hubs),
            "__CROSS__": ", ".join(f"{q(nodes[e['a']]['name'])} – {q(nodes[e['b']]['name'])}" for e in cross[:6]),
            "__NCROSS__": str(len(cross)), "__ISO__": ", ".join(q(x) for x in iso) or "none",
            "__MOVED__": ("The opening stories were moved closer on the map, as in the book map report. " if moved else "")}
    page = TEMPLATE
    for k, v in fill.items():
        page = page.replace(k, v)
    path.write_text(page, encoding="utf-8")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trails between book topics</title>
<style>
:root{--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--b0:#2a78d6;--b1:#eb6834;--edge:#9a9993;--path:#0b0b0b;--hl:#eda100}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835;--b0:#3987e5;--b1:#d95926;--edge:#6b6b66;--path:#f0efec;--hl:#c98500}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}p.sub,p.cap{color:var(--ink2);max-width:900px}p.cap{font-size:13px;margin:8px 2px 0}
h2.sec{font-size:17px;margin:28px 0 6px;padding-top:12px;border-top:1px solid var(--line)}
ol,ul{max-width:900px;padding-left:20px}li{margin:6px 0}code{background:var(--line);padding:1px 5px;border-radius:4px;font-size:12.5px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.mapwrap{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:16px;align-items:start}
canvas{width:100%;display:block;cursor:pointer}
.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:0 0 10px;font-size:13px}
.controls label{display:inline-flex;gap:6px;align-items:center;cursor:pointer}
select,button{background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:4px 8px;font:inherit;font-size:13px;max-width:100%}
button{cursor:pointer}.chips{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 0}.chips button{font-size:12px;padding:2px 8px}
.side{font-size:12.5px;max-height:820px;overflow:auto}.side h3{font-size:14px;margin:10px 0 4px}.side .muted{color:var(--ink2)}
.step{border-left:3px solid var(--hl);padding:4px 0 6px 10px;margin:6px 0}.step b{font-size:13px}
.br{color:var(--ink2);font-size:12px;margin-top:3px}.br i{color:var(--ink)}
.sw{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:4px;vertical-align:-1px}
table{border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:8px;font-size:12.5px;width:100%}
th,td{padding:5px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{color:var(--ink2);font-weight:600}
td.r,th.r{text-align:right;font-variant-numeric:tabular-nums}.tw{overflow-x:auto}
@media (max-width:960px){.mapwrap{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<h1>Trails between the books' topics <span style="font-weight:400;color:var(--ink2)">· variant <code>textonly</code></span></h1>
<p class="sub">On the book map, sections sit next to related sections, and following those neighbours reads like a
line of argument: school → indoctrination → state → democracy. This report turns those neighbourhoods into a
graph of links between the __NN__ book sections, shows its backbone, and lets you find the chain of strongest
links between any two topics, with the passages that bridge each step.</p>

<h2 class="sec">What was done</h2>
<ol>
 <li><strong>Links from the text itself.</strong> For every book chunk (embedded from its own text only), its
 __K__ most similar book chunks were taken. Every time a neighbour belongs to a different section, that counts
 as one link between the two sections.</li>
 <li><strong>Strength = lift:</strong> the number of links divided by the number expected if neighbours were
 picked at random (bigger sections would get more links by chance). Lift 10 = ten times more often than chance.
 Only links with lift ≥ __MINLIFT__ and at least __MINL__ neighbour pairs are kept: __NE__ links.</li>
 <li><strong>Backbone:</strong> the maximum spanning tree, the __NT__ strongest links that connect all sections
 without loops. It is the skeleton of the trails.</li>
 <li><strong>Paths:</strong> between two sections, the chain whose links are strongest overall (each step costs
 1 / lift, and the cheapest chain wins). For every step, the most similar pair of passages across it is shown:
 the actual “bridge” between the two topics.</li>
 <li><strong>Positions</strong> are the section centres on the textonly book map (same UMAP as the book map
 report). __MOVED__Distances on the map are only a guide; the links are computed in the full space.</li>
</ol>

<h2 class="sec">The trail map</h2>
<div class="controls">
 <span>From</span><select id="from"></select><span>to</span><select id="to"></select><button id="clear">Clear</button>
 <label><input type="checkbox" id="all" checked>All links</label><label><input type="checkbox" id="tree" checked>Backbone</label>
 <label>Min lift <input type="range" id="lift" min="2" max="15" step="0.5" value="__MINLIFT__"><span id="liftv">__MINLIFT__</span></label>
</div>
<div class="chips" id="chips"><span style="color:var(--ink2);font-size:12px">Try:</span></div>
<div class="mapwrap" style="margin-top:10px">
 <div class="panel"><canvas id="cv"></canvas></div>
 <div class="panel side" id="side"><div class="muted">Click a section to see its links, or pick “from” and “to” (or click two sections in a row) to find the trail between them.</div></div>
</div>
<p class="cap"><strong>How to read it:</strong> circles are book sections (<span class="sw" style="background:var(--b0)"></span><em>Anarchokapitalismus</em>,
<span class="sw" style="background:var(--b1)"></span><em>Anarchoagorismus</em>), sized by the number of chunks; faint dots are the chunks. Black
lines are the backbone, grey lines the other links; thicker = stronger. A found trail is drawn in yellow, and the
side panel lists its steps with the passages that connect them.</p>

<h2 class="sec">The backbone</h2>
<div class="tw"><table id="treeTab"><thead><tr><th>Link</th><th class="r">Lift</th><th class="r">Neighbour pairs</th><th>Strongest bridge</th></tr></thead><tbody></tbody></table></div>
<p class="cap"><strong>Reading it:</strong> the strongest links are __STRONG__: sections that continue each other's
argument. The most connected sections (hubs of the trail network) are __HUBS__. __NCROSS__ backbone links join
the two books, for example __CROSS__: these are the places where <em>Anarchokapitalismus</em> and
<em>Anarchoagorismus</em> talk about the same thing. Unconnected: __ISO__.</p>

<h2 class="sec">Conclusions</h2>
<ul>
 <li><strong>The books' topics form connected trails, not isolated islands.</strong> Chains such as school →
 indoctrination → democracy, money → cryptocurrency → crypto-anarchy → agora, or drugs → smugglers and dealers →
 liberation follow the books' own lines of argument, found from the text alone.</li>
 <li><strong>The hubs are the broad, summarising sections:</strong> __HUBS__. They draw together themes from
 across the books, so many trails pass through them; the specific chapters (money, drugs, school) sit at the
 ends of trails.</li>
 <li><strong>For UrzaLLM:</strong> the trails are a ready-made structure for the reasoning-tree interface. A
 debate that starts at one topic can follow the strongest links to the next argument, citing the bridging
 passages along the way.</li>
 <li><strong>Caveat:</strong> a link means the passages are similar in content, not that the author argues from
 one to the other; the direction of a trail is the reader's.</li>
</ul>
</div>
<script>
__LABELJS__
const N=__NODES__, E=__EDGES__, PT=__PTS__, SUG=__SUGG__, PAD=18;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const cv=document.getElementById("cv"),side=document.getElementById("side"),fromS=document.getElementById("from"),toS=document.getElementById("to");
let minLift=+document.getElementById("lift").value,showAll=true,showTree=true,sel=-1,pathN=[],pick=[],hover=-1;
[fromS,toS].forEach(s=>{s.innerHTML='<option value="-1">–</option>'+N.map(n=>`<option value="${n.id}">${esc(n.name)}</option>`).join("")});
fromS.onchange=toS.onchange=()=>route(+fromS.value,+toS.value);
document.getElementById("clear").onclick=()=>{fromS.value=toS.value=-1;pathN=[];pick=[];sel=-1;side.innerHTML='<div class="muted">Click a section, or pick “from” and “to”.</div>';draw()};
document.getElementById("all").onchange=e=>{showAll=e.target.checked;draw()};document.getElementById("tree").onchange=e=>{showTree=e.target.checked;draw()};
document.getElementById("lift").oninput=e=>{minLift=+e.target.value;document.getElementById("liftv").textContent=minLift;if(pathN.length)route(pathN[0],pathN[pathN.length-1]);else draw()};
const chips=document.getElementById("chips");SUG.forEach(([a,b])=>{const c=document.createElement("button");c.textContent=`${N[a].name} → ${N[b].name}`;c.onclick=()=>{fromS.value=a;toS.value=b;route(a,b)};chips.appendChild(c)});
const live=()=>E.filter(e=>e.lift>=minLift||e.tree);
function route(s,t){if(s<0||t<0||s===t){pathN=[];draw();return}const adj=N.map(()=>[]);live().forEach(e=>{if(e.lift<minLift)return;adj[e.a].push([e.b,1/e.lift,e]);adj[e.b].push([e.a,1/e.lift,e])});
 const d=N.map(()=>Infinity),prev=N.map(()=>null);d[s]=0;const done=new Set();
 while(true){let u=-1,best=Infinity;N.forEach((_,i)=>{if(!done.has(i)&&d[i]<best){best=d[i];u=i}});if(u<0||u===t)break;done.add(u);adj[u].forEach(([v,c,e])=>{if(d[u]+c<d[v]){d[v]=d[u]+c;prev[v]=[u,e]}})}
 if(d[t]===Infinity){pathN=[];side.innerHTML=`<h3>No trail</h3><div class="muted">No chain of links with lift ≥ ${minLift} joins these sections. Lower the minimum lift.</div>`;draw();return}
 const steps=[];let v=t;while(v!==s){const[u,e]=prev[v];steps.push([u,v,e]);v=u}steps.reverse();pathN=[s,...steps.map(x=>x[1])];
 side.innerHTML=`<h3>${esc(N[s].name)} → ${esc(N[t].name)}</h3><div class="muted">${steps.length} step${steps.length>1?"s":""}</div>`+steps.map(([u,v,e])=>{const br=e.bridges[0],fw=N[u].id===e.a;
  const [secU,txtU,secV,txtV]=fw?[br[1],br[2],br[3],br[4]]:[br[3],br[4],br[1],br[2]];
  return `<div class="step"><b>${esc(N[u].name)} → ${esc(N[v].name)}</b> <span class="muted">lift ${e.lift}, ${e.links} pairs</span>
  <div class="br"><i>${esc(secU)}:</i> ${esc(txtU)}…</div><div class="br"><i>${esc(secV)}:</i> ${esc(txtV)}…</div></div>`}).join("");draw()}
function nodeInfo(k){const n=N[k],es=live().filter(e=>(e.a===k||e.b===k)&&(e.lift>=minLift)).sort((x,y)=>y.lift-x.lift);
 side.innerHTML=`<h3>${esc(n.name)}</h3><div class="muted">${esc(n.book)} · ${n.n} chunks · ${es.length} links</div>`+es.map(e=>{const o=e.a===k?e.b:e.a,br=e.bridges[0],fw=e.a===k;
  return `<div class="step"><b>→ ${esc(N[o].name)}</b> <span class="muted">lift ${e.lift}${e.tree?" · backbone":""}</span><div class="br"><i>${esc(fw?br[3]:br[1])}:</i> ${esc(fw?br[4]:br[2])}…</div></div>`}).join("")}
const W=Math.max(...PT.map(p=>p[0])),H=Math.max(...PT.map(p=>p[1]));
function size(){const r=cv.getBoundingClientRect(),d=devicePixelRatio||1,h=(r.width-2*PAD)*H/W+2*PAD;cv.style.height=h+"px";cv.width=r.width*d;cv.height=h*d}
function xy(x,y){const d=devicePixelRatio||1,k=(cv.width-2*PAD*d)/W;return[PAD*d+x*k,cv.height-PAD*d-y*k]}
const rad=n=>(3+Math.sqrt(n.n)*1.6)*(devicePixelRatio||1);
function draw(){const g=cv.getContext("2d"),d=devicePixelRatio||1,col=[css("--b0"),css("--b1")],bg=css("--card");g.clearRect(0,0,cv.width,cv.height);
 PT.forEach(p=>{const[x,y]=xy(p[0],p[1]);g.globalAlpha=.12;g.fillStyle=col[p[2]];g.beginPath();g.arc(x,y,2*d,0,7);g.fill()});g.globalAlpha=1;
 const onPath=new Set();for(let i=1;i<pathN.length;i++)onPath.add([Math.min(pathN[i-1],pathN[i]),Math.max(pathN[i-1],pathN[i])].join("-"));
 const focus=sel>=0||pathN.length;
 E.forEach(e=>{const key=[Math.min(e.a,e.b),Math.max(e.a,e.b)].join("-");const isP=onPath.has(key),isSel=sel>=0&&(e.a===sel||e.b===sel);
  if(!isP&&!isSel){if(e.lift<minLift)return;if(e.tree?!showTree:!showAll)return}
  const[x1,y1]=xy(N[e.a].x,N[e.a].y),[x2,y2]=xy(N[e.b].x,N[e.b].y);
  g.strokeStyle=isP?css("--hl"):e.tree?css("--path"):css("--edge");g.globalAlpha=isP?1:focus&&!isSel?.12:(e.tree?.75:.35);
  g.lineWidth=(isP?4:Math.min(.6+e.lift*.15,3.2))*d;g.beginPath();g.moveTo(x1,y1);g.lineTo(x2,y2);g.stroke()});g.globalAlpha=1;
 N.forEach(n=>{const[x,y]=xy(n.x,n.y),r=rad(n),act=pathN.includes(n.id)||n.id===sel||n.id===hover;g.globalAlpha=focus&&!act&&!(sel>=0&&E.some(e=>(e.a===sel&&e.b===n.id||e.b===sel&&e.a===n.id)&&e.lift>=minLift))?.35:1;
  g.fillStyle=col[n.book==="Anarchokapitalismus"?0:1];g.strokeStyle=act?css("--hl"):bg;g.lineWidth=(act?3:1.5)*d;g.beginPath();g.arc(x,y,r,0,7);g.fill();g.stroke()});g.globalAlpha=1;
 const order=[...N].sort((a,b)=>(pathN.includes(b.id)||b.id===sel)-(pathN.includes(a.id)||a.id===sel)||b.n-a.n);
 placeLabels(g,order.map(n=>{const[x,y]=xy(n.x,n.y),imp=pathN.includes(n.id)||n.id===sel||n.id===hover;return{x,y,text:n.name,r:rad(n),bold:imp,color:imp?css("--ink"):css("--ink2")}}),d,bg,css("--ink"),css("--ink2"))}
function hit(e){const r=cv.getBoundingClientRect(),d=devicePixelRatio||1,mx=(e.clientX-r.left)*d,my=(e.clientY-r.top)*d;let best=-1,bd=Infinity;
 N.forEach(n=>{const[x,y]=xy(n.x,n.y),dd=Math.hypot(x-mx,y-my);if(dd<rad(n)+6*d&&dd<bd){bd=dd;best=n.id}});return best}
cv.onmousemove=e=>{const h=hit(e);if(h!==hover){hover=h;draw()}};
cv.onclick=e=>{const h=hit(e);if(h<0)return;pick.push(h);if(pick.length>2)pick=[h];
 if(pick.length===2){fromS.value=pick[0];toS.value=pick[1];sel=-1;route(pick[0],pick[1])}else{sel=h;pathN=[];nodeInfo(h);draw()}};
// backbone table
document.querySelector("#treeTab tbody").innerHTML=E.filter(e=>e.tree).sort((a,b)=>b.lift-a.lift).map(e=>{const br=e.bridges[0];
 return `<tr><td><span class="sw" style="background:var(--b${N[e.a].book==="Anarchokapitalismus"?0:1})"></span>${esc(N[e.a].name)} – <span class="sw" style="background:var(--b${N[e.b].book==="Anarchokapitalismus"?0:1})"></span>${esc(N[e.b].name)}</td>
 <td class="r">${e.lift}</td><td class="r">${e.links}</td><td style="color:var(--ink2);font-size:12px">${esc(br[2].slice(0,110))}… ↔ ${esc(br[4].slice(0,110))}…</td></tr>`}).join("");
addEventListener("resize",()=>{size();draw()});matchMedia("(prefers-color-scheme: dark)").addEventListener("change",draw);size();draw();
if(SUG.length){fromS.value=SUG[0][0];toS.value=SUG[0][1];route(SUG[0][0],SUG[0][1])}
</script></body></html>"""


if __name__ == "__main__":
    main()
