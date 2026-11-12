"""Evolution over time: the source-centred UMAP map with a year slider (textonly embeddings).

    python Projection/timeline_map.py        # from the repo root (run project_compare.py --variant textonly first)

Articles and videos are dated (2015–2026). The two books have no year in the data (both
were published in 2026); they stay on the map as a constant reference layer and give
the concept regions their names: every article and video chunk is assigned to the book
section it is closest to in meaning.

Writes Projection/output/timeline_map_textonly.html (+ .png)
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import LABEL_JS, place_labels  # noqa: E402
from book_map import chapter_group, esc  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "output"
VARIANT = "textonly"
SOURCES = ["article", "book", "video"]
K_REGION = 5           # nearest book chunks that vote on a chunk's region
TOP_REGIONS = 6        # regions listed per year
FEW = 30               # years with fewer dated chunks than this are marked uncertain


def region_of(r):
    """Concept region = book chapter group; the broad '19. Praxe' is split into its subsections."""
    g = chapter_group(r)
    if g.startswith("19. Praxe") and r.get("subchapter"):
        return r["subchapter"]
    return g


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rows = pq.read_table(ROOT / "Preprocessing" / "output" / "chunks.parquet",
                         columns=["chunk_id", "source", "kind", "year", "date", "title", "book", "chapter",
                                  "subchapter", "text"]).to_pylist()
    ids = [r["chunk_id"] for r in rows]
    coords = pq.read_table(OUT / f"coords_{VARIANT}.parquet").to_pylist()
    assert [c["chunk_id"] for c in coords] == ids, "run project_compare.py --variant textonly first"
    Cvec = np.load(ROOT / "Embedding" / f"results_{VARIANT}" / "embeddings_source_centred.npy")
    P = np.array([[c["x_centred"], c["y_centred"]] for c in coords])
    P = (P - P.min(0)) / (P.max(0) - P.min(0)).max()
    src = np.array([r["source"] for r in rows])
    kind = np.array([r["kind"] for r in rows])
    year = np.array([r["year"] or 0 for r in rows])

    # Concept regions from the books
    bidx = np.where(src == "book")[0]
    breg = np.array([region_of(rows[i]) for i in bidx])
    region = np.array([""] * len(rows), dtype=object)
    region[bidx] = breg
    match = np.zeros(len(rows))
    other = np.where(src != "book")[0]
    S = Cvec[other] @ Cvec[bidx].T
    top = np.argsort(-S, axis=1)[:, :K_REGION]
    for n, i in enumerate(other):
        votes = Counter()
        for j in top[n]:
            votes[breg[j]] += S[n, j]
        region[i] = votes.most_common(1)[0][0]
        match[i] = S[n, top[n][0]]

    regions = sorted(set(breg), key=lambda g: int(bidx[breg == g][0]))
    rid = {g: k for k, g in enumerate(regions)}
    reg_info = [{"id": rid[g], "name": g, "book": rows[int(bidx[breg == g][0])]["book"],
                 "n": int((breg == g).sum()),
                 "x": float(P[bidx[breg == g], 0].mean()), "y": float(P[bidx[breg == g], 1].mean())} for g in regions]

    # Per-year statistics (articles + videos, announcements left out)
    dated = (src != "book") & (kind != "announcement") & (year > 0)
    years = sorted(set(year[dated]))
    ystats = []
    for y in years:
        m = dated & (year == y)
        regs = Counter(region[m])
        n = int(m.sum())
        ystats.append({
            "year": int(y), "n": n,
            "counts": {s: int((m & (src == s)).sum()) for s in ("article", "video")},
            "cx": float(P[m, 0].mean()), "cy": float(P[m, 1].mean()),
            "top": [[rid[g], round(c / n, 3)] for g, c in regs.most_common(TOP_REGIONS)],
            "few": n < FEW, "partial": bool(y == max(years)),
        })

    # Period comparison for the conclusions: first and last thirds of the years
    third = max(1, len(years) // 3)
    early, late = years[:third], years[-third:]

    def share(ys):
        m = dated & np.isin(year, ys)
        c = Counter(region[m])
        return {g: c[g] / m.sum() for g in c}, int(m.sum())
    se, ne = share(early)
    sl, nl = share(late)
    rising = sorted(regions, key=lambda g: sl.get(g, 0) - se.get(g, 0), reverse=True)[:5]
    falling = sorted(regions, key=lambda g: sl.get(g, 0) - se.get(g, 0))[:5]
    drift = float(np.hypot(ystats[-1]["cx"] - ystats[0]["cx"], ystats[-1]["cy"] - ystats[0]["cy"]))
    book_of = {r["name"]: r["book"] for r in reg_info}
    bshare = lambda sh: sum(v for g, v in sh.items() if book_of.get(g) == "Anarchoagorismus")
    for st in ystats:   # yearly share of chunks in Anarchoagorismus regions
        m = dated & (year == st["year"])
        st["anag"] = round(float(np.mean([book_of.get(g) == "Anarchoagorismus" for g in region[m]])), 3)
        for so in ("article", "video"):
            ms = m & (src == so)
            st[f"anag_{so}"] = (round(float(np.mean([book_of.get(g) == "Anarchoagorismus" for g in region[ms]])), 3)
                                if ms.sum() else None)
    anag_e, anag_l = bshare(se), bshare(sl)

    pts = [[round(float(P[i, 0]), 4), round(float(P[i, 1]), 4), SOURCES.index(src[i]), int(year[i]),
            1 if kind[i] == "announcement" else 0, rid.get(region[i], -1), rows[i]["title"],
            rows[i]["text"][:160], round(float(match[i]), 2)] for i in range(len(rows))]

    OUT.mkdir(exist_ok=True)
    png(P, src, year, dated, ystats, reg_info, OUT / f"timeline_map_{VARIANT}.png")
    html(pts, reg_info, ystats, early, late, se, sl, ne, nl, rising, falling, drift, OUT / f"timeline_map_{VARIANT}.html",
         anag_e, anag_l)
    print(f"Anarchoagorismus regions: {anag_e:.0%} -> {anag_l:.0%}", [(st["year"], st["anag"]) for st in ystats])
    print("years:", [(s["year"], s["n"]) for s in ystats])
    print("early", early, "→ late", late)
    print("rising:", [(g, f"{se.get(g, 0):.0%}→{sl.get(g, 0):.0%}") for g in rising])
    print("falling:", [(g, f"{se.get(g, 0):.0%}→{sl.get(g, 0):.0%}") for g in falling])
    print(f"Wrote {OUT / f'timeline_map_{VARIANT}.html'}")


def png(P, src, year, dated, ystats, reg_info, path):
    """Static overview: books as reference, dated chunks coloured by year, yearly centres as a trail."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 11), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    b = src == "book"
    ax.scatter(P[b, 0], P[b, 1], s=8, c="#eb6834", alpha=0.25, linewidths=0, label="Books (reference)")
    from matplotlib.colors import LinearSegmentedColormap
    blues = plt.get_cmap("Blues")
    cmap = LinearSegmentedColormap.from_list("years", [blues(0.3), blues(1.0)])   # earliest year still visible
    sc = ax.scatter(P[dated, 0], P[dated, 1], s=7, c=year[dated], cmap=cmap, vmin=min(year[dated]),
                    vmax=max(year[dated]), alpha=0.8, linewidths=0)
    cx = [s["cx"] for s in ystats]
    cy = [s["cy"] for s in ystats]
    ax.plot(cx, cy, color="#0b0b0b", lw=1.2)
    for s in ystats:
        ax.scatter([s["cx"]], [s["cy"]], s=26, c="#0b0b0b", zorder=3)
        ax.text(s["cx"] + 0.006, s["cy"] + 0.006, str(s["year"]), fontsize=8, color="#0b0b0b", zorder=4)
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("year (articles and videos)", fontsize=9, color="#52514e")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#e4e3df")
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title("Articles and videos by year on the source-centred map; black line = centre of each year",
                 loc="left", fontsize=11.5)
    place_labels(ax, [(r["x"], r["y"], r["name"]) for r in sorted(reg_info, key=lambda r: -r["n"])],
                 fontsize=6.5, color="#52514e")
    fig.savefig(path, dpi=150, facecolor="#fcfcfb", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def html(pts, reg_info, ystats, early, late, se, sl, ne, nl, rising, falling, drift, path, anag_e, anag_l):
    p = lambda v: f"{round(v * 100)}%"
    q = lambda s: f"“{esc(s)}”"
    span = lambda ys: f"{ys[0]}–{ys[-1]}" if len(ys) > 1 else str(ys[0])
    few = [s["year"] for s in ystats if s["few"]]
    chg = lambda gs: ", ".join(f"{q(g)} ({p(se.get(g, 0))} → {p(sl.get(g, 0))})" for g in gs)
    fill = {"__LABELJS__": LABEL_JS,
        "__DATA__": json.dumps(pts, ensure_ascii=False).replace("</", "<\\/"),
        "__REGIONS__": json.dumps(reg_info, ensure_ascii=False).replace("</", "<\\/"),
        "__YEARS__": json.dumps(ystats, ensure_ascii=False),
        "__Y0__": str(ystats[0]["year"]), "__Y1__": str(ystats[-1]["year"]),
        "__EARLY__": span(early), "__LATE__": span(late), "__NE__": f"{ne:,}", "__NL__": f"{nl:,}",
        "__RISING__": chg(rising), "__FALLING__": chg(falling),
        "__FEW__": (", ".join(map(str, few)) + f" (fewer than {FEW} dated chunks)") if few else "none",
        "__FEWN__": str(FEW), "__K__": str(K_REGION), "__NREG__": str(len(reg_info)),
        "__ANAG_E__": p(anag_e), "__ANAG_L__": p(anag_l),
        "__ANAG_TABLE__": (
            "<div class='tw'><table><thead><tr><th>Year</th>" + "".join(f"<th>{st['year']}</th>" for st in ystats)
            + "</tr></thead><tbody>"
            + "".join(f"<tr><td>{lab}</td>" + "".join(
                f"<td>{p(st[key]) if st[key] is not None else '–'}{'*' if key != 'anag' and st['counts'][key[5:]] < FEW and st[key] is not None else ''}</td>"
                for st in ystats) + "</tr>"
                for lab, key in (("All", "anag"), ("Articles", "anag_article"), ("Videos", "anag_video")))
            + "</tbody></table></div>"),
    }
    page = TEMPLATE
    for k, v in fill.items():
        page = page.replace(k, v)
    path.write_text(page, encoding="utf-8")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ideas over time</title>
<style>
:root{--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--s0:#2a78d6;--s1:#eb6834;--s2:#1baf7a;--hl:#0b0b0b;--dim:#c9c8c3}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835;--s0:#3987e5;--s1:#d95926;--s2:#199e70;--hl:#f0efec;--dim:#4a4a46}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1240px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}p.sub,p.cap{color:var(--ink2);max-width:900px}p.cap{font-size:13px;margin:8px 2px 0}
h2.sec{font-size:17px;margin:28px 0 6px;padding-top:12px;border-top:1px solid var(--line)}
ol,ul{max-width:900px;padding-left:20px}li{margin:6px 0}
code{background:var(--line);padding:1px 5px;border-radius:4px;font-size:12.5px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.mapwrap{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:16px;align-items:start}
canvas{width:100%;display:block;cursor:crosshair}
.controls{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:0 0 10px}
.controls label{display:inline-flex;gap:6px;align-items:center;cursor:pointer}
button{background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:4px 12px;cursor:pointer;font:inherit}
input[type=range]{flex:1;min-width:220px;accent-color:var(--s0)}
.yearlab{font-size:22px;font-weight:600;min-width:120px;font-variant-numeric:tabular-nums}
.sw{width:10px;height:10px;border-radius:50%;display:inline-block;flex:none}
.side h3{font-size:14px;margin:0 0 4px}.side .muted{color:var(--ink2);font-size:12.5px}
.reg{display:grid;grid-template-columns:1fr 44px;gap:2px 8px;align-items:center;font-size:12.5px;margin:8px 0 0}
.bar{height:8px;border-radius:0 4px 4px 0;background:var(--s0);margin-top:2px}
.vol{display:flex;gap:3px;align-items:flex-end;height:70px;margin-top:6px}
.vol .y{flex:1;display:flex;flex-direction:column;justify-content:flex-end;gap:2px;cursor:pointer;position:relative}
.vol .y span{display:block;border-radius:3px 3px 0 0}.vol .y.on{outline:2px solid var(--ink);outline-offset:2px;border-radius:3px}
.voll{display:flex;gap:3px;font-size:10px;color:var(--ink2)}.voll span{flex:1;text-align:center}
table{border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:8px;font-size:12.5px;font-variant-numeric:tabular-nums;margin-top:8px}
th,td{padding:5px 9px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600}tr:last-child td{border-bottom:0}.tw{overflow-x:auto}
#tip{position:fixed;pointer-events:none;background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:8px;
 padding:8px 10px;font-size:12.5px;max-width:380px;box-shadow:0 4px 16px rgba(0,0,0,.12);display:none;z-index:9}
#tip .t{font-weight:600}#tip .m{color:var(--ink2);margin-top:4px}
@media (max-width:900px){.mapwrap{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<h1>Ideas over time <span style="font-weight:400;color:var(--ink2)">· source-centred UMAP, variant <code>textonly</code></span></h1>
<p class="sub">Where in the space of ideas was Urza in each year? The map is the same source-centred UMAP map as
in the textonly projection report. Move the slider (or press play) to light up one year's articles and videos.
The books stay on the map as a fixed reference: they carry no year in the data (both editions were published in
2026), but their chapters name the regions of the map.</p>

<h2 class="sec">What was done</h2>
<ol>
 <li><strong>Map.</strong> The coordinates are those of the source-centred UMAP map of all chunks (textonly
 embeddings, each source's average vector subtracted), so this map matches the one in the projection report.</li>
 <li><strong>Concept regions.</strong> Every article and video chunk was given the region of the book section
 closest to it in meaning: its __K__ most similar book chunks (source-centred vectors) vote, weighted by
 similarity. There are __NREG__ regions: the book chapters, with the appendix and AMEN split into their topics and
 the broad “19. Praxe” split into its subsections. Region labels sit at the average position of the book chunks.</li>
 <li><strong>Years.</strong> Articles and videos carry their publication year (__Y0__–__Y1__). Announcements
 (short event notices) are shown but left out of all yearly numbers. For each year, the black trail marks the
 average position of its chunks: the year's centre of gravity.</li>
</ol>

<h2 class="sec">The map</h2>
<div class="panel" style="margin-bottom:10px">
 <div class="controls"><button id="play">▶ Play</button><span class="yearlab" id="ylab">All years</span>
  <input type="range" id="slider" min="0" max="1" step="1" value="0"></div>
 <div class="controls" style="margin:0">
  <label><input type="checkbox" id="books" checked><span class="sw" style="background:var(--s1)"></span>Books (reference)</label>
  <label><input type="checkbox" id="trail" checked>Yearly centres</label>
  <label><input type="checkbox" id="labels" checked>Region labels</label>
  <label><input type="checkbox" id="cum">Show earlier years too</label>
  <span><span class="sw" style="background:var(--s0)"></span> Articles</span>
  <span><span class="sw" style="background:var(--s2)"></span> Videos</span></div>
</div>
<div class="mapwrap">
 <div class="panel"><canvas id="cv"></canvas></div>
 <div class="panel side"><h3 id="sy">All years</h3><div class="muted" id="scount"></div>
  <div class="muted" style="margin-top:10px">Most frequent concept regions</div><div class="reg" id="sreg"></div>
  <div class="muted" style="margin-top:14px">Dated chunks per year (click a year)</div>
  <div class="vol" id="vol"></div><div class="voll" id="voll"></div>
  <div class="muted" style="margin-top:4px"><span class="sw" style="background:var(--s0)"></span> articles &nbsp;
  <span class="sw" style="background:var(--s2)"></span> videos</div></div>
</div>
<p class="cap"><strong>How to read it:</strong> grey dots are all other chunks, orange the books. The selected
year's articles (blue) and videos (green) are drawn on top; announcements are hollow. The side panel lists the
regions that year's chunks fall into most often. Hover a dot for its title and region; “Show earlier years too”
accumulates the years up to the one selected.</p>

<h2 class="sec">How the focus moved</h2>
<p class="sub">Shares of concept regions in the first years (__EARLY__, __NE__ chunks) against the last years
(__LATE__, __NL__ chunks), articles and videos together, announcements excluded.</p>
<ul>
 <li><strong>Grew the most:</strong> __RISING__.</li>
 <li><strong>Shrank the most:</strong> __FALLING__.</li>
 <li><strong>From one book's ideas towards the other's.</strong> Share of chunks whose nearest book section is in
 <em>Anarchoagorismus</em> (personal freedom, obedience, self-development) rather than <em>Anarchokapitalismus</em>
 (institutions, economics): __ANAG_E__ in __EARLY__ against __ANAG_L__ in __LATE__.</li>
</ul>
__ANAG_TABLE__
<p class="cap">* fewer than __FEWN__ chunks of that source in that year.</p>
<p class="cap"><strong>Reading it:</strong> the regions that grew are those of the free school Urza co-founded
(“Ježek bez klece”) and of the personal-freedom chapters of <em>Anarchoagorismus</em>; the ones that shrank are
the institutional topics of <em>Anarchokapitalismus</em> (work, roads, guns) and the stories, which early
narrative blog posts resembled. The shift towards <em>Anarchoagorismus</em> happens around 2021 and appears
<strong>within articles and within videos separately</strong>, so it is not just the move from articles to
videos that happened at the same time.</p>

<h2 class="sec">Caveats</h2>
<ul>
 <li><strong>The source mix changes over time.</strong> Until 2020 most dated chunks are articles; from 2021 most
 are videos (see the volume chart). The map is source-centred, which removes most of the difference in how the
 two sources embed, but part of any change can still be a change of format rather than of subject.</li>
 <li><strong>Videos are represented by their descriptions</strong>, not by what is said in them, so their region
 reflects the summary Urza wrote.</li>
 <li><strong>Regions are the nearest book section, not a topic label.</strong> A chunk about something the books
 do not cover still gets the closest section (hover shows how close: the best similarity to a book chunk).
 Narrative or personal posts tend to land near the books' stories and personal chapters.</li>
 <li><strong>Uncertain years:</strong> __FEW__. The last year (__Y1__) is incomplete.</li>
</ul>
</div><div id="tip"></div>
<script>
__LABELJS__
const P=__DATA__, REG=__REGIONS__, YS=__YEARS__, PAD=14;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const cv=document.getElementById("cv"), tip=document.getElementById("tip"), slider=document.getElementById("slider");
let idx=0, hover=-1, timer=null;
const opt={books:true,trail:true,labels:true,cum:false};
["books","trail","labels","cum"].forEach(k=>document.getElementById(k).onchange=e=>{opt[k]=e.target.checked;draw()});
slider.max=YS.length;slider.oninput=()=>{idx=+slider.value;update()};
document.getElementById("play").onclick=()=>{if(timer){stop();return}document.getElementById("play").textContent="❚❚ Pause";
 if(idx>=YS.length)idx=0;timer=setInterval(()=>{idx=idx>=YS.length?1:idx+1;slider.value=idx;update();if(idx===YS.length)stop()},1300)};
function stop(){clearInterval(timer);timer=null;document.getElementById("play").textContent="▶ Play"}
const W=Math.max(...P.map(p=>p[0])),H=Math.max(...P.map(p=>p[1]));
// volume chart
const vmax=Math.max(...YS.map(s=>s.n)),vol=document.getElementById("vol"),voll=document.getElementById("voll");
YS.forEach((s,k)=>{const d=document.createElement("div");d.className="y";d.title=`${s.year}: ${s.counts.article} articles, ${s.counts.video} videos`;
 d.innerHTML=`<span style="height:${60*s.counts.video/vmax}px;background:var(--s2)"></span><span style="height:${60*s.counts.article/vmax}px;background:var(--s0)"></span>`;
 d.onclick=()=>{idx=k+1;slider.value=idx;update()};vol.appendChild(d);
 const l=document.createElement("span");l.textContent=k%2===0?String(s.year).slice(2):"";voll.appendChild(l)});
function size(){const r=cv.getBoundingClientRect(),d=devicePixelRatio||1,h=(r.width-2*PAD)*H/W+2*PAD;cv.style.height=h+"px";cv.width=r.width*d;cv.height=h*d}
function xy(x,y){const d=devicePixelRatio||1,k=(cv.width-2*PAD*d)/W;return[PAD*d+x*k,cv.height-PAD*d-y*k]}
const yr=()=>idx===0?null:YS[idx-1].year;
const active=p=>{if(p[2]===1)return false;const y=yr();if(y===null)return true;return opt.cum?p[3]<=y&&p[3]>0:p[3]===y};
function update(){const y=yr();document.getElementById("ylab").textContent=y===null?"All years":y;
 const s=y===null?null:YS[idx-1];document.getElementById("sy").textContent=y===null?"All years":`${y}${s.partial?" (incomplete)":""}${s.few?" · few chunks":""}`;
 const sel=P.filter(p=>p[2]!==1&&p[4]===0&&(y===null?p[3]>0:(opt.cum?p[3]<=y&&p[3]>0:p[3]===y)));
 const na=sel.filter(p=>p[2]===0).length,nv=sel.filter(p=>p[2]===2).length;
 document.getElementById("scount").textContent=`${na} article chunks, ${nv} video chunks (announcements excluded)`;
 const c={};sel.forEach(p=>c[p[5]]=(c[p[5]]||0)+1);const top=Object.entries(c).sort((a,b)=>b[1]-a[1]).slice(0,6);
 const mx=top.length?top[0][1]/sel.length:1;
 document.getElementById("sreg").innerHTML=top.map(([r,n])=>`<div>${esc(REG[r].name)}<div class="bar" style="width:${100*(n/sel.length)/mx}%"></div></div><div style="text-align:right;font-variant-numeric:tabular-nums">${Math.round(100*n/sel.length)}%</div>`).join("");
 document.querySelectorAll(".vol .y").forEach((d,k)=>d.classList.toggle("on",k===idx-1));draw()}
function draw(){const g=cv.getContext("2d"),d=devicePixelRatio||1,col=[css("--s0"),css("--s1"),css("--s2")],bg=css("--card"),dim=css("--dim"),ink=css("--ink");
 g.clearRect(0,0,cv.width,cv.height);const y=yr();
 P.forEach(p=>{if(p[2]===1||active(p))return;const[x,yy]=xy(p[0],p[1]);g.globalAlpha=.35;g.fillStyle=dim;g.beginPath();g.arc(x,yy,2*d,0,7);g.fill()});
 if(opt.books)P.forEach(p=>{if(p[2]!==1)return;const[x,yy]=xy(p[0],p[1]);g.globalAlpha=.35;g.fillStyle=col[1];g.beginPath();g.arc(x,yy,2.3*d,0,7);g.fill()});
 P.forEach(p=>{if(!active(p))return;const[x,yy]=xy(p[0],p[1]);const recent=y===null||p[3]===y;g.globalAlpha=recent?.85:.35;
  if(p[4]){g.strokeStyle=col[p[2]];g.lineWidth=1.2*d;g.beginPath();g.arc(x,yy,2.6*d,0,7);g.stroke()}else{g.fillStyle=col[p[2]];g.beginPath();g.arc(x,yy,2.8*d,0,7);g.fill()}});
 g.globalAlpha=1;
 if(opt.labels)placeLabels(g,[...REG].sort((a,b)=>b.n-a.n).map(r=>{const[x,yy]=xy(r.x,r.y);return{x,y:yy,text:r.name,r:0,color:css("--ink2")}}),d,bg,css("--ink"),css("--ink2"),{size:10.5});
 if(opt.trail){const upto=y===null?YS.length:idx;g.strokeStyle=ink;g.lineWidth=1.5*d;g.beginPath();
  YS.slice(0,upto).forEach((s,k)=>{const[x,yy]=xy(s.cx,s.cy);k?g.lineTo(x,yy):g.moveTo(x,yy)});g.stroke();
  g.font=`600 ${11*d}px system-ui,sans-serif`;g.textAlign="left";g.textBaseline="bottom";
  YS.slice(0,upto).forEach((s,k)=>{const[x,yy]=xy(s.cx,s.cy),cur=k===upto-1&&y!==null;g.fillStyle=bg;g.beginPath();g.arc(x,yy,(cur?6:4)*d,0,7);g.fill();
   g.fillStyle=ink;g.beginPath();g.arc(x,yy,(cur?4.5:3)*d,0,7);g.fill();if(cur||y===null){g.fillText(s.year,x+6*d,yy-3*d)}})}
 if(hover>=0){const[x,yy]=xy(P[hover][0],P[hover][1]);g.strokeStyle=bg;g.lineWidth=4*d;g.beginPath();g.arc(x,yy,7*d,0,7);g.stroke();g.strokeStyle=css("--hl");g.lineWidth=2*d;g.beginPath();g.arc(x,yy,7*d,0,7);g.stroke()}}
cv.onmousemove=e=>{const r=cv.getBoundingClientRect(),d=devicePixelRatio||1,mx=(e.clientX-r.left)*d,my=(e.clientY-r.top)*d;
 let best=-1,bd=(8*d)**2;P.forEach((p,i)=>{if(!(active(p)||(p[2]===1&&opt.books)))return;const[x,y]=xy(p[0],p[1]),dd=(x-mx)**2+(y-my)**2;if(dd<bd){bd=dd;best=i}});
 if(best!==hover){hover=best;draw()}if(best<0){tip.style.display="none";return}const p=P[best],src=["Article","Book","Video"][p[2]];
 tip.innerHTML=`<div class="t">${esc(p[6])}</div><div class="m">${src}${p[3]?" · "+p[3]:""}${p[4]?" · announcement":""} · region: ${p[5]>=0?esc(REG[p[5]].name):"–"}${p[2]!==1?` (similarity ${p[8]})`:""}</div><div class="m">${esc(p[7])}…</div>`;
 tip.style.display="block";const tw=tip.offsetWidth,th=tip.offsetHeight;tip.style.left=Math.min(e.clientX+14,innerWidth-tw-8)+"px";tip.style.top=Math.min(e.clientY+14,innerHeight-th-8)+"px"};
cv.onmouseleave=()=>{hover=-1;tip.style.display="none";draw()};
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
addEventListener("resize",()=>{size();draw()});matchMedia("(prefers-color-scheme: dark)").addEventListener("change",draw);
size();update();
</script></body></html>"""


if __name__ == "__main__":
    main()
