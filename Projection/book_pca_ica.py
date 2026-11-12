"""One-off report: linear projections (PCA, ICA) of the book chunks, textonly embeddings.

    python Projection/book_pca_ica.py            # from the repo root

UMAP and t-SNE (book_map.py) are non-linear and only keep neighbourhoods. PCA and ICA
are linear: every map axis is a fixed direction in the 1,024-dim embedding space, so
the axes themselves can be read as themes.

Writes Projection/output/book_pca_ica_textonly.html  (+ _pca.png, _ica.png)
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import f_oneway, kurtosis, skew
from sklearn.decomposition import PCA, FastICA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import LABEL_JS, place_labels  # noqa: E402
from book_map import BOOK_COLORS, chapter_group, esc  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "output"
VARIANT = "textonly"
N_PCA_SHOW = 20      # components in the variance chart
N_PCA_AXES = 6       # components selectable as map axes
N_ICA = 16           # independent components
SEED = 42


def spread_ratio(Q, grp, rng):
    """Median over chapters of: chapter spread / spread of random sets of the same size."""
    out = []
    for g, n in Counter(grp).items():
        if n < 3:
            continue
        ix = np.where(grp == g)[0]
        sp = lambda i: np.sqrt(((Q[i] - Q[i].mean(0)) ** 2).sum(1)).mean()
        out.append(sp(ix) / np.mean([sp(rng.choice(len(Q), n, replace=False)) for _ in range(50)]))
    return float(np.median(out)), sum(1 for r in out if r < 0.2), len(out)


def ends(v, grp, k=4):
    m = {g: v[grp == g].mean() for g in set(grp)}
    o = sorted(m, key=m.get)
    return o[:k], o[-k:][::-1]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rows = pq.read_table(ROOT / "Preprocessing" / "output" / "chunks.parquet",
                         columns=["chunk_id", "source", "book", "chapter", "subchapter", "text"]).to_pylist()
    E = np.load(ROOT / "Embedding" / f"results_{VARIANT}" / "embeddings.npy")
    ids = (ROOT / "Embedding" / f"results_{VARIANT}" / "chunk_ids.txt").read_text(encoding="utf-8").split()
    assert ids == [r["chunk_id"] for r in rows], "embeddings don't match chunks.parquet"
    idx = [i for i, r in enumerate(rows) if r["source"] == "book"]
    B, X = [rows[i] for i in idx], E[idx]
    grp = np.array([chapter_group(r) for r in B])
    book = np.array([r["book"] for r in B])
    Xc = X - X.mean(0)
    rng = np.random.default_rng(0)

    # PCA
    pca = PCA(N_PCA_SHOW, random_state=SEED).fit(Xc)
    S = pca.transform(Xc)
    evr = pca.explained_variance_ratio_
    full_pca = PCA(random_state=SEED).fit(Xc)
    n50 = int(np.searchsorted(np.cumsum(full_pca.explained_variance_ratio_), 0.5) + 1)
    pc_ends = [ends(S[:, k], grp) for k in range(N_PCA_AXES)]

    # ICA: with only 2 components ICA is just a rotated, rescaled PCA, so take more.
    ica = FastICA(N_ICA, whiten="unit-variance", random_state=SEED, max_iter=5000)
    Z = ica.fit_transform(Xc)
    Z *= np.sign(skew(Z, axis=0))             # heavy tail positive: high score = "about this theme"
    F = np.array([f_oneway(*[Z[grp == g, k] for g in set(grp)]).statistic for k in range(N_ICA)])
    order = list(np.argsort(-F))
    comps = []
    for rank, k in enumerate(order):
        m = {g: Z[grp == g, k].mean() for g in set(grp)}
        top = sorted(m, key=m.get, reverse=True)[:3]
        top20 = Counter(grp[np.argsort(-Z[:, k])[:20]]).most_common(1)[0]
        best = int(np.argmax(Z[:, k]))
        comps.append({"k": int(k), "label": f"IC{rank + 1}", "F": float(F[k]), "kurt": float(kurtosis(Z[:, k])),
                      "top": top, "single": top20[1] >= 16, "single_group": top20[0],
                      "best_path": " › ".join(x for x in (B[best]["chapter"], B[best]["subchapter"]) if x),
                      "best_text": B[best]["text"][:200]})
    # Default map axes: the two most chapter-separating *focused* themes (long tail, kurtosis > 2)
    # that aren't one chapter on its own.
    themed = [c for c in comps if not c["single"] and c["kurt"] > 2] or [c for c in comps if not c["single"]]
    ax_i, ax_j = themed[0], themed[1]

    # Tightness of chapters on the default maps, compared with the non-linear maps
    pca_tight = spread_ratio(S[:, :2], grp, rng)
    ica_tight = spread_ratio(Z[:, [ax_i["k"], ax_j["k"]]], grp, rng)
    stats_path = OUT / f"book_stats_{VARIANT}.json"
    nonlin = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}

    # Chapters (book order) and per-point data
    chapters, seen = [], {}
    for i, g in enumerate(grp):
        if g not in seen:
            seen[g] = len(chapters)
            chapters.append({"id": seen[g], "name": g, "book": book[i], "n": int((grp == g).sum())})
    pts = [[0 if book[i] == "Anarchokapitalismus" else 1, seen[grp[i]],
            " › ".join(x for x in (B[i]["chapter"], B[i]["subchapter"]) if x), B[i]["text"][:240],
            [round(float(v), 4) for v in S[i, :N_PCA_AXES]],
            [round(float(Z[i, c["k"]]), 3) for c in comps]] for i in range(len(B))]

    OUT.mkdir(exist_ok=True)
    png(S[:, 0], S[:, 1], book, grp, f"PC1 ({evr[0]:.1%} of variance)", f"PC2 ({evr[1]:.1%})",
        "Book chunks, PCA (textonly)", OUT / f"book_pca_ica_{VARIANT}_pca.png")
    png(Z[:, ax_i["k"]], Z[:, ax_j["k"]], book, grp, f"{ax_i['label']}: {', '.join(ax_i['top'][:2])} …",
        f"{ax_j['label']}: {', '.join(ax_j['top'][:2])} …", "Book chunks, ICA (textonly)",
        OUT / f"book_pca_ica_{VARIANT}_ica.png")
    html(pts, chapters, evr, n50, pc_ends, comps, ax_i, ax_j, pca_tight, ica_tight, nonlin, len(B),
         OUT / f"book_pca_ica_{VARIANT}.html")
    print(f"PCA: PC1 {evr[0]:.1%}, PC2 {evr[1]:.1%}, 2 PCs {evr[:2].sum():.1%}, {n50} PCs for 50%")
    print("PCA map median spread", round(pca_tight[0], 2), "| ICA map", round(ica_tight[0], 2))
    for c in comps:
        print(f"  {c['label']} F={c['F']:.1f} kurt={c['kurt']:.1f} single={c['single']} top={c['top']}")
    print(f"Wrote {OUT / f'book_pca_ica_{VARIANT}.html'}")


def png(x, y, book, grp, xl, yl, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10 * np.ptp(y) / np.ptp(x) + 0.8), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for b, col in BOOK_COLORS.items():
        m = book == b
        ax.scatter(x[m], y[m], s=10, c=col, alpha=0.6, linewidths=0, label=b)
    ax.axhline(0, color="#e4e3df", lw=0.8, zorder=0)
    ax.axvline(0, color="#e4e3df", lw=0.8, zorder=0)
    ax.set_aspect("equal")
    ax.set_xlabel(xl, fontsize=9, color="#52514e")
    ax.set_ylabel(yl, fontsize=9, color="#52514e")
    ax.tick_params(colors="#52514e", labelsize=8, length=0)
    for sp in ax.spines.values():
        sp.set_color("#e4e3df")
    ax.legend(frameon=False, loc="upper left", markerscale=2, fontsize=9)
    ax.set_title(title, loc="left", fontsize=12)
    place_labels(ax, [(x[grp == g].mean(), y[grp == g].mean(), g) for g, _ in Counter(grp).most_common()], fontsize=7)
    fig.savefig(path, dpi=150, facecolor="#fcfcfb", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def variance_svg(evr):
    """Bar chart of variance explained per principal component (inline SVG, single series)."""
    w, h, pl, pb, pt = 640, 200, 36, 28, 12
    bw = (w - pl - 8) / len(evr)
    top = max(evr) * 1.15
    bars = []
    for k, v in enumerate(evr):
        x = pl + k * bw + 2
        bh = (h - pb - pt) * v / top
        y = h - pb - bh
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw - 4:.1f}" height="{bh:.1f}" rx="3" '
                    f'fill="var(--b0)"><title>PC{k + 1}: {v:.1%} of variance</title></rect>')
        if k < 3:
            bars.append(f'<text x="{x + (bw - 4) / 2:.1f}" y="{y - 4:.1f}" text-anchor="middle" '
                        f'class="lab">{v:.1%}</text>')
        if k % 2 == 0:
            bars.append(f'<text x="{x + (bw - 4) / 2:.1f}" y="{h - pb + 16}" text-anchor="middle" '
                        f'class="ax">PC{k + 1}</text>')
    grid = "".join(f'<line x1="{pl}" x2="{w - 8}" y1="{h - pb - (h - pb - pt) * t / top:.1f}" '
                   f'y2="{h - pb - (h - pb - pt) * t / top:.1f}" class="grid"/>'
                   f'<text x="{pl - 6}" y="{h - pb - (h - pb - pt) * t / top + 4:.1f}" text-anchor="end" '
                   f'class="ax">{t:.0%}</text>' for t in (0.01, 0.02, 0.03, 0.04) if t < top)
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Variance explained per principal component" '
            f'style="width:100%;max-width:{w}px;height:auto">{grid}{"".join(bars)}'
            f'<line x1="{pl}" x2="{w - 8}" y1="{h - pb}" y2="{h - pb}" class="base"/></svg>')


def html(pts, chapters, evr, n50, pc_ends, comps, ax_i, ax_j, pca_tight, ica_tight, nonlin, n, path):
    q = lambda s: f"“{esc(s)}”"
    lst = lambda xs: ", ".join(q(x) for x in xs)
    pc_opts = [{"label": f"PC{k + 1} ({evr[k]:.1%})", "low": pc_ends[k][0], "high": pc_ends[k][1]}
               for k in range(N_PCA_AXES)]
    ic_opts = [{"label": c["label"], "top": c["top"]} for c in comps]
    ic_rows = "".join(
        f"<tr><td><strong>{c['label']}</strong></td><td>{lst(c['top'])}</td>"
        f"<td><span class='path'>{esc(c['best_path'])}</span><br>{esc(c['best_text'])}…</td>"
        f"<td>{c['F']:.1f}</td><td>{c['kurt']:.1f}</td></tr>" for c in comps)
    single = [c for c in comps if c["single"]]
    nl = ""
    if nonlin:
        nl = (f" For comparison, the non-linear maps of the same chunks: UMAP {nonlin['median_ratio']:.2f}, "
              f"t-SNE {nonlin.get('median_ratio_tsne', float('nan')):.2f}.")
    fill = {"__LABELJS__": LABEL_JS,
        "__N__": str(n), "__N_ICA__": str(N_ICA), "__VARSVG__": variance_svg(evr),
        "__PC1__": f"{evr[0]:.1%}", "__PC2__": f"{evr[1]:.1%}", "__PC12__": f"{evr[:2].sum():.1%}",
        "__PC20__": f"{evr.sum():.0%}", "__N50__": str(n50),
        "__PC1LOW__": lst(pc_ends[0][0]), "__PC1HIGH__": lst(pc_ends[0][1]),
        "__PC2LOW__": lst(pc_ends[1][0]), "__PC2HIGH__": lst(pc_ends[1][1]),
        "__PCA_TIGHT__": f"{pca_tight[0]:.2f}", "__PCA_TIGHTN__": f"{pca_tight[1]} of {pca_tight[2]}",
        "__ICA_TIGHT__": f"{ica_tight[0]:.2f}", "__ICA_TIGHTN__": f"{ica_tight[1]} of {ica_tight[2]}",
        "__NONLIN__": nl, "__IC_ROWS__": ic_rows,
        "__AXI__": ax_i["label"], "__AXJ__": ax_j["label"],
        "__AXI_TOP__": lst(ax_i["top"]), "__AXJ_TOP__": lst(ax_j["top"]),
        "__SINGLE__": ("; ".join(f"{c['label']} is driven mostly by {q(c['single_group'])}" for c in single)
                       + ", so it describes one chapter's style rather than a shared theme.") if single else "",
        "__THEMES__": "; ".join(f"{c['label']}: {lst(c['top'][:2])}" for c in comps if not c["single"]),
        "__DATA__": json.dumps(pts, ensure_ascii=False).replace("</", "<\\/"),
        "__CHAPTERS__": json.dumps(chapters, ensure_ascii=False).replace("</", "<\\/"),
        "__PCOPTS__": json.dumps(pc_opts, ensure_ascii=False).replace("</", "<\\/"),
        "__ICOPTS__": json.dumps(ic_opts, ensure_ascii=False).replace("</", "<\\/"),
        "__AXIDX__": json.dumps([comps.index(ax_i), comps.index(ax_j)]),
    }
    page = TEMPLATE
    for k, v in fill.items():
        page = page.replace(k, v)
    path.write_text(page, encoding="utf-8")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Book PCA and ICA</title>
<style>
:root{--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--b0:#2a78d6;--b1:#eb6834;--hl:#0b0b0b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835;--b0:#3987e5;--b1:#d95926;--hl:#f0efec}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1200px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}p.sub,p.cap{color:var(--ink2);max-width:900px}p.cap{font-size:13px;margin:8px 2px 0}
h2.sec{font-size:17px;margin:28px 0 6px;padding-top:12px;border-top:1px solid var(--line)}
ol,ul{max-width:900px;padding-left:20px}li{margin:6px 0}
code{background:var(--line);padding:1px 5px;border-radius:4px;font-size:12.5px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.mapwrap{display:grid;grid-template-columns:minmax(0,1fr) 260px;gap:16px;align-items:start}
canvas{width:100%;display:block;cursor:crosshair}
.side{max-height:760px;overflow:auto;font-size:12.5px}.side h3{font-size:13px;margin:0 0 6px}
.side div.c{padding:3px 6px;border-radius:5px;cursor:pointer;display:flex;gap:6px;align-items:center}
.side div.c:hover,.side div.c.on{background:var(--line)}.side .n{color:var(--ink2);margin-left:auto}
.controls{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:0 0 10px}
select{background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:4px 6px}
.sw{width:10px;height:10px;border-radius:50%;display:inline-block;flex:none}
.axes{color:var(--ink2);font-size:12.5px;margin:6px 2px 0}
#tip{position:fixed;pointer-events:none;background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:8px;
 padding:8px 10px;font-size:12.5px;max-width:380px;box-shadow:0 4px 16px rgba(0,0,0,.12);display:none;z-index:9}
#tip .t{font-weight:600}#tip .m{color:var(--ink2);margin-top:4px}
table{border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:8px;font-size:13px}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
td:nth-child(4),td:nth-child(5),th:nth-child(4),th:nth-child(5){text-align:right;font-variant-numeric:tabular-nums}
th{color:var(--ink2);font-weight:600}.path{color:var(--ink2);font-size:12px}
.tw{overflow-x:auto}
svg .grid{stroke:var(--line);stroke-width:1}svg .base{stroke:var(--ink2);stroke-width:1}
svg .ax{fill:var(--ink2);font-size:10px}svg .lab{fill:var(--ink);font-size:10.5px}
@media (max-width:860px){.mapwrap{grid-template-columns:1fr}.side{max-height:280px}}
</style></head><body><div class="wrap">
<h1>Books: linear projections (PCA and ICA) <span style="font-weight:400;color:var(--ink2)">· variant <code>textonly</code></span></h1>
<p class="sub">A one-off companion to the book map report. UMAP and t-SNE bend the space to keep neighbours
together; their axes mean nothing. PCA and ICA are <strong>linear</strong>: each axis is one fixed direction in
the 1,024-dimensional embedding space, so the axes themselves can be read as themes. Data: the __N__ book chunks,
embedded from their own text only (no chapter path, title or other context line).</p>

<h2 class="sec">What was done</h2>
<ol>
 <li><strong>Centring.</strong> The average book vector was subtracted from every chunk (both methods need it).</li>
 <li><strong>PCA</strong> (principal component analysis) finds the directions along which the chunks differ most,
 in order: PC1 is the single direction with the largest spread, PC2 the largest one at right angles to PC1, and so
 on. A PCA map shows the chunks' coordinates along two of these directions, at true scale.</li>
 <li><strong>ICA</strong> (independent component analysis, FastICA) looks for __N_ICA__ directions whose scores are
 as statistically independent and as non-Gaussian as possible. In text embeddings such a direction typically
 picks out a <em>theme</em>: most chunks score near zero, and the chunks about that theme score high. Each
 component's sign was set so that its long tail is positive. ICA with only 2 components would be a mere rotation
 of the first two PCA axes, which is why __N_ICA__ were computed and any two can be plotted.</li>
 <li><strong>Reading the axes.</strong> For every axis, the chapters with the highest and lowest average score
 name what it stands for. Chapter labels on the maps sit at the average position of their chunks.</li>
</ol>

<h2 class="sec">PCA: how much does each direction explain?</h2>
<div class="panel">__VARSVG__</div>
<p class="cap"><strong>Reading it:</strong> PC1 explains __PC1__ of the differences between chunks, PC2 __PC2__,
together __PC12__; the first 20 components explain __PC20__, and __N50__ components are needed for half. There is
no dominant direction: the books' content is spread over many independent dimensions. A 2-dimensional PCA map
therefore shows only a thin slice of the structure, which is why non-linear maps (UMAP, t-SNE) are needed for a
readable picture.</p>

<h2 class="sec">PCA map</h2>
<div class="controls"><label>X axis <select id="pcx"></select></label><label>Y axis <select id="pcy"></select></label>
<span><span class="sw" style="background:var(--b0)"></span> Anarchokapitalismus</span>
<span><span class="sw" style="background:var(--b1)"></span> Anarchoagorismus</span></div>
<div class="mapwrap">
 <div class="panel"><canvas id="cv0"></canvas><div class="axes" id="ax0"></div></div>
 <div class="panel side"><h3>Chapters (click to highlight)</h3><div id="list"></div></div>
</div>
<p class="cap"><strong>What you see (PC1 × PC2):</strong> PC1 runs from __PC1LOW__ (low) to __PC1HIGH__ (high):
from personal freedom, obedience and the future towards economics and how institutions work. It is essentially
the difference between the two books, which therefore sit side by side along PC1. PC2 runs from __PC2LOW__ (low)
to __PC2HIGH__ (high): from abstract arguments about the state and democracy towards concrete, practical or
narrative passages. Chapters overlap much more than on the UMAP map: their median spread is __PCA_TIGHT__ of a
random set (__PCA_TIGHTN__ chapters below 0.20).__NONLIN__ Use the selectors to look at further components.</p>

<h2 class="sec">ICA: independent themes</h2>
<p class="sub">The __N_ICA__ components, ordered by how strongly they separate chapters (F statistic of a one-way
analysis of variance across chapter groups: higher = the component's scores differ more between chapters than
within them). <strong>Kurtosis</strong> above 0 means a long tail: a few chunks score very high, i.e. a focused
theme.</p>
<div class="tw"><table><thead><tr><th>Component</th><th>Chapters scoring highest</th><th>Highest-scoring chunk</th><th>F</th><th>Kurtosis</th></tr></thead>
<tbody>__IC_ROWS__</tbody></table></div>
<p class="cap"><strong>Reading it:</strong> almost every component is a recognisable theme of the books, found
without any chapter information: __THEMES__. __SINGLE__ The themes cut across the two books where the subject
is shared (school appears in both), which the chapter structure alone does not show.</p>

<h2 class="sec">ICA map</h2>
<div class="controls"><label>X axis <select id="icx"></select></label><label>Y axis <select id="icy"></select></label></div>
<div class="panel"><canvas id="cv1"></canvas><div class="axes" id="ax1"></div></div>
<p class="cap"><strong>What you see (default __AXI__ × __AXJ__):</strong> the two focused themes (kurtosis
above 2) that separate chapters best, leaving out components that describe a single chapter. __AXI__ is high for __AXI_TOP__;
__AXJ__ for __AXJ_TOP__. Most chunks sit near the origin (0, 0), and each theme forms an arm reaching out
along its axis: an ICA map shows <em>how strongly</em> a chunk is about a theme rather than a landscape of
regions. Median chapter spread on this map: __ICA_TIGHT__ (__ICA_TIGHTN__ below 0.20). Pick other components
to see other themes; hovering a chunk highlights it on both maps.</p>

<h2 class="sec">Conclusions</h2>
<ul>
 <li><strong>The main linear direction is the difference between the books.</strong> PC1 separates the
 economic and institutional arguments of <em>Anarchokapitalismus</em> from the personal freedom and obedience
 themes of <em>Anarchoagorismus</em>, but explains only __PC1__ of the variation.</li>
 <li><strong>PCA is a poor map, but an honest one.</strong> With no dominant directions, any 2D linear view
 overlaps most chapters. The compact chapter regions of UMAP and t-SNE come from combining many directions,
 not from one or two.</li>
 <li><strong>ICA recovers the books' topics on its own.</strong> Without knowing the chapters, its components
 line up with them: money, school and obedience, drugs and guns, law and courts, the social system and so on.
 This is independent, linear evidence that the embedding encodes Urza's ideas as distinct directions.</li>
 <li><strong>ICA components are candidate “idea axes” for the coverage map.</strong> A political question
 could be scored along each component to see which of Urza's themes it touches. The components depend on the
 number chosen (__N_ICA__ here) and on the random start, so they are a lens, not a fixed taxonomy.</li>
</ul>
</div><div id="tip"></div>
<script>
__LABELJS__
const P=__DATA__, CH=__CHAPTERS__, PCO=__PCOPTS__, ICO=__ICOPTS__, AXIDX=__AXIDX__, PAD=16;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const tip=document.getElementById("tip");
let sel=-1, hover=-1;
const MAPS=[{cv:document.getElementById("cv0"),key:4,sx:document.getElementById("pcx"),sy:document.getElementById("pcy"),
             opts:PCO,init:[0,1],ax:document.getElementById("ax0"),
             desc:o=>`${o.label}: low = ${o.low.slice(0,2).join(", ")} … high = ${o.high.slice(0,2).join(", ")}`},
            {cv:document.getElementById("cv1"),key:5,sx:document.getElementById("icx"),sy:document.getElementById("icy"),
             opts:ICO,init:AXIDX,ax:document.getElementById("ax1"),desc:o=>`${o.label}: high = ${o.top.join(", ")}`}];
MAPS.forEach(m=>{[m.sx,m.sy].forEach((s,j)=>{m.opts.forEach((o,i)=>{const e=document.createElement("option");e.value=i;e.textContent=o.label;s.appendChild(e)});
 s.value=m.init[j];s.onchange=()=>{setup(m);draw(m)}});setup(m)});
const list=document.getElementById("list");
CH.forEach(c=>{const d=document.createElement("div");d.className="c";d.dataset.g=c.id;
 d.innerHTML=`<span class="sw" style="background:var(--b${c.book==="Anarchokapitalismus"?0:1})"></span>${esc(c.name)}<span class="n">${c.n}</span>`;
 d.onclick=()=>select(c.id);list.appendChild(d)});
function select(g){sel=sel===g?-1:g;document.querySelectorAll(".side div.c").forEach(d=>d.classList.toggle("on",+d.dataset.g===sel));drawAll()}
function setup(m){const i=+m.sx.value,j=+m.sy.value;m.i=i;m.j=j;
 const xs=P.map(p=>p[m.key][i]),ys=P.map(p=>p[m.key][j]);
 m.x0=Math.min(...xs);m.x1=Math.max(...xs);m.y0=Math.min(...ys);m.y1=Math.max(...ys);
 const asp=Math.min(1.4,Math.max(.45,(m.y1-m.y0)/(m.x1-m.x0)));m.asp=asp;
 m.ax.textContent=`X → ${m.desc(m.opts[i])}   ·   Y ↑ ${m.desc(m.opts[j])}`;
 const g={};P.forEach(p=>{const c=p[1];(g[c]=g[c]||[0,0,0]);g[c][0]+=p[m.key][i];g[c][1]+=p[m.key][j];g[c][2]++});
 m.lab=CH.map(c=>({...c,x:g[c.id][0]/g[c.id][2],y:g[c.id][1]/g[c.id][2]}));size(m)}
function size(m){const cv=m.cv,r=cv.getBoundingClientRect(),d=devicePixelRatio||1,h=(r.width-2*PAD)*m.asp+2*PAD;
 cv.style.height=h+"px";cv.width=r.width*d;cv.height=h*d}
function xy(m,x,y){const d=devicePixelRatio||1,w=m.cv.width-2*PAD*d,h=m.cv.height-2*PAD*d,
 k=Math.min(w/(m.x1-m.x0),h/(m.y1-m.y0)),ox=(w-(m.x1-m.x0)*k)/2,oy=(h-(m.y1-m.y0)*k)/2;
 return[PAD*d+ox+(x-m.x0)*k,m.cv.height-PAD*d-oy-(y-m.y0)*k]}
function drawAll(){MAPS.forEach(draw)}
function draw(m){const cv=m.cv,g=cv.getContext("2d"),d=devicePixelRatio||1,col=[css("--b0"),css("--b1")],bg=css("--card"),ink=css("--ink");
 g.clearRect(0,0,cv.width,cv.height);
 const[zx,zy]=xy(m,0,0);g.strokeStyle=css("--line");g.lineWidth=1*d;g.beginPath();g.moveTo(zx,0);g.lineTo(zx,cv.height);g.moveTo(0,zy);g.lineTo(cv.width,zy);g.stroke();
 P.forEach(p=>{const[x,y]=xy(m,p[m.key][m.i],p[m.key][m.j]);g.globalAlpha=sel<0?.65:(p[1]===sel?.95:.08);g.fillStyle=col[p[0]];
  g.beginPath();g.arc(x,y,(sel===p[1]?4:3)*d,0,7);g.fill()});g.globalAlpha=1;
 if(hover>=0){const p=P[hover],[x,y]=xy(m,p[m.key][m.i],p[m.key][m.j]);g.strokeStyle=bg;g.lineWidth=4*d;g.beginPath();g.arc(x,y,7*d,0,7);g.stroke();
  g.strokeStyle=css("--hl");g.lineWidth=2*d;g.beginPath();g.arc(x,y,7*d,0,7);g.stroke()}
 placeLabels(g,[...m.lab].sort((a,b)=>(b.id===sel)-(a.id===sel)||b.n-a.n).map(c=>{const[x,y]=xy(m,c.x,c.y);
  return{x,y,text:c.name,r:0,bold:c.id===sel,color:sel<0||c.id===sel?ink:css("--ink2")}}),d,bg,ink,css("--ink2"))}
MAPS.forEach(m=>{const cv=m.cv;
 cv.onmousemove=e=>{const r=cv.getBoundingClientRect(),d=devicePixelRatio||1,mx=(e.clientX-r.left)*d,my=(e.clientY-r.top)*d;
  let best=-1,bd=(9*d)**2;P.forEach((p,i)=>{const[x,y]=xy(m,p[m.key][m.i],p[m.key][m.j]),dd=(x-mx)**2+(y-my)**2;if(dd<bd){bd=dd;best=i}});
  if(best!==hover){hover=best;drawAll()}if(best<0){tip.style.display="none";return}const p=P[best];
  tip.innerHTML=`<div class="t">${esc(p[2])}</div><div class="m">${esc(p[3])}…</div>`;tip.style.display="block";
  const tw=tip.offsetWidth,th=tip.offsetHeight;tip.style.left=Math.min(e.clientX+14,innerWidth-tw-8)+"px";tip.style.top=Math.min(e.clientY+14,innerHeight-th-8)+"px"};
 cv.onmouseleave=()=>{hover=-1;tip.style.display="none";drawAll()};cv.onclick=()=>{if(hover>=0)select(P[hover][1])}});
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
addEventListener("resize",()=>{MAPS.forEach(size);drawAll()});matchMedia("(prefers-color-scheme: dark)").addEventListener("change",drawAll);
drawAll();
</script></body></html>"""


if __name__ == "__main__":
    main()
