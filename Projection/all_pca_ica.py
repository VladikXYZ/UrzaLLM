"""One-off report: PCA and ICA of the full dataset (articles, books, videos), textonly embeddings.

    python Projection/all_pca_ica.py            # from the repo root

Companion to book_pca_ica.py (books only). Here there are no chapters for all chunks,
so axes are read from their extreme chunks, the video tags and book chapters that are
over-represented at the high end, and how much each axis is explained by the source.

Writes Projection/output/all_pca_ica_textonly.html  (+ _pca.png, _ica.png)
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA, FastICA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cumvar import cumvar_svg, needed  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "output"
VARIANT = "textonly"
SOURCES = ["article", "book", "video"]
SRC_LABEL = {"article": "Articles", "book": "Books", "video": "Videos"}
COLORS = {"article": "#2a78d6", "book": "#eb6834", "video": "#1baf7a"}   # validated slots 1–3
N_PCA_SHOW, N_PCA_AXES, N_ICA, SEED = 20, 6, 20, 42
TOP = 30               # chunks at an axis' high end used to describe it
ONE_SOURCE = 0.8       # a component whose top chunks are ≥ this share one source is a format/series component


def eta2(v, lab):
    """Share of an axis' variance explained by the source (0 = none, 1 = all)."""
    m = v.mean()
    return float(sum((lab == l).sum() * (v[lab == l].mean() - m) ** 2 for l in set(lab)) / ((v - m) ** 2).sum())


def same_source_rate(Q, src, k=10):
    d = ((Q[:, None, :] - Q[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    nn = np.argsort(d, axis=1)[:, :k]
    return {s: float((src[nn[src == s]] == s).mean()) for s in SOURCES}


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def describe(v, rows, src, alltags, nvid, top=TOP):
    """What sits at the high end of an axis: source mix, enriched video tags, book chapters, titles."""
    hi = np.argsort(-v)[:top]
    mix = Counter(src[hi])
    vt = Counter(t for i in hi if src[i] == "video" for t in set(rows[i]["tags"]))
    nv = max(sum(1 for i in hi if src[i] == "video"), 1)
    tags = [t for t, c in sorted(vt.items(), key=lambda tc: -(tc[1] / nv) / (alltags[tc[0]] / nvid)) if c >= 3][:4]
    chs = [c for c, _ in Counter(rows[i]["chapter"] for i in hi if src[i] == "book").most_common(2)]
    titles, seen = [], set()
    for i in hi:
        if rows[i]["doc_id"] not in seen:
            seen.add(rows[i]["doc_id"])
            titles.append((src[i], rows[i]["title"]))
        if len(titles) == 4:
            break
    return {"mix": {s: int(mix.get(s, 0)) for s in SOURCES}, "tags": tags, "chapters": chs, "titles": titles}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rows = pq.read_table(ROOT / "Preprocessing" / "output" / "chunks.parquet",
                         columns=["chunk_id", "doc_id", "source", "kind", "title", "year", "tags", "chapter",
                                  "subchapter", "text"]).to_pylist()
    X = np.load(ROOT / "Embedding" / f"results_{VARIANT}" / "embeddings.npy")
    ids = (ROOT / "Embedding" / f"results_{VARIANT}" / "chunk_ids.txt").read_text(encoding="utf-8").split()
    assert ids == [r["chunk_id"] for r in rows], "embeddings don't match chunks.parquet"
    src = np.array([r["source"] for r in rows])
    alltags = Counter(t for r in rows if r["source"] == "video" for t in set(r["tags"]))
    nvid = int((src == "video").sum())
    Xc = X - X.mean(0)

    # PCA
    pca = PCA(N_PCA_SHOW, random_state=SEED).fit(Xc)
    S = pca.transform(Xc)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(PCA(random_state=SEED).fit(Xc).explained_variance_ratio_)
    n50 = needed(cum, 0.5)
    pcs = []
    for k in range(N_PCA_AXES):
        pcs.append({"label": f"PC{k + 1}", "evr": float(evr[k]), "eta": eta2(S[:, k], src),
                    "means": {s: float(S[src == s, k].mean()) for s in SOURCES},
                    "low": describe(-S[:, k], rows, src, alltags, nvid),
                    "high": describe(S[:, k], rows, src, alltags, nvid)})

    # ICA
    ica = FastICA(N_ICA, whiten="unit-variance", random_state=SEED, max_iter=5000)
    Z = ica.fit_transform(Xc)
    Z *= np.sign(skew(Z, axis=0))
    comps = []
    for k in range(N_ICA):
        d = describe(Z[:, k], rows, src, alltags, nvid)
        share = max(d["mix"].values()) / TOP
        comps.append({"k": k, "kurt": float(kurtosis(Z[:, k])), "eta": eta2(Z[:, k], src), **d,
                      "theme": share < ONE_SOURCE,
                      "main_src": max(d["mix"], key=d["mix"].get)})
    comps.sort(key=lambda c: (not c["theme"], -c["kurt"]))
    for r, c in enumerate(comps):
        c["label"] = f"IC{r + 1}"
    themes = [c for c in comps if c["theme"]]
    ax_i, ax_j = themes[0], themes[1]

    # Source mixing on the 2D maps, against the UMAP numbers of the same variant
    rp = same_source_rate(S[:, :2], src)
    ri = same_source_rate(Z[:, [ax_i["k"], ax_j["k"]]], src)
    rates_path = OUT / f"rates_{VARIANT}.json"
    umap_rates = json.loads(rates_path.read_text(encoding="utf-8")) if rates_path.exists() else None

    pts = [[SOURCES.index(src[i]), rows[i]["title"],
            f"{SRC_LABEL[src[i]][:-1]} · {rows[i]['kind']}" + (f" · {rows[i]['year']}" if rows[i]["year"] else "")
            + (f" · {rows[i]['chapter']}" if src[i] == "book" else ""),
            rows[i]["text"][:180],
            [round(float(v), 4) for v in S[i, :N_PCA_AXES]],
            [round(float(Z[i, c["k"]]), 3) for c in comps]] for i in range(len(rows))]

    OUT.mkdir(exist_ok=True)
    png(S[:, 0], S[:, 1], src, f"PC1 ({evr[0]:.1%} of variance)", f"PC2 ({evr[1]:.1%})",
        "All chunks, PCA (textonly)", OUT / f"all_pca_ica_{VARIANT}_pca.png")
    png(Z[:, ax_i["k"]], Z[:, ax_j["k"]], src, f"{ax_i['label']}: {short(ax_i)}", f"{ax_j['label']}: {short(ax_j)}",
        "All chunks, ICA (textonly)", OUT / f"all_pca_ica_{VARIANT}_ica.png")
    html(pts, src, evr, n50, pcs, comps, ax_i, ax_j, rp, ri, umap_rates, cum, OUT / f"all_pca_ica_{VARIANT}.html")
    print(f"PCA: PC1 {evr[0]:.1%} (source eta2 {pcs[0]['eta']:.2f}), PC2 {evr[1]:.1%}; {n50} PCs for 50%")
    for c in comps:
        print(f"  {c['label']} theme={c['theme']} kurt={c['kurt']:.1f} mix={c['mix']} tags={c['tags'][:3]} "
              f"ch={c['chapters'][:1]} | {c['titles'][0][1][:40]}")
    print("same-source 10-NN, PCA map:", {s: f"{v:.0%}" for s, v in rp.items()},
          "ICA map:", {s: f"{v:.0%}" for s, v in ri.items()})
    print(f"Wrote {OUT / f'all_pca_ica_{VARIANT}.html'}")


def short(c):
    bits = c["tags"][:2] or c["chapters"][:1] or [c["titles"][0][1]]
    return ", ".join(bits)[:60]


def png(x, y, src, xl, yl, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, min(13, 10 * np.ptp(y) / np.ptp(x)) + 0.8), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for s in SOURCES:
        m = src == s
        ax.scatter(x[m], y[m], s=7, c=COLORS[s], alpha=0.55, linewidths=0, label=SRC_LABEL[s])
    ax.axhline(0, color="#e4e3df", lw=0.8, zorder=0)
    ax.axvline(0, color="#e4e3df", lw=0.8, zorder=0)
    ax.set_aspect("equal")
    ax.set_xlabel(xl, fontsize=9, color="#52514e")
    ax.set_ylabel(yl, fontsize=9, color="#52514e")
    ax.tick_params(colors="#52514e", labelsize=8, length=0)
    for sp in ax.spines.values():
        sp.set_color("#e4e3df")
    ax.legend(frameon=False, loc="upper left", markerscale=2.5, fontsize=9)
    ax.set_title(title, loc="left", fontsize=12)
    fig.savefig(path, dpi=150, facecolor="#fcfcfb", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def variance_svg(evr):
    w, h, pl, pb, pt = 640, 200, 36, 28, 12
    bw = (w - pl - 8) / len(evr)
    top = max(evr) * 1.15
    out = []
    for t in (0.02, 0.04, 0.06):
        if t < top:
            y = h - pb - (h - pb - pt) * t / top
            out.append(f'<line x1="{pl}" x2="{w - 8}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                       f'<text x="{pl - 6}" y="{y + 4:.1f}" text-anchor="end" class="ax">{t:.0%}</text>')
    for k, v in enumerate(evr):
        x = pl + k * bw + 2
        bh = (h - pb - pt) * v / top
        y = h - pb - bh
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw - 4:.1f}" height="{bh:.1f}" rx="3" fill="var(--s0)">'
                   f'<title>PC{k + 1}: {v:.1%} of variance</title></rect>')
        if k < 3:
            out.append(f'<text x="{x + (bw - 4) / 2:.1f}" y="{y - 4:.1f}" text-anchor="middle" class="lab">{v:.1%}</text>')
        if k % 2 == 0:
            out.append(f'<text x="{x + (bw - 4) / 2:.1f}" y="{h - pb + 16}" text-anchor="middle" class="ax">PC{k + 1}</text>')
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Variance explained per principal component" '
            f'style="width:100%;max-width:{w}px;height:auto">{"".join(out)}'
            f'<line x1="{pl}" x2="{w - 8}" y1="{h - pb}" y2="{h - pb}" class="base"/></svg>')


def mixbar(mix):
    """Small stacked bar of the source mix of an axis' top chunks (with a 2px surface gap)."""
    n = sum(mix.values()) or 1
    segs = "".join(f'<span style="flex:{mix[s]};background:var(--s{SOURCES.index(s)})" title="{SRC_LABEL[s]}: '
                   f'{mix[s]}"></span>' for s in SOURCES if mix[s])
    txt = " · ".join(f"{SRC_LABEL[s][0]} {mix[s]}" for s in SOURCES if mix[s])
    return f'<div class="mix">{segs}</div><div class="mixt">{txt}</div>'


def titles_html(ts):
    return "<br>".join(f'<span class="sw sm" style="background:var(--s{SOURCES.index(s)})"></span>{esc(t)}'
                       for s, t in ts[:3])


def html(pts, src, evr, n50, pcs, comps, ax_i, ax_j, rp, ri, umap_rates, cum, path):
    p = lambda v: f"{round(v * 100)}%"
    rnd = {s: ((src == s).sum() - 1) / (len(src) - 1) for s in SOURCES}
    pc_rows = "".join(
        f"<tr><td><strong>{c['label']}</strong></td><td>{c['evr']:.1%}</td><td>{c['eta']:.2f}</td>"
        f"<td>{titles_html(c['low']['titles'])}</td><td>{titles_html(c['high']['titles'])}</td></tr>" for c in pcs)
    ic_rows = "".join(
        f"<tr><td><strong>{c['label']}</strong><br><span class='tag {'th' if c['theme'] else 'fo'}'>"
        f"{'theme' if c['theme'] else 'format / series'}</span></td><td>{mixbar(c['mix'])}</td>"
        f"<td>{esc(', '.join(c['tags'])) or '–'}</td><td>{esc(', '.join(c['chapters'])) or '–'}</td>"
        f"<td>{titles_html(c['titles'])}</td><td>{c['kurt']:.1f}</td></tr>" for c in comps)
    themes = [c for c in comps if c["theme"]]
    formats = [c for c in comps if not c["theme"]]
    cmp_rows = (f"<tr><td>PCA map (PC1 × PC2)</td>" + "".join(f"<td>{p(rp[s])}</td>" for s in SOURCES) + "</tr>"
                + f"<tr><td>ICA map ({ax_i['label']} × {ax_j['label']})</td>" + "".join(f"<td>{p(ri[s])}</td>" for s in SOURCES) + "</tr>")
    if umap_rates:
        cmp_rows += (f"<tr><td>UMAP map, raw</td>" + "".join(f"<td>{p(umap_rates['raw_2d'][s])}</td>" for s in SOURCES) + "</tr>"
                     + f"<tr><td>UMAP map, source-centred</td>" + "".join(f"<td>{p(umap_rates['centred_2d'][s])}</td>" for s in SOURCES) + "</tr>")
    cmp_rows += "<tr><td>Expected at random</td>" + "".join(f"<td>{p(rnd[s])}</td>" for s in SOURCES) + "</tr>"
    fill = {
        "__N__": f"{len(pts):,}", "__NA__": f"{(src == 'article').sum():,}", "__NB__": f"{(src == 'book').sum():,}",
        "__NV__": f"{(src == 'video').sum():,}", "__N_ICA__": str(N_ICA), "__VARSVG__": variance_svg(evr),
        "__PC1__": f"{evr[0]:.1%}", "__PC2__": f"{evr[1]:.1%}", "__PC20__": f"{evr.sum():.0%}", "__N50__": str(n50), "__UNITS__": "chunks",
        "__CUMSVG__": cumvar_svg(cum, 'var(--s0)'), "__NPC__": str(len(cum)), "__CUM2__": f"{cum[1]:.0%}",
        "__N80__": str(needed(cum, 0.8)), "__N90__": str(needed(cum, 0.9)), "__N95__": str(needed(cum, 0.95)),
        "__PC1ETA__": f"{pcs[0]['eta']:.2f}", "__PC_ROWS__": pc_rows, "__IC_ROWS__": ic_rows,
        "__PC1MEANS__": ", ".join(f"{SRC_LABEL[s].lower()} {pcs[0]['means'][s]:+.2f}" for s in SOURCES),
        "__NTHEME__": str(len(themes)), "__NFORMAT__": str(len(formats)),
        "__THEMES__": "; ".join(f"{c['label']}: {esc(short(c))}" for c in themes),
        "__FORMATS__": "; ".join(f"{c['label']}: {esc(c['titles'][0][1][:45])} ({SRC_LABEL[c['main_src']].lower()})"
                                 for c in formats),
        "__AXI__": ax_i["label"], "__AXJ__": ax_j["label"], "__AXI_D__": esc(short(ax_i)), "__AXJ_D__": esc(short(ax_j)),
        "__CMP_ROWS__": cmp_rows,
        "__DATA__": json.dumps(pts, ensure_ascii=False).replace("</", "<\\/"),
        "__PCOPTS__": json.dumps([{"label": f"{c['label']} ({c['evr']:.1%})",
                                   "desc": f"low: {pc_end(c['low'])} … high: {pc_end(c['high'])}"} for c in pcs],
                                 ensure_ascii=False).replace("</", "<\\/"),
        "__ICOPTS__": json.dumps([{"label": f"{c['label']}{'' if c['theme'] else ' (format)'}",
                                   "desc": f"high: {short(c)}"} for c in comps], ensure_ascii=False).replace("</", "<\\/"),
        "__AXIDX__": json.dumps([comps.index(ax_i), comps.index(ax_j)]),
    }
    page = TEMPLATE
    for k, v in fill.items():
        page = page.replace(k, v)
    path.write_text(page, encoding="utf-8")


def pc_end(d):
    return ", ".join(t[:28] for _, t in d["titles"][:2])


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Full dataset PCA and ICA</title>
<style>
:root{--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--s0:#2a78d6;--s1:#eb6834;--s2:#1baf7a;--hl:#0b0b0b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835;--s0:#3987e5;--s1:#d95926;--s2:#199e70;--hl:#f0efec}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1200px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}p.sub,p.cap{color:var(--ink2);max-width:900px}p.cap{font-size:13px;margin:8px 2px 0}
h2.sec{font-size:17px;margin:28px 0 6px;padding-top:12px;border-top:1px solid var(--line)}
ol,ul{max-width:900px;padding-left:20px}li{margin:6px 0}
code{background:var(--line);padding:1px 5px;border-radius:4px;font-size:12.5px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
canvas{width:100%;display:block;cursor:crosshair}
.controls{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:0 0 10px}
.controls label{display:inline-flex;gap:6px;align-items:center;cursor:pointer}
select,input[type=search]{background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:4px 8px}
.sw{width:10px;height:10px;border-radius:50%;display:inline-block;flex:none}.sw.sm{width:8px;height:8px;margin-right:5px}
.axes{color:var(--ink2);font-size:12.5px;margin:6px 2px 0}
#tip{position:fixed;pointer-events:none;background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:8px;
 padding:8px 10px;font-size:12.5px;max-width:380px;box-shadow:0 4px 16px rgba(0,0,0,.12);display:none;z-index:9}
#tip .t{font-weight:600}#tip .m{color:var(--ink2);margin-top:4px}
table{border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:8px;font-size:12.5px}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--ink2);font-weight:600}td.r,th.r{text-align:right;font-variant-numeric:tabular-nums}
.tw{overflow-x:auto}
.mix{display:flex;gap:2px;width:120px;height:8px;border-radius:4px;overflow:hidden}.mix span{display:block}
.mixt{color:var(--ink2);font-size:11.5px;margin-top:3px;white-space:nowrap}
.tag{font-size:11px;border-radius:10px;padding:1px 7px;border:1px solid var(--line);color:var(--ink2);white-space:nowrap}
.tag.th{color:var(--ink)}
svg .grid{stroke:var(--line);stroke-width:1}svg .base{stroke:var(--ink2);stroke-width:1}
svg .ax{fill:var(--ink2);font-size:10px}svg .lab{fill:var(--ink);font-size:10.5px}
</style></head><body><div class="wrap">
<h1>Full dataset: linear projections (PCA and ICA) <span style="font-weight:400;color:var(--ink2)">· variant <code>textonly</code></span></h1>
<p class="sub">A one-off companion to the book-only PCA/ICA report, now for all __N__ chunks: __NA__ from articles,
__NB__ from the books, __NV__ from videos, each embedded from its own text only. PCA and ICA are linear: every axis
is one fixed direction in the 1,024-dimensional embedding space, so the axes can be read. Unlike the books, most
chunks have no chapter, so each axis is described by the chunks at its ends, the video tags and book chapters that
are over-represented there, and how much of it is explained by the source.</p>

<h2 class="sec">What was done</h2>
<ol>
 <li><strong>Centring.</strong> The average vector of all chunks was subtracted (no per-source correction: the
 source effect is part of what this report measures).</li>
 <li><strong>PCA</strong> finds the directions of largest variation, in order (PC1 the largest).</li>
 <li><strong>ICA</strong> (FastICA, __N_ICA__ components) finds directions whose scores are as independent and
 long-tailed as possible; each tends to pick out one theme or one kind of text. Signs were set so the long tail is
 positive.</li>
 <li><strong>Describing an axis.</strong> For the 30 chunks scoring highest (and for PCA also lowest): their mix of
 sources, the video tags most over-represented among them compared with all videos (shown if at least 3 of the
 videos carry the tag), the book chapters they come from, and the first titles.</li>
 <li><strong>Source effect.</strong> η² (eta squared) = share of an axis' variance explained by the source alone,
 from 0 (none) to 1 (the axis only tells articles, books and videos apart).</li>
</ol>

<h2 class="sec">PCA: how much does each direction explain?</h2>
<div class="panel">__VARSVG__</div>
<p class="cap"><strong>Reading it:</strong> PC1 explains __PC1__, PC2 __PC2__; the first 20 components explain
__PC20__, and __N50__ are needed for half. PC1 stands out more than in the books alone, and the table below shows
why: it is largely the kind of text.</p>
<p class="cap" style="margin-top:14px"><strong>Cumulative variance:</strong></p>
<div class="panel">__CUMSVG__</div>
<p class="cap"><strong>Reading it:</strong> the share of all differences between __UNITS__ explained by the first k
principal components together (out of __NPC__ components in total). Two components, the most a flat PCA map can
show, explain __CUM2__. Half needs __N50__ components, 80% __N80__, 90% __N90__ and 95% __N95__. The curve rises
slowly and without a clear elbow: no small set of directions captures most of the content.</p>
<div class="tw"><table><thead><tr><th>Axis</th><th class="r">Variance</th><th class="r">Source η²</th><th>Lowest-scoring texts</th><th>Highest-scoring texts</th></tr></thead>
<tbody>__PC_ROWS__</tbody></table></div>
<p class="cap"><strong>Reading it:</strong> PC1 has a source η² of __PC1ETA__: it mostly separates book prose
(low end) from video descriptions (high end); average PC1 scores: __PC1MEANS__. This is the source clustering found
in the UMAP reports, appearing as the single strongest linear direction. The further components are mostly
topical (small η²): capitalism and anarchist theory against practical life, crypto against
self-governance, and so on.</p>

<h2 class="sec">PCA map</h2>
<div class="controls" id="ctl0"><label>X <select id="pcx"></select></label><label>Y <select id="pcy"></select></label></div>
<div class="panel"><canvas id="cv0"></canvas><div class="axes" id="ax0"></div></div>
<p class="cap"><strong>What you see (PC1 × PC2):</strong> the sources line up along PC1: books on the left,
videos on the right, articles spread across the middle. PC2 runs across all three sources. Use the selectors to
view PC3–PC6, where the colours mix much more, and the search box to highlight a subject (try <em>drog</em>,
<em>škol</em>, <em>bitcoin</em>).</p>

<h2 class="sec">ICA: themes and formats</h2>
<p class="sub">The __N_ICA__ components, split into <strong>themes</strong> (their top chunks come from several
sources) and <strong>format / series</strong> components (at least 80% of their top chunks come from one source,
typically one recurring kind of post). Within each group they are ordered by kurtosis (higher = more focused).</p>
<div class="tw"><table><thead><tr><th>Component</th><th>Sources of top 30</th><th>Over-represented video tags</th><th>Book chapters</th><th>Top texts</th><th class="r">Kurtosis</th></tr></thead>
<tbody>__IC_ROWS__</tbody></table></div>
<p class="cap"><strong>Reading it:</strong> __NTHEME__ components are themes that run across articles, books and
videos: __THEMES__. They are the linear counterpart of the topic regions on the maps, and they connect the same
subject across sources. __NFORMAT__ components describe a kind of text rather than a subject: __FORMATS__.
Most are recurring formats (announcements, yearly notices, a video series, conference recordings) whose texts are
alike because of their template. A few may instead be subjects that only one source covers (check their top
texts): that is itself a coverage finding, not a defect.</p>

<h2 class="sec">ICA map</h2>
<div class="controls"><label>X <select id="icx"></select></label><label>Y <select id="icy"></select></label></div>
<div class="panel"><canvas id="cv1"></canvas><div class="axes" id="ax1"></div></div>
<p class="cap"><strong>What you see (default __AXI__ × __AXJ__):</strong> the two most focused themes:
__AXI__ (__AXI_D__) and __AXJ__ (__AXJ_D__). Most chunks sit near the origin; each theme reaches out as an arm,
and the arms are made of several colours: the theme is covered by more than one source. Pick a format component to
see a single-colour arm instead.</p>

<h2 class="sec">How much do sources mix on each map?</h2>
<p class="sub">Same-source share of each chunk's 10 nearest neighbours on the map (2D distance), compared with the
UMAP maps of the same embeddings and with chance.</p>
<div class="tw"><table><thead><tr><th>Map</th><th class="r">Articles</th><th class="r">Books</th><th class="r">Videos</th></tr></thead>
<tbody>__CMP_ROWS__</tbody></table></div>
<p class="cap"><strong>Reading it:</strong> lower is more mixed. A map built on the two focused ICA themes mixes
sources far more than PC1 × PC2, because it leaves out the source direction that dominates PC1.</p>

<h2 class="sec">Conclusions</h2>
<ul>
 <li><strong>The strongest linear direction is the kind of text.</strong> PC1 separates book prose from video
 descriptions (η² __PC1ETA__), confirming in linear form what the UMAP reports found: style and length are a large
 part of the embedding.</li>
 <li><strong>Beneath it, the directions are topical.</strong> From PC2 on, and in most ICA components, the source
 explains little; the axes are subjects.</li>
 <li><strong>ICA separates themes from formats.</strong> __NTHEME__ of __N_ICA__ components are cross-source themes,
 __NFORMAT__ are formats or series. The theme components show where Urza addresses the same idea in books,
 articles and videos; the format components identify template texts (announcements, notices, series intros) that a
 coverage measure may want to discount.</li>
 <li><strong>For the coverage map:</strong> scoring a political question along the theme components would show
 which of Urza's themes it touches; the format components are a tool for spotting boilerplate. As in the book
 report, components depend on the number chosen and the random start: a lens, not a fixed taxonomy.</li>
</ul>
</div><div id="tip"></div>
<script>
const P=__DATA__, PCO=__PCOPTS__, ICO=__ICOPTS__, AXIDX=__AXIDX__, PAD=16, SRC=["Articles","Books","Videos"];
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const tip=document.getElementById("tip");
let hover=-1, query="", on=[true,true,true];
const ctl=document.getElementById("ctl0");
SRC.forEach((s,i)=>{const l=document.createElement("label");l.innerHTML=`<input type="checkbox" checked><span class="sw" style="background:var(--s${i})"></span>${s}`;
 l.querySelector("input").onchange=e=>{on[i]=e.target.checked;drawAll()};ctl.appendChild(l)});
const q=document.createElement("input");q.type="search";q.placeholder="Highlight titles containing…";q.oninput=()=>{query=q.value.trim().toLowerCase();drawAll()};ctl.appendChild(q);
const MAPS=[{cv:document.getElementById("cv0"),key:4,sx:document.getElementById("pcx"),sy:document.getElementById("pcy"),opts:PCO,init:[0,1],ax:document.getElementById("ax0")},
            {cv:document.getElementById("cv1"),key:5,sx:document.getElementById("icx"),sy:document.getElementById("icy"),opts:ICO,init:AXIDX,ax:document.getElementById("ax1")}];
MAPS.forEach(m=>{[m.sx,m.sy].forEach((s,j)=>{m.opts.forEach((o,i)=>{const e=document.createElement("option");e.value=i;e.textContent=o.label;s.appendChild(e)});
 s.value=m.init[j];s.onchange=()=>{setup(m);draw(m)}});setup(m)});
function setup(m){const i=+m.sx.value,j=+m.sy.value;m.i=i;m.j=j;const xs=P.map(p=>p[m.key][i]),ys=P.map(p=>p[m.key][j]);
 m.x0=Math.min(...xs);m.x1=Math.max(...xs);m.y0=Math.min(...ys);m.y1=Math.max(...ys);
 m.asp=Math.min(1.3,Math.max(.45,(m.y1-m.y0)/(m.x1-m.x0)));
 m.ax.textContent=`X → ${m.opts[i].label}: ${m.opts[i].desc}   ·   Y ↑ ${m.opts[j].label}: ${m.opts[j].desc}`;size(m)}
function size(m){const cv=m.cv,r=cv.getBoundingClientRect(),d=devicePixelRatio||1,h=(r.width-2*PAD)*m.asp+2*PAD;cv.style.height=h+"px";cv.width=r.width*d;cv.height=h*d}
function xy(m,x,y){const d=devicePixelRatio||1,w=m.cv.width-2*PAD*d,h=m.cv.height-2*PAD*d,
 k=Math.min(w/(m.x1-m.x0),h/(m.y1-m.y0)),ox=(w-(m.x1-m.x0)*k)/2,oy=(h-(m.y1-m.y0)*k)/2;return[PAD*d+ox+(x-m.x0)*k,m.cv.height-PAD*d-oy-(y-m.y0)*k]}
const match=p=>query&&p[1].toLowerCase().includes(query);
function drawAll(){MAPS.forEach(draw)}
function draw(m){const cv=m.cv,g=cv.getContext("2d"),d=devicePixelRatio||1,col=[css("--s0"),css("--s1"),css("--s2")],bg=css("--card");
 g.clearRect(0,0,cv.width,cv.height);const[zx,zy]=xy(m,0,0);g.strokeStyle=css("--line");g.lineWidth=d;g.beginPath();g.moveTo(zx,0);g.lineTo(zx,cv.height);g.moveTo(0,zy);g.lineTo(cv.width,zy);g.stroke();
 P.forEach(p=>{if(!on[p[0]])return;const[x,y]=xy(m,p[m.key][m.i],p[m.key][m.j]);g.globalAlpha=query&&!match(p)?.08:.6;g.fillStyle=col[p[0]];g.beginPath();g.arc(x,y,2.4*d,0,7);g.fill()});
 g.globalAlpha=1;if(query)P.forEach(p=>{if(!on[p[0]]||!match(p))return;const[x,y]=xy(m,p[m.key][m.i],p[m.key][m.j]);g.fillStyle=col[p[0]];g.strokeStyle=bg;g.lineWidth=1.5*d;g.beginPath();g.arc(x,y,4*d,0,7);g.fill();g.stroke()});
 if(hover>=0){const p=P[hover],[x,y]=xy(m,p[m.key][m.i],p[m.key][m.j]);g.strokeStyle=bg;g.lineWidth=4*d;g.beginPath();g.arc(x,y,7*d,0,7);g.stroke();g.strokeStyle=css("--hl");g.lineWidth=2*d;g.beginPath();g.arc(x,y,7*d,0,7);g.stroke()}}
MAPS.forEach(m=>{const cv=m.cv;
 cv.onmousemove=e=>{const r=cv.getBoundingClientRect(),d=devicePixelRatio||1,mx=(e.clientX-r.left)*d,my=(e.clientY-r.top)*d;
  let best=-1,bd=(8*d)**2;P.forEach((p,i)=>{if(!on[p[0]])return;const[x,y]=xy(m,p[m.key][m.i],p[m.key][m.j]),dd=(x-mx)**2+(y-my)**2;if(dd<bd){bd=dd;best=i}});
  if(best!==hover){hover=best;drawAll()}if(best<0){tip.style.display="none";return}const p=P[best];
  tip.innerHTML=`<div class="t">${esc(p[1])}</div><div class="m">${esc(p[2])}</div><div class="m">${esc(p[3])}…</div>`;tip.style.display="block";
  const tw=tip.offsetWidth,th=tip.offsetHeight;tip.style.left=Math.min(e.clientX+14,innerWidth-tw-8)+"px";tip.style.top=Math.min(e.clientY+14,innerHeight-th-8)+"px"};
 cv.onmouseleave=()=>{hover=-1;tip.style.display="none";drawAll()}});
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
addEventListener("resize",()=>{MAPS.forEach(size);drawAll()});matchMedia("(prefers-color-scheme: dark)").addEventListener("change",drawAll);
drawAll();
</script></body></html>"""


if __name__ == "__main__":
    main()
