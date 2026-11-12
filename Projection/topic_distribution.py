"""Topic distribution: which book concepts do the articles and videos cover? (textonly embeddings)

    python Projection/topic_distribution.py        # from the repo root

1. The book chunks define the concept space: PCA fitted on the books only, keeping the
   directions that beat a random baseline (broken-stick rule).
2. Concept centres: the average of each book section's chunks in that space.
3. Articles and videos are projected into the same space and assigned to the nearest
   centre (cosine). How well each text fits the books at all is measured separately, in the full space.
Each source's own average vector is subtracted first, so differences in style between
books, articles and videos don't decide the assignment.

Writes Projection/output/topic_distribution_textonly.html
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from book_map import esc  # noqa: E402
from timeline_map import region_of  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "output"
VARIANT = "textonly"
N_WORST = 15           # worst-fitting texts listed
K_ALT = (50, 184)      # alternative space sizes for the robustness check
FRAMING = {"Tři příběhy", "Pár slov úvodem", "Předmluva", "Doslov", "Závěr; a co dál?", "Anarchokapitalismus",
           "AMEN: aforismy"}   # sections that frame the books rather than discuss one topic


def unit(A):
    return A / np.linalg.norm(A, axis=-1, keepdims=True)


def broken_stick_k(evr):
    r = len(evr)
    bs = np.array([sum(1 / j for j in range(i, r + 1)) / r for i in range(1, r + 1)])
    return int(np.argmax(evr < bs))


def assign(X, b, o, grp, groups, k):
    """Fit PCA (k comps) on book rows b; return labels/similarities for rows o and book validation."""
    pca = PCA(k, random_state=0).fit(X[b])
    B, O = pca.transform(X[b]), pca.transform(X[o])
    captured = np.linalg.norm(O, axis=1) / np.linalg.norm(X[o], axis=1)
    Bn, On = unit(B), unit(O)
    sums = np.array([B[grp == g].sum(0) for g in groups])
    cnt = np.array([(grp == g).sum() for g in groups])
    C = unit(sums / cnt[:, None])
    # Leave-one-out on books: each book chunk against centres computed without it
    gi = np.array([groups.index(g) for g in grp])
    hit1 = hit3 = 0
    own = np.zeros(len(b))
    for i in range(len(b)):
        cc = C.copy()
        if cnt[gi[i]] > 1:
            cc[gi[i]] = unit((sums[gi[i]] - B[i]) / (cnt[gi[i]] - 1))
        s = cc @ Bn[i]
        order = np.argsort(-s)
        hit1 += order[0] == gi[i]
        hit3 += gi[i] in order[:3]
        own[i] = s[gi[i]]
    sim = On @ C.T
    best = sim.max(1)
    second = np.sort(sim, axis=1)[:, -2]
    lab = np.array([groups[j] for j in sim.argmax(1)], dtype=object)
    return {"lab": lab, "best": best, "margin": best - second, "captured": captured,
            "acc1": hit1 / len(b), "acc3": hit3 / len(b), "evr": float(pca.explained_variance_ratio_.sum())}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rows = pq.read_table(ROOT / "Preprocessing" / "output" / "chunks.parquet",
                         columns=["chunk_id", "source", "kind", "year", "title", "url", "book", "chapter",
                                  "subchapter", "text"]).to_pylist()
    E = np.load(ROOT / "Embedding" / f"results_{VARIANT}" / "embeddings.npy")
    ids = (ROOT / "Embedding" / f"results_{VARIANT}" / "chunk_ids.txt").read_text(encoding="utf-8").split()
    assert ids == [r["chunk_id"] for r in rows], "embeddings don't match chunks.parquet"
    src = np.array([r["source"] for r in rows])
    kind = np.array([r["kind"] for r in rows])
    X = E.copy()
    for s in ("article", "book", "video"):
        X[src == s] -= X[src == s].mean(0)          # remove each source's average (style) offset

    b = np.where(src == "book")[0]
    o = np.where((src != "book") & (kind != "announcement"))[0]
    grp = np.array([region_of(rows[i]) for i in b], dtype=object)
    groups = sorted(set(grp), key=lambda g: int(b[grp == g][0]))       # book order
    gbook = {g: rows[int(b[grp == g][0])]["book"] for g in groups}

    K = broken_stick_k(PCA(random_state=0).fit(X[b]).explained_variance_ratio_)
    res = assign(X, b, o, grp, groups, K)
    lab = res["lab"].copy()
    osrc = src[o]

    # Fit to the books, measured in the full space: a text's similarity to its nearest section centre,
    # compared with how similar book chunks are to their own section (centre computed without them).
    gi = np.array([groups.index(g) for g in grp])
    sums = np.array([X[b][grp == g].sum(0) for g in groups])
    cnt = np.bincount(gi, minlength=len(groups))
    Cf = unit(sums / cnt[:, None])
    Bf = unit(X[b])
    own = np.array([unit((sums[gi[i]] - X[b][i]) / (cnt[gi[i]] - 1)) @ Bf[i] if cnt[gi[i]] > 1
                    else Cf[gi[i]] @ Bf[i] for i in range(len(b))])
    fit = (unit(X[o]) @ Cf.T).max(1)
    worst, seen = [], set()
    for j in np.argsort(fit):
        t = rows[o[j]]["title"]
        if t in seen:
            continue
        seen.add(t)
        worst.append([0 if osrc[j] == "article" else 1, t, rows[o[j]].get("url") or "",
                      int(rows[o[j]]["year"] or 0), round(float(fit[j]), 3), lab[j]])
        if len(worst) == N_WORST:
            break

    # Distribution per group and source
    dist = []
    n_art, n_vid = int((osrc == "article").sum()), int((osrc == "video").sum())
    nb = len(b)
    for g in groups:
        m = lab == g
        a, v = int((m & (osrc == "article")).sum()), int((m & (osrc == "video")).sum())
        bshare = float((grp == g).mean())
        share = (a + v) / len(o)
        # most typical examples: highest similarity among assigned chunks (distinct titles)
        ex, seen = [], set()
        for j in np.argsort(-res["best"] * m):
            if not m[j] or len(ex) == 3:
                break
            t = rows[o[j]]["title"]
            if t not in seen:
                seen.add(t)
                ex.append([0 if osrc[j] == "article" else 1, t, rows[o[j]].get("url") or "", int(rows[o[j]]["year"] or 0)])
        dist.append({"g": g, "book": gbook[g], "framing": g in FRAMING, "a": a, "v": v, "share": share,
                     "sa": a / n_art, "sv": v / n_vid, "bshare": bshare,
                     "ratio": (share / bshare) if bshare else None, "ex": ex,
                     "order": groups.index(g) if g in groups else len(groups)})

    # Per book (topical only and all)
    per_book = {}
    for name, sel in (("all", np.ones(len(o), bool)), ("article", osrc == "article"), ("video", osrc == "video")):
        c = Counter(gbook[l] for l in lab[sel])
        per_book[name] = {k: c[k] / sel.sum() for k in ("Anarchokapitalismus", "Anarchoagorismus")}

    # Robustness: shares of the top groups under other space sizes
    top_groups = [d["g"] for d in sorted(dist, key=lambda d: -d["share"])][:10]
    robust = {}
    for k in K_ALT:
        r2 = assign(X, b, o, grp, groups, k)
        l2 = r2["lab"]
        c2 = Counter(l2)
        robust[k] = {"shares": {g: c2[g] / len(o) for g in top_groups},
                     "agree": float(np.mean(l2 == lab)), "acc1": r2["acc1"]}

    OUT.mkdir(exist_ok=True)
    html(dist, res, K, len(o), n_art, n_vid, nb, per_book, robust, top_groups, len(groups),
         OUT / f"topic_distribution_{VARIANT}.html", own, fit, osrc, worst)
    print(f"K={K} (book variance {res['evr']:.0%}), LOO acc {res['acc1']:.0%} / top-3 {res['acc3']:.0%}, "
          f"captured median {np.median(res['captured']):.2f}; fit median {np.median(fit):.3f} "
          f"vs books own {np.median(own):.3f}")
    for d in sorted(dist, key=lambda d: -d["share"])[:15]:
        print(f"  {d['share']:.1%}  A{d['a']:>3} V{d['v']:>3}  {d['g']}  (book {d['bshare'] or 0:.1%})")
    print("per book:", {k: {kk: f'{vv:.0%}' for kk, vv in v.items()} for k, v in per_book.items()})
    print("robust:", {k: (f"agree {v['agree']:.0%}") for k, v in robust.items()})
    print(f"Wrote {OUT / f'topic_distribution_{VARIANT}.html'}")


def fit_svg(own, fit, osrc):
    """Distributions of fit: book chunks to their own section vs. articles and videos to their nearest section."""
    w, h, pl, pb, pt = 640, 210, 40, 34, 14
    lo, hi = -0.05, 0.75
    bins = np.linspace(lo, hi, 33)
    series = [("Book chunks → own section", own, "var(--s1)"),
              ("Article chunks → nearest section", fit[osrc == "article"], "var(--s0)"),
              ("Video chunks → nearest section", fit[osrc == "video"], "var(--s2)")]
    dens = [np.histogram(np.clip(v, lo, hi - 1e-9), bins)[0] / len(v) for _, v, _ in series]
    top = max(d.max() for d in dens) * 1.1
    xpos = lambda v: pl + (v - lo) / (hi - lo) * (w - pl - 10)
    ypos = lambda v: h - pb - v / top * (h - pb - pt)
    out = []
    for t in (0.0, 0.2, 0.4, 0.6):
        out.append(f'<line x1="{xpos(t):.1f}" x2="{xpos(t):.1f}" y1="{pt}" y2="{h - pb}" class="grid"/>'
                   f'<text x="{xpos(t):.1f}" y="{h - pb + 14}" text-anchor="middle" class="ax">{t:.1f}</text>')
    for (name, v, col), d in zip(series, dens):
        pts = " ".join(f"{xpos(bins[i]):.1f},{ypos(d[i]):.1f} {xpos(bins[i + 1]):.1f},{ypos(d[i]):.1f}"
                       for i in range(len(d)))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2">'
                   f'<title>{name}: median {np.median(v):.2f}</title></polyline>')
        mx = xpos(np.median(v))
        out.append(f'<line x1="{mx:.1f}" x2="{mx:.1f}" y1="{pt}" y2="{h - pb}" stroke="{col}" '
                   f'stroke-width="1" stroke-dasharray="3 3"/>')
    out.append(f'<text x="{(pl + w) / 2:.0f}" y="{h - 4}" text-anchor="middle" class="ax">cosine similarity to '
               f'the section centre (dashed lines: medians)</text>')
    legend = "".join(f'<span><span class="sw" style="background:{c}"></span>{esc(n)}</span>' for n, _, c in series)
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Fit to the books" '
            f'style="width:100%;max-width:{w}px;height:auto">{"".join(out)}'
            f'<line x1="{pl}" x2="{w - 10}" y1="{h - pb}" y2="{h - pb}" class="base"/></svg>'
            f'<div class="legend">{legend}</div>')


def html(dist, res, K, n, n_art, n_vid, nb, per_book, robust, top_groups, ngroups, path, own, fit, osrc, worst):
    p = lambda v: f"{v * 100:.0f}%"
    q = lambda s: f"“{esc(s)}”"
    topical = [d for d in dist if not d["framing"]]
    ranked = sorted(topical, key=lambda d: -d["share"])
    top5 = ", ".join(f"{q(d['g'])} ({p(d['share'])})" for d in ranked[:5])
    covered = [d for d in topical if d["a"] + d["v"] > 0]
    empty = [d for d in topical if d["a"] + d["v"] == 0]
    emph = sorted([d for d in topical if d["bshare"] and d["bshare"] >= 0.008], key=lambda d: -(d["ratio"] or 0))
    over = emph[:6]
    under = sorted(emph, key=lambda d: d["ratio"] or 0)[:6]
    q25 = float(np.percentile(own, 25))
    worst_html = "".join(
        f"<tr><td><span class='sw' style='background:var(--s{2 if w_[0] else 0})'></span>"
        + (f"<a href='{esc(w_[2])}' target='_blank' rel='noopener'>{esc(w_[1])}</a>" if w_[2] else esc(w_[1]))
        + f"</td><td>{w_[3] or '–'}</td><td>{w_[4]:.2f}</td><td class='l'>{esc(w_[5])}</td></tr>" for w_ in worst)
    rob_rows = "".join(
        f"<tr><td>{esc(g)}</td><td>{p(next(d['share'] for d in dist if d['g'] == g))}</td>"
        + "".join(f"<td>{p(robust[k]['shares'][g])}</td>" for k in robust) + "</tr>" for g in top_groups)
    rob_rows += ("<tr><td><strong>Same section as main result</strong></td><td>–</td>"
                 + "".join(f"<td>{p(robust[k]['agree'])}</td>" for k in robust) + "</tr>")
    pb = per_book
    fill = {
        "__DATA__": json.dumps(dist, ensure_ascii=False).replace("</", "<\\/"),
        "__N__": f"{n:,}", "__NA__": f"{n_art:,}", "__NV__": f"{n_vid:,}", "__NB__": f"{nb:,}",
        "__NG__": str(ngroups), "__K__": str(K), "__EVR__": p(res["evr"]),
        "__ACC1__": p(res["acc1"]), "__ACC3__": p(res["acc3"]), "__CHANCE__": f"{100 / ngroups:.1f}%",
        "__CAP__": f"{np.median(res['captured']):.2f}",
        "__FITSVG__": fit_svg(own, fit, osrc), "__WORST__": worst_html, "__NWORST__": str(len(worst)),
        "__OWN_MED__": f"{np.median(own):.2f}", "__FIT_A__": f"{np.median(fit[osrc == 'article']):.2f}",
        "__FIT_V__": f"{np.median(fit[osrc == 'video']):.2f}", "__Q25__": f"{q25:.2f}",
        "__BELOW_Q25__": p(float(np.mean(fit < q25))),
        "__BELOW_P10__": p(float(np.mean(fit < np.percentile(own, 10)))),
        "__BELOW_P05__": p(float(np.mean(fit < np.percentile(own, 5)))),
        "__TOP5__": top5, "__NCOV__": str(len(covered)), "__NTOP__": str(len(topical)),
        "__EMPTY__": ", ".join(q(d["g"]) for d in empty) or "none",
        "__OVER__": ", ".join(f"{q(d['g'])} ({p(d['share'])} vs {p(d['bshare'])} of the books)" for d in over),
        "__UNDER__": ", ".join(f"{q(d['g'])} ({p(d['share'])} vs {p(d['bshare'])})" for d in under),
        "__PB_ANKAP__": p(pb["all"]["Anarchokapitalismus"]), "__PB_ANAG__": p(pb["all"]["Anarchoagorismus"]),
        "__PB_A__": f"{p(pb['article']['Anarchokapitalismus'])} / {p(pb['article']['Anarchoagorismus'])}",
        "__PB_V__": f"{p(pb['video']['Anarchokapitalismus'])} / {p(pb['video']['Anarchoagorismus'])}",
        "__ROB_HEAD__": "".join(f"<th class='r'>{k} dims</th>" for k in robust),
        "__ROB_ROWS__": rob_rows,
        "__ROB_ACC__": ", ".join(f"{k} dims {p(robust[k]['acc1'])}" for k in robust),
        "__ROB_AGREE__": " and ".join(p(robust[k]["agree"]) for k in robust),
    }
    page = TEMPLATE
    for k, v in fill.items():
        page = page.replace(k, v)
    path.write_text(page, encoding="utf-8")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Topic distribution</title>
<style>
:root{--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--s0:#2a78d6;--s1:#eb6834;--s2:#1baf7a;--muted:#b8b7b1}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835;--s0:#3987e5;--s1:#d95926;--s2:#199e70;--muted:#5a5a55}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}p.sub,p.cap{color:var(--ink2);max-width:880px}p.cap{font-size:13px;margin:8px 2px 0}
h2.sec{font-size:17px;margin:28px 0 6px;padding-top:12px;border-top:1px solid var(--line)}
ol,ul{max-width:880px;padding-left:20px}li{margin:6px 0}
code{background:var(--line);padding:1px 5px;border-radius:4px;font-size:12.5px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.controls{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:0 0 12px;font-size:13px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:7px;overflow:hidden}
.seg button{background:var(--card);color:var(--ink2);border:0;padding:5px 11px;cursor:pointer;font:inherit}
.seg button.on{background:var(--line);color:var(--ink);font-weight:600}
.sw{width:10px;height:10px;border-radius:3px;display:inline-block;margin-right:5px;vertical-align:-1px}
.row{display:grid;grid-template-columns:260px minmax(0,1fr) 54px;gap:10px;align-items:center;padding:3px 0;font-size:12.5px;cursor:pointer}
.row:hover{background:var(--bg)}.row .nm{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.row .nm small{color:var(--ink2)}.row.fr .nm{color:var(--ink2);font-style:italic}
.bars{position:relative;height:14px}.stack{display:flex;gap:2px;height:14px}
.stack span{display:block;height:14px}.stack span:last-child{border-radius:0 4px 4px 0}
.tick{position:absolute;top:-2px;width:2px;height:18px;background:var(--ink)}
.num{text-align:right;font-variant-numeric:tabular-nums}
.detail{grid-column:1/-1;background:var(--bg);border-radius:6px;padding:6px 10px;font-size:12.5px;color:var(--ink2);display:none}
.detail.on{display:block}.detail a{color:inherit}
.legend{font-size:12.5px;color:var(--ink2);display:flex;gap:16px;flex-wrap:wrap;margin-top:10px}
table{border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:8px;font-size:12.5px}
th,td{padding:5px 10px;border-bottom:1px solid var(--line);text-align:left}th{color:var(--ink2);font-weight:600}
td:not(:first-child),th.r{text-align:right;font-variant-numeric:tabular-nums}td.l,th.l{text-align:left}
svg .grid{stroke:var(--line);stroke-width:1}svg .base{stroke:var(--ink2);stroke-width:1}svg .ax{fill:var(--ink2);font-size:10px}.tw{overflow-x:auto}
.kpis{display:flex;gap:12px;flex-wrap:wrap;margin:8px 0}.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;min-width:150px}
.kpi b{display:block;font-size:20px}.kpi span{color:var(--ink2);font-size:12px}
@media (max-width:700px){.row{grid-template-columns:150px minmax(0,1fr) 46px}}
</style></head><body><div class="wrap">
<h1>Topic distribution <span style="font-weight:400;color:var(--ink2)">· which book concepts do the articles and videos cover? · <code>textonly</code></span></h1>
<p class="sub">The two books are the most systematic statement of Urza's ideas. Here they define the concepts:
each book section is a cluster, with a conceptual centre. The __NA__ article chunks and __NV__ video chunks are
projected into the space the books define and assigned to the nearest centre. The result is how Urza's other
output is distributed over the books' concepts, and which texts fit none of them well.</p>

<div class="kpis">
 <div class="kpi"><b>__ACC1__</b><span>book chunks put back in their own section (chance __CHANCE__)</span></div>
 <div class="kpi"><b>__NCOV__ / __NTOP__</b><span>topical book sections that articles or videos reach</span></div>
 <div class="kpi"><b>__FIT_A__ / __FIT_V__</b><span>median fit of articles / videos to the books (books to themselves: __OWN_MED__)</span></div>
</div>

<h2 class="sec">What was done</h2>
<ol>
 <li><strong>Embeddings:</strong> all chunks, embedded from their own text only (variant <code>textonly</code>).
 Each source's average vector was subtracted first, so the difference in style between book prose, blog posts
 and video descriptions doesn't decide the outcome (without this, one section about Urza's organisation
 “Svobodný přístav” attracted a third of all videos).</li>
 <li><strong>The books' concept space:</strong> PCA fitted on the __NB__ book chunks only. Kept are the __K__
 directions that carry more variance than a random split would (broken-stick rule); they hold __EVR__ of the
 books' variation. Articles and videos are projected into this space; directions the books don't use are
 dropped. A typical article or video chunk keeps __CAP__ of its length in it.</li>
 <li><strong>Conceptual centres:</strong> the average of each of the __NG__ book sections' chunks in this space
 (chapters; the appendix and AMEN split into their topics; “19. Praxe” split into its subsections).</li>
 <li><strong>Assignment:</strong> every article and video chunk goes to the centre with the highest cosine
 similarity. Announcements are left out.</li>
 <li><strong>Check on the books:</strong> each book chunk was assigned with its own section's centre computed
 without it. It lands in its own section __ACC1__ of the time and among the top three __ACC3__ (chance:
 __CHANCE__), so the centres are distinct enough to classify by.</li>
</ol>

<h2 class="sec">Distribution over the books' concepts</h2>
<div class="controls">
 <span>Sort</span><div class="seg" id="sort"><button data-v="share" class="on">By share</button><button data-v="book">Book order</button><button data-v="ratio">Vs. the books</button></div>
 <span>Show</span><div class="seg" id="src"><button data-v="all" class="on">Articles + videos</button><button data-v="a">Articles</button><button data-v="v">Videos</button></div>
</div>
<div class="panel" id="chart"></div>
<div class="legend"><span><span class="sw" style="background:var(--s0)"></span>articles</span>
<span><span class="sw" style="background:var(--s2)"></span>videos</span>
<span><span class="sw" style="background:var(--ink);width:2px"></span>share of the books themselves</span>
<span><em>italic</em> = framing sections (stories, introductions, epilogues)</span>
<span>AK = <em>Anarchokapitalismus</em>, AAG = <em>Anarchoagorismus</em></span></div>
<p class="cap"><strong>How to read it:</strong> each bar is the share of all article and video chunks assigned to
that book section, split into articles (blue) and videos (green). The black tick marks how big the section is in
the books, so a bar reaching past its tick means the topic gets more attention in articles and videos than in
the books. Click a row for the three most typical texts assigned to it. “Vs. the books” sorts by that ratio.</p>
<p class="cap"><strong>What it shows:</strong> the most covered concepts are __TOP5__. Much more present than in the
books: __OVER__. Much less: __UNDER__. Not reached at all: __EMPTY__.</p>

<h2 class="sec">Between the two books</h2>
<div class="tw"><table><thead><tr><th>Assigned to sections of…</th><th class="r">All</th><th class="r">Articles</th><th class="r">Videos</th></tr></thead><tbody>
<tr><td><em>Anarchokapitalismus</em> (institutions, economics)</td><td>__PB_ANKAP__</td><td colspan="2" rowspan="2" style="text-align:center">articles __PB_A__ · videos __PB_V__<br><span style="color:var(--ink2)">(Anarchokapitalismus / Anarchoagorismus)</span></td></tr>
<tr><td><em>Anarchoagorismus</em> (freedom, obedience, self-development)</td><td>__PB_ANAG__</td></tr>
</tbody></table></div>
<p class="cap">The timeline report shows how this balance shifted from the first book's themes towards the
second's after 2020.</p>

<h2 class="sec">How well do articles and videos fit the books at all?</h2>
<p class="sub">Every text gets a section, even one about something the books never discuss. To see how well texts
fit, their similarity to the nearest section centre is compared, in the full 1,024-dimensional space, with how
similar the books' own chunks are to their own section (centre computed without the chunk).</p>
<div class="panel">__FITSVG__</div>
<p class="cap"><strong>Reading it:</strong> article and video chunks fit the books' sections less tightly (medians
__FIT_A__ and __FIT_V__) than the books fit themselves (__OWN_MED__). That is expected, since they also contain
current affairs, events and people the books don't mention. There is no natural cut-off for “outside the books”:
__BELOW_P05__ of article and video chunks fit worse than the bottom 5% of book chunks, __BELOW_P10__ worse than the
bottom 10%, and __BELOW_Q25__ worse than the bottom quarter. So the report shows the fit as a distribution and lists
the worst-fitting texts instead of one “outside” number.</p>
<div class="tw"><table><thead><tr><th>The __NWORST__ worst-fitting texts</th><th>Year</th><th>Fit</th><th class="l">Nearest section anyway</th></tr></thead>
<tbody>__WORST__</tbody></table></div>
<p class="cap">These are the texts furthest from all of the books' concepts: mostly current events, politicians,
media and specific campaigns. They are where Urza's output goes beyond what the books systematise.</p>

<h2 class="sec">How robust is this?</h2>
<p class="sub">The size of the book space is a choice. The same assignment with fewer and with more directions:</p>
<div class="tw"><table><thead><tr><th>Section</th><th class="r">__K__ dims (main)</th>__ROB_HEAD__</tr></thead><tbody>__ROB_ROWS__</tbody></table></div>
<p class="cap"><strong>Reading it:</strong> the leading sections stay the same, and __ROB_AGREE__ of chunks get the
same section with the other sizes; the book check gives __ROB_ACC__. The ranking of covered concepts is
solid; individual borderline chunks can switch between neighbouring sections.</p>

<h2 class="sec">Conclusions</h2>
<ul>
 <li><strong>The books' concepts cover most of Urza's other output:</strong> __NCOV__ of __NTOP__ topical
 sections are reached. Articles and videos fit the sections less tightly than the books themselves, and the
 worst-fitting texts (current affairs, politicians, media, campaigns) mark where Urza goes beyond the books.</li>
 <li><strong>The distribution is uneven</strong>, led by __TOP5__. The practical, organisational and education
 topics are far more present in articles and videos than in the books; parts of the economic theory are mainly
 in the books.</li>
 <li><strong>Caveats:</strong> videos are represented by their descriptions; a chunk gets exactly one section
 even when it touches several; sections are clusters of the author's own text, so a topic the books treat
 briefly has a weaker centre. Framing sections (stories, introductions, epilogues) attract texts by tone
 rather than topic and are shown in italics.</li>
 <li><strong>For the coverage map:</strong> the same centres can score political questions. A question that
 lands outside every centre, or in a section few texts reach, points to a gap in Urza's coverage.</li>
</ul>
</div>
<script>
const D=__DATA__;let sort="share",src="all";const open=new Set();
const val=d=>src==="a"?d.sa:src==="v"?d.sv:d.share;
function render(){const rows=[...D];
 if(sort==="share")rows.sort((a,b)=>val(b)-val(a));else if(sort==="book")rows.sort((a,b)=>a.order-b.order);
 else rows.sort((a,b)=>((b.bshare?val(b)/b.bshare:-1)-(a.bshare?val(a)/a.bshare:-1)));
 const mx=Math.max(...D.map(d=>Math.max(val(d),d.bshare||0)));
 document.getElementById("chart").innerHTML=rows.map((d,i)=>{const tot=val(d),a=src==="v"?0:(src==="a"?d.sa:d.a/(d.a+d.v||1)*d.share),v=tot-a;
  const tip=`${d.g}: ${d.a} article + ${d.v} video chunks = ${(d.share*100).toFixed(1)}% (articles ${(d.sa*100).toFixed(1)}%, videos ${(d.sv*100).toFixed(1)}%)`+(d.bshare?`; ${(d.bshare*100).toFixed(1)}% of the books`:"");
  return `<div class="row${d.framing?" fr":""}" data-i="${D.indexOf(d)}" title="${esc(tip)}"><div class="nm">${esc(d.g)} <small>${d.book==="Anarchoagorismus"?"AAG":d.book?"AK":""}</small></div>
  <div class="bars"><div class="stack" style="width:${100*tot/mx}%">${a>0?`<span style="flex:${100*a/tot};background:var(--s0)"></span>`:""}${v>0?`<span style="flex:${100*v/tot};background:var(--s2)"></span>`:""}</div>
  ${d.bshare?`<div class="tick" style="left:${100*d.bshare/mx}%"></div>`:""}</div><div class="num">${(tot*100).toFixed(1)}%</div>
  <div class="detail${open.has(D.indexOf(d))?" on":""}">${d.ex.length?d.ex.map(e=>`<span class="sw" style="background:var(--s${e[0]?2:0})"></span>${e[2]?`<a href="${esc(e[2])}" target="_blank" rel="noopener">${esc(e[1])}</a>`:esc(e[1])}${e[3]?` (${e[3]})`:""}`).join("<br>"):"no texts assigned"}</div></div>`}).join("");
 document.querySelectorAll(".row").forEach(r=>r.onclick=e=>{if(e.target.tagName==="A")return;const i=+r.dataset.i;open.has(i)?open.delete(i):open.add(i);render()})}
function seg(id,set){document.querySelectorAll(`#${id} button`).forEach(b=>b.onclick=()=>{document.querySelectorAll(`#${id} button`).forEach(x=>x.classList.toggle("on",x===b));set(b.dataset.v);render()})}
seg("sort",v=>sort=v);seg("src",v=>src=v);
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
render();
</script></body></html>"""


if __name__ == "__main__":
    main()
