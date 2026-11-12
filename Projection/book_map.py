"""Map of the two books only (raw embeddings, no source-centring), labelled by chapter.

    python Projection/book_map.py --variant nosource      # from the repo root

Question: do regions of the map correspond to ideas? The books are the best test,
because their chapters are human-made topic labels.

Reads  chunks.parquet and results_<variant>/embeddings.npy of the dataset chosen in dataset.py
       (default Preprocessing/output + Embedding/; URZA_DATASET=final: Chunks/output/)
Writes <OUT_DIR>/book_map_<variant>.html   interactive, self-contained report
       <OUT_DIR>/book_map_<variant>.png    static map
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from labels import LABEL_JS, place_labels  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import CHUNKS, ENGLISH, OUT_DIR, nm, read_chunks, results_dir  # noqa: E402
OUT = OUT_DIR
BOOK_COLORS = {"Anarchokapitalismus": "#2a78d6", "Anarchoagorismus": "#eb6834"}  # validated slots 1–2
UMAP_PARAMS = dict(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
K = 10


def chapter_group(r) -> str:
    """The idea-level unit: a chapter, or for the appendix its topic / AMEN category."""
    ch, sub = r["chapter"], (r["subchapter"] or "").split(" › ")
    if ch in ("dodatek", "Appendix"):                       # Czech chapter names, or their English display twins
        return f"AMEN: {sub[1]}" if sub[0].startswith("AMEN") and len(sub) > 1 else sub[-1]
    if ch.startswith("AMEN"):                                # "AMEN: aforismy …" → "AMEN: aforismy" ("AMEN: aphorisms")
        return "AMEN: " + ch.split(":", 1)[1].split()[0]
    return ch


ISLAND_LINK = 0.04     # points closer than this (map units) belong to the same island
ISLAND_GAP = 0.12      # an island farther than this from the main cloud gets pulled in…
ISLAND_NEW_GAP = 0.05  # …to this distance, keeping its own shape


def compact(P, grp):
    """Pull far-away islands towards the main cloud so the map has no large empty areas.

    Shapes and scale inside the main cloud and inside each island are unchanged (pure
    translation); only the empty gap between an island and the rest shrinks. Returns the
    display coordinates (rescaled to start at 0, same scale on both axes) and a list of
    moved islands with how much their gap was shortened.
    """
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import radius_neighbors_graph
    _, comp = connected_components(radius_neighbors_graph(P, ISLAND_LINK), directed=False)
    sizes = np.bincount(comp)
    main = comp == sizes.argmax()
    # Grow the main cloud with every component that is already close to it.
    changed = True
    while changed:
        changed = False
        for c in np.unique(comp[~main]):
            m = comp == c
            gap = np.sqrt(((P[m][:, None] - P[main][None]) ** 2).sum(-1)).min()
            if gap <= ISLAND_GAP:
                main |= m
                changed = True
    Q = P.copy()
    moved = []
    centre = P[main].mean(0)
    for c in np.unique(comp[~main]):
        m = comp == c
        d = np.sqrt(((Q[m][:, None] - Q[main][None]) ** 2).sum(-1))
        gap = d.min()
        direction = Q[m].mean(0) - centre
        direction /= np.linalg.norm(direction)
        # Move along the line to the main cloud until the gap is ISLAND_NEW_GAP.
        lo, hi = 0.0, gap
        for _ in range(40):
            t = (lo + hi) / 2
            g = np.sqrt(((Q[m] - t * direction)[:, None] - Q[main][None]) ** 2).sum(-1).min()
            lo, hi = (t, hi) if g > ISLAND_NEW_GAP else (lo, t)
        Q[m] -= lo * direction
        groups = Counter(grp[m]).most_common()
        moved.append({"idx": np.where(m)[0], "gap_before": float(gap), "gap_after": ISLAND_NEW_GAP,
                      "groups": [g for g, _ in groups]})
        main |= m
    Q -= Q.min(0)
    Q /= Q.max()
    return Q, moved


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

    rows = read_chunks(CHUNKS, ["chunk_id", "doc_id", "source", "book", "chapter", "subchapter",
                                  "title", "text"])
    E = np.load(results_dir(variant) / "embeddings.npy")
    ids = (results_dir(variant) / "chunk_ids.txt").read_text(encoding="utf-8").split()
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

    # Facts about the chunks themselves, so the report text follows the data.
    import re
    nxt = [(a, b) for a, b in zip(B, B[1:]) if a["doc_id"] == b["doc_id"]]

    def overlaps(a, b):
        last = re.split(r"(?<=[.!?])\s+", a["text"].strip())[-1]
        return len(last) > 15 and last in b["text"]
    facts = {
        "words": float(np.median([len(r["text"].split()) for r in B])),
        "single_para": float(np.mean(["\n\n" not in r["text"].strip() for r in B])),
        "overlap": float(np.mean([overlaps(a, b) for a, b in nxt])) if nxt else 0.0,
        "subsections": {g: len({(B[i]["subchapter"] or "").split(" › ")[0] for i in np.where(grp == g)[0]} - {""})
                        for g in set(grp)},
    }

    print(f"UMAP on {len(B)} book chunks …")
    P = umap.UMAP(**UMAP_PARAMS).fit_transform(X)
    P = (P - P.min(0)) / (P.max(0) - P.min(0)).max()
    # Display copy: far-off islands pulled in (measurements below use the original P).
    Pd, moved = compact(P, grp)

    print("t-SNE …")
    from sklearn.manifold import TSNE
    T = TSNE(n_components=2, metric="cosine", perplexity=30, init="pca", random_state=42).fit_transform(X)
    a, b = P - P.mean(0), T - T.mean(0)
    u, _, vt = np.linalg.svd(b.T @ a)          # rotate/flip t-SNE to line up with UMAP (no distortion)
    T = b @ (u @ vt)
    T = (T - T.min(0)) / (T.max(0) - T.min(0)).max()
    Td, moved_t = compact(T, grp)

    # Clumps on the (original) UMAP map, as in compact(): used to explain spread-out groups.
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import radius_neighbors_graph
    _, comp = connected_components(radius_neighbors_graph(P, ISLAND_LINK), directed=False)
    sub1 = np.array([(r["subchapter"] or "").split(" › ")[0] for r in B])

    rng = np.random.default_rng(0)

    def spread(ix, Q=None):
        Q = (P if Q is None else Q)[ix]
        return float(np.sqrt(((Q - Q.mean(0)) ** 2).sum(1)).mean())

    chapters = []
    for g, cnt in Counter(grp).items():
        ix = np.where(grp == g)[0]
        ratio = ratio_t = None
        if cnt >= 3:
            draws = [rng.choice(len(P), cnt, replace=False) for _ in range(50)]
            ratio = spread(ix) / np.mean([spread(r) for r in draws])
            ratio_t = spread(ix, T) / np.mean([spread(r, T) for r in draws])
        nn_same = float((grp[nn[ix]] == g).mean())
        clumps = [k for _, k in Counter(comp[ix]).most_common()]
        # Each subsection sits in exactly one clump, and different subsections in different clumps?
        by_sub = {s: set(comp[ix][sub1[ix] == s]) for s in set(sub1[ix])}
        per_sub = (len(by_sub) > 1 and all(len(v) == 1 for v in by_sub.values())
                   and len(set.union(*by_sub.values())) == len(by_sub))
        chapters.append({"name": g, "book": B[ix[0]]["book"], "n": int(cnt), "ratio": ratio, "ratio_t": ratio_t,
                         "clumps": clumps, "per_sub": per_sub,
                         "tx": float(Td[ix, 0].mean()), "ty": float(Td[ix, 1].mean()),
                         "nn_same": nn_same, "cx": float(Pd[ix, 0].mean()), "cy": float(Pd[ix, 1].mean()),
                         "ox": float(P[ix, 0].mean()), "oy": float(P[ix, 1].mean()),
                         "first": int(min(ix))})
    chapters.sort(key=lambda c: c["first"])
    for k, c in enumerate(chapters):
        c["id"] = k
    gid = {c["name"]: c["id"] for c in chapters}

    # Which chapters are closest in meaning? (centroids of the full vectors)
    names = [c["name"] for c in chapters]
    C = np.array([X[grp == nm].mean(0) for nm in names])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    CS = C @ C.T
    np.fill_diagonal(CS, -1)
    nearest = {nm: [names[j] for j in np.argsort(-CS[i])[:2]] for i, nm in enumerate(names)}
    # Map distance from each chapter's centre to the closest other chapter centre (isolation)
    cen = np.array([[c["ox"], c["oy"]] for c in chapters])   # original layout
    dd = np.sqrt(((cen[:, None] - cen[None]) ** 2).sum(-1))
    np.fill_diagonal(dd, np.inf)
    isolated = names[int(dd.min(1).argmax())]

    OUT.mkdir(exist_ok=True)
    ratios = sorted(c["ratio"] for c in chapters if c["ratio"] is not None)
    ratios_t = sorted(c["ratio_t"] for c in chapters if c["ratio_t"] is not None)
    stats = {"rates": rates, "cross": cross, "median_ratio": ratios[len(ratios) // 2],
             "median_ratio_tsne": ratios_t[len(ratios_t) // 2], "tight_tsne": sum(1 for r in ratios_t if r < 0.2),
             "tight": sum(1 for r in ratios if r < 0.2), "n_ratios": len(ratios)}
    (OUT / f"book_stats_{variant}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    png(Pd, book, chapters, moved, OUT / f"book_map_{variant}.png")
    png(Td, book, chapters, moved_t, OUT / f"book_map_{variant}_tsne.png", keys=("tx", "ty"),
        title="Book chunks, t-SNE projection, labelled at each chapter's centre")
    pts = [[round(float(Pd[i, 0]), 4), round(float(Pd[i, 1]), 4), 0 if b["book"] == "Anarchokapitalismus" else 1,
            gid[grp[i]], " › ".join(x for x in (b["chapter"], b["subchapter"]) if x), b["text"][:240],
            round(float(Td[i, 0]), 4), round(float(Td[i, 1]), 4)]
           for i, b in enumerate(B)]
    html(pts, chapters, rates, cross, variant, len(B), OUT / f"book_map_{variant}.html", nearest, isolated, moved, Pd,
         moved_t, Td, facts)
    print("same-section / chapter / book neighbours:",
          {k: f"{v[0]:.0%} vs {v[1]:.0%}" for k, v in rates.items()}, f"cross-section same chapter {cross:.0%}")
    print(f"Wrote {OUT / f'book_map_{variant}.html'}")


def png(P, book, chapters, moved, path, keys=("cx", "cy"),
        title="Book chunks (raw embeddings), labelled at each chapter's centre"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    W, H = P.max(0)
    fig, ax = plt.subplots(figsize=(11, max(5.0, 11 * H / W + 0.6)), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for b, col in BOOK_COLORS.items():
        m = book == b
        ax.scatter(P[m, 0], P[m, 1], s=10, c=col, alpha=0.6, linewidths=0, label=b)
    from matplotlib.patches import FancyBboxPatch
    for mv in moved:
        Q = P[mv["idx"]]
        lo, hi = Q.min(0) - 0.012, Q.max(0) + 0.012
        ax.add_patch(FancyBboxPatch(lo, *(hi - lo), boxstyle="round,pad=0,rounding_size=0.01", fill=False,
                                    ec="#52514e", lw=0.8, ls=(0, (3, 3))))
        ax.text(lo[0], hi[1] + 0.008, f"moved closer: really {mv['gap_before'] / mv['gap_after']:.0f}× farther away",
                fontsize=7, color="#52514e", va="bottom")
    ax.set_aspect("equal")
    ax.set_xlim(-0.03, W + 0.03); ax.set_ylim(-0.03, H + 0.05)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#e4e3df")
    ax.legend(frameon=False, loc="upper left", markerscale=2)
    ax.set_title(title, loc="left", fontsize=12)
    fig.tight_layout()
    place_labels(ax, [(c[keys[0]], c[keys[1]], c["name"]) for c in sorted(chapters, key=lambda c: -c["n"])], fontsize=7)
    fig.savefig(path, dpi=150, facecolor="#fcfcfb", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


PROBES = [nm(x) for x in ["AMEN: Peníze a bankovnictví", "AMEN: Ekonomie", "AMEN: Přirozené právo", "8. Školství", "11. Zločin",
          "Drogy", "10. Zdravotnictví"]]
CONTEXT = {"withsource": "“Kniha Anarchokapitalismus › 10. Zdravotnictví”",
           "nosource": "“Anarchokapitalismus › 10. Zdravotnictví”",
           "textonly": "nothing: only the chunk's own text"}


def html(pts, chapters, rates, cross, variant, n, path, nearest, isolated, moved, Pd, moved_t, Td, facts):
    p = lambda v: f"{round(v * 100)}%"
    q = lambda s: f"“{esc(s)}”"
    dims = Pd.max(0)
    islands = [{"x0": float(Pd[m["idx"], 0].min()) - 0.012, "y0": float(Pd[m["idx"], 1].min()) - 0.012,
                "x1": float(Pd[m["idx"], 0].max()) + 0.012, "y1": float(Pd[m["idx"], 1].max()) + 0.012,
                "f": round(m["gap_before"] / m["gap_after"])} for m in moved]
    by_names = {}
    for m in moved:
        by_names.setdefault(tuple(m["groups"][:3]), []).append(m)

    def island_desc(names, ms):
        fs = sorted(round(m["gap_before"] / m["gap_after"]) for m in ms)
        f = f"{fs[0]}×" if fs[0] == fs[-1] else f"{fs[0]}–{fs[-1]}×"
        label = ", ".join(q(g) for g in names)
        if len(ms) == 1:
            return f"the island {label} was moved closer to the rest (its gap shortened about {f}; dashed box on the map)"
        return (f"{len(ms)} separate islands of {label} were moved closer to the rest (gaps shortened about {f}; "
                f"dashed boxes on the map)")
    island_note = ("To avoid a mostly empty picture, " + "; ".join(island_desc(k, v) for k, v in by_names.items())
                   + ". Distances inside the main cloud and inside each island are unchanged; all measurements use "
                     "the original layout.") if moved else ""
    chunk_note = (f"each one paragraph of the book, median {facts['words']:.0f} words"
                  if facts["single_para"] >= 0.95 else f"median {facts['words']:.0f} words each")
    sec_note = ("neighbouring passages, which also share one overlapping sentence" if facts["overlap"] >= 0.1
                else "neighbouring paragraphs of the same text")
    dims2 = Td.max(0)
    islands2 = [{"x0": float(Td[m["idx"], 0].min()) - 0.012, "y0": float(Td[m["idx"], 1].min()) - 0.012,
                 "x1": float(Td[m["idx"], 0].max()) + 0.012, "y1": float(Td[m["idx"], 1].max()) + 0.012,
                 "f": round(m["gap_before"] / m["gap_after"])} for m in moved_t]
    ratios = sorted(c["ratio"] for c in chapters if c["ratio"] is not None)
    med = ratios[len(ratios) // 2]
    tight = sum(1 for r in ratios if r < 0.2)
    rt = sorted(c["ratio_t"] for c in chapters if c["ratio_t"] is not None)
    med_t, tight_t = rt[len(rt) // 2], sum(1 for r in rt if r < 0.2)
    both = [c for c in chapters if c["ratio"] is not None]
    from scipy.stats import spearmanr
    rho = spearmanr([c["ratio"] for c in both], [c["ratio_t"] for c in both]).correlation
    loose_t = [c["name"] for c in sorted(both, key=lambda c: -c["ratio_t"])[:4]]
    tsne_side = (f"Median chapter spread vs random: <strong>{med_t:.2f}</strong> (UMAP {med:.2f}). Chapters below "
                 f"0.20: <strong>{tight_t} of {len(rt)}</strong> (UMAP {tight}). Agreement on which chapters are "
                 f"tight or loose (rank correlation of the two columns in the table below): "
                 f"<strong>{rho:.2f}</strong>.")
    agree = "agree closely" if rho >= 0.7 else "broadly agree" if rho >= 0.4 else "agree only weakly"
    tsne_caption = (
        f"<strong>What you see:</strong> chapters form islands in the t-SNE map too ({tight_t} of {len(rt)} below "
        f"0.20, against {tight} in UMAP), and the two methods "
        f"{agree} on which chapters are compact and which are spread out (rank correlation {rho:.2f}). t-SNE "
        f"packs chapters {'tighter' if med_t < med else 'less tightly'} overall (median {med_t:.2f} vs "
        f"{med:.2f} for UMAP)"
        + (", which is typical of the method: it spreads clusters more evenly over the map, so a chapter's "
           "spread is smaller relative to a random set of chunks" if med_t < med else
           ": t-SNE keeps more space inside groups and pulls clusters less apart than UMAP, so the same "
           "chapters look looser relative to a random set of chunks")
        + f". The loosest groups in t-SNE are {', '.join(q(x) for x in loose_t)}. "
        + ("Because a differently built method shows the same chapter structure, the islands are a property of "
           "the embeddings, not an artefact of UMAP." if rho >= 0.4 else
           "With this little agreement, the exact ranking of chapters depends on the projection method; only "
           "the measurements in the full 1,024 dimensions below are independent of it."))
    if moved_t:
        tsne_caption += (" As on the UMAP map, far-away islands were moved closer (dashed box); measurements "
                         "use the original layout.")
    by_ratio = [c for c in sorted(chapters, key=lambda c: c["ratio"] if c["ratio"] is not None else 9)
                if c["ratio"] is not None]
    tightest = ", ".join(q(c["name"]) for c in by_ratio[:6])
    pairs = "; ".join(f"{q(nm)} → {q(nearest[nm][0])}, {q(nearest[nm][1])}" for nm in PROBES if nm in nearest)
    book_of = {c["name"]: c["book"] for c in chapters}
    cross_pairs = [(nm, nb) for nm in PROBES if nm in nearest for nb in nearest[nm] if book_of[nb] != book_of[nm]]
    meet = (f"; where they meet, it is through related chapters (such as {q(cross_pairs[0][0])} → "
            f"{q(cross_pairs[0][1])} in the list above)" if cross_pairs else "")
    apart = "largely" if rates["book"][0] - rates["book"][1] >= 0.2 else "partly"
    caption = (
        f"<strong>What you see:</strong> chapters form islands with their label on top; the most compact are "
        f"{tightest}. <strong>Related chapters are neighbours.</strong> Closest chapters in meaning (average "
        f"vector of each chapter, full 1,024 dimensions): {pairs}. <strong>The two books</strong> keep {apart} "
        f"apart: {p(rates['book'][0])} of a chunk's nearest neighbours are from the same book "
        f"({p(rates['book'][1])} by chance), in line with <em>Anarchokapitalismus</em> arguing about institutions "
        f"and economics and <em>Anarchoagorismus</em> about freedom, obedience and self-development{meet}. The "
        f"most isolated group on the map is {q(isolated)}.")
    # The effect of the context line, measured if the other variants' book maps exist.
    other = {v: json.loads((path.parent / f"book_stats_{v}.json").read_text(encoding="utf-8"))
             for v in ("withsource", "nosource", "textonly")
             if v != variant and (path.parent / f"book_stats_{v}.json").exists()}
    other[variant] = {"rates": rates, "cross": cross, "median_ratio": med, "tight": tight, "n_ratios": len(ratios)}
    if "textonly" in other and len(other) > 1:
        rows_html = "".join(
            f"<tr><td>{'<strong>' if v == variant else ''}<code>{v}</code>{'</strong>' if v == variant else ''}: "
            f"{CONTEXT[v]}</td><td>{p(o['rates']['chapter'][0])}</td><td>{p(o['cross'])}</td>"
            f"<td>{o['median_ratio']:.2f}</td><td>{o['tight']} of {o['n_ratios']}</td></tr>"
            for v, o in sorted(other.items(), key=lambda kv: ["withsource", "nosource", "textonly"].index(kv[0])))
        t, ref = other["textonly"], other.get("nosource") or other.get("withsource")
        caveat = (
            "<strong>The chapter name in the context line inflated the effect, but the content alone still "
            "groups chapters.</strong> The same chunks embedded with and without the chapter path at the start "
            f"(chance: {p(rates['chapter'][1])} same-chapter neighbours, spread 1.00):</li></ul>"
            "<div class='tw'><table><thead><tr><th>Variant: text in front of each chunk</th><th>Neighbours from "
            "same chapter</th><th>…from other sections</th><th>Median spread vs random</th><th>Chapters below "
            f"0.20</th></tr></thead><tbody>{rows_html}</tbody></table></div><ul><li style='list-style:none'>"
            f"Without the chapter path, same-chapter neighbours fall from {p(ref['rates']['chapter'][0])} to "
            f"{p(t['rates']['chapter'][0])} and the median chapter is about {1 / t['median_ratio']:.0f}× tighter "
            f"than random instead of {1 / ref['median_ratio']:.0f}×. So about "
            f"{p((ref['rates']['chapter'][0] - t['rates']['chapter'][0]) / (ref['rates']['chapter'][0] - rates['chapter'][1]))}"
            " of the above-chance same-chapter neighbours came from the shared label; the rest comes from what the "
            "chapters actually say.")
    else:
        caveat = ("<strong>Caveat: the chapter name is in the text.</strong> Every chunk starts with its chapter "
                  "path, which helps chunks of one chapter find each other. The <code>textonly</code> variant "
                  "measures how much.")
    if variant == "textonly":
        verdict = (f"<strong>Yes, regions correspond to ideas, on content alone.</strong> Without any label in "
                   f"the text, {tight} of {len(ratios)} chapters still form a compact region (median {1 / med:.0f}× "
                   f"tighter than random), and the nearest chapters in meaning are the related ones listed under "
                   f"the map. This is the honest evidence that the embedding organises Urza's texts by topic.")
    else:
        verdict = (f"<strong>Regions of the map correspond to ideas.</strong> {tight} of {len(ratios)} chapters "
                   f"form a compact region and related chapters lie next to each other; see the caveat below "
                   f"about the chapter name in the text.")
    loose = [c for c in sorted(chapters, key=lambda c: -(c["ratio"] or 0)) if (c["ratio"] or 0) >= 0.3]
    # Why is a loose group loose? Split into separate clumps (but still its own neighbours), or mixed content.
    split, mixed = [], []
    for c in loose:
        big = [k for k in c["clumps"] if k >= 3]
        if c["nn_same"] >= 0.6 and len(big) >= 2:
            split.append((c, f"lies in {len(big)} separate clumps" + (" (one per subsection)" if c["per_sub"] else "")))
        elif c["nn_same"] >= 0.6 and c["clumps"][0] < c["n"]:
            rest = c["n"] - c["clumps"][0]
            split.append((c, f"has {c['clumps'][0]} of its {c['n']} chunks in one clump and the other "
                             f"{'one' if rest == 1 else rest} elsewhere"))
        elif c["nn_same"] < 0.6:
            mixed.append(c)
    broad = []
    for c in mixed:
        if nm("19. Praxe") in c["name"] and facts["subsections"].get(c["name"], 0) > 1:
            broad.append(f"{q(c['name'])} is {facts['subsections'][c['name']]} different subsections ({nm('Ježek bez klece')}, "
                         f"{nm('Svobodný přístav')}, {nm('Kryptoanarchie')}, Agora, …)")
        elif nm("Boření mýtů") in c["name"]:
            broad.append(f"{q(c['name'])} debunks myths from many areas")
    split_txt = "; ".join(f"{q(c['name'])} ({p(c['nn_same'])} same-chapter neighbours) {d}" for c, d in split)
    if not loose:
        loose_note = "No group has a spread ratio of 0.30 or more."
    else:
        loose_note = (f"The loosest groups (0.30 or more) are: "
                      + ", ".join(f"{esc(c['name'])} ({c['ratio']:.2f})" for c in loose) + ".")
        if mixed:
            lead = (f"All {len(mixed)}" if len(mixed) == len(loose) else f"{len(mixed)} of them")
            loose_note += (f" {lead} also have few neighbours from their own chapter (at most "
                           f"{p(max(c['nn_same'] for c in mixed))}), so their chunks really are scattered"
                           + (": " + ", and ".join(broad) + (", so they are" if len(broad) > 1 else ", so it is")
                              + " spread because the content is." if broad else "."))
        if split:
            loose_note += (" Others are loose for a different reason: their chunks still find each other, but the "
                           f"map places them apart. {split_txt}.")
    if not loose:
        spread_verdict = "<strong>No chapter is badly spread out</strong> on the map: every group is below 0.30."
    else:
        spread_verdict = "<strong>A spread-out chapter is not a failure of the embedding.</strong> "
        if mixed:
            spread_verdict += ("Loose groups with few same-chapter neighbours really are mixed in content"
                               + (", the clearest being the broad ones (" + ", ".join(
                                   q(c["name"]) for c in mixed if nm("19. Praxe") in c["name"] or nm("Boření mýtů") in c["name"])
                                  + ")" if broad else "") + ". ")
        if split:
            spread_verdict += (", ".join(q(c["name"]) for c, _ in split) + (" keeps" if len(split) == 1 else " keep")
                               + " mostly same-chapter neighbours and" + (" is" if len(split) == 1 else " are")
                               + " loose only because the map splits " + ("it" if len(split) == 1 else "them")
                               + " into separate clumps.")
    rows = sorted([c for c in chapters if c["ratio"] is not None], key=lambda c: c["ratio"])
    table = "".join(
        f'<tr data-g="{c["id"]}"><td><span class="sw" style="background:var(--b{0 if c["book"] == "Anarchokapitalismus" else 1})"></span>'
        f'{esc(c["name"])}</td><td>{c["n"]}</td><td>{c["ratio"]:.2f}</td><td>{c["ratio_t"]:.2f}</td><td>{p(c["nn_same"])}</td></tr>' for c in rows)
    fill = {"__LABELJS__": LABEL_JS,
        "__N__": str(n), "__NCH__": str(len(chapters)), "__APPX__": ", ".join(esc(nm(x)) for x in ("Drogy", "Zbraně", "Věda", "Práce")),
        "__AMENCATS__": ", ".join(esc(nm(x)) for x in ("Ekonomie", "Historie")),
        "__VARIANT__": variant,
        "__SEC__": p(rates["section"][0]), "__SECR__": p(rates["section"][1]),
        "__CH__": p(rates["chapter"][0]), "__CHR__": p(rates["chapter"][1]),
        "__BK__": p(rates["book"][0]), "__BKR__": p(rates["book"][1]),
        "__CROSS__": p(cross), "__MED__": f"{med:.2f}", "__MEDX__": f"{1 / med:.0f}",
        "__TIGHT__": str(tight), "__NRAT__": str(len(ratios)),
        "__LOOSE_NOTE__": loose_note, "__SPREAD_VERDICT__": spread_verdict,
        "__CHUNK_NOTE__": chunk_note, "__SEC_NOTE__": sec_note,
        "__ISLAND_NOTE__": island_note, "__ISLANDS__": json.dumps(islands),
        "__ISLANDS2__": json.dumps(islands2), "__W2__": f"{dims2[0]:.4f}", "__H2__": f"{dims2[1]:.4f}",
        "__TSNE_SIDE__": tsne_side, "__TSNE_CAPTION__": tsne_caption, "__W__": f"{dims[0]:.4f}", "__H__": f"{dims[1]:.4f}",
        "__TABLE__": table, "__CAPTION__": caption, "__CAVEAT__": caveat, "__VERDICT__": verdict,
        "__CONTEXT__": CONTEXT.get(variant, variant),
        "__DATA__": json.dumps(pts, ensure_ascii=False).replace("</", "<\\/"),
        "__CHAPTERS__": json.dumps([{k: c[k] for k in ("id", "name", "n", "cx", "cy", "tx", "ty", "book")} for c in chapters],
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
canvas{width:100%;display:block;cursor:crosshair}
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
 <li><strong>Data.</strong> The __N__ book chunks (__CHUNK_NOTE__) from <em>Anarchokapitalismus</em> and
 <em>Anarchoagorismus</em>, with their bge-m3 embeddings (variant <code>__VARIANT__</code>; text in front of each chunk: __CONTEXT__).</li>
 <li><strong>Chapter groups.</strong> Each chunk belongs to one of __NCH__ groups: a chapter, or in the appendix
 of <em>Anarchokapitalismus</em> its own topic (__APPX__) or AMEN category (__AMENCATS__, …). All AMEN aphorisms of <em>Anarchoagorismus</em> form one group.</li>
 <li><strong>Map.</strong> UMAP to 2D (cosine distance, 15 neighbours, min_dist 0.1, fixed seed) on these chunks
 only. Each chapter's name is placed at the average position of its chunks. __ISLAND_NOTE__</li>
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
<p class="cap">__CAPTION__</p>

<h2 class="sec">A second projection: t-SNE</h2>
<p class="sub">UMAP is one way to flatten 1,024 dimensions into two. To check that the chapter islands are a
property of the embeddings and not of UMAP, the same chunks are projected with <strong>t-SNE</strong>, a method
built on a different principle (it matches neighbour probabilities rather than a neighbourhood graph). Settings:
cosine distance, perplexity 30, PCA initialisation, fixed seed; the layout is rotated to line up with the UMAP
map. Colours, labels, hover and chapter highlighting work as above and are linked between the two maps.</p>
<div class="mapwrap">
 <div class="panel"><canvas id="cv2"></canvas></div>
 <div class="panel side"><h3>t-SNE in numbers</h3><p style="margin:0;color:var(--ink2)">__TSNE_SIDE__</p></div>
</div>
<p class="cap">__TSNE_CAPTION__</p>

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
against __CHR__ by chance. Part of that is trivial: __SEC__ are from the very same section (__SEC_NOTE__). Looking only at neighbours from <em>other</em>
sections, __CROSS__ are still from the same chapter group, i.e. different parts of a chapter find each other.
__BK__ of neighbours are from the same book (__BKR__ by chance): the two books differ in subject and tone.</p>

<h2 class="sec">How tight is each chapter?</h2>
<p class="sub"><strong>How it is computed:</strong> the average distance of a chapter's chunks from their centre
on a map, divided by the same for random sets of chunks of the same size (50 draws), once for the UMAP map and
once for the t-SNE map.
<strong>1.00</strong> = no more clustered than random; <strong>0.10</strong> = packed into a tenth of the
spread. “Neighbours in chapter” is the share of the chunks' 10 nearest neighbours from the same chapter.
Click a row to highlight it on the map.</p>
<div class="scroll"><table id="ctab"><thead><tr><th>Chapter group</th><th>Chunks</th><th>Spread vs random (UMAP)</th><th>Spread vs random (t-SNE)</th><th>Neighbours in chapter</th></tr></thead>
<tbody>__TABLE__</tbody></table></div>
<p class="cap"><strong>Reading it:</strong> the median chapter has a spread ratio of __MED__, i.e. its chunks are
about __MEDX__× more concentrated than random; __TIGHT__ of __NRAT__ groups are below 0.20. __LOOSE_NOTE__</p>

<h2 class="sec">Conclusions</h2>
<ul>
 <li>__VERDICT__</li>
 <li>__SPREAD_VERDICT__</li>
 <li>__CAVEAT__</li>
 <li><strong>The books are a good “skeleton” for the full map.</strong> Their chapter labels can name the
 regions of the full map (articles and videos included), since the same topics recur across sources.</li>
</ul>
</div><div id="tip"></div>
<script>
__LABELJS__
const P=__DATA__, CH=__CHAPTERS__, TOP=18, PAD=14;
const MAPS=[{cv:document.getElementById("cv"),xi:0,yi:1,cx:"cx",cy:"cy",isl:__ISLANDS__,W:__W__,H:__H__},
            {cv:document.getElementById("cv2"),xi:6,yi:7,cx:"tx",cy:"ty",isl:__ISLANDS2__,W:__W2__,H:__H2__}];
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const tip=document.getElementById("tip");
let sel=-1, hover=-1, showLabels=true;
const list=document.getElementById("list");
CH.forEach(c=>{const d=document.createElement("div");d.className="c";d.dataset.g=c.id;
 d.innerHTML=`<span class="sw" style="background:var(--b${c.book==="Anarchokapitalismus"?0:1})"></span>${esc(c.name)}<span class="n">${c.n}</span>`;
 d.onclick=()=>select(c.id);list.appendChild(d)});
document.querySelectorAll("#ctab tbody tr").forEach(tr=>tr.onclick=()=>{select(+tr.dataset.g);MAPS[0].cv.scrollIntoView({behavior:"smooth",block:"center"})});
document.getElementById("labels").onchange=e=>{showLabels=e.target.checked;drawAll()};
function select(g){sel=sel===g?-1:g;document.querySelectorAll(".side div.c").forEach(d=>d.classList.toggle("on",+d.dataset.g===sel));drawAll()}
function size(m){const cv=m.cv,r=cv.getBoundingClientRect(),d=devicePixelRatio||1,h=(r.width-2*PAD)*m.H/m.W+2*PAD+TOP;
 cv.style.height=h+"px";cv.width=r.width*d;cv.height=h*d}
function xy(m,x,y){const d=devicePixelRatio||1,k=(m.cv.width-2*PAD*d)/m.W;return[PAD*d+x*k,m.cv.height-PAD*d-y*k]}
function drawAll(){MAPS.forEach(draw)}
function draw(m){const cv=m.cv,g=cv.getContext("2d"),d=devicePixelRatio||1,col=[css("--b0"),css("--b1")],bg=css("--card"),ink=css("--ink");
 g.clearRect(0,0,cv.width,cv.height);
 P.forEach(p=>{const[x,y]=xy(m,p[m.xi],p[m.yi]);g.globalAlpha=sel<0?.7:(p[3]===sel?.95:.1);g.fillStyle=col[p[2]];
  g.beginPath();g.arc(x,y,(sel===p[3]?4:3)*d,0,7);g.fill()});
 g.globalAlpha=1;
 if(hover>=0){const[x,y]=xy(m,P[hover][m.xi],P[hover][m.yi]);g.strokeStyle=bg;g.lineWidth=4*d;g.beginPath();g.arc(x,y,7*d,0,7);g.stroke();
  g.strokeStyle=css("--hl");g.lineWidth=2*d;g.beginPath();g.arc(x,y,7*d,0,7);g.stroke()}
 g.setLineDash([4*d,4*d]);g.strokeStyle=css("--ink2");g.lineWidth=1*d;g.font=`${10.5*d}px system-ui,sans-serif`;
 g.fillStyle=css("--ink2");g.textAlign="left";g.textBaseline="bottom";
 m.isl.forEach(b=>{const[x0,y0]=xy(m,b.x0,b.y1),[x1,y1]=xy(m,b.x1,b.y0);g.strokeRect(x0,y0,x1-x0,y1-y0);
  g.fillText(`moved closer: really ${b.f}× farther away`,Math.min(x0,cv.width-200*d),y0-3*d)});
 g.setLineDash([]);
 if(!showLabels&&sel<0)return;
 const order=CH.filter(c=>showLabels||c.id===sel).sort((a,b)=>(b.id===sel)-(a.id===sel)||b.n-a.n);
 placeLabels(g,order.map(c=>{const[x,y]=xy(m,c[m.cx],c[m.cy]);return{x,y,text:c.name,r:0,bold:c.id===sel}}),d,bg,ink,css("--ink2"))}
MAPS.forEach(m=>{const cv=m.cv;
 cv.onmousemove=e=>{const r=cv.getBoundingClientRect(),d=devicePixelRatio||1,mx=(e.clientX-r.left)*d,my=(e.clientY-r.top)*d;
  let best=-1,bd=(9*d)**2;P.forEach((p,i)=>{const[x,y]=xy(m,p[m.xi],p[m.yi]),dd=(x-mx)**2+(y-my)**2;if(dd<bd){bd=dd;best=i}});
  if(best!==hover){hover=best;drawAll()}if(best<0){tip.style.display="none";return}const p=P[best];
  tip.innerHTML=`<div class="t">${esc(p[4])}</div><div class="m">${esc(p[5])}…</div>`;tip.style.display="block";
  const tw=tip.offsetWidth,th=tip.offsetHeight;tip.style.left=Math.min(e.clientX+14,innerWidth-tw-8)+"px";tip.style.top=Math.min(e.clientY+14,innerHeight-th-8)+"px"};
 cv.onmouseleave=()=>{hover=-1;tip.style.display="none";drawAll()};
 cv.onclick=()=>{if(hover>=0)select(P[hover][3])}});
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
addEventListener("resize",()=>{MAPS.forEach(size);drawAll()});matchMedia("(prefers-color-scheme: dark)").addEventListener("change",drawAll);
MAPS.forEach(size);drawAll();
</script></body></html>"""


if __name__ == "__main__":
    main()
