"""Coverage of the general political space: which political topics, questions and concepts does Urza address?

    python Questions/embed_questions.py      # once: embeds Questions/political_space.tsv
    python Projection/political_space.py     # from the repo root

The political space (Questions/political_space.tsv) is independent of Urza: 22 policy areas built on
the Comparative Agendas Project topics, 85 subtopics, 168 concrete questions and 73 concepts of political
philosophy. Urza's chunks (textonly embeddings) are measured against it in two ways:
  volume  each article/book/video chunk is assigned to the nearest subtopic centre (source-centred vectors)
  depth   for each topic/question/concept, the similarity of Urza's best-matching passage (raw vectors,
          as for any search query)

Writes Projection/output/political_space_textonly.html (+ _map.png)
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import LABEL_JS, place_labels  # noqa: E402
from book_map import esc  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import CHUNKS, ENGLISH, OUT_DIR, nm, read_chunks, results_dir  # noqa: E402
OUT = OUT_DIR
VARIANT = "textonly"
SOURCES = ["article", "book", "video"]
QTEXT = {}   # item id -> shown text (for the caveat questions)
STRONG, SOME = 0.60, 0.55     # depth tiers (cosine of best passage); set from the validation examples
K_PLACE = 5                   # nearest items used to place a chunk on the map
SEED = 42
# spot-check questions: the first N_CORE are on Urza's core subjects (drugs, schooling, crypto, taxes, guns, roads),
# the rest on subjects he treats only in passing (refugees, climate, abortion, Ukraine)
CHECK = ["Q12.4a", "Q06.1a", "Q15.4c", "Q01.2b", "Q12.5a", "Q10.1a", "Q09.1a", "Q07.1a", "Q03.4b", "Q16.3a"]
N_CORE = 6


def unit(A):
    return A / np.linalg.norm(A, axis=-1, keepdims=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import umap
    items = list(csv.DictReader(open(ROOT / "Questions" / "political_space.tsv", encoding="utf-8"), delimiter="\t"))
    en_tsv = ROOT / "Questions" / "political_space_en.tsv"
    if ENGLISH and en_tsv.exists():   # show the English translation; the embeddings stay those of the Czech items
        en = list(csv.DictReader(open(en_tsv, encoding="utf-8"), delimiter="\t"))
        assert [q["id"] for q in en] == [q["id"] for q in items], "political_space_en.tsv does not match the Czech list"
        items = [{**q, "description": q.get("description") or ""} for q in en]
    QTEXT.update({q["id"]: q["text"] for q in items})
    QE = np.load(ROOT / "Questions" / "output" / "embeddings.npy")
    assert (ROOT / "Questions" / "output" / "ids.txt").read_text(encoding="utf-8").split() == [q["id"] for q in items]
    rows = read_chunks(CHUNKS, ["chunk_id", "doc_id", "source", "kind", "year", "title", "url", "chapter",
                                  "text"])
    E = np.load(results_dir(VARIANT) / "embeddings.npy")
    assert (results_dir(VARIANT) / "chunk_ids.txt").read_text(encoding="utf-8").split() \
        == [r["chunk_id"] for r in rows]
    src = np.array([r["source"] for r in rows])
    keep = np.array([r["kind"] != "announcement" for r in rows])
    K = np.where(keep)[0]
    layer = np.array([q["layer"] for q in items])
    major = np.array([q["major"] for q in items])
    sub = np.array([q["sub"] for q in items])

    # ---- depth: best passage per item (raw vectors, like a search query)
    S = QE @ E[K].T                                  # [items, kept chunks]
    best_j = S.argmax(1)
    best = S.max(1)
    docs_strong = []
    for i in range(len(items)):
        hits = np.where(S[i] >= SOME)[0]
        docs_strong.append(len({rows[K[j]]["doc_id"] for j in hits}))
    top3 = np.argsort(-S, axis=1)[:, :3]

    def passage(j):
        r = rows[K[j]]
        return [SOURCES.index(r["source"]), r["title"], r.get("url") or "", int(r["year"] or 0), r["text"][:220]]

    # ---- volume: chunks to nearest subtopic centre (source-centred chunks, centred items)
    X = E.copy()
    for s in SOURCES:
        X[src == s] -= X[src == s].mean(0)
    QX = QE - QE.mean(0)
    subs = [q["sub"] for q in items if q["layer"] == "topic"]
    smaj = {q["sub"]: q["major"] for q in items if q["layer"] == "topic"}
    Cs = unit(np.array([QX[(sub == s) & (layer != "concept")].mean(0) for s in subs]))
    assign = np.array(subs, dtype=object)[(unit(X[K]) @ Cs.T).argmax(1)]
    ksrc = src[K]

    # ---- map: UMAP of the political space, chunks placed among their nearest items
    import warnings
    warnings.filterwarnings("ignore")
    # the map: political items only, each item type (topic / question / concept) centred on its own average, so the
    # layout follows what the items are about rather than how each type is worded; fixed from here on
    QT = QE.copy()
    for l in ("topic", "question", "concept"):
        QT[layer == l] -= QE[layer == l].mean(0)
    um = umap.UMAP(n_neighbors=10, min_dist=0.25, metric="cosine", random_state=SEED).fit(QT)
    P = um.embedding_.copy()
    nc = layer != "concept"

    def nb_rate(Pm, lab):                            # share of an item's 10 nearest map neighbours with the same label
        D = ((Pm[:, None] - Pm[None]) ** 2).sum(-1)
        np.fill_diagonal(D, np.inf)
        return float((lab[np.argsort(D, 1)[:, :10]] == lab[:, None]).mean())
    Praw = umap.UMAP(n_neighbors=10, min_dist=0.25, metric="cosine", random_state=SEED).fit_transform(QE)
    layout = {"type": nb_rate(P, layer), "area": nb_rate(P[nc], major[nc]),
              "type_raw": nb_rate(Praw, layer), "area_raw": nb_rate(Praw[nc], major[nc]),
              "type_ch": float(sum((layer == l).mean() ** 2 for l in set(layer))),
              "area_ch": float(sum((major[nc] == m).mean() ** 2 for m in set(major[nc])))}
    SC = unit(X[K]) @ unit(QT).T                     # chunk-to-item similarity, both centred
    nn = np.argsort(-SC, axis=1)[:, :K_PLACE]
    w = np.take_along_axis(SC, nn, 1)
    w = np.exp((w - w.max(1, keepdims=True)) * 20)
    CP = (P[nn] * w[..., None]).sum(1) / w.sum(1, keepdims=True)     # placement 1: among the nearest items
    # placement 2: UMAP's own transform into the fitted map. Both sides are centred (items per type, chunks per
    # source), so neither the wording of the items nor Urza's style decides where a chunk lands.
    CT = um.transform(X[K])
    # placement 3: the same transform, but with only one average removed from all of Urza's chunks (his overall
    # "Urza-ness"); the differences between articles, books and videos stay in
    CG = um.transform(E[K] - E[K].mean(0))
    lo = np.minimum.reduce([P.min(0), CT.min(0), CG.min(0)])
    span = np.maximum.reduce([P.max(0), CT.max(0), CG.max(0)]) - lo
    norm = lambda A: (A - lo) / span.max()
    P, CP, CT, CG = norm(P), norm(CP), norm(CT), norm(CG)
    jit = np.random.default_rng(SEED).normal(0, 0.006, CP.shape)
    CP += jit
    CT += jit
    CG += jit
    pstats = {}
    ilo, ihi = P.min(0), P.max(0)
    src_i = np.array([SOURCES.index(x) for x in ksrc])
    pstats["src_ch"] = float(sum((src_i == k).mean() ** 2 for k in range(3)))
    for name, C in (("knn", CP), ("tr", CT), ("g", CG)):
        g = np.floor((C - ilo) / (ihi - ilo) * 40).clip(0, 39).astype(int)
        cnt = np.bincount(g[:, 0] * 40 + g[:, 1], minlength=1600)
        near = np.sqrt(((C[:, None] - P[None]) ** 2).sum(-1)).argmin(1)       # nearest item on the map
        pstats[name] = {"cells": int((cnt > 0).sum()), "dense": float(np.sort(cnt)[::-1][:80].sum() / len(C)),
                        "outside": float(((C < ilo - 0.02) | (C > ihi + 0.02)).any(1).mean()),
                        "faith": float(np.mean([near[k] in nn[k] for k in range(len(C))]))}
        Dc = ((C[:, None] - C[None]) ** 2).sum(-1)              # same-source share among a chunk's 10 map neighbours
        np.fill_diagonal(Dc, np.inf)
        pstats[name]["same_src"] = float((src_i[np.argsort(Dc, 1)[:, :10]] == src_i[:, None]).mean())
        del Dc
    pstats["moved"] = float(np.median(np.sqrt(((CP - CT) ** 2).sum(1))))
    pstats["moved_g"] = float(np.median(np.sqrt(((CG - CT) ** 2).sum(1))))      # third placement vs second

    # ---- aggregates
    majors = list(dict.fromkeys(q["major"] for q in items if q["layer"] != "concept"))
    n = len(K)
    areas = []
    for m in majors:
        msubs = [s for s in subs if smaj[s] == m]
        cnt = {s_: int(((np.isin(assign, msubs)) & (ksrc == s_)).sum()) for s_ in SOURCES}
        its = np.where((major == m) & (layer != "concept"))[0]
        subrows = []
        for s in msubs:
            ii = np.where((sub == s) & (layer != "concept"))[0]
            subrows.append({"name": s, "vol": int((assign == s).sum()) / n,
                            "cnt": {s_: int(((assign == s) & (ksrc == s_)).sum()) for s_ in SOURCES},
                            "depth": float(best[ii].mean()),
                            "items": [[items[i]["id"], items[i]["layer"], items[i]["text"], round(float(best[i]), 3),
                                       docs_strong[i], passage(best_j[i])] for i in ii]})
        areas.append({"name": m, "vol": sum(cnt.values()) / n, "cnt": cnt, "depth": float(best[its].mean()),
                      "strong": float(np.mean(best[its] >= STRONG)), "some": float(np.mean(best[its] >= SOME)),
                      "subs": subrows})
    q_idx = np.where(layer == "question")[0]
    c_idx = np.where(layer == "concept")[0]
    rank = lambda ix, rev: sorted(ix, key=lambda i: -best[i] if rev else best[i])
    qrow = lambda i: [items[i]["id"], items[i]["major"] if items[i]["layer"] != "concept" else items[i]["sub"],
                      items[i]["text"], round(float(best[i]), 3), docs_strong[i], passage(best_j[i])]
    checks = [[items[i]["text"], [[round(float(S[i, j]), 3)] + passage(j) for j in top3[i]]]
              for i in [ [q["id"] for q in items].index(c) for c in CHECK ]]

    pts_items = [[round(float(P[i, 0]), 4), round(float(P[i, 1]), 4), ["topic", "question", "concept"].index(layer[i]),
                  items[i]["text"], items[i]["major"] if layer[i] != "concept" else items[i]["sub"], round(float(best[i]), 3),
                  docs_strong[i], passage(best_j[i])] for i in range(len(items))]
    pts_chunks = [[round(float(CP[k, 0]), 4), round(float(CP[k, 1]), 4), SOURCES.index(ksrc[k]),
                   rows[K[k]]["title"], int(rows[K[k]]["year"] or 0),
                   round(float(CT[k, 0]), 4), round(float(CT[k, 1]), 4),
                   round(float(CG[k, 0]), 4), round(float(CG[k, 1]), 4)] for k in range(n)]
    labels = []
    for m in majors:   # label on the area's most central item (medoid): areas are not compact on the map
        A = P[(major == m) & (layer != "concept")]
        med = A[np.argmin(np.sqrt(((A[:, None] - A[None]) ** 2).sum(-1)).sum(1))]
        labels.append({"name": m, "x": float(med[0]), "y": float(med[1])})

    OUT.mkdir(exist_ok=True)
    png(P, layer, best, CP, ksrc, labels, OUT / f"political_space_{VARIANT}_map.png",
        "Urza's chunks (placed among their nearest items)")
    png(P, layer, best, CT, ksrc, labels, OUT / f"political_space_{VARIANT}_map_transform.png",
        "Urza's chunks (UMAP transform, each source's average removed)")
    png(P, layer, best, CG, ksrc, labels, OUT / f"political_space_{VARIANT}_map_transform_urza.png",
        "Urza's chunks (UMAP transform, one average of all his texts removed)")
    stats = {
        "n": n, "n_items": len(items), "n_q": int(len(q_idx)), "n_c": int(len(c_idx)), "n_t": int((layer == "topic").sum()),
        "n_major": len(majors), "used_subs": int(len(set(assign))), "n_subs": len(subs),
        "q_strong": float(np.mean(best[q_idx] >= STRONG)), "q_some": float(np.mean(best[q_idx] >= SOME)),
        "c_strong": float(np.mean(best[c_idx] >= STRONG)),
        "src_n": {s: int((ksrc == s).sum()) for s in SOURCES}, "place": pstats, "layout": layout,
    }
    html(stats, areas, [qrow(i) for i in rank(q_idx, True)[:15]], [qrow(i) for i in rank(q_idx, False)[:15]],
         [qrow(i) for i in rank(c_idx, True)[:10]], [qrow(i) for i in rank(c_idx, False)[:10]], checks,
         pts_items, pts_chunks, labels, OUT / f"political_space_{VARIANT}.html")
    print("areas by volume:", [(a["name"], f"{a['vol']:.1%}", f"depth {a['depth']:.2f}") for a in sorted(areas, key=lambda a: -a["vol"])])
    print(f"questions strong {stats['q_strong']:.0%}, some {stats['q_some']:.0%}; concepts strong {stats['c_strong']:.0%}")
    print("placement:", pstats)
    print(f"Wrote {OUT / f'political_space_{VARIANT}.html'}")


def png(P, layer, best, CP, ksrc, labels, path, chunk_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 11), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.scatter(CP[:, 0], CP[:, 1], s=4, c="#9a9993", alpha=0.25, linewidths=0, label=chunk_label)
    sc = ax.scatter(P[:, 0], P[:, 1], s=[60 if l == "topic" else 26 for l in layer], c=best, cmap="Blues", vmin=0.45,
                    vmax=0.72, edgecolors="#fcfcfb", linewidths=0.8, marker="o")
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("depth: similarity of Urza's best passage", fontsize=9, color="#52514e")
    ax.set_xticks([]); ax.set_yticks([])
    for spn in ax.spines.values():
        spn.set_color("#e4e3df")
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="upper left", fontsize=9, markerscale=3)
    ax.set_title("The political space (topics, questions, concepts) coloured by how deeply Urza covers them",
                 loc="left", fontsize=11.5)
    place_labels(ax, [(lab["x"], lab["y"], lab["name"]) for lab in labels], fontsize=8)
    fig.savefig(path, dpi=150, facecolor="#fcfcfb", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def html(st, areas, q_top, q_bot, c_top, c_bot, checks, pts_items, pts_chunks, labels, path):
    p = lambda v: f"{v * 100:.0f}%"
    q = lambda s: f"“{esc(s)}”"
    by_vol = sorted(areas, key=lambda a: -a["vol"])
    by_depth = sorted(areas, key=lambda a: -a["depth"])
    j = lambda o: json.dumps(o, ensure_ascii=False).replace("</", "<\\/")
    fill = {"__LABELJS__": LABEL_JS, "__ROADQ__": esc(QTEXT["Q10.1a"]), "__TAXQ__": esc(QTEXT["Q01.2b"]),
        "__AREAS__": j(areas), "__QTOP__": j(q_top), "__QBOT__": j(q_bot), "__CTOP__": j(c_top), "__CBOT__": j(c_bot),
        "__CHECKS__": j(checks), "__ITEMS__": j(pts_items), "__CHUNKS__": j(pts_chunks), "__LABELS__": j(labels),
        "__N__": f"{st['n']:,}", "__NI__": str(st["n_items"]), "__NQ__": str(st["n_q"]), "__NC__": str(st["n_c"]),
        "__NT__": str(st["n_t"]), "__NM__": str(st["n_major"]), "__USED__": str(st["used_subs"]), "__NS__": str(st["n_subs"]),
        "__QSTRONG__": p(st["q_strong"]), "__QSOME__": p(st["q_some"]), "__CSTRONG__": p(st["c_strong"]),
        "__NA__": f"{st['src_n']['article']:,}", "__NB__": f"{st['src_n']['book']:,}", "__NV__": f"{st['src_n']['video']:,}",
        "__STRONG__": f"{STRONG:.2f}", "__SOME__": f"{SOME:.2f}",
        "__VOLTOP__": ", ".join(f"{q(a['name'])} ({p(a['vol'])})" for a in by_vol[:4]),
        "__VOLBOT__": ", ".join(f"{q(a['name'])} ({p(a['vol'])})" for a in by_vol[-4:]),
        "__DEPTOP__": ", ".join(f"{q(a['name'])} ({a['depth']:.2f})" for a in by_depth[:4]),
        "__DEPBOT__": ", ".join(f"{q(a['name'])} ({a['depth']:.2f})" for a in by_depth[-4:]),
        "__BLIND__": ", ".join(q(r[2]) for r in q_bot[:6]),
        "__KCELLS__": str(st["place"]["knn"]["cells"]), "__TCELLS__": str(st["place"]["tr"]["cells"]),
        "__KDENSE__": p(st["place"]["knn"]["dense"]), "__TDENSE__": p(st["place"]["tr"]["dense"]),
        "__TOUT__": f"{st['place']['tr']['outside'] * 100:.0f}%",
        "__KFAITH__": p(st["place"]["knn"]["faith"]), "__TFAITH__": p(st["place"]["tr"]["faith"]),
        "__MOVED__": f"{st['place']['moved']:.2f}",
        "__GCELLS__": str(st["place"]["g"]["cells"]), "__GDENSE__": p(st["place"]["g"]["dense"]),
        "__GOUT__": f"{st['place']['g']['outside'] * 100:.0f}%", "__GFAITH__": p(st["place"]["g"]["faith"]),
        "__SSK__": p(st["place"]["knn"]["same_src"]), "__SST__": p(st["place"]["tr"]["same_src"]),
        "__SSG__": p(st["place"]["g"]["same_src"]), "__SSCH__": p(st["place"]["src_ch"]),
        **check_text(checks, q), **place_text(st["place"]),
        "__BLINDNO__": ("have no passage that reaches " + f"{SOME:.2f}" + ", i.e. none that addresses them directly"
                        if max(r[3] for r in q_bot[:6]) < SOME else "have only weakly matching passages"),
        "__BLINDNO2__": ("get no passage that addresses them directly" if max(r[3] for r in q_bot[:6]) < SOME
                         else "get only weakly matching passages"),
        "__LTYPE__": p(st["layout"]["type"]), "__LTYPERAW__": p(st["layout"]["type_raw"]), "__LTYPECH__": p(st["layout"]["type_ch"]),
        "__LAREA__": p(st["layout"]["area"]), "__LAREARAW__": p(st["layout"]["area_raw"]), "__LAREACH__": p(st["layout"]["area_ch"]),
    }
    page = TEMPLATE
    for k, v in fill.items():
        page = page.replace(k, v)
    path.write_text(page, encoding="utf-8")


def check_text(checks, q):
    """Spot-check caption from the actual best passages (core subjects vs subjects treated in passing)."""
    core, rest = checks[:N_CORE], checks[N_CORE:]
    top = lambda cs: [c[1][0][0] for c in cs]
    rng = lambda v: f"{min(v):.2f}–{max(v):.2f}"
    on = [c for c in core if c[1][0][0] >= SOME]
    titles = list(dict.fromkeys(c[1][0][2] for c in on))[:4]
    txt = (f"the {len(core)} questions on his core subjects find, as best passage, his own texts on those subjects "
           f"(such as {', '.join(q(t) for t in titles)}), at similarities of {rng(top(core))}.")
    if rest:
        lower = max(top(rest)) < min(top(core))
        txt += (f" For the {len(rest)} questions on subjects he treats only in passing, his best passages score "
                f"{'lower' if lower else 'mostly lower'} ({rng(top(rest))}).")
    return {"__CHECKTXT__": txt}


def place_text(pl):
    """Placement captions, worded from the placement statistics."""
    k, t, g = pl["knn"], pl["tr"], pl["g"]
    p = lambda v: f"{v * 100:.0f}%"
    if k["outside"] == 0:
        kout = "never puts a chunk outside the area of the political items"
    else:
        kout = f"puts {p(k['outside'])} of chunks outside the area of the political items"
    out = (f"it lets some fall away from the items ({p(t['outside'])} land outside the items' area)" if t["outside"] > 0
           else "it keeps all of them within the items' area")
    if t["cells"] > k["cells"] and t["dense"] > k["dense"]:
        spread = (f"reaches more cells ({t['cells']}) but also piles more chunks onto a few spots "
                  f"({p(t['dense'])} in the densest 5%)")
    else:
        spread = f"reaches {t['cells']} cells (densest 5%: {p(t['dense'])})"
    trtxt = f"<em>UMAP transform</em> {spread}, and {out}."
    diff = abs(g["same_src"] - t["same_src"])
    if diff < 0.05:
        gtxt = ("It keeps the differences between sources, but they barely show: ")
        gend = ("So once Urza's overall average is removed, the remaining differences between articles, books and "
                "videos hardly change where his texts land in the political space")
    else:
        gtxt = "It keeps the differences between sources, and they show: "
        gend = ("So the differences between articles, books and videos do change where his texts land in the "
                "political space")
    if pl["moved_g"] < pl["moved"] / 2:
        gend += (f"; a chunk moves by a median of only {pl['moved_g']:.2f} map widths between this placement and the "
                 f"second one (against {pl['moved']:.2f} between the first two).")
    else:
        gend += (f"; a chunk moves by a median of {pl['moved_g']:.2f} map widths between this placement and the second "
                 f"one (against {pl['moved']:.2f} between the first two).")
    return {"__KOUT__": kout, "__TRTXT__": trtxt, "__GTXT__": gtxt, "__GEND__": gend}


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Political space coverage</title>
<style>
:root{--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--s0:#2a78d6;--s1:#eb6834;--s2:#1baf7a;--dim:#c9c8c3;--hl:#0b0b0b;
 --q1:#cde2fb;--q2:#86b6ef;--q3:#3987e5;--q4:#1c5cab}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835;--s0:#3987e5;--s1:#d95926;--s2:#199e70;--dim:#4a4a46;--hl:#f0efec;
 --q1:#184f95;--q2:#256abf;--q3:#5598e7;--q4:#9ec5f4}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1200px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}p.sub,p.cap{color:var(--ink2);max-width:900px}p.cap{font-size:13px;margin:8px 2px 0}
h2.sec{font-size:17px;margin:28px 0 6px;padding-top:12px;border-top:1px solid var(--line)}
ol,ul{max-width:900px;padding-left:20px}li{margin:6px 0}
code{background:var(--line);padding:1px 5px;border-radius:4px;font-size:12.5px}a{color:inherit}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.kpis{display:flex;gap:12px;flex-wrap:wrap;margin:8px 0}.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;min-width:170px}
.kpi b{display:block;font-size:20px}.kpi span{color:var(--ink2);font-size:12px}
.mapwrap{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:16px;align-items:start}
canvas{width:100%;display:block;cursor:crosshair}
.controls{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:0 0 10px;font-size:13px}
.controls label{display:inline-flex;gap:6px;align-items:center;cursor:pointer}
.sw{width:10px;height:10px;border-radius:50%;display:inline-block;flex:none;margin-right:4px;vertical-align:-1px}
.side{font-size:12.5px;max-height:760px;overflow:auto}.side h3{font-size:14px;margin:0 0 4px}.side .muted{color:var(--ink2)}
.psg{border-top:1px solid var(--line);padding:6px 0}.psg b{font-weight:600}
.ramp{display:inline-flex;gap:2px;vertical-align:middle}.ramp span{width:22px;height:8px;border-radius:2px}
table{border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:8px;font-size:12.5px;width:100%}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{color:var(--ink2);font-weight:600}
td.r,th.r{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.tw{overflow-x:auto}
.row{display:grid;grid-template-columns:240px minmax(0,1fr) 64px 64px;gap:10px;align-items:center;padding:4px 0;font-size:12.5px;cursor:pointer;border-bottom:1px solid var(--line)}
.row:hover{background:var(--bg)}.row.subr{padding-left:18px;color:var(--ink2);cursor:default;grid-template-columns:222px minmax(0,1fr) 64px 64px}
.stack{display:flex;gap:2px;height:12px}.stack span{display:block;height:12px}.stack span:last-child{border-radius:0 4px 4px 0}
.dep{text-align:center;border-radius:4px;font-variant-numeric:tabular-nums;padding:1px 0}
.det{grid-column:1/-1;display:none}.det.on{display:block}
.head{display:grid;grid-template-columns:240px minmax(0,1fr) 64px 64px;gap:10px;font-size:12px;color:var(--ink2);padding:0 0 4px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:7px;overflow:hidden}.seg button{background:var(--card);color:var(--ink2);border:0;padding:4px 10px;cursor:pointer;font:inherit}.seg button.on{background:var(--line);color:var(--ink);font-weight:600}
@media (max-width:900px){.mapwrap,.two{grid-template-columns:1fr}.row,.head{grid-template-columns:140px minmax(0,1fr) 52px 52px}}
</style></head><body><div class="wrap">
<h1>Urza in the political space <span style="font-weight:400;color:var(--ink2)">· which general political topics does he address? · <code>textonly</code></span></h1>
<p class="sub">Everything so far measured Urza against himself. Here the reference is independent of him: a map of
politics in general (__NM__ policy areas, __NT__ subtopics, __NQ__ concrete questions, __NC__ concepts of political
philosophy). Urza's __N__ chunks (__NA__ article, __NB__ book, __NV__ video) are projected into that space to see
which parts of it he attends to, how deeply, and where his blind spots are.</p>
<div class="kpis">
 <div class="kpi"><b>__QSTRONG__</b><span>of questions with a strongly matching Urza passage (≥ __STRONG__)</span></div>
 <div class="kpi"><b>__QSOME__</b><span>of questions with at least a relevant passage (≥ __SOME__)</span></div>
 <div class="kpi"><b>__USED__ / __NS__</b><span>subtopics that some of his texts are closest to</span></div>
</div>

<h2 class="sec">What was done</h2>
<ol>
 <li><strong>The political space</strong> (<code>Questions/political_space.tsv</code>, editable): the policy areas
 of the Comparative Agendas Project, an academic classification of what governments deal with, adapted to Czech
 politics and extended by “Morality and personal freedom”; under each area its subtopics, under each subtopic 1–3
 concrete questions as Czech voters meet them; and a separate layer of concepts (state, legitimacy, social
 contract, ideologies, rights, justice, economics, ethics). __NI__ items in total, drafted for this report.</li>
 <li><strong>Embedding:</strong> the same model as Urza's texts (bge-m3), computed on this machine; a test showed
 identical vectors to the GPU run (cosine 1.0000). Topics and concepts are embedded as “name: description”,
 questions as written.</li>
 <li><strong>Depth</strong> (per topic, question, concept): the similarity of Urza's single best-matching passage,
 exactly as a search query would rank them. From the checks below, passages above __SOME__ are typically on
 topic and above __STRONG__ squarely on it; these thresholds are a reading aid, not a hard boundary. Also counted:
 how many different texts reach __SOME__.</li>
 <li><strong>Volume</strong> (per area): every chunk is assigned to the nearest subtopic, whose centre is the
 average of the subtopic and its questions. As in the earlier reports, each source's average vector is subtracted
 first, so the style of book prose or video descriptions doesn't decide the outcome. Announcements are left out.</li>
 <li><strong>Map:</strong> the political items alone are laid out with UMAP, after subtracting the average vector
 of each item type separately (topics, questions, concepts). Each type is worded in its own way (a question versus
 a “name: description” entry), and without this the map sorts items by type rather than by subject, the same
 format effect as between Urza's sources. The map is then fixed, and
 Urza's chunks are added into it without refitting, in one of three ways (switch above the map):
 <em>among nearest items</em>, at a similarity-weighted average of the positions of the five items each chunk is most
 similar to; <em>UMAP transform</em>, UMAP's own method for placing new points into a fitted map, with each chunk's
 source average subtracted (as everywhere in this report: this removes both what all of Urza's texts share and
 what is typical of articles, books or videos); or <em>UMAP transform, one Urza average</em>, the same but with a
 single average of all his chunks subtracted, which removes his overall “Urza-ness” and keeps the differences
 between the three sources.</li>
</ol>

<h2 class="sec">Does it find the right texts? Spot checks</h2>
<p class="sub">Questions on subjects Urza is known for, and his three best-matching passages:</p>
<div id="checks"></div>
<p class="cap"><strong>Reading it:</strong> __CHECKTXT__</p>

<h2 class="sec">The map of the political space</h2>
<div class="controls"><span>Urza's chunks:</span>
<label><input type="checkbox" data-s="0" checked><span class="sw" style="background:var(--s0)"></span>articles</label>
<label><input type="checkbox" data-s="1" checked><span class="sw" style="background:var(--s1)"></span>books</label>
<label><input type="checkbox" data-s="2" checked><span class="sw" style="background:var(--s2)"></span>videos</label>
<label><input type="checkbox" id="lab" checked>Area labels</label>
<span>Placement</span><span class="seg" id="place"><button data-v="0" class="on">among nearest items</button><button data-v="1">UMAP transform</button><button data-v="2">UMAP transform, one Urza average</button></span>
<span>Item depth: <span class="ramp"><span style="background:var(--q1)"></span><span style="background:var(--q2)"></span><span style="background:var(--q3)"></span><span style="background:var(--q4)"></span></span> weak → strong</span></div>
<div class="mapwrap">
 <div class="panel"><canvas id="cv"></canvas></div>
 <div class="panel side" id="side"><h3>Hover a topic, question or concept</h3><div class="muted">Large dots: subtopics · medium: questions · squares: concepts. Colour = depth of Urza's best passage. Small dots behind: Urza's chunks.</div></div>
</div>
<p class="cap"><strong>How to read it:</strong> nearby items are about related things. A policy area is often spread over several parts of the map (questions about the state touch many subjects), so each area label sits on its most central item. Because each item type was centred separately, items no longer group by how they are worded: __LTYPE__ of an item's 10 nearest neighbours on the map are of the same type (chance __LTYPECH__; __LTYPERAW__ without the centring), and __LAREA__ of a topic's or question's neighbours belong to the same policy area (chance __LAREACH__; __LAREARAW__ without the centring). The depth and volume numbers do not depend on the map. Dark blue items are ones Urza
addresses squarely, pale ones only weakly; the cloud of small dots shows where his texts gather. Hover an item for
its best passages.</p>
<p class="cap"><strong>Nearest items vs UMAP transform:</strong> in all three placements, the political items stay
exactly where they are; only Urza's chunks are placed differently. <em>Among nearest items</em> spreads them over __KCELLS__ of 1,600 grid cells
(the 5% densest cells hold __KDENSE__ of the chunks) and __KOUT__.
__TRTXT__ A chunk moves by a median of __MOVED__ map widths between
the two. Neither keeps a chunk next to the items it is most similar to: the item closest to a chunk on the map is one
of its five most similar items for only __KFAITH__ (nearest items) and __TFAITH__ (transform) of chunks, because a 2D
map cannot keep __NI__ items and thousands of passages in their true neighbourhoods. Read both maps for where his
texts gather overall; for which items they are about, use the depth numbers and the hover panel, which do not depend
on the placement.</p>
<p class="cap"><strong>One Urza average vs one average per source:</strong> the third placement removes only what
all of Urza's texts share. It reaches __GCELLS__ cells (densest 5%: __GDENSE__), __GOUT__ land outside the items'
area, and the nearest item on the map is one of the five most similar for __GFAITH__ of chunks. __GTXT__among a chunk's 10 nearest chunks on the map, __SSG__ come
from the same source (articles, books or videos), against __SST__ with the per-source averages removed, __SSK__ for
“among nearest items”, and __SSCH__ if sources were mixed at random. __GEND__</p>

<h2 class="sec">Coverage by policy area</h2>
<div class="controls"><span>Sort</span><span id="sortb"></span></div>
<div class="panel"><div class="head"><span>Area (click to expand)</span><span>Volume: share of Urza's chunks</span><span style="text-align:center">Volume</span><span style="text-align:center">Depth</span></div><div id="areas"></div></div>
<p class="cap"><strong>How to read it:</strong> the bar is how much of Urza's output falls into the area, split
into articles, books and videos; depth is the average similarity of his best passage over the area's topics and
questions (darker = more squarely covered). The two can differ: an area can get few texts that answer its
questions precisely, or many texts that only touch it.</p>
<p class="cap"><strong>What it shows:</strong> by volume, Urza writes and speaks most about __VOLTOP__, and least
about __VOLBOT__. By depth, he answers questions best in __DEPTOP__, and least in __DEPBOT__.</p>

<h2 class="sec">Best covered and blind spots</h2>
<div class="two"><div><h3 style="font-size:14px;margin:0 0 6px">Best-covered questions</h3><div id="qtop"></div></div>
<div><h3 style="font-size:14px;margin:0 0 6px">Least covered questions (blind spots)</h3><div id="qbot"></div></div></div>
<p class="cap"><strong>Reading it:</strong> “texts” counts how many different articles, book sections or videos
reach __SOME__ for the question. The least covered questions, such as __BLIND__, __BLINDNO__;
the nearest one is shown to make the gap visible.</p>
<div class="two" style="margin-top:16px"><div><h3 style="font-size:14px;margin:0 0 6px">Best-covered concepts</h3><div id="ctop"></div></div>
<div><h3 style="font-size:14px;margin:0 0 6px">Least covered concepts</h3><div id="cbot"></div></div></div>

<h2 class="sec">Caveats</h2>
<ul>
 <li><strong>The list defines the result.</strong> It was drafted for this report and is one reasonable map of
 politics, not the only one. It deliberately includes a few classic questions from the anarcho-capitalist
 debate (“__ROADQ__”, “__TAXQ__”), which Urza naturally covers well. Edit
 <code>Questions/political_space.tsv</code> and rerun both scripts to change it.</li>
 <li><strong>Similarity is not agreement.</strong> A high match means Urza discusses the question, not which
 answer he gives.</li>
 <li><strong>Videos are represented by their descriptions</strong>, so subjects he talks about at length in
 videos but doesn't name in the description are under-counted.</li>
 <li><strong>Volume depends partly on how finely an area is divided:</strong> an area with more subtopics offers
 more centres to be closest to (the state and public administration has 6, migration 2). Depth doesn't have this
 problem, since it is an average over the area's items.</li>
 <li><strong>Thresholds are a reading aid.</strong> The depth values are continuous; __SOME__ and __STRONG__ were
 chosen from the spot checks.</li>
</ul>

<h2 class="sec">Conclusions</h2>
<ul>
 <li><strong>Urza covers a large part of the political space:</strong> __QSOME__ of the questions have at least a
 relevant passage and __QSTRONG__ a strongly matching one; his texts reach __USED__ of the __NS__ subtopics.</li>
 <li><strong>His attention is uneven, in a recognisable way:</strong> most of it goes to __VOLTOP__; the deepest
 answers are in __DEPTOP__.</li>
 <li><strong>Blind spots</strong> lie in areas of conventional policy-making; the lowest depth is in __DEPBOT__.
 Questions such as __BLIND__ __BLINDNO2__. An anarchist framework tends to answer them only
 through general arguments about the state.</li>
 <li><strong>Next step for UrzaLLM:</strong> the blind-spot questions are where a debating assistant would have
 to reason from Urza's general principles rather than quote him, and where it should say so.</li>
</ul>
</div><div id="tip"></div>
<script>
__LABELJS__
const AREAS=__AREAS__, QTOP=__QTOP__, QBOT=__QBOT__, CTOP=__CTOP__, CBOT=__CBOT__, CHECKS=__CHECKS__, IT=__ITEMS__, CH=__CHUNKS__, LAB=__LABELS__;
const SRC=["article","book","video"], PAD=16;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const depthVar=d=>d>=0.65?"--q4":d>=0.6?"--q3":d>=0.55?"--q2":"--q1";
const link=p=>`<span class="sw" style="background:var(--s${p[0]})"></span>${p[2]?`<a href="${esc(p[2])}" target="_blank" rel="noopener">${esc(p[1])}</a>`:esc(p[1])}${p[3]?` (${p[3]})`:""}`;
// spot checks
document.getElementById("checks").innerHTML=`<div class="tw"><table><thead><tr><th>Question</th><th>Best passages</th><th class="r">Similarity</th></tr></thead><tbody>`+
 CHECKS.map(c=>c[1].map((p,k)=>`<tr>${k?"":`<td rowspan="3"><strong>${esc(c[0])}</strong></td>`}<td>${link(p.slice(1))}</td><td class="r">${p[0].toFixed(2)}</td></tr>`).join("")).join("")+"</tbody></table></div>";
// question / concept tables
function qtable(id,rows){document.getElementById(id).innerHTML=`<div class="tw"><table><thead><tr><th>Item</th><th class="r">Depth</th><th class="r">Texts</th></tr></thead><tbody>`+
 rows.map(r=>`<tr><td>${esc(r[2])}<div style="color:var(--ink2);font-size:11.5px">${esc(r[1])} · best: ${link(r[5])}</div></td><td class="r"><span class="dep" style="background:var(${depthVar(r[3])});color:${r[3]>=0.6?"#fff":"inherit"};padding:1px 6px">${r[3].toFixed(2)}</span></td><td class="r">${r[4]}</td></tr>`).join("")+"</tbody></table></div>"}
qtable("qtop",QTOP);qtable("qbot",QBOT);qtable("ctop",CTOP);qtable("cbot",CBOT);
// areas
let sortBy="vol";const openA=new Set();
const sb=document.getElementById("sortb");[["vol","Volume"],["depth","Depth"],["name","Name"]].forEach(([k,l])=>{const b=document.createElement("button");b.textContent=l;
 b.style.cssText="background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:3px 10px;margin-right:6px;cursor:pointer";b.onclick=()=>{sortBy=k;renderAreas()};sb.appendChild(b)});
function stack(cnt,tot,mx){const t=SRC.reduce((a,s)=>a+cnt[s],0)||1;return `<div class="stack" style="width:${100*tot/mx}%">${SRC.map((s,i)=>cnt[s]?`<span style="flex:${100*cnt[s]/t};background:var(--s${i})"></span>`:"").join("")}</div>`}
function renderAreas(){const a=[...AREAS].sort((x,y)=>sortBy==="name"?x.name.localeCompare(y.name):y[sortBy]-x[sortBy]);const mx=Math.max(...AREAS.map(x=>x.vol));
 document.getElementById("areas").innerHTML=a.map((x,k)=>{const o=openA.has(x.name);return `<div class="row" data-n="${esc(x.name)}" title="${esc(x.name)}: ${SRC.map(s=>x.cnt[s]+" "+s+"s").join(", ")}"><span>${o?"▾":"▸"} ${esc(x.name)}</span>${stack(x.cnt,x.vol,mx)}<span class="dep">${(x.vol*100).toFixed(1)}%</span><span class="dep" style="background:var(${depthVar(x.depth)});color:${x.depth>=0.6?"#fff":"inherit"}">${x.depth.toFixed(2)}</span></div>`+
  (o?x.subs.map(s=>`<div class="row subr"><span>${esc(s.name)}</span>${stack(s.cnt,s.vol,mx)}<span class="dep">${(s.vol*100).toFixed(1)}%</span><span class="dep" style="background:var(${depthVar(s.depth)});color:${s.depth>=0.6?"#fff":"inherit"}">${s.depth.toFixed(2)}</span>
   <div class="det on" style="font-size:12px;padding:2px 0 6px">${s.items.map(it=>`<div>${it[1]==="question"?"❓":"•"} ${esc(it[2])} <span style="color:var(--ink2)">— ${it[3].toFixed(2)}, best: ${link(it[5])}</span></div>`).join("")}</div></div>`).join(""):"")}).join("");
 document.querySelectorAll("#areas .row[data-n]").forEach(r=>r.onclick=()=>{const n=r.dataset.n;openA.has(n)?openA.delete(n):openA.add(n);renderAreas()})}
renderAreas();
// map
const cv=document.getElementById("cv"),side=document.getElementById("side");const on=[true,true,true];let showLab=true,hover=-1;
document.querySelectorAll("input[data-s]").forEach(i=>i.onchange=()=>{on[+i.dataset.s]=i.checked;draw()});
let pl=0;document.querySelectorAll("#place button").forEach(b=>b.onclick=()=>{document.querySelectorAll("#place button").forEach(x=>x.classList.toggle("on",x===b));pl=+b.dataset.v;draw()});document.getElementById("lab").onchange=e=>{showLab=e.target.checked;draw()};
const W=Math.max(...IT.map(p=>p[0]),...CH.map(p=>Math.max(p[0],p[5],p[7]))),H=Math.max(...IT.map(p=>p[1]),...CH.map(p=>Math.max(p[1],p[6],p[8])));
function size(){const r=cv.getBoundingClientRect(),d=devicePixelRatio||1,h=(r.width-2*PAD)*H/W+2*PAD;cv.style.height=h+"px";cv.width=r.width*d;cv.height=h*d}
function xy(x,y){const d=devicePixelRatio||1,k=(cv.width-2*PAD*d)/W;return[PAD*d+x*k,cv.height-PAD*d-y*k]}
function draw(){const g=cv.getContext("2d"),d=devicePixelRatio||1,col=[css("--s0"),css("--s1"),css("--s2")],bg=css("--card");g.clearRect(0,0,cv.width,cv.height);
 CH.forEach(p=>{if(!on[p[2]])return;const[x,y]=xy(p[2*pl+(pl?3:0)],p[2*pl+(pl?4:1)]);g.globalAlpha=.28;g.fillStyle=col[p[2]];g.beginPath();g.arc(x,y,1.8*d,0,7);g.fill()});g.globalAlpha=1;
 IT.forEach((p,i)=>{const[x,y]=xy(p[0],p[1]),r=(p[2]===0?6.5:p[2]===1?4.2:4.6)*d;g.fillStyle=css(depthVar(p[5]));g.strokeStyle=bg;g.lineWidth=1.5*d;g.beginPath();
  if(p[2]===2){g.rect(x-r,y-r,2*r,2*r)}else{g.arc(x,y,r,0,7)}g.fill();g.stroke();if(i===hover){g.strokeStyle=css("--hl");g.lineWidth=2*d;g.beginPath();g.arc(x,y,r+4*d,0,7);g.stroke()}});
 if(showLab)placeLabels(g,LAB.map(l=>{const[x,y]=xy(l.x,l.y);return{x,y,text:l.name,r:6.5*d,bold:true}}),d,bg,css("--ink"),css("--ink2"))}
cv.onmousemove=e=>{const r=cv.getBoundingClientRect(),d=devicePixelRatio||1,mx=(e.clientX-r.left)*d,my=(e.clientY-r.top)*d;let best=-1,bd=(10*d)**2;
 IT.forEach((p,i)=>{const[x,y]=xy(p[0],p[1]),dd=(x-mx)**2+(y-my)**2;if(dd<bd){bd=dd;best=i}});if(best===hover)return;hover=best;draw();if(best<0)return;const p=IT[best];
 side.innerHTML=`<h3>${esc(p[3])}</h3><div class="muted">${["Subtopic","Question","Concept"][p[2]]} · ${esc(p[4])}</div><p>Depth <b>${p[5].toFixed(2)}</b> · ${p[6]} texts reach ${"__SOME__"}</p><div class="psg"><b>Best passage:</b> ${link(p[7])}<div class="muted" style="margin-top:4px">${esc(p[7][4])}…</div></div>`};
addEventListener("resize",()=>{size();draw()});matchMedia("(prefers-color-scheme: dark)").addEventListener("change",draw);size();draw();
</script></body></html>"""


if __name__ == "__main__":
    main()
