"""Chunking report: what the final Czech and English chunks look like and how they differ from the earlier chunking.

    python Chunks/report.py          # from the repo root, after Chunks/chunk.py

Writes Chunks/output/chunking_report.html (self-contained; copied to FinalReports/ by Chunks/build_reports.py)
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "output"
SOURCES = ["article", "video", "book"]
LABEL = {"article": "Articles", "video": "Videos", "book": "Books"}
COLOR = {"article": "var(--s0)", "book": "var(--s1)", "video": "var(--s2)"}


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def pct(v):
    return f"{v:.0%}"


def q(v, p):
    return float(np.percentile(v, p))


def hist_svg(cs):
    """Small multiples: words per chunk, one panel per source, log-spaced bins."""
    bins = np.array([1, 2, 5, 10, 20, 50, 100, 150, 200, 300, 500, 800])
    labels = ["1", "2–4", "5–9", "10–19", "20–49", "50–99", "100–149", "150–199", "200–299", "300–499", "500+"]
    w, h, pl, pb, pt, gap = 300, 170, 30, 44, 22, 18
    panels = []
    for k, src in enumerate(SOURCES):
        v = np.array([r["n_words"] for r in cs if r["source"] == src])
        cnt = np.histogram(np.clip(v, 1, 799), bins)[0]
        sh = cnt / len(v)
        top = max(sh.max() * 1.15, 0.05)
        bw = (w - pl - 6) / len(cnt)
        x0 = k * (w + gap)
        el = [f'<text x="{x0 + pl}" y="14" class="lab">{LABEL[src]} (n = {len(v):,})</text>']
        for t in (0.1, 0.2, 0.3, 0.4):
            if t < top:
                y = h - pb - (h - pb - pt) * t / top
                el.append(f'<line x1="{x0 + pl}" x2="{x0 + w - 6}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                          f'<text x="{x0 + pl - 4}" y="{y + 3:.1f}" text-anchor="end" class="ax">{t:.0%}</text>')
        for i, s in enumerate(sh):
            bh = (h - pb - pt) * s / top
            el.append(f'<rect x="{x0 + pl + i * bw + 1:.1f}" y="{h - pb - bh:.1f}" width="{bw - 2:.1f}" height="{bh:.1f}" '
                      f'rx="2" fill="{COLOR[src]}"><title>{LABEL[src]}, {labels[i]} words: {cnt[i]} chunks ({s:.1%})</title></rect>')
            if i % 2 == 0:
                el.append(f'<text x="{x0 + pl + i * bw + bw / 2:.1f}" y="{h - pb + 12}" text-anchor="end" class="ax" '
                          f'transform="rotate(-40 {x0 + pl + i * bw + bw / 2:.1f} {h - pb + 12})">{labels[i]}</text>')
        el.append(f'<line x1="{x0 + pl}" x2="{x0 + w - 6}" y1="{h - pb}" y2="{h - pb}" class="base"/>')
        panels.append("".join(el))
    W = 3 * w + 2 * gap
    return (f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="Words per chunk by source" '
            f'style="width:100%;height:auto">{"".join(panels)}</svg>')


def years_svg(cs):
    """Stacked bars: article and video chunks per year."""
    yrs = sorted({r["year"] for r in cs if r["source"] != "book" and r["year"]})
    cnt = {s: Counter(r["year"] for r in cs if r["source"] == s and r["year"]) for s in ("article", "video")}
    tot = [cnt["article"][y] + cnt["video"][y] for y in yrs]
    w, h, pl, pb, pt = 900, 230, 40, 26, 12
    top = max(tot) * 1.1
    bw = (w - pl - 8) / len(yrs)
    el = []
    step = 50 if top < 300 else 100
    for t in range(step, int(top) + 1, step):
        y = h - pb - (h - pb - pt) * t / top
        el.append(f'<line x1="{pl}" x2="{w - 8}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                  f'<text x="{pl - 6}" y="{y + 4:.1f}" text-anchor="end" class="ax">{t}</text>')
    for i, yv in enumerate(yrs):
        x = pl + i * bw + 3
        base = h - pb
        for s in ("article", "video"):
            c = cnt[s][yv]
            bh = (h - pb - pt) * c / top
            el.append(f'<rect x="{x:.1f}" y="{base - bh:.1f}" width="{bw - 6:.1f}" height="{bh:.1f}" fill="{COLOR[s]}">'
                      f'<title>{yv}: {c} {LABEL[s].lower()}</title></rect>')
            base -= bh + (1 if c else 0)
        el.append(f'<text x="{x + (bw - 6) / 2:.1f}" y="{h - pb + 16}" text-anchor="middle" class="ax">{yv}</text>')
    el.append(f'<line x1="{pl}" x2="{w - 8}" y1="{h - pb}" y2="{h - pb}" class="base"/>')
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Article and video chunks per year" '
            f'style="width:100%;height:auto">{"".join(el)}</svg>')


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(ROOT / "Projection"))
    from book_map import chapter_group  # noqa: E402
    cs = pq.read_table(OUT / "chunks_cs.parquet").to_pylist()
    en = pq.read_table(OUT / "chunks_en.parquet").to_pylist()
    pipe = pq.read_table(OUT / "chunks.parquet").to_pylist()
    old = pq.read_table(ROOT / "Preprocessing" / "output" / "chunks.parquet",
                        columns=["source", "doc_id", "words", "kind"]).to_pylist()
    assert [r["chunk_id"] for r in cs] == [r["chunk_id"] for r in en] == [r["chunk_id"] for r in pipe]

    rows = []
    for s in SOURCES:
        c = [r for r in cs if r["source"] == s]
        e = [r for r in en if r["source"] == s]
        o = [r for r in old if r["source"] == s]
        wc, we, wo = [r["n_words"] for r in c], [r["n_words"] for r in e], [r["words"] for r in o]
        rows.append({"src": s, "n": len(c), "docs": len({r["doc_id"] for r in c}),
                     "cs_med": q(wc, 50), "cs_p90": q(wc, 90), "cs_max": max(wc), "en_med": q(we, 50),
                     "old_n": len(o), "old_docs": len({r["doc_id"] for r in o}), "old_med": q(wo, 50)})
    ratio = np.array([b["n_words"] / max(a["n_words"], 1) for a, b in zip(cs, en)])
    same = sum(1 for a, b in zip(cs, en) if a["text"] == b["text"])
    tiny = [r for r in cs if r["n_words"] <= 3]
    en_text = {r["chunk_id"]: r["text"] for r in en}
    dup = Counter(r["text"] for r in cs if r["source"] == "article")
    dups = [r for r in cs if r["source"] == "article" and dup[r["text"]] > 1]
    kinds = Counter((r["source"], r["kind"]) for r in pipe)
    raw_v = json.load(open(ROOT / "preprocessed" / "yt_tr.json", encoding="utf-8"))
    raw_w = sum(len(d[1].split()) for d in raw_v)
    kept_w = sum(r["n_words"] for r in cs if r["source"] == "video")
    raw_med = float(np.median([len(d[1].split()) for d in raw_v]))
    groups = Counter(chapter_group({"chapter": r["chapter_en"], "subchapter": r["subchapter_en"]})   # English names
                     for r in pipe if r["source"] == "book")
    gtop = groups.most_common(6)
    ex = {s: next(r for r in pipe if r["source"] == s and r["words"] > 40) for s in SOURCES}

    def row_html(r):
        return (f"<tr><td><span class=sw style='background:{COLOR[r['src']]}'></span>{LABEL[r['src']]}</td>"
                f"<td class=r>{r['n']:,}</td><td class=r>{r['docs']:,}</td><td class=r>{r['cs_med']:.0f}</td>"
                f"<td class=r>{r['cs_p90']:.0f}</td><td class=r>{r['cs_max']:,}</td><td class=r>{r['en_med']:.0f}</td>"
                f"<td class=r>{r['old_n']:,}</td><td class=r>{r['old_med']:.0f}</td></tr>")
    R = {r["src"]: r for r in rows}
    fill = {
        "__N__": f"{len(cs):,}", "__ROWS__": "".join(row_html(r) for r in rows),
        "__NA__": f"{R['article']['n']:,}", "__NV__": f"{R['video']['n']:,}", "__NB__": f"{R['book']['n']:,}",
        "__NOLD__": f"{len(old):,}", "__OLDB__": f"{R['book']['old_n']:,}", "__OLDBMED__": f"{R['book']['old_med']:.0f}",
        "__BMED__": f"{R['book']['cs_med']:.0f}", "__AMED__": f"{R['article']['cs_med']:.0f}",
        "__VMED__": f"{R['video']['cs_med']:.0f}", "__AMAX__": f"{R['article']['cs_max']:,}",
        "__VMAX__": f"{R['video']['cs_max']:,}", "__BMAX__": f"{R['book']['cs_max']:,}",
        "__OLDA__": f"{R['article']['old_n']:,}", "__OLDV__": f"{R['video']['old_n']:,}",
        "__HIST__": hist_svg(cs), "__YEARS__": years_svg(cs),
        "__RMED__": f"{np.median(ratio):.2f}", "__RLO__": f"{q(ratio, 5):.2f}", "__RHI__": f"{q(ratio, 95):.2f}",
        "__SAME__": str(same), "__NTINY__": str(len(tiny)),
        "__TINY__": "".join(f"<tr><td><code>{esc(r['chunk_id'])}</code></td><td>{LABEL[r['source']][:-1]}</td>"
                            f"<td>{esc(en_text[r['chunk_id']][:90])}</td><td class=m>{esc(r['text'][:90])}</td></tr>"
                            for r in tiny),
        "__NDUP__": str(len(dups)), "__NDUPT__": str(sum(1 for t, c in dup.items() if c > 1)),
        "__NANN__": str(kinds[("article", "announcement")]), "__NART__": str(kinds[("article", "article")]),
        "__NSTORY__": str(kinds[("book", "book_story")]), "__NFAQ__": str(kinds[("book", "book_faq")]),
        "__NGROUPS__": str(len(groups)), "__BPCUT__": pct(1 - kept_w / raw_w), "__RAWVMED__": f"{raw_med:.0f}",
        "__GTOP__": ", ".join(f"“{esc(g)}” ({c})" for g, c in gtop),
        "__TAGGED__": str(sum(1 for r in pipe if r["source"] == "video" and r["tags"])),
        "__EX__": "".join(
            f"<tr><td>{LABEL[s][:-1]}</td><td><code>{esc(ex[s]['embed_text'].split(chr(10))[0][:110])}</code></td>"
            f"<td><code>{esc(ex[s]['embed_text_nosource'].split(chr(10))[0][:110])}</code></td><td>(none)</td></tr>"
            for s in SOURCES),
    }
    page = TEMPLATE
    for k, v in fill.items():
        page = page.replace(k, v)
    (OUT / "chunking_report.html").write_text(page, encoding="utf-8")
    print(f"Wrote {OUT / 'chunking_report.html'}")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Final chunking</title>
<style>
:root{--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--s0:#2a78d6;--s1:#eb6834;--s2:#1baf7a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835;--s0:#3987e5;--s1:#d95926;--s2:#199e70}}
:root[data-theme="dark"]{--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835;--s0:#3987e5;--s1:#d95926;--s2:#199e70}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}p.sub,p.cap{color:var(--ink2);max-width:900px}p.cap{font-size:13px;margin:8px 2px 0}
h2.sec{font-size:17px;margin:28px 0 6px;padding-top:12px;border-top:1px solid var(--line)}
ol,ul{max-width:900px;padding-left:20px}li{margin:6px 0}code{background:var(--line);padding:1px 5px;border-radius:4px;font-size:12px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.kpis{display:flex;gap:12px;flex-wrap:wrap;margin:8px 0}.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;min-width:170px}
.kpi b{display:block;font-size:20px}.kpi span{color:var(--ink2);font-size:12px}
table{border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:8px;font-size:12.5px;width:100%}
th,td{padding:5px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{color:var(--ink2);font-weight:600}
td.r,th.r{text-align:right;font-variant-numeric:tabular-nums}.tw{overflow-x:auto}
.sw{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:6px;vertical-align:-1px}
svg .grid{stroke:var(--line);stroke-width:1}svg .base{stroke:var(--ink2);stroke-width:1}
svg .ax{fill:var(--ink2);font-size:10px}svg .lab{fill:var(--ink);font-size:11.5px;font-weight:600}
</style></head><body><div class="wrap">
<h1>Final chunking <span style="font-weight:400;color:var(--ink2)">· Czech and English, aligned</span></h1>
<p class="sub">The corpus behind the final reports: Urza's articles, video descriptions and two books, in Czech
(embedded) and English (translation, aligned chunk by chunk). One simple rule per source decides what a chunk is.</p>
<div class="kpis">
 <div class="kpi"><b>__N__</b><span>chunks per language</span></div>
 <div class="kpi"><b>__NA__ / __NV__ / __NB__</b><span>article / video / book chunks</span></div>
 <div class="kpi"><b>0</b><span>chunks where Czech and English don't line up</span></div>
</div>

<h2 class="sec">What was done</h2>
<ol>
 <li><strong>Input</strong> (<code>preprocessed/</code>): articles with text, all video descriptions, and the two books
 split into sections, each in Czech and in an English translation of the same items in the same order.</li>
 <li><strong>Rules</strong> (<code>Chunks/chunk.py</code>): an article's text is one chunk; a video's description is
 one chunk; in the books, every paragraph (text between blank lines) is one chunk. Nothing is split further or
 merged. From video descriptions, the recurring boilerplate is removed first: every line found in 20 or more
 descriptions (donation details, crypto addresses and bank account, Urza's signature line, separators), separately
 in each language. That was __BPCUT__ of all video words (median description __RAWVMED__ words before, __VMED__ after);
 left in, it would make videos resemble each other through the same lines rather than their content.</li>
 <li><strong>Alignment:</strong> row <em>i</em> of <code>chunks_cs</code> and <code>chunks_en</code> is the same chunk
 with the same <code>chunk_id</code> (<code>art-0000</code>, <code>vid-0000</code>, <code>AC-033-009</code> =
 book, section, paragraph). The script stops if the number of items, sections or paragraphs ever differs.</li>
 <li><strong>Metadata from the earlier pipeline:</strong> every article and video matches an earlier document by
 URL, so each chunk keeps its kind (article, announcement, video; __NANN__ announcements), video tags (__TAGGED__
 videos), series and guests; video dates come from the channel metadata. Book chapters are stored the same way
 as before, giving the same __NGROUPS__ chapter groups.</li>
 <li><strong>Three embedding texts</strong>, as in all earlier reports; only the Czech chunks are embedded
 (bge-m3), each variant separately:
 <div class="tw" style="margin-top:6px"><table><thead><tr><th>Source</th><th><code>withsource</code>: line in front (Czech, as embedded)</th><th><code>nosource</code>: line in front</th><th><code>textonly</code></th></tr></thead><tbody>__EX__</tbody></table></div></li>
</ol>

<h2 class="sec">Chunks by source, compared with the earlier chunking</h2>
<div class="tw"><table><thead><tr><th>Source</th><th class=r>Chunks</th><th class=r>Documents</th><th class=r>Median words (CZ)</th><th class=r>90th pct</th><th class=r>Max</th><th class=r>Median words (EN)</th><th class=r>Earlier: chunks</th><th class=r>Earlier: median words</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<p class="cap"><strong>What it shows:</strong> articles and videos change little: the earlier pipeline also kept
almost every article (__OLDA__ chunks) and video (__OLDV__) whole, splitting only a handful of long ones, and it
also removed the video boilerplate; now the
longest article has __AMAX__ words and the longest description __VMAX__, both well within what the model reads
(about 8,000 tokens). The books change most: one chunk per paragraph gives __NB__ chunks with a median of __BMED__
words, against __OLDB__ chunks of about __OLDBMED__ words before, so each book chunk is closer to a single
argument.</p>

<h2 class="sec">How long the chunks are</h2>
<div class="panel">__HIST__</div>
<p class="cap"><strong>How to read it:</strong> share of each source's chunks by number of words (Czech).
<strong>What it shows:</strong> articles (median __AMED__ words) and video descriptions (median __VMED__) are
spread widely, from a hashtag to several hundred words; book paragraphs are shorter and more even (median __BMED__,
longest __BMAX__). Very short book chunks are mostly lines of dialogue in the opening stories and the headings of
the money chapter.</p>

<h2 class="sec">Articles and videos over time</h2>
<div class="panel">__YEARS__</div>
<p class="cap"><strong>How to read it:</strong> article (blue) and video (green) chunks per year; one chunk is one
article or one video. Books have no year here. <strong>What it shows:</strong> the change from writing to video
over the years, which the timeline report follows in detail.</p>

<h2 class="sec">Czech and English</h2>
<p class="cap" style="margin-top:0"><strong>Lengths:</strong> an English chunk has a median of __RMED__ × the words
of its Czech original (90% between __RLO__ and __RHI__), as expected for a translation (English uses more, shorter
words). <strong>Untranslated:</strong> __SAME__ chunks are identical in both languages (names, links or
English-language texts). Only the Czech chunks are embedded; the English twin of any chunk is found by its
<code>chunk_id</code>.</p>

<h2 class="sec">Odd chunks kept by the rules</h2>
<div class="tw"><table><thead><tr><th>Chunk</th><th>Source</th><th>Text (English)</th><th>Czech original</th></tr></thead><tbody>__TINY__</tbody></table></div>
<p class="cap"><strong>What it shows:</strong> the __NTINY__ chunks of three words or fewer: hashtag-only articles,
link-only video descriptions, headings in the books and single lines of dialogue. They follow the rules and are
kept; on the maps they land wherever their few words point. In addition, __NDUP__ article chunks share their text
with another article (__NDUPT__ texts), also kept.</p>

<h2 class="sec">Book chunks by section</h2>
<p class="cap" style="margin-top:0"><strong>What it shows:</strong> the books split into __NGROUPS__ chapter groups,
the same as in the earlier reports. Paragraph chunking changes their weight: the largest are __GTOP__. The
opening stories (“Three stories”, __NSTORY__ chunks) are dialogue, so nearly every line is its own paragraph;
they now weigh much more in the book maps than before. The two AMEN sections (objections and aphorisms) have __NFAQ__ chunks.</p>

<h2 class="sec">Conclusions</h2>
<ul>
 <li><strong>The chunks are simple and traceable:</strong> one article, one video description or one book paragraph,
 with the same ID in Czech and English.</li>
 <li><strong>Articles and videos are as before</strong> (video boilerplate removed, as before); the books are now chunked by paragraph, finer than
 before (__NB__ instead of __OLDB__ chunks).</li>
 <li><strong>The earlier analysis carries over:</strong> the same kinds, tags and chapter groups, and the same
 three embedding variants, so the final reports can be read side by side with the earlier ones.</li>
 <li><strong>Watch for:</strong> the opening stories' many short dialogue chunks, and __NTINY__ near-empty chunks.</li>
</ul>
</div></body></html>"""


if __name__ == "__main__":
    main()
