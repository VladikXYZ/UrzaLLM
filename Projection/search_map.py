"""Search the map: type a topic or question, see the top-k passages highlighted on the two maps.

    python Projection/search_map.py              # from the repo root, then open http://127.0.0.1:8765
    python Projection/search_map.py --port 9000 --device cpu
    python Projection/search_map.py --corpus cs  # without the video transcripts

Corpus (default `all`): Vladimir's real_data (real_data/chunked.json, 2,911 chunks: the two books,
stoky.urza.cz articles and YouTube video descriptions, Czech + English; bge-m3 vectors of the Czech text in
real_data/embeddings_out_cs/) plus the 57,693 chunks of Urza's speech from the video transcripts
(real_data/transcripts_urza.json from Transcription/; bge-m3 vectors in embeddings_out_rollo_cs/). Shown on
two UMAP maps: raw vectors and source-centred vectors (each source's average subtracted). Passages are
shown in English where a translation exists (the transcripts are Czech until translated); a transcript
hit also shows the question it answers. bge-m3 is multilingual, so a query can be in Czech or English.

A query is embedded with the same model as the chunks (BAAI/bge-m3), scored against the raw vectors by
cosine similarity, and the top k passages are highlighted.

Where the question lands: UMAP is not re-fitted, so the question is placed at the similarity-weighted
average position of its PLACE_K best passages on each map (weights exp((s - s_best) / PLACE_TAU), so the
closest passages dominate). This is how UMAP itself initialises new points; if the best passages sit in
two distant regions, the point lands between them, and the lines to the hits show that.

Map coordinates: real_data/projection/coords_bge_m3_<corpus>.parquet, made by Projection/real_data_coords.py
(UMAP does not import in the main Python on this machine, so they are computed once and kept in git).

Needs  sentence-transformers + torch (Embedding/requirements.txt) and the model in the local Hugging Face
       cache. It loads offline; on a machine without the model, run once with --online (~2.3 GB download).
"""
import argparse
import gzip
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_data_coords import TRANSCRIPTS, chunk_ids, coords_path, emb_paths, source_of  # noqa: E402

DATA = ROOT / "real_data"
BOOKS_EN = {"ancap": "Anarcho-capitalism", "anag": "Anarchoagorism"}
KIND = {"book": "chapter", "article": "article", "video": "video description", "transcript": "Urza in a video"}
# --model: the vectors in the folders, the model that embeds the query, its query prompt (the chunks were
# embedded as raw text, without prompts: re-embedding reproduces the stored vectors exactly)
MODELS = {"bge_m3": {"hf": "BAAI/bge-m3", "prompt": None},
          "embeddinggemma_2": {"hf": "google/embeddinggemma-2", "prompt": "task: search result | query: "}}
SOURCES = ["article", "book", "video", "transcript"]
PLACE_K = 10
PLACE_TAU = 0.02
MAX_K = 200


def norm01(P):
    """The same scaling as projection_compare.py, so positions match that page."""
    lo, hi = P.min(0), P.max(0)
    return (P - lo) / (hi - lo).max()


def load_maps(corpus, model="bge_m3"):
    """(chunk ids, [raw map, centred map]) scaled to [0, 1] like projection_compare.py."""
    path = coords_path(model, corpus)
    if not path.exists():
        sys.exit(f"No map coordinates: {path} not found. Run Projection/real_data_coords.py.")
    c = pq.read_table(path).to_pylist()
    return ([r["chunk_id"] for r in c], [norm01(np.array([[r["x_raw"], r["y_raw"]] for r in c])),
                                         norm01(np.array([[r["x_centred"], r["y_centred"]] for r in c]))])


def load_rows(with_transcripts=True):
    """real_data/chunked.json (+ the transcript chunks) as rows for the page, in English where translated."""
    chunks = json.loads((DATA / "chunked.json").read_text(encoding="utf-8"))
    dates = {}
    for y in json.loads((DATA / "yt_full.json").read_text(encoding="utf-8")):      # date: 20221231
        d = str(y.get("date") or "")
        dates[y["url"]] = f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 else d
    for a in json.loads((DATA / "arts_full.json").read_text(encoding="utf-8")):    # date: 2022-12-31 10:00:00
        dates[a["url"]] = str(a.get("date") or "")[:10]
    rows = []
    for cid, c in zip(chunk_ids(chunks), chunks):
        s = source_of(c)
        book = s == "book"
        rows.append({"chunk_id": cid, "source": s, "kind": KIND[s],
                     "title": (c["subtitle_en"] or c["title_en"]) if book else c["title_en"],
                     "book": BOOKS_EN.get(c["type"], BOOKS_EN.get(c["src"], c["src"])) if book else "",
                     "chapter": c["title_en"] if book else "",
                     "subchapter": c["subtitle_en"] if book else "",
                     "date": dates.get(c["src"], ""), "url": c["src"] if c["src"].startswith("http") else None,
                     "text": c["text_en"], "text_cs": c["text_cs"], "guests": [], "words": len(c["text_cs"].split())})
    if with_transcripts:
        tr = json.loads(TRANSCRIPTS.read_text(encoding="utf-8"))
        ids = chunk_ids(chunks + tr)[len(chunks):]          # the same ids as real_data_coords.py
        for cid, c in zip(ids, tr):
            # two formats exist: Transcription/to_chunked_format.py (id, part, url with &t=, ...) and the later
            # one with time/end_time, "part 2/5" in subtitle_en and text_en a Czech placeholder until translated
            translated = c.get("translation_status", "translated" if c.get("text_en") else "") == "translated"
            en = c.get("text_en") if translated else ""
            t = c.get("start", c.get("time"))
            url = c["src"] if "&t=" in c["src"] or t is None else f"{c['src']}&t={int(float(t))}s"
            ctx = c.get("context_en") if translated and c.get("context_en") else c.get("context_cs")
            rows.append({"chunk_id": cid, "source": "transcript", "kind": KIND["transcript"],
                         "title": c.get("title_en") or c["title_cs"], "book": "", "chapter": "", "subchapter": "",
                         "path": c.get("subtitle_en") or "",
                         "date": c.get("date") or dates.get(c["src"].split("&")[0], ""), "url": url,
                         "text": en or c["text_cs"], "text_cs": c["text_cs"] if en else None,
                         "lang_note": "" if en else "Czech, not translated yet",
                         "context": {"speaker": c.get("context_speaker") or "", "text": ctx} if ctx else None,
                         "housekeeping": bool(c.get("housekeeping")),
                         "guests": [], "words": c.get("n_words") or len(c["text_cs"].split())})
    return rows


class Index:
    def __init__(self, device, corpus="all", online=False, model="bge_m3", query_prompt=True):
        rows = load_rows(corpus == "all")
        self.english = True
        map_ids, self.maps = load_maps(corpus, model)
        if map_ids != [r["chunk_id"] for r in rows]:
            sys.exit("The chunks and the map coordinates don't hold the same chunks in the same order: "
                     f"rerun Projection/real_data_coords.py --model {model}.")
        self.rows = rows
        emb_cs, emb_tr = emb_paths(model)
        embs = [np.load(emb_cs)] + ([np.load(emb_tr)] if corpus == "all" else [])
        self.emb = np.concatenate(embs).astype(np.float32)
        if len(self.emb) != len(rows):
            sys.exit(f"{len(self.emb)} vectors for {len(rows)} chunks.")
        self.emb /= np.linalg.norm(self.emb, axis=1, keepdims=True)
        self.src = np.array([r["source"] for r in rows])
        # "announcements" filter: housekeeping chunks of the transcripts (next lecture, donations, goodbyes)
        self.announce = np.array([r["kind"] == "announcement" or r.get("housekeeping", False) for r in rows])
        # Offline by default: with the model cached as pytorch_model.bin, transformers otherwise starts a
        # background download of a 2.3 GB safetensors copy. Must be set before huggingface_hub is imported.
        if not online:
            os.environ["HF_HUB_OFFLINE"] = "1"
        import torch
        from sentence_transformers import SentenceTransformer
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        hf = MODELS[model]["hf"]
        self.model_name, self.prompt = hf, (MODELS[model]["prompt"] if query_prompt else None)
        print(f"Loading {hf} on {device} …", flush=True)
        t = time.time()
        try:                                   # float32: bfloat16 is slow on CPUs
            self.model = SentenceTransformer(hf, device=device, model_kwargs={"dtype": torch.float32})
        except OSError as e:
            if online:
                raise
            sys.exit(f"{hf} is not in the local Hugging Face cache ({e.__class__.__name__}). "
                     "Run once with --online to download it.")
        self.device = device
        print(f"  ready in {time.time() - t:.0f} s", flush=True)

    def points(self):
        out = []
        for i, r in enumerate(self.rows):
            path = (" › ".join(x for x in (r["book"], r["chapter"], r["subchapter"]) if x) if r["source"] == "book"
                    else r.get("path", ""))
            a, b = self.maps
            out.append([r["chunk_id"], SOURCES.index(r["source"]), r["kind"], r["title"], r["date"] or "", path,
                        round(float(a[i, 0]), 4), round(float(a[i, 1]), 4), round(float(b[i, 0]), 4), round(float(b[i, 1]), 4)])
        return out

    def search(self, query, k, sources, announcements):
        t = time.time()
        q = self.model.encode([query], normalize_embeddings=True, prompt=self.prompt)[0].astype(np.float32)
        s = self.emb @ q
        ok = np.isin(self.src, sources) & (announcements | ~self.announce)
        cand = np.flatnonzero(ok)
        order = cand[np.argsort(-s[cand])][:k]
        place = order[:PLACE_K]
        w = np.exp((s[place] - s[place].max()) / PLACE_TAU)
        w /= w.sum()
        pos = [[round(float(v), 4) for v in (w[:, None] * m[place]).sum(0)] for m in self.maps]
        hits = []
        for i in order:
            r = self.rows[i]
            hits.append({"i": int(i), "score": round(float(s[i]), 4), "text": r["text"], "url": r["url"],
                         "text_cs": r.get("text_cs") if self.english else None,
                         "context": r.get("context"), "lang_note": r.get("lang_note", ""),
                         "guests": r["guests"] or [], "words": r["words"]})
        return {"query": query, "pos": pos, "hits": hits, "ms": round((time.time() - t) * 1000),
                "candidates": int(len(cand))}


def make_handler(index, page):
    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.1 keeps the connection open after the response: with the default HTTP/1.0 the server closes it
        # right after writing, and on Windows a large page (60k points, ~9 MB) then arrived cut off
        protocol_version = "HTTP/1.1"

        def _send(self, code, body, ctype):
            data = body.encode("utf-8")
            gz = len(data) > 2048 and "gzip" in self.headers.get("Accept-Encoding", "")
            if gz:
                data = gzip.compress(data, 5)
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            if gz:
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            self.wfile.flush()

        def do_GET(self):
            path, _, query = self.path.partition("?")
            if path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            elif path == "/api/chunk":                # one chunk's text for a clicked dot (texts are not in the page)
                try:
                    i = int(dict(p.split("=", 1) for p in query.split("&") if "=" in p)["i"])
                    r = index.rows[i]
                    keys = ("chunk_id", "source", "kind", "title", "date", "url", "text", "text_cs", "context",
                            "lang_note", "words", "book", "chapter", "subchapter", "path", "housekeeping")
                    self._send(200, json.dumps({"i": i, **{k: r.get(k) for k in keys}}, ensure_ascii=False),
                               "application/json; charset=utf-8")
                except Exception as e:
                    self._send(400, json.dumps({"error": str(e)}), "application/json")
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/search":
                return self._send(404, "not found", "text/plain")
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                query = str(req.get("query", "")).strip()[:2000]
                if not query:
                    raise ValueError("empty query")
                k = max(1, min(MAX_K, int(req.get("k", 20))))
                sources = [s for s in req.get("sources", SOURCES) if s in SOURCES] or SOURCES
                res = index.search(query, k, sources, bool(req.get("announcements", False)))
                print(f"  {res['ms']:>5} ms  k={k}  {query[:70]!r}", flush=True)
                self._send(200, json.dumps(res, ensure_ascii=False), "application/json; charset=utf-8")
            except Exception as e:  # report the problem to the page instead of dropping the connection
                self._send(400, json.dumps({"error": str(e)}), "application/json")

        def log_message(self, *a):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to let others on your network use it")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--online", action="store_true",
                    help="allow Hugging Face downloads (first run on a machine without the model cached)")
    ap.add_argument("--corpus", choices=["all", "cs"], default="all",
                    help="all: Vladimir's chunks + the video transcript chunks (default); cs: Vladimir's chunks only")
    ap.add_argument("--model", choices=MODELS, default="bge_m3",
                    help="which embeddings to search and map: bge_m3 (default) or embeddinggemma_2")
    ap.add_argument("--no-query-prompt", action="store_true", help="embed the query without the model's query prompt")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    index = Index(args.device, args.corpus, args.online, args.model, not args.no_query_prompt)
    data = json.dumps(index.points(), ensure_ascii=False).replace("</", "<\\/")
    lang = ("Books, articles and video descriptions are shown in English (Czech original one click away); the "
            "video transcripts are Czech until translated, and each shows the question it answers. The search "
            "runs on the embeddings of the Czech texts, and a query can be in English or Czech.")
    report = str(coords_path(args.model, args.corpus).relative_to(ROOT)).replace("\\", "/")
    page = (TEMPLATE.replace("__DATA__", data).replace("__N__", f"{len(index.rows):,}")
            .replace("__MODEL__", index.model_name + (" + query prompt" if index.prompt else ""))
            .replace("__PLACE_K__", str(PLACE_K)).replace("__LANG__", lang).replace("__REPORT__", report)
            .replace("__HAS_ANN__", "true" if any(r["kind"] == "announcement" for r in index.rows) else "false"))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(index, page))
    print(f"Open http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Urza map search</title>
<style>
:root{--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--soft:#f0efec;
 --s0:#2a78d6;--s1:#eb6834;--s2:#1baf7a;--s3:#8b5cf6;--hl:#0b0b0b;--q:#c2185b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
 --line:#383835;--soft:#262624;--s0:#3987e5;--s1:#d95926;--s2:#199e70;--s3:#a78bfa;--hl:#f0efec;--q:#f06292}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1400px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px} p.sub{color:var(--ink2);margin:0 0 16px;max-width:920px}
form{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:0 0 10px}
#q{flex:1 1 380px;min-width:0;background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:10px 12px;font:inherit;font-size:15px}
#q:focus,#k:focus{outline:2px solid var(--s0);outline-offset:1px}
#k{width:72px;background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:9px 8px;font:inherit}
button{background:var(--ink);color:var(--bg);border:0;border-radius:8px;padding:10px 18px;font:inherit;font-weight:600;cursor:pointer}
button:disabled{opacity:.5;cursor:wait} button.ghost{background:transparent;color:var(--ink2);border:1px solid var(--line);font-weight:400}
.controls{display:flex;flex-wrap:wrap;gap:16px;align-items:center;margin:0 0 12px;color:var(--ink2)}
.controls label{display:inline-flex;align-items:center;gap:6px;cursor:pointer;color:var(--ink)}
.sw{width:10px;height:10px;border-radius:50%;display:inline-block;flex:none}
#status{color:var(--ink2);min-height:1.5em;margin:0 0 10px}
#status.err{color:#c62828}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.panel h2{font-size:15px;margin:0}
.detail{margin:16px 0 0;position:relative} .detail .x{position:absolute;top:8px;right:10px;padding:2px 9px}
.detail .dt{font-weight:600;font-size:15px;margin:0 2.5em 2px 0} .detail .tx,.detail .mt,.detail .q,.detail details.cs{margin-left:0}
.detail .tx{max-height:40vh;overflow:auto} .detail details.cs{display:block}
.ph{display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap;margin:0 0 8px}
.zb{display:inline-flex;gap:4px}
.zb button{background:var(--soft);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:2px 9px;font-weight:600;font-size:13px;line-height:1.6;cursor:pointer}
.zb button:hover{border-color:var(--ink2)} .zb button:disabled{opacity:.4;cursor:default}
canvas{width:100%;aspect-ratio:1/1;display:block;cursor:crosshair}
canvas.drag{cursor:grabbing}
.hint{color:var(--ink2);font-size:12.5px;margin:6px 0 0}
#tip{position:fixed;pointer-events:none;background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:8px;
 padding:8px 10px;font-size:12.5px;max-width:340px;box-shadow:0 4px 16px rgba(0,0,0,.12);display:none;z-index:9}
#tip .t{font-weight:600;margin-bottom:2px}#tip .m{color:var(--ink2)}
h2.sec{font-size:17px;margin:24px 0 8px;padding-top:12px;border-top:1px solid var(--line)}
ol#hits{list-style:none;padding:0;margin:0;display:grid;gap:8px}
#hits li{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px;cursor:pointer}
#hits li.on{border-color:var(--ink2);box-shadow:0 0 0 1px var(--ink2)}
.hh{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.rk{font-weight:700;font-variant-numeric:tabular-nums;min-width:2.2em}
.sc{font-variant-numeric:tabular-nums;color:var(--ink2);font-size:12.5px}
.bar{display:inline-block;height:6px;border-radius:3px;background:var(--ink2);opacity:.5;vertical-align:middle;margin-left:4px}
.ti{font-weight:600;flex:1 1 300px;min-width:0}
.mt{color:var(--ink2);font-size:12.5px;margin:2px 0 4px 3.2em}
.tx{margin:0 0 0 3.2em;color:var(--ink);white-space:pre-wrap;overflow-wrap:anywhere}
li:not(.open) .tx{display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
details.cs{margin:6px 0 0 3.2em;color:var(--ink2)} li:not(.open) details.cs{display:none}
.q{margin:2px 0 4px 3.2em;color:var(--ink2);font-size:13px;border-left:3px solid var(--s3);padding-left:8px}
li:not(.open) .q{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
details.cs summary{cursor:pointer;font-size:12.5px} details.cs p{white-space:pre-wrap;overflow-wrap:anywhere;margin:4px 0 0}
a{color:var(--s0)} code{background:var(--soft);padding:1px 5px;border-radius:4px;font-size:12.5px}
.legend{display:inline-flex;align-items:center;gap:6px}
@media (max-width:820px){.grid{grid-template-columns:1fr}.mt,.tx,.q{margin-left:0}}
</style></head><body><div class="wrap">
<h1>Search the map</h1>
<p class="sub">Write a topic or a question. The __N__ chunks are scored by cosine similarity to it with the same
model that embedded them (<code>__MODEL__</code>, raw vectors), and the best ones are highlighted
on the two maps (UMAP, <code>__REPORT__</code>).
<span class="legend"><svg width="14" height="14" viewBox="-7 -7 14 14"><path d="M0-6L1.8-1.8 6 0 1.8 1.8 0 6-1.8 1.8-6 0-1.8-1.8Z" fill="var(--q)"/></svg>
marks where the question lands</span>: the similarity-weighted position of its __PLACE_K__ best passages.
__LANG__</p>
<form id="f">
 <input id="q" type="search" placeholder="e.g. Should the state regulate drugs?  ·  compulsory schooling  ·  who would build the roads" autocomplete="off" autofocus>
 <label title="number of passages">top <input id="k" type="number" min="1" max="200" value="20"></label>
 <button id="go" type="submit">Search</button>
 <button id="clr" class="ghost" type="button">Clear</button>
</form>
<div class="controls" id="controls"></div>
<div id="status">Loading…</div>
<div class="grid">
 <div class="panel"><div class="ph"><h2>Raw embeddings</h2><div class="zb" data-k="0"></div></div><canvas id="c0"></canvas></div>
 <div class="panel"><div class="ph"><h2>After subtracting each source's average</h2><div class="zb" data-k="1"></div></div><canvas id="c1"></canvas></div>
</div>
<p class="hint"><b>Click any dot to read its chunk</b> (it opens under the maps). Scroll over a map to zoom at the cursor, drag to move it, double-click to fit the hits again
(or the whole map if there is no search). The buttons do the same; each map zooms on its own.</p>
<div class="panel detail" id="detail" hidden></div>
<h2 class="sec" id="rh" hidden>Top passages</h2>
<ol id="hits"></ol>
</div><div id="tip"></div>
<script>
const P=__DATA__;
const SRC=["article","book","video","transcript"],LABEL={article:"Articles",book:"Books",video:"Video descriptions",transcript:"Video transcripts (Urza)"};
const DOT=P.length>10000?1.3:2.6;
// drawing order: the 57k transcript dots first, so the ~3k book/article/video dots stay visible on top
const ORDER=[...P.keys()].sort((a,b)=>(P[a][1]===3?0:1)-(P[b][1]===3?0:1));
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const $=id=>document.getElementById(id);
let on=SRC.map(()=>true),ann=false,lines=true,zoom=true,hover=-1,res=null,rank=new Map(),sel=-1;
let view=[{x0:0,y0:0,s:1},{x0:0,y0:0,s:1}];
const ctr=$("controls");
SRC.forEach((s,i)=>{const l=document.createElement("label");
 l.innerHTML=`<input type="checkbox" checked><span class="sw" style="background:var(--s${i})"></span>${LABEL[s]} (${P.filter(p=>p[1]===i).length.toLocaleString()})`;
 l.querySelector("input").onchange=e=>{on[i]=e.target.checked;rerun()};ctr.appendChild(l)});
function opt(text,val,fn){const l=document.createElement("label");l.innerHTML=`<input type="checkbox"${val?" checked":""}>${text}`;
 l.querySelector("input").onchange=e=>fn(e.target.checked);ctr.appendChild(l)}
if(__HAS_ANN__)opt("Include housekeeping (announcements, goodbyes)",ann,v=>{ann=v;rerun()});
opt("Lines to hits",lines,v=>{lines=v;draw()});
opt("Auto-zoom to hits",zoom,v=>{zoom=v;fit();draw()});
const cv=[0,1].map(i=>$("c"+i)),tip=$("tip"),PAD=12;
function size(){cv.forEach(c=>{const r=c.getBoundingClientRect(),d=devicePixelRatio||1;c.width=r.width*d;c.height=r.width*d})}
// Map coordinates are in [0,1]; the view picks the square [x0, x0+s] × [y0, y0+s] to show.
// Zoom frames the question and the central 70% of the hits, so a few far-away hits don't shrink the
// main group back to a dot; the hits left outside are counted on the map and still reachable by their lines.
const WHOLE={x0:0,y0:0,s:1};
function hitsView(k){const qt=(a,p)=>{const s=[...a].sort((x,y)=>x-y);return s[Math.round(p*(s.length-1))]};
 if(!res||!res.hits.length)return{...WHOLE};
 const xs=res.hits.map(h=>P[h.i][6+2*k]),ys=res.hits.map(h=>P[h.i][7+2*k]),[qx,qy]=res.pos[k];
 const lx=Math.min(qx,qt(xs,.15)),hx=Math.max(qx,qt(xs,.85)),ly=Math.min(qy,qt(ys,.15)),hy=Math.max(qy,qt(ys,.85));
 const s=Math.min(1,Math.max(.08,1.6*Math.max(hx-lx,hy-ly)));return{x0:(lx+hx-s)/2,y0:(ly+hy-s)/2,s}}
function fit(){view=[0,1].map(k=>zoom?hitsView(k):{...WHOLE})}
// Manual zoom: keep the map point under (px, py) (canvas pixels) fixed while the scale changes by f.
const SMIN=.005,SMAX=2;
function zoomAt(k,f,px,py){const c=cv[k],d=devicePixelRatio||1,w=c.width-2*PAD*d,v=view[k];
 if(px===undefined){px=c.width/2;py=c.height/2}
 const ux=v.x0+(px-PAD*d)/w*v.s,uy=v.y0+(c.height-PAD*d-py)/w*v.s,s=Math.min(SMAX,Math.max(SMIN,v.s*f));
 view[k]={x0:ux-(px-PAD*d)/w*s,y0:uy-(c.height-PAD*d-py)/w*s,s};draw()}
document.querySelectorAll(".zb").forEach(box=>{const k=+box.dataset.k;
 [["+","Zoom in",()=>zoomAt(k,1/1.6)],["−","Zoom out",()=>zoomAt(k,1.6)],
  ["Fit hits","Frame the question and its hits",()=>{view[k]=hitsView(k);draw()},"fit"],
  ["Whole map","Show the whole map",()=>{view[k]={...WHOLE};draw()}]].forEach(([t,title,fn,role])=>{
  const b=document.createElement("button");b.type="button";b.textContent=t;b.title=title;b.onclick=fn;if(role)b.dataset.role=role;box.appendChild(b)})});
function buttons(){document.querySelectorAll('.zb [data-role="fit"]').forEach(b=>b.disabled=!res)}
function outside(k){const v=view[k];return res?res.hits.filter(h=>{const x=P[h.i][6+2*k],y=P[h.i][7+2*k];
 return x<v.x0||x>v.x0+v.s||y<v.y0||y>v.y0+v.s}).length:0}
function sxy(c,x,y,k){const d=devicePixelRatio||1,w=c.width-2*PAD*d,v=view[k];return[PAD*d+(x-v.x0)/v.s*w,c.height-PAD*d-(y-v.y0)/v.s*w]}
const xy=(c,p,k)=>sxy(c,p[6+2*k],p[7+2*k],k);
function star(g,x,y,r){g.beginPath();for(let j=0;j<8;j++){const a=j*Math.PI/4-Math.PI/2,rr=j%2?r*.38:r;g.lineTo(x+rr*Math.cos(a),y+rr*Math.sin(a))}g.closePath()}
function draw(){const col=SRC.map((_,i)=>css("--s"+i)),hl=css("--hl"),bg=css("--card"),qc=css("--q");
 cv.forEach((c,k)=>{const g=c.getContext("2d"),d=devicePixelRatio||1;g.clearRect(0,0,c.width,c.height);
  ORDER.forEach(i=>{const p=P[i];if(!on[p[1]])return;const[x,y]=xy(c,p,k);g.globalAlpha=res&&!rank.has(i)?.18:.7;
   g.fillStyle=col[p[1]];g.beginPath();g.arc(x,y,DOT*d,0,7);g.fill()});
  g.globalAlpha=1;
  if(res){const[qx,qy]=sxy(c,res.pos[k][0],res.pos[k][1],k);
   if(lines){g.strokeStyle=qc;g.lineWidth=1*d;res.hits.forEach((h,r)=>{const[x,y]=xy(c,P[h.i],k);
    g.globalAlpha=Math.max(.08,.45*(1-r/res.hits.length));g.beginPath();g.moveTo(qx,qy);g.lineTo(x,y);g.stroke()});g.globalAlpha=1}
   [...res.hits].reverse().forEach(h=>{const p=P[h.i],[x,y]=xy(c,p,k),r=rank.get(h.i);
    g.fillStyle=col[p[1]];g.strokeStyle=bg;g.lineWidth=1.5*d;g.beginPath();g.arc(x,y,(r<10?5.5:4.2)*d,0,7);g.fill();g.stroke();
    if(r<10){g.fillStyle="#fff";g.font=`600 ${7.5*d}px system-ui,sans-serif`;g.textAlign="center";g.textBaseline="middle";g.fillText(r+1,x,y+.3*d)}});
   // translucent marker with a solid outline, so hits underneath stay visible
   star(g,qx,qy,13*d);g.globalAlpha=.3;g.fillStyle=qc;g.fill();g.globalAlpha=1;
   g.strokeStyle=bg;g.lineWidth=4*d;g.stroke();g.strokeStyle=qc;g.lineWidth=2*d;g.stroke()}
  const v=view[k],n=outside(k),moved=v.s!==1||v.x0!==0||v.y0!==0;
  const note=(moved?`zoom ×${(1/v.s).toFixed(1)}`:"")+(n?`${moved?" · ":""}${n} hit${n>1?"s":""} outside the view, follow the lines`:"");
  if(note){g.font=`${11*d}px system-ui,sans-serif`;g.textAlign="left";g.textBaseline="bottom";g.fillStyle=css("--ink2");g.fillText(note,6*d,c.height-4*d)}
  for(const i of [hover,sel])if(i>=0){const[x,y]=xy(c,P[i],k);g.strokeStyle=bg;g.lineWidth=5*d;g.beginPath();g.arc(x,y,8.5*d,0,7);g.stroke();
   g.strokeStyle=hl;g.lineWidth=2*d;g.beginPath();g.arc(x,y,8.5*d,0,7);g.stroke()}})}
function nearest(c,k,e){const r=c.getBoundingClientRect(),d=devicePixelRatio||1,mx=(e.clientX-r.left)*d,my=(e.clientY-r.top)*d;
 let best=-1,bd=(10*d)**2;P.forEach((p,i)=>{if(!on[p[1]])return;const[x,y]=xy(c,p,k),dd=(x-mx)**2+(y-my)**2,w=rank.has(i)?0.6:1;if(dd*w<bd){bd=dd*w;best=i}});return best}
// Wheel zooms at the cursor; dragging pans (a drag of more than 3 px is not a click); double-click re-fits.
let drag=null,dragged=false;
cv.forEach((c,k)=>{
 c.addEventListener("wheel",e=>{e.preventDefault();const r=c.getBoundingClientRect(),d=devicePixelRatio||1;
  const dy=e.deltaMode===1?e.deltaY*33:e.deltaY;zoomAt(k,Math.exp(Math.max(-300,Math.min(300,dy))*.0015),(e.clientX-r.left)*d,(e.clientY-r.top)*d)},{passive:false});
 c.addEventListener("mousedown",e=>{if(e.button!==0)return;drag={k,x:e.clientX,y:e.clientY,v:{...view[k]}};dragged=false});
 c.ondblclick=e=>{e.preventDefault();view[k]=hitsView(k);draw()};   // whole map when there is no search
});
addEventListener("mousemove",e=>{if(!drag)return;const c=cv[drag.k],d=devicePixelRatio||1,w=c.width-2*PAD*d;
 const dx=(e.clientX-drag.x)*d,dy=(e.clientY-drag.y)*d;if(!dragged&&Math.hypot(dx,dy)<3*d)return;
 dragged=true;c.classList.add("drag");tip.style.display="none";hover=-1;
 view[drag.k]={x0:drag.v.x0-dx/w*drag.v.s,y0:drag.v.y0+dy/w*drag.v.s,s:drag.v.s};draw()});
addEventListener("mouseup",()=>{if(drag)cv[drag.k].classList.remove("drag");drag=null});
cv.forEach((c,k)=>{c.onmousemove=e=>{if(drag&&dragged)return;const best=nearest(c,k,e);
  if(best!==hover){hover=best;draw();mark()}
  if(best<0){tip.style.display="none";return}const p=P[best],r=rank.get(best);
  tip.innerHTML=`<div class="t">${r!==undefined?"#"+(r+1)+" · ":""}${esc(p[3])}</div><div class="m">${LABEL[SRC[p[1]]]} · ${p[2]}${p[4]?" · "+p[4]:""}${r!==undefined?" · similarity "+res.hits[r].score.toFixed(3):""}</div>${p[5]?`<div class="m">${esc(p[5])}</div>`:""}<div class="m">${p[0]}</div>`;
  tip.style.display="block";const tw=tip.offsetWidth,th=tip.offsetHeight;
  tip.style.left=Math.min(e.clientX+14,innerWidth-tw-8)+"px";tip.style.top=Math.min(e.clientY+14,innerHeight-th-8)+"px"};
 c.onclick=e=>{if(dragged){dragged=false;return}const best=nearest(c,k,e);if(best<0)return;
  sel=best;mark();draw();showChunk(best);
  if(rank.has(best))$("hits").children[rank.get(best)].classList.add("open")};
 c.onmouseleave=()=>{hover=-1;tip.style.display="none";draw();mark()}});
function esc(s){return String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
// Clicked dot: fetch its chunk (the page holds only positions and titles) and show it under the maps.
async function showChunk(i){const box=$("detail"),p=P[i];box.hidden=false;
 box.innerHTML=`<div class="dt">${esc(p[3])}</div><div class="mt">Loading…</div>`;
 try{const r=await (await fetch("/api/chunk?i="+i)).json();if(r.error)throw new Error(r.error);if(sel!==i)return;
  const rk=rank.get(i),score=rk!==undefined?` · #${rk+1} for “${esc(res.query)}”, similarity ${res.hits[rk].score.toFixed(3)}`:"";
  const where=r.source==="book"?[r.book,r.chapter,r.subchapter].filter(Boolean).join(" › "):(r.path||"");
  const meta=[`<span class="sw" style="background:var(--s${p[1]})"></span> ${LABEL[SRC[p[1]]]}`,r.date,where,r.words+" words",r.lang_note,
   r.url?`<a href="${esc(r.url)}" target="_blank" rel="noopener">${p[1]===3?"watch at this moment":"source"}</a>`:"",`<code>${esc(r.chunk_id)}</code>`].filter(Boolean).join(" · ");
  box.innerHTML=`<button class="ghost x" title="close">✕</button><div class="dt">${esc(r.title)}</div><div class="mt">${meta}${score}</div>`+
   (r.context?`<p class="q"><b>Question${r.context.speaker?" ("+esc(r.context.speaker)+")":""}:</b> ${esc(r.context.text)}</p>`:"")+
   `<p class="tx">${esc(r.text)}</p>`+(r.text_cs?`<details class="cs"><summary>Czech original</summary><p>${esc(r.text_cs)}</p></details>`:"");
  box.querySelector(".x").onclick=()=>{box.hidden=true;sel=-1;draw();mark()};
  const top=box.getBoundingClientRect().top;if(top>innerHeight-120||top<0)box.scrollIntoView({behavior:"smooth",block:"nearest"});
 }catch(err){box.innerHTML=`<div class="mt">Could not load the chunk: ${esc(err.message)}</div>`}}
function mark(){[...$("hits").children].forEach((li,r)=>li.classList.toggle("on",res&&(res.hits[r].i===hover||res.hits[r].i===sel)))}
function list(){const ol=$("hits");ol.innerHTML="";$("rh").hidden=!res;if(!res)return;
 const top=res.hits[0]?.score||1;
 res.hits.forEach((h,r)=>{const p=P[h.i],li=document.createElement("li");
  const meta=[LABEL[SRC[p[1]]].replace(/s$/,""),p[2],p[4]||(p[1]===1?"book":""),p[5],h.guests.length?"guests: "+h.guests.join(", "):"",h.words+" words"].filter(Boolean).map(esc).join(" · ");
  li.innerHTML=`<div class="hh"><span class="rk">#${r+1}</span><span class="sw" style="background:var(--s${p[1]})"></span><span class="ti">${esc(p[3])}</span>
   <span class="sc">${h.score.toFixed(3)}<span class="bar" style="width:${Math.max(4,60*h.score/top)}px"></span></span></div>
   <div class="mt">${meta}${h.lang_note?" · "+esc(h.lang_note):""}${h.url?` · <a href="${esc(h.url)}" target="_blank" rel="noopener">${p[1]===3?"watch at this moment":"source"}</a>`:""} · <code>${esc(p[0])}</code></div>${h.context?`<p class="q"><b>Question${h.context.speaker?" ("+esc(h.context.speaker)+")":""}:</b> ${esc(h.context.text)}</p>`:""}<p class="tx">${esc(h.text)}</p>${h.text_cs?`<details class="cs"><summary>Czech original</summary><p>${esc(h.text_cs)}</p></details>`:""}`;
  li.onmouseenter=()=>{hover=h.i;draw();mark()};li.onmouseleave=()=>{hover=-1;draw();mark()};
  li.onclick=e=>{if(e.target.tagName==="A"||e.target.closest("details"))return;li.classList.toggle("open");sel=li.classList.contains("open")?h.i:-1;draw();mark()};
  ol.appendChild(li)})}
let last="";
async function run(query){query=query.trim();if(!query)return;last=query;
 const k=Math.max(1,Math.min(200,+$("k").value||20));$("go").disabled=true;$("status").className="";$("status").textContent="Searching…";
 try{const r=await fetch("/api/search",{method:"POST",headers:{"Content-Type":"application/json"},
   body:JSON.stringify({query,k,sources:SRC.filter((_,i)=>on[i]),announcements:ann})});
  const j=await r.json();if(!r.ok)throw new Error(j.error||r.status);
  res=j;rank=new Map(j.hits.map((h,r)=>[h.i,r]));sel=-1;fit();buttons();
  const s=j.hits.map(h=>h.score);
  $("status").textContent=`Top ${j.hits.length} of ${j.candidates.toLocaleString()} chunks for “${query}” · similarity ${s.at(-1)?.toFixed(3)}–${s[0]?.toFixed(3)} · ${j.ms} ms`;
  const u=new URL(location);u.searchParams.set("q",query);u.searchParams.set("k",k);history.replaceState(null,"",u);
 }catch(err){$("status").className="err";$("status").textContent="Search failed: "+err.message}
 $("go").disabled=false;list();draw()}
function rerun(){if(res)run(last);else draw()}
$("f").onsubmit=e=>{e.preventDefault();run($("q").value)};
$("clr").onclick=()=>{res=null;rank=new Map();sel=-1;fit();buttons();$("q").value="";$("status").textContent="";history.replaceState(null,"",location.pathname);list();draw();$("q").focus()};
new ResizeObserver(()=>{size();draw()}).observe(cv[0]);   // also catches the scrollbar appearing
matchMedia("(prefers-color-scheme: dark)").addEventListener("change",draw);
size();buttons();draw();$("status").textContent="";
const u=new URLSearchParams(location.search);if(u.get("k"))$("k").value=u.get("k");if(u.get("q")){$("q").value=u.get("q");run(u.get("q"))}
</script></body></html>"""


if __name__ == "__main__":
    main()
