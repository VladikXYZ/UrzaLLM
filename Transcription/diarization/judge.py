"""Decide by ear which diarization system finds Urza best: a small local web app.

    python Transcription/diarization/judge.py          # then open http://127.0.0.1:8770

It finds every stretch (>= 0.5 s) of the 7 benchmark clips where the compared systems, or the reference,
disagree about whether Urza is speaking. Where they all agree, the answer cannot change which system wins,
so only the disagreements are worth listening to. For each stretch you hear it with a little context and
answer: Urza / someone else / both at once / nobody (or skip). By default you don't see what each system
said (blind), so it can't sway you.

After every answer the scoreboard is recomputed:
  wrong seconds   how much of the stretches you judged each system got wrong (a system says "Urza" there
                  and you heard someone else or nobody, or the other way round)
  Urza F1         the benchmark's "auto" Urza F1 (10 ms frames, 0.25 s collar, mean over clips), but against
                  the reference corrected by your answers; before any answer it equals the benchmark table
  sure            how often the leader stays ahead of each other system when the judged stretches are
                  resampled (bootstrap): ~50% = a coin toss, 95%+ = a real difference
Which speakers count as Urza is taken exactly as in score.py: the system speaker closest to Urza's voice,
plus any other with cosine >= 0.45 (the similarities are read from hyp/scores.json).

Answers are saved after every click to judge_verdicts.json next to this script, so you can stop and go on
later. Inputs (not in git, default ~/urza_diar): clips/<clip>.wav, hyp/<system>/<clip>.rttm with
hyp/scores.json (the MetaCentrum run), words/<clip>.json (Whisper word times, for the text). Reference
labels: reference/ next to this script.
"""
import argparse
import io
import json
import sys
import wave
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = Path.home() / "urza_diar"
FR, COLLAR, TAU = 0.01, 0.25, 0.45          # as in score.py
DEFAULT_SYSTEMS = ["sortformer-v2", "sortformer-v2.1", "pyannote-community-1", "pyannote-3.1"]
ANSWERS = {"urza": True, "both": True, "other": False, "nobody": False, "skip": None}
CONTEXT_BEFORE, CONTEXT_AFTER = 2.0, 1.0
BOOT = 2000


def read_rttm(p: Path):
    out = []
    for ln in p.read_text().splitlines():
        f = ln.split()
        if f and f[0] == "SPEAKER" and float(f[4]) > 0:
            out.append((float(f[3]), float(f[3]) + float(f[4]), f[7]))
    return out


def read_uem(p: Path, dur: float):
    if not p.exists():
        return [(0.0, dur)]
    return [(float(l.split()[2]), float(l.split()[3])) for l in p.read_text().splitlines() if l.strip()]


def mask(segs, n):
    m = np.zeros(n, bool)
    for s, e in segs:
        m[int(s / FR):int(e / FR)] = True
    return m


def collar_mask(truth, known):
    """known minus 0.25 s around every boundary of truth (score.py leaves those out)."""
    m = known.copy()
    edges = np.flatnonzero(np.diff(truth.astype(np.int8))) + 1
    c = int(COLLAR / FR)
    for b in edges:
        m[max(0, b - c):b + c] = False
    return m


def f1(truth, hyp, m):
    r, h = truth & m, hyp & m
    tp = (r & h).sum()
    p, rc = tp / max(h.sum(), 1), tp / max(r.sum(), 1)
    return 2 * p * rc / max(p + rc, 1e-9)


class Bench:
    def __init__(self, ref_dir, hyp_dir, clips_dir, words_dir, systems, min_dur):
        self.clips_dir, self.systems = clips_dir, systems
        scores = json.loads((hyp_dir / "scores.json").read_text())
        self.clips, self.items, self.data = [], [], {}
        worst = 0.0
        for rp in sorted(ref_dir.glob("*.rttm")):
            clip = rp.stem
            with wave.open(str(clips_dir / f"{clip}.wav")) as w:
                dur = w.getnframes() / w.getframerate()
            n = int(dur / FR) + 1
            ref = read_rttm(rp)
            known = mask(read_uem(ref_dir / f"{clip}.uem", dur), n)
            ref_urza = mask([(s, e) for s, e, l in ref if l == "Urza"], n)
            # score.py's mask: the collar around the reference's own Urza boundaries
            m0 = known.copy()
            c = int(COLLAR / FR)
            for s, e, l in ref:
                if l == "Urza":
                    for b in (s, e):
                        m0[max(0, int((b - COLLAR) / FR)):int((b + COLLAR) / FR)] = False
            tracks = {}
            for sname in systems:
                hyp = read_rttm(hyp_dir / sname / f"{clip}.rttm")
                row = next(r for r in scores[sname]["clips"] if r["clip"] == clip)
                sims = row["sims"]
                best = max(sims, key=sims.get)
                sure = {l for l, v in sims.items() if v > TAU + 0.005 or l == best}
                edge = [l for l, v in sims.items() if abs(v - TAU) <= 0.005 and l != best]
                # scores.json rounds the similarities to 2 decimals; for a speaker right at 0.45, take the
                # choice that reproduces the benchmark's own Urza F1
                options = [sure | set(x) for x in ([[]] + [edge] if edge else [[]])]
                cand = [mask([(s, e) for s, e, l in hyp if l in o], n) for o in options]
                errs = [abs(f1(ref_urza, h, m0) - row["urza_auto_f1"]) for h in cand]
                tracks[sname] = cand[int(np.argmin(errs))]
                worst = max(worst, min(errs))
            self.data[clip] = {"n": n, "dur": dur, "known": known, "ref": ref_urza, "tracks": tracks}
            self.clips.append(clip)
            words = []
            wp = words_dir / f"{clip}.json"
            if wp.exists():
                words = json.loads(wp.read_text(encoding="utf-8"))
            self.items += self._regions(clip, n, known, ref_urza, tracks, words, min_dur)
        self.check = worst
        self.by_id = {it["id"]: it for it in self.items}

    def _regions(self, clip, n, known, ref, tracks, words, min_dur):
        S = np.array([tracks[s] for s in self.systems])
        refstate = np.where(known, ref.astype(np.int8), 2)           # 0 / 1, 2 = not in the reference
        allsame = S.all(0) | ~S.any(0)
        contested = ~allsame | (known & (ref != S[0]))
        key = np.vstack([S.astype(np.int8), refstate[None]])
        out, i = [], 0
        while i < n:
            if not contested[i]:
                i += 1
                continue
            j = i
            while j < n and contested[j] and (key[:, j] == key[:, i]).all():
                j += 1
            s, e = i * FR, j * FR
            if e - s >= min_dur:
                wi = [(w["w"], w["s"], w["e"], s <= (w["s"] + w["e"]) / 2 <= e) for w in words
                      if w["e"] > s - CONTEXT_BEFORE - 1 and w["s"] < e + CONTEXT_AFTER + 1]
                vid, off = clip.rsplit("_", 1)[0].split("_", 1)[1], int(clip.rsplit("_", 1)[1])
                out.append({"id": f"{clip}@{s:.2f}-{e:.2f}", "clip": clip, "start": round(s, 2), "end": round(e, 2),
                            "votes": {sname: bool(S[k, i]) for k, sname in enumerate(self.systems)},
                            "ref": None if refstate[i] == 2 else bool(refstate[i]),
                            "words": wi, "url": f"https://www.youtube.com/watch?v={vid}&t={int(off + s)}s"})
            i = j
        return out

    def audio(self, clip, s, e) -> bytes:
        if clip not in self.data:
            raise ValueError("unknown clip")
        with wave.open(str(self.clips_dir / f"{clip}.wav")) as w:
            sr = w.getframerate()
            a, b = max(0, int(s * sr)), min(w.getnframes(), int(e * sr))
            w.setpos(a)
            frames, params = w.readframes(max(0, b - a)), w.getparams()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as o:
            o.setparams(params)
            o.writeframes(frames)
        return buf.getvalue()

    def scoreboard(self, verdicts: dict) -> dict:
        names = self.systems + ["reference"]
        judged = [(it, ANSWERS[verdicts[it["id"]]["label"]]) for it in self.items
                  if it["id"] in verdicts and ANSWERS.get(verdicts[it["id"]]["label"]) is not None]
        dur = np.array([it["end"] - it["start"] for it, _ in judged])
        wrong = {}
        for nm in names:
            votes = [it["votes"][nm] if nm != "reference" else it["ref"] for it, _ in judged]
            wrong[nm] = np.array([0.0 if v is None and nm == "reference" else float(v != t)
                                  for v, (_, t) in zip(votes, judged)]) * (dur if len(dur) else 1)
        # Urza F1 against the reference corrected by the answers (mean over clips, as in the benchmark)
        f1s = {nm: [] for nm in self.systems}
        for clip in self.clips:
            d = self.data[clip]
            truth, known = d["ref"].copy(), d["known"].copy()
            for it, t in judged:
                if it["clip"] == clip:
                    a, b = int(round(it["start"] / FR)), int(round(it["end"] / FR))
                    truth[a:b], known[a:b] = t, True
            m = collar_mask(truth, known)
            for nm in self.systems:
                f1s[nm].append(f1(truth, d["tracks"][nm], m))
        rows = [{"name": nm, "wrong": round(float(wrong[nm].sum()), 1),
                 "f1": round(float(np.mean(f1s[nm])) * 100, 2) if nm in f1s else None} for nm in names]
        sysrows = sorted([r for r in rows if r["name"] != "reference"], key=lambda r: (r["wrong"], -r["f1"]))
        leader, sure = sysrows[0]["name"], {}
        if len(judged) >= 2:
            rng = np.random.default_rng(0)
            idx = rng.integers(0, len(judged), (BOOT, len(judged)))
            for r in sysrows[1:]:
                diff = wrong[r["name"]][idx].sum(1) - wrong[leader][idx].sum(1)
                sure[r["name"]] = round(float((diff > 0).mean() + 0.5 * (diff == 0).mean()) * 100)
        total = sum(it["end"] - it["start"] for it in self.items)
        return {"rows": rows, "leader": leader, "sure": sure, "n_judged": len(judged),
                "n_answered": sum(1 for it in self.items if it["id"] in verdicts),
                "sec_judged": round(float(dur.sum()), 1) if len(dur) else 0.0, "sec_total": round(total, 1)}


def make_handler(bench, page, vpath):
    def load():
        return json.loads(vpath.read_text(encoding="utf-8")) if vpath.exists() else {}

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False), "application/json; charset=utf-8")

        def do_GET(self):
            u = urlparse(self.path)
            try:
                if u.path in ("/", "/index.html"):
                    return self._send(200, page, "text/html; charset=utf-8")
                if u.path == "/api/state":
                    v = load()
                    return self._json({"items": bench.items, "systems": bench.systems, "verdicts": v,
                                       "scores": bench.scoreboard(v)})
                if u.path == "/api/audio":
                    q = parse_qs(u.query)
                    return self._send(200, bench.audio(q["clip"][0], float(q["s"][0]), float(q["e"][0])), "audio/wav")
                self._send(404, "not found", "text/plain")
            except Exception as e:
                self._json({"error": repr(e)}, 400)

        def do_POST(self):
            if self.path != "/api/verdict":
                return self._send(404, "not found", "text/plain")
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if req.get("id") not in bench.by_id:
                    raise ValueError("unknown stretch")
                v = load()
                if req.get("label") is None:
                    v.pop(req["id"], None)
                elif req["label"] in ANSWERS:
                    v[req["id"]] = {"label": req["label"], "time": datetime.now().isoformat(timespec="seconds")}
                else:
                    raise ValueError("unknown answer")
                vpath.write_text(json.dumps(v, indent=1, ensure_ascii=False), encoding="utf-8")
                self._json({"verdicts": v, "scores": bench.scoreboard(v)})
            except Exception as e:
                self._json({"error": repr(e)}, 400)

        def log_message(self, *a):
            pass

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=Path, default=WORK / "clips")
    ap.add_argument("--hyp", type=Path, default=WORK / "hyp_mc", help="system outputs + scores.json (MetaCentrum run)")
    ap.add_argument("--words", type=Path, default=WORK / "words")
    ap.add_argument("--ref", type=Path, default=HERE / "reference")
    ap.add_argument("--systems", nargs="+", default=DEFAULT_SYSTEMS)
    ap.add_argument("--min-dur", type=float, default=0.5, help="shortest disagreement worth listening to (s)")
    ap.add_argument("--verdicts", type=Path, default=HERE / "judge_verdicts.json")
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    bench = Bench(args.ref, args.hyp, args.clips, args.words, args.systems, args.min_dur)
    total = sum(it["end"] - it["start"] for it in bench.items)
    print(f"{len(bench.items)} stretches to judge ({total / 60:.1f} min) in {len(bench.clips)} clips; "
          f"systems: {', '.join(args.systems)}", flush=True)
    print(f"check: Urza tracks reproduce the benchmark's Urza F1 to within {bench.check * 100:.2f} points", flush=True)
    page = TEMPLATE.replace("__BEFORE__", str(CONTEXT_BEFORE)).replace("__AFTER__", str(CONTEXT_AFTER))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(bench, page, args.verdicts))
    print(f"Open http://127.0.0.1:{args.port}  (Ctrl+C to stop); answers go to {args.verdicts}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Diarization judge</title>
<style>
:root{--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--soft:#f0efec;
 --u:#eb6834;--o:#2a78d6;--ok:#1baf7a;--bad:#c62828;--hl:rgba(235,104,52,.16)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;
 --ink2:#c3c2b7;--line:#383835;--soft:#262624;--u:#d95926;--o:#3987e5;--ok:#199e70;--bad:#ef5350;--hl:rgba(217,89,38,.22)}}
*{box-sizing:border-box} [hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:22px 16px;display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:18px}
@media (max-width:900px){.wrap{grid-template-columns:1fr}}
h1{font-size:21px;margin:0 0 2px} .sub{color:var(--ink2);margin:0 0 14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.prog{height:6px;background:var(--soft);border-radius:3px;overflow:hidden;margin:6px 0 4px}.prog i{display:block;height:100%;background:var(--ok)}
.meta{color:var(--ink2);font-size:13px;display:flex;flex-wrap:wrap;gap:4px 14px;align-items:center}
.meta b{color:var(--ink)} a{color:var(--o)}
.tl{position:relative;height:34px;margin:14px 0 6px;background:var(--soft);border-radius:6px;overflow:hidden;cursor:pointer}
.tl .reg{position:absolute;top:0;bottom:0;background:var(--hl);border-left:2px solid var(--u);border-right:2px solid var(--u)}
.tl .ph{position:absolute;top:0;bottom:0;width:2px;background:var(--ink)}
.tl .lab{position:absolute;bottom:2px;font-size:11px;color:var(--ink2)}
.tracks{display:grid;grid-template-columns:150px 1fr;gap:4px 8px;align-items:center;margin:4px 0 8px;font-size:12.5px}
.tracks .bar{position:relative;height:12px;background:var(--soft);border-radius:3px;overflow:hidden}
.tracks .bar span{position:absolute;top:0;bottom:0}
.text{font-size:15px;line-height:1.7;margin:10px 0 14px;padding:10px 12px;background:var(--soft);border-radius:8px}
.text .in{background:var(--hl);border-bottom:2px solid var(--u);font-weight:600}
.text .out{color:var(--ink2)}
.q{font-weight:600;margin:4px 0 8px}
.btns{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:9px;padding:9px 10px;cursor:pointer;text-align:left}
button:hover{border-color:var(--ink2)} button kbd{float:right;color:var(--ink2);font-size:12px;font-family:inherit}
button.sel{border-color:var(--ink);box-shadow:0 0 0 1px var(--ink);background:var(--soft)}
.ans-urza{border-left:5px solid var(--u)} .ans-other{border-left:5px solid var(--o)} .ans-both{border-left:5px solid #9c6ade}
.ans-nobody{border-left:5px solid var(--ink2)} .ans-skip{border-left:5px solid var(--line)}
.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:12px 0 0}
.row button{padding:6px 12px}
.row label{display:inline-flex;gap:6px;align-items:center;color:var(--ink2);cursor:pointer}
select{font:inherit;background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:5px 8px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:6px 6px;border-bottom:1px solid var(--line);text-align:right} th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:500;font-size:12px}
tr.lead td{font-weight:700} tr.ref td{color:var(--ink2)}
.verdict{margin:12px 0 0;padding:10px 12px;border-radius:8px;background:var(--soft);font-size:13.5px}
.hint{color:var(--ink2);font-size:12.5px;margin:10px 0 0}
#err{color:var(--bad)}
</style></head><body>
<div class="wrap">
<main>
 <h1>Which diarization finds Urza best?</h1>
 <p class="sub">Only the stretches where the systems (or the reference) disagree about whether Urza is speaking.
 Listen, say who you hear in the <span style="background:var(--hl);padding:0 4px;border-radius:3px">highlighted part</span>.</p>
 <div class="card">
  <div class="meta"><span id="pos"></span><span id="sec"></span></div>
  <div class="prog"><i id="bar"></i></div>
  <div class="meta" style="margin-top:8px"><b id="clip"></b><span id="when"></span><span id="len"></span><a id="yt" target="_blank" rel="noopener">open on YouTube</a></div>
  <div class="tl" id="tl" title="click to play from here"><div class="reg" id="reg"></div><div class="ph" id="ph"></div></div>
  <div class="tracks" id="tracks" hidden></div>
  <div class="text" id="text"></div>
  <div class="q">Who is speaking in the highlighted part?</div>
  <div class="btns" id="btns"></div>
  <div class="row">
   <button id="play">▶ Play <kbd>Space</kbd></button>
   <button id="only">Highlighted only <kbd>R</kbd></button>
   <button id="prev">← Back</button><button id="next">Next →</button>
   <label><input type="checkbox" id="reveal"> show what each system said</label>
   <label>order <select id="order"><option value="big">longest first</option><option value="time">by clip and time</option></select></label>
   <label><input type="checkbox" id="todo" checked> skip answered</label>
  </div>
  <p id="err"></p>
 </div>
 <p class="hint">Keys: <b>1</b> Urza · <b>2</b> someone else · <b>3</b> both at once · <b>4</b> nobody / noise ·
 <b>0</b> can't tell · <b>Space</b> play · <b>R</b> highlighted part only · <b>←/→</b> move. Every answer is saved at once.
 "Both" counts as Urza (he is speaking); "can't tell" is left out of the scores.</p>
</main>
<aside>
 <div class="card">
  <h2 style="font-size:16px;margin:0 0 8px">Scoreboard</h2>
  <table><thead><tr><th>system</th><th title="seconds of the judged stretches this system got wrong">wrong s</th>
  <th title="benchmark Urza F1, with the reference corrected by your answers">Urza F1</th><th title="how often the leader stays ahead of this system when the judged stretches are resampled">leader ahead</th></tr></thead>
  <tbody id="score"></tbody></table>
  <div class="verdict" id="verdict"></div>
  <p class="hint"><b>wrong s</b>: of the stretches you judged, how many seconds the system got wrong.
  <b>Urza F1</b>: the benchmark number, recomputed with your answers replacing the reference where you listened.
  <b>leader ahead</b>: ~50% is a coin toss, 95% or more a real difference.</p>
 </div>
</aside>
</div>
<script>
const $=id=>document.getElementById(id), BEFORE=__BEFORE__, AFTER=__AFTER__;
const ANS=[["urza","Urza","1"],["other","Someone else","2"],["both","Both at once","3"],["nobody","Nobody / noise","4"],["skip","Can't tell","0"]];
let S=null,order=[],cur=0,audio=null,raf=0,win=[0,1];
function fmt(t){t=Math.max(0,t);return Math.floor(t/60)+":"+String(Math.floor(t%60)).padStart(2,"0")}
function short(c){return c.split("_")[0]}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
ANS.forEach(([k,l,key])=>{const b=document.createElement("button");b.className="ans-"+k;b.dataset.k=k;
 b.innerHTML=`${l}<kbd>${key}</kbd>`;b.onclick=()=>answer(k);$("btns").appendChild(b)});
async function load(){try{const r=await fetch("/api/state");S=await r.json()}
 catch(e){$("err").textContent="Waiting for the server…";return setTimeout(load,1500)}
 if(S.error){$("err").textContent=S.error;return}$("err").textContent="";
 sortItems();const first=order.findIndex(i=>!S.verdicts[S.items[i].id]);cur=first<0?0:first;show();board()}
function sortItems(){const idx=S.items.map((_,i)=>i);
 order=$("order").value==="big"?idx.sort((a,b)=>(S.items[b].end-S.items[b].start)-(S.items[a].end-S.items[a].start)):idx}
function item(){return S.items[order[cur]]}
function show(){stop();const it=item();if(!it)return;const v=S.verdicts[it.id];
 const done=S.items.filter(x=>S.verdicts[x.id]).length;
 $("pos").innerHTML=`<b>${cur+1}</b> of ${S.items.length} · ${done} answered`;
 $("sec").textContent=`${S.scores.sec_judged}s of ${S.scores.sec_total}s judged`;
 $("bar").style.width=(100*done/S.items.length)+"%";
 $("clip").textContent=short(it.clip)+" clip";$("when").textContent=`at ${fmt(it.start)}–${fmt(it.end)}`;
 $("len").textContent=`${(it.end-it.start).toFixed(1)} s`;$("yt").href=it.url;
 win=[Math.max(0,it.start-BEFORE),it.end+AFTER];const W=win[1]-win[0];
 $("reg").style.left=(100*(it.start-win[0])/W)+"%";$("reg").style.width=(100*(it.end-it.start)/W)+"%";$("ph").style.left="0";
 $("text").innerHTML=it.words.length?it.words.map(([w,s,e,inside])=>`<span class="${inside?"in":"out"}">${esc(w)}</span>`).join(" "):"<span class=out>(no transcribed words here)</span>";
 document.querySelectorAll("#btns button").forEach(b=>b.classList.toggle("sel",!!v&&v.label===b.dataset.k));
 const tr=$("tracks");tr.hidden=!$("reveal").checked;tr.innerHTML="";
 [...S.systems,"reference"].forEach(nm=>{const val=nm==="reference"?it.ref:it.votes[nm];
  tr.insertAdjacentHTML("beforeend",`<div>${nm}</div><div class="bar"><span style="left:${100*(it.start-win[0])/W}%;width:${100*(it.end-it.start)/W}%;background:${val===null?"var(--line)":val?"var(--u)":"var(--o)"}"></span></div>`)});
 if(!tr.hidden)tr.insertAdjacentHTML("beforeend",`<div></div><div class="meta">orange = says Urza · blue = says not Urza · grey = reference has no label</div>`)}
function board(){const sc=S.scores,lead=sc.leader;
 $("score").innerHTML=sc.rows.map(r=>`<tr class="${r.name===lead?"lead":""} ${r.name==="reference"?"ref":""}"><td>${r.name}</td><td>${r.wrong.toFixed(1)}</td><td>${r.f1===null?"–":r.f1.toFixed(1)}</td><td>${r.name===lead?"leader":sc.sure[r.name]!==undefined?sc.sure[r.name]+"%":"–"}</td></tr>`).join("");
 const others=Object.entries(sc.sure);let msg;
 if(sc.n_judged<5)msg=`Answer a few stretches first (${sc.n_judged} so far). The F1 column starts at the benchmark values.`;
 else{const weakest=others.length?Math.min(...others.map(x=>x[1])):100,runner=others.find(x=>x[1]===weakest);
  msg=weakest>=95?`<b>${lead}</b> is ahead of every other system in at least 95% of resamples: a real difference.`
   :`<b>${lead}</b> leads, but it is only ahead of <b>${runner[0]}</b> in ${weakest}% of resamples: keep listening (${sc.n_judged} of ${S.items.length} answered).`}
 const ref=sc.rows.find(r=>r.name==="reference");
 if(sc.n_judged)msg+=`<br>The reference was wrong for ${ref.wrong.toFixed(1)} s of what you judged.`;
 $("verdict").innerHTML=msg}
async function answer(k){const it=item();if(!it)return;
 const r=await fetch("/api/verdict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id:it.id,label:k})});
 const j=await r.json();if(j.error){$("err").textContent=j.error;return}$("err").textContent="";
 S.verdicts=j.verdicts;S.scores=j.scores;board();advance(1)}
function advance(d){stop();let i=cur;for(let n=0;n<order.length;n++){i=(i+d+order.length)%order.length;
  if(!$("todo").checked||!S.verdicts[S.items[order[i]].id]||d<0)break}cur=i;show()}
function stop(){if(audio){audio.pause();audio=null}cancelAnimationFrame(raf)}
function play(a,b){stop();const it=item();if(!it)return;audio=new Audio(`/api/audio?clip=${encodeURIComponent(it.clip)}&s=${a}&e=${b}`);
 audio.play().catch(e=>$("err").textContent="Audio: "+e.message);const W=win[1]-win[0];
 const tick=()=>{if(!audio)return;$("ph").style.left=(100*(a+audio.currentTime-win[0])/W)+"%";if(!audio.ended)raf=requestAnimationFrame(tick)};tick()}
$("play").onclick=()=>play(win[0],win[1]);$("only").onclick=()=>{const it=item();play(it.start,it.end)};
$("tl").onclick=e=>{const r=$("tl").getBoundingClientRect();play(win[0]+(e.clientX-r.left)/r.width*(win[1]-win[0]),win[1])};
$("prev").onclick=()=>{stop();cur=(cur-1+order.length)%order.length;show()};$("next").onclick=()=>advance(1);
$("reveal").onchange=show;$("order").onchange=()=>{const id=item().id;sortItems();cur=order.findIndex(i=>S.items[i].id===id);show()};
addEventListener("keydown",e=>{if(e.target.tagName==="SELECT"||e.ctrlKey||e.metaKey||e.altKey)return;
 const a=ANS.find(x=>x[2]===e.key);if(a){e.preventDefault();answer(a[0]);return}
 if(e.key===" "){e.preventDefault();audio&&!audio.paused?stop():play(win[0],win[1])}
 else if(e.key==="r"||e.key==="R"){const it=item();play(it.start,it.end)}
 else if(e.key==="ArrowRight")advance(1);else if(e.key==="ArrowLeft")$("prev").onclick()});
load();
</script></body></html>"""


if __name__ == "__main__":
    main()
