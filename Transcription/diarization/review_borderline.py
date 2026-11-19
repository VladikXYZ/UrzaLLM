"""Listen to the speakers near the Urza threshold and say who they are: a small local web app.

    python Transcription/diarization/review_borderline.py      # then open http://127.0.0.1:8771

Lists every speaker whose normalised score z (rescore_urza.py) is between --lo and --hi (4 and 8): the
ones the voice comparison can't settle. For each it plays a few of that speaker's segments, spread over the
video, through the YouTube player (no audio download), with the transcript under each, and shows the other
speakers of the same video with their scores and a play button, to compare with the Urza of that video.
You answer: Urza / not Urza / mixed (Sortformer put Urza and someone else into this one speaker) / can't
tell. Answers are saved after every click to urza_review.json next to this script; rescore_urza.py then
takes them as fixed decisions (for "mixed", only the speaker's segments that score as Urza on their own).

Inputs (default ~/urza_diar): channel/urza_videos.json + channel/<video>.json from rescore_urza.py and
diarize_channel.py, transcripts_border/<video>__faster-whisper-large-v3/transcript.json for the text.
"""
import argparse
import json
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path.home() / "urza_diar"
ANSWERS = ("urza", "other", "mixed", "skip")
SAMPLES, MIN_SAMPLE = 4, 4.0


def words_in(tr, segs, label, s, e, pad=2.0):
    """The words around a clip as [word, mine]: mine = Sortformer put this word into `label`."""
    if not tr:
        return []
    out = []
    for seg in tr["segments"]:
        if seg["end"] < s - pad or seg["start"] > e + pad:
            continue
        for w in seg["words"]:
            mid = (w["start"] + w["end"]) / 2
            if s - pad <= mid <= e + pad:
                who = [l for a, b, l, _ in segs if a <= mid <= b]
                out.append([w["word"], label in who])
    return out


def samples(segs, label, n=SAMPLES):
    """the speaker's longest segment + others of >= MIN_SAMPLE s spread over the video, in time order."""
    mine = [(s, e) for s, e, l, _ in segs if l == label]
    if not mine:
        return []
    longest = max(mine, key=lambda x: x[1] - x[0])
    rest = [x for x in mine if x[1] - x[0] >= MIN_SAMPLE and x != longest]
    pick = [rest[round(i * (len(rest) - 1) / max(n - 2, 1))] for i in range(min(n - 1, len(rest)))] if rest else []
    return sorted({longest, *pick})


def build(work: Path, lo: float, hi: float):
    vids = json.loads((work / "channel" / "urza_videos.json").read_text(encoding="utf-8"))
    tau = vids.get("z_threshold", 6.0)
    items = []
    for v in vids["videos"]:
        border = [l for l, s in v["speakers"].items() if s["z"] is not None and lo <= s["z"] < hi]
        if not border:
            continue
        d = json.loads((work / "channel" / f"{v['video_id']}.json").read_text(encoding="utf-8"))
        tp = work / "transcripts_border" / f"{v['video_id']}__faster-whisper-large-v3" / "transcript.json"
        tr = json.loads(tp.read_text(encoding="utf-8")) if tp.exists() else None
        segs = d["segments"]

        def clips(label, n=SAMPLES):
            return [{"s": round(s, 1), "e": round(e, 1), "words": words_in(tr, segs, label, s, e)[:200]}
                    for s, e in samples(segs, label, n)]

        others = [{"label": l, "seconds": s["seconds"], "z": s["z"], "urza": l in v["urza_speakers"],
                   "clip": (clips(l, 1) or [None])[0]} for l, s in sorted(v["speakers"].items()) if l not in border]
        for l in border:
            s = v["speakers"][l]
            items.append({"id": f"{v['video_id']}/{l}", "video": v["video_id"], "title": v["title"] or v["video_id"],
                          "label": l, "seconds": s["seconds"], "z": s["z"], "sim": s["sim"],
                          "counted": l in v["urza_speakers"], "clips": clips(l),
                          "others": others + [{"label": o, "seconds": v["speakers"][o]["seconds"], "z": v["speakers"][o]["z"],
                                               "urza": o in v["urza_speakers"], "clip": (clips(o, 1) or [None])[0]}
                                              for o in border if o != l]})
    items.sort(key=lambda it: it["z"])
    return items, tau


def make_handler(items, tau, page, rpath):
    ids = {it["id"] for it in items}

    def load():
        return json.loads(rpath.read_text(encoding="utf-8")) if rpath.exists() else {}

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.split("?")[0] in ("/", "/index.html"):
                return self._send(200, page, "text/html; charset=utf-8")
            if self.path == "/api/state":
                return self._send(200, json.dumps({"items": items, "tau": tau, "answers": load()}, ensure_ascii=False))
            self._send(404, '"not found"')

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path != "/api/answer" or req.get("id") not in ids:
                    raise ValueError("unknown speaker")
                a = load()
                if req.get("label") is None:
                    a.pop(req["id"], None)
                elif req["label"] in ANSWERS:
                    it = next(x for x in items if x["id"] == req["id"])
                    a[req["id"]] = {"label": req["label"], "note": str(req.get("note") or "")[:500], "z": it["z"],
                                    "title": it["title"], "time": datetime.now().isoformat(timespec="seconds")}
                else:
                    raise ValueError("unknown answer")
                rpath.write_text(json.dumps(a, indent=1, ensure_ascii=False), encoding="utf-8")
                self._send(200, json.dumps({"answers": a}, ensure_ascii=False))
            except Exception as e:
                self._send(400, json.dumps({"error": repr(e)}))

        def log_message(self, *a):
            pass

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, default=WORK)
    ap.add_argument("--lo", type=float, default=4.0)
    ap.add_argument("--hi", type=float, default=8.0)
    ap.add_argument("--review", type=Path, default=HERE / "urza_review.json")
    ap.add_argument("--port", type=int, default=8771)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    items, tau = build(args.work, args.lo, args.hi)
    print(f"{len(items)} speakers with z in [{args.lo}, {args.hi}) in {len({i['video'] for i in items})} videos", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(items, tau, TEMPLATE, args.review))
    print(f"Open http://127.0.0.1:{args.port}  (Ctrl+C to stop); answers go to {args.review}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Urza or not</title>
<style>
:root{--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--soft:#f0efec;
 --u:#eb6834;--o:#2a78d6;--m:#9c6ade;--ok:#1baf7a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;
 --ink2:#c3c2b7;--line:#383835;--soft:#262624;--u:#d95926;--o:#3987e5;--m:#a57fe0;--ok:#199e70}}
*{box-sizing:border-box} [hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:20px 16px;display:grid;grid-template-columns:300px minmax(0,1fr);gap:18px}
@media (max-width:900px){.wrap{grid-template-columns:1fr}#list{max-height:240px}}
h1{font-size:20px;margin:0 0 2px} .sub{color:var(--ink2);margin:0 0 12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
#list{max-height:calc(100vh - 140px);overflow:auto;padding:4px}
.li{display:flex;gap:8px;align-items:center;padding:6px 8px;border-radius:8px;cursor:pointer;font-size:13px}
.li:hover{background:var(--soft)} .li.cur{background:var(--soft);outline:1px solid var(--ink2)}
.li .zz{font-variant-numeric:tabular-nums;color:var(--ink2);min-width:2.6em}
.li .t{flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.dot{width:10px;height:10px;border-radius:50%;flex:none;background:var(--line)}
.d-urza{background:var(--u)}.d-other{background:var(--o)}.d-mixed{background:var(--m)}.d-skip{background:var(--ink2)}
.prog{height:6px;background:var(--soft);border-radius:3px;overflow:hidden;margin:6px 0 10px}.prog i{display:block;height:100%;background:var(--ok)}
.meta{color:var(--ink2);font-size:13px;display:flex;flex-wrap:wrap;gap:4px 14px}.meta b{color:var(--ink)}
.player{position:relative;width:100%;aspect-ratio:16/9;background:#000;border-radius:10px;overflow:hidden;margin:10px 0}
.player iframe{position:absolute;inset:0;width:100%;height:100%;border:0}
.clips{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;padding:7px 11px;cursor:pointer}
button:hover{border-color:var(--ink2)} button.on{border-color:var(--ink);box-shadow:0 0 0 1px var(--ink)}
.text{background:var(--soft);border-radius:8px;padding:9px 11px;margin:6px 0 12px;min-height:2.5em;font-size:14.5px;line-height:1.7}
.text .me{font-weight:700;color:var(--ink)} .text .nm{color:var(--ink2);opacity:.75}
.how{background:var(--soft);border-radius:8px;padding:9px 12px;margin:0 0 10px;font-size:13px}
.how ul{margin:4px 0 0;padding-left:18px}
.ans{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin:8px 0}
.ans button{text-align:left;padding:10px 12px}.ans kbd{float:right;color:var(--ink2);font-family:inherit;font-size:12px}
.a-urza{border-left:5px solid var(--u)}.a-other{border-left:5px solid var(--o)}.a-mixed{border-left:5px solid var(--m)}.a-skip{border-left:5px solid var(--ink2)}
.ans button.sel{background:var(--soft);border-color:var(--ink);box-shadow:0 0 0 1px var(--ink)}
input[type=text]{width:100%;font:inherit;background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:7px 10px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums;margin-top:6px}
th,td{padding:5px 6px;border-bottom:1px solid var(--line);text-align:left} th{color:var(--ink2);font-weight:500;font-size:12px}
.tag{font-size:11.5px;padding:1px 7px;border-radius:10px;background:var(--soft);color:var(--ink2)}.tag.u{background:var(--u);color:#fff}
.hint{color:var(--ink2);font-size:12.5px;margin:8px 0 0} a{color:var(--o)} #err{color:#c62828}
</style></head><body>
<div class="wrap">
<aside>
 <h1>Urza or not?</h1>
 <p class="sub" id="count"></p>
 <div class="prog"><i id="bar"></i></div>
 <div class="card" id="list"></div>
</aside>
<main>
 <div class="card">
  <div class="meta"><b id="title"></b></div>
  <div class="meta" style="margin-top:4px"><span id="spk"></span><span id="score"></span><span id="now"></span><a id="yt" target="_blank" rel="noopener">open on YouTube</a></div>
  <div class="player"><iframe id="yt-frame" allow="autoplay; encrypted-media" allowfullscreen></iframe></div>
  <div class="how"><b>What you are judging:</b> the program split the video into anonymous "speakers" and isn't
  sure whether this one is Urza. A speaker is just the pieces of audio the program grouped as one voice, and it
  can be wrong at the edges (a few words of the next person) or mix two people.
  <ul><li>Listen to the clips. <b>Bold</b> words = what the program gave to this speaker, grey = someone else
  (2 s around the clip, for context).</li>
  <li>Answer for the <b>bold parts, in most clips</b>: Urza → <b>1</b>, someone else → <b>2</b>, clearly
  both (e.g. one clip Urza, another not) → <b>3 Mixed</b>. A word or two of the wrong person at an edge
  doesn't matter.</li></ul></div>
  <div class="meta">This speaker's clips (spread over the video):</div>
  <div class="clips" id="clips"></div>
  <div class="text" id="text"></div>
  <div style="font-weight:600">In most clips, is the bold speaker Urza?</div>
  <div class="ans" id="ans"></div>
  <input type="text" id="note" placeholder="optional note (e.g. 'Urza only in the first clip')">
  <p id="err"></p>
  <div class="meta" style="margin-top:10px">Other speakers in this video, to compare voices. The "Urza" tag is
  the program's current guess and can be wrong too:</div>
  <table><thead><tr><th>speaker</th><th title="all its pieces added up, over the whole video">total minutes of speech</th>
  <th title="how much it sounds like Urza; Urza from 6 up">Urza score (Urza ≥ 6)</th><th>program's guess</th><th>its longest piece</th></tr></thead><tbody id="others"></tbody></table>
  <p class="hint">Keys: <b>1</b> Urza · <b>2</b> not Urza · <b>3</b> mixed (partly Urza) · <b>0</b> can't tell ·
  <b>←/→</b> previous/next clip · <b>↑/↓</b> previous/next speaker. Answers are saved at once; the list is sorted
  by score z (Urza if z ≥ <span id="tau"></span>). "Mixed" = Sortformer merged Urza and someone else into this speaker.</p>
 </div>
</main>
</div>
<script>
const $=id=>document.getElementById(id);
const ANS=[["urza","Urza","1"],["other","Not Urza","2"],["mixed","Mixed (partly Urza)","3"],["skip","Can't tell","0"]];
let S=null,cur=0,clip=0;
function fmt(t){t=Math.max(0,Math.round(t));const h=Math.floor(t/3600),m=Math.floor(t%3600/60),s=t%60;return(h?h+":"+String(m).padStart(2,"0"):m)+":"+String(s).padStart(2,"0")}
function esc(s){return String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
ANS.forEach(([k,l,key])=>{const b=document.createElement("button");b.className="a-"+k;b.dataset.k=k;b.innerHTML=`${l}<kbd>${key}</kbd>`;b.onclick=()=>answer(k);$("ans").appendChild(b)});
async function load(){try{const r=await fetch("/api/state");S=await r.json()}catch(e){$("err").textContent="Waiting for the server…";return setTimeout(load,1500)}
 $("tau").textContent=S.tau;const f=S.items.findIndex(it=>!S.answers[it.id]);cur=f<0?0:f;list();show()}
function list(){const n=Object.keys(S.answers).length;$("count").textContent=`${n} of ${S.items.length} answered`;
 $("bar").style.width=(100*n/S.items.length)+"%";
 $("list").innerHTML=S.items.map((it,i)=>`<div class="li${i===cur?" cur":""}" data-i="${i}"><span class="dot d-${S.answers[it.id]?.label||""}"></span><span class="zz">${it.z.toFixed(1)}</span><span class="t" title="${esc(it.title)}">${esc(it.title)}</span></div>`).join("");
 document.querySelectorAll(".li").forEach(e=>e.onclick=()=>{cur=+e.dataset.i;clip=0;list();show()});
 document.querySelector(".li.cur")?.scrollIntoView({block:"nearest"})}
function words(ws){return ws&&ws.length?ws.map(([w,me])=>`<span class="${me?"me":"nm"}">${esc(w)}</span>`).join(" "):"(no transcribed words)"}
function play(video,c){if(!c)return;const s=Math.max(0,Math.floor(c.s)),e=Math.ceil(c.e);
 $("yt-frame").src=`https://www.youtube-nocookie.com/embed/${video}?start=${s}&end=${e}&autoplay=1&rel=0`;
 $("yt").href=`https://www.youtube.com/watch?v=${video}&t=${s}s`;$("now").textContent=`playing ${fmt(c.s)}–${fmt(c.e)}`}
function show(){const it=S.items[cur];if(!it)return;const a=S.answers[it.id];
 $("title").textContent=it.title;$("spk").textContent=`${it.label} · ${(it.seconds/60).toFixed(1)} min of speech in total (all its pieces)`;
 $("score").innerHTML=`z ${it.z.toFixed(2)} · similarity ${it.sim.toFixed(3)} · now ${it.counted?"<b>counted as Urza</b>":"not counted as Urza"}`;
 $("clips").innerHTML=it.clips.map((c,i)=>`<button class="${i===clip?"on":""}" data-i="${i}">${fmt(c.s)} (${Math.round(c.e-c.s)} s)</button>`).join("");
 document.querySelectorAll("#clips button").forEach(b=>b.onclick=()=>{clip=+b.dataset.i;show()});
 const c=it.clips[clip];$("text").innerHTML=words(c?.words);play(it.video,c);
 document.querySelectorAll("#ans button").forEach(b=>b.classList.toggle("sel",!!a&&a.label===b.dataset.k));
 $("note").value=a?.note||"";
 $("others").innerHTML=it.others.map((o,i)=>`<tr><td>${esc(o.label)}</td><td>${(o.seconds/60).toFixed(1)}</td><td>${o.z==null?"–":o.z.toFixed(1)}</td><td>${o.urza?'<span class="tag u">Urza</span>':o.z!=null&&o.z>=4&&o.z<8?'<span class="tag">borderline</span>':""}</td><td>${o.clip?`<button data-o="${i}">▶ ${fmt(o.clip.s)}</button>`:""}</td></tr>`).join("");
 document.querySelectorAll("#others button").forEach(b=>b.onclick=()=>{const o=it.others[+b.dataset.o];play(it.video,o.clip);$("text").innerHTML=`<i>[${esc(o.label)}, bold = its words]</i> `+words(o.clip.words)})}
async function answer(k){const it=S.items[cur];
 const r=await fetch("/api/answer",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id:it.id,label:k,note:$("note").value})});
 const j=await r.json();if(j.error){$("err").textContent=j.error;return}$("err").textContent="";S.answers=j.answers;
 const nxt=S.items.findIndex((x,i)=>i>cur&&!S.answers[x.id]);if(nxt>=0){cur=nxt;clip=0}list();show()}
addEventListener("keydown",e=>{if(e.target.tagName==="INPUT"||e.ctrlKey||e.metaKey||e.altKey)return;const it=S?.items[cur];if(!it)return;
 const a=ANS.find(x=>x[2]===e.key);if(a){e.preventDefault();return answer(a[0])}
 if(e.key==="ArrowRight"&&clip<it.clips.length-1){clip++;show()}else if(e.key==="ArrowLeft"&&clip>0){clip--;show()}
 else if(e.key==="ArrowDown"&&cur<S.items.length-1){e.preventDefault();cur++;clip=0;list();show()}
 else if(e.key==="ArrowUp"&&cur>0){e.preventDefault();cur--;clip=0;list();show()}});
load();
</script></body></html>"""


if __name__ == "__main__":
    main()
