"""See a video with its transcription, diarization, Urza decision and chunks, in sync: a local web app.

    python Transcription/visualize.py        # then open http://127.0.0.1:8772

Pick a video on the left. The YouTube player plays it; below it:
  timeline     one lane per Sortformer speaker (Urza's segments orange, others blue/grey) and a lane with the
               chunks (Urza / other speakers / housekeeping), as an overview of the whole video and a zoomed
               10-minute window around the playhead; click to jump
  transcript   every word, coloured by the speaker the chunker gave it (after flicker smoothing and snapping
               speaker changes to sentence ends), the current word highlighted while the video plays, words
               Whisper was unsure about (prob < 0.5) underlined; click a word to jump there
  chunks       a marker where each chunk starts; click it to see the chunk with its context and flags

Inputs: transcripts and diarization under --work (default ~/urza_diar), and the
two transcript chunk arrays under --chunks-dir (default real_data).
"""
import argparse
import gzip
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunk_videos import (  # noqa: E402
    OUTPUT_FILES,
    ROOT,
    chunk_video,
    load_dates_titles,
    smooth_flicker,
    snap_to_sentences,
    speakers_of_words,
)
from transcribe_einfra import safe_name  # noqa: E402


def chunk_records(file):
    """Yield record offsets from the indented export or an older compact file."""
    while True:
        offset = file.tell()
        line = file.readline()
        if not line:
            return
        record = line.strip()
        if record in (b"", b"[", b"]", b"[]"):
            continue
        if record == b"{":
            lines = [line]
            for field in iter(file.readline, b""):
                lines.append(field)
                if field.startswith(b" }"):
                    break
            record = b"".join(lines).strip()
        yield offset, json.loads(record.removesuffix(b","))


def chunk_for_viewer(chunk):
    """Give the viewer the text and context fields it displays."""
    record = {key: chunk.get(key) for key in (
        "id", "speaker", "start", "end", "part", "parts", "n_words",
        "mean_word_prob", "housekeeping", "housekeeping_signals", "context_dropped",
    )}
    if "type" in chunk:
        record["source"] = "video_urza" if chunk["type"] == "transcript" else "video_other"
        record["text"] = chunk["text_cs"]
        record["context"] = None
        if chunk["context_speaker"] is not None:
            record["context"] = {
                "speaker": chunk["context_speaker"],
                "text": chunk["context_cs"],
                "truncated": chunk["context_truncated"],
            }
    else:
        record.update(source=chunk["source"], text=chunk["text"], context=chunk["context"])
    return record


class Data:
    def __init__(self, work: Path, chunk_paths: list[Path], meta: Path):
        self.work = work
        self.meta = load_dates_titles(meta)
        uv = json.loads((work / "channel" / "urza_videos.json").read_text(encoding="utf-8"))["videos"]
        self.uv = {v["video_id"]: v for v in uv}
        self.offsets = {}
        for path in chunk_paths:
            with path.open("rb") as file:
                for offset, chunk in chunk_records(file):
                    video_id = chunk.get("video_id") or parse_qs(urlparse(chunk["src"]).query)["v"][0]
                    by_file = self.offsets.setdefault(video_id, {})
                    by_file.setdefault(path, []).append(offset)
        self.videos = []
        for vid, v in self.uv.items():
            title, date = self.meta.get(vid, (v.get("title"), None))
            self.videos.append({"id": vid, "title": title or vid, "date": date or "", "duration": v["duration"],
                                "urza_share": v["urza_share_of_speech"],
                                "chunks": sum(map(len, self.offsets.get(vid, {}).values()))})
        self.videos.sort(key=lambda x: x["date"], reverse=True)

    def video(self, vid):
        if vid not in self.uv:
            raise KeyError(vid)
        t = json.loads((self.work / "transcripts" / f"{safe_name(vid)}__faster-whisper-large-v3" / "transcript.json")
                       .read_text(encoding="utf-8"))
        d = json.loads((self.work / "channel" / f"{vid}.json").read_text(encoding="utf-8"))
        urza = set(self.uv[vid]["urza_segments"])
        words = [w for s in t["segments"] for w in s["words"] if w["word"].strip()]
        spk = snap_to_sentences(words, smooth_flicker(speakers_of_words(words, d["segments"], urza))) if words else []
        names = sorted({l for _, _, l, _ in d["segments"]})
        sp_ids = ["Urza"] + names
        idx = {s: i for i, s in enumerate(sp_ids)}
        chunks = []
        details = None
        kinds = {name: kind for kind, name in OUTPUT_FILES.items()}
        for path, offsets in self.offsets.get(vid, {}).items():
            with path.open("rb") as file:
                for offset in offsets:
                    file.seek(offset)
                    _, chunk = next(chunk_records(file))
                    if "time" in chunk and "video_id" not in chunk:
                        # The export has only the corpus fields. Derive speaker and context
                        # from the source of this video using the same chunking rules.
                        if details is None:
                            details = {(record["type"], record["start"]): record
                                       for record in chunk_video(t, d["segments"], urza, None)}
                        source = details[(kinds[path.name], chunk["time"])]
                        chunks.append(chunk_for_viewer(source | {"text_cs": chunk["text_cs"]}))
                    else:
                        chunks.append(chunk_for_viewer(chunk))
        chunks.sort(key=lambda chunk: chunk["start"])
        title, date = self.meta.get(vid, (t.get("title"), None))
        return {"id": vid, "title": title, "date": date, "duration": t["duration"], "speakers": sp_ids,
                "urza_share": self.uv[vid]["urza_share_of_speech"], "urza_seconds": self.uv[vid]["urza_seconds"],
                "words": [[round(w["start"], 2), round(w["end"], 2), w["word"],
                           None if w.get("prob") is None else round(w["prob"], 2), idx.get(s, 0)]
                          for w, s in zip(words, spk)],
                "segments": [[round(s, 2), round(e, 2), idx[l], k in urza] for k, (s, e, l, _) in enumerate(d["segments"])],
                "chunks": chunks}


def make_handler(data, page):
    class H(BaseHTTPRequestHandler):
        # HTTP/1.1 keeps the connection open after the response: with the default HTTP/1.0 the server closes
        # it right after writing, and on Windows large responses then arrived cut off (connection reset)
        protocol_version = "HTTP/1.1"

        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            b = body.encode("utf-8")
            gz = len(b) > 2048 and "gzip" in self.headers.get("Accept-Encoding", "")
            if gz:
                b = gzip.compress(b, 5)
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            if gz:
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            self.wfile.flush()

        def do_GET(self):
            u = urlparse(self.path)
            try:
                if u.path in ("/", "/index.html"):
                    return self._send(200, page, "text/html; charset=utf-8")
                if u.path == "/api/videos":
                    return self._send(200, json.dumps(data.videos, ensure_ascii=False))
                if u.path == "/api/video":
                    return self._send(200, json.dumps(data.video(parse_qs(u.query)["id"][0]), ensure_ascii=False))
                self._send(404, '"not found"')
            except Exception as e:
                self._send(400, json.dumps({"error": repr(e)}))

        def log_message(self, *a):
            pass
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, default=Path.home() / "urza_diar")
    chunk_input = ap.add_mutually_exclusive_group()
    chunk_input.add_argument("--chunks-dir", type=Path, default=ROOT / "real_data",
                             help="directory containing both transcript chunk arrays")
    chunk_input.add_argument("--chunks", type=Path, help="read an older video_chunks.jsonl file")
    ap.add_argument("--meta", type=Path, default=ROOT / "real_data" / "yt_full.json")
    ap.add_argument("--port", type=int, default=8772)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    chunk_paths = [args.chunks] if args.chunks else [args.chunks_dir / name for name in OUTPUT_FILES.values()]
    data = Data(args.work, chunk_paths, args.meta)
    print(f"{len(data.videos)} videos; open http://127.0.0.1:{args.port}  (Ctrl+C to stop)", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(data, TEMPLATE))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Urza video viewer</title>
<style>
:root{--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--soft:#f0efec;
 --u:#eb6834;--o1:#2a78d6;--o2:#1baf7a;--o3:#9c6ade;--o4:#c9a227;--hk:#9a9893;--cur:#ffe58a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;
 --ink2:#c3c2b7;--line:#383835;--soft:#262624;--u:#d95926;--o1:#3987e5;--o2:#199e70;--o3:#a57fe0;--o4:#b8932a;--hk:#7d7b76;--cur:#6b5a12}}
*{box-sizing:border-box} [hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{display:grid;grid-template-columns:300px minmax(0,1fr);gap:16px;padding:16px;max-width:1500px;margin:0 auto}
@media (max-width:900px){.wrap{grid-template-columns:1fr}#vlist{max-height:220px}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}
h1{font-size:18px;margin:0 0 8px} input[type=search]{width:100%;font:inherit;padding:7px 10px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink)}
#vlist{max-height:calc(100vh - 130px);overflow:auto;margin-top:8px}
.v{padding:6px 8px;border-radius:8px;cursor:pointer;font-size:13px}.v:hover{background:var(--soft)}.v.cur{background:var(--soft);outline:1px solid var(--ink2)}
.v .m{color:var(--ink2);font-size:12px}
.top{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:14px}
@media (max-width:1200px){.top{grid-template-columns:1fr}}
.player{position:relative;width:100%;aspect-ratio:16/9;background:#000;border-radius:10px;overflow:hidden}
.player>div,.player iframe{position:absolute;inset:0;width:100%;height:100%}
.meta{color:var(--ink2);font-size:13px;display:flex;flex-wrap:wrap;gap:4px 14px;margin:6px 0}.meta b{color:var(--ink)}
.legend{display:flex;flex-wrap:wrap;gap:6px 12px;font-size:12.5px;color:var(--ink2);margin:6px 0}
.sw{display:inline-block;width:11px;height:11px;border-radius:3px;vertical-align:-1px;margin-right:4px}
canvas{width:100%;display:block;cursor:pointer;border-radius:6px;background:var(--soft)}
.tl-label{font-size:12px;color:var(--ink2);margin:8px 0 3px}
#chunk{font-size:13.5px;max-height:330px;overflow:auto}
#chunk .ctx{background:var(--soft);border-left:3px solid var(--ink2);padding:6px 9px;border-radius:6px;margin:6px 0;color:var(--ink2)}
.tag{display:inline-block;font-size:11.5px;padding:0 7px;border-radius:9px;background:var(--soft);color:var(--ink2);margin-right:4px}
.tag.u{background:var(--u);color:#fff}.tag.hk{background:var(--hk);color:#fff}
#tx{max-height:58vh;overflow:auto;font-size:15px;line-height:1.75;margin-top:10px;padding-right:6px}
.turn{margin:0 0 10px;padding-left:10px;border-left:4px solid var(--line)}
.who{font-size:12px;font-weight:600;color:var(--ink2);margin-bottom:1px}
.w{cursor:pointer;border-radius:3px}.w:hover{background:var(--soft)}
.w.low{text-decoration:underline dotted;text-underline-offset:3px}
.w.now{background:var(--cur)}
.cm{display:inline-block;font-size:11px;font-weight:600;padding:0 6px;border-radius:8px;margin:0 3px;cursor:pointer;vertical-align:1px;color:#fff}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;font-size:13px;color:var(--ink2)}
#err{color:#c62828}
</style></head><body>
<div class="wrap">
<aside class="card">
 <h1>Urza videos</h1>
 <input type="search" id="q" placeholder="filter by title or id…">
 <div id="vlist"></div>
</aside>
<main>
 <div class="card">
  <div class="top">
   <div>
    <div class="player"><div id="yt"></div></div>
    <div class="meta" id="vmeta"></div>
   </div>
   <div>
    <div class="legend" id="legend"></div>
    <div class="tl-label">Whole video</div><canvas id="tl0" height="70"></canvas>
    <div class="tl-label" id="zlabel">10 minutes around the playhead</div><canvas id="tl1" height="110"></canvas>
    <div class="tl-label">Chunk (click a ▸ marker in the text or a chunk on the timeline)</div>
    <div id="chunk" class="card" style="padding:9px"><span class="m" style="color:var(--ink2)">–</span></div>
   </div>
  </div>
  <div class="row" style="margin-top:8px"><label><input type="checkbox" id="follow" checked> follow the video</label>
   <span>underlined = Whisper unsure (prob &lt; 0.5) · click a word to jump there</span><span id="err"></span></div>
  <div id="tx"></div>
 </div>
</main>
</div>
<script>
const $=id=>document.getElementById(id);
let V=[],D=null,player=null,ready=false,cur=-1,starts=[],els=[];
const OTH=["--o1","--o2","--o3","--o4"];
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function color(i){return i===0?css("--u"):css(OTH[(i-1)%OTH.length])}
function fmt(t){t=Math.max(0,Math.round(t));const h=Math.floor(t/3600),m=Math.floor(t%3600/60),s=t%60;return(h?h+":"+String(m).padStart(2,"0"):m)+":"+String(s).padStart(2,"0")}
function esc(s){return String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
async function init(){V=await (await fetch("/api/videos")).json();list();
 const id=new URLSearchParams(location.search).get("v")||V[0].id;open(id)}
function list(){const q=$("q").value.toLowerCase();
 $("vlist").innerHTML=V.filter(v=>!q||v.title.toLowerCase().includes(q)||v.id.toLowerCase().includes(q)).slice(0,400)
  .map(v=>`<div class="v${D&&D.id===v.id?" cur":""}" data-id="${v.id}"><div>${esc(v.title)}</div><div class="m">${v.date} · ${fmt(v.duration)} · Urza ${Math.round(v.urza_share*100)}% · ${v.chunks} chunks</div></div>`).join("");
 document.querySelectorAll(".v").forEach(e=>e.onclick=()=>open(e.dataset.id))}
$("q").oninput=list;
async function open(id){$("err").textContent="Loading…";const r=await fetch("/api/video?id="+encodeURIComponent(id));D=await r.json();
 if(D.error){$("err").textContent=D.error;return}$("err").textContent="";history.replaceState(null,"","?v="+id);list();
 starts=D.words.map(w=>w[0]);cur=-1;
 $("vmeta").innerHTML=`<b>${esc(D.title)}</b><span>${D.date||""}</span><span>${fmt(D.duration)}</span><span>Urza ${Math.round(D.urza_share*100)}% of speech (${(D.urza_seconds/60).toFixed(0)} min)</span><span>${D.chunks.filter(c=>c.source==="video_urza").length} Urza chunks, ${D.chunks.filter(c=>c.source!=="video_urza").length} other</span><a href="https://www.youtube.com/watch?v=${id}" target="_blank" rel="noopener">YouTube</a>`;
 $("legend").innerHTML=D.speakers.map((s,i)=>`<span><span class="sw" style="background:${color(i)}"></span>${esc(s)}</span>`).join("")+
  `<span><span class="sw" style="background:${css("--hk")}"></span>housekeeping chunk</span>`;
 text();draw();$("chunk").innerHTML="–";
 if(ready)player.cueVideoById(id);else window._pending=id}
function text(){const byStart=new Map(D.chunks.map(c=>[c.start,c]));let html="",last=-1,open=false,i=0;
 const cstarts=D.chunks.map(c=>c.start).sort((a,b)=>a-b);let ci=0;
 for(const [s,e,w,p,k] of D.words){
  while(ci<cstarts.length&&cstarts[ci]<=s+0.01){const c=byStart.get(cstarts[ci]);
   html+=`${k!==last&&open?"":""}<span class="cm" data-c="${esc(c.id)}" style="background:${c.housekeeping?css("--hk"):c.source==="video_urza"?css("--u"):css("--o1")}">▸ ${esc(c.id.split("-").pop())}${c.parts>1?" "+c.part+"/"+c.parts:""}</span>`;ci++}
  if(k!==last){if(open)html+="</div>";html+=`<div class="turn" style="border-left-color:${color(k)}"><div class="who" style="color:${color(k)}">${esc(D.speakers[k])} · ${fmt(s)}</div>`;open=true;last=k}
  html+=`<span class="w${p!==null&&p<0.5?" low":""}" data-i="${i}" title="${fmt(s)}${p!==null?" · prob "+p:""}">${esc(w)}</span> `;i++}
 if(open)html+="</div>";$("tx").innerHTML=html;els=[...document.querySelectorAll("#tx .w")];
 $("tx").onclick=e=>{const w=e.target.closest(".w"),m=e.target.closest(".cm");
  if(w)seek(D.words[+w.dataset.i][0]);if(m)showChunk(D.chunks.find(c=>c.id===m.dataset.c),true)}}
function showChunk(c,jump){if(!c)return;
 const tags=[c.source==="video_urza"?'<span class="tag u">Urza</span>':`<span class="tag">${esc(c.speaker)}</span>`,
  c.parts>1?`<span class="tag">part ${c.part}/${c.parts}</span>`:"",`<span class="tag">${c.n_words} words</span>`,
  c.mean_word_prob!=null?`<span class="tag">confidence ${c.mean_word_prob}</span>`:"",
  c.housekeeping?`<span class="tag hk">housekeeping: ${esc((c.housekeeping_signals||[]).join(", "))}</span>`:"",
  c.context_dropped?'<span class="tag">question removed (housekeeping)</span>':""].join("");
 $("chunk").innerHTML=`<div><b>${esc(c.id)}</b> · ${fmt(c.start)}–${fmt(c.end)}</div><div style="margin:4px 0">${tags}</div>`+
  (c.context?`<div class="ctx"><b>Context (${esc(c.context.speaker)}${c.context.truncated?", end of a longer turn":""}):</b> ${esc(c.context.text)}</div>`:"")+
  `<div>${esc(c.text)}</div>`;if(jump)seek(c.start)}
function seek(t){if(player&&ready){player.seekTo(t,true);player.playVideo()}}
function draw(){if(!D)return;const T=D.duration,now=ready&&player.getCurrentTime?player.getCurrentTime():0;
 [[0,0,T],[1,Math.max(0,now-300),Math.max(0,now-300)+600]].forEach(([k,a,b])=>{const c=$("tl"+k),d=devicePixelRatio||1;
  c.width=c.clientWidth*d;c.height=c.clientHeight*d;const g=c.getContext("2d"),W=c.width,H=c.height,n=D.speakers.length;
  const lanes=n+1,lh=(H-6*d)/lanes,X=t=>(t-a)/(b-a)*W;g.clearRect(0,0,W,H);
  D.segments.forEach(([s,e,i,u])=>{if(e<a||s>b)return;const lane=u?0:i;g.fillStyle=u?color(0):color(i);
   g.globalAlpha=u?0.95:0.75;g.fillRect(X(s),lane*lh+1,Math.max(1,X(e)-X(s)),lh-2)});
  g.globalAlpha=1;D.chunks.forEach(c=>{if(c.end<a||c.start>b)return;g.fillStyle=c.housekeeping?css("--hk"):c.source==="video_urza"?color(0):css("--o1");
   g.globalAlpha=0.55;g.fillRect(X(c.start),n*lh+3*d,Math.max(1,X(c.end)-X(c.start)-1*d),lh-2)});
  g.globalAlpha=1;g.fillStyle=css("--ink");g.fillRect(X(now)-d,0,2*d,H);
  if(k===1){g.fillStyle=css("--ink2");g.font=`${10*d}px system-ui`;g.fillText(fmt(a),3*d,H-3*d);const t2=fmt(b);g.fillText(t2,W-g.measureText(t2).width-3*d,H-3*d)}
  c.onclick=e=>{const r=c.getBoundingClientRect(),t=a+(e.clientX-r.left)/r.width*(b-a);
   const ch=D.chunks.find(x=>x.start<=t&&t<=x.end);if(ch&&(e.clientY-r.top)/r.height>n/(n+1))showChunk(ch,false);seek(t)}})}
function tick(){if(D&&ready&&player.getCurrentTime){const t=player.getCurrentTime();
  let lo=0,hi=starts.length-1,i=-1;while(lo<=hi){const m=(lo+hi)>>1;if(starts[m]<=t){i=m;lo=m+1}else hi=m-1}
  if(i!==cur){if(cur>=0&&els[cur])els[cur].classList.remove("now");if(i>=0&&els[i]){els[i].classList.add("now");
   if($("follow").checked&&player.getPlayerState()===1){const tx=$("tx"),el=els[i],r=el.offsetTop-tx.offsetTop;
    if(r<tx.scrollTop+40||r>tx.scrollTop+tx.clientHeight-80)tx.scrollTop=r-tx.clientHeight/3}}cur=i}
  draw()}setTimeout(tick,250)}
window.onYouTubeIframeAPIReady=()=>{player=new YT.Player("yt",{videoId:window._pending||"",playerVars:{rel:0},
 events:{onReady:()=>{ready=true;if(window._pending&&D)player.cueVideoById(D.id);tick()}}})};
addEventListener("resize",draw);
init();
</script>
<script src="https://www.youtube.com/iframe_api"></script>
</body></html>"""


if __name__ == "__main__":
    main()
