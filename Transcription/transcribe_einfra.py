"""Transcribe the Urza audio with the e-INFRA CZ Whisper API (https://llm.ai.e-infra.cz/v1/).

Every audio file gets its own folder, <out>/<video_name>__API-<model>/ (e.g. abc123__API-whisper-large-v3), with
  transcript.json   all segments with times in the whole video, plus metadata
  transcript.txt    the same as readable lines, "[hh:mm:ss] text"
  windows.jsonl     the raw API answer for every window (so an interrupted run resumes without re-sending)

How the audio is sent. Whisper hears 30 seconds at a time. Given a longer upload, the server cuts it into
~30 s chunks itself, at points that ignore words: a word straddling a cut is dropped or doubled, and its
reported times drift by up to a second per chunk. So the script does the cutting: it splits the audio into
windows of --min-window to --max-window seconds (under 30), each ending at a pause (the latest moment
almost as quiet as the quietest one), and sends every window as its own 16 kHz mono WAV request, which the
server never needs to cut. Windows that are pure silence are not sent (Whisper invents text in silence).
Requests go out at most --workers at a time (the API allows 4 parallel requests per key).

Run (needs ffmpeg on PATH, numpy, requests, and EINFRA_API_KEY or E_INFRA_API_TOKEN set):
    python Transcription/transcribe_einfra.py --audio-dir /storage/brno2/home/FRIEND/audio --limit 1
"""
import argparse
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

API_URL = "https://llm.ai.e-infra.cz/v1/audio/transcriptions"
AUDIO_EXTS = {".mp3", ".m4a", ".opus", ".webm", ".ogg", ".wav", ".flac", ".mka", ".aac", ".mp4"}
SR = 16000
SERVER_CHUNK = 30.0                    # longer uploads get cut by the server; windows must stay below this
RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_TRIES = 8
FRAME = 320                            # 20 ms energy frames
SMOOTH = 10                            # 200 ms moving average: a cut lands in a pause, not between two phonemes
CUT_SLACK_DB = 3.0                     # a frame this close to the quietest one counts as an equally good cut
END_TOLERANCE = 0.5                    # a segment starting this long after its window's end is invented
# Whisper's own thresholds for a segment that is probably invented or garbled
SUSPECT_COMPRESSION = 2.4
SUSPECT_LOGPROB = -1.0
# Whisper's Czech training data had subtitle credits; in quiet audio it likes to "transcribe" them
HALLUCINATION_RE = re.compile(r"titulky\s+(vytvořil|připravil|pro\s+vás)|johnyx", re.IGNORECASE)

_print_lock = threading.Lock()


def log(msg: str):
    with _print_lock:
        print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def safe_name(s: str) -> str:
    """Folder-safe name: keeps letters (with diacritics), digits, '.', '-', '_'."""
    return re.sub(r"[^\w.-]+", "_", s).strip("_") or "unnamed"


def hms(t: float) -> str:
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def write_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------- audio

def load_pcm(path: Path) -> np.ndarray:
    """The whole file as 16 kHz mono int16 (a 3-hour video is ~350 MB)."""
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
                        "-ar", str(SR), "-f", "s16le", "-"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.int16)


def frame_db(pcm: np.ndarray) -> np.ndarray:
    """Smoothed energy (dBFS) per 20 ms frame."""
    n = len(pcm) // FRAME
    x = pcm[:n * FRAME].astype(np.float32).reshape(n, FRAME) / 32768.0
    e = np.convolve((x * x).mean(axis=1), np.ones(SMOOTH) / SMOOTH, mode="same")
    return 10 * np.log10(e + 1e-12)


def make_windows(db: np.ndarray, n_samples: int, min_win: float, max_win: float,
                 silent_db: float) -> list[dict]:
    """Cut into windows of min_win..max_win seconds, each ending at a pause."""
    fps = SR / FRAME
    wins, start = [], 0
    while start < n_samples:
        if n_samples - start <= max_win * SR:
            end = n_samples
        else:
            a = int(start / FRAME + min_win * fps)
            b = int(start / FRAME + max_win * fps)
            # the latest frame within CUT_SLACK_DB of the quietest: a real pause, and long windows
            seg = db[a:b]
            end = (a + int(np.nonzero(seg <= seg.min() + CUT_SLACK_DB)[0][-1])) * FRAME + FRAME // 2
        f0, f1 = start // FRAME, max(start // FRAME + 1, end // FRAME)
        wins.append({"k": len(wins), "start": round(start / SR, 3), "end": round(end / SR, 3),
                     "silent": bool(db[f0:f1].max(initial=-120.0) < silent_db)})
        start = end
    return wins


def wav_bytes(pcm: np.ndarray, w: dict) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(pcm[round(w["start"] * SR):round(w["end"] * SR)].tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------- API

def transcribe(audio: bytes, name: str, args, api_key: str) -> dict:
    data = {"model": args.model, "language": args.language,
            "response_format": "verbose_json", "temperature": "0"}
    for attempt in range(1, MAX_TRIES + 1):
        try:
            r = requests.post(args.api_url, headers={"Authorization": f"Bearer {api_key}"},
                              files={"file": (name, audio, "audio/wav")}, data=data, timeout=(30, 600))
            if r.status_code == 200:
                return r.json()
            if r.status_code not in RETRY_STATUS:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            why = f"HTTP {r.status_code}"
        except (requests.ConnectionError, requests.Timeout) as e:
            why = type(e).__name__
        if attempt == MAX_TRIES:
            raise RuntimeError(f"gave up after {MAX_TRIES} tries ({why})")
        wait = min(300, 10 * 2 ** (attempt - 1)) + random.uniform(0, 5)
        log(f"   {name}: {why}, retry {attempt}/{MAX_TRIES - 1} in {wait:.0f}s")
        time.sleep(wait)


def merge(results: list[tuple[dict, dict]]) -> tuple[list[dict], list[dict]]:
    """Shift every segment to whole-video time. Segments that start after their window's audio ended, or
    are a known Whisper subtitle-credit hallucination, are returned separately as dropped."""
    segs, dropped = [], []
    for w, resp in results:
        length = w["end"] - w["start"]
        for s in resp.get("segments") or []:
            text = (s.get("text") or "").strip()
            if not text:
                continue
            st, en = float(s.get("start", 0)), float(s.get("end", 0))
            if st > length + END_TOLERANCE or HALLUCINATION_RE.search(text):
                dropped.append({"window": w["k"], "start_in_window": st, "text": text,
                                "why": "after the audio" if st > length + END_TOLERANCE else "known hallucination"})
                continue
            seg = {
                "id": len(segs),
                "start": round(w["start"] + min(st, length), 2),
                "end": round(w["start"] + min(max(en, st), length), 2),
                "text": text,
                "window": w["k"],
                "avg_logprob": s.get("avg_logprob"),
                "compression_ratio": s.get("compression_ratio"),
            }
            cr, lp = seg["compression_ratio"], seg["avg_logprob"]
            seg["suspect"] = bool((cr is not None and cr > SUSPECT_COMPRESSION)
                                  or (lp is not None and lp < SUSPECT_LOGPROB))
            segs.append(seg)
    return segs, dropped


# ---------------------------------------------------------------- one file

def folder_for(src: Path, args) -> Path:
    return args.out / f"{safe_name(src.stem)}__API-{safe_name(args.model)}"


def load_cache(path: Path) -> dict:
    """Window answers saved by an earlier, interrupted run, keyed by (start, end)."""
    cache = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                c = json.loads(line)
                cache[(c["start"], c["end"])] = c["response"]
            except (json.JSONDecodeError, KeyError):
                pass  # a line cut off by a crash
    return cache


def process(src: Path, args, api_key: str, pool: ThreadPoolExecutor, titles: dict) -> str:
    folder = folder_for(src, args)
    if (folder / "transcript.json").exists():
        return "skipped"
    folder.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    pcm = load_pcm(src)
    duration = len(pcm) / SR
    wins = make_windows(frame_db(pcm), len(pcm), args.min_window, args.max_window, args.silent_db)
    spoken = [w for w in wins if not w["silent"]]
    cache_path = folder / "windows.jsonl"
    cache = load_cache(cache_path)
    cache_lock = threading.Lock()
    log(f"{src.name}: {hms(duration)}, {len(spoken)} windows to send ({len(wins) - len(spoken)} silent), "
        f"{sum((w['start'], w['end']) in cache for w in spoken)} cached")

    def one(w: dict) -> dict:
        if (w["start"], w["end"]) in cache:
            return cache[(w["start"], w["end"])]
        resp = transcribe(wav_bytes(pcm, w), f"{src.stem}_{w['k']:05d}.wav", args, api_key)
        line = json.dumps({"k": w["k"], "start": w["start"], "end": w["end"], "response": resp},
                          ensure_ascii=False)
        with cache_lock, open(cache_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return resp

    responses = [f.result() for f in [pool.submit(one, w) for w in spoken]]
    segs, dropped = merge(list(zip(spoken, responses)))
    took = time.time() - t0
    meta = titles.get(src.stem, {})
    write_json(folder / "transcript.json", {
        "video_id": src.stem,
        "title": meta.get("title"),
        "upload_date": meta.get("upload_date"),
        "source_file": str(src),
        "model": args.model,
        "api_url": args.api_url,
        "language": args.language,
        "duration": round(duration, 2),
        "n_windows": len(wins),
        "n_silent_windows": len(wins) - len(spoken),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds_taken": round(took, 1),
        "n_segments": len(segs),
        "n_suspect": sum(s["suspect"] for s in segs),
        "n_dropped": len(dropped),
        "segments": segs,
        "dropped": dropped,
    })
    (folder / "transcript.txt").write_text(
        "".join(f"[{hms(s['start'])}] {s['text']}\n" for s in segs), encoding="utf-8")
    (folder / "error.txt").unlink(missing_ok=True)
    log(f"   {src.stem}: {len(segs)} segments in {took:.0f}s ({duration / max(took, 1e-6):.0f}x real time)")
    return "done"


def load_titles(path: Path) -> dict:
    if not path or not path.exists():
        return {}
    items = json.loads(path.read_text(encoding="utf-8"))
    return {v["id"]: {"title": v.get("title"), "upload_date": v.get("upload_date")}
            for v in items if "id" in v}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("transcriptions"))
    ap.add_argument("--model", default="whisper-large-v3")
    ap.add_argument("--language", default="cs")
    ap.add_argument("--api-url", default=API_URL)
    ap.add_argument("--files", nargs="*", help="only these file names or stems (video ids)")
    ap.add_argument("--limit", type=int, help="process at most this many files (that are not done yet)")
    ap.add_argument("--shard", help="K/N: only the K-th of N disjoint parts (e.g. 1/2 and 2/2 for two API keys)")
    ap.add_argument("--workers", type=int, default=4, help="parallel API requests (API max: 4 per key)")
    ap.add_argument("--file-workers", type=int, default=2, help="files decoded and queued in parallel")
    ap.add_argument("--min-window", type=float, default=10.0, help="shortest window, seconds")
    ap.add_argument("--max-window", type=float, default=29.5, help="longest window, seconds (below 30)")
    ap.add_argument("--silent-db", type=float, default=-55.0,
                    help="a window whose loudest 200 ms stays below this (dBFS) is silence and not sent")
    ap.add_argument("--metadata", type=Path,
                    default=Path(__file__).resolve().parent.parent / "jsons" / "channel_metadata.json",
                    help="channel metadata for titles and dates (optional)")
    args = ap.parse_args()
    if not 0 < args.min_window < args.max_window < SERVER_CHUNK:
        sys.exit("Need 0 < --min-window < --max-window < 30.")

    api_key = os.environ.get("EINFRA_API_KEY") or os.environ.get("E_INFRA_API_TOKEN")
    if not api_key:
        sys.exit("Set EINFRA_API_KEY (or E_INFRA_API_TOKEN) first.")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is not on PATH.")

    files = sorted(p for p in args.audio_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    if args.files:
        wanted = set(args.files)
        files = [p for p in files if p.name in wanted or p.stem in wanted]
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        files = [p for p in files if zlib.crc32(p.stem.encode()) % n == k - 1]
    todo = [p for p in files if not (folder_for(p, args) / "transcript.json").exists()]
    n_done = len(files) - len(todo)
    if args.limit is not None:
        todo = todo[:args.limit]
    log(f"{len(files)} audio file(s), {n_done} already done, {len(todo)} to do now -> {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    titles = load_titles(args.metadata)

    counts = {"done": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(args.workers) as requests_pool, ThreadPoolExecutor(args.file_workers) as files_pool:
        def run(src: Path):
            try:
                status = process(src, args, api_key, requests_pool, titles)
            except Exception as e:  # keep going with the other files
                status = "failed"
                folder = folder_for(src, args)
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "error.txt").write_text(f"{datetime.now().isoformat()}\n{e!r}\n", encoding="utf-8")
                log(f"FAILED {src.name}: {e!r}")
            with _print_lock:
                counts[status] += 1
                done = counts["done"]
            if status == "done":
                log(f"done {src.name} ({done}/{len(todo)})")

        list(files_pool.map(run, todo))
    log(f"finished: {counts['done']} done, {counts['failed']} failed, {counts['skipped']} skipped")


if __name__ == "__main__":
    main()
