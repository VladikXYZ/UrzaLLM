"""Sortformer + Canary + Whisper large-v3 + Whisper large-v3-turbo.

  python ASRBench/transcribe_pipeline.py audio/            # every video in the folder
  python ASRBench/transcribe_pipeline.py audio/x.webm      # or single files

Per video, written to out/<name>/:
  diarization.json            Sortformer speaker turns
  canary.json                 Canary, audio cut at speaker turns (long turns split at pauses)
  whisper-large-v3.json       Whisper on the whole file, each word gets the Sortformer speaker
  whisper-large-v3-turbo.json same with turbo
Each transcript is a list of segments {start, end, speaker, text, words}; Whisper words
also carry "prob" (confidence).

Optimization: three processes share the GPU (Sortformer -> Canary, Whisper v3,
Whisper turbo), each model loads once, every model runs batched.

WAVs are made one video at a time, just before they are needed, and deleted when all
three processes are done with them; together they never exceed WAV_LIMIT_GB (16).

Finished videos are skipped. While a model works on a video, out/<name>/<model>.unfinished
exists; if the run is stopped, that flag stays and the video is redone next time.
"""
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

SR = 16000
OUT = Path("out")
BATCH = 16
WAV_LIMIT = float(os.environ.get("WAV_LIMIT_GB", 16)) * 1e9
SCRATCH = Path(os.environ.get("SCRATCHDIR") or tempfile.gettempdir()) / "asr_wavs"
MEDIA = {".webm", ".mp4", ".mkv", ".mov", ".m4a", ".mp3", ".opus", ".ogg", ".wav", ".flac"}
WHISPER = ["large-v3", "large-v3-turbo"]


# ---------------------------------------------------------------- files

def save(name, filename, data):
    """Write atomically, so a stopped run never leaves a half-written result."""
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / (filename + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, d / filename)


def load(name, filename):
    return json.loads((OUT / name / filename).read_text(encoding="utf-8"))


def has(name, filename):
    return (OUT / name / filename).exists()


def nemo_done(name):
    return has(name, "diarization.json") and has(name, "canary.json")


def whisper_done(name, model):
    return has(name, f"whisper-{model}.json") or has(name, f"whisper-{model}.raw.json")


def needs_wav(name):
    return not nemo_done(name) or not all(whisper_done(name, m) for m in WHISPER)


def wav_seconds(video):
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(video)], capture_output=True, text=True, check=True)
        return float(r.stdout.strip())
    except Exception:
        return 10 * 3600  # unknown: assume long


# ---------------------------------------------------------------- per-video loop shared by workers

def each_video(worker, jobs, is_done, process):
    """Run process(name, wav) for every unfinished video, in order."""
    for name, _ in jobs:
        marker = SCRATCH / f"{name}.done.{worker}"
        if is_done(name):
            marker.touch()
            continue
        while not (SCRATCH / f"{name}.ready").exists():  # the main process makes the WAV
            if (SCRATCH / f"{name}.failed").exists():
                break
            time.sleep(2)
        if (SCRATCH / f"{name}.ready").exists():
            flag = OUT / name / f"{worker}.unfinished"
            flag.parent.mkdir(parents=True, exist_ok=True)
            if flag.exists():
                print(f"{name}: {worker} was interrupted last time, redoing", flush=True)
            flag.touch()
            process(name, str(SCRATCH / f"{name}.wav"))
            flag.unlink()
            print(f"{name}: {worker} done", flush=True)
        marker.touch()


# ---------------------------------------------------------------- speaker helpers

def turns(diar, audio, max_len=30.0, min_len=15.0):
    """Speaker turns without overlap, same-speaker gaps < 0.5 s merged,
    turns longer than max_len split at the quietest point between min_len and max_len."""
    out = []
    for d in sorted(diar, key=lambda d: d["start"]):
        s, e = d["start"], d["end"]
        if out:
            s = max(s, out[-1]["end"])
        if e <= s:
            continue
        if out and out[-1]["speaker"] == d["speaker"] and s - out[-1]["end"] < 0.5:
            out[-1]["end"] = e
        else:
            out.append({"start": s, "end": e, "speaker": d["speaker"]})
    pieces = []
    for t in out:
        s = t["start"]
        while t["end"] - s > max_len:
            win = audio[int((s + min_len) * SR):int((s + max_len) * SR)]
            hop = SR // 100
            rms = np.sqrt((win[:len(win) // hop * hop].reshape(-1, hop) ** 2).mean(axis=1))
            rms = np.convolve(rms, np.ones(20) / 20, mode="same")
            cut = s + min_len + int(np.argmin(rms)) * hop / SR
            pieces.append({"start": s, "end": cut, "speaker": t["speaker"]})
            s = cut
        pieces.append({"start": s, "end": t["end"], "speaker": t["speaker"]})
    return pieces


def speaker_at(diar, s, e):
    best, best_ov = None, 0.0
    for d in diar:
        ov = min(e, d["end"]) - max(s, d["start"])
        if ov > best_ov:
            best, best_ov = d["speaker"], ov
    if best is None and diar:
        mid = (s + e) / 2
        best = min(diar, key=lambda d: min(abs(mid - d["start"]), abs(mid - d["end"])))["speaker"]
    return best


def words_to_segments(words, diar):
    segs = []
    for w in words:
        spk = speaker_at(diar, w["start"], w["end"])
        if segs and segs[-1]["speaker"] == spk and w["start"] - segs[-1]["end"] <= 1.0:
            segs[-1]["end"] = w["end"]
            segs[-1]["words"].append(w)
        else:
            segs.append({"start": w["start"], "end": w["end"], "speaker": spk, "words": [w]})
    for s in segs:
        s["text"] = " ".join(w["word"] for w in s["words"])
    return segs


def merge_whisper(name):
    """Whisper words + Sortformer speakers -> final Whisper files."""
    if not has(name, "diarization.json"):
        return
    diar = load(name, "diarization.json")
    for m in WHISPER:
        if has(name, f"whisper-{m}.raw.json") and not has(name, f"whisper-{m}.json"):
            save(name, f"whisper-{m}.json", words_to_segments(load(name, f"whisper-{m}.raw.json"), diar))
            (OUT / name / f"whisper-{m}.raw.json").unlink()


# ---------------------------------------------------------------- workers (one process each)

def run_nemo(jobs):
    """Sortformer, then Canary on the speaker turns."""
    import soundfile as sf
    import torch
    from nemo.collections.asr.models import ASRModel, SortformerEncLabelModel

    diar_model = SortformerEncLabelModel.from_pretrained(
        "nvidia/diar_streaming_sortformer_4spk-v2", map_location="cuda").eval()
    m = diar_model.sortformer_modules  # long-form streaming setting from the model card
    m.chunk_len, m.chunk_right_context, m.fifo_len = 340, 40, 40
    m.spkcache_update_period, m.spkcache_len = 340, 188
    canary = ASRModel.from_pretrained("nvidia/canary-1b-v2", map_location="cuda").eval()

    def process(name, wav):
        if has(name, "diarization.json"):
            diar = load(name, "diarization.json")
        else:
            with torch.inference_mode():
                raw = diar_model.diarize(audio=[wav], batch_size=1)
            raw = raw[0] if isinstance(raw, tuple) else raw
            diar = []
            for item in raw[0]:
                s, e, spk = item.split()[:3] if isinstance(item, str) else item[:3]
                diar.append({"start": float(s), "end": float(e), "speaker": str(spk)})
            save(name, "diarization.json", diar)

        audio, _ = sf.read(wav, dtype="float32")
        dur = len(audio) / SR
        chunks = turns(diar, audio)
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            paths = []
            for i, c in enumerate(chunks):
                a, b = max(0.0, c["start"] - 0.2), min(dur, c["end"] + 0.2)
                c["offset"] = a
                p = os.path.join(tmp, f"{i:06d}.wav")
                sf.write(p, audio[int(a * SR):int(b * SR)], SR, subtype="PCM_16")
                paths.append(p)
            with torch.inference_mode():
                hyps = canary.transcribe(paths, batch_size=BATCH, source_lang="cs",
                                         target_lang="cs", timestamps=True)
        hyps = hyps[0] if isinstance(hyps, tuple) else hyps
        segs = []
        for c, h in zip(chunks, hyps):
            ts = h.timestamp.get("word", []) if isinstance(getattr(h, "timestamp", None), dict) else []
            words = [{"word": w.get("word", "").strip(), "start": round(c["offset"] + w["start"], 2),
                      "end": round(c["offset"] + w["end"], 2)} for w in ts if "start" in w]
            segs.append({"start": round(c["start"], 2), "end": round(c["end"], 2),
                         "speaker": c["speaker"], "text": h.text.strip(), "words": words})
        save(name, "canary.json", segs)

    each_video("nemo", jobs, nemo_done, process)


def run_whisper(model_name, jobs):
    """Whisper on the whole file, batched; saves words with timestamps and confidence."""
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    pipe = BatchedInferencePipeline(model=WhisperModel(model_name, device="cuda", compute_type="float16"))

    def process(name, wav):
        segments, _ = pipe.transcribe(wav, language="cs", batch_size=BATCH, beam_size=5,
                                      word_timestamps=True)
        words = [{"word": w.word.strip(), "start": round(w.start, 2), "end": round(w.end, 2),
                  "prob": round(w.probability, 3)}
                 for s in segments for w in (s.words or []) if w.word.strip()]
        save(name, f"whisper-{model_name}.raw.json", words)

    each_video(f"whisper-{model_name}", jobs, lambda n: whisper_done(n, model_name), process)


# ---------------------------------------------------------------- main process

def run(jobs, workers):
    """Start the workers and feed them WAVs, at most WAV_LIMIT bytes on disk at once."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for f in SCRATCH.glob("*"):  # leftovers of a stopped run
        if f.is_file():
            f.unlink()
    for f in sorted(OUT.glob("*/*.unfinished")):
        print(f"{f.parent.name}: {f.stem} was interrupted last time, will be redone", flush=True)

    ctx = get_context("spawn")
    procs = {w: ctx.Process(target=t, args=a, name=w) for w, t, a in workers}
    for p in procs.values():
        p.start()

    todo = [(n, v) for n, v in jobs if needs_wav(n)]
    print(f"{len(jobs)} videos, {len(jobs) - len(todo)} already finished", flush=True)
    on_disk = []
    try:
        while True:
            # delete WAVs every worker is done with (a crashed worker counts as done)
            for name in list(on_disk):
                if all((SCRATCH / f"{name}.done.{w}").exists() or not p.is_alive()
                       for w, p in procs.items()):
                    for f in SCRATCH.glob(f"{name}.*"):
                        f.unlink()
                    on_disk.remove(name)
                    merge_whisper(name)
            # make the next WAVs while they fit into the limit
            while todo and any(p.is_alive() for p in procs.values()):
                name, video = todo[0]
                used = sum((SCRATCH / f"{n}.wav").stat().st_size for n in on_disk
                           if (SCRATCH / f"{n}.wav").exists())
                if on_disk and used + wav_seconds(video) * SR * 2 > WAV_LIMIT:
                    break
                todo.pop(0)
                on_disk.append(name)
                try:
                    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(video),
                                    "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le",
                                    str(SCRATCH / f"{name}.wav")], check=True)
                    (SCRATCH / f"{name}.ready").touch()
                except subprocess.CalledProcessError:
                    print(f"{name}: could not read the audio, skipping", flush=True)
                    (SCRATCH / f"{name}.failed").touch()
            if not any(p.is_alive() for p in procs.values()):
                break
            time.sleep(2)
    finally:
        for p in procs.values():
            if p.is_alive():
                p.terminate()
        for f in SCRATCH.glob("*"):
            if f.is_file():
                f.unlink()
    for name, _ in jobs:
        merge_whisper(name)
    failed = [w for w, p in procs.items() if p.exitcode != 0]
    print("FAILED: " + ", ".join(failed) if failed else "all done", flush=True)
    return not failed


def main():
    videos = []
    for x in map(Path, sys.argv[1:]):
        videos += sorted(p for p in x.iterdir() if p.suffix.lower() in MEDIA) if x.is_dir() else [x]
    if not videos:
        sys.exit(__doc__)

    # faster-whisper needs the cuDNN/cuBLAS that came with PyTorch; set before the
    # worker processes start so they inherit it
    libs = []
    for mod in ("nvidia.cudnn.lib", "nvidia.cublas.lib"):
        try:
            libs.append(list(importlib.import_module(mod).__path__)[0])
        except Exception:
            pass
    os.environ["LD_LIBRARY_PATH"] = ":".join(libs + [os.environ.get("LD_LIBRARY_PATH", "")])

    jobs = [(v.stem, str(v)) for v in videos]
    workers = [("nemo", run_nemo, (jobs,))] + [(f"whisper-{m}", run_whisper, (m, jobs)) for m in WHISPER]
    sys.exit(0 if run(jobs, workers) else 1)


if __name__ == "__main__":
    main()
