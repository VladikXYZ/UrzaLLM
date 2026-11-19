"""Transcribe the Urza audio on a GPU with faster-whisper large-v3, with a confidence for every word.

Every audio file gets its own folder, <out>/<video_name>__faster-whisper-<model>/, with
  transcript.json   segments with text, language, Whisper's quality signals, and every word as
                    {"word", "start", "end", "prob"}: prob is Whisper's probability for that word
                    (0-1; below ~0.5 it was unsure), start/end its time in the whole video
  transcript.txt    "[hh:mm:ss] text" lines, for reading

The audio is cut into windows of 10-29.5 s that end at a pause (the cutter from transcribe_einfra.py,
which won the benchmark), silent windows are skipped, and each window is transcribed on its own, one after
another. Before that, Whisper detects each window's language: clearly English windows are transcribed as
English, clearly non-English ones as Czech, and unsure ones both ways, keeping the version Whisper is more
confident in (see UNSURE_LOW). With the language forced, Whisper translates instead of transcribing.
Known Whisper inventions (subtitle credits, a web address it learned from training subtitles) are moved
to "dropped".

    python transcribe_local.py --audio-dir AUDIO --out transcriptions [--shard K/N] [--limit 1]
    python transcribe_local.py --audio-dir AUDIO --out transcriptions --fix [--shard K/N]

Finished videos (with transcript.json) are skipped, so an interrupted job just runs again.
--fix brings transcripts made by older versions of this script up to date (see fix_file): it redoes only
the windows whose language changes or is unsure, applies the cleanup, and sets "version" (videos already
at the current version are skipped). When it re-transcribes anything, the previous file is kept as
transcript.v<old version>.json.
"""
import argparse
import json
import re
import shutil
import sys
import time
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcribe_einfra import (AUDIO_EXTS, HALLUCINATION_RE, SR, SUSPECT_COMPRESSION,  # noqa: E402
                               SUSPECT_LOGPROB, frame_db, hms, load_titles, make_windows,
                               safe_name, write_json)

LOW_PROB = 0.5          # a word below this probability counts as "unsure" in the stats
# Language of a window from Whisper's probability of English (p_en): <= UNSURE_LOW Czech, >= UNSURE_HIGH
# English, in between transcribed both ways, keeping the version with the higher mean word probability.
# Why: Urza's Czech-accented English often gets p_en 0.3-0.5, but so does some Czech speech, and
# Czech transcribed as English comes out as a translation with word prob ~0.63 (real English ~0.9).
UNSURE_LOW, UNSURE_HIGH = 0.2, 0.8
FIX_VERSION = 2         # 1: language check + cleanup; 2: unsure windows decided by word probability

# Whisper writes these where nobody speaks: they come from the subtitles it was trained on. Counted on
# all 980 videos: "www.hradeckralove.org" 893x, "www.arkance-systems.cz" 7x, never actually said.
INVENTED_RE = re.compile(HALLUCINATION_RE.pattern + r"|hradeckralove|arkance-systems", re.IGNORECASE)
# Real at the end of a talk, invented in silence (38x with no_speech_prob > 0.6).
THANKS_RE = re.compile(r"^děkuj(i|eme)\s+za\s+pozornost\W*$", re.IGNORECASE)
SILENCE_NO_SPEECH = 0.6


def folder_for(src: Path, args) -> Path:
    return args.out / f"{safe_name(src.stem)}__faster-whisper-{safe_name(args.model)}"


def r2(x):
    return None if x is None else round(float(x), 2)


def decode(src: Path) -> np.ndarray:
    """The file as int16 mono 16 kHz, via PyAV (faster_whisper.decode_audio passes an argument that
    newer PyAV versions no longer accept)."""
    import av
    parts = []
    with av.open(str(src)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SR)
        for frame in container.decode(stream):
            for f in resampler.resample(frame):
                parts.append(f.to_ndarray().reshape(-1))
        for f in resampler.resample(None):
            parts.append(f.to_ndarray().reshape(-1))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int16)


def load_audio(src: Path, args):
    """(audio as float32, duration, all windows, spoken windows)."""
    pcm = decode(src)
    wins = make_windows(frame_db(pcm), len(pcm), args.min_window, args.max_window, args.silent_db)
    return pcm.astype(np.float32) / 32768.0, len(pcm) / SR, wins, [w for w in wins if not w["silent"]]


def piece(audio, w):
    return audio[round(w["start"] * SR):round(w["end"] * SR)]


def unsure(w) -> bool:
    return w.get("p_en") is not None and UNSURE_LOW < w["p_en"] < UNSURE_HIGH


def detect_languages(model, audio, windows, args):
    """Set w["p_en"] and w["language"] ("cs", "en", or "unsure" = decided in transcribe_windows) on
    every window; with a forced language, that language and p_en None."""
    for w in windows:
        if args.language != "auto":
            w["language"], w["p_en"] = args.language, None
            continue
        _, _, probs = model.detect_language(piece(audio, w))
        w["p_en"] = round(float(dict(probs).get("en", 0.0)), 3)
        w["language"] = "unsure" if unsure(w) else ("en" if w["p_en"] >= UNSURE_HIGH else "cs")


def why_dropped(seg: dict):
    if INVENTED_RE.search(seg["text"]):
        return "known hallucination"
    if THANKS_RE.match(seg["text"]) and (seg["no_speech_prob"] or 0) > SILENCE_NO_SPEECH:
        return "known hallucination in silence"
    return None


def transcribe_window(model, audio, w, language, args) -> list[dict]:
    """One window with WhisperModel.transcribe. If Whisper stops before the end of the window, it
    continues from its last timestamp, so nothing is skipped (this is what the benchmark measured: 2.4%
    WER-std on the debate clip; BatchedInferencePipeline decodes each window only once and dropped words,
    6.0%)."""
    segs = []
    out, _ = model.transcribe(piece(audio, w), language=language, beam_size=args.beam,
                              word_timestamps=True, condition_on_previous_text=False, vad_filter=False)
    t = lambda x: r2(min(w["start"] + x, w["end"]))   # noqa: E731  (window time -> video time)
    for s in out:
        text = s.text.strip()
        if not text:
            continue
        words = [{"word": x.word.strip(), "start": t(x.start), "end": t(x.end),
                  "prob": round(float(x.probability), 3)} for x in (s.words or []) if x.word.strip()]
        seg = {"id": 0, "start": t(s.start), "end": t(s.end), "language": language, "text": text,
               "avg_logprob": r2(s.avg_logprob), "no_speech_prob": r2(s.no_speech_prob),
               "compression_ratio": r2(s.compression_ratio),
               "min_word_prob": min((x["prob"] for x in words), default=None), "words": words}
        seg["suspect"] = bool((seg["compression_ratio"] or 0) > SUSPECT_COMPRESSION
                              or (seg["avg_logprob"] is not None and seg["avg_logprob"] < SUSPECT_LOGPROB))
        segs.append(seg)
    return segs


def mean_word_prob(segs) -> float:
    probs = [x["prob"] for s in segs for x in s["words"]]
    return round(float(np.mean(probs)), 3) if probs else 0.0


def transcribe_windows(model, audio, windows, args) -> list[dict]:
    """Every window in its language; an unsure window both in Czech and in English, keeping the version
    with the higher mean word probability (recorded in w["word_prob"])."""
    segs = []
    for w in windows:
        if not unsure(w) or args.language != "auto":
            segs += transcribe_window(model, audio, w, w["language"], args)
            continue
        tries = {lang: transcribe_window(model, audio, w, lang, args) for lang in ("cs", "en")}
        w["word_prob"] = {lang: mean_word_prob(s) for lang, s in tries.items()}
        w["language"] = max(w["word_prob"], key=w["word_prob"].get)
        segs += tries[w["language"]]
    return segs


def finish(segs: list[dict], dropped: list[dict], spoken: list[dict]) -> dict:
    """Clean up, sort and number the segments; the stats fields of transcript.json."""
    kept = []
    for s in segs:
        why = why_dropped(s)
        if why:
            dropped.append({"start": s["start"], "text": s["text"], "why": why})
        else:
            kept.append(s)
    kept.sort(key=lambda s: s["start"])
    for i, s in enumerate(kept):
        s["id"] = i
    dropped.sort(key=lambda d: d["start"] or 0)
    probs = [w["prob"] for s in kept for w in s["words"]]
    seconds = Counter()
    for w in spoken:
        seconds[w["language"]] += w["end"] - w["start"]
    return {
        "languages": {k: round(v, 1) for k, v in seconds.most_common()},
        "windows": [{k: w[k] for k in ("start", "end", "language", "p_en", "word_prob") if k in w}
                    for w in spoken],
        "n_segments": len(kept), "n_suspect": sum(s["suspect"] for s in kept),
        "n_words": len(probs),
        "mean_word_prob": round(float(np.mean(probs)), 3) if probs else None,
        f"share_words_below_{LOW_PROB}": round(sum(p < LOW_PROB for p in probs) / len(probs), 3) if probs else None,
        "n_dropped": len(dropped),
        "segments": kept, "dropped": dropped,
    }


def transcribe_file(src: Path, model, args, titles: dict) -> dict:
    t0 = time.time()
    audio, duration, wins, spoken = load_audio(src, args)
    t_decode = time.time() - t0
    detect_languages(model, audio, spoken, args)
    segs = transcribe_windows(model, audio, spoken, args)
    took = time.time() - t0
    meta = titles.get(src.stem, {})
    result = {
        "video_id": src.stem, "title": meta.get("title"), "upload_date": meta.get("upload_date"),
        "source_file": str(src),
        "version": FIX_VERSION, "model": args.model, "engine": f"faster-whisper (beam {args.beam})",
        "language": args.language, "duration": round(duration, 2),
        "n_windows": len(wins), "n_silent_windows": len(wins) - len(spoken),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds_taken": round(took, 1), "seconds_decoding_audio": round(t_decode, 1),
        "speed_x_realtime": round(duration / max(took, 1e-6), 1),
    }
    result.update(finish(segs, [], spoken))
    return result


def fix_file(src: Path, old: dict, model, args) -> dict:
    """Bring an older transcript up to FIX_VERSION; windows whose language is clear keep their segments.
    No language check yet (first run, Czech forced): detect every window's language, redo the English
    and unsure ones. Version 1 (unsure windows were decided by p_en alone): redo the unsure ones."""
    t0 = time.time()
    segs = old["segments"]
    for s in segs:
        s.setdefault("language", "cs")
    if version(old) == 1:
        spoken = old["windows"]
        redo = [w for w in spoken if unsure(w)]
        for w in redo:
            w["language"] = "unsure"
        audio = load_audio(src, args)[0] if redo else None
    else:
        audio, _, _, spoken = load_audio(src, args)
        detect_languages(model, audio, spoken, args)
        redo = [w for w in spoken if w["language"] != "cs"]
    if redo:
        segs = [s for s in segs if not any(w["start"] <= s["start"] < w["end"] for w in redo)]
        segs += transcribe_windows(model, audio, redo, args)
    old.update(finish(segs, old.get("dropped", []), spoken))
    old["language"], old["version"] = args.language, FIX_VERSION
    old["fixed"] = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "windows_retranscribed": len(redo) + old.get("fixed", {}).get("windows_retranscribed", 0),
                    "seconds_taken": round(time.time() - t0, 1)}
    return old


def version(j: dict) -> int:
    """0: first run (Czech forced), 1: language check by p_en alone + cleanup, 2: current."""
    return j.get("version", 1 if "windows" in j else 0)


def write_outputs(folder: Path, result: dict):
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "transcript.json", result)
    (folder / "transcript.txt").write_text(
        "".join(f"[{hms(s['start'])}] {s['text']}\n" for s in result["segments"]), encoding="utf-8")
    (folder / "error.txt").unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("transcriptions"))
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--language", default="auto",
                    help="auto (default): English windows as English, the rest as Czech; or force one, e.g. cs")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--fix", action="store_true",
                    help="upgrade finished transcripts made before the language check (see above)")
    ap.add_argument("--files", nargs="*", help="only these file names or stems (video ids)")
    ap.add_argument("--limit", type=int, help="process at most this many files (that are not done yet)")
    ap.add_argument("--shard", help="K/N: only the K-th of N disjoint parts (one per job-array task)")
    ap.add_argument("--min-window", type=float, default=10.0)
    ap.add_argument("--max-window", type=float, default=29.5)
    ap.add_argument("--silent-db", type=float, default=-55.0)
    ap.add_argument("--metadata", type=Path,
                    default=Path(__file__).resolve().parent.parent / "jsons" / "channel_metadata.json")
    args = ap.parse_args()

    files = sorted(p for p in args.audio_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    if args.files:
        wanted = set(args.files)
        files = [p for p in files if p.name in wanted or p.stem in wanted]
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        files = [p for p in files if zlib.crc32(p.stem.encode()) % n == k - 1]
    done = [p for p in files if (folder_for(p, args) / "transcript.json").exists()]
    todo = done if args.fix else [p for p in files if p not in done]
    print(f"{len(files)} file(s), {len(done)} with a transcript", flush=True)
    if args.limit is not None:
        todo = todo[:args.limit]
    if not todo:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    titles = load_titles(args.metadata)

    from faster_whisper import WhisperModel
    t = time.time()
    model = WhisperModel(args.model, device="cuda", compute_type="float16")
    print(f"model {args.model} loaded in {time.time() - t:.0f}s", flush=True)

    failed = skipped = 0
    for i, src in enumerate(todo, 1):
        folder = folder_for(src, args)
        try:
            if args.fix:
                old = json.loads((folder / "transcript.json").read_text(encoding="utf-8"))
                if version(old) >= FIX_VERSION:
                    skipped += 1
                    continue
                before, redone = version(old), old.get("fixed", {}).get("windows_retranscribed", 0)
                result = fix_file(src, old, model, args)
                if result["fixed"]["windows_retranscribed"] > redone:   # text changes: keep the old file
                    backup = folder / f"transcript.v{before}.json"
                    if not backup.exists():
                        shutil.copy2(folder / "transcript.json", backup)
                write_outputs(folder, result)
                print(f"[{i}/{len(todo)}] {src.stem}: languages {result['languages']}, "
                      f"{result['fixed']['windows_retranscribed']} window(s) redone, "
                      f"{result['n_dropped']} dropped, {result['fixed']['seconds_taken']:.0f}s", flush=True)
                continue
            result = transcribe_file(src, model, args, titles)
            write_outputs(folder, result)
            print(f"[{i}/{len(todo)}] {src.stem}: {hms(result['duration'])} in {result['seconds_taken']:.0f}s "
                  f"({result['speed_x_realtime']:.0f}x), languages {result['languages']}, "
                  f"{result['n_words']} words, mean word prob {result['mean_word_prob']}", flush=True)
        except Exception as e:  # keep going with the other files
            failed += 1
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "error.txt").write_text(f"{datetime.now().isoformat()}\n{e!r}\n", encoding="utf-8")
            print(f"[{i}/{len(todo)}] FAILED {src.name}: {e!r}", flush=True)
    print(f"finished: {len(todo) - failed - skipped} done, {skipped} already fixed, {failed} failed", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
