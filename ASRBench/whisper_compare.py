"""Whisper large-v3 vs. large-v3-turbo, fully automatic, on one or more videos.

  python whisper_compare.py lecture1.webm lecture2.mp4
  python whisper_compare.py path/to/folder          # every audio/video file in it

Both models transcribe every file with identical settings (Czech, batched, beam 5,
word timestamps). Models are loaded once for all files. No hand-made reference is
needed; the report compares the two on:

  speed          seconds of GPU time per file, x realtime
  disagreement   share of words where the two transcripts differ (not an error rate:
                 it says how much is at stake, not who is right)
  failures       repetition loops ("jenom jenom jenom ..."), junk tokens ("[Music]",
                 non-Latin text) and 30 s stretches where one model wrote far less
  vs. others     if out/<name>/ holds outputs of non-Whisper models (e.g. Canary from
                 run_all.sh), how far each Whisper is from them. The two Whispers share
                 most of their network, so their errors overlap; an independent model
                 is a fair referee: the closer Whisper is likely the more accurate one
  vs. reference  if a hand-corrected references/<name>.txt exists: the real error rate

Writes out/<name>/whisper-large-v3_plain.* and whisper-large-v3-turbo_plain.* (reused on
later runs; --force redoes them), out/whisper_compare/report.md and disagreements.csv.
"""
import argparse
import csv
import json
import re
import time
from pathlib import Path

from common import fmt_ts, progress, read_segments, say, write_outputs
from make_review import align, tokens
from to_wav import convert

HERE = Path(__file__).parent
MODELS = [("large-v3", "large-v3"), ("large-v3-turbo", "turbo")]
MEDIA = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".mp3", ".m4a", ".wav", ".opus", ".ogg", ".flac", ".aac"}
WIN = 60.0          # alignment window (s); keeps alignment fast on 2-hour files
JUNK = re.compile(r"[　-鿿가-힯]|^\[.*\]$|^\(.*\)$|♪")


# ---------------------------------------------------------------- transcription

def transcribe_all(model_name, items, args):
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "?"
    pipe = None
    for it in items:
        stem = f"whisper-{model_name}_plain"
        if (it["out"] / f"{stem}.json").exists() and not args.force:
            say(f"{model_name}: {it['name']} already transcribed, reusing it")
            continue
        if pipe is None:
            say(f"loading {model_name}")
            t0 = time.perf_counter()
            pipe = BatchedInferencePipeline(WhisperModel(model_name, device="cuda", compute_type="float16"))
            load = time.perf_counter() - t0
        bar = progress(it["secs"], f"{model_name}: {it['name']}")
        t0 = time.perf_counter()
        segs, _ = pipe.transcribe(str(it["wav"]), language=args.lang, batch_size=args.batch_size,
                                  beam_size=args.beam_size, word_timestamps=True)
        segments = []
        for s in segs:  # generator: decoding happens while iterating
            words = [{"word": w.word.strip(), "start": float(w.start), "end": float(w.end)}
                     for w in s.words or [] if w.word.strip()]
            segments.append({"start": float(s.start), "end": float(s.end), "speaker": "S",
                             "text": s.text.strip(), "words": words})
            bar.n = min(round(s.end), bar.total)
            bar.refresh()
        secs = time.perf_counter() - t0
        bar.n = bar.total
        bar.close()
        write_outputs(it["out"], stem, segments, {
            "audio": str(it["wav"]), "duration": it["secs"], "asr_model": model_name,
            "asr_seconds": secs, "diar_seconds": 0.0, "load_seconds": load, "gpu": gpu,
            "settings": {"language": args.lang, "batch_size": args.batch_size, "beam_size": args.beam_size}})
        say(f"{model_name}: {it['name']} in {secs:.0f} s = {it['secs'] / secs:.0f}x realtime")


# ---------------------------------------------------------------- analysis

def words_of(segs):
    return tokens(segs, -1, 1e9)  # (surface, key, time, speaker)


def compare(a, b):
    """Align b against a in WIN-second windows. Returns (edits, spans): edits = number of
    word substitutions + deletions + insertions; spans = places where they differ."""
    edits, spans = 0, []
    end = max([t[2] for t in a + b] or [0])
    w0 = 0.0
    while w0 <= end:
        A = [t for t in a if w0 <= t[2] < w0 + WIN]
        B = [t for t in b if w0 <= t[2] < w0 + WIN]
        w0 += WIN
        if not A and not B:
            continue
        match, ins = align([t[1] for t in B], [t[1] for t in A])
        cur = None

        def start(j):
            t = A[j][2] if j < len(A) else (A[-1][2] if A else B[0][2])
            return {"t": t, "a": [], "b": [], "ctx": " ".join(x[0] for x in A[max(0, j - 4):j])}

        for j in range(len(A) + 1):
            extra = ins.get(j, [])           # turbo words with no counterpart, before A[j]
            if extra:
                cur = cur or start(j)
                cur["b"] += [B[i][0] for i in extra]
                edits += len(extra)
            if j == len(A):
                break
            i = match[j]
            if i is not None and B[i][1] == A[j][1]:   # same word: close an open span
                if cur:
                    spans.append(cur)
                cur = None
            else:                                      # different word, or missing in turbo
                cur = cur or start(j)
                cur["a"].append(A[j][0])
                if i is not None:
                    cur["b"].append(B[i][0])
                edits += 1
        if cur:
            spans.append(cur)
    return edits, spans


def loops(toks):
    """Repetition loops: one word 4+ times in a row, or a 2-4 word phrase 3+ times."""
    keys, found, i = [t[1] for t in toks], [], 0
    while i < len(keys):
        for n in (1, 2, 3, 4):
            reps = 1
            while keys[i + reps * n:i + (reps + 1) * n] == keys[i:i + n]:
                reps += 1
            if reps >= (4 if n == 1 else 3):
                found.append((toks[i][2], " ".join(keys[i:i + n]), reps))
                i += reps * n
                break
        else:
            i += 1
    return found


def dropouts(a, b, end, win=30.0):
    """30 s stretches where one model wrote much less than the other."""
    out, t = [], 0.0
    while t < end:
        na = sum(1 for x in a if t <= x[2] < t + win)
        nb = sum(1 for x in b if t <= x[2] < t + win)
        if max(na, nb) >= 20 and min(na, nb) <= 0.3 * max(na, nb):
            out.append((t, na, nb))
        t += win
    return out


def reference_for(name, outdir):
    for p in (HERE / "references" / f"{name}.txt", outdir / "reference.txt"):
        if p.exists():
            return p
    return None


def analyse(it):
    out = it["out"]
    hyp = {}
    for model, label in MODELS:
        segs, meta = read_segments(out / f"whisper-{model}_plain.json")
        hyp[label] = (segs, meta, words_of(segs))
    (sa, ma, a), (sb, mb, b) = hyp["large-v3"], hyp["turbo"]
    edits, spans = compare(a, b)
    r = {"name": it["name"], "minutes": it["secs"] / 60,
         "time": {k: v[1]["asr_seconds"] for k, v in hyp.items()},
         "words": {k: len(v[2]) for k, v in hyp.items()},
         "disagree": edits / max(1, len(a)), "spans": spans,
         "loops": {k: loops(v[2]) for k, v in hyp.items()},
         "junk": {k: [(t[2], t[0]) for t in v[2] if JUNK.search(t[0])] for k, v in hyp.items()},
         "dropouts": dropouts(a, b, it["secs"]), "others": {}, "ref": None}

    # independent referees: outputs of non-Whisper models in the same folder
    for p in sorted(out.glob("*.json")):
        if p.stem.startswith("whisper") or p.stem.endswith("_diar_only") or p.stem.startswith("reference"):
            continue
        segs, meta = read_segments(p)
        model = meta.get("asr_model", p.stem)
        if model in r["others"] or not any(s["text"] for s in segs):
            continue
        o = words_of(segs)
        cov = max(s["end"] for s in segs)
        dist = {}
        for k, v in hyp.items():
            mine = [t for t in v[2] if t[2] <= cov]
            dist[k] = compare(o, mine)[0] / max(1, len(o))
        r["others"][model] = dist

    ref_path = reference_for(it["name"], out)
    if ref_path:
        from score import crop, wer
        ref, _ = read_segments(ref_path)
        lo, hi = min(s["start"] for s in ref), max(s["end"] for s in ref)
        r["ref"] = {"file": ref_path.name, "range": (lo, hi),
                    "wer": {k: wer(ref, crop(v[0], lo, hi)) for k, v in hyp.items()}}
    return r


# ---------------------------------------------------------------- report

def report(results, path_md, path_csv):
    L = ["# Whisper large-v3 vs. large-v3-turbo", ""]
    tot = {k: sum(r["time"][k] for r in results) for _, k in MODELS}
    mins = sum(r["minutes"] for r in results)
    L.append(f"{len(results)} file(s), {mins:.0f} minutes of audio. Same settings for both models.")
    L.append("")
    L.append("| file | min | large-v3 time | turbo time | turbo speed-up | words v3 / turbo | words that differ |")
    L.append("|---|---|---|---|---|---|---|")
    for r in results:
        t3, tt = r["time"]["large-v3"], r["time"]["turbo"]
        L.append(f"| {r['name']} | {r['minutes']:.0f} | {t3:.0f} s ({r['minutes'] * 60 / t3:.0f}x) | "
                 f"{tt:.0f} s ({r['minutes'] * 60 / tt:.0f}x) | {t3 / tt:.1f}x | "
                 f"{r['words']['large-v3']} / {r['words']['turbo']} | {r['disagree']:.1%} |")
    L.append(f"| **all** | {mins:.0f} | {tot['large-v3']:.0f} s | {tot['turbo']:.0f} s | "
             f"**{tot['large-v3'] / tot['turbo']:.1f}x** | | |")

    L += ["", "## Failures (found automatically)", "",
          "| file | repetition loops v3 / turbo | junk tokens v3 / turbo | stretches where one wrote much less |",
          "|---|---|---|---|"]
    for r in results:
        drops = "; ".join(f"{fmt_ts(t)[:8]} (v3 {na} / turbo {nb} words)" for t, na, nb in r["dropouts"][:5]) or "none"
        L.append(f"| {r['name']} | {len(r['loops']['large-v3'])} / {len(r['loops']['turbo'])} | "
                 f"{len(r['junk']['large-v3'])} / {len(r['junk']['turbo'])} | {drops} |")
    for r in results:
        for k in ("large-v3", "turbo"):
            for t, phrase, reps in r["loops"][k][:5]:
                L.append(f"- {r['name']} {fmt_ts(t)[:8]}: {k} repeats \"{phrase}\" {reps}x")

    if any(r["others"] for r in results):
        L += ["", "## Closeness to independent models (lower = closer, likely more accurate)", "",
              "| file | referee model | large-v3 | turbo | closer |", "|---|---|---|---|---|"]
        for r in results:
            for model, d in r["others"].items():
                closer = "large-v3" if d["large-v3"] < d["turbo"] - 0.002 else \
                         "turbo" if d["turbo"] < d["large-v3"] - 0.002 else "tie"
                L.append(f"| {r['name']} | {model} | {d['large-v3']:.1%} | {d['turbo']:.1%} | {closer} |")

    if any(r["ref"] for r in results):
        L += ["", "## Against hand-corrected references (real word error rate, lower is better)", "",
              "| file | reference | part | large-v3 WER | turbo WER |", "|---|---|---|---|---|"]
        for r in results:
            if r["ref"]:
                lo, hi = r["ref"]["range"]
                L.append(f"| {r['name']} | {r['ref']['file']} | {fmt_ts(lo)[:8]}-{fmt_ts(hi)[:8]} | "
                         f"{r['ref']['wer']['large-v3']:.1%} | {r['ref']['wer']['turbo']:.1%} |")

    L += ["", "## How to read this", "",
          "- *Words that differ* is how much is at stake, not an error rate: where they differ, either may be right.",
          "- Loops, junk and stretches with far fewer words are real failures; check them in "
          "`disagreements.csv` or the `.txt` outputs.",
          "- The referee comparison only appears when non-Whisper outputs (e.g. Canary from run_all.sh) "
          "exist for a file; the reference table only when references/<name>.txt exists.",
          f"- All places where the two differ, with time and context: `{path_csv.name}`."]
    path_md.write_text("\n".join(L) + "\n", encoding="utf-8")

    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "time", "context before", "large-v3", "turbo"])
        for r in results:
            for s in r["spans"]:
                w.writerow([r["name"], fmt_ts(s["t"])[:11], s["ctx"], " ".join(s["a"]), " ".join(s["b"])])
    return "\n".join(L)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="video/audio files or folders")
    ap.add_argument("--lang", default="cs")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--force", action="store_true", help="transcribe again even if outputs exist")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    files = []
    for x in map(Path, args.inputs):
        files += sorted(p for p in x.iterdir() if p.suffix.lower() in MEDIA) if x.is_dir() else [x]
    if not files:
        raise SystemExit("no audio/video files given")

    items = []
    for f in files:
        say(f"reading audio: {f.name}")
        wav, secs, _ = convert(f, HERE / "audio")
        items.append({"name": wav.stem, "wav": wav, "secs": secs, "out": Path(args.out) / wav.stem})

    for model, _ in MODELS:
        transcribe_all(model, items, args)

    say("comparing")
    results = [analyse(it) for it in items]
    rep_dir = Path(args.out) / "whisper_compare"
    rep_dir.mkdir(parents=True, exist_ok=True)
    text = report(results, rep_dir / "report.md", rep_dir / "disagreements.csv")
    print("\n" + text)
    say(f"report: {rep_dir / 'report.md'}")


if __name__ == "__main__":
    main()
