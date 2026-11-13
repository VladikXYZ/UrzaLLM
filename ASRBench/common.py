"""Shared helpers for the ASR + diarization benchmark.

Every approach writes the same two files per run:
  <name>.json  segments + words + diarization + timing (used by score.py)
  <name>.txt   one line per segment, editable by hand (used as the reference)

Line format of the .txt files:
  [00:01:02.30 --> 00:01:15.80] SPEAKER: text of the segment
Speaker names are free-form (S0, URZA, Q1, ...); scoring is permutation-invariant.
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

SR = 16000
FRAME = 0.01  # seconds, timeline resolution for turn building and DER


# ---------------------------------------------------------------- terminal output
# run_all.sh sends each runner's full output (library warnings etc.) to a log file.
# Progress bars and short status lines still go to the terminal via /dev/tty.

_TERM = None


def terminal():
    """Stream that reaches the user's terminal, or stderr if there is none."""
    global _TERM
    if _TERM is None:
        _TERM = sys.stderr
        if not sys.stderr.isatty():
            try:
                _TERM = open("/dev/tty", "w")
            except OSError:
                pass
    return _TERM


def say(msg):
    """Status line: into the log (stdout) and onto the terminal."""
    print(msg, flush=True)
    t = terminal()
    if not sys.stdout.isatty() and t is not sys.stderr:
        print(f"  {msg}", file=t, flush=True)


def progress(total, desc, unit="s"):
    """A tqdm bar on the terminal; total in seconds of audio by default."""
    from tqdm import tqdm
    return tqdm(total=round(total), desc=f"  {desc}", unit=unit, file=terminal(),
                dynamic_ncols=True, mininterval=0.5,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.0f}/{total:.0f} {unit} "
                           "[{elapsed}<{remaining}]")


# ---------------------------------------------------------------- audio

def load_audio(path):
    """Read a 16 kHz mono WAV (as produced by to_wav.py) into float32."""
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        raise ValueError(f"{path}: expected {SR} Hz, got {sr} Hz; run to_wav.py first")
    return audio


def write_wav(path, audio):
    import soundfile as sf
    sf.write(str(path), audio, SR, subtype="PCM_16")


def split_long(audio, start, end, max_len=30.0, min_len=15.0):
    """Split [start, end] into pieces of at most max_len seconds,
    cutting at the quietest 200 ms between min_len and max_len."""
    pieces, s = [], start
    hop, win = int(SR * FRAME), 20
    while end - s > max_len:
        seg = audio[int((s + min_len) * SR):int((s + max_len) * SR)]
        frames = seg[: len(seg) // hop * hop].reshape(-1, hop)
        if len(frames) == 0:
            break
        rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
        smooth = np.convolve(rms, np.ones(win) / win, mode="same")
        cut = s + min_len + (int(np.argmin(smooth)) * hop + hop / 2) / SR
        pieces.append((s, cut))
        s = cut
    pieces.append((s, end))
    return pieces


# ---------------------------------------------------------------- diarization -> turns

def exclusive_turns(diar, duration, merge_gap=0.5, min_turn=0.3):
    """Turn possibly-overlapping diarization segments into non-overlapping turns.

    In overlapped speech the speaker who already holds the floor keeps it.
    Same-speaker turns separated by < merge_gap are merged; turns shorter than
    min_turn are absorbed into the previous turn. Each turn records how many
    seconds of it were overlapped speech.
    """
    spks = sorted({d["speaker"] for d in diar})
    if not spks:
        return []
    n = int(np.ceil(duration / FRAME)) + 1
    act = np.zeros((len(spks), n), dtype=bool)
    for d in diar:
        act[spks.index(d["speaker"]), int(d["start"] / FRAME):int(np.ceil(d["end"] / FRAME))] = True
    nact = act.sum(axis=0)
    label = np.where(nact > 0, act.argmax(axis=0), -1)
    total = act.sum(axis=1)
    for t in np.flatnonzero(nact > 1):
        active = np.flatnonzero(act[:, t])
        prev = label[t - 1] if t > 0 else -1
        label[t] = prev if prev in active else active[np.argmax(total[active])]

    bounds = np.concatenate([[0], np.flatnonzero(np.diff(label)) + 1, [n]])
    turns = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if label[a] < 0:
            continue
        turns.append({"start": float(a * FRAME), "end": float(b * FRAME), "speaker": spks[label[a]],
                      "overlap": float((nact[a:b] > 1).sum() * FRAME)})

    out = []
    for tr in turns:
        close = out and tr["start"] - out[-1]["end"] < merge_gap
        if close and (tr["speaker"] == out[-1]["speaker"] or tr["end"] - tr["start"] < min_turn):
            out[-1]["end"] = tr["end"]
            out[-1]["overlap"] += tr["overlap"]
        else:
            out.append(dict(tr))
    return out


def make_chunks(audio, duration, diar, mode, pad=0.2):
    """Cut the audio for transcription. Returns (start, end, speaker_or_None, offset, audio).

    turns: one chunk per speaker turn (long turns split at pauses), so each chunk has
           one speaker; offset is where the padded chunk audio starts
    fixed: <=30 s chunks at pauses, ignoring speakers
    """
    if mode == "turns":
        spans = [(s, e, t["speaker"])
                 for t in exclusive_turns(diar, duration)
                 for s, e in split_long(audio, t["start"], min(t["end"], duration))]
    else:
        spans = [(s, e, None) for s, e in split_long(audio, 0.0, duration)]
    chunks = []
    for s, e, spk in spans:
        a, b = max(0.0, s - pad), min(duration, e + pad)
        chunks.append((s, e, spk, a, audio[int(a * SR):int(b * SR)]))
    return chunks


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def speaker_at(diar, start, end):
    """Speaker with the most overlap with [start, end]; nearest segment if none."""
    best, best_ov = None, 0.0
    for d in diar:
        ov = min(end, d["end"]) - max(start, d["start"])
        if ov > best_ov:
            best, best_ov = d["speaker"], ov
    if best is None and diar:
        mid = (start + end) / 2
        best = min(diar, key=lambda d: min(abs(mid - d["start"]), abs(mid - d["end"])))["speaker"]
    return best or "S?"


def words_to_segments(words, diar, max_gap=1.0):
    """Give each word a speaker from the diarization and group runs into segments."""
    segs = []
    for w in words:
        spk = speaker_at(diar, w["start"], w["end"])
        if segs and segs[-1]["speaker"] == spk and w["start"] - segs[-1]["end"] <= max_gap:
            segs[-1]["end"] = max(segs[-1]["end"], w["end"])
            segs[-1]["words"].append(w)
        else:
            segs.append({"start": w["start"], "end": w["end"], "speaker": spk, "words": [w]})
    for s in segs:
        s["text"] = " ".join(w["word"] for w in s["words"]).strip()
    return segs


# ---------------------------------------------------------------- output files

def fmt_ts(t):
    t = max(0.0, float(t))
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def parse_ts(s):
    parts = [float(p) for p in s.strip().split(":")]
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec


LINE_RE = re.compile(r"^\[\s*([\d:.]+)\s*-->\s*([\d:.]+)\s*\]\s*([^:]+?)\s*:\s*(.*)$")


def write_outputs(out_dir, name, segments, meta):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    segments = sorted(segments, key=lambda s: s["start"])
    data = dict(meta, name=name, segments=segments)
    (out_dir / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"# {name}"]
    for s in segments:
        lines.append(f"[{fmt_ts(s['start'])} --> {fmt_ts(s['end'])}] {s['speaker']}: {s['text']}")
    (out_dir / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / name}.json/.txt ({len(segments)} segments)")


def write_diar_only(out_dir, diar_name, diar, wav, duration, secs):
    """Save a diarization result so other runners can reuse it with --diar-json."""
    write_outputs(out_dir, f"{diar_name}_diar_only", [dict(d, text="") for d in diar], {
        "audio": str(wav), "duration": duration, "diar_name": diar_name,
        "diar_seconds": secs, "diarization": diar})


def load_diar_json(path):
    """Return (diarization, seconds it took, diarizer name) from any output JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    diar = data.get("diarization") or [
        {k: s[k] for k in ("start", "end", "speaker")} for s in data["segments"]]
    name = data.get("diar_name") or Path(path).stem.replace("_diar_only", "")
    return diar, float(data.get("diar_seconds") or 0.0), name


def read_segments(path):
    """Load segments from a .json output or a (hand-edited) .txt file."""
    path = Path(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["segments"], data
    segs = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = LINE_RE.match(line)
        if not m:
            raise ValueError(f"{path}:{i}: cannot parse line: {line!r}")
        segs.append({"start": parse_ts(m[1]), "end": parse_ts(m[2]),
                     "speaker": m[3].strip(), "text": m[4].strip()})
    return segs, {"name": path.stem}


# ---------------------------------------------------------------- text normalisation

def normalize_words(text):
    """Lowercase, drop punctuation, split into words (Czech diacritics kept)."""
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return text.split()
