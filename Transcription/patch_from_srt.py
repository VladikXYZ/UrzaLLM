"""Fill a gap in a Whisper transcript with YouTube's captions (an .srt file, e.g. from DownSub).

    python patch_from_srt.py TRANSCRIPT_JSON CAPTIONS_SRT --range 2:57 3:26

Whisper sometimes takes quiet or distant speech for silence and writes nothing there (or a stock phrase
that the cleanup drops). This copies the caption lines that lie entirely inside --range and don't overlap
any existing Whisper segment into transcript.json as new segments, one per caption line, marked
"source": "youtube-captions". Captions have no per-word times or confidence, so each word gets an evenly
spread time inside its line and "prob": null. Caption text is kept as YouTube wrote it.

Before changing anything, the file is copied to transcript.v<version>.json (once); the change is listed
under "patches" in the file, and transcript.txt is rewritten.
"""
import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


def secs(x: str) -> float:
    """'3:26', '1:02:03', '206.4' or SRT '00:03:26,390' -> seconds."""
    x = x.replace(",", ".")
    out = 0.0
    for part in x.split(":"):
        out = out * 60 + float(part)
    return out


def hms(t: float) -> str:
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def read_srt(p: Path):
    lines = []
    for block in re.split(r"\n\s*\n", p.read_text(encoding="utf-8-sig").strip()):
        rows = block.strip().splitlines()
        tline = next((i for i, r in enumerate(rows) if "-->" in r), None)
        if tline is None:
            continue
        a, b = (secs(x.strip()) for x in rows[tline].split("-->"))
        text = " ".join(r.strip() for r in rows[tline + 1:] if r.strip())
        if text:
            lines.append((a, b, text))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("transcript", type=Path)
    ap.add_argument("srt", type=Path)
    ap.add_argument("--range", nargs=2, required=True, metavar=("START", "END"))
    ap.add_argument("--language", default="cs")
    args = ap.parse_args()
    lo, hi = (secs(x) for x in args.range)
    t = json.loads(args.transcript.read_text(encoding="utf-8"))
    have = [(s["start"], s["end"]) for s in t["segments"]]
    new = []
    for a, b, text in read_srt(args.srt):
        if a < lo or b > hi or any(a < e and b > s for s, e in have):
            continue
        ws = text.split()
        step = (b - a) / len(ws)
        new.append({"id": 0, "start": round(a, 2), "end": round(b, 2), "language": args.language, "text": text,
                    "avg_logprob": None, "no_speech_prob": None, "compression_ratio": None, "min_word_prob": None,
                    "words": [{"word": w, "start": round(a + i * step, 2), "end": round(a + (i + 1) * step, 2), "prob": None}
                              for i, w in enumerate(ws)],
                    "suspect": False, "source": "youtube-captions"})
    if not new:
        raise SystemExit("no caption line lies entirely inside the range without overlapping Whisper's segments")

    folder = args.transcript.parent
    backup = folder / f"transcript.v{t.get('version', 0)}.json"
    if not backup.exists():
        shutil.copy2(args.transcript, backup)
    segs = sorted(t["segments"] + new, key=lambda s: s["start"])
    for i, s in enumerate(segs):
        s["id"] = i
    probs = [w["prob"] for s in segs for w in s["words"] if w["prob"] is not None]
    t.update({"segments": segs, "n_segments": len(segs), "n_words": sum(len(s["words"]) for s in segs),
              "mean_word_prob": round(sum(probs) / len(probs), 3) if probs else None,
              "share_words_below_0.5": round(sum(p < 0.5 for p in probs) / len(probs), 3) if probs else None})
    t.setdefault("patches", []).append({"source": "youtube-captions", "file": args.srt.name, "range": [lo, hi],
                                        "segments_added": len(new), "words_added": sum(len(s["words"]) for s in new),
                                        "date": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    args.transcript.write_text(json.dumps(t, ensure_ascii=False, indent=1), encoding="utf-8")
    (folder / "transcript.txt").write_text("".join(f"[{hms(s['start'])}] {s['text']}\n" for s in segs), encoding="utf-8")
    print(f"added {len(new)} caption lines ({hms(new[0]['start'])}-{hms(new[-1]['end'])}), "
          f"{sum(len(s['words']) for s in new)} words; backup {backup.name}")
    for s in new:
        print(f"  [{hms(s['start'])}] {s['text']}")


if __name__ == "__main__":
    main()
