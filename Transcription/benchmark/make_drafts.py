"""Build one reference draft per clip for people to correct while listening.

    python make_drafts.py --hyp hyp/ --out reference/

All models were run on the same windows, so the draft goes window by window. For each window it shows
the text of the model that agrees most with the others (the editable line) and, as '#' comment lines,
what every other model heard there, with the words that differ marked «like this». Correctors fix the
editable line to exactly what was said; score.py then reads only the editable lines.
"""
import argparse
import json
from pathlib import Path

from textnorm import align, errors, normalize

HEADER = """\
# DRAFT - delete this line when the whole file is corrected (score.py skips files that still have it)
# Reference transcript: clip {clip}
# Video: https://youtu.be/{vid}?t={t0}  (the clip is {dur:.0f} s from {t0_hms} in the video)
#
# HOW TO CORRECT: under each [mm:ss-mm:ss] line (time inside the clip; the link plays that spot on
# YouTube) fix the one text line so it is EXACTLY what was said, word for word, in Czech, including
# filler words that are real words ("no", "jo", "prostě"), repetitions and unfinished words.
# Leave out non-words ("ehm", "hmm"). Write numbers as you would in a normal text ("2017", "pět lidí").
# Punctuation and capitals do not matter. Unintelligible: write [?] and move on.
# Listen to the whole window, not only to the marked spots: the models can all be wrong the same way.
# The '#' lines below each window show what the other models heard (differences in «»); they are
# ignored when scoring. Draft line taken from: {base}.
"""


def hms(t: float) -> str:
    t = int(round(t))
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60:02d}:{t % 60:02d}"


def mark(base: str, other: str) -> str:
    """`other` with the words that differ from `base` wrapped in «»."""
    b, o = normalize(base), other.split()
    o_norm = [" ".join(normalize(w)) for w in o]
    keep = [i for i, w in enumerate(o_norm) if w]          # tokens that are words after normalising
    steps = align(b, [o_norm[i] for i in keep])
    diff = {keep[j] for op, _, j in steps if op in ("sub", "ins")}
    out, open_ = [], False
    for k, w in enumerate(o):
        if k in diff and not open_:
            w, open_ = "«" + w, True
        if open_ and (k + 1 not in diff):
            w, open_ = w + "»", False
        out.append(w)
    gaps = sum(op == "del" for op, _, _ in steps)
    return " ".join(out) + (f"   (+{gaps} word(s) missing)" if gaps else "")


def window_texts(h: dict, windows: list[dict]) -> list[str]:
    """Text per shared window. Hypotheses with their own segments (long-form) are mapped by midpoint."""
    if len(h["windows"]) == len(windows) and all(
            abs(a["start"] - b["start"]) < 0.01 for a, b in zip(h["windows"], windows)):
        return [w.get("text", "") for w in h["windows"]]
    texts = [[] for _ in windows]
    for s in h["windows"]:
        mid = (s["start"] + s["end"]) / 2
        k = next((i for i, w in enumerate(windows) if w["start"] <= mid < w["end"]), len(windows) - 1)
        texts[k].append(s["text"])
    return [" ".join(t) for t in texts]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hyp", type=Path, required=True, help="hyp/<model>/<clip>.json from bench_asr.py")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    models = sorted(p.name for p in args.hyp.iterdir() if p.is_dir())
    clips = sorted({f.stem for m in models for f in (args.hyp / m).glob("*.json")})
    for clip in clips:
        hyps = {m: json.loads((args.hyp / m / f"{clip}.json").read_text(encoding="utf-8"))
                for m in models if (args.hyp / m / f"{clip}.json").exists()}
        windowed = [m for m in hyps if m != "whisper-large-v3-longform"]
        windows = hyps[windowed[0]]["windows"]
        texts = {m: window_texts(h, windows) for m, h in hyps.items()}
        # the draft comes from the model closest to all the others (fewest summed word differences)
        words = {m: normalize(h["text"]) for m, h in hyps.items()}
        base = min(windowed, key=lambda m: sum(sum(v for k, v in errors(words[o], words[m]).items() if k != "n")
                                               for o in hyps if o != m))
        vid, t0 = clip.split("_")[1], int(clip.split("_")[-1])
        dur = hyps[base]["audio_seconds"]
        lines = [HEADER.format(clip=clip, vid=vid, t0=t0, t0_hms=hms(t0), dur=dur, base=base)]
        for k, w in enumerate(windows):
            lines.append(f"[{hms(w['start'])}-{hms(w['end'])}]  https://youtu.be/{vid}?t={t0 + int(w['start'])}")
            lines.append(texts[base][k])
            for m in sorted(windowed):   # long-form cuts the audio its own way, so it does not line up here
                if m == base:
                    continue
                other = texts[m][k]
                if normalize(other) != normalize(texts[base][k]):
                    lines.append(f"#   {m:26s} {mark(texts[base][k], other) or '(nothing)'}")
            lines.append("")
        path = args.out / f"{clip}.txt"
        if path.exists():
            path = args.out / f"{clip}.draft-new.txt"   # never overwrite a file someone is correcting
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"{path}  (draft from {base})")


if __name__ == "__main__":
    main()
