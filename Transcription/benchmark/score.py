"""Score every model against the corrected reference transcripts.

    python score.py --hyp hyp/ --ref reference/

Two word error rates per model:
  WER-std  colloquial and standard Czech forms of the same word count as equal (týdletý = této,
           jinejch = jiných, existujou = existují, ...; see textnorm.standardize). This is the one that
           matters for the RAG corpus, where writing the standard form is fine, even better.
  WER      strict, word for word against the verbatim reference.
plus the character error rate (CER, strict), the per-clip WER-std, and the speed (times real time)
measured on the GPU job. A reference file counts only once its first line no longer says DRAFT:
correctors delete that line when they finish the file.
"""
import argparse
import json
from pathlib import Path

from textnorm import errors, normalize, standardize


def read_reference(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    text = [ln for ln in lines if ln.strip() and not ln.startswith("#") and not ln.startswith("[")]
    return normalize(" ".join(text).replace("[?]", " "))


def rate(c: dict) -> float:
    return (c["sub"] + c["del"] + c["ins"]) / max(c["n"], 1)


def add(total: dict, c: dict):
    for k in total:
        total[k] += c[k]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hyp", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--include-drafts", action="store_true", help="also score files still marked DRAFT")
    ap.add_argument("--clips", nargs="*", help="only clips whose name starts with one of these (e.g. debate)")
    args = ap.parse_args()

    refs = {}
    for p in sorted(args.ref.glob("*.txt")):
        if p.name.endswith(".draft-new.txt"):
            continue
        if args.clips and not any(p.stem.startswith(c) for c in args.clips):
            continue
        if p.read_text(encoding="utf-8").startswith("# DRAFT") and not args.include_drafts:
            print(f"skipping {p.name}: still marked DRAFT")
            continue
        refs[p.stem] = read_reference(p)
    if not refs:
        raise SystemExit("No finished reference files.")

    models = sorted(p.name for p in args.hyp.iterdir() if p.is_dir())
    clips = sorted(refs)
    rows = []
    for m in models:
        strict, lenient, chars = ({"sub": 0, "del": 0, "ins": 0, "n": 0} for _ in range(3))
        per_clip, audio, gpu = {}, 0.0, 0.0
        for c in clips:
            f = args.hyp / m / f"{c}.json"
            if not f.exists():
                continue
            h = json.loads(f.read_text(encoding="utf-8"))
            hyp = normalize(h["text"])
            add(strict, errors(refs[c], hyp))
            le = errors([standardize(w) for w in refs[c]], [standardize(w) for w in hyp])
            add(lenient, le)
            add(chars, errors(list(" ".join(refs[c])), list(" ".join(hyp))))
            per_clip[c] = rate(le)
            audio += h["audio_seconds"]
            gpu += h["seconds"]
        if per_clip:
            rows.append((rate(lenient), m, rate(strict), rate(chars), per_clip, lenient,
                         audio / gpu if gpu else 0, len(per_clip)))

    short = [c.split("_")[0] for c in clips]
    print(f"\n{len(clips)} clip(s), {sum(len(r) for r in refs.values())} reference words")
    print("WER-std = colloquial/standard forms count as equal; per-clip columns are WER-std\n")
    print(f"{'model':28s} {'WER-std':>7s} {'WER':>6s} {'CER':>6s}  " + "  ".join(f"{s:>7s}" for s in short)
          + "   sub/del/ins (std)   speed")
    for wer_std, m, wer, cer, per_clip, t, speed, n in sorted(rows):
        cells = "  ".join(f"{per_clip[c]:7.1%}" if c in per_clip else f"{'-':>7s}" for c in clips)
        flag = "" if n == len(clips) else f"  (only {n} clips)"
        print(f"{m:28s} {wer_std:7.1%} {wer:6.1%} {cer:6.1%}  {cells}   {t['sub']:4d}/{t['del']:3d}/{t['ins']:3d}"
              f"        {speed:5.0f}x{flag}")


if __name__ == "__main__":
    main()
