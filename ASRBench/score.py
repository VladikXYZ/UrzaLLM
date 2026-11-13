"""Score every approach against the hand-corrected reference.

  python score.py out/X/reference.txt out/X/*.json

Metrics (lower is better):
  WER    words only, speakers ignored
  cpWER  words + speakers: each speaker's text is compared with the best-matching
         reference speaker; wrong speaker labels count as errors. The main number.
  DER    speaker timing only (miss + false alarm + confusion), 0.25 s collar.
         Approximate, since reference times are hand-corrected segment bounds.

Only the time range the reference covers is scored, so for a long video it is enough
to correct one continuous 5-10 minute part and delete the rest. Each output is cut to
that range by word timestamps where it has them, otherwise by segment midpoints.

Writes a markdown table next to the reference (scores.md).
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from common import FRAME, fmt_ts, normalize_words, read_segments

try:
    from rapidfuzz.distance import Levenshtein

    def edit_distance(a, b):
        return Levenshtein.distance(a, b)
except ImportError:
    def edit_distance(a, b):
        prev = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            cur = [i]
            for j, y in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
            prev = cur
        return prev[-1]


def words_of(segs):
    return [w for s in sorted(segs, key=lambda s: s["start"]) for w in normalize_words(s["text"])]


def per_speaker(segs):
    out = {}
    for s in sorted(segs, key=lambda s: s["start"]):
        out.setdefault(s["speaker"], []).extend(normalize_words(s["text"]))
    return out


def wer(ref, hyp):
    r, h = words_of(ref), words_of(hyp)
    return edit_distance(r, h) / max(1, len(r))


def cpwer(ref, hyp):
    R, H = per_speaker(ref), per_speaker(hyp)
    rk, hk = list(R), list(H)
    k = max(len(rk), len(hk))
    cost = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            r = R[rk[i]] if i < len(rk) else []
            h = H[hk[j]] if j < len(hk) else []
            cost[i, j] = edit_distance(r, h)
    ri, ci = linear_sum_assignment(cost)
    n_ref = sum(len(v) for v in R.values())
    mapping = {hk[j]: rk[i] for i, j in zip(ri, ci) if i < len(rk) and j < len(hk)}
    return cost[ri, ci].sum() / max(1, n_ref), mapping


def activity(segs, n):
    spks = sorted({s["speaker"] for s in segs})
    act = np.zeros((len(spks), n), dtype=bool)
    for s in segs:
        act[spks.index(s["speaker"]), int(s["start"] / FRAME):int(np.ceil(s["end"] / FRAME))] = True
    return act


def der(ref, hyp, collar=0.25):
    end = max(s["end"] for s in ref + hyp)
    n = int(np.ceil(end / FRAME)) + 1
    R, H = activity(ref, n), activity(hyp, n)
    keep = np.ones(n, dtype=bool)
    c = int(collar / FRAME)
    for s in ref:
        for b in (s["start"], s["end"]):
            f = int(b / FRAME)
            keep[max(0, f - c):f + c] = False
    R, H = R[:, keep], H[:, keep]
    overlap = R.astype(np.int64) @ H.T.astype(np.int64)
    ri, ci = linear_sum_assignment(-overlap)
    correct = overlap[ri, ci].sum()
    nr, nh = R.sum(axis=0), H.sum(axis=0)
    miss = np.clip(nr - nh, 0, None).sum()
    fa = np.clip(nh - nr, 0, None).sum()
    conf = np.minimum(nr, nh).sum() - correct
    return (miss + fa + conf) / max(1, nr.sum())


def crop(segs, lo, hi):
    """Keep only the part of an output inside [lo, hi] seconds."""
    out = []
    for s in segs:
        words = [w for w in s.get("words") or [] if "start" in w]
        if words:
            kept = [w for w in words if lo <= (w["start"] + w["end"]) / 2 <= hi]
            if kept:
                out.append(dict(s, start=max(lo, kept[0]["start"]), end=min(hi, kept[-1]["end"]),
                                text=" ".join(w["word"] for w in kept), words=kept))
        elif lo <= (s["start"] + s["end"]) / 2 <= hi:
            out.append(dict(s, start=max(lo, s["start"]), end=min(hi, s["end"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", help="hand-corrected .txt")
    ap.add_argument("hyps", nargs="+", help="approach outputs (.json or .txt)")
    ap.add_argument("--collar", type=float, default=0.25)
    args = ap.parse_args()

    ref, _ = read_segments(args.reference)
    if not ref:
        raise SystemExit(f"{args.reference} has no lines")
    ref_path = Path(args.reference).resolve()
    n_ref_spk = len({s["speaker"] for s in ref})
    lo, hi = min(s["start"] for s in ref), max(s["end"] for s in ref)
    rows = []
    for path in args.hyps:
        p = Path(path)
        if p.resolve() == ref_path or p.stem == "reference" or p.stem.endswith("_diar_only"):
            continue
        hyp, meta = read_segments(p)
        if not any(s["text"] for s in hyp):
            continue
        hyp = crop(hyp, lo, hi)
        if not hyp:
            print(f"{p.name}: nothing inside the reference's time range, skipped")
            continue
        cp, mapping = cpwer(ref, hyp)
        dur = meta.get("duration")
        secs = (meta.get("asr_seconds") or 0) + (meta.get("diar_seconds") or 0)
        rows.append({
            "name": meta.get("name", p.stem), "wer": wer(ref, hyp), "cpwer": cp,
            "der": der(ref, hyp, args.collar),
            "spk": len({s["speaker"] for s in hyp}),
            "speed": (f"{dur / secs:.0f}x" if dur / secs >= 10 else f"{dur / secs:.1f}x") if dur and secs else "-",
            "mapping": ", ".join(f"{h}->{r}" for h, r in sorted(mapping.items())),
        })
    rows.sort(key=lambda r: r["cpwer"])

    lines = [
        f"Reference: `{args.reference}`: {fmt_ts(lo)}-{fmt_ts(hi)}, "
        f"{len(words_of(ref))} words, {n_ref_spk} speakers. Speed is for the whole file.\n",
        "| approach | cpWER | WER | DER | speakers | speed (realtime) | speaker mapping |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['name']} | {r['cpwer']:.1%} | {r['wer']:.1%} | {r['der']:.1%} | "
                     f"{r['spk']} | {r['speed']} | {r['mapping']} |")
    table = "\n".join(lines)
    print(table)
    out = Path(args.reference).with_name("scores.md")
    out.write_text(table + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
