"""Build a review app for making the hand-corrected reference quickly.

  python make_review.py out/NAME --start 25:00 --end 35:00

Takes one output per transcription model (Canary, Whisper, Czech Whisper, Qwen3-ASR,
VibeVoice ...), aligns them word by word against a backbone (Canary with Sortformer
speaker turns, which has word timestamps), and marks every place where the models
disagree. The result is out/NAME/review/index.html plus audio.mp3: open index.html in
a browser (double-click), listen, fix, and download reference.txt.

Default text in the app = what most models wrote. Highlighted words are where they
disagree; click one to hear it and pick the right version (or type your own).
"""
import argparse
import hashlib
import json
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path

from common import fmt_ts, parse_ts, read_segments

HERE = Path(__file__).parent
LABELS = {  # asr_model -> (short id, name shown in the app)
    "nvidia/canary-1b-v2": ("C", "Canary"),
    "large-v3": ("W", "Whisper large-v3"),
    "large-v3-turbo": ("Wt", "Whisper turbo"),
    "models/whisper-cs-ct2": ("Wcs", "Whisper Czech"),
    "Qwen/Qwen3-ASR-1.7B": ("Q", "Qwen3-ASR"),
    "microsoft/VibeVoice-ASR-HF": ("V", "VibeVoice"),
    "nvidia/parakeet-tdt-0.6b-v3": ("P", "Parakeet"),
}


def key(word):
    """Comparison form: lowercase, no punctuation."""
    w = unicodedata.normalize("NFC", word).lower()
    return "".join(c for c in w if unicodedata.category(c)[0] not in "PS")


def tokens(segs, lo, hi):
    """(surface, key, time, speaker) for every word whose middle is in [lo, hi]."""
    out = []
    for s in segs:
        words = [w for w in s.get("words") or [] if "start" in w]
        if not words:  # no word timestamps: spread the words evenly over the segment
            parts = s["text"].split()
            step = (s["end"] - s["start"]) / max(1, len(parts))
            words = [{"word": p, "start": s["start"] + i * step, "end": s["start"] + (i + 1) * step}
                     for i, p in enumerate(parts)]
        for w in words:
            k = key(w["word"])
            if k and lo <= (w["start"] + w["end"]) / 2 <= hi:
                out.append((w["word"].strip(), k, round(w["start"], 2), s["speaker"]))
    return out


def similar(a, b):
    return a == b or (len(a) > 3 and len(b) > 3 and a[:4] == b[:4])


def align(hyp, ref):
    """Word alignment (edit distance with backtrace) of hyp keys against ref keys.
    Returns (match[j] = hyp index or None for each ref word, ins[g] = hyp indices
    inserted before ref word g; g == len(ref) means after the last one)."""
    n, m = len(hyp), len(ref)
    INDEL, SIM, SUB = 1.0, 0.6, 1.0
    prev = [j * INDEL for j in range(m + 1)]
    back = [bytes([2]) * (m + 1)]  # 0 = diagonal, 1 = hyp word inserted, 2 = ref word deleted
    for i in range(1, n + 1):
        h = hyp[i - 1]
        cur = [i * INDEL] + [0.0] * m
        bp = bytearray(m + 1)
        bp[0] = 1
        for j in range(1, m + 1):
            r = ref[j - 1]
            d = prev[j - 1] + (0.0 if h == r else SIM if similar(h, r) else SUB)
            u = prev[j] + INDEL
            left = cur[j - 1] + INDEL
            if d <= u and d <= left:
                cur[j] = d
            elif u <= left:
                cur[j], bp[j] = u, 1
            else:
                cur[j], bp[j] = left, 2
        prev = cur
        back.append(bytes(bp))
    match, ins = [None] * m, {}
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j] if i > 0 else 2
        if i > 0 and j > 0 and op == 0:
            match[j - 1] = i - 1
            i, j = i - 1, j - 1
        elif i > 0 and (op == 1 or j == 0):
            ins.setdefault(j, []).insert(0, i - 1)
            i -= 1
        else:
            j -= 1
    return match, ins


def speaker_map(other, base, lo, hi):
    """Map another diarizer's labels onto the backbone's by largest overlap."""
    ov = Counter()
    for a in other:
        for b in base:
            o = min(a["end"], b["end"], hi) - max(a["start"], b["start"], lo)
            if o > 0:
                ov[(a["speaker"], b["speaker"])] += o
    best = {}
    for (a, b), o in ov.most_common():
        best.setdefault(a, b)
    return best


def dominant(segs, s, e):
    c = Counter()
    for x in segs:
        o = min(e, x["end"]) - max(s, x["start"])
        if o > 0:
            c[x["speaker"]] += o
    return c.most_common(1)[0][0] if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", help="out/NAME folder with the approach outputs")
    ap.add_argument("--start", default="0", help="e.g. 25:00")
    ap.add_argument("--end", help="e.g. 35:00 (default: end of file)")
    ap.add_argument("--audio", help="default: audio/NAME.wav")
    ap.add_argument("--backbone", default="sortformer_canary_turns")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    name = outdir.name
    audio = Path(args.audio or HERE / "audio" / f"{name}.wav")
    base_segs, base_meta = read_segments(outdir / f"{args.backbone}.json")
    duration = base_meta.get("duration") or max(s["end"] for s in base_segs)
    lo = parse_ts(args.start)
    hi = min(parse_ts(args.end), duration) if args.end else duration

    # one output per transcription model; prefer the one with the backbone's speaker turns
    outputs = {}
    for p in sorted(outdir.glob("*.json"), key=lambda p: (p.stem != args.backbone,
                                                          not p.stem.startswith(base_meta.get("diar_name", "")))):
        if p.stem.endswith("_diar_only"):
            continue
        segs, meta = read_segments(p)
        model = meta.get("asr_model", p.stem)
        if model not in outputs and any(s["text"] for s in segs):
            outputs[model] = (p.stem, segs)
    base_model = base_meta.get("asr_model", args.backbone)
    systems = []
    for model, (stem, segs) in outputs.items():
        sid, label = LABELS.get(model, (stem[:6], stem))
        cov = max(s["end"] for s in segs)
        systems.append({"id": sid, "label": label, "file": stem, "segs": segs,
                        "covers_until": round(cov, 2), "base": model == base_model})
    systems.sort(key=lambda s: not s["base"])
    base = systems[0]
    print("models:", ", ".join(f"{s['label']} ({s['file']})" for s in systems))

    # backbone words and lines (one line per backbone segment whose middle is in range)
    lines, btok = [], []
    for s in base_segs:
        if not lo <= (s["start"] + s["end"]) / 2 <= hi:
            continue
        toks = tokens([s], -1, 1e9)
        lines.append({"start": round(s["start"], 2), "end": round(s["end"], 2),
                      "speaker": s["speaker"], "slots": []})
        btok += [(t, len(lines) - 1) for t in toks]
    if not lines:
        raise SystemExit("no backbone lines in that time range")
    lo, hi = min(lo, lines[0]["start"]), max(hi, lines[-1]["end"])
    ref_keys = [t[1] for t, _ in btok]
    m = len(ref_keys)

    # align every other model against the backbone
    aligned = {}
    for sysd in systems[1:]:
        toks = tokens(sysd["segs"], lo, hi)
        k = sum(1 for (t, _) in btok if t[2] <= sysd["covers_until"])  # backbone words it covers
        print(f"aligning {sysd['label']}: {len(toks)} words against {k} ...", flush=True)
        match, ins = align([t[1] for t in toks], ref_keys[:k])
        aligned[sysd["id"]] = (toks, match + [None] * (m - k), ins, k)

    def covers(sid, j):
        return sid == base["id"] or j < aligned[sid][3]

    def add_slot(line, opts, t, is_ins, base_choice):
        """opts: list of (surface, key, sid). Collapse to one slot."""
        groups = {}
        for surf, k, sid in opts:
            g = groups.setdefault(k, {"w": surf, "sys": []})
            g["sys"].append(sid)
        ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]["sys"]), kv[0] != base_choice))
        if len(ordered) == 1:
            if ordered[0][0]:  # everyone agrees on a word
                line["slots"].append({"w": ordered[0][1]["w"], "t": t})
            return
        slot = {"o": [[g["w"], g["sys"]] for _, g in ordered], "c": 0, "t": t}
        if is_ins:
            slot["ins"] = 1
        line["slots"].append(slot)

    sids = [s["id"] for s in systems]
    for g in range(m + 1):
        # insertions in the gap before backbone word g
        if any(g in aligned[s][2] for s in aligned):
            jj = g if g < m else m - 1
            line = lines[btok[max(0, g - 1)][1]]
            opts = []
            for sid in sids:
                if sid == base["id"] or not covers(sid, jj):
                    if sid == base["id"]:
                        opts.append(("", "", sid))
                    continue
                toks, _, ins, _ = aligned[sid]
                idx = ins.get(g, [])
                opts.append((" ".join(toks[i][0] for i in idx), "".join(toks[i][1] for i in idx), sid))
            t = btok[g][0][2] if g < m else btok[-1][0][2]
            add_slot(line, opts, t, True, "")
        if g == m:
            break
        (surf, k, t, _), li = btok[g]
        opts = [(surf, k, base["id"])]
        for sid in sids[1:]:
            if not covers(sid, g):
                continue
            toks, match, _, _ = aligned[sid]
            i = match[g]
            opts.append(("", "", sid) if i is None else (toks[i][0], toks[i][1], sid))
        add_slot(lines[li], opts, t, False, k)

    # speaker cross-check: other diarizers' opinion per line
    others = []
    for f, label in (("diarizen_diar_only", "DiariZen"), ("sortformer_diar_only", "Sortformer")):
        if (outdir / f"{f}.json").exists() and not args.backbone.startswith(f.split("_")[0]):
            segs, _ = read_segments(outdir / f"{f}.json")
            others.append((label, segs))
    vv = next((s for s in systems if s["id"] == "V"), None)
    if vv:
        others.append(("VibeVoice", vv["segs"]))
    for label, segs in others:
        mp = speaker_map(segs, base_segs, lo, hi)
        for ln in lines:
            d = dominant(segs, ln["start"], ln["end"])
            if d is not None and mp.get(d, d) != ln["speaker"]:
                ln.setdefault("spk_alt", []).append([label, mp.get(d, d)])

    # audio clip for the range
    out = outdir / "review"
    out.mkdir(exist_ok=True)
    print("cutting audio ...")
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", str(lo), "-t", str(hi - lo),
                    "-i", str(audio), "-ac", "1", "-b:a", "64k", str(out / "audio.mp3")], check=True)

    data = {"name": name, "offset": lo, "end": hi, "audio": "audio.mp3",
            "systems": [{k: s[k] for k in ("id", "label", "file", "covers_until")} for s in systems],
            "speakers": sorted({ln["speaker"] for ln in lines}), "lines": lines}
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    data_id = hashlib.sha1(blob.encode()).hexdigest()[:10]
    html = (HERE / "review_template.html").read_text(encoding="utf-8")
    html = html.replace("__DATA__", blob).replace("__DATA_ID__", data_id)
    (out / "index.html").write_text(html, encoding="utf-8")

    n_dis = sum(1 for ln in lines for s in ln["slots"] if "o" in s)
    n_words = sum(1 for ln in lines for s in ln["slots"] if "w" in s or "o" in s)
    print(f"{len(lines)} lines, {n_dis} disagreements in ~{n_words} words, "
          f"{fmt_ts(lo)}-{fmt_ts(hi)}\nopen {out / 'index.html'}")


if __name__ == "__main__":
    main()
