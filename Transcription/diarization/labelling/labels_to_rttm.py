"""Hand labels (piece ranges -> speaker) to reference RTTM + UEM.

A piece is one or more Whisper words (times from the word timestamps). Neighbouring pieces of the same speaker
are joined when the pause between them is < 0.5 s. '?' pieces are cut out of the UEM (not scored), together
with the pauses next to them. Usage: labels_to_rttm.py LABELS PIECES CLIPS OUT"""
import json, sys, pathlib
import soundfile as sf

L, P, C, O = map(pathlib.Path, sys.argv[1:5]); O.mkdir(parents=True, exist_ok=True)
NAMES = {"U": "Urza"}

for lab in sorted(L.glob("*.lab")):
    clip = lab.stem
    meta = json.load(open(P / f"{clip}.json"))
    spk = [None] * len(meta)
    for ln in open(lab, encoding="utf-8"):
        ln = ln.split("#")[0].strip()
        if not ln:
            continue
        rng, who = ln.split()
        a, b = (rng.split("-") + [rng])[:2]
        for i in range(int(a), int(b) + 1):
            spk[i] = who
    missing = [i for i, s in enumerate(spk) if s is None]
    assert not missing, f"{clip}: unlabelled pieces {missing[:10]}"
    dur = sf.info(C / f"{clip}.wav").duration
    # turns
    turns = []
    for m, s in zip(meta, spk):
        if s == "?":
            continue
        name = NAMES.get(s, s)
        if turns and turns[-1][2] == name and m["s"] - turns[-1][1] < 0.5:
            turns[-1][1] = m["e"]
        else:
            turns.append([m["s"], m["e"], name])
    with open(O / f"{clip}.rttm", "w") as f:
        for s, e, n in turns:
            f.write(f"SPEAKER {clip} 1 {s:.3f} {e - s:.3f} <NA> <NA> {n} <NA> <NA>\n")
    # UEM: everything except '?' pieces, extended to the middle of the pause on each side
    holes = []
    for i, (m, s) in enumerate(zip(meta, spk)):
        if s != "?":
            continue
        lo = (meta[i - 1]["e"] + m["s"]) / 2 if i else 0.0
        hi = (m["e"] + meta[i + 1]["s"]) / 2 if i + 1 < len(meta) else dur
        holes.append([lo, hi])
    # long stretches without any transcribed word (> 2 s) are unknown too: silence, or speech Whisper missed
    edges = [0.0] + [x for m in meta for x in (m["s"], m["e"])] + [dur]
    for a, b in zip(edges[0::2], edges[1::2]):
        if b - a > 2.0:
            holes.append([a + 0.5, b - 0.5])
    holes.sort()
    merged = []
    for lo, hi in holes:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    holes = merged
    keep, t = [], 0.0
    for lo, hi in holes:
        if lo > t:
            keep.append((t, lo))
        t = hi
    if t < dur:
        keep.append((t, dur))
    with open(O / f"{clip}.uem", "w") as f:
        for s, e in keep:
            f.write(f"{clip} 1 {s:.3f} {e:.3f}\n")
    scored = sum(e - s for s, e in keep)
    urza = sum(e - s for s, e, n in turns if n == "Urza")
    other = sum(e - s for s, e, n in turns if n != "Urza")
    print(f"{clip}: scored {scored:.0f}/{dur:.0f} s, Urza {urza:.0f} s, others {other:.0f} s, speakers {sorted({n for *_, n in turns})}")
