"""Cut each clip's words into small pieces for hand labelling: a new piece at every pause > 0.3 s, every
speaker-change point proposed by ANY system in hyp/ (so a piece is very likely one speaker), and every 12 words.
System labels are NOT shown, only where they think the speaker might change, so the speaker identity in the
reference is decided from content and voice pitch alone.
Usage: make_pieces.py WORDS CLIPS HYP OUT"""
import json, sys, pathlib
import numpy as np, soundfile as sf, librosa

W, C, H, O = map(pathlib.Path, sys.argv[1:5]); O.mkdir(exist_ok=True)

for wf in sorted(W.glob("*.json")):
    clip = wf.stem
    words = json.load(open(wf, encoding="utf-8"))
    changes = []
    for rt in H.glob(f"*/{clip}.rttm"):
        segs = sorted((float(l.split()[3]), float(l.split()[3]) + float(l.split()[4]), l.split()[7]) for l in open(rt))
        for (s1, e1, a), (s2, e2, b) in zip(segs, segs[1:]):
            if a != b:
                changes.append((e1 + s2) / 2 if s2 > e1 else s2)
    changes = np.array(sorted(changes))
    y, sr = sf.read(C / f"{clip}.wav", dtype="float32")
    f0, vf, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr, frame_length=1024, hop_length=320)
    t = np.arange(len(f0)) * 320 / sr
    pieces, cur = [], []
    for w in words:
        if cur:
            gap = w["s"] - cur[-1]["e"]
            mid = (cur[-1]["e"] + w["s"]) / 2
            near_change = len(changes) and np.min(np.abs(changes - mid)) < 0.4
            if gap > 0.3 or near_change or len(cur) >= 12:
                pieces.append(cur); cur = []
        cur.append(w)
    if cur:
        pieces.append(cur)
    lines, meta = [], []
    for i, p in enumerate(pieces):
        s, e = p[0]["s"], p[-1]["e"]
        m = (t >= s) & (t <= e) & vf
        pitch = int(np.nanmedian(f0[m])) if m.sum() > 5 else 0
        lines.append(f"{i:03d} {s:6.1f} {pitch:3d} | " + " ".join(w["w"] for w in p))
        meta.append({"i": i, "s": s, "e": e})
    (O / f"{clip}.txt").write_text("\n".join(lines), encoding="utf-8")
    json.dump(meta, open(O / f"{clip}.json", "w"))
    print(clip, len(pieces))
