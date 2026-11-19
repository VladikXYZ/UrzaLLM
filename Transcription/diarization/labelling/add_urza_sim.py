"""Add to every labelling piece its voice similarity to Urza (ECAPA cosine vs. his solo livestream, another video).
Pieces shorter than 1 s get no score. Usage: add_urza_sim.py PIECES CLIPS"""
import json, sys, pathlib
import numpy as np, soundfile as sf, torch
from speechbrain.inference.speaker import EncoderClassifier

P, C = map(pathlib.Path, sys.argv[1:3])
enc = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cuda"})


def emb(x):
    with torch.inference_mode():
        v = enc.encode_batch(torch.from_numpy(x)[None].cuda()).squeeze().cpu().numpy()
    return v / np.linalg.norm(v)


y, sr = sf.read(C / "enroll_solo_pBuDpAGGgnA.wav", dtype="float32")
U = np.mean([emb(y[int(t * sr):int((t + 3) * sr)]) for t in np.arange(30, len(y) / sr - 30, 3)], axis=0)
U /= np.linalg.norm(U)
for pj in sorted(P.glob("*.json")):
    meta = json.load(open(pj))
    y, sr = sf.read(C / f"{pj.stem}.wav", dtype="float32")
    for m in meta:
        m["u"] = round(float(emb(y[int(m["s"] * sr):int(m["e"] * sr)]) @ U), 2) if m["e"] - m["s"] >= 1.0 else None
    json.dump(meta, open(pj, "w"))
    lines = (P / f"{pj.stem}.txt").read_text(encoding="utf-8").splitlines()
    out = []
    for m, ln in zip(meta, lines):
        head, text = ln.split(" | ", 1)
        out.append(f"{head} {'' if m['u'] is None else format(m['u'], '.2f'):>5s} | {text}")
    (P / f"{pj.stem}.txt").write_text("\n".join(out), encoding="utf-8")
    us = [m["u"] for m in meta if m["u"] is not None]
    print(pj.stem, np.percentile(us, [10, 50, 90]).round(2))
