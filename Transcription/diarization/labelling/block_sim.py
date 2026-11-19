"""Labelling aid: are two blocks of pieces the same voice? ECAPA centroid cosine between blocks.
Usage: block_sim.py CLIP a-b c-d ... (piece ranges)"""
import json, sys, itertools, pathlib
import numpy as np, soundfile as sf, torch
from speechbrain.inference.speaker import EncoderClassifier

clip, ranges = sys.argv[1], sys.argv[2:]
meta = json.load(open(f"pieces/{clip}.json"))
y, sr = sf.read(f"clips/{clip}.wav", dtype="float32")
enc = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cuda:0"})
cent = {}
for r in ranges:
    a, b = map(int, r.split("-"))
    vs = []
    for m in meta[a:b + 1]:
        if m["e"] - m["s"] >= 1.0:
            with torch.inference_mode():
                v = enc.encode_batch(torch.from_numpy(y[int(m["s"] * sr):int(m["e"] * sr)])[None].cuda()).squeeze().cpu().numpy()
            vs.append(v / np.linalg.norm(v))
    c = np.mean(vs, 0); cent[r] = c / np.linalg.norm(c)
for x, z in itertools.combinations(ranges, 2):
    print(f"{x:>9s} vs {z:<9s} {cent[x] @ cent[z]:.2f}")
