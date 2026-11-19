"""Score diarization systems against the hand-made references.

DER / JER (pyannote.metrics, 0.25 s collar, overlap scored) and the number that matters for us:
how well "Urza speaks here" comes out, measured on 10 ms frames, in two ways:
  oracle  - each system speaker mapped to the reference speaker it overlaps most (Hungarian, as in DER)
  auto    - stage 3 for real: system speakers whose voice (ECAPA centroid) is close to Urza's voice from a
            different video (solo livestream pBuDpAGGgnA) are Urza; the closest one always, others if cos >= TAU
Usage: python score.py reference hyp clips [systems...]   -> prints the table, writes hyp/scores.json
"""
import sys, json, pathlib, collections
import numpy as np, soundfile as sf
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

REF, HYP, CLIPS = map(pathlib.Path, sys.argv[1:4])
SYSTEMS = sys.argv[4:] or sorted(p.name for p in HYP.iterdir() if p.is_dir())
FR, COLLAR, TAU = 0.01, 0.25, 0.45


def read_rttm(p):
    a = Annotation()
    for ln in open(p):
        f = ln.split()
        if f and f[0] == "SPEAKER":
            s, d = float(f[3]), float(f[4])
            if d > 0:
                a[Segment(s, s + d), len(a)] = f[7]
    return a


def read_uem(p, dur):
    if not p.exists():
        return Timeline([Segment(0, dur)])
    return Timeline([Segment(float(l.split()[2]), float(l.split()[3])) for l in open(p) if l.strip()])


def frames(ann_or_tl, n, labels=None):
    m = np.zeros(n, bool)
    it = ann_or_tl.itertracks(yield_label=True) if isinstance(ann_or_tl, Annotation) else ((s, None, None) for s in ann_or_tl)
    for seg, _, lab in it:
        if labels is None or lab in labels:
            m[int(seg.start / FR):int(seg.end / FR)] = True
    return m


_enc = None
def embed(y, sr, segs):
    """ECAPA centroid of the given (start, end) pieces, 3 s windows."""
    global _enc
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    if _enc is None:
        _enc = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cuda:0"})
    chunks = []
    for s, e in segs:
        for t in np.arange(s, e - 1.0, 3.0):
            c = y[int(t * sr):int(min(t + 3.0, e) * sr)]
            if len(c) >= sr:
                chunks.append(c)
    if not chunks:
        return None
    vs = []
    with torch.inference_mode():
        for c in chunks:
            vs.append(_enc.encode_batch(torch.from_numpy(c)[None].cuda()).squeeze().cpu().numpy())
    v = np.mean([x / np.linalg.norm(x) for x in vs], axis=0)
    return v / np.linalg.norm(v)


def urza_voice():
    cache = HYP / "urza_voice.npy"
    if cache.exists():
        return np.load(cache)
    y, sr = sf.read(CLIPS / "enroll_solo_pBuDpAGGgnA.wav", dtype="float32")
    v = embed(y, sr, [(30, len(y) / sr - 30)])
    np.save(cache, v)
    return v


def prf(ref, hyp, mask):
    r, h = ref & mask, hyp & mask
    tp = (r & h).sum()
    p = tp / max(h.sum(), 1); rc = tp / max(r.sum(), 1)
    return p, rc, 2 * p * rc / max(p + rc, 1e-9)


def main():
    U = urza_voice()
    refs = sorted(REF.glob("*.rttm"))
    rows, per_clip = collections.defaultdict(list), {}
    for rp in refs:
        clip = rp.stem
        y, sr = sf.read(CLIPS / f"{clip}.wav", dtype="float32")
        dur = len(y) / sr; n = int(dur / FR) + 1
        ref = read_rttm(rp); uem = read_uem(REF / f"{clip}.uem", dur)
        ref_urza = frames(ref, n, {"Urza"})
        mask = frames(uem, n)
        # collar around Urza's reference boundaries is not scored
        for seg, _, lab in ref.itertracks(yield_label=True):
            if lab == "Urza":
                for b in (seg.start, seg.end):
                    mask[max(0, int((b - COLLAR) / FR)):int((b + COLLAR) / FR)] = False
        for sysname in SYSTEMS:
            hp = HYP / sysname / f"{clip}.rttm"
            if not hp.exists():
                continue
            hyp = read_rttm(hp)
            der = DiarizationErrorRate(collar=2 * COLLAR, skip_overlap=False)
            d = der(ref, hyp, uem=uem, detailed=True)
            jer = JaccardErrorRate(collar=2 * COLLAR)(ref, hyp, uem=uem)
            mapping = der.optimal_mapping(ref, hyp)
            oracle = frames(hyp, n, {h for h, r in mapping.items() if r == "Urza"})
            # auto identification
            sims = {}
            for lab in hyp.labels():
                tl = hyp.label_timeline(lab).support()
                segs = [(s.start, s.end) for s in tl if s.duration >= 1.0]
                v = embed(y, sr, segs) if sum(e - s for s, e in segs) >= 3 else None
                if v is not None:
                    sims[lab] = float(v @ U)
            best = max(sims, key=sims.get) if sims else None
            auto_labels = {l for l, s in sims.items() if s >= TAU or l == best}
            auto = frames(hyp, n, auto_labels)
            po, ro, fo = prf(ref_urza, oracle, mask)
            pa, ra, fa = prf(ref_urza, auto, mask)
            row = dict(clip=clip, der=d["diarization error rate"], miss=d["missed detection"] / d["total"],
                       fa=d["false alarm"] / d["total"], conf=d["confusion"] / d["total"], jer=jer,
                       n_ref=len(ref.labels()), n_hyp=len(hyp.labels()),
                       urza_oracle_f1=fo, urza_auto_p=pa, urza_auto_r=ra, urza_auto_f1=fa,
                       sims={k: round(v, 2) for k, v in sorted(sims.items(), key=lambda x: -x[1])})
            rows[sysname].append(row)
    out = {}
    for s, rs in rows.items():
        tot = {k: float(np.mean([r[k] for r in rs])) for k in ["der", "miss", "fa", "conf", "jer", "urza_oracle_f1", "urza_auto_p", "urza_auto_r", "urza_auto_f1"]}
        tf = HYP / s / "times.json"
        tot["sec_per_10min"] = float(np.mean(list(json.load(open(tf)).values()))) if tf.exists() else float("nan")
        out[s] = {"mean": tot, "clips": rs}
    json.dump(out, open(HYP / "scores.json", "w"), indent=1)
    hdr = f"{'system':38s} {'DER':>6s} {'miss':>5s} {'FA':>5s} {'conf':>5s} {'JER':>6s} {'U-orF1':>6s} {'U-P':>5s} {'U-R':>5s} {'U-F1':>5s} {'s/10min':>7s}"
    print(hdr)
    for s, v in sorted(out.items(), key=lambda x: x[1]["mean"]["der"]):
        m = v["mean"]
        print(f"{s:38s} {m['der']*100:6.1f} {m['miss']*100:5.1f} {m['fa']*100:5.1f} {m['conf']*100:5.1f} {m['jer']*100:6.1f} "
              f"{m['urza_oracle_f1']*100:6.1f} {m['urza_auto_p']*100:5.1f} {m['urza_auto_r']*100:5.1f} {m['urza_auto_f1']*100:5.1f} {m['sec_per_10min']:7.1f}")
    print()
    for s, v in sorted(out.items()):
        for r in v["clips"]:
            print(f"{s:38s} {r['clip']:32s} DER {r['der']*100:5.1f} conf {r['conf']*100:5.1f} spk {r['n_ref']}/{r['n_hyp']} "
                  f"U-F1 oracle {r['urza_oracle_f1']*100:5.1f} auto {r['urza_auto_f1']*100:5.1f}  {r['sims']}")


if __name__ == "__main__":
    main()
