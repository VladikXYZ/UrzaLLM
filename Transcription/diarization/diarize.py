"""Run one diarization system over every clip -> hyp/<system>/<clip>.rttm (+ times.json, seconds per clip).

    python diarize.py --system sortformer-v2.1 --clips clips --out hyp

Systems (each needs its own environment, see setup_env.pbs):
  sortformer-v2.1, sortformer-v2   NVIDIA streaming Sortformer, max 4 speakers         (venv with nemo_toolkit[asr])
  sortformer-v1                    NVIDIA offline Sortformer, max 4 speakers, trained on <= 90 s
  diarizen-large, diarizen-large-v2  BUT DiariZen (WavLM + pyannote clustering)         (venv_dz)
  pyannote-community-1, pyannote-3.1  pyannote.audio pipelines, gated: needs an HF token (venv_pya)
  ecapa-ahc                        baseline: energy VAD + ECAPA windows + agglomerative clustering (venv_pya)
"""
import argparse, json, pathlib, sys, time
import numpy as np, soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--system", required=True)
ap.add_argument("--clips", default="clips")
ap.add_argument("--out", default="hyp")
ap.add_argument("--only", nargs="*", help="clip names to run (default: all except enroll_*)")
a = ap.parse_args()

import torch
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clips = [p for p in sorted(pathlib.Path(a.clips).glob("*.wav"))
         if not p.name.startswith("enroll") and (not a.only or p.stem in a.only)]
out = pathlib.Path(a.out) / a.system
out.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- one loader per family, each returns f(wav) -> [(start, end, label)]
def load_sortformer(name):
    from nemo.collections.asr.models import SortformerEncLabelModel
    repo = {"sortformer-v2.1": "nvidia/diar_streaming_sortformer_4spk-v2.1",
            "sortformer-v2": "nvidia/diar_streaming_sortformer_4spk-v2",
            "sortformer-v1": "nvidia/diar_sortformer_4spk-v1"}[name]
    m = SortformerEncLabelModel.from_pretrained(repo, map_location=dev).eval()
    if "streaming" in repo:
        # "high latency" setting from the model card: the most accurate one, latency does not matter offline
        sm = m.sortformer_modules
        sm.chunk_len, sm.chunk_right_context, sm.fifo_len, sm.spkcache_update_period, sm.spkcache_len = 340, 40, 40, 300, 188
        sm._check_streaming_parameters()

    def run(wav):
        with torch.inference_mode():
            segs = m.diarize(audio=[str(wav)], batch_size=1)[0]
        return [(float(s.split()[0]), float(s.split()[1]), s.split()[2]) for s in segs]
    return run


def load_diarizen(name):
    sys.modules["spacy"] = None  # pyannote.database imports spacy if present; a broken one would crash it
    from diarizen.pipelines.inference import DiariZenPipeline
    from pyannote.audio.core.task import Specifications, Problem, Resolution
    # torch >= 2.6 loads checkpoints weights-only; allow the plain classes these checkpoints contain
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion, Specifications, Problem, Resolution])
    repo = {"diarizen-large": "BUT-FIT/diarizen-wavlm-large-s80-md",
            "diarizen-large-v2": "BUT-FIT/diarizen-wavlm-large-s80-md-v2"}[name]
    p = DiariZenPipeline.from_pretrained(repo)

    def run(wav):
        res = p(str(wav), sess_name=wav.stem)
        return [(t.start, t.end, f"spk{s}") for t, _, s in res.itertracks(yield_label=True)]
    return run


def load_pyannote(name):
    from pyannote.audio import Pipeline
    repo = {"pyannote-community-1": "pyannote/speaker-diarization-community-1",
            "pyannote-3.1": "pyannote/speaker-diarization-3.1"}[name]
    p = Pipeline.from_pretrained(repo)  # token: `hf auth login` once, or HF_TOKEN in the environment
    if p is None:
        sys.exit(f"could not load {repo}: accept its conditions on huggingface.co and log in (see README)")
    p.to(dev)

    def run(wav):
        y, sr = sf.read(wav, dtype="float32")
        res = p({"waveform": torch.from_numpy(y)[None], "sample_rate": sr})
        ann = getattr(res, "speaker_diarization", res)  # pyannote 4 returns an object, 3.x an Annotation
        return [(t.start, t.end, s) for t, _, s in ann.itertracks(yield_label=True)]
    return run


def load_ecapa(name, threshold=0.7, win=3.0, hop=1.5):
    from speechbrain.inference.speaker import EncoderClassifier
    from sklearn.cluster import AgglomerativeClustering
    enc = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cuda:0" if dev.type == "cuda" else "cpu"})

    def run(wav):
        y, sr = sf.read(wav, dtype="float32")
        n = int(0.03 * sr)
        db = 20 * np.log10(np.sqrt(np.mean(y[:len(y) // n * n].reshape(-1, n) ** 2, axis=1)) + 1e-6)
        speech = db > np.percentile(db, 5) + 15           # 15 dB over the quietest 5 % (noise floor)
        starts = [s for s in np.arange(0, len(y) / sr - win, hop)
                  if speech[int(s / 0.03):int((s + win) / 0.03)].mean() > 0.5]
        x = torch.stack([torch.from_numpy(y[int(s * sr):int(s * sr) + int(win * sr)]) for s in starts])
        with torch.inference_mode():
            e = torch.cat([enc.encode_batch(x[i:i + 256].to(dev)).squeeze(1).cpu() for i in range(0, len(x), 256)]).numpy()
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        lab = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                      distance_threshold=threshold).fit_predict(e)
        # tiny clusters (< 3 % of the windows: noise, laughter, music) join the nearest big one
        big = [l for l in set(lab) if (lab == l).mean() >= 0.03] or [np.bincount(lab).argmax()]
        cent = np.stack([e[lab == l].mean(0) for l in big])
        lab = np.array([l if l in big else big[int(np.argmax(cent @ v))] for l, v in zip(lab, e)])
        c = (win - hop) / 2
        return [(s + c, s + c + hop, f"spk{l}") for s, l in zip(starts, lab)]
    return run


LOADERS = {"sortformer": load_sortformer, "diarizen": load_diarizen, "pyannote": load_pyannote, "ecapa": load_ecapa}
run = LOADERS[a.system.split("-")[0]](a.system)

times = json.load(open(out / "times.json")) if (out / "times.json").exists() else {}
for wav in clips:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    segs = run(wav)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    times[wav.stem] = round(time.time() - t0, 2)
    with open(out / f"{wav.stem}.rttm", "w") as f:
        for s, e, lab in segs:
            if e > s:
                f.write(f"SPEAKER {wav.stem} 1 {s:.3f} {e - s:.3f} <NA> <NA> {lab} <NA> <NA>\n")
    print(f"{a.system} {wav.stem}: {len({l for *_, l in segs})} speakers, {times[wav.stem]} s", flush=True)
json.dump(times, open(out / "times.json", "w"), indent=1)
