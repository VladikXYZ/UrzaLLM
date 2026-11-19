"""Run one ASR model over the benchmark clips, on the same pause-cut windows for every model.

    python bench_asr.py --model canary-1b-v2 --clips clips/ --out hyp/

Every clip is cut into windows of 10-29.5 s that end at a pause (the same cutter as
Transcription/transcribe_einfra.py), each window is transcribed on its own, and the texts are joined.
So all models see exactly the same audio pieces and differences come from the model alone.
`whisper-large-v3-longform` is the exception: faster-whisper's own batched long-form mode on the whole
clip, as used in the earlier pipeline, to see whether its own segmentation helps.

Writes <out>/<model>/<clip>.json: windows with their text, the joined text, and the GPU time taken.
"""
import argparse
import json
import os
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transcribe_einfra import SR, frame_db, make_windows  # noqa: E402

MODELS = ["whisper-large-v3", "whisper-large-v3-turbo", "whisper-large-v3-longform",
          "canary-1b-v2", "parakeet-tdt-0.6b-v3", "seamless-m4t-v2-large"]


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as f:
        assert f.getframerate() == SR and f.getnchannels() == 1 and f.getsampwidth() == 2, path
        return np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)


def write_wav(path: Path, pcm: np.ndarray):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(pcm.tobytes())


def windows_of(pcm: np.ndarray) -> list[dict]:
    return [w for w in make_windows(frame_db(pcm), len(pcm), 10.0, 29.5, -55.0) if not w["silent"]]


def cut(pcm, w):
    return pcm[round(w["start"] * SR):round(w["end"] * SR)]


# ---------------------------------------------------------------- one transcriber per model family
# Each returns a function: list of int16 arrays -> list of texts.

def faster_whisper(name: str):
    from faster_whisper import WhisperModel
    model = WhisperModel(name, device="cuda", compute_type="float16")

    def run(pieces):
        out = []
        for p in pieces:
            segs, _ = model.transcribe(p.astype(np.float32) / 32768.0, language="cs", beam_size=5,
                                       condition_on_previous_text=False, vad_filter=False)
            out.append(" ".join(s.text.strip() for s in segs))
        return out
    return run


def nemo(name: str, lang_args: dict):
    import torch
    from nemo.collections.asr.models import ASRModel
    model = ASRModel.from_pretrained(name, map_location="cuda").eval()
    tmp = Path(tempfile.mkdtemp(dir=os.environ.get("SCRATCHDIR")))

    def run(pieces):
        paths = []
        for i, p in enumerate(pieces):
            paths.append(str(tmp / f"{i:05d}.wav"))
            write_wav(Path(paths[-1]), p)
        with torch.inference_mode():
            hyps = model.transcribe(paths, batch_size=8, **lang_args)
        hyps = hyps[0] if isinstance(hyps, tuple) else hyps
        return [(h.text if hasattr(h, "text") else str(h)).strip() for h in hyps]
    return run


def seamless():
    import torch
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
    proc = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        "facebook/seamless-m4t-v2-large", torch_dtype=torch.float16).to("cuda").eval()

    def run(pieces):
        out = []
        for p in pieces:
            audio = p.astype(np.float32) / 32768.0
            try:
                inputs = proc(audio=audio, sampling_rate=SR, return_tensors="pt")
            except TypeError:
                inputs = proc(audios=audio, sampling_rate=SR, return_tensors="pt")
            inputs = {k: (v.to("cuda", torch.float16) if v.dtype.is_floating_point else v.to("cuda"))
                      for k, v in inputs.items()}
            with torch.inference_mode():
                tokens = model.generate(**inputs, tgt_lang="ces", num_beams=5, max_new_tokens=400)
            out.append(proc.decode(tokens[0], skip_special_tokens=True).strip())
        return out
    return run


def longform():
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    pipe = BatchedInferencePipeline(model=WhisperModel("large-v3", device="cuda", compute_type="float16"))

    def run_clip(pcm):
        segs, _ = pipe.transcribe(pcm.astype(np.float32) / 32768.0, language="cs", batch_size=16, beam_size=5)
        return [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segs]
    return run_clip


def build(model: str):
    if model == "whisper-large-v3":
        return faster_whisper("large-v3")
    if model == "whisper-large-v3-turbo":
        return faster_whisper("large-v3-turbo")
    if model == "canary-1b-v2":
        return nemo("nvidia/canary-1b-v2", {"source_lang": "cs", "target_lang": "cs"})
    if model == "parakeet-tdt-0.6b-v3":
        return nemo("nvidia/parakeet-tdt-0.6b-v3", {})  # detects the language itself
    if model == "seamless-m4t-v2-large":
        return seamless()
    raise ValueError(model)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    clips = sorted(args.clips.glob("*.wav"))
    out_dir = args.out / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    t_load = time.time()
    if args.model == "whisper-large-v3-longform":
        run_clip = longform()
    else:
        transcribe = build(args.model)
    load_s = time.time() - t_load
    print(f"{args.model}: loaded in {load_s:.0f}s", flush=True)

    for clip in clips:
        pcm = read_wav(clip)
        t0 = time.time()
        if args.model == "whisper-large-v3-longform":
            wins = run_clip(pcm)
        else:
            wins = windows_of(pcm)
            for w, text in zip(wins, transcribe([cut(pcm, w) for w in wins])):
                w["text"] = text
        took = time.time() - t0
        text = " ".join(w["text"] for w in wins if w["text"])
        (out_dir / f"{clip.stem}.json").write_text(json.dumps({
            "model": args.model, "clip": clip.stem, "audio_seconds": round(len(pcm) / SR, 1),
            "seconds": round(took, 1), "load_seconds": round(load_s, 1),
            "windows": wins, "text": text}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {clip.stem}: {len(pcm) / SR / took:.0f}x real time, {len(text.split())} words", flush=True)


if __name__ == "__main__":
    main()
