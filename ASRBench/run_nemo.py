"""Streaming Sortformer diarization + Canary-1b-v2 / Parakeet-tdt-0.6b-v3 transcription.

  python run_nemo.py audio/X.wav                               # all 4 variants
  python run_nemo.py audio/X.wav --asr canary --chunking turns  # just one

Diarization runs once; every (asr, chunking) combination reuses it.
  python run_nemo.py audio/X.wav --diar-json out/X/diarizen_diar_only.json   # DiariZen instead

Outputs are named <diarizer>_<asr>_<chunking>.

Chunking modes:
  turns  cut the audio at speaker turns (long turns split at pauses), so each chunk
         has one speaker; the chunk's text gets that speaker's label
  fixed  cut the audio every <=30 s at pauses, ignoring speakers, then give each word
         the speaker active at its timestamp (the WhisperX-style approach)

Outputs: out/<clip>/sortformer_<asr>_<chunking>.json/.txt
"""
import argparse
import gc
import tempfile
import time
from pathlib import Path

import torch

from common import (SR, exclusive_turns, load_audio, load_diar_json, split_long,
                    words_to_segments, write_diar_only, write_outputs, write_wav)

ASR_MODELS = {
    "canary": "nvidia/canary-1b-v2",
    "parakeet": "nvidia/parakeet-tdt-0.6b-v3",
}

# Long-form, high-accuracy streaming setting (the "very high latency" setup from the
# Sortformer v2 model card). Larger chunks = better accuracy and throughput offline.
SORTFORMER_STREAMING = dict(chunk_len=340, chunk_right_context=40, fifo_len=40,
                            spkcache_update_period=300, spkcache_len=188)


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def diarize(wav, model_name):
    from nemo.collections.asr.models import SortformerEncLabelModel
    model = SortformerEncLabelModel.from_pretrained(model_name, map_location="cuda").eval()
    mods = model.sortformer_modules
    for k, v in SORTFORMER_STREAMING.items():
        if hasattr(mods, k):
            setattr(mods, k, v)
    if hasattr(mods, "_check_streaming_parameters"):
        mods._check_streaming_parameters()

    with torch.inference_mode():
        raw, secs = timed(lambda: model.diarize(audio=[str(wav)], batch_size=1))
    if isinstance(raw, tuple):  # (segments, tensors) in some versions
        raw = raw[0]
    diar = []
    for item in raw[0]:
        s, e, spk = item.split()[:3] if isinstance(item, str) else item[:3]
        diar.append({"start": float(s), "end": float(e), "speaker": str(spk)})
    del model
    free_gpu()
    return sorted(diar, key=lambda d: d["start"]), secs


def hyp_words(h, offset):
    """Word timestamps from a NeMo Hypothesis, shifted to clip time."""
    ts = getattr(h, "timestamp", None)
    if not isinstance(ts, dict):
        return []
    words = []
    for w in ts.get("word", []):
        if "start" not in w:
            continue
        txt = (w.get("word") or w.get("segment") or "").strip()
        if txt:
            words.append({"word": txt, "start": offset + float(w["start"]),
                          "end": offset + float(w["end"])})
    return words


def transcribe(model, asr, paths, lang, batch_size):
    kw = dict(batch_size=batch_size, timestamps=True)
    if asr == "canary":
        kw.update(source_lang=lang, target_lang=lang)  # parakeet detects language itself
    with torch.inference_mode():
        hyps = model.transcribe(paths, **kw)
    if isinstance(hyps, tuple):
        hyps = hyps[0]
    return hyps


def make_chunks(audio, duration, diar, mode, pad):
    """Return (start, end, speaker_or_None, chunk_offset, chunk_audio)."""
    if mode == "turns":
        spans = [(s, e, t["speaker"])
                 for t in exclusive_turns(diar, duration)
                 for s, e in split_long(audio, t["start"], min(t["end"], duration))]
    else:
        spans = [(s, e, None) for s, e in split_long(audio, 0.0, duration)]
    chunks = []
    for s, e, spk in spans:
        a, b = max(0.0, s - pad), min(duration, e + pad)
        chunks.append((s, e, spk, a, audio[int(a * SR):int(b * SR)]))
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--asr", nargs="+", default=list(ASR_MODELS), choices=list(ASR_MODELS))
    ap.add_argument("--chunking", nargs="+", default=["turns", "fixed"], choices=["turns", "fixed"])
    ap.add_argument("--lang", default="cs")
    ap.add_argument("--diar-model", default="nvidia/diar_streaming_sortformer_4spk-v2")
    ap.add_argument("--diar-json", help="reuse another diarization (e.g. out/X/diarizen_diar_only.json) instead of Sortformer")
    ap.add_argument("--pad", type=float, default=0.2, help="seconds of audio added around turn chunks")
    ap.add_argument("--batch-size", type=int, default=8, help="lower if the GPU runs out of memory")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    wav = Path(args.wav)
    out_dir = Path(args.out) / wav.stem
    audio = load_audio(wav)
    duration = len(audio) / SR
    print(f"{wav.name}: {duration / 60:.1f} min, GPU: {torch.cuda.get_device_name(0)}")

    if args.diar_json:
        diar, diar_secs, diar_name = load_diar_json(args.diar_json)
        print(f"using {diar_name} diarization from {args.diar_json}")
    else:
        print("diarizing with", args.diar_model)
        diar, diar_secs = diarize(wav, args.diar_model)
        diar_name = "sortformer"
        write_diar_only(out_dir, diar_name, diar, wav, duration, diar_secs)
    print(f"  {len(diar)} segments, {len({d['speaker'] for d in diar})} speakers, {diar_secs:.1f} s")

    from nemo.collections.asr.models import ASRModel
    for asr in args.asr:
        print("loading", ASR_MODELS[asr])
        model = ASRModel.from_pretrained(ASR_MODELS[asr], map_location="cuda").eval()
        for mode in args.chunking:
            chunks = make_chunks(audio, duration, diar, mode, args.pad)
            with tempfile.TemporaryDirectory() as tmp:
                paths = []
                for i, c in enumerate(chunks):
                    p = Path(tmp) / f"{i:05d}.wav"
                    write_wav(p, c[4])
                    paths.append(str(p))
                hyps, asr_secs = timed(lambda: transcribe(model, asr, paths, args.lang, args.batch_size))

            if mode == "turns":
                segments = [{"start": s, "end": e, "speaker": spk, "text": h.text.strip(),
                             "words": hyp_words(h, off)}
                            for (s, e, spk, off, _), h in zip(chunks, hyps)]
            else:
                words = []
                for (s, e, _, off, _), h in zip(chunks, hyps):
                    ws = hyp_words(h, off)
                    if not ws and h.text.strip():  # no timestamps: spread the chunk evenly
                        toks = h.text.split()
                        step = (e - s) / len(toks)
                        ws = [{"word": t, "start": s + i * step, "end": s + (i + 1) * step}
                              for i, t in enumerate(toks)]
                    words.extend(ws)
                segments = words_to_segments(words, diar)

            name = f"{diar_name}_{asr}_{mode}"
            total = diar_secs + asr_secs
            print(f"  {name}: {len(chunks)} chunks, ASR {asr_secs:.1f} s, "
                  f"total {total:.1f} s = {duration / total:.0f}x realtime")
            write_outputs(out_dir, name, segments, {
                "audio": str(wav), "duration": duration, "asr_model": ASR_MODELS[asr],
                "diar_model": args.diar_json or args.diar_model, "diar_name": diar_name,
                "chunking": mode,
                "diar_seconds": diar_secs, "asr_seconds": asr_secs,
                "gpu": torch.cuda.get_device_name(0), "diarization": diar,
            })
        del model
        free_gpu()


if __name__ == "__main__":
    main()
