"""Whisper (faster-whisper, batched, word timestamps) + speaker turns from another runner.

Each word gets the speaker active at its timestamp. The speaker turns come from a
*_diar_only.json written by run_nemo.py (Sortformer) or run_diarizen.py (DiariZen).

  python run_whisper.py audio/X.wav --diar-json out/X/sortformer_diar_only.json
  python run_whisper.py audio/X.wav --model models/whisper-cs-ct2 --label whisper-cs \
      --diar-json out/X/diarizen_diar_only.json        # Czech fine-tune + DiariZen

Output: out/<clip>/whisper-<model>_<diar>.json/.txt
"""
import argparse
import gc
import time
from pathlib import Path

import torch

from common import SR, load_audio, load_diar_json, progress, say, words_to_segments, write_outputs


def transcribe(audio, duration, model_name, lang, batch_size, beam_size, prompt, desc):
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    model = WhisperModel(model_name, device="cuda", compute_type="float16")
    pipe = BatchedInferencePipeline(model=model)
    t0 = time.perf_counter()
    segments, _ = pipe.transcribe(audio, language=lang, batch_size=batch_size,
                                  beam_size=beam_size, word_timestamps=True,
                                  initial_prompt=prompt)
    words = []
    bar = progress(duration, desc)
    for seg in segments:  # generator: decoding happens while iterating
        for w in seg.words or []:
            if w.word.strip():
                words.append({"word": w.word.strip(), "start": float(w.start), "end": float(w.end)})
        if seg.end > bar.n:
            bar.n = min(round(seg.end), bar.total)
            bar.refresh()
    bar.n = bar.total
    bar.close()
    secs = time.perf_counter() - t0
    del pipe, model
    gc.collect()
    return words, secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--model", default="large-v3",
                    help="faster-whisper model name or path to a CTranslate2 model (e.g. a Czech fine-tune)")
    ap.add_argument("--lang", default="cs")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--prompt", help="initial prompt: names and terms that appear in the lecture")
    ap.add_argument("--diar-json", required=True,
                    help="speaker turns to use: out/X/sortformer_diar_only.json or out/X/diarizen_diar_only.json")
    ap.add_argument("--label", help="name for the ASR part of the output file (default: whisper-<model>)")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    wav = Path(args.wav)
    out_dir = Path(args.out) / wav.stem
    diar, diar_secs, diar_name = load_diar_json(args.diar_json)
    audio = load_audio(wav)
    duration = len(audio) / SR
    print(f"{wav.name}: {duration / 60:.1f} min, GPU: {torch.cuda.get_device_name(0)}")

    label = args.label or f"whisper-{Path(args.model).name}"
    say(f"loading {args.model}; speaker turns from {diar_name}")
    words, asr_secs = transcribe(audio, duration, args.model, args.lang, args.batch_size,
                                 args.beam_size, args.prompt, f"{label} transcription")

    segments = words_to_segments(words, diar)
    name = f"{label}_{diar_name}" + ("_prompt" if args.prompt else "")
    total = asr_secs + diar_secs
    say(f"{name}: {len(words)} words, transcription {asr_secs:.0f} s, "
        f"total {total:.0f} s = {duration / total:.0f}x realtime")
    write_outputs(out_dir, name, segments, {
        "audio": str(wav), "duration": duration, "asr_model": args.model,
        "diar_model": args.diar_json, "diar_name": diar_name,
        "chunking": "fixed",
        "diar_seconds": diar_secs, "asr_seconds": asr_secs,
        "gpu": torch.cuda.get_device_name(0), "diarization": diar,
    })


if __name__ == "__main__":
    main()
