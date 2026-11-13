"""Qwen3-ASR (Alibaba, LLM-based speech recognition, Czech supported) on the GPU,
with speaker turns from another runner. The audio is cut per speaker turn (long
turns split at pauses, <=30 s), so every chunk already has its speaker.

  python run_qwen.py audio/X.wav --diar-json out/X/sortformer_diar_only.json out/X/diarizen_diar_only.json

The model is loaded once and run for each speaker-turn file given.
Output: out/<name>/<diarizer>_qwen3asr_turns.json/.txt

Uses the ungated base model Qwen/Qwen3-ASR-1.7B (Apache-2.0). The Czech fine-tune
inferRouter/qwen3-asr-cs-1.7b would need a Hugging Face login, so it is not used.
Qwen's timestamp aligner does not support Czech, so there are no word timestamps.
"""
import argparse
import gc
import time
from pathlib import Path

import torch

from common import (SR, batches, load_audio, load_diar_json, make_chunks, progress, say,
                    write_outputs)

MODEL = "Qwen/Qwen3-ASR-1.7B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--diar-json", nargs="+", required=True,
                    help="one or more out/X/<diarizer>_diar_only.json files")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--language", default="Czech")
    ap.add_argument("--batch-size", type=int, default=16, help="halved automatically on out-of-memory")
    ap.add_argument("--pad", type=float, default=0.2)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    from qwen_asr import Qwen3ASRModel

    wav = Path(args.wav)
    out_dir = Path(args.out) / wav.stem
    turn_files = [load_diar_json(p) for p in args.diar_json]  # fail early if one is missing
    audio = load_audio(wav)
    duration = len(audio) / SR
    print(f"{wav.name}: {duration / 60:.1f} min, GPU: {torch.cuda.get_device_name(0)}")

    say(f"loading {args.model} (CUDA, bf16)")
    model = Qwen3ASRModel.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0",
        max_inference_batch_size=args.batch_size, max_new_tokens=512)

    batch = args.batch_size
    for diar, diar_secs, diar_name in turn_files:
        chunks = make_chunks(audio, duration, diar, "turns", args.pad)
        texts = []
        bar = progress(sum(e - s for s, e, *_ in chunks), f"qwen3-asr transcription ({diar_name} turns)")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        todo = list(chunks)
        while todo:
            group = todo[:batch]
            try:
                res = model.transcribe(audio=[(c[4], SR) for c in group],
                                       language=[args.language] * len(group))
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                if batch == 1:
                    raise
                batch //= 2
                say(f"out of GPU memory, batch size now {batch}")
                continue
            texts.extend(r.text.strip() for r in res)
            todo = todo[len(group):]
            bar.update(sum(c[1] - c[0] for c in group))
        torch.cuda.synchronize()
        asr_secs = time.perf_counter() - t0
        bar.close()

        segments = [{"start": s, "end": e, "speaker": spk, "text": t}
                    for (s, e, spk, _, _), t in zip(chunks, texts)]
        name = f"{diar_name}_qwen3asr_turns"
        total = asr_secs + diar_secs
        say(f"{name}: {len(chunks)} chunks, transcription {asr_secs:.0f} s, "
            f"total {total:.0f} s = {duration / total:.0f}x realtime")
        write_outputs(out_dir, name, segments, {
            "audio": str(wav), "duration": duration, "asr_model": args.model,
            "diar_name": diar_name, "chunking": "turns",
            "diar_seconds": diar_secs, "asr_seconds": asr_secs,
            "gpu": torch.cuda.get_device_name(0), "diarization": diar,
        })


if __name__ == "__main__":
    main()
