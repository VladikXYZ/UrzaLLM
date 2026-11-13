"""DiariZen diarization (BUT Brno, WavLM + pyannote). Writes only speaker turns;
run_nemo.py / run_whisper.py / run_qwen.py add the text with --diar-json.

  python run_diarizen.py audio/X.wav      # -> out/X/diarizen_diar_only.json

Needs its own environment (Python 3.10, torch 2.1.1); setup.sh builds it.
Model weights are CC BY-NC 4.0 (non-commercial).

The model's default batch size (32) runs out of memory on a 12 GB GPU, so this uses 8
and halves it again whenever the GPU runs out of memory.
"""
import argparse
import functools
import gc
import time
from pathlib import Path

import torch

from common import SR, load_audio, progress, say, write_diar_only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--model", default="BUT-FIT/diarizen-wavlm-large-s80-md-v2")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    from diarizen.pipelines.inference import DiariZenPipeline

    wav = Path(args.wav)
    duration = len(load_audio(wav)) / SR
    print(f"{wav.name}: {duration / 60:.1f} min, GPU: {torch.cuda.get_device_name(0)}")
    say(f"loading {args.model}")
    pipe = DiariZenPipeline.from_pretrained(args.model)  # uses cuda:0 automatically

    # progress bar for the segmentation pass (the slow part); pyannote reports chunks done
    bar = None

    def hook(*_, completed=None, total=None, **__):
        nonlocal bar
        if total is None or completed is None:
            return
        if bar is None:
            bar = progress(total, "speaker segmentation", unit="chunks")
        bar.n = min(completed, total)
        bar.refresh()

    orig = pipe.get_segmentations
    pipe.get_segmentations = functools.partial(orig, hook=hook)

    batch = args.batch_size
    t0 = time.perf_counter()
    while True:
        pipe._segmentation.batch_size = batch
        if hasattr(pipe, "embedding_batch_size"):
            pipe.embedding_batch_size = min(pipe.embedding_batch_size, batch)
        try:
            ann = pipe(str(wav), sess_name=wav.stem)
            break
        except (MemoryError, torch.cuda.OutOfMemoryError) as e:
            if bar is not None:
                bar.close()
                bar = None
            if batch == 1:
                raise
            gc.collect()
            torch.cuda.empty_cache()
            batch //= 2
            say(f"out of GPU memory ({type(e).__name__}), retrying with batch size {batch}")
            t0 = time.perf_counter()
    torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    if bar is not None:
        bar.close()

    diar = sorted(({"start": float(t.start), "end": float(t.end), "speaker": f"S{spk}"}
                   for t, _, spk in ann.itertracks(yield_label=True)), key=lambda d: d["start"])
    say(f"{len(diar)} segments, {len({d['speaker'] for d in diar})} speakers, {secs:.0f} s "
        f"(batch size {batch})")
    write_diar_only(Path(args.out) / wav.stem, "diarizen", diar, wav, duration, secs)


if __name__ == "__main__":
    main()
