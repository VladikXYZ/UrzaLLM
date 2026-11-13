"""DiariZen diarization (BUT Brno, WavLM + pyannote). Writes only speaker turns;
run_nemo.py / run_whisper.py add the text with --diar-json.

  python run_diarizen.py audio/X.wav      # -> out/X/diarizen_diar_only.json

Needs its own environment (Python 3.10, torch 2.1.1); setup.sh builds it.
Model weights are CC BY-NC 4.0 (non-commercial).
"""
import argparse
import time
from pathlib import Path

import torch

from common import SR, load_audio, write_diar_only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--model", default="BUT-FIT/diarizen-wavlm-large-s80-md-v2")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    from diarizen.pipelines.inference import DiariZenPipeline

    wav = Path(args.wav)
    duration = len(load_audio(wav)) / SR
    print(f"{wav.name}: {duration / 60:.1f} min, GPU: {torch.cuda.get_device_name(0)}")
    print("diarizing with", args.model)
    pipe = DiariZenPipeline.from_pretrained(args.model)  # uses cuda:0 automatically

    t0 = time.perf_counter()
    ann = pipe(str(wav), sess_name=wav.stem)
    torch.cuda.synchronize()
    secs = time.perf_counter() - t0

    diar = sorted(({"start": float(t.start), "end": float(t.end), "speaker": f"S{spk}"}
                   for t, _, spk in ann.itertracks(yield_label=True)), key=lambda d: d["start"])
    print(f"  {len(diar)} segments, {len({d['speaker'] for d in diar})} speakers, {secs:.1f} s")
    write_diar_only(Path(args.out) / wav.stem, "diarizen", diar, wav, duration, secs)


if __name__ == "__main__":
    main()
