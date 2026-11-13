"""VibeVoice-ASR (Microsoft): one model does transcription + speakers + timestamps.

  python run_vibevoice.py audio/X.wav                 # 4-bit, fits a 12 GB GPU
  python run_vibevoice.py audio/X.wav --quant offload # full precision, spills to CPU RAM (slow)
  python run_vibevoice.py audio/X.wav --prompt "Urza, anarchokapitalismus, agorismus"

The model has ~8B parameters (~16 GB in bf16), so on a 12 GB card it runs either
4-bit quantized (default) or partly on the CPU. Czech is not explicitly listed among
its ~50 languages, so this run is also a check of whether it handles Czech at all.

Long audio is processed in windows of --window-min minutes (cut at pauses); a window
that still runs out of GPU memory is split in half and retried. Speaker numbers are
only consistent inside one window, so with several windows speakers are named
W<window>S<speaker>.

Output: out/<name>/vibevoice-<quant>.json/.txt (+ raw model output in .raw.txt)
"""
import argparse
import gc
import json
import re
import time
from pathlib import Path

import torch

from common import SR, load_audio, progress, say, split_long, write_outputs

MODEL = "microsoft/VibeVoice-ASR-HF"
# Parts that turn audio into embeddings stay in bf16; only the language model is quantized.
SKIP_QUANT = ["acoustic_tokenizer", "semantic_tokenizer", "acoustic_connector",
              "semantic_connector", "multi_modal_projector", "audio_tower", "lm_head"]


def load_model(quant):
    from transformers import AutoProcessor, BitsAndBytesConfig, VibeVoiceAsrForConditionalGeneration
    processor = AutoProcessor.from_pretrained(MODEL)
    kw = {"dtype": torch.bfloat16, "device_map": "auto"}
    if quant == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, llm_int8_skip_modules=SKIP_QUANT)
    elif quant == "8bit":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=SKIP_QUANT)
    else:  # offload: keep bf16, put what does not fit on the GPU into CPU RAM
        free = torch.cuda.mem_get_info()[0] / 2**30
        kw["max_memory"] = {0: f"{max(1, int(free) - 2)}GiB", "cpu": "64GiB"}
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(MODEL, **kw).eval()
    return processor, model


class EndTimeStreamer:
    """Moves the progress bar while the model writes: reads the latest "End": <sec>."""

    def __init__(self, tokenizer, bar, offset):
        self.tok, self.bar, self.offset = tokenizer, bar, offset
        self.ids, self.prompt_seen = [], False

    def put(self, value):
        if not self.prompt_seen:  # the first call carries the prompt
            self.prompt_seen = True
            return
        self.ids.extend(value.flatten().tolist())
        if len(self.ids) % 32:
            return
        try:
            ends = re.findall(r'"End"\s*:\s*([\d.]+)', self.tok.decode(self.ids[-400:]))
            if ends:
                pos = self.offset + float(ends[-1])
                if pos > self.bar.n:
                    self.bar.n = min(round(pos), self.bar.total)
                    self.bar.refresh()
        except Exception:
            pass

    def end(self):
        pass


def parse(processor, ids):
    parsed = processor.decode(ids, return_format="parsed")[0]
    raw = processor.decode(ids)[0]
    if isinstance(parsed, str):  # parser failed: try to pull the JSON list out ourselves
        m = re.search(r"\[.*\]", parsed, re.S)
        try:
            parsed = json.loads(m.group(0)) if m else []
        except json.JSONDecodeError:
            parsed = []
    segs = []
    for p in parsed:
        text = str(p.get("Content", "")).strip()
        if text:
            segs.append({"start": float(p.get("Start", 0)), "end": float(p.get("End", 0)),
                         "speaker": str(p.get("Speaker", "?")), "text": text})
    return segs, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--quant", default="4bit", choices=["4bit", "8bit", "offload"])
    ap.add_argument("--prompt", help="context: names and terms that appear in the lecture")
    ap.add_argument("--window-min", type=float, default=20, help="minutes of audio per model call")
    ap.add_argument("--max-new-tokens", type=int, default=16384)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    wav = Path(args.wav)
    out_dir = Path(args.out) / wav.stem
    audio = load_audio(wav)
    duration = len(audio) / SR
    print(f"{wav.name}: {duration / 60:.1f} min, GPU: {torch.cuda.get_device_name(0)}")

    say(f"loading {MODEL} ({args.quant})")
    processor, model = load_model(args.quant)
    model_sr = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 24000)

    win = args.window_min * 60
    windows = split_long(audio, 0.0, duration, max_len=win, min_len=win * 0.8)
    bar = progress(duration, "VibeVoice transcription")
    results, raws = [], []

    def run_window(s, e):
        chunk = torch.from_numpy(audio[int(s * SR):int(e * SR)].copy())
        if model_sr != SR:
            import torchaudio
            chunk = torchaudio.functional.resample(chunk, SR, model_sr)
        kw = {"prompt": args.prompt} if args.prompt else {}
        inputs = processor.apply_transcription_request(audio=chunk.numpy(), **kw)
        inputs = inputs.to(model.device, model.dtype)
        streamer = EndTimeStreamer(processor.tokenizer, bar, s)
        try:
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, streamer=streamer)
        except torch.cuda.OutOfMemoryError:
            del inputs
            gc.collect()
            torch.cuda.empty_cache()
            if e - s < 120:
                raise
            mid = split_long(audio, s, e, max_len=(e - s) / 2 + 30, min_len=(e - s) / 2 - 30)[0][1]
            say(f"out of GPU memory on {(e - s) / 60:.0f} min of audio, splitting it in two")
            run_window(s, mid)
            run_window(mid, e)
            return
        segs, raw = parse(processor, out[:, inputs["input_ids"].shape[1]:])
        results.append((s, segs))
        raws.append(f"===== window {s:.1f}-{e:.1f} s\n{raw}")
        bar.n = min(round(e), bar.total)
        bar.refresh()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s, e in windows:
        run_window(s, e)
    torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    bar.close()

    many = len(results) > 1
    segments = [{"start": off + sg["start"], "end": off + sg["end"], "text": sg["text"],
                 "speaker": f"W{i + 1}S{sg['speaker']}" if many else f"S{sg['speaker']}"}
                for i, (off, segs) in enumerate(sorted(results, key=lambda r: r[0])) for sg in segs]

    name = f"vibevoice-{args.quant}" + ("_prompt" if args.prompt else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.raw.txt").write_text("\n\n".join(raws), encoding="utf-8")
    if not segments:
        say(f"could not parse any segments; see {out_dir / name}.raw.txt")
    say(f"{len(segments)} segments in {len(results)} window(s), {secs:.0f} s = "
        f"{duration / secs:.1f}x realtime")
    write_outputs(out_dir, name, segments, {
        "audio": str(wav), "duration": duration, "asr_model": MODEL, "quant": args.quant,
        "diar_name": "vibevoice", "asr_seconds": secs, "diar_seconds": 0.0,
        "windows": len(results), "gpu": torch.cuda.get_device_name(0)})


if __name__ == "__main__":
    main()
