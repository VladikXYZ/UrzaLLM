"""VibeVoice-ASR (Microsoft): one model does transcription + speakers + timestamps
over the whole clip in a single pass.

  python run_vibevoice.py audio/X.wav                 # 4-bit, fits a 12 GB GPU
  python run_vibevoice.py audio/X.wav --quant offload # full precision, spills to CPU RAM (slow)
  python run_vibevoice.py audio/X.wav --prompt "Urza, anarchokapitalismus, agorismus"

The model has ~8B parameters (~16 GB in bf16), so on a 12 GB card it runs either
4-bit quantized (default) or partly on the CPU. Czech is not explicitly listed among
its ~50 languages, so this run is also a check of whether it handles Czech at all.

Output: out/<clip>/vibevoice-<quant>.json/.txt (+ raw model output in .raw.txt)
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch

from common import SR, load_audio, write_outputs

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


def make_inputs(processor, wav, prompt):
    kw = {"prompt": prompt} if prompt else {}
    try:
        return processor.apply_transcription_request(audio=str(wav), **kw)
    except Exception as e:  # some versions only take arrays: resample ourselves
        print(f"  path input failed ({e}); passing the waveform instead")
        sr = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 24000)
        audio = torch.from_numpy(load_audio(wav))
        if sr != SR:
            import torchaudio
            audio = torchaudio.functional.resample(audio, SR, sr)
        return processor.apply_transcription_request(audio=audio.numpy(), **kw)


def parse(processor, ids):
    parsed = processor.decode(ids, return_format="parsed")[0]
    raw = processor.decode(ids)[0]
    if isinstance(parsed, str):  # parser failed: try to pull the JSON list out ourselves
        m = re.search(r"\[.*\]", parsed, re.S)
        parsed = json.loads(m.group(0)) if m else []
    segs = []
    for p in parsed:
        text = str(p.get("Content", "")).strip()
        if text:
            segs.append({"start": float(p.get("Start", 0)), "end": float(p.get("End", 0)),
                         "speaker": f"S{p.get('Speaker', '?')}", "text": text})
    return segs, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--quant", default="4bit", choices=["4bit", "8bit", "offload"])
    ap.add_argument("--prompt", help="context: names and terms that appear in the lecture")
    ap.add_argument("--max-new-tokens", type=int, default=16384)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    wav = Path(args.wav)
    out_dir = Path(args.out) / wav.stem
    duration = len(load_audio(wav)) / SR
    print(f"{wav.name}: {duration / 60:.1f} min, GPU: {torch.cuda.get_device_name(0)}")

    print(f"loading {MODEL} ({args.quant})")
    processor, model = load_model(args.quant)
    inputs = make_inputs(processor, wav, args.prompt).to(model.device, model.dtype)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    torch.cuda.synchronize()
    secs = time.perf_counter() - t0

    segments, raw = parse(processor, out[:, inputs["input_ids"].shape[1]:])
    name = f"vibevoice-{args.quant}" + ("_prompt" if args.prompt else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.raw.txt").write_text(raw, encoding="utf-8")
    if not segments:
        print(f"  could not parse any segments; see {out_dir / name}.raw.txt")
    print(f"  {len(segments)} segments, {len({s['speaker'] for s in segments})} speakers, "
          f"{secs:.1f} s = {duration / secs:.1f}x realtime")
    write_outputs(out_dir, name, segments, {
        "audio": str(wav), "duration": duration, "asr_model": MODEL, "quant": args.quant,
        "diar_name": "vibevoice", "asr_seconds": secs, "diar_seconds": 0.0,
        "gpu": torch.cuda.get_device_name(0)})


if __name__ == "__main__":
    main()
