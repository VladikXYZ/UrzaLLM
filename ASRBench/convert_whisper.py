"""Convert a Hugging Face Whisper fine-tune to the CTranslate2 format faster-whisper uses.

  python convert_whisper.py                                   # mikr/whisper-large-v3-czech-cv13
  python convert_whisper.py --model BuzzASR/czech --out models/whisper-cs-buzz-ct2

Files the fine-tune does not ship (tokenizer, 128-mel preprocessor config) are
taken from openai/whisper-large-v3, which every large-v3 fine-tune shares.
"""
import argparse
import shutil
from pathlib import Path

from ctranslate2.converters import TransformersConverter
from huggingface_hub import hf_hub_download, list_repo_files

BASE = "openai/whisper-large-v3"
EXTRA = ["tokenizer.json", "preprocessor_config.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mikr/whisper-large-v3-czech-cv13")
    ap.add_argument("--out", default="models/whisper-cs-ct2")
    args = ap.parse_args()

    out = Path(args.out)
    if (out / "model.bin").exists():
        print(f"{out} already exists, skipping")
        return
    present = set(list_repo_files(args.model))
    copy = [f for f in EXTRA if f in present]
    print(f"converting {args.model} -> {out}")
    TransformersConverter(args.model, copy_files=copy, load_as_float16=True).convert(
        str(out), quantization="float16", force=True)
    for f in EXTRA:
        if f not in copy:
            print(f"  {f} missing in {args.model}, using the one from {BASE}")
            shutil.copy(hf_hub_download(BASE, f), out / f)
    print("done")


if __name__ == "__main__":
    main()
