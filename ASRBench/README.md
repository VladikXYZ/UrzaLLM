# ASRBench: transcription + speaker labels, compared on one video

Runs seven free transcription + speaker-detection approaches on one lecture video on a
single consumer GPU (tested on an RTX 3060 12 GB). You correct a 5–10 minute part of
one output by hand, and every approach is scored against it.

## The approaches

| # | output name | speaker detection | transcription |
|---|---|---|---|
| 1 | `sortformer_canary_turns` | NVIDIA Streaming Sortformer v2 | NVIDIA Canary-1b-v2, audio cut per speaker turn |
| 2 | `diarizen_canary_turns` | DiariZen large v2 (BUT Brno) | Canary-1b-v2, audio cut per speaker turn |
| 3 | `whisper-cs_diarizen` | DiariZen large v2 | Whisper large-v3 Czech fine-tune (`mikr/whisper-large-v3-czech-cv13`) |
| 4 | `whisper-large-v3_sortformer` | Sortformer (reused from #1) | Whisper large-v3 (baseline) |
| 5 | `vibevoice-4bit` | VibeVoice-ASR, one model for both | same model, 4-bit to fit 12 GB |
| 6 | `sortformer_qwen3asr_turns` | Sortformer (reused) | Qwen3-ASR-1.7B on CUDA, audio cut per speaker turn |
| 7 | `diarizen_qwen3asr_turns` | DiariZen (reused) | Qwen3-ASR-1.7B on CUDA, audio cut per speaker turn |

Notes:
- **VibeVoice** has ~8B parameters, so on 12 GB it runs 4-bit quantized. Czech is not
  explicitly in its language list, so this run also shows whether it handles Czech.
  Long audio is processed in 20-minute windows; its speaker numbers restart in each
  window (`W1S0`, `W2S0`, ...). If its output looks broken, try
  `run_vibevoice.py --quant offload` (full precision, part of the model in CPU RAM, slow).
- **Qwen3-ASR** is an LLM-based speech recognizer with official Czech support. It uses the
  base model; the Czech fine-tune (`inferRouter/qwen3-asr-cs-1.7b`) needs a Hugging Face
  login, so it isn't used. It gives no word timestamps for Czech, which the per-turn cutting doesn't need.
- **DiariZen** runs with batch size 8 (its default of 32 doesn't fit in 12 GB) and halves
  it again by itself if the GPU still runs out of memory.
- **Licenses:** DiariZen weights are CC BY-NC 4.0 (non-commercial). Canary: CC-BY-4.0.
  Sortformer: NVIDIA open model license. Qwen3-ASR: Apache-2.0. Whisper, VibeVoice: MIT.
- **No Hugging Face account needed.** All models used here are public downloads.

## Requirements

Linux or WSL2 on Windows (NeMo and DiariZen don't work on native Windows), the NVIDIA driver
(on WSL2 the normal Windows driver is enough), ~50 GB of free disk and ~16 GB RAM.

## Run (one command)

```bash
cd ASRBench
./run_all.sh lecture.mp4
```

(If it says "not an executable file", run `chmod +x run_all.sh` once, or use `bash run_all.sh`.)

Any format ffmpeg reads works (mp4, mkv, webm, mov, or audio like mp3/m4a). The whole
file is used. A full 2-hour lecture works; on an RTX 3060 it takes very roughly 1 hour
for all approaches after installation, most of it VibeVoice. A 10-minute video takes a few minutes.

The first time, it installs everything it needs (later runs skip this):
- missing system tools (`ffmpeg`, `git`, `build-essential`, `curl`) through `apt`, which asks for the password
- `uv`, a Python environment manager, from astral.sh
- five separate Python environments in `envs/` (the models need conflicting package versions);
  each shows a running timer, and the install output goes to `envs/<env>.setup.log`
- the Czech Whisper fine-tune, converted into `models/`

Then it extracts the audio to `audio/<name>.wav` (16 kHz mono) and runs the approaches.
The terminal shows one line per step with progress bars; the full output of each step is
in `out/<name>/logs/`. Results go to `out/<name>/<approach>.json` and `.txt`, where
`<name>` is the video's file name without extension. A failing step doesn't stop the
others; it prints its last log lines and the summary at the end shows what worked.

If the NVIDIA driver is older than 570, put `TORCH_INDEX=https://download.pytorch.org/whl/cu124`
in front of the command the first time.

## Make the reference by hand, then run the same command again

After the first run there is `out/<name>/reference.txt.todo`, a copy of one approach's output.

1. **Pick one continuous 5–10 minute part**, ideally with audience questions, and delete
   all lines before and after it. Only the time range left in the file is scored, so you
   never have to correct a whole 2-hour lecture.
2. **Correct that part** while listening to `audio/<name>.wav` (VLC or Audacity can jump to a time):
   - every word exactly as spoken. Decide once whether filler words ("ehm", "jako", "hele")
     and colloquial forms ("správný lidi" vs. "správné lidi") are written as spoken, then stick
     to it. The models differ here: Canary tidies speech into standard Czech, Whisper keeps it as spoken
   - speaker labels; names are free (`URZA`, `Q1`, `Q2`); scoring matches them automatically
   - times at **speaker changes**, roughly (±0.5 s is fine); split a line where a speaker change was missed
   - numbers the way you want them in the final transcripts (`100` vs. `sto`); a different form counts as an error
   - Slovak speakers: write what they said in Slovak (all models are told the audio is Czech, so they'll get these wrong)
3. **Rename it** to `reference.txt` and run the **same command** again. Finished approaches are
   skipped, so it only prints the scores and saves them to `out/<name>/scores.md`
   (`FORCE=1 ./run_all.sh ...` redoes the approaches).

Starting from one approach's output biases the reference toward it (mistakes you overlook
count in its favour), so listen carefully. If two approaches end up close, correct a
second part starting from a different approach's output.

The scores:
- **cpWER**: words + speakers together, the main number to pick by
- **WER**: words only
- **DER**: speaker timing only, approximate since reference times are hand-set
- **speed**: × realtime for the whole file on this GPU (an A100 is several times faster)

## Running pieces by hand

| script | env | what it does |
|---|---|---|
| `to_wav.py` | whisper | extract 16 kHz mono audio from a video or audio file |
| `run_diarizen.py` | diarizen | DiariZen speaker turns → `diarizen_diar_only.json`, `--batch-size` |
| `run_nemo.py` | nemo | Sortformer (or `--diar-json`) + Canary / Parakeet, `--chunking turns|fixed` |
| `run_whisper.py` | whisper | faster-whisper + speaker turns from `--diar-json`, `--model`, `--prompt` |
| `run_qwen.py` | qwen | Qwen3-ASR + speaker turns from one or more `--diar-json` files |
| `run_vibevoice.py` | vibevoice | VibeVoice-ASR, `--quant 4bit|8bit|offload`, `--window-min`, `--prompt` |
| `convert_whisper.py` | whisper | convert another Whisper fine-tune, e.g. `--model BuzzASR/czech` |
| `score.py` | any | score outputs against `reference.txt` |

Run them as e.g. `envs/nemo/bin/python run_nemo.py audio/NAME.wav --asr parakeet canary --chunking turns fixed`.
