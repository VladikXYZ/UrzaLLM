# ASRBench: transcription + speaker labels, compared on one video

Runs the five best free approaches on one 5–10 minute lecture video on a single
consumer GPU (target: RTX 3060 12 GB). You correct one output by hand, and every
approach is scored against it.

## The five approaches

| # | output name | speaker detection | transcription |
|---|---|---|---|
| 1 | `sortformer_canary_turns` | NVIDIA Streaming Sortformer v2 | NVIDIA Canary-1b-v2, audio cut per speaker turn |
| 2 | `diarizen_canary_turns` | DiariZen large v2 (BUT Brno) | Canary-1b-v2, audio cut per speaker turn |
| 3 | `whisper-cs_diarizen` | DiariZen large v2 | Whisper large-v3 Czech fine-tune (`mikr/whisper-large-v3-czech-cv13`) |
| 4 | `whisper-large-v3_sortformer` | Sortformer (reused from #1) | Whisper large-v3 (baseline) |
| 5 | `vibevoice-4bit` | VibeVoice-ASR, one model for both | same model, 4-bit to fit 12 GB |

Notes:
- **VibeVoice** has ~8B parameters, so on 12 GB it runs 4-bit quantized. Czech is not
  explicitly in its language list, so this run also shows whether it handles Czech.
  If its output looks broken, try `run_vibevoice.py --quant offload` (full precision,
  part of the model in CPU RAM, slow; needs ~32 GB system RAM).
- **Licenses:** DiariZen weights are CC BY-NC 4.0 (non-commercial). Canary: CC-BY-4.0.
  Sortformer: NVIDIA open model license. Whisper, VibeVoice: MIT.
- **No Hugging Face account needed.** All models used here are public downloads.

## Requirements

Linux or WSL2 on Windows (NeMo and DiariZen don't work on native Windows), the NVIDIA driver
(on WSL2 the normal Windows driver is enough), ~40 GB of free disk and ~16 GB RAM.

## Run (one command)

Give it a lecture video that is already cut to the test section, ideally 5–10 minutes
**with audience questions** (otherwise speaker detection isn't really tested). The whole
file is used; nothing is cut.

```bash
bash ASRBench/run_all.sh lecture.mp4
```

Any format ffmpeg reads works (mp4, mkv, webm, mov, or audio like mp3/m4a). On WSL2, files on
the Windows disk are under `/mnt/c/...`, e.g. `/mnt/c/Users/NAME/Videos/lecture.mp4`.

That's it. The first time, it installs everything it needs:
- missing system tools (`ffmpeg`, `git`, `build-essential`, `curl`) through `apt`, which asks for the password
- `uv`, a Python environment manager, from astral.sh
- four separate Python environments in `envs/` (the models need conflicting package versions)
- the Czech Whisper fine-tune, converted into `models/`

The installation takes a while and downloads a lot. Later runs skip it.

Then it extracts the audio to `audio/<name>.wav` (16 kHz mono), runs all five approaches,
and writes `out/<name>/<approach>.json` and `.txt` plus logs in `out/<name>/logs/`,
where `<name>` is the video's file name without extension.
A failing approach doesn't stop the others; the summary at the end shows what worked.

If the NVIDIA driver is older than 570, put `TORCH_INDEX=https://download.pytorch.org/whl/cu124`
in front of the command the first time.

## Make the reference by hand, then run the same command again

After the first run there is `out/<name>/reference.txt.todo`, a copy of one approach's output.
Open it next to the audio (VLC or Audacity) and fix it line by line:

- correct every word exactly as spoken; decide once whether filler words ("ehm", "jako") stay in, then stick to it
- fix speaker labels; names are free (`URZA`, `Q1`, `Q2`); scoring matches them automatically
- fix the times at **speaker changes** roughly (±0.5 s is fine); split a line where a speaker change was missed
- write numbers the way you want them in the final transcripts (`1989` vs. `tisíc devět set osmdesát devět`);
  a different form counts as an error

Starting from one approach's output biases the reference toward it (mistakes you overlook
count in its favour), so listen carefully. If two approaches end up close, correct a
second video starting from a different approach.

When done, rename it to `reference.txt` and run the **same command** again. Finished
approaches are skipped, so it only prints the scores and saves them to `out/<name>/scores.md`
(use `FORCE=1 bash run_all.sh ...` to redo the approaches):

- **cpWER**: words + speakers together, the main number to pick by
- **WER**: words only
- **DER**: speaker timing only, approximate since reference times are hand-set
- **speed**: × realtime on this GPU (an A100 is several times faster)

## Running pieces by hand

| script | env | what it does |
|---|---|---|
| `to_wav.py` | whisper | extract 16 kHz mono audio from a video or audio file |
| `run_diarizen.py` | diarizen | DiariZen speaker turns → `diarizen_diar_only.json` |
| `run_nemo.py` | nemo | Sortformer (or `--diar-json`) + Canary / Parakeet, `--chunking turns|fixed` |
| `run_whisper.py` | whisper | faster-whisper + speaker turns from `--diar-json`, `--model`, `--prompt` |
| `run_vibevoice.py` | vibevoice | VibeVoice-ASR, `--quant 4bit|8bit|offload`, `--prompt` |
| `convert_whisper.py` | whisper | convert another Whisper fine-tune, e.g. `--model BuzzASR/czech` |
| `score.py` | any | score outputs against `reference.txt` |

Run them as e.g. `envs/nemo/bin/python run_nemo.py audio/NAME.wav --asr parakeet canary --chunking turns fixed`.
