# ASR benchmark: which model transcribes Urza's Czech best

Four 3-minute clips from different recording conditions, every model run on exactly the same audio
windows, scored against transcripts corrected by hand.

| clip | video | condition |
|---|---|---|
| `studio_01Xg98XMRXE_1800` | Studio Svobodného přístavu: Učíme (se) (ne)svobodně, from 30:00 | studio, several speakers |
| `debate_06zLLuAPI4k_5400` | Urza s Tomášem Guthem Jarkovským, from 1:30:00 | two-person debate |
| `talk_0ACKjGY6VDg_2700` | Gabriela a Urza, Liberec, from 45:00 | live talk in a room |
| `echo_0Nhnw5DxKaY_2400` | Gabriela a Urza, Polná (zvuk s ozvěnou), from 40:00 | echoey room |

Models: `whisper-large-v3`, `whisper-large-v3-turbo` (faster-whisper, beam 5), `whisper-large-v3-longform`
(faster-whisper's own batched long-form mode on the whole clip, as in the earlier pipeline),
`canary-1b-v2`, `parakeet-tdt-0.6b-v3` (NeMo), `seamless-m4t-v2-large` (transformers), and
`whisper-large-v3` through the e-INFRA API (`Transcription/transcribe_einfra.py`).

All models except long-form get the same windows of 10–29.5 s cut at pauses (the cutter from
`transcribe_einfra.py`), so differences come from the model, not from how the audio was cut.

## Files

- `bench_asr.py`: one model over the clips -> `hyp/<model>/<clip>.json` (window texts, joined text, GPU time)
- `setup_env.pbs`: CPU job, builds the Python environment (`venv/`) once
- `run_bench.pbs`: GPU job, runs every model, one process per model
- `make_drafts.py`: reference drafts to correct, one per clip
- `score.py`: WER / CER per model and clip, and speed
- `textnorm.py`: normalisation (lowercase, no punctuation, hyphens inside words dropped) and alignment

## Run (MetaCentrum, everything as jobs; the frontend only submits)

Layout in `~/urza_asr_bench`: `clips/` (16 kHz mono WAV), `code/` (a copy of `Transcription/`),
`uvpkg/bin/uv`, `logs/`.

```bash
cd ~/urza_asr_bench
S=$(qsub -o logs/ code/benchmark/setup_env.pbs)
qsub -o logs/ -W depend=afterok:$S code/benchmark/run_bench.pbs
# one model only: qsub -o logs/ -v MODELS=canary-1b-v2 code/benchmark/run_bench.pbs
```

## Reference transcripts

```bash
python make_drafts.py --hyp hyp/ --out reference/
```

Each `reference/<clip>.txt` lists the clip window by window: a YouTube link to that exact second, one
editable line (the model that agrees most with the others), and `#` lines with what the other models
heard, differing words marked «like this». Fix the editable lines to exactly what was said (rules at the
top of each file), then delete the `# DRAFT` first line. Then:

```bash
python score.py --hyp hyp/ --ref reference/
```
