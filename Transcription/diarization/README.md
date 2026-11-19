# Diarization benchmark: which system finds when Urza speaks

Stage 2 + 3 of the pipeline: split each video into speakers, then tell which speaker is Urza. Seven
10-minute clips from real channel videos, labelled by hand, every system run on the same audio.

| clip | video | situation | speakers in the reference |
|---|---|---|---|
| `debate_06zLLuAPI4k_5400` | Urza s Tomášem Guthem Jarkovským, from 1:30:00 | fast two-man debate + moderator | Urza, Tomáš, moderator |
| `studio_01Xg98XMRXE_1800` | Studio: Učíme (se) (ne)svobodně, from 30:00 | studio, Urza + a student | Urza, student |
| `talk_0ACKjGY6VDg_2700` | Gabriela a Urza, Liberec, from 45:00 | room, woman + audience | Urza, Gabriela, audience man |
| `echo_0Nhnw5DxKaY_2400` | Gabriela a Urza, Polná, from 40:00 | echoey room | Urza, 2 women |
| `lectureqa_1PkAoDz1DHg_5400` | O sebeřízení anarchii v Bratislavě, from 1:30:00 | lecture Q&A, Slovak questions | Urza, 3 audience members |
| `fourppl_8ROi_ZmPt5M_1800` | Teri, Tim, Šimon a Urza, from 30:00 | online podcast, 4 people | Urza, Teri, Tim, Šimon |
| `podcast_emzWOz2r_Po_1200` | Reportér pozval Urzu do podcastu, from 20:00 | interview | Urza, interviewer |

Urza's reference voice for stage 3 comes from another video (`pBuDpAGGgnA`, a solo livestream), so
recognising him is tested the way it will run on new videos.

## Results so far (laptop RTX 4070, before the MetaCentrum run)

| system | DER % | miss | false alarm | confusion | Urza F1, oracle % | Urza F1, automatic % | s per 10 min |
|---|---|---|---|---|---|---|---|
| `sortformer-v2.1` (NVIDIA streaming, ≤ 4 speakers) | **11.7** | 5.2 | 3.1 | 3.4 | 95.3 | **95.3** | 5.7 |
| `ecapa-ahc` (baseline) | 17.2 | 8.6 | 2.4 | 6.2 | 89.0 | 88.9 | 7.0 |

- **Urza F1, automatic** is the number that matters: precision/recall of "Urza speaks here" on 10 ms frames,
  where Urza = the system speaker(s) whose voice is closest to his reference voice (ECAPA, cosine ≥ 0.45, the
  closest one always). **Oracle** maps system speakers to reference speakers as DER does, for comparison.
  For Sortformer the speaker picked automatically was the right one in every clip (oracle = automatic).
- Sortformer v2.1 beats the baseline on Urza F1 in 6 of 7 clips (podcast 99.3 %, Liberec 97.4 %; the studio
  clip is the exception, 92.1 vs 93.1) but on DER only in 4 of 7: on the debate, Bratislava and studio clips
  the baseline has fewer misses. Its big wins are the echoey room (12.7 vs 47.5 % DER) and the podcast
  (1.6 vs 10.8 %). Weakest clip: Liberec (18.5 % DER, 10.5 % confusion, 4 speakers found where the reference has 3).
- DiariZen-large ran on one clip only before the laptop GPU (8 GB) failed with a CUDA error; on the debate clip:
  19.4 % DER vs. 14.3 % for Sortformer v2.1. Not enough to judge, hence the MetaCentrum run.
- `results_local/` holds these outputs and `scores.txt` with the per-clip table.

## MetaCentrum results (all 8 systems)

| system | DER % | Urza F1, automatic % | s per 10 min |
|---|---|---|---|
| `sortformer-v2` | 10.4 | 95.6 | 1.8 |
| `pyannote-community-1` | 11.0 | 95.1 | 11.2 |
| `sortformer-v2.1` | 11.2 | 95.7 | 3.2 |
| `pyannote-3.1` | 11.2 | 95.4 | 10.3 |
| `sortformer-v1` | 14.6 | 93.4 | 3.7 |
| `diarizen-large`, `-v2` | 14.8 | 93.5 | ~25 |
| `ecapa-ahc` | 17.1 | 89.1 | 0.7 |

The top four are within half a point, less than the uncertainty of a reference made without listening,
and the per-clip winner changes from clip to clip.

## Deciding by ear: `judge.py`

```bash
python Transcription/diarization/judge.py      # PowerShell, from the repo root; then open http://127.0.0.1:8770
```

A local page that plays only the stretches (>= 0.5 s) where the top four systems or the reference disagree
about whether Urza is speaking: 244 stretches, 4 minutes of audio. For each you answer Urza / someone else /
both / nobody (keys 1–4, 0 = can't tell), blind by default. The scoreboard then shows, per system, the
seconds it got wrong, its Urza F1 against the reference corrected by your answers (before any answer, exactly
the table above), and how often the leader stays ahead under resampling (95%+ = a real difference). Answers
are saved after every click to `judge_verdicts.json`. It reads the MetaCentrum outputs from
`~/urza_diar/hyp_mc/` (copied from `~/urza_diar/hyp/` on MetaCentrum), the clips from `~/urza_diar/clips/`
and the Whisper word times from `~/urza_diar/words/`; `--systems` compares other systems.

**Result (17 Nov, 71 longest stretches judged by ear):** both Sortformers got about 16–17 s of the judged
audio wrong, both pyannotes about 57–59 s (Sortformer ahead in 100% of resamples); Sortformer v2 vs v2.1 is a
tie (16.1 vs 17.1 s, 81%). The reference was wrong for 66.6 s of it, so the Claude-made labels are not
reliable exactly where the systems disagree. Choice: **Sortformer v2.1** (same accuracy as v2, the newest
release of the model). Its limit is 4 speakers; none of the test clips has more.

## Run on MetaCentrum (everything as jobs; the frontend only submits)

Layout in `~/urza_diar`: `code/` (a copy of `Transcription/`), `logs/`. Uses the ASR benchmark's `uv` and
its `venv` (torch cu128 + NeMo) for Sortformer, and its Hugging Face cache.

```bash
mkdir -p ~/urza_diar/logs && cd ~/urza_diar        # put a copy of Transcription/ there as code/
S=$(qsub -o logs/ code/diarization/setup_env.pbs)  # builds venv_dz (DiariZen) and venv_pya (pyannote, scoring)
qsub -o logs/ -W depend=afterok:$S code/diarization/run_diar.pbs
```

`run_diar.pbs` cuts the clips from the shared audio folder (first run only), runs every system, then prints the
score table to `hyp/scores.txt`. One system only: `qsub -o logs/ -v SYSTEMS="diarizen-large" code/diarization/run_diar.pbs`.

**pyannote** (`pyannote-community-1`, `pyannote-3.1`) is gated: once, accept the conditions on
<https://huggingface.co/pyannote/speaker-diarization-community-1> (and `.../speaker-diarization-3.1`) with your
HF account, then save a read token where the jobs look for it (HF_HOME is the ASR benchmark's cache):

```bash
HF_HOME=/storage/praha1/home/rolloj/urza_asr_bench/hf-cache \
  ~/urza_diar/venv_pya/bin/python -c "from huggingface_hub import login; login()"
```

Without it the two pyannote systems fail and the rest still run.

## Systems (`diarize.py --system ...`)

| system | what | limits |
|---|---|---|
| `sortformer-v2.1`, `sortformer-v2` | NVIDIA streaming Sortformer, "high latency" setting | at most 4 speakers |
| `sortformer-v1` | NVIDIA offline Sortformer | at most 4 speakers, trained on ≤ 90 s |
| `diarizen-large`, `diarizen-large-v2` | BUT DiariZen: WavLM segmentation + pyannote clustering | needs torch < 2.9 |
| `pyannote-community-1`, `pyannote-3.1` | pyannote.audio pipelines | gated (HF token) |
| `ecapa-ahc` | baseline: energy VAD + 3 s ECAPA windows + agglomerative clustering | its 2 settings were tuned on these clips |

## How the reference was made (`reference/`, `labelling/`)

Labelled by Claude from the transcript, not by ear, with three independent cues, and never from the outputs
of the systems tested:

1. **Content**: Whisper large-v3 words with timestamps (`labelling/transcribe_clips.py`), cut into small
   pieces at pauses > 0.3 s, every 12 words, and wherever any system proposed a speaker change
   (`make_pieces.py`). Only the cut points come from the systems, never their speaker labels.
2. **Pitch** per piece (pYIN), e.g. the studio clip has only Urza and a female student and was labelled by pitch.
3. **Voice similarity**: each piece's ECAPA cosine to Urza's solo video (`add_urza_sim.py`), and between long
   blocks to check whether two blocks are the same person (`block_sim.py`). Same person ≈ 0.6–0.9, others ≈ 0.0–0.4.

`reference/<clip>.lab` lists the pieces per speaker (`labelling/pieces/<clip>.txt` shows each piece's
time, pitch, similarity to Urza and text); `labels_to_rttm.py` turns them into `.rttm` + `.uem`.
Not scored (`.uem` holes): pieces marked `?` (crosstalk, one-word backchannels, unclear voices) and
stretches > 2 s with no transcribed word. 80–100 % of each clip is scored.

Known weaknesses: boundaries follow Whisper's word times (the 0.25 s collar absorbs most of it); overlapping
speech has one speaker only; who is who among the non-Urza speakers is less certain than Urza vs. not-Urza;
ECAPA was used both for labelling help and for the automatic Urza recognition, which may flatter all systems
a little on that column (equally, since they all use the same recognition step).
