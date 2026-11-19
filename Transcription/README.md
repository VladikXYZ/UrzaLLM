# Transcription: the Urza videos as text

## The pipeline

| Stage | What | Tool | Status |
|---|---|---|---|
| 0. Audio | one `<youtube_id>.webm` per video (Opus) in the shared folder; new videos via `yt_scraper.py` | yt-dlp | 980 files |
| 1. **Transcription** | Czech (and English) text with a start, end and **confidence for every word** | `transcribe_local.py` + `run_local.pbs` (faster-whisper large-v3 on MetaCentrum GPUs); gaps can be filled from YouTube captions with `patch_from_srt.py` | done: 980 videos, 1,543 h |
| 2. Speakers | who speaks when (diarization) | NeMo Sortformer v2.1 (won the benchmark in `diarization/`, decided by ear with `judge.py`); `diarization/diarize_channel.py` + `run_channel.pbs` | done: 980 videos |
| 3. Urza | which segments are Urza: voiceprints (ECAPA) against 12 Urza references, score-normalised, decided per segment; 61 borderline speakers checked by ear (`review_borderline.py`) | `diarization/rescore_urza.py` | done: 872 h of Urza's speech, 43 videos without him |
| 4. Checks | unsure words, suspect segments, share of Urza per video | stats in the stage 1–3 outputs | partly |
| 5. RAG chunks | Speaker turns split at sentence ends and pauses, with the preceding question attached to Urza's answers | `chunk_videos.py` | done: 57,693 Urza chunks and 10,534 other-speaker chunks |

Why faster-whisper large-v3: it was the most accurate model in `benchmark/` (debate clip: 2.4% word error
rate counting colloquial = standard Czech, vs 3.9% for the same model through the e-INFRA API and 8.8% for
large-v3-turbo). Running it ourselves also gives word-level timestamps and probabilities, which the
e-INFRA API does not return.

## Build the RAG chunks

Run from the repository root with Python and NumPy installed.

```bash
python Transcription/chunk_videos.py
```

The script reads timed transcripts and speaker decisions from `~/urza_diar`. It groups words into speaker
turns, splits long turns near 225 words, and attaches up to 80 words from the preceding question to each
part of Urza's answer. Housekeeping chunks are flagged and their question is omitted.

The script writes `real_data/transcripts_urza.json` and `real_data/transcripts_other.json` in Vladimir's
transcription format, with one field per line. `time` and `end_time` give the clip boundaries
in seconds, and `speaker` identifies Urza or the video's local speaker label. `context_cs` holds
the preceding question or remark in its original language. `context_en` holds the same context
when it was spoken in English, or stays empty until translated.

`translation_status` describes `text_en`: `original_english` for speech already in English,
`translated` for a supplied translation, and `source_placeholder` for a copy of untranslated
source text. The viewer uses the transcripts and diarization for word confidence and speaker timelines.

```bash
python Transcription/visualize.py
```

Use `--work` to point to a different transcript and diarization directory. To try a small run, pass
`--limit 10 --out-dir PATH` and give the viewer `--chunks-dir PATH` to inspect that output.

## Stage 1: `transcribe_local.py`

Every audio file gets its own folder:

```
transcriptions/
  <video_name>__faster-whisper-large-v3/
    transcript.json      metadata + segments + words with confidence
    transcript.txt       "[hh:mm:ss] text" lines, for reading
    error.txt            only if the video failed; removed once it succeeds
```

`transcript.json`:

```json
{
  "video_id": "08hwEuoSR_o", "title": "...", "upload_date": "20220514",
  "model": "large-v3", "engine": "faster-whisper (beam 5)", "language": "auto", "duration": 1153.2,
  "languages": {"cs": 1101.2, "en": 52.0},
  "n_words": 2310, "mean_word_prob": 0.91, "share_words_below_0.5": 0.04, "n_suspect": 0,
  "windows": [{"start": 0.0, "end": 24.6, "language": "cs", "p_en": 0.01}],
  "segments": [
    {"id": 0, "start": 0.0, "end": 24.6, "language": "cs", "text": "Dobrý den, ...",
     "avg_logprob": -0.21, "no_speech_prob": 0.01, "compression_ratio": 1.5, "suspect": false,
     "min_word_prob": 0.42,
     "words": [{"word": "Dobrý", "start": 0.0, "end": 0.38, "prob": 0.97},
               {"word": "den,",  "start": 0.38, "end": 0.71, "prob": 0.99}]}
  ],
  "dropped": [{"start": 340.55, "text": "www.hradeckralove.org", "why": "known hallucination"}]
}
```

- **`prob`** is Whisper's probability for the word (0–1). Around 0.9+ it is sure; below ~0.5 it was unsure,
  which is where errors cluster: names, quiet or overlapping speech.
- **A segment is one window** of 10–29.5 s cut at a pause; silent windows are skipped (Whisper invents text
  in silence). Each window is transcribed on its own, one after another; if Whisper stops early in a window
  it continues from its last timestamp, so no speech is skipped.
- **Speed**: 10–22x real time depending on the GPU; the whole channel (1,543 h) took 101 GPU-hours. Batched
  decoding (`BatchedInferencePipeline`) was ~2x faster but dropped words (debate clip: 6.0% vs 2.4% word
  error rate), so it was removed.
- **`--fix`**: upgrades transcripts made before the language check and cleanup (the first full run forced
  Czech): detects every window's language, re-transcribes only the English windows, applies the cleanup,
  and adds `fixed` to the file.
- **Language per window**: Whisper first detects each window's language (`windows[].p_en` = probability
  of English). A window is English when p_en > 0.5, or > 0.2 next to such a window (Urza's Czech-accented
  English often scores only 0.3–0.5; Czech and Slovak stayed below 0.16). English windows are transcribed
  as English, the rest as Czech. With the language forced to Czech, Whisper *translates* English talks.
- **`suspect`**: Whisper's own warning signs (compression ratio > 2.4 = repetition loop, average
  log-probability < -1.0). Kept, but flagged.
- **`dropped`**: text Whisper invents where nobody speaks, learned from its training subtitles: the
  "Titulky vytvořil JohnyX" credits, `www.hradeckralove.org` (893x in the first run, never said),
  `www.arkance-systems.cz`, and "Děkuji/Děkujeme za pozornost." when no_speech_prob > 0.6. Real addresses
  (`opristavu.urza.cz`) and short replies ("Jo, přesně tak.") stay.
- **Resumable**: finished videos are skipped, so a job that ran out of time is just submitted again.

### Run on MetaCentrum (everything as jobs; the frontend only submits)

Layout in `~/urza_transcribe`: `code/` (this folder's `transcribe_local.py`, `transcribe_einfra.py`,
`run_local.pbs`), `jsons/channel_metadata.json` (titles and dates), `logs/`. The Python environment is the
one built by `benchmark/setup_env.pbs` (`~/urza_asr_bench/venv`, models cached in `~/urza_asr_bench/hf-cache`).

```bash
cd ~/urza_transcribe
qsub -o logs/ -v FILES=08hwEuoSR_o code/run_local.pbs     # one video, as a test
qsub -o logs/ -J 1-20 -v NSHARDS=20 code/run_local.pbs     # everything, 20 GPU tasks in parallel
qsub -o logs/ -J 1-10 -v NSHARDS=10,FIX=1 code/run_local.pbs   # upgrade old transcripts (see --fix)
```

`qsub` needs a Kerberos ticket on the frontend (`kinit`). A login session keeps it in its own file
(`/tmp/krb5cc_<uid>_XXXXXX`); another SSH session uses it with `export KRB5CCNAME=FILE:/tmp/krb5cc_...`.

Outputs go to `~/urza_transcribe/transcriptions/` (change with `-v OUT_DIR=...`). The audio is read from the
shared folder `/storage/brno2/home/vladikxyz/scraping/audio` (read access is enough).

## Alternative: `transcribe_einfra.py` (e-INFRA CZ Whisper API)

The same model through `https://llm.ai.e-infra.cz/v1/audio/transcriptions`, no GPU jobs needed: about 50x
real time with 4 parallel requests per key, segment times but **no word timestamps or word confidence**.
Same windowing (the server would otherwise cut every upload at fixed 30 s marks, splitting words), output
in `<video_name>__API-whisper-large-v3/` with `windows.jsonl` for resuming. API keys are personal; large batch
runs should be agreed with CERIT-SC support (k8s@cerit-sc.cz) in advance.

```bash
export EINFRA_API_KEY=$(cat ~/.einfra_key)
python3 Transcription/transcribe_einfra.py --audio-dir AUDIO_DIR --out transcriptions --limit 1
```

## Sharing a folder on MetaCentrum storage (careful)

Do **not** use `nfs4_setfacl -R -a A:fd:LOGIN@META:...` on a folder: adding the first inheritable entry
without one for the owner makes every new file there unreadable to its own owner. Set the full list instead
(owner + the other person, both inherited), for example read access for `LOGIN`:

```bash
find DIR -type d -exec nfs4_setfacl -s "A::OWNER@:rwaDxtTcCy,A::LOGIN@META:rxtcy,A::GROUP@:tcy,A::EVERYONE@:tcy,A:fdi:OWNER@:rwaDxtTcCy,A:fdi:LOGIN@META:rxtcy,A:fdi:GROUP@:tcy,A:fdi:EVERYONE@:tcy" {} +
```
