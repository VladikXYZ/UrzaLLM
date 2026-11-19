"""Build video transcript chunks in Vladimir's transcription format.

    python Transcription/chunk_videos.py

Inputs under --work (default ~/urza_diar):
  transcripts/<video>__faster-whisper-large-v3/transcript.json   timed words
  channel/<video>.json                                        Sortformer segments
  channel/urza_videos.json                                    segments attributed to Urza
Titles and dates come from --meta (default real_data/yt_full.json).

1. Assign speakers by word midpoint and smooth small errors at speaker changes.
2. Group consecutive words into turns, dropping interjections between parts of a turn.
3. Split turns into roughly 225-word chunks, using sentence ends and pauses.
   Keep Urza turns of at least 30 words and other speakers' turns of at least 150.
4. Attach up to 80 words from the preceding question to each part of Urza's answer.
   Flag organisational talk as housekeeping and omit its question.

Outputs under --out-dir (default real_data):
  transcripts_urza.json   Urza's speech
  transcripts_other.json  other speakers' speech
Both are indented JSON arrays with type "transcription", start and end times in
seconds, speaker, and question context. translation_status marks whether text_en
is original English, a translation, or a copy of the source text.
"""
import argparse
import json
import re
import sys
from collections import Counter
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TARGET_WORDS = 225
MIN_URZA_WORDS, MIN_OTHER_WORDS = 30, 150
INTERJ_SEC, INTERJ_WORDS = 3.0, 8
SNAP_WORDS, FLICKER_WORDS = 3, 2
PAUSE = 1.5
CTX_WORDS, CTX_MAX, CTX_LONG = 70, 80, 150
SENT_END = re.compile(r"[.?!…]+[\"'“”»)]*$")
FRAME = 0.05
CZECH = set("áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ")
EN_WORDS = {"the", "of", "and", "on", "for", "with", "at", "is", "to", "in", "from", "about"}
OUTPUT_FILES = {
    "transcript": "transcripts_urza.json",
    "transcript_other": "transcripts_other.json",
}


def load_video_metadata(path):
    if not path.exists():
        return {}
    return {video["id"]: video for video in json.loads(path.read_text(encoding="utf-8"))}


def video_date(metadata):
    date = str(metadata.get("date") or "")
    return f"{date[:4]}-{date[4:6]}-{date[6:8]}" if len(date) == 8 else date


def load_dates_titles(path):
    return {vid: (video.get("title_cs"), video_date(video))
            for vid, video in load_video_metadata(path).items()}


def english_title(metadata):
    title = metadata.get("title_cs") or ""
    words = {word.lower() for word in re.findall(r"[A-Za-z]+", title)}
    if not CZECH.intersection(title) and EN_WORDS.intersection(words):
        return title
    return metadata.get("title_en") or ""


def speakers_of_words(words, segs, urza_idx):
    """Speaker of every word: 'Urza' or the diarization label, from the segment covering the word's middle,
    else the nearest one within 1 s, else the speaker of the word before.

    Segments come from Sortformer grouped by speaker, NOT sorted by time, so they are painted onto a time
    grid (50 ms frames) instead of searched. Where two overlap, Urza's segment wins (painted last)."""
    if not segs:
        return ["?"] * len(words)
    n = int(max(max(e for _, e, _, _ in segs), words[-1]["end"]) / FRAME) + 2
    owner = np.full(n, -1, int)
    for k in sorted(range(len(segs)), key=lambda k: (k in urza_idx, -(segs[k][1] - segs[k][0]))):
        owner[int(segs[k][0] / FRAME):int(np.ceil(segs[k][1] / FRAME))] = k    # longer first, Urza last
    # frames without a segment: the nearest owned frame within 1 s
    reach = int(1.0 / FRAME)
    idx = np.arange(n)
    has = owner >= 0
    prev = np.where(has, idx, -10**9)
    prev = np.maximum.accumulate(prev)
    nxt = np.where(has, idx, 10**9)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    use_prev = (idx - prev) <= (nxt - idx)
    near = np.where(use_prev, prev, nxt)
    dist = np.minimum(idx - prev, nxt - idx)
    filled = np.where(has, owner, np.where(dist <= reach, owner[np.clip(near, 0, n - 1)], -1))
    out = []
    for w in words:
        k = filled[min(n - 1, int((w["start"] + w["end"]) / 2 / FRAME))]
        out.append(None if k < 0 else ("Urza" if k in urza_idx else segs[k][2]))
    for i in range(len(out)):                # words far from any segment: the speaker before them
        if out[i] is None:
            out[i] = out[i - 1] if i else next((x for x in out if x), "?")
    return out


# Organisational talk: next lecture, donations, QR codes, technical set-up, goodbyes. A chunk with at least
# HOUSEKEEPING_MIN of these (different ones) is flagged "housekeeping" (kept, but skipped by retrieval by default).
HOUSEKEEPING = {name: re.compile(p, re.IGNORECASE) for name, p in {
    "next event": r"příští\w* (přednášk|stream|díl|besed|setkán)|(přednášk|stream)\w* (nebude|bude ve? |bude zase)|"
                  r"první středa|zvu vás|pozvu vás|přijďte",
    # not plain "příspěvek": it is also a (Facebook) post
    "donations": r"podpo\w+ (svobodný přístav|studio|nás|kanál|tenhle)|pokladničk|qr kód|"
                 r"číslo účtu|variabilní symbol|bitcoinov\w+ adres|daruj|dobrovoln\w+ (příspěv|platb)|"
                 r"(finanční|pravideln\w+|trval\w+|jakýkoliv|měsíční) příspěv|přispě\w+ (nám|na provoz|na studio)|"
                 r"(ruší|zrušil\w*|posíla\w+) příspěvk",
    "thanks/goodbye": r"děkuj\w* (vám )?za pozornost|na shledanou|mějte se|to by bylo pro dnešek|"
                      r"tím bych (to )?(ukončil|skončil)|díky,? že jste (přišli|se dívali|tu byli)",
    "tech": r"slyšíte (mě|nás)|je (mě|to) slyšet|mikrofon|zvuk (je|nejde)|stream (běží|jede|spadl)|"
            r"vidíte (mě|obrazovku|prezentaci)|sdílím obrazovku",
    "channel": r"odebír\w+|lajk|kanál\w* svobodného přístavu|popisk\w* (videa|pod videem)|odkaz\w* (pod videem|v popisku)",
}.items()}
HOUSEKEEPING_MIN = 2


def housekeeping(text):
    return sorted(name for name, rx in HOUSEKEEPING.items() if rx.search(text))


def smooth_flicker(spk, max_len=FLICKER_WORDS):
    """A run of <= max_len words between two runs of the same speaker gets that speaker: the diarization
    sometimes flips for a word or two inside someone's sentence ("…Jako všeho, protože [zjišťovali se] to
    vlastně vykat."), and snap_to_sentences would otherwise treat the flip as a real speaker change."""
    spk = list(spk)
    while True:                                  # each round fixes every flicker whose neighbours are untouched
        rs = speaker_runs(spk)
        done, k = 0, 1
        while k < len(rs) - 1:
            if len(rs[k][1]) <= max_len and rs[k - 1][0] == rs[k + 1][0] != rs[k][0]:
                for i in rs[k][1]:
                    spk[i] = rs[k - 1][0]
                done += 1
                k += 2                           # the merged run's neighbours are re-checked next round
            else:
                k += 1
        if not done:
            return spk


def snap_to_sentences(words, spk, max_move=SNAP_WORDS):
    """Move speaker changes that fall within max_move words of a sentence end onto that sentence end.
    The diarization often switches a word or two early or late: "…když už jich bude | 5 %? Já si myslím…"
    (the question's end went to the answer) or "…protože to vlastně vykat. Vida, | to je dobrá připomínka"
    (the answer's first word went to the question)."""
    spk = list(spk)
    n = len(spk)
    ends = [bool(SENT_END.search(w["word"])) for w in words]
    i = 1
    while i < n:
        if spk[i] == spk[i - 1] or ends[i - 1]:
            i += 1
            continue
        run_end = i
        while run_end < n and spk[run_end] == spk[i]:
            run_end += 1
        # the new speaker's first words finish the previous speaker's sentence
        m = next((k for k in range(i, min(i + max_move, run_end)) if ends[k]), None)
        if m is not None and m + 1 < run_end:
            for k in range(i, m + 1):
                spk[k] = spk[i - 1]
            i = m + 1
            continue
        # the previous speaker's last words start the new speaker's sentence
        run_start = i - 1
        while run_start > 0 and spk[run_start - 1] == spk[i - 1]:
            run_start -= 1
        e = next((k for k in range(i - 2, max(i - 2 - max_move, run_start - 1), -1) if ends[k]), None)
        if e is not None and e >= run_start:
            for k in range(e + 1, i):
                spk[k] = spk[i]
        i += 1
    return spk


def speaker_runs(speakers):
    """Group consecutive word indices with the same speaker."""
    return [
        (speaker, [index for index, _ in run])
        for speaker, run in groupby(enumerate(speakers), key=lambda item: item[1])
    ]


def turns(words, speakers):
    """Join runs of the same speaker across short interjections."""
    runs = speaker_runs(speakers)
    position = 0
    while position < len(runs):
        speaker, indices = runs[position]
        while position + 2 < len(runs) and runs[position + 2][0] == speaker:
            interruption = runs[position + 1][1]
            duration = words[interruption[-1]]["end"] - words[interruption[0]]["start"]
            if duration >= INTERJ_SEC or len(interruption) > INTERJ_WORDS:
                break
            indices.extend(runs[position + 2][1])
            position += 2
        yield speaker, indices
        position += 1


def cut_points(words, idx, n_parts):
    """Split a turn (word indices) into n_parts at sentence ends near equal targets, preferring pauses."""
    n = len(idx)
    cuts = []
    for p in range(1, n_parts):
        target = round(p * n / n_parts)
        lo, hi = max(cuts[-1] + 1 if cuts else 1, target - 75), min(n - 1, target + 75)
        best, best_score = None, None
        for j in range(lo, hi):                           # cut after position j-1 (j = first word of the next part)
            prev = words[idx[j - 1]]
            if not SENT_END.search(prev["word"]):
                continue
            pause = words[idx[j]]["start"] - prev["end"]
            score = abs(j - target) - (60 if pause > PAUSE else 0) - min(pause, 3) * 5
            if best_score is None or score < best_score:
                best, best_score = j, score
        if best is None:                                  # no sentence end: the longest pause
            best = max(range(lo, hi), key=lambda j: words[idx[j]]["start"] - words[idx[j - 1]]["end"], default=target)
        cuts.append(best)
    bounds = [0] + cuts + [n]
    return [idx[a:b] for a, b in zip(bounds, bounds[1:]) if b > a]


def text_of(words, idx):
    return re.sub(r"\s+([,.?!…:;])", r"\1", " ".join(words[i]["word"] for i in idx)).strip()


def question_excerpt(words, indices):
    """Return word indices from a sentence start close to 70 words from the end."""
    tail = indices[-CTX_MAX:]
    starts = [0] if 10 <= len(indices) <= CTX_MAX else []
    starts.extend(j for j in range(1, len(tail) - 9)
                  if SENT_END.search(words[tail[j - 1]]["word"]))
    if not starts:
        return indices[-CTX_WORDS:]
    start = min(starts, key=lambda j: abs(len(tail) - j - CTX_WORDS))
    return tail[start:]


def chunk_video(transcript, segments, urza_indices, metadata):
    """Yield corpus records for one video in timeline order."""
    words, languages = [], []
    for segment in transcript["segments"]:
        for word in segment["words"]:
            if word["word"].strip():
                words.append(word)
                languages.append(segment.get("language", "cs"))
    if not words:
        return
    speakers = speakers_of_words(words, segments, urza_indices)
    speakers = snap_to_sentences(words, smooth_flicker(speakers))
    video_id = transcript["video_id"]
    title_cs = (metadata or {}).get("title_cs") or transcript.get("title") or ""
    title_en = english_title(metadata or {})
    date = video_date(metadata) if metadata is not None else None
    counts = Counter()
    question_speaker, question, question_en, question_truncated = None, "", "", False

    for speaker, indices in turns(words, speakers):
        is_urza = speaker == "Urza"
        if not is_urza and len(indices) >= 3:
            question_speaker = speaker
            excerpt = question_excerpt(words, indices)
            question = text_of(words, excerpt)
            question_en = question if all(languages[i] == "en" for i in excerpt) else ""
            question_truncated = len(indices) > CTX_LONG

        min_words = MIN_URZA_WORDS if is_urza else MIN_OTHER_WORDS
        if len(indices) < min_words:
            continue

        chunk_type = "transcript" if is_urza else "transcript_other"
        id_prefix = "u" if is_urza else "o"
        parts = cut_points(words, indices, max(1, round(len(indices) / TARGET_WORDS)))
        for part_number, part in enumerate(parts, 1):
            counts[chunk_type] += 1
            text = text_of(words, part)
            signals = housekeeping(text)
            is_housekeeping = len(signals) >= HOUSEKEEPING_MIN
            part_languages = {languages[i] for i in part}
            language = languages[part[0]] if len(part_languages) == 1 else "mixed"
            probabilities = [words[i]["prob"] for i in part if words[i].get("prob") is not None]
            start = words[part[0]]["start"]
            part_label = f"{part_number}/{len(parts)}" if len(parts) > 1 else ""
            has_context = is_urza and question_speaker is not None
            keep_context = has_context and not is_housekeeping
            yield {
                "title_cs": title_cs,
                "subtitle_cs": f"část {part_label}" if part_label else "",
                "text_cs": text,
                "title_en": title_en,
                "subtitle_en": f"part {part_label}" if part_label else "",
                "text_en": text if language == "en" else "",
                "src": f"https://www.youtube.com/watch?v={video_id}&t={int(start)}s",
                "type": chunk_type,
                "id": f"{video_id}-{id_prefix}{counts[chunk_type]:04d}",
                "video_id": video_id,
                "speaker": speaker,
                "date": date,
                "start": round(start, 2),
                "end": round(words[part[-1]]["end"], 2),
                "part": part_number,
                "parts": len(parts),
                "language": language,
                "n_words": len(part),
                "mean_word_prob": round(sum(probabilities) / len(probabilities), 3) if probabilities else None,
                "low_conf_words": sum(probability < 0.5 for probability in probabilities),
                "context_speaker": question_speaker if keep_context else None,
                "context_cs": question if keep_context else "",
                "context_truncated": keep_context and question_truncated,
                "context_en": question_en if keep_context else "",
                "context_dropped": has_context and is_housekeeping,
                "housekeeping": is_housekeeping,
                "housekeeping_signals": signals,
            }


def corpus_chunks(transcript_paths, channel, urza_by_video, metadata):
    for path in transcript_paths:
        transcript = json.loads(path.read_text(encoding="utf-8"))
        video_id = transcript["video_id"]
        diarization_path = channel / f"{video_id}.json"
        if video_id not in urza_by_video or not diarization_path.exists():
            print(f"skip {video_id}: no diarization", flush=True)
            continue
        segments = json.loads(diarization_path.read_text(encoding="utf-8"))["segments"]
        yield from chunk_video(transcript, segments, urza_by_video[video_id], metadata.get(video_id))


def export_record(chunk):
    """Keep the original corpus fields first, followed by the video metadata."""
    if chunk["language"] == "en":
        translation_status = "original_english"
    elif chunk["text_en"]:
        translation_status = "translated"
    else:
        translation_status = "source_placeholder"

    return {
        "title_cs": chunk["title_cs"],
        "subtitle_cs": chunk["subtitle_cs"],
        "text_cs": chunk["text_cs"],
        "title_en": chunk["title_en"],
        "subtitle_en": chunk["subtitle_en"],
        "text_en": chunk["text_en"] or chunk["text_cs"],
        "src": f"https://www.youtube.com/watch?v={chunk['video_id']}",
        "type": "transcription",
        "time": chunk["start"],
        "context_cs": chunk["context_cs"],
        "context_en": chunk["context_en"],
        "end_time": chunk["end"],
        "speaker": chunk["speaker"],
        "translation_status": translation_status,
    }


def write_corpus(chunks, out_dir):
    """Write both JSON arrays incrementally, then replace the previous outputs."""
    counts = Counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out_dir) as temporary_dir:
        staging = Path(temporary_dir)
        urza_path = staging / OUTPUT_FILES["transcript"]
        other_path = staging / OUTPUT_FILES["transcript_other"]
        with (
            urza_path.open("w", encoding="utf-8", newline="\r\n") as urza_file,
            other_path.open("w", encoding="utf-8", newline="\r\n") as other_file,
        ):
            files = {"transcript": urza_file, "transcript_other": other_file}
            for file in files.values():
                file.write("[")
            for chunk in chunks:
                chunk_type = chunk["type"]
                file = files[chunk_type]
                file.write(",\n" if counts[chunk_type] else "\n")
                record = json.dumps(export_record(chunk), ensure_ascii=False, indent=1)
                file.write(" " + record.replace("\n", "\n "))
                counts[chunk_type] += 1
            for chunk_type, file in files.items():
                file.write("\n]" if counts[chunk_type] else "]")
        for chunk_type, name in OUTPUT_FILES.items():
            output = out_dir / name
            (staging / name).replace(output)
            print(f"{output}: {counts[chunk_type]:,} chunks")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, default=Path.home() / "urza_diar")
    ap.add_argument("--transcripts", type=Path, help="default: <work>/transcripts")
    ap.add_argument("--meta", type=Path, default=ROOT / "real_data" / "yt_full.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "real_data")
    ap.add_argument("--limit", type=int, help="process only the first N transcripts")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be positive")

    transcript_dir = args.transcripts or args.work / "transcripts"
    paths = sorted(transcript_dir.glob("*/transcript.json"))[:args.limit]
    if not paths:
        ap.error(f"no transcript.json files found under {transcript_dir}")

    channel = args.work / "channel"
    decisions = json.loads((channel / "urza_videos.json").read_text(encoding="utf-8"))
    urza_by_video = {video["video_id"]: set(video["urza_segments"]) for video in decisions["videos"]}
    chunks = corpus_chunks(paths, channel, urza_by_video, load_video_metadata(args.meta))
    write_corpus(chunks, args.out_dir)


if __name__ == "__main__":
    main()
