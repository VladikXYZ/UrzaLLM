"""Build the translation package: every video's transcript as numbered sentences, to translate into English.

    python Transcription/make_translation_package.py [--work ~/urza_diar] [--out Transcription/translation_package]

Why sentences: the translation must survive any later re-chunking. chunk_videos.py cuts only at sentence
ends, so a chunk is always a run of whole sentences, and its English text is the translations of those
sentences put together, whatever the chunk sizes are. (Whisper segments would not do: a chunk boundary can
fall inside one.)

A sentence ends at . ? ! or …, where the language changes, where another speaker takes over (a run of at
least 4 words, so a stray word at a speaker change does not split a sentence), and at the longest pause
once it reaches 60 words (unpunctuated stretches). Each sentence gets the speaker who says most of its words:
"Urza" or the diarization label of someone else ("speaker_1", ...; numbered per video).

Writes one flat folder: <out>/<video_id>.json (the sentences; with --txt also <video_id>.txt, readable with
speakers) and <out>/index.json
(every video with its counts). Videos with no Czech speech are left out. Inputs as for chunk_videos.py.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunk_videos import ROOT, SENT_END, load_dates_titles, speakers_of_words  # noqa: E402

MAX_WORDS, SPEAKER_RUN = 60, 4


def sentences(words, langs, spk):
    n = len(words)
    cut = [False] * n                          # cut[i]: a new sentence starts at word i
    for i in range(1, n):
        if SENT_END.search(words[i - 1]["word"]) or langs[i] != langs[i - 1]:
            cut[i] = True
    i, last = 0, 0
    while i < n:                               # speaker changes that last >= SPEAKER_RUN words ...
        j = i
        while j < n and spk[j] == spk[i]:
            j += 1
        last = max([k for k in range(last, i + 1) if cut[k]] or [last])
        # ... unless that would leave a stub of < SPEAKER_RUN words before it: the diarization often puts a
        # speaker change a word early ("Já" | "jsem nevěděl, že ..."), and the stub belongs to the sentence
        if i > 0 and j - i >= SPEAKER_RUN and i - last >= SPEAKER_RUN:
            cut[i] = True
        i = j
    out, start = [], 0
    for i in range(1, n + 1):
        if i == n or cut[i]:
            out += split_long(words, start, i)
            start = i
    return out


def split_long(words, a, b):
    """[(a, b)] word ranges of at most MAX_WORDS, cut at the longest pauses."""
    if b - a <= MAX_WORDS:
        return [(a, b)]
    j = max(range(a + MAX_WORDS // 3, b - MAX_WORDS // 3 + 1) if b - a >= MAX_WORDS else range(a + 1, b),
            key=lambda k: words[k]["start"] - words[k - 1]["end"])
    return split_long(words, a, j) + split_long(words, j, b)


def text_of(ws):
    return re.sub(r"\s+([,.?!…:;])", r"\1", " ".join(w["word"] for w in ws)).strip()


def hms(t):
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, default=Path.home() / "urza_diar")
    ap.add_argument("--meta", type=Path, default=ROOT / "real_data" / "yt_full.json")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "translation_package")
    ap.add_argument("--txt", action="store_true", help="also write a readable <video_id>.txt next to each json")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    meta = load_dates_titles(args.meta)
    uv = {v["video_id"]: v for v in json.loads((args.work / "channel" / "urza_videos.json").read_text(encoding="utf-8"))["videos"]}
    index, total = [], Counter()
    for p in sorted((args.work / "transcripts").glob("*/transcript.json")):
        t = json.loads(p.read_text(encoding="utf-8"))
        vid = t["video_id"]
        words, langs = [], []
        for s in t["segments"]:
            for w in s["words"]:
                if w["word"].strip():
                    words.append(w)
                    langs.append(s.get("language", "cs"))
        if not words or "cs" not in langs:
            total["skipped (no Czech)"] += 1
            continue
        dp = args.work / "channel" / f"{vid}.json"
        if dp.exists() and vid in uv:
            spk = speakers_of_words(words, json.loads(dp.read_text(encoding="utf-8"))["segments"], set(uv[vid]["urza_segments"]))
        else:
            spk = ["?"] * len(words)
        sents = []
        for a, b in sentences(words, langs, spk):
            who = Counter(spk[a:b]).most_common(1)[0][0]
            lang = langs[a]
            sents.append({"id": f"s{len(sents) + 1:05d}", "start": round(words[a]["start"], 2), "end": round(words[b - 1]["end"], 2),
                          "speaker": who, "language": lang, "translate": lang != "en", "text": text_of(words[a:b])})
        title, date = meta.get(vid, (t.get("title"), None))
        args.out.mkdir(parents=True, exist_ok=True)
        doc = {"video_id": vid, "title": title, "date": date, "url": f"https://www.youtube.com/watch?v={vid}",
               "duration": t["duration"], "n_sentences": len(sents),
               "n_to_translate": sum(s["translate"] for s in sents),
               "words_to_translate": sum(len(s["text"].split()) for s in sents if s["translate"]),
               "sentences": sents}
        (args.out / f"{vid}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        if args.txt:
            (args.out / f"{vid}.txt").write_text(
                f"{title}\n{doc['url']}\n\n" + "".join(f"[{hms(s['start'])}] {s['id']} {s['speaker']}: {s['text']}\n" for s in sents),
                encoding="utf-8")
        index.append({k: doc[k] for k in ("video_id", "title", "date", "duration", "n_sentences", "n_to_translate", "words_to_translate")})
        total["videos"] += 1
        total["sentences"] += doc["n_to_translate"]
        total["words"] += doc["words_to_translate"]
    (args.out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{total['videos']} videos, {total['sentences']:,} sentences / {total['words']:,} words to translate; "
          f"{total['skipped (no Czech)']} English-only videos left out -> {args.out}")


if __name__ == "__main__":
    main()
