"""Translate the transcript sentences (translation_package/) into English with an LLM on e-INFRA CZ.

    python Transcription/translate_einfra.py [--model gemma4] [--workers 2] [--offpeak] [--files ID ...] [--limit N]

Reads   translation_package/<video_id>.json   (make_translation_package.py: numbered sentences, Czech)
Writes  translations/<video_id>.json          the same header, and per sentence id, start, end, speaker,
                                               language, text_cs (the original text) and text_en
        translations/.progress/<video_id>.jsonl  finished batches, so an interrupted run resumes
        translations/usage.jsonl               one line per request: tokens, seconds, model (for estimates)

API: https://llm.ai.e-infra.cz/v1 (OpenAI-compatible), key in the environment variable E_INFRA_API_TOKEN.
e-INFRA rules (docs.cerit-sc.cz, AI as a service): at most 4 parallel requests per account (more get HTTP 429),
30 min timeout per non-streaming request, "limit concurrent requests and test on a small sample before
processing a large dataset", heavy use at peak times slows the shared queue for others. So: --workers 2 by
default (half the account limit), exponential back-off on 429/5xx, and --offpeak to send requests only on
weekday evenings/nights (18:00-08:00) and weekends.

How a video is translated: its sentences are cut into batches of consecutive sentences (<= BATCH_WORDS
Czech words, <= BATCH_SENTENCES), each sent with the BATCH_CONTEXT sentences before it as read-only context
and the speaker of every sentence. The model must answer JSON {"translations": [{"id", "en"}, ...]} with
exactly the ids it was given; anything else is retried (lower temperature), then the batch is split in two.
Sentences spoken in English ("translate": false) are copied, not sent.
"""
import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = "https://llm.ai.e-infra.cz/v1/chat/completions"
BATCH_WORDS, BATCH_SENTENCES, BATCH_CONTEXT = 900, 80, 4

SYSTEM = """You translate Czech speech transcripts into English.

The transcripts come from the YouTube channel of the Czech anarcho-capitalist Urza (Free Haven): lectures,
interviews and debates about freedom, the state, economics, education and drugs. They were made by speech
recognition, so they are colloquial and contain recognition errors.

You get a list of numbered sentences in order, with who says each one. Translate every sentence into
natural spoken English, keeping the tone (colloquial stays colloquial; drop or lighten fillers like "jako",
"prostě", "no"). Sentences are often fragments that continue in the next one: translate each fragment as a
fragment, do not merge or split sentences, and never move words between ids. If a word is clearly misheard,
translate the most plausible meaning; never add content. Keep names as they are.

Fixed terms: Urza = Urza; Svobodný přístav = Free Haven; Dům Svobodného přístavu = Free Haven House;
Studio Svobodného přístavu = Free Haven Studio; Ježek bez klece = Ježek bez klece (a free school);
anarchokapitalismus / ankap = anarcho-capitalism / ancap; anarchoagorismus = anarcho-agorism;
sebeřízené vzdělávání = self-directed education; Svoboda učení = Freedom to Learn; etatista = statist.

Answer with JSON only: {"translations": [{"id": "<id>", "en": "<English>"}, ...]} with exactly one entry
for every id under TRANSLATE, in the same order, and nothing for the CONTEXT lines."""


class Usage:
    def __init__(self, path):
        self.path, self.lock = path, threading.Lock()

    def log(self, **kw):
        with self.lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), **kw}) + "\n")


def offpeak_wait(enabled):
    """Sleep until weekday 18:00-08:00 or a weekend, when --offpeak is on."""
    while enabled:
        now = datetime.now()
        if now.weekday() >= 5 or now.hour >= 18 or now.hour < 8:
            return
        time.sleep(300)


def batches(sents, *, max_words=BATCH_WORDS, max_sentences=BATCH_SENTENCES):
    """[(context sentences, sentences to translate)] over the sentences to translate, in order."""
    todo = [i for i, s in enumerate(sents) if s["translate"]]
    out, cur, words = [], [], 0
    for i in todo:
        n = len(sents[i]["text"].split())
        if cur and (words + n > max_words or len(cur) >= max_sentences):
            out.append(cur)
            cur, words = [], 0
        cur.append(i)
        words += n
    if cur:
        out.append(cur)
    return [([sents[j] for j in range(max(0, b[0] - BATCH_CONTEXT), b[0])], [sents[j] for j in b]) for b in out]


def prompt(title, ctx, todo):
    lines = [f"Video: {title}", ""]
    if ctx:
        lines += ["CONTEXT (do not translate):"] + [f"[{s['speaker']}] {s['text']}" for s in ctx] + [""]
    lines += ["TRANSLATE:"] + [f"{s['id']} [{s['speaker']}] {s['text']}" for s in todo]
    return "\n".join(lines)


def call(model, messages, temperature, usage, key, tag):
    body = json.dumps({"model": model, "messages": messages, "temperature": temperature,
                       "response_format": {"type": "json_object"}, "stream": False}).encode()
    for attempt in range(8):
        req = urllib.request.Request(API, data=body, headers={"Authorization": f"Bearer {key}",
                                                                "Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=1900) as r:
                d = json.loads(r.read())
            u = d.get("usage") or {}
            usage.log(tag=tag, model=model, seconds=round(time.time() - t0, 1), prompt_tokens=u.get("prompt_tokens"),
                      completion_tokens=u.get("completion_tokens"), status=200)
            return d["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            usage.log(tag=tag, model=model, seconds=round(time.time() - t0, 1), status=e.code)
            if e.code in (408, 429, 500, 502, 503, 504):
                time.sleep(min(600, 15 * 2 ** attempt) * (0.8 + 0.4 * random.random()))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            usage.log(tag=tag, model=model, seconds=round(time.time() - t0, 1), status=f"network: {e}")
            time.sleep(min(600, 15 * 2 ** attempt))
    raise RuntimeError("API kept failing")


def parse(text, ids):
    """{id: en} if the answer has exactly these ids with non-empty English, else None."""
    try:
        s = text.strip()
        if s.startswith("```"):
            s = s.strip("`").split("\n", 1)[1]
        d = json.loads(s[s.find("{"):s.rfind("}") + 1])
        got = {str(x["id"]): str(x["en"]).strip() for x in d["translations"]}
    except Exception:
        return None
    if set(got) != set(ids) or any(not got[i] for i in ids):
        return None
    return got


def translate_batch(model, title, ctx, todo, usage, key, tag):
    ids = [s["id"] for s in todo]
    for temperature in (0.2, 0.0):
        text = call(model, [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": prompt(title, ctx, todo)}], temperature, usage, key, tag)
        got = parse(text, ids)
        if got:
            return got
        usage.log(tag=tag, model=model, problem="bad answer", ids=len(ids))
    if len(todo) == 1:
        raise RuntimeError(f"{tag}: no valid translation for {ids[0]}")
    half = len(todo) // 2                     # split and try the halves
    a = translate_batch(model, title, ctx, todo[:half], usage, key, tag + "a")
    b = translate_batch(model, title, ctx + todo[:half], todo[half:], usage, key, tag + "b")
    return {**a, **b}


def translate_video(src: Path, out_dir: Path, model, pool, usage, key, offpeak):
    doc = json.loads(src.read_text(encoding="utf-8"))
    vid = doc["video_id"]
    prog = out_dir / ".progress" / f"{vid}.jsonl"
    done = {}
    if prog.exists():
        for line in prog.read_text(encoding="utf-8").splitlines():
            done.update(json.loads(line))
    bs = [(k, c, t) for k, (c, t) in enumerate(batches(doc["sentences"])) if not all(s["id"] in done for s in t)]
    lock = threading.Lock()

    def work(item):
        k, ctx, todo = item
        offpeak_wait(offpeak)
        got = translate_batch(model, doc["title"] or vid, ctx, todo, usage, key, f"{vid}#{k}")
        with lock, open(prog, "a", encoding="utf-8") as f:
            f.write(json.dumps(got, ensure_ascii=False) + "\n")
        return got

    for got in pool.map(work, bs):
        done.update(got)
    out = {k: doc[k] for k in ("video_id", "title", "date", "url", "duration", "n_sentences", "n_to_translate",
                               "words_to_translate")}
    out["model"] = model
    out["sentences"] = [{"id": s["id"], "start": s["start"], "end": s["end"], "speaker": s["speaker"],
                         "language": s["language"], "text_cs": s["text"],
                         "text_en": done.get(s["id"], "") if s["translate"] else s["text"]} for s in doc["sentences"]]
    missing = sum(1 for s in out["sentences"] if not s["text_en"])
    tmp = out_dir / f"{vid}.json.tmp"
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(out_dir / f"{vid}.json")
    return vid, len(bs), missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", type=Path, default=HERE / "translation_package")
    ap.add_argument("--out", type=Path, default=HERE / "translations")
    ap.add_argument("--model", default="gemma4")
    ap.add_argument("--workers", type=int, default=2, help="parallel requests (e-INFRA allows at most 4 per account)")
    ap.add_argument("--offpeak", action="store_true", help="only send requests weekdays 18-08 and on weekends")
    ap.add_argument("--files", nargs="*", help="only these video ids")
    ap.add_argument("--limit", type=int, help="at most this many videos")
    ap.add_argument("--max-batches", type=int, help="test: translate only the first N batches of each video")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    key = os.environ.get("E_INFRA_API_TOKEN") or os.environ.get("EINFRA_API_KEY")
    if not key:
        sys.exit("set E_INFRA_API_TOKEN")
    if args.workers > 4:
        sys.exit("e-INFRA allows at most 4 parallel requests per account")
    (args.out / ".progress").mkdir(parents=True, exist_ok=True)
    usage = Usage(args.out / "usage.jsonl")
    srcs = sorted(p for p in args.package.glob("*.json") if p.name != "index.json")
    if args.files:
        srcs = [p for p in srcs if p.stem in set(args.files)]
    srcs = [p for p in srcs if not (args.out / p.name).exists()]
    if args.limit:
        srcs = srcs[:args.limit]
    if args.max_batches:                      # test mode: cut each video to its first batches
        global batches
        full = batches
        batches = lambda sents: full(sents)[:args.max_batches]   # noqa: E731
    print(f"{len(srcs)} videos to translate with {args.model}, {args.workers} parallel requests"
          + (", off-peak only" if args.offpeak else ""), flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        for i, src in enumerate(srcs, 1):
            t0 = time.time()
            try:
                vid, nb, missing = translate_video(src, args.out, args.model, pool, usage, key, args.offpeak)
                print(f"[{i}/{len(srcs)}] {vid}: {nb} batches in {time.time() - t0:.0f}s"
                      + (f", {missing} sentences missing" if missing else ""), flush=True)
            except Exception as e:
                print(f"[{i}/{len(srcs)}] FAILED {src.stem}: {e!r}", flush=True)


if __name__ == "__main__":
    main()
