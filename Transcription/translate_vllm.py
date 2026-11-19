"""Translate the transcript sentences into English with an open model on our own GPU (vLLM, offline batch).

    python translate_vllm.py --model mistralai/Mistral-Small-3.2-24B-Instruct-2506 --package P --out O [--shard K/N]
    python translate_vllm.py --model ... --package P --out O --test VIDEO_ID ... [--test-batches 2]

Uses the same transcript and output formats as translate_einfra.py. Run smaller batches with
--batch-sentences and --batch-words to reduce sentence drift. A malformed response is retried
at temperature 0 and then split until each unresolved request contains a single sentence.

--test   translate only the first --test-batches batches of the given videos, write them to <out> and
         report the speed (output tokens per second over the whole batch run)
--shard  K/N: the K-th of N disjoint parts of the videos (crc32 of the id), one per job-array task
Complete videos from the same model are skipped. An incomplete video is left for a retry.
"""
import argparse
import json
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from translate_einfra import BATCH_CONTEXT, BATCH_SENTENCES, BATCH_WORDS, SYSTEM, batches, prompt  # noqa: E402

MISTRAL_FORMAT = ("mistralai/",)       # Mistral's own checkpoints load with their tokenizer/config format
TRANSLATION_SYSTEM = SYSTEM + """

Translate the text immediately after each ID under TRANSLATE into that ID's own English entry.
If an item is a hesitation, filler or incomplete sentence, give it its own short translation.
Keep every fragment and speaker separate, even when joining them would sound more natural."""


def schema_for(ids):
    return {"type": "object", "required": ["translations"], "additionalProperties": False,
            "properties": {"translations": {"type": "array", "minItems": len(ids), "maxItems": len(ids),
                                            "items": {"type": "object", "required": ["id", "en"],
                                                      "additionalProperties": False,
                                                      "properties": {"id": {"type": "string", "enum": ids},
                                                                     "en": {"type": "string", "minLength": 1}}}}}}


def sampling(ids, words, temperature):
    from vllm import SamplingParams
    kw = dict(temperature=temperature, top_p=0.95, max_tokens=min(12000, 8 * words + 600))
    try:                                       # vLLM >= 0.11
        from vllm.sampling_params import StructuredOutputsParams
        return SamplingParams(structured_outputs=StructuredOutputsParams(json=schema_for(ids)), **kw)
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        return SamplingParams(guided_decoding=GuidedDecodingParams(json=schema_for(ids)), **kw)


def load_llm(model, max_len, quantization=None):
    from vllm import LLM
    kw = dict(model=model, max_model_len=max_len, gpu_memory_utilization=0.90, enable_prefix_caching=True)
    if quantization:                           # e.g. "fp8": 8-bit weights, half the memory of 16-bit
        kw["quantization"] = quantization
    if model.startswith(MISTRAL_FORMAT):
        kw.update(tokenizer_mode="mistral", config_format="mistral", load_format="mistral")
    return LLM(**kw)


def parse_ordered(text, ids):
    """Accept a single nonempty string per expected ID, in the original order."""
    try:
        entries = json.loads(text)["translations"]
        if [entry["id"] for entry in entries] != ids:
            return None
        translations = {entry["id"]: entry["en"].strip() for entry in entries}
    except (ValueError, TypeError, KeyError, AttributeError):
        return None
    return translations if all(translations.values()) else None


def run(llm, jobs, temperature):
    """jobs: [(key, title, ctx, todo)] -> ({key: {id: en}}, [failed keys], output tokens)."""
    convs = [[{"role": "system", "content": TRANSLATION_SYSTEM}, {"role": "user", "content": prompt(title, ctx, todo)}]
             for _, title, ctx, todo in jobs]
    params = [sampling([s["id"] for s in todo], sum(len(s["text"].split()) for s in todo), temperature)
              for _, _, _, todo in jobs]
    outs = llm.chat(convs, params, use_tqdm=False)
    got, failed, ntok = {}, [], 0
    for (key, _, _, todo), o in zip(jobs, outs):
        ntok += len(o.outputs[0].token_ids)
        r = parse_ordered(o.outputs[0].text, [s["id"] for s in todo])
        if r:
            got[key] = r
        else:
            failed.append(key)
    return got, failed, ntok


def translate_all(llm, docs, max_batches=None, *, batch_words=BATCH_WORDS, batch_sentences=BATCH_SENTENCES):
    """{video_id: {sentence id: en}} for the given videos, plus stats."""
    jobs = []
    for vid, doc in docs.items():
        groups = batches(doc["sentences"], max_words=batch_words, max_sentences=batch_sentences)
        for k, (ctx, todo) in enumerate(groups[:max_batches]):
            jobs.append(((vid, k), doc["title"] or vid, ctx, todo))
    t0 = time.time()
    got, failed, ntok = run(llm, jobs, 0.2) if jobs else ({}, [], 0)
    by_key = {j[0]: j for j in jobs}
    if failed:                                  # retry at temperature 0
        g2, failed, n2 = run(llm, [by_key[k] for k in failed], 0.0)
        got.update(g2)
        ntok += n2
    unresolved = 0
    while failed:
        halves = []
        for key in failed:
            _, title, ctx, todo = by_key[key]
            if len(todo) == 1:
                unresolved += 1
                continue
            mid = len(todo) // 2
            left, right = todo[:mid], todo[mid:]
            halves.append(((*key, "a"), title, ctx, left))
            halves.append(((*key, "b"), title, (ctx + left)[-BATCH_CONTEXT:], right))
        if not halves:
            break
        g3, failed, n3 = run(llm, halves, 0.0)
        ntok += n3
        for k, v in g3.items():
            got.setdefault(k[:2], {}).update(v)
        by_key.update({h[0]: h for h in halves})
    res = {vid: {} for vid in docs}
    for (vid, *_), v in got.items():
        res[vid].update(v)
    return res, {"batches": len(jobs), "failed_at_end": unresolved, "output_tokens": ntok,
                 "seconds": round(time.time() - t0, 1)}


def write(doc, en, out_dir, model, *, allow_partial=False):
    out = {k: doc[k] for k in ("video_id", "title", "date", "url", "duration", "n_sentences", "n_to_translate",
                               "words_to_translate")}
    out["model"] = model
    out["sentences"] = [{"id": s["id"], "start": s["start"], "end": s["end"], "speaker": s["speaker"],
                         "language": s["language"], "text_cs": s["text"],
                         "text_en": en.get(s["id"], "") if s["translate"] else s["text"]} for s in doc["sentences"]]
    missing = sum(1 for s in out["sentences"] if not s["text_en"])
    if missing and not allow_partial:
        return missing
    tmp = out_dir / f"{doc['video_id']}.json.tmp"
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(out_dir / f"{doc['video_id']}.json")
    return missing


def is_complete(path, model):
    if not path.is_file():
        return False
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        sentences = doc["sentences"]
        return (
            doc["video_id"] == path.stem and doc["model"] == model
            and len(sentences) == doc["n_sentences"]
            and all(sent["text_en"].strip() for sent in sentences)
        )
    except (ValueError, TypeError, KeyError, AttributeError):
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard")
    ap.add_argument("--test", nargs="*")
    ap.add_argument("--test-batches", type=int, default=2)
    ap.add_argument("--group", type=int, default=40, help="videos per vLLM run (results are written after each)")
    ap.add_argument("--batch-sentences", type=int, default=BATCH_SENTENCES)
    ap.add_argument("--batch-words", type=int, default=BATCH_WORDS)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--quantization", help="e.g. fp8: fits 22-30B models on 40-48 GB GPUs")
    args = ap.parse_args()
    if min(args.group, args.batch_sentences, args.batch_words) < 1:
        ap.error("--group, --batch-sentences and --batch-words must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    srcs = sorted(p for p in args.package.glob("*.json") if p.name != "index.json")
    if args.test:
        srcs = [p for p in srcs if p.stem in set(args.test)]
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        srcs = [p for p in srcs if zlib.crc32(p.stem.encode()) % n == k - 1]
    if not args.test:
        srcs = [p for p in srcs if not is_complete(args.out / p.name, args.model)]
    print(f"{len(srcs)} videos, model {args.model}", flush=True)
    if not srcs:
        return
    t = time.time()
    llm = load_llm(args.model, args.max_len, args.quantization)
    print(f"model loaded in {time.time() - t:.0f}s", flush=True)
    total = {"batches": 0, "output_tokens": 0, "seconds": 0.0, "failed_at_end": 0}
    incomplete = []
    for g in range(0, len(srcs), args.group):
        docs = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in srcs[g:g + args.group]}
        en, st = translate_all(
            llm, docs, args.test_batches if args.test else None,
            batch_words=args.batch_words, batch_sentences=args.batch_sentences,
        )
        for vid, doc in docs.items():
            missing = write(doc, en[vid], args.out, args.model, allow_partial=bool(args.test))
            if missing and not args.test:
                incomplete.append(vid)
                print(f"  {vid}: {missing} sentences without translation; left for retry", flush=True)
        for k in total:
            total[k] += st[k]
        print(f"[{min(g + args.group, len(srcs))}/{len(srcs)}] {st['batches']} batches, {st['output_tokens']:,} output "
              f"tokens in {st['seconds']:.0f}s ({st['output_tokens'] / max(st['seconds'], 1):.0f} tok/s), "
              f"{st['failed_at_end']} batches failed", flush=True)
    print(f"TOTAL {json.dumps(total)} -> {total['output_tokens'] / max(total['seconds'], 1):.0f} output tok/s", flush=True)
    if incomplete:
        raise SystemExit(f"{len(incomplete)} incomplete videos: {' '.join(incomplete)}")


if __name__ == "__main__":
    main()
