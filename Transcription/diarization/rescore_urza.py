"""Decide which speakers are Urza, with references for many recording conditions (no GPU needed).

    python Transcription/diarization/rescore_urza.py [--dir ~/urza_diar/channel] [--rebuild]

Input: the output of diarize_channel.py for every video (<video>.json + <video>.npz with the voiceprint of
every speech segment). The first pass compared each speaker with ONE reference (Urza's solo livestream,
threshold 0.45); his voice from a hall, an online call or in English then scored only ~0.39-0.45 and
whole videos with him came out "no Urza".

Here:
 1. seeds: segments of >= 3 s scoring >= 0.6 against the first reference: certainly Urza (53k segments,
    207 h, from 762 videos), at most 60 per video so no video dominates
 2. references: the seeds grouped into K=12 clusters (spherical k-means) = Urza in 12 kinds of recording
    (urza_refs.npy next to this script; rebuilt with --rebuild)
 3. each speaker's voice (duration-weighted mean of its segment voiceprints) scored by the closest reference
    (`sim`, cosine)
 4. score normalisation (s-norm): `sim` minus the mean, divided by the std, of the speaker's 100 highest
    similarities to a cohort of clearly-other voices (speakers of >= 30 s scoring <= 0.30 against the first
    reference in videos where Urza is identified; urza_cohort.npy). A voice that resembles many people gets
    marked down, one that resembles specifically Urza stands out (`z`)
 5. Urza = z >= --z (6)

Checked on 18 Nov, with half of the clearly-other speakers held out of the cohort:
                              sim (12 references)          z (normalised)
  certain Urza (621)          1%: 0.73  5%: 0.82           1%: 8.7  5%: 10.8
  clearly other (405 held out) 99%: 0.45  max: 0.46         99%: 1.7  max: 4.6
Known cases: Urza asking from the audience (Nm0TzaQDfu0, confirmed by ear) z 8.2, "Urza and Lawrence Reed"
10.4, Deloitte online conference 9.2, HCPP17 in English 9.8; Tomáš Roud's talk 1.2, the O. T. lecture 1.6.
With one reference and 0.45, all four Urza cases had failed.

Answers from review_borderline.py (urza_review.json: speakers checked by ear) override the speaker score.

Final decision per segment (urza_segments): Sortformer sometimes merges two people into one speaker, so a
segment's own voice (its z, normalised the same way) can overrule its speaker when it is clear:
  speaker is Urza (z >= 6 or answered "urza")      every segment except those with z < 2 (clearly not him)
  speaker answered "other"                          none
  speaker is not Urza, or answered "mixed"          only segments with z >= 6 (clearly him)
  segments < 1 s (no voiceprint) and z 2-6          follow the speaker
Per segment, z >= 6 keeps 97.2% of certain-Urza time and takes 0.5% of held-out other speakers' time; on
the 28 clips of "mixed" speakers the reviewer described clip by clip, it agreed on all 28. Channel-wide
this removes ~12 h of other people from Urza speakers and recovers ~7 h of Urza hidden in other speakers
(e.g. "Urza v prvním dílu Nejlepšího podcastu": 0% -> 64 min).

Writes <dir>/urza_videos.json (per video: Urza speakers, seconds, share, every speaker's scores) and
<dir>/urza_borderline.md (speakers with z between 4 and 8, with a YouTube link to their longest segment,
to check by ear).
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SEED_SIM, SEED_MIN_DUR, SEEDS_PER_VIDEO, K = 0.6, 3.0, 60, 12
SEG_HI, SEG_LO = 6.0, 2.0     # a segment's own z: >= HI clearly Urza, < LO clearly not (see main)


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-9)


def load(d: Path):
    for f in sorted(glob.glob(str(d / "*.json"))):
        if f.endswith(("urza_videos.json",)):
            continue
        j = json.loads(Path(f).read_text(encoding="utf-8"))
        z = np.load(f[:-5] + ".npz")
        yield j, z["segment_voice"].astype(np.float32), z["has_voice"]


def build_refs(d: Path, rng):
    seeds = []
    for j, E, has in load(d):
        seg = j["segments"]
        ok = [k for k, (s, e, _, sim) in enumerate(seg) if has[k] and sim is not None and sim >= SEED_SIM
              and e - s >= SEED_MIN_DUR]
        if len(ok) > SEEDS_PER_VIDEO:
            ok = list(rng.choice(ok, SEEDS_PER_VIDEO, replace=False))
        seeds += [E[k] for k in ok]
    S = unit(np.array(seeds))
    C = S[rng.choice(len(S), K, replace=False)]
    for _ in range(50):                                       # spherical k-means
        a = np.argmax(S @ C.T, 1)
        C = unit(np.array([S[a == k].mean(0) if (a == k).any() else C[k] for k in range(K)]))
    print(f"references: {K} from {len(S):,} seed segments, cluster sizes {np.bincount(a, minlength=K).tolist()}")
    return C


def speaker_voices(d: Path):
    """[(video_id, label, seconds, first-pass sim, voice)] for every speaker with a usable voice."""
    out = []
    for j, E, has in load(d):
        dur = np.array([e - s for s, e, _, _ in j["segments"]])
        lab = np.array([l for _, _, l, _ in j["segments"]])
        for l, v in j["speakers"].items():
            m = (lab == l) & has
            if m.any() and v["sim_urza"] is not None:
                out.append((j["video_id"], l, v["seconds"], v["sim_urza"], unit((E[m] * dur[m, None]).sum(0))))
    return out


def build_cohort(spk):
    """Voices of speakers that are clearly not Urza: >= 30 s, <= 0.30 against the first reference, in a video
    where another speaker scored >= 0.6 (so Urza is there, and these are the other people)."""
    best = {}
    for v, _, _, s, _ in spk:
        best[v] = max(best.get(v, -1), s)
    coh = [x for v, _, sec, s, x in spk if sec >= 30 and s <= 0.30 and best[v] >= 0.6]
    print(f"cohort: {len(coh)} clearly-other voices")
    return np.array(coh)


def snorm(x, sim, coh, top=100):
    s = coh @ x
    s = np.sort(s[s < 0.999])[-top:]        # not against itself
    return (sim - s.mean()) / s.std()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path.home() / "urza_diar" / "channel")
    ap.add_argument("--refs", type=Path, default=HERE / "urza_refs.npy")
    ap.add_argument("--cohort", type=Path, default=HERE / "urza_cohort.npy")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the references and the cohort")
    ap.add_argument("--z", type=float, default=6.0, help="Urza = normalised score >= this")
    ap.add_argument("--review", type=Path, default=HERE / "urza_review.json",
                    help="answers from review_borderline.py: fixed decisions for those speakers")
    ap.add_argument("--titles", type=Path, default=ROOT / "real_data" / "yt_full.json")
    args = ap.parse_args()
    titles = ({y["id"]: y.get("title_cs") for y in json.loads(args.titles.read_text(encoding="utf-8"))}
              if args.titles.exists() else {})
    if args.rebuild or not args.refs.exists():
        np.save(args.refs, build_refs(args.dir, np.random.default_rng(0)).astype(np.float32))
    C = unit(np.load(args.refs))
    if args.rebuild or not args.cohort.exists():
        np.save(args.cohort, build_cohort(speaker_voices(args.dir)).astype(np.float32))
    coh = unit(np.load(args.cohort))
    review = json.loads(args.review.read_text(encoding="utf-8")) if args.review.exists() else {}
    review = {k: v["label"] for k, v in review.items() if v["label"] in ("urza", "other", "mixed")}

    videos, border = [], []
    for j, E, has in load(args.dir):
        seg = j["segments"]
        dur = np.array([e - s for s, e, _, _ in seg])
        lab = np.array([l for _, _, l, _ in seg])
        speakers = {}
        for l, v in j["speakers"].items():
            m = (lab == l) & has
            sim = z = None
            if m.any() and v["sim_urza"] is not None:
                x = unit((E[m] * dur[m, None]).sum(0))
                sim = float((x @ C.T).max())
                z = float(snorm(x, sim, coh))
            speakers[l] = {"seconds": v["seconds"], "sim_first": v["sim_urza"],
                           "sim": None if sim is None else round(sim, 3), "z": None if z is None else round(z, 2)}
            ans = review.get(f"{j['video_id']}/{l}")
            if ans:
                speakers[l]["review"] = ans
            if z is not None and 4 <= z < 8 and not ans:
                k = max(np.flatnonzero(lab == l), key=lambda i: dur[i])
                border.append((z, j["video_id"], l, v["seconds"], seg[k][0]))
        urza = [l for l, v in speakers.items()
                if v.get("review") == "urza" or (v.get("review") is None and v["z"] is not None and v["z"] >= args.z)]
        # Per segment: the speaker decides by default, a segment's own voice overrules it when it is clear.
        seg_z = np.full(len(seg), np.nan)
        if has.any():
            X = E[has]
            S = np.sort(X @ coh.T, 1)[:, -100:]
            seg_z[has] = ((X @ C.T).max(1) - S.mean(1)) / S.std(1)
        is_urza = np.zeros(len(seg), bool)
        for l, v in speakers.items():
            m = lab == l
            if l in urza:                               # Urza: all but the pieces clearly not him
                is_urza |= m & ~(seg_z < SEG_LO)
            elif v.get("review") == "other":            # checked by ear: not Urza, nothing taken
                continue
            else:                                       # mixed or not Urza: only pieces clearly him
                is_urza |= m & (seg_z >= SEG_HI)
        speech = float(dur.sum())
        usec = float(dur[is_urza].sum())
        spk_sec = float(sum(dur[lab == l].sum() for l in urza))
        videos.append({"video_id": j["video_id"], "title": titles.get(j["video_id"]), "duration": j["duration"],
                       "speech_seconds": round(speech, 1), "urza_speakers": urza,
                       "urza_segments": np.flatnonzero(is_urza).tolist(), "urza_seconds": round(usec, 1),
                       "urza_seconds_by_speaker": round(spk_sec, 1),
                       "urza_share_of_speech": round(usec / speech, 3) if speech else 0.0,
                       "urza_speakers_first_pass": j["urza_speakers"], "speakers": speakers})
    out = {"z_threshold": args.z, "references": args.refs.name, "cohort": args.cohort.name, "videos": videos}
    (args.dir / "urza_videos.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    border.sort()
    lines = ["# Speakers near the Urza threshold (normalised score z between 4 and 8)", "",
             f"Urza = z >= {args.z}. Listen to the link (the speaker's longest segment) and note who it is.", "",
             "| z | video | title | speaker | seconds | listen |", "|---|---|---|---|---|---|"]
    lines += [f"| {s:.2f} | {v} | {(titles.get(v) or '')[:60]} | {l} | {sec:.0f} | "
              f"[{int(t // 60)}:{int(t % 60):02d}](https://www.youtube.com/watch?v={v}&t={int(t)}s) |"
              for s, v, l, sec, t in border]
    (args.dir / "urza_borderline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    hours = lambda x: x / 3600                              # noqa: E731
    share = np.array([v["urza_share_of_speech"] for v in videos])
    print(f"{len(videos)} videos, {hours(sum(v['duration'] for v in videos)):.0f} h, "
          f"speech {hours(sum(v['speech_seconds'] for v in videos)):.0f} h, "
          f"Urza {hours(sum(v['urza_seconds'] for v in videos)):.0f} h "
          f"(by whole speakers: {hours(sum(v['urza_seconds_by_speaker'] for v in videos)):.0f} h)")
    bins = [("no Urza", share == 0), ("< 10%", (share > 0) & (share < .1)), ("10-50%", (share >= .1) & (share < .5)),
            ("50-90%", (share >= .5) & (share < .9)), (">= 90%", share >= .9)]
    print("videos by Urza's share of speech: " + ", ".join(f"{n} {m.sum()}" for n, m in bins))
    changed = [v for v in videos if set(v["urza_speakers"]) != set(v["urza_speakers_first_pass"])]
    print(f"videos whose Urza speakers changed vs. the first pass: {len(changed)}")
    print(f"speakers near the threshold (z 4-8): {len(border)}, {sum(b[3] for b in border) / 3600:.1f} h "
          f"-> {args.dir / 'urza_borderline.md'}")


if __name__ == "__main__":
    main()
