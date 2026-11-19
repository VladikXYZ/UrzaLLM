"""Diarize every channel video with Sortformer v2.1 and find which speaker is Urza (stage 2 + 3).

Two steps, each in its own environment (see run_channel.pbs), over the same share of the videos:

    python diarize_channel.py sortformer --audio-dir AUDIO --wav-dir SCRATCH --out OUT [--shard K/N]
    python diarize_channel.py identify   --audio-dir AUDIO --wav-dir SCRATCH --out OUT --urza urza_voice.npy [--shard K/N]

sortformer  decodes each video to 16 kHz mono WAV (kept in --wav-dir for the second step) and runs NVIDIA's
            streaming Sortformer v2.1 with the "high latency" settings that won the benchmark
            -> OUT/<video>.rttm    who speaks when (speaker_0..3: at most 4 speakers per video)
identify    turns every speech segment of >= 1 s into a voiceprint (ECAPA, SpeechBrain; 3 s windows averaged,
            exactly as score.py does), averages them per speaker and compares with Urza's reference voice
            (cosine similarity) -> OUT/<video>.json  speakers, their similarity to Urza, which are Urza, how
                                                     much of the video is Urza, every segment with its own
                                                     similarity
                                   OUT/<video>.npz   the segment voiceprints (float16), so the Urza decision
                                                     can be redone later (other references, other threshold)
                                                     without the GPU

A speaker is Urza when its similarity is >= --tau (0.45, the benchmark's threshold). Unlike the benchmark,
the closest speaker is NOT taken as Urza when nobody passes: the channel has videos without him (guest
talks), and those must come out with no Urza. In the benchmark clips Urza's speaker scored 0.54-0.73 and the
others at most 0.39 (Sortformer v2).

Finished videos are skipped (an .rttm in step one, a .json in step two), so a job can just be resubmitted.
"""
import argparse
import json
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SR = 16000
AUDIO_EXTS = {".webm", ".m4a", ".mp3", ".opus", ".ogg", ".wav", ".mp4", ".mkv"}
REPO = "nvidia/diar_streaming_sortformer_4spk-v2.1"
MIN_SEG, WIN = 1.0, 3.0           # as score.py: segments >= 1 s, 3 s windows, a piece needs >= 1 s


def videos(args):
    files = sorted(p for p in args.audio_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    if args.files:
        files = [p for p in files if p.stem in set(args.files) or p.name in set(args.files)]
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        files = [p for p in files if zlib.crc32(p.stem.encode()) % n == k - 1]
    return files[:args.limit] if args.limit else files


def decode(src: Path) -> np.ndarray:
    """int16 mono 16 kHz via PyAV."""
    import av
    parts = []
    with av.open(str(src)) as c:
        rs = av.AudioResampler(format="s16", layout="mono", rate=SR)
        for fr in c.decode(c.streams.audio[0]):
            parts += [f.to_ndarray().reshape(-1) for f in rs.resample(fr)]
        parts += [f.to_ndarray().reshape(-1) for f in rs.resample(None)]
    return np.concatenate(parts) if parts else np.zeros(0, np.int16)


def read_rttm(p: Path):
    out = []
    for ln in p.read_text().splitlines():
        f = ln.split()
        if f and f[0] == "SPEAKER" and float(f[4]) > 0:
            out.append((float(f[3]), float(f[3]) + float(f[4]), f[7]))
    return out


# ------------------------------------------------------------------------------------------- step one
MAX_PART = 3 * 3600   # Sortformer holds a whole recording in GPU memory: 6 h needed ~15 GB, 12 h > 44 GB


def diarize_wav(m, wav, sf, torch):
    """[(start, end, label)] for the whole file. Recordings longer than MAX_PART are diarized in parts;
    labels then get the part number (p1_speaker_0, ...), since speaker numbers are per part."""
    def run(path):
        with torch.inference_mode():
            out = m.diarize(audio=[str(path)], batch_size=1)[0]
        return [(float(s.split()[0]), float(s.split()[1]), s.split()[2]) for s in out]
    n = sf.info(wav).frames
    if n <= MAX_PART * SR:
        return run(wav)
    segs = []
    for k, a in enumerate(range(0, n, MAX_PART * SR), 1):
        part = wav.with_name(f"{wav.stem}.part{k}.wav")
        sf.write(part, sf.read(wav, start=a, stop=min(n, a + MAX_PART * SR), dtype="int16")[0], SR, subtype="PCM_16")
        try:
            segs += [(s + a / SR, e + a / SR, f"p{k}_{l}") for s, e, l in run(part)]
        finally:
            part.unlink(missing_ok=True)
        torch.cuda.empty_cache()
    return segs


def step_sortformer(args):
    import soundfile as sf
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel
    todo = [p for p in videos(args) if not (args.out / f"{p.stem}.rttm").exists()]
    need_wav = [p for p in videos(args) if not (args.out / f"{p.stem}.json").exists()]
    print(f"sortformer: {len(todo)} to diarize, {len(need_wav)} need audio for identify", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    args.wav_dir.mkdir(parents=True, exist_ok=True)
    m = None
    for i, src in enumerate(need_wav, 1):
        wav = args.wav_dir / f"{src.stem}.wav"
        try:
            t0 = time.time()
            if not wav.exists():
                sf.write(wav, decode(src), SR, subtype="PCM_16")
            t_dec = time.time() - t0
            rttm = args.out / f"{src.stem}.rttm"
            if rttm.exists():
                continue
            if m is None:
                m = SortformerEncLabelModel.from_pretrained(REPO, map_location="cuda").eval()
                sm = m.sortformer_modules   # "high latency" setting from the model card, as in the benchmark
                sm.chunk_len, sm.chunk_right_context, sm.fifo_len, sm.spkcache_update_period, sm.spkcache_len = 340, 40, 40, 300, 188
                sm._check_streaming_parameters()
            t1 = time.time()
            if any(ch in str(wav) for ch in "[]*?"):
                raise ValueError(f"{wav}: NeMo reads [ ] * ? in a path as a pattern; use a --wav-dir without them")
            segs = diarize_wav(m, wav, sf, torch)
            tmp = rttm.with_suffix(".tmp")
            tmp.write_text("".join(f"SPEAKER {src.stem} 1 {s:.3f} {e - s:.3f} <NA> <NA> {l} <NA> <NA>\n"
                                   for s, e, l in segs if e > s))
            tmp.replace(rttm)
            dur = sf.info(wav).duration
            print(f"[{i}/{len(need_wav)}] {src.stem}: {dur / 60:.0f} min, {len({l for *_, l in segs})} speakers, "
                  f"decode {t_dec:.0f}s, sortformer {time.time() - t1:.0f}s", flush=True)
        except Exception as e:
            print(f"[{i}/{len(need_wav)}] FAILED {src.stem}: {e!r}", flush=True)


# ------------------------------------------------------------------------------------------- step two
class Voice:
    def __init__(self):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        self.torch = torch
        self.enc = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cuda:0"})

    def pieces(self, y, segs):
        """score.py's windows: per segment >= 1 s, 3 s steps, pieces of >= 1 s. -> list of (segment index, audio)."""
        out = []
        for k, (s, e) in enumerate(segs):
            if e - s < MIN_SEG:
                continue
            for t in np.arange(s, e - 1.0, WIN):
                c = y[int(t * SR):int(min(t + WIN, e) * SR)]
                if len(c) >= SR:
                    out.append((k, c))
        return out

    def embed(self, chunks, batch=64):
        """unit voiceprints of the audio pieces, batched (padded, with relative lengths)."""
        vs = []
        with self.torch.inference_mode():
            for i in range(0, len(chunks), batch):
                b = chunks[i:i + batch]
                L = max(len(c) for c in b)
                x = np.zeros((len(b), L), np.float32)
                for j, c in enumerate(b):
                    x[j, :len(c)] = c
                lens = self.torch.tensor([len(c) / L for c in b])
                v = self.enc.encode_batch(self.torch.from_numpy(x).cuda(), lens.cuda()).squeeze(1).cpu().numpy()
                vs.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        return np.concatenate(vs) if vs else np.zeros((0, 192), np.float32)


def unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


def step_identify(args):
    import soundfile as sf
    urza = unit(np.load(args.urza).astype(np.float32))
    todo = [p for p in videos(args) if (args.out / f"{p.stem}.rttm").exists()
            and not (args.out / f"{p.stem}.json").exists()]
    print(f"identify: {len(todo)} video(s)", flush=True)
    voice = None
    for i, src in enumerate(todo, 1):
        try:
            t0 = time.time()
            wav = args.wav_dir / f"{src.stem}.wav"
            y = sf.read(wav, dtype="float32")[0] if wav.exists() else decode(src).astype(np.float32) / 32768
            dur = len(y) / SR
            segs = read_rttm(args.out / f"{src.stem}.rttm")
            voice = voice or Voice()
            pcs = voice.pieces(y, [(s, e) for s, e, _ in segs])
            E = voice.embed([c for _, c in pcs])
            owner = np.array([k for k, _ in pcs], int)
            seg_v = np.zeros((len(segs), E.shape[1] if len(E) else 192), np.float32)
            has = np.zeros(len(segs), bool)
            for k in np.unique(owner):
                seg_v[k], has[k] = unit(E[owner == k].mean(0)), True
            speakers = {}
            for lab in sorted({l for *_, l in segs}):
                idx = [k for k, (_, _, l) in enumerate(segs) if l == lab]
                secs = sum(segs[k][1] - segs[k][0] for k in idx)
                win = E[np.isin(owner, idx)]
                # score.py: a speaker needs >= 3 s of usable pieces; its voice = mean of the piece voiceprints
                usable = sum(segs[k][1] - segs[k][0] for k in idx if segs[k][1] - segs[k][0] >= MIN_SEG)
                sim = float(unit(win.mean(0)) @ urza) if len(win) and usable >= 3 else None
                speakers[lab] = {"seconds": round(secs, 1), "segments": len(idx),
                                 "sim_urza": None if sim is None else round(sim, 3)}
            urza_spk = [l for l, v in speakers.items() if v["sim_urza"] is not None and v["sim_urza"] >= args.tau]
            urza_sec = sum(e - s for s, e, l in segs if l in urza_spk)
            speech = sum(e - s for s, e, _ in segs)
            out = {"video_id": src.stem, "model": REPO, "voice_model": "speechbrain/spkrec-ecapa-voxceleb",
                   "urza_reference": args.urza.name, "tau": args.tau, "duration": round(dur, 2),
                   "speech_seconds": round(speech, 1), "urza_speakers": urza_spk, "urza_seconds": round(urza_sec, 1),
                   "urza_share_of_speech": round(urza_sec / speech, 3) if speech else 0.0,
                   "best_sim_urza": max((v["sim_urza"] for v in speakers.values() if v["sim_urza"] is not None), default=None),
                   "speakers": speakers,
                   "segments": [[round(s, 2), round(e, 2), l, round(float(seg_v[k] @ urza), 3) if has[k] else None]
                                for k, (s, e, l) in enumerate(segs)],
                   "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            np.savez_compressed(args.out / f"{src.stem}.npz", segment_voice=seg_v.astype(np.float16), has_voice=has)
            tmp = args.out / f"{src.stem}.json.tmp"
            tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
            tmp.replace(args.out / f"{src.stem}.json")
            print(f"[{i}/{len(todo)}] {src.stem}: Urza {urza_spk or '-'} {out['urza_share_of_speech']:.0%} of speech, "
                  f"sims {[v['sim_urza'] for v in speakers.values()]}, {time.time() - t0:.0f}s", flush=True)
            if wav.exists():
                wav.unlink()       # scratch space
        except Exception as e:
            print(f"[{i}/{len(todo)}] FAILED {src.stem}: {e!r}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["sortformer", "identify"])
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--wav-dir", type=Path, required=True, help="decoded WAVs, shared by the two steps (scratch)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--urza", type=Path, help="Urza's reference voiceprint (.npy), for identify")
    ap.add_argument("--tau", type=float, default=0.45)
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", help="K/N: the K-th of N disjoint parts")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.step == "sortformer":
        step_sortformer(args)
    else:
        if not args.urza:
            ap.error("identify needs --urza")
        step_identify(args)


if __name__ == "__main__":
    main()
