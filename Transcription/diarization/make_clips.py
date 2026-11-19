"""Cut the test clips listed in clips.tsv from the channel audio into 16 kHz mono WAV.

    python make_clips.py --audio-dir /storage/brno2/home/vladikxyz/scraping/audio --out clips

duration_s 0 = the whole video. Needs PyAV (`av`), which decodes the .webm/.m4a/.mp4 audio without ffmpeg.
"""
import argparse, csv, pathlib
import av, numpy as np, soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--audio-dir", required=True)
ap.add_argument("--out", default="clips")
ap.add_argument("--tsv", default=str(pathlib.Path(__file__).with_name("clips.tsv")))
a = ap.parse_args()
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
SR = 16000

for row in csv.DictReader(open(a.tsv, encoding="utf-8"), delimiter="\t"):
    dst = out / f"{row['clip']}.wav"
    if dst.exists():
        continue
    src = next(pathlib.Path(a.audio_dir).glob(f"{row['video_id']}.*"))
    start, dur = float(row["start_s"]), float(row["duration_s"])
    with av.open(str(src)) as c:
        st = c.streams.audio[0]
        if start:
            c.seek(int(start / st.time_base), stream=st)
        res = av.AudioResampler(format="flt", layout="mono", rate=SR)
        chunks = []
        for frame in c.decode(st):
            for f in res.resample(frame):
                t = float(f.pts * f.time_base) if f.pts is not None else None
                x = f.to_ndarray().reshape(-1)
                if t is not None and t + len(x) / SR < start:
                    continue
                if t is not None and t < start:      # first frame after the seek
                    x = x[int((start - t) * SR):]
                chunks.append(x)
            if dur and sum(map(len, chunks)) >= dur * SR:
                break
    y = np.concatenate(chunks)
    if dur:
        y = y[:int(dur * SR)]
    sf.write(dst, y, SR, subtype="PCM_16")
    print(dst.name, round(len(y) / SR, 1), "s")
