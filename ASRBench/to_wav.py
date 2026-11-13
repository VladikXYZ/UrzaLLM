"""Convert a lecture video (or audio) file into the 16 kHz mono WAV all runners use.
The whole file is converted; nothing is cut.

  python to_wav.py lecture.mp4        # -> audio/lecture.wav

Any format ffmpeg reads works (mp4, mkv, webm, mov, mp3, m4a, wav, ...).
"""
import argparse
import subprocess
import sys
import wave
from pathlib import Path

from common import SR

LONG = 15 * 60  # seconds; the benchmark is meant for 5-10 minute videos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="local video or audio file")
    ap.add_argument("--out", default="audio", help="output folder")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.is_file():
        sys.exit(f"no such file: {src}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / f"{src.stem}.wav"

    if wav.exists() and wav.stat().st_mtime >= src.stat().st_mtime:
        print(f"already converted, reusing it\nwav: {wav}")
        return

    subprocess.run([
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(wav),
    ], check=True)

    with wave.open(str(wav)) as w:
        secs = w.getnframes() / w.getframerate()
    if secs > LONG:
        print(f"note: {src.name} is {secs / 60:.0f} minutes long; the benchmark is meant for "
              f"5-10 minutes, so this will be slow and VibeVoice may run out of GPU memory")
    print(f"wav: {wav}")


if __name__ == "__main__":
    main()
