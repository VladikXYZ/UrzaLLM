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

LONG = 15 * 60  # seconds; the full benchmark is meant for 5-10 minute videos


def convert(src, out_dir="audio"):
    """Return (wav path, seconds, reused?) for a video/audio file."""
    src, out_dir = Path(src), Path(out_dir)
    if not src.is_file():
        raise FileNotFoundError(f"no such file: {src}")
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / f"{src.stem}.wav"
    reused = wav.exists() and wav.stat().st_mtime >= src.stat().st_mtime
    if not reused:
        subprocess.run([
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
            "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(wav),
        ], check=True)
    with wave.open(str(wav)) as w:
        secs = w.getnframes() / w.getframerate()
    return wav, secs, reused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="local video or audio file")
    ap.add_argument("--out", default="audio", help="output folder")
    args = ap.parse_args()
    try:
        wav, secs, reused = convert(args.source, args.out)
    except FileNotFoundError as e:
        sys.exit(str(e))
    if reused:
        print("already converted, reusing it")
    if secs > LONG:
        print(f"note: {Path(args.source).name} is {secs / 60:.0f} minutes long; the benchmark is meant "
              f"for 5-10 minutes, so this will be slow and VibeVoice may run out of GPU memory")
    print(f"wav: {wav}")


if __name__ == "__main__":
    main()
