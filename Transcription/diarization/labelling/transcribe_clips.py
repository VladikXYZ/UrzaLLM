"""Word-timestamped transcripts of the test clips (used to build the speaker reference by hand)."""
import json, sys, pathlib
import soundfile as sf
from faster_whisper import WhisperModel

clips = sorted(pathlib.Path(sys.argv[1]).glob("*.wav"))
out = pathlib.Path(sys.argv[2]); out.mkdir(exist_ok=True)
m = WhisperModel("large-v3", device="cuda", compute_type="float16")
for c in clips:
    if c.name.startswith("enroll") or (out / (c.stem + ".json")).exists():
        continue
    audio, _ = sf.read(c, dtype="float32")  # 16 kHz mono already; avoids the PyAV version clash
    segs, info = m.transcribe(audio, language="cs", beam_size=5, word_timestamps=True,
                              vad_filter=True, condition_on_previous_text=False)
    words = []
    for s in segs:
        for w in s.words:
            words.append({"w": w.word.strip(), "s": round(w.start, 2), "e": round(w.end, 2), "p": round(w.probability, 2)})
    json.dump(words, open(out / (c.stem + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(c.stem, len(words), flush=True)
