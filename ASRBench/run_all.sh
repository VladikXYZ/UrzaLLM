#!/usr/bin/env bash
# The only script to run. Installs everything (first time), runs all approaches
# on a lecture video, and scores them once you've made the reference.
#
#   ./run_all.sh lecture.mp4
#
# Re-running the same command skips everything already done, so after you've
# written out/<name>/reference.txt, run it again and it only does the scoring.
# FORCE=1 ./run_all.sh ...   re-runs all approaches.
#
# The terminal shows one line per step with progress bars; the full output of each
# step (library warnings etc.) is in out/<name>/logs/. A failing step doesn't stop
# the others.
set -uo pipefail

USAGE='usage: ./run_all.sh VIDEO_FILE    e.g. ./run_all.sh lecture.mp4'
[ $# -eq 1 ] || { echo "$USAGE"; exit 1; }
# resolve the video path before changing into the script's folder
VIDEO=$(realpath -- "$1" 2>/dev/null) && [ -f "$VIDEO" ] || { echo "no such file: $1"; exit 1; }
cd "$(dirname "$0")"

echo "== setup"
bash setup.sh || { echo "setup failed, see the messages above"; exit 1; }
export PATH="$HOME/.local/bin:$PATH"

echo "== reading audio from $(basename "$VIDEO")"
WAV=$(envs/whisper/bin/python to_wav.py "$VIDEO" | tee /dev/stderr | tail -n1 | sed 's/^wav: //')
[ -f "$WAV" ] || { echo "could not read the audio from $VIDEO"; exit 1; }

NAME=$(basename "$WAV" .wav)
OUT=out/$NAME
LOG=$OUT/logs
mkdir -p "$LOG"
DZ=$OUT/diarizen_diar_only.json
SF=$OUT/sortformer_diar_only.json

# less GPU memory fragmentation on long files (not understood by DiariZen's old torch)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# faster-whisper needs the cuDNN / cuBLAS libraries that came with PyTorch
WLIB=$(envs/whisper/bin/python -c "
import importlib
ps = []
for m in ('nvidia.cudnn.lib', 'nvidia.cublas.lib'):
    try: ps.append(list(importlib.import_module(m).__path__)[0])
    except Exception: pass
print(':'.join(ps))" 2>/dev/null)
WHISPER="env LD_LIBRARY_PATH=$WLIB:${LD_LIBRARY_PATH:-} envs/whisper/bin/python"

STEPS=8
STEP=0
STATUS=()
T_ALL=$SECONDS
# run <step name> <output it produces> <command...>
run() {
  local name=$1 result=$OUT/$2; shift 2
  STEP=$((STEP + 1))
  if [ -f "$result" ] && [ -z "${FORCE:-}" ]; then
    echo "== [$STEP/$STEPS] $name: already done"
    STATUS+=("done  $name")
    return
  fi
  echo "== [$STEP/$STEPS] $name"
  local t0=$SECONDS
  if "$@" >"$LOG/$name.log" 2>&1 && [ -f "$result" ]; then
    echo "   OK ($((SECONDS - t0)) s)"
    STATUS+=("OK    $name  ($((SECONDS - t0)) s)")
  else
    echo "   FAILED after $((SECONDS - t0)) s; last lines of $LOG/$name.log:"
    grep -v '^\s*$' "$LOG/$name.log" | tail -n 6 | sed 's/^/      /'
    STATUS+=("FAIL  $name  (see $LOG/$name.log)")
  fi
}

run "speaker detection: DiariZen"   diarizen_diar_only.json \
    env -u PYTORCH_CUDA_ALLOC_CONF envs/diarizen/bin/python run_diarizen.py "$WAV"
run "#1 Sortformer + Canary"        sortformer_canary_turns.json \
    envs/nemo/bin/python run_nemo.py "$WAV" --asr canary --chunking turns
run "#2 DiariZen + Canary"          diarizen_canary_turns.json \
    envs/nemo/bin/python run_nemo.py "$WAV" --asr canary --chunking turns --diar-json "$DZ"
run "#3 Czech Whisper + DiariZen"   whisper-cs_diarizen.json \
    $WHISPER run_whisper.py "$WAV" --model models/whisper-cs-ct2 --label whisper-cs --diar-json "$DZ"
run "#4 Whisper large-v3 + Sortformer" whisper-large-v3_sortformer.json \
    $WHISPER run_whisper.py "$WAV" --model large-v3 --diar-json "$SF"
run "#6 Sortformer + Qwen3-ASR"     sortformer_qwen3asr_turns.json \
    envs/qwen/bin/python run_qwen.py "$WAV" --diar-json "$SF"
run "#7 DiariZen + Qwen3-ASR"       diarizen_qwen3asr_turns.json \
    envs/qwen/bin/python run_qwen.py "$WAV" --diar-json "$DZ"
run "#5 VibeVoice (slowest)"        vibevoice-4bit.json \
    envs/vibevoice/bin/python run_vibevoice.py "$WAV" --quant 4bit

echo
echo "== summary ($(( (SECONDS - T_ALL) / 60 )) min)"
printf '   %s\n' "${STATUS[@]}"

if [ -f "$OUT/reference.txt" ]; then
  echo; echo "== scores"
  envs/whisper/bin/python score.py "$OUT/reference.txt" "$OUT"/*.json
else
  START=""
  for f in sortformer_canary_turns diarizen_canary_turns sortformer_qwen3asr_turns whisper-cs_diarizen whisper-large-v3_sortformer vibevoice-4bit; do
    [ -f "$OUT/$f.txt" ] && { START=$OUT/$f.txt; break; }
  done
  [ -n "$START" ] || { echo "no approach produced output; check the logs in $LOG"; exit 1; }
  [ -f "$OUT/reference.txt.todo" ] || cp "$START" "$OUT/reference.txt.todo"  # never overwrite edits
  cat <<EOF

Next step: the hand-corrected reference.
  1. Open $OUT/reference.txt.todo. Pick one continuous 5-10 minute part (ideally with
     audience questions), delete all lines before and after it, and correct the rest
     while listening to $WAV: words exactly as spoken, speaker names like URZA / Q1,
     times at speaker changes. Only that part is scored.
  2. Rename it to $OUT/reference.txt
  3. Run the same command again: it skips the finished approaches and prints the scores.
EOF
fi
