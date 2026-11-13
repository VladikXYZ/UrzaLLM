#!/usr/bin/env bash
# The only script to run. Installs everything (first time), runs the five
# approaches on a prepared 5-10 minute lecture video, and scores them once
# you've made the reference.
#
#   bash run_all.sh lecture.mp4
#
# Re-running the same command skips everything already done, so after you've
# written out/<name>/reference.txt, run it again and it only does the scoring.
# FORCE=1 bash run_all.sh ...   re-runs all approaches.
#
# A failing approach doesn't stop the others; each has a log in out/<name>/logs/.
set -uo pipefail

USAGE='usage: bash run_all.sh VIDEO_FILE    e.g. bash run_all.sh lecture.mp4'
[ $# -eq 1 ] || { echo "$USAGE"; exit 1; }
# resolve the video path before changing into the script's folder
VIDEO=$(realpath -- "$1" 2>/dev/null) && [ -f "$VIDEO" ] || { echo "no such file: $1"; exit 1; }
cd "$(dirname "$0")"

bash setup.sh || { echo "setup failed, see the messages above"; exit 1; }
export PATH="$HOME/.local/bin:$PATH"

WAV=$(envs/whisper/bin/python to_wav.py "$VIDEO" | tee /dev/stderr | tail -n1 | sed 's/^wav: //')
[ -f "$WAV" ] || { echo "could not read the audio from $VIDEO"; exit 1; }

NAME=$(basename "$WAV" .wav)
OUT=out/$NAME
LOG=$OUT/logs
mkdir -p "$LOG"
DZ=$OUT/diarizen_diar_only.json

# faster-whisper needs the cuDNN / cuBLAS libraries that came with PyTorch
WLIB=$(envs/whisper/bin/python -c "
import importlib
ps = []
for m in ('nvidia.cudnn.lib', 'nvidia.cublas.lib'):
    try: ps.append(list(importlib.import_module(m).__path__)[0])
    except Exception: pass
print(':'.join(ps))" 2>/dev/null)
WHISPER="env LD_LIBRARY_PATH=$WLIB:${LD_LIBRARY_PATH:-} envs/whisper/bin/python"

STATUS=()
# run <step name> <output it produces> <command...>
run() {
  local name=$1 result=$OUT/$2; shift 2
  if [ -f "$result" ] && [ -z "${FORCE:-}" ]; then
    STATUS+=("done  $name  (already there)"); return
  fi
  echo; echo "=================== $name"
  local t0=$SECONDS
  if "$@" 2>&1 | tee "$LOG/$name.log" && [ -f "$result" ]; then
    STATUS+=("OK    $name  ($((SECONDS - t0)) s)")
  else
    STATUS+=("FAIL  $name  (see $LOG/$name.log)")
  fi
}

run diarizen              diarizen_diar_only.json \
    envs/diarizen/bin/python run_diarizen.py "$WAV"
run 1_sortformer_canary   sortformer_canary_turns.json \
    envs/nemo/bin/python run_nemo.py "$WAV" --asr canary --chunking turns
run 2_diarizen_canary     diarizen_canary_turns.json \
    envs/nemo/bin/python run_nemo.py "$WAV" --asr canary --chunking turns --diar-json "$DZ"
run 3_whisper-cs_diarizen whisper-cs_diarizen.json \
    $WHISPER run_whisper.py "$WAV" --model models/whisper-cs-ct2 --label whisper-cs --diar-json "$DZ"
run 4_whisper_sortformer  whisper-large-v3_sortformer.json \
    $WHISPER run_whisper.py "$WAV" --model large-v3 --diar-json "$OUT/sortformer_diar_only.json"
run 5_vibevoice           vibevoice-4bit.json \
    envs/vibevoice/bin/python run_vibevoice.py "$WAV" --quant 4bit

echo; echo "=================== summary"
printf '%s\n' "${STATUS[@]}"

if [ -f "$OUT/reference.txt" ]; then
  echo; echo "=================== scores"
  envs/whisper/bin/python score.py "$OUT/reference.txt" "$OUT"/*.json
else
  START=""
  for f in sortformer_canary_turns diarizen_canary_turns whisper-cs_diarizen whisper-large-v3_sortformer vibevoice-4bit; do
    [ -f "$OUT/$f.txt" ] && { START=$OUT/$f.txt; break; }
  done
  [ -n "$START" ] || { echo "no approach produced output; check the logs in $LOG"; exit 1; }
  [ -f "$OUT/reference.txt.todo" ] || cp "$START" "$OUT/reference.txt.todo"  # never overwrite edits
  cat <<EOF

Next step: the hand-corrected reference.
  1. Open $OUT/reference.txt.todo and correct it while listening to $WAV
     (words exactly as spoken, speaker names like URZA / Q1, times at speaker changes).
  2. Rename it to $OUT/reference.txt
  3. Run the same command again: it skips the finished approaches and prints the scores.
EOF
fi
