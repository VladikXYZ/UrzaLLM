#!/usr/bin/env bash
# Whisper large-v3 vs. large-v3-turbo, fully automatic. One command:
#
#   ./compare_whisper.sh lecture1.webm lecture2.mp4 ...
#   ./compare_whisper.sh path/to/folder
#
# Installs only what Whisper needs (first time), transcribes every file with both
# models, and writes out/whisper_compare/report.md + disagreements.csv.
# FORCE=1 ./compare_whisper.sh ...   transcribes again even if outputs exist.
set -uo pipefail

[ $# -ge 1 ] || { echo "usage: ./compare_whisper.sh VIDEO_OR_FOLDER [MORE ...]"; exit 1; }
ARGS=()
for x in "$@"; do
  p=$(realpath -- "$x" 2>/dev/null) && [ -e "$p" ] || { echo "no such file or folder: $x"; exit 1; }
  ARGS+=("$p")
done
cd "$(dirname "$0")"

echo "== setup"
ONLY=whisper bash setup.sh || { echo "setup failed, see the messages above"; exit 1; }

# faster-whisper needs the cuDNN / cuBLAS libraries that came with PyTorch
WLIB=$(envs/whisper/bin/python -c "
import importlib
ps = []
for m in ('nvidia.cudnn.lib', 'nvidia.cublas.lib'):
    try: ps.append(list(importlib.import_module(m).__path__)[0])
    except Exception: pass
print(':'.join(ps))" 2>/dev/null)

mkdir -p out/whisper_compare
LOG=out/whisper_compare/run.log
echo "== comparing (full output in $LOG)"
EXTRA=()
[ -n "${FORCE:-}" ] && EXTRA+=(--force)
# progress bars and status lines still reach the terminal (via /dev/tty)
env LD_LIBRARY_PATH="$WLIB:${LD_LIBRARY_PATH:-}" \
  envs/whisper/bin/python whisper_compare.py "${ARGS[@]}" "${EXTRA[@]}" >"$LOG" 2>&1
status=$?
if [ "$status" -ne 0 ]; then
  echo "FAILED; last lines of $LOG:"; tail -n 15 "$LOG" | sed 's/^/   /'; exit "$status"
fi
echo
cat out/whisper_compare/report.md
