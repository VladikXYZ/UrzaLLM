#!/usr/bin/env bash
# Installs everything the benchmark needs. run_all.sh calls this automatically;
# environments that are already finished are skipped, so re-running is cheap.
#
#   bash setup.sh                       # everything
#   ONLY="nemo whisper" bash setup.sh   # just some environments (nemo whisper diarizen vibevoice qwen)
#
# Creates five separate Python environments in envs/ (the models need conflicting
# package versions) and converts the Czech Whisper fine-tune into models/.
# Install output goes to envs/<env>.setup.log; the terminal shows one line per environment.
# One environment failing doesn't stop the others; the approaches needing it fail later.
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$PWD

# PyTorch build. cu128 needs a recent NVIDIA driver (570+); with an older driver use cu124.
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}

# ---------------------------------------------------------------- system tools
command -v nvidia-smi >/dev/null || {
  echo "No NVIDIA driver found. Install it first (on WSL2: the normal Windows NVIDIA driver)."; exit 1; }

missing=()
command -v ffmpeg >/dev/null || missing+=(ffmpeg)
command -v git >/dev/null || missing+=(git)
command -v gcc >/dev/null || missing+=(build-essential)
command -v curl >/dev/null || missing+=(curl)
if [ ${#missing[@]} -gt 0 ]; then
  command -v apt-get >/dev/null || { echo "Please install: ${missing[*]}"; exit 1; }
  echo "=== installing ${missing[*]} (asks for your password)"
  sudo apt-get update && sudo apt-get install -y "${missing[@]}" || exit 1
fi

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  echo "=== installing uv (Python environment manager, from astral.sh)"
  curl -LsSf https://astral.sh/uv/install.sh | sh || exit 1
fi

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"

# ---------------------------------------------------------------- environments
only() { [ -z "${ONLY:-}" ] || [[ " $ONLY " == *" $1 "* ]]; }
mkdir -p envs models
FAILED=()
BUILT=()

# build <env>: run setup_<env> unless envs/<env>/.done exists; mark it done on success.
# Output goes to a log; the terminal shows a running timer.
build() {
  local e=$1 log=envs/$e.setup.log
  only "$e" || return 0
  [ -f "envs/$e/.done" ] && return 0
  ( set -e; "setup_$e" ) >"$log" 2>&1 &
  local pid=$! t0=$SECONDS
  while kill -0 $pid 2>/dev/null; do
    printf "\r  installing %-10s %4ds  (log: %s)" "$e" $((SECONDS - t0)) "$log"
    sleep 1
  done
  if wait $pid; then
    touch "envs/$e/.done"
    BUILT+=("$e")
    printf "\r  installed  %-10s %4ds%-40s\n" "$e" $((SECONDS - t0)) ""
  else
    printf "\r  FAILED     %-10s %4ds  (see %s)\n" "$e" $((SECONDS - t0)) "$log"
    tail -n 5 "$log" | sed 's/^/      /'
    FAILED+=("$e")
  fi
}

setup_nemo() {  # Sortformer, Canary
  uv venv --python 3.11 envs/nemo
  uv pip install -p envs/nemo/bin/python torch torchaudio --index-url "$TORCH_INDEX"
  uv pip install -p envs/nemo/bin/python -r requirements-nemo.txt
}

setup_whisper() {  # faster-whisper, clip cutting, scoring
  uv venv --python 3.11 envs/whisper
  uv pip install -p envs/whisper/bin/python torch torchaudio --index-url "$TORCH_INDEX"
  uv pip install -p envs/whisper/bin/python -r requirements-whisper.txt
  echo "--- converting the Czech Whisper fine-tune"
  envs/whisper/bin/python convert_whisper.py --out models/whisper-cs-ct2
}

setup_diarizen() {  # Python 3.10 + torch 2.1.1, as DiariZen requires
  local SRC=envs/DiariZen-src PY=$ROOT/envs/diarizen/bin/python
  [ -d "$SRC" ] || git clone --recurse-submodules https://github.com/BUTSpeechFIT/DiariZen.git "$SRC"
  uv venv --python 3.10 envs/diarizen
  uv pip install -p "$PY" torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
      --index-url https://download.pytorch.org/whl/cu121
  (cd "$SRC" && uv pip install -p "$PY" -r requirements.txt -c constraints.txt \
      && uv pip install -p "$PY" -e . -c constraints.txt)
  (cd "$SRC/pyannote-audio" && uv pip install -p "$PY" -e ".[dev,testing]" -c ../constraints.txt)
  uv pip install -p "$PY" soundfile -c "$SRC/constraints.txt"
}

setup_vibevoice() {  # transformers >= 5.3, bitsandbytes
  uv venv --python 3.11 envs/vibevoice
  uv pip install -p envs/vibevoice/bin/python torch torchaudio --index-url "$TORCH_INDEX"
  uv pip install -p envs/vibevoice/bin/python -r requirements-vibevoice.txt
}

setup_qwen() {  # Qwen3-ASR (Python 3.12 as Qwen recommends)
  uv venv --python 3.12 envs/qwen
  uv pip install -p envs/qwen/bin/python torch torchaudio --index-url "$TORCH_INDEX"
  uv pip install -p envs/qwen/bin/python -r requirements-qwen.txt
}

build whisper
build nemo
build diarizen
build vibevoice
build qwen

for e in "${BUILT[@]}"; do  # check that freshly built environments see the GPU
  "envs/$e/bin/python" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || echo "  WARNING: the $e environment cannot use the GPU (CUDA not available)"
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "  setup failed for: ${FAILED[*]} (the next run retries just those)"
  # the whisper env is needed for reading the video and scoring
  [[ " ${FAILED[*]} " == *" whisper "* ]] && exit 1
fi
exit 0
