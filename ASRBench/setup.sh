#!/usr/bin/env bash
# Installs everything the benchmark needs. run_all.sh calls this automatically;
# environments that are already finished are skipped, so re-running is cheap.
#
#   bash setup.sh                       # everything
#   ONLY="nemo whisper" bash setup.sh   # just some environments (nemo whisper diarizen vibevoice)
#
# Creates four separate Python environments in envs/ (the models need conflicting
# package versions) and converts the Czech Whisper fine-tune into models/.
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

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ---------------------------------------------------------------- environments
only() { [ -z "${ONLY:-}" ] || [[ " $ONLY " == *" $1 "* ]]; }
mkdir -p envs models
FAILED=()

# build <env>: run setup_<env> unless envs/<env>/.done exists; mark it done on success
build() {
  local e=$1
  only "$e" || return 0
  if [ -f "envs/$e/.done" ]; then echo "=== $e env: already set up"; return 0; fi
  echo "=== $e env"
  if ( set -e; "setup_$e" ); then
    touch "envs/$e/.done"
  else
    echo "!!! $e env failed"; FAILED+=("$e")
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

build whisper
build nemo
build diarizen
build vibevoice

echo
echo "=== GPU check"
for e in nemo whisper diarizen vibevoice; do
  [ -x "envs/$e/bin/python" ] || continue
  printf "%-10s " "$e"
  "envs/$e/bin/python" -c "import torch; print(torch.__version__, 'cuda ok' if torch.cuda.is_available() else 'NO CUDA')"
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "setup finished with failures: ${FAILED[*]} (re-run to retry just those)"
  # the whisper env is needed for cutting clips and scoring
  [[ " ${FAILED[*]} " == *" whisper "* ]] && exit 1
fi
echo "setup done"
