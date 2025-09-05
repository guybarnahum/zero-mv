#!/usr/bin/env bash
set -e

# zero-mv setup:
# - Loads .env if present
# - Picks Python 3.11/3.12
# - Installs minimal system deps
# - Creates & activates venv
# - Installs PyTorch (CPU/MPS/CUDA)
# - Installs project in editable mode
# - Optional HF auth
# - Set venv alias 

# ------------- Auto-yes handling -------------
AUTO_YES=""
for arg in "$@"; do
  case "$arg" in
    --yes|-y) AUTO_YES="--yes" ;;
    cuda_gpu) VARIANT="cuda_gpu" ;;   # allow explicit arg
    cpu) VARIANT="cpu" ;;         # allow explicit arg
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

ask_yes_no() {
  local prompt="$1"
  if [[ -n "$AUTO_YES" ]]; then
    echo "Auto-yes: $prompt -> yes"; return 0
  fi
  read -p "$prompt " -n 1 -r; echo
  [[ $REPLY =~ ^[Yy]$ ]]
}

# ------------- Safer colors + cleanup -------------

if tput setaf 7 >/dev/null 2>&1 && tput dim >/dev/null 2>&1; then
  COLOR_GRAY="$(tput dim)$(tput setaf 7)"   # dim light gray
  COLOR_RESET="$(tput sgr0)"
else
  COLOR_GRAY=$'\033[2m'                     # ANSI dim
  COLOR_RESET=$'\033[0m'
fi

cleanup_render() { printf '\r\033[K%s' "${COLOR_RESET}"; tput cnorm 2>/dev/null || true; }
trap cleanup_render EXIT INT TERM

# ------------- run_and_log -------------
run_and_log() {
  local log_file; log_file=$(mktemp)
  local description="$1"; shift
  tput civis 2>/dev/null || true
  local prev_render=""; local cols; cols=$(tput cols 2>/dev/null || echo 120)
  (
    frames=( '⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏' ); i=0
    while :; do
      local last_line=""
      if [[ -s "$log_file" ]]; then
        last_line=$(tail -n 1 "$log_file" | sed -E 's/\x1B\[[0-9;?]*[ -/]*[@-~]//g')
      fi
      local plain_prefix="${frames[i]} ${description} : "
      local plain="${plain_prefix}${last_line}"
      (( ${#plain} > cols )) && plain="${plain:0:cols-1}"
      local visible_tail=""; (( ${#plain} >= ${#plain_prefix} )) && visible_tail="${plain:${#plain_prefix}}"
      local visible_head="${plain:0:${#plain_prefix}}"
      local render="${COLOR_RESET}${visible_head}${COLOR_GRAY}${visible_tail}${COLOR_RESET}"
      if [[ "$render" != "$prev_render" ]]; then printf '\r\033[K%s' "$render"; prev_render="$render"; fi
      i=$(( (i + 1) % ${#frames[@]} )); sleep 0.25
    done
  ) & local spinner_pid=$!

  if ! "$@" >"$log_file" 2>&1; then
    kill "$spinner_pid" &>/dev/null || true; wait "$spinner_pid" &>/dev/null || true
    printf '\r\033[K%s' "${COLOR_RESET}"; printf "❌ %s failed.\n" "$description"
    echo "ERROR LOG :"; cat "$log_file"; echo "END OF ERROR LOG"; rm -f "$log_file"; exit 1
  fi
  kill "$spinner_pid" &>/dev/null || true; wait "$spinner_pid" &>/dev/null || true
  printf '\r\033[K%s' "${COLOR_RESET}"; printf '✅ %s\n' "$description"; rm -f "$log_file"
}

have() { command -v "$1" >/dev/null 2>&1; }

# --- Step 1: Load .env if present ---
if [[ -f ".env" ]]; then
  echo "Sourcing .env"; set -a; source .env; set +a
fi
[[ -z "${HF_TOKEN:-}" && -n "${HUGGINGFACE_HUB_TOKEN:-}" ]] && export HF_TOKEN="${HUGGINGFACE_HUB_TOKEN}"

VENV_DIR=".venv"
PYTHON_BIN=""

# --- Step 2: Pick Python (3.11/3.12 preferred) ---
echo "Searching for a compatible Python (3.11 or 3.12 preferred)"
if   have python3.11; then PYTHON_BIN="python3.11"
elif have python3.12; then PYTHON_BIN="python3.12"
elif have python3;    then PYTHON_BIN="python3"; echo "⚠️  Falling back to default 'python3'."
else echo "❌ No suitable Python found. Please install Python 3.11 or 3.12."; exit 1
fi
echo "✅ Using: $($PYTHON_BIN --version)"

# --- Step 3: Minimal system deps ---
UNAME_S="$(uname -s)"
if [[ "$UNAME_S" == "Linux" ]]; then
  if ! have g++ || ! have make; then
    run_and_log "Installing build tools (Linux)" sudo apt-get update && sudo apt-get install -y build-essential
  fi
elif [[ "$UNAME_S" == "Darwin" ]]; then
  if ! xcode-select -p >/dev/null 2>&1; then echo "Installing Xcode Command Line Tools..."; xcode-select --install || true; fi
fi

# --- Step 4: Create & activate venv ---
[[ ! -d "$VENV_DIR" ]] && run_and_log "Creating virtual environment" "$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
run_and_log "Upgrading pip/setuptools/wheel" pip install -U pip setuptools wheel

# --- Variant selection (fixes UNAME var & removes duplicate block) ---
if [[ -z "${VARIANT:-}" ]]; then
  if [[ "$UNAME_S" == "Darwin" ]]; then
    VARIANT="cpu"
  elif command -v nvidia-smi &>/dev/null; then
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
    echo "Detected GPU: ${gpu_name:-unknown}"
    # Check for both T4 and L4 GPUs, as both can be used for CUDA
    if [[ "$gpu_name" == *"T4"* ]] || [[ "$gpu_name" == *"L4"* ]]; then
        if [[ -n "$AUTO_YES" ]] || ask_yes_no "Use the GPU install ([cuda_gpu])? [y/N]"; then VARIANT="cuda_gpu"; else VARIANT="cpu"; fi
    else
        VARIANT="cpu"
    fi
  else
    VARIANT="cpu"
  fi
fi
echo "Chosen variant: ${VARIANT}"

# --- ABI guards ---
run_and_log "Ensure NumPy/SciPy ABI compatibility" pip install "numpy<2" "scipy<1.13" --upgrade

# --- Step 5: Torch pins (visible, overridable) ---
TORCH_CHANNEL="${TORCH_CHANNEL:-cu124}"

# macOS/CPU
TORCH_VER_CPU="${TORCH_VER_CPU:-2.2.2}"
VISION_VER_CPU="${VISION_VER_CPU:-0.17.2}"

# CUDA 12.4 (exists on cu124 index)
TORCH_VER_GPU="${TORCH_VER_GPU:-2.4.1}"
VISION_VER_GPU="${VISION_VER_GPU:-0.19.1}"

echo "Torch plan:"
echo "  UNAME_S=$UNAME_S  VARIANT=$VARIANT  CHANNEL=$TORCH_CHANNEL"
echo "  CPU/Mac  -> torch==$TORCH_VER_CPU  torchvision==$VISION_VER_CPU"
echo "  CUDA     -> torch==$TORCH_VER_GPU  torchvision==$VISION_VER_GPU (from $TORCH_CHANNEL)"

install_torch_mac(){ run_and_log "Torch (macOS)" \
  pip install "torch==${TORCH_VER_CPU}" "torchvision==${VISION_VER_CPU}"; }

install_torch_cpu(){ run_and_log "Torch (CPU)" \
  pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  "torch==${TORCH_VER_CPU}" "torchvision==${VISION_VER_CPU}"; }

install_torch_gpu(){ run_and_log "Torch (CUDA ${TORCH_CHANNEL})" \
  pip install --extra-index-url "https://download.pytorch.org/whl/${TORCH_CHANNEL}" \
  "torch==${TORCH_VER_GPU}" "torchvision==${VISION_VER_GPU}"; }

if [[ "$UNAME_S" == "Darwin" ]]; then
  install_torch_mac
else
  case "$VARIANT" in
    cpu)    install_torch_cpu ;;
    cuda_gpu) install_torch_gpu ;;
    *) echo "❌ Unknown variant '$VARIANT'"; exit 1 ;;
  esac
fi

# --- Step 6: Install DreamGaussian Dependencies and compile extensions ---
if [[ "$VARIANT" == "cuda_gpu" ]]; then
    echo "DreamGaussian requires specific CUDA extensions. Checking for 'nvcc'..."
    if ! have nvcc; then
        echo "❌ 'nvcc' (the CUDA compiler) is not found. Please ensure the CUDA toolkit is installed and on your PATH to compile the necessary extensions."
        exit 1
    fi
    
    # Install DreamGaussian dependencies. The --no-input flag prevents pip from
    # attempting to prompt for user credentials, which can happen with certain
    # git configurations even for public repos.
    run_and_log "Installing DreamGaussian dependencies (GPU)" pip install --no-input "einops>=0.7" "fire" "lpips" "plyfile" "scikit-image" "trimesh" "xatlas"

    run_and_log "Compiling DreamGaussian CUDA extensions" bash ./setup_dreamgaussian.sh

    if [[ -f "validate_dreamgaussian.py" ]]; then
        run_and_log "Running DreamGaussian validation" python validate_dreamgaussian.py
    else
        echo "ℹ️  Skipping validation (validate_dreamgaussian.py not found)."
    fi

else
    echo "No compatible GPU detected. DreamGaussian relies on a CUDA-enabled NVIDIA GPU for its C++ and CUDA extensions."
    echo "Since this system is not compatible, the DreamGaussian installation is not possible and will be skipped to prevent errors."
fi

# --- Step 7: Install project ---
run_and_log "Installing zero-mv" pip install -e .

# --- Step 8: Optional Hugging Face auth ---
if [[ -n "${HF_TOKEN:-}" ]]; then
  run_and_log "Hugging Face login" huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
fi

# --- Step 9: Offer a 'venv' alias (bash/zsh only, idempotent) ---
SHELL_NAME="$(basename "${SHELL:-}")"
if [[ "$SHELL_NAME" == "zsh" ]]; then
  SHELL_RC="${ZDOTDIR:-$HOME}/.zshrc"
elif [[ "$SHELL_NAME" == "bash" ]]; then
  SHELL_RC="$HOME/.bashrc"
  [[ -f "$HOME/.bash_profile" && ! -f "$SHELL_RC" ]] && SHELL_RC="$HOME/.bash_profile"
else
  SHELL_RC="$HOME/.profile"
fi

BLOCK_START="# >>> zero-mv venv alias >>>"
BLOCK_END="# <<< zero-mv venv alias <<<"

if ! grep -qF "$BLOCK_START" "$SHELL_RC" 2>/dev/null; then
  {
    echo "$BLOCK_START"
    echo "alias venv='source \"$(pwd)/.venv/bin/activate\"'"
    echo "$BLOCK_END"
  } >> "$SHELL_RC"
  echo "✅ Added 'venv' alias to $SHELL_RC (run: source $SHELL_RC)"
else
  echo "ℹ️  'venv' alias already present in $SHELL_RC"
fi

echo
echo "✅ Setup complete."
echo "Activate your env with:  source ${VENV_DIR}/bin/activate  (or: venv)"