#!/usr/bin/env bash
#
# zero-mv setup:
# - Loads .env if present (for MVGEN_CMD_TEMPLATE, etc.)
# - Finds Python (prefers 3.11/3.12)
# - Installs minimal system deps (Linux/macOS)
# - Creates & activates venv
# - Installs PyTorch 2.2.2:
#     * Linux + NVIDIA GPU (e.g., T4) -> CUDA 12.1 wheels
#     * macOS -> MPS-enabled build
#     * Otherwise -> CPU wheels
# - Installs project in editable mode (minimal deps only)
# - Optional Hugging Face auth
#
set -e

# ------------- Auto-yes handling (AUTO_YES is '--yes' or empty) -------------
AUTO_YES=""
for arg in "$@"; do
  case "$arg" in
    --yes|-y) AUTO_YES="--yes" ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

ask_yes_no() {
  local prompt="$1"
  if [[ -n "$AUTO_YES" ]]; then
    echo "Auto-yes: $prompt -> yes"
    return 0
  fi
  read -p "$prompt " -n 1 -r
  echo
  [[ $REPLY =~ ^[Yy]$ ]]
}

# ------------- Safer colors + cleanup -------------
if tput setaf 0 >/dev/null 2>&1; then
  COLOR_GRAY="$(tput setaf 8)"
  COLOR_RESET="$(tput sgr0)"
else
  COLOR_GRAY=$'\033[90m'
  COLOR_RESET=$'\033[0m'
fi

cleanup_render() {
  printf '\r\033[K%s' "${COLOR_RESET}"
  tput cnorm 2>/dev/null || true
}
trap cleanup_render EXIT INT TERM

# ------------- Improved run_and_log (ANSI-safe, truncation-safe, last-line preview) -------------
run_and_log() {
  local log_file
  log_file=$(mktemp)
  local description="$1"
  shift

  # printf "⏳ %s\n" "$description"
  tput civis 2>/dev/null || true

  local prev_render=""
  local cols
  cols=$(tput cols 2>/dev/null || echo 120)

  (
    frames=( '⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏' )
    i=0
    while :; do
      local last_line=""
      if [[ -s "$log_file" ]]; then
        last_line=$(tail -n 1 "$log_file" | sed -E 's/\x1B\[[0-9;?]*[ -/]*[@-~]//g')
      fi

      local plain_prefix="${frames[i]} ${description} : "
      local plain="${plain_prefix}${last_line}"
      if (( ${#plain} > cols )); then
        plain="${plain:0:cols-1}"
      fi

      local visible_tail=""
      if (( ${#plain} >= ${#plain_prefix} )); then
        visible_tail="${plain:${#plain_prefix}}"
      fi
      local visible_head="${plain:0:${#plain_prefix}}"

      local render="${COLOR_RESET}${visible_head}${COLOR_GRAY}${visible_tail}${COLOR_RESET}"

      if [[ "$render" != "$prev_render" ]]; then
        printf '\r\033[K%s' "$render"
        prev_render="$render"
      fi

      i=$(( (i + 1) % ${#frames[@]} ))
      sleep 0.25
    done
  ) &
  local spinner_pid=$!

  if ! "$@" >"$log_file" 2>&1; then
    kill "$spinner_pid" &>/dev/null || true
    wait "$spinner_pid" &>/dev/null || true
    printf '\r\033[K%s' "${COLOR_RESET}"
    printf "❌ %s failed.\n" "$description"
    echo "ERROR LOG :"
    cat "$log_file"
    echo "END OF ERROR LOG"
    rm -f "$log_file"
    exit 1
  fi

  kill "$spinner_pid" &>/dev/null || true
  wait "$spinner_pid" &>/dev/null || true
  printf '\r\033[K%s' "${COLOR_RESET}"
  printf '✅ %s\n' "$description"
  rm -f "$log_file"
}

# -------- helpers --------
have() { command -v "$1" >/dev/null 2>&1; }

# --- Step 1: Load .env if present ---
if [[ -f ".env" ]]; then
  echo "Sourcing .env"
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

VENV_DIR=".venv"
PYTHON_BIN=""

# Bridge HF token envs (optional)
if [[ -z "${HF_TOKEN:-}" && -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then export HF_TOKEN="${HUGGINGFACE_HUB_TOKEN}"; fi

# --- Step 2: Pick Python (3.11/3.12 preferred) ---
echo "Searching for a compatible Python (3.11 or 3.12 preferred)"
if have python3.11; then
  PYTHON_BIN="python3.11"
elif have python3.12; then
  PYTHON_BIN="python3.12"
elif have python3; then
  PYTHON_BIN="python3"
  echo "⚠️  Falling back to default 'python3'."
else
  echo "❌ No suitable Python found. Please install Python 3.11 or 3.12."
  exit 1
fi
echo "✅ Using: $($PYTHON_BIN --version)"

# --- Step 3: Minimal system deps (Linux/macOS) ---
UNAME_S="$(uname -s)"
if [[ "$UNAME_S" == "Linux" ]]; then
  if ! have g++ || ! have make; then
    run_and_log "Installing build tools (Linux)" sudo apt-get update && sudo apt-get install -y build-essential
  fi
elif [[ "$UNAME_S" == "Darwin" ]]; then
  if ! xcode-select -p >/dev/null 2>&1; then
    echo "Installing Xcode Command Line Tools..."
    xcode-select --install || true
  fi
fi

# --- Step 4: Create & activate venv ---
if [[ ! -d "$VENV_DIR" ]]; then
  run_and_log "Creating virtual environment" "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
run_and_log "Upgrading pip/setuptools/wheel" pip install -U pip setuptools wheel

# -------- Variant selection --------
if [[ -z "$VARIANT" ]]; then
  if [[ "$uname_s" == "Darwin" ]]; then
    VARIANT="cpu"
  elif command -v nvidia-smi &>/dev/null; then
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
    echo "Detected GPU: ${gpu_name:-unknown}"
    if [[ -n "$AUTO_YES" ]] || ask_yes_no "Use the GPU install ([t4_gpu])? [y/N]"; then VARIANT="t4_gpu"; else VARIANT="cpu"; fi
  else VARIANT="cpu"; fi
fi
echo "Chosen variant: ${VARIANT}"
[[ "$WITH_VIZ" -eq 1 ]] && echo "Extra: [viz] will be installed"

# -------- ABI guards --------
run_and_log "Ensure NumPy/SciPy ABI compatibility" pip install "numpy<2" "scipy<1.13" --upgrade

# -------- Step 5: Torch pins --------
TORCH_CHANNEL="${TORCH_CHANNEL:-cu124}"
TORCH_VER_CPU="2.2.2"; VISION_VER_CPU="0.17.2"
TORCH_VER_GPU="2.4.2"; VISION_VER_GPU="0.19.1"

install_torch_mac(){   run_and_log "Torch (macOS)" pip install "torch==${TORCH_VER_CPU}" "torchvision==${VISION_VER_CPU}"; }
install_torch_cpu(){   run_and_log "Torch (CPU)"   pip install --extra-index-url https://download.pytorch.org/whl/cpu \
                                           "torch==${TORCH_VER_CPU}" "torchvision==${VISION_VER_CPU}"; }
install_torch_gpu(){   run_and_log "Torch (CUDA ${TORCH_CHANNEL})" pip install --extra-index-url "https://download.pytorch.org/whl/${TORCH_CHANNEL}" \
                                           "torch==${TORCH_VER_GPU}" "torchvision==${VISION_VER_GPU}"; }

if [[ "$uname_s" == "Darwin" ]]; then
  install_torch_mac
else
  case "$VARIANT" in
    cpu)    install_torch_cpu ;;
    t4_gpu) install_torch_gpu ;;
    *) echo "❌ Unknown variant '$VARIANT'"; exit 1 ;;
  esac
fi
  
# --- Step 6: Install project (minimal deps needed to orchestrate Zero123++) ---
run_and_log "Installing zero-mv" pip install -e .

# -------- Step 7: Optional Hugging Face auth --------
if [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  run_and_log "Hugging Face login" huggingface-cli login --token "$HUGGINGFACE_HUB_TOKEN" --add-to-git-credential
fi

echo
echo "✅ Setup complete."
echo "Activate your env with:  source ${VENV_DIR}/bin/activate"
