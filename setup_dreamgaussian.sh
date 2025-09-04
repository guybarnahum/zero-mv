#!/usr/bin/env bash
set -euo pipefail

# Run from repo root
cd "$(dirname "$0")"

# Non-interactive git
export GIT_TERMINAL_PROMPT=0
export GIT_ASKPASS=

# Helpful for L4/T4 (compute 8.9 / 7.5). Adjust if needed.
: "${TORCH_CUDA_ARCH_LIST:=7.5;8.9}"
export TORCH_CUDA_ARCH_LIST

have() { command -v "$1" >/dev/null 2>&1; }

clone_or_zip_install () {
  local REPO_URL="$1"   # e.g. https://github.com/ashawkey/simple-knn
  local NAME="$2"       # e.g. simple-knn
  local TMP_DIR
  TMP_DIR="$(mktemp -d)"

  echo "→ Installing ${NAME} …"

  # Try a git clone with all auth disabled and headers cleared
  if git \
    -c credential.helper= \
    -c http.https://github.com/.extraheader= \
    -c url."https://github.com/".insteadof=git@github.com: \
    -c url."https://github.com/".insteadof=ssh://git@github.com/ \
    -c url."https://".insteadof=git:// \
    clone --depth=1 "${REPO_URL}" "${TMP_DIR}/${NAME}"; then
      python -m pip install --no-input "${TMP_DIR}/${NAME}"
      rm -rf "${TMP_DIR}"
      return 0
  fi

  echo "   git clone blocked; falling back to zip download…"

  # Fallback: download the default branch zip via codeload
  # (works with proxies that block git, still anonymous)
  local ZIP_URL
  ZIP_URL="${REPO_URL/https:\/\/github.com\//https:\/\/codeload.github.com\/}.zip/refs/heads/master"
  # If master doesn’t exist, try main
  if ! curl -fsSL -o "${TMP_DIR}/${NAME}.zip" "${ZIP_URL}"; then
    ZIP_URL="${REPO_URL/https:\/\/github.com\//https:\/\/codeload.github.com\/}.zip/refs/heads/main"
    curl -fsSL -o "${TMP_DIR}/${NAME}.zip" "${ZIP_URL}"
  fi

  unzip -q "${TMP_DIR}/${NAME}.zip" -d "${TMP_DIR}"
  # folder will be NAME-<branch>
  local EXTRACT_DIR
  EXTRACT_DIR="$(find "${TMP_DIR}" -maxdepth 1 -type d -name "${NAME}-*" | head -n1)"
  python -m pip install --no-input "${EXTRACT_DIR}"
  rm -rf "${TMP_DIR}"
}

# simple-knn
clone_or_zip_install "https://github.com/ashawkey/simple-knn" "simple-knn"

# diff-gaussian-rasterization
clone_or_zip_install "https://github.com/ashawkey/diff-gaussian-rasterization" "diff-gaussian-rasterization"

echo "✅ DreamGaussian CUDA extensions installed."
