#!/usr/bin/env bash
set -euo pipefail

# Installs DreamGaussian’s CUDA extensions from the official repo:
#   https://github.com/dreamgaussian/dreamgaussian
# We only build the native kernels (diff-gaussian-rasterization, simple-knn).

# --- prerequisites ---
if ! command -v nvcc >/dev/null 2>&1; then
  echo "❌ nvcc (CUDA toolkit) not found on PATH. Install CUDA to build DreamGaussian extensions."
  exit 1
fi

# Keep venv python/pip
PYTHON_BIN="${PYTHON_BIN:-python}"
PIP_BIN="${PIP_BIN:-python -m pip}"

# Temp workspace
WG=$(mktemp -d 2>/dev/null || mktemp -d -t dreamgaussian)
cleanup() { rm -rf "$WG"; }
trap cleanup EXIT

echo "→ Working dir: $WG"

# Disable any interactive git credentials; force HTTPS
export GIT_TERMINAL_PROMPT=0
export GIT_ASKPASS=
GIT_ARGS=(-c credential.helper= -c url."https://github.com/".insteadof=git@github.com:
          -c url."https://github.com/".insteadof=ssh://git@github.com/
          -c url."https://".insteadof=git://)

echo "→ Cloning dreamgaussian/dreamgaussian (recursive)…"
git "${GIT_ARGS[@]}" clone --recursive --depth=1 https://github.com/dreamgaussian/dreamgaussian "$WG/dreamgaussian"

# In case submodules weren’t pulled by the shallow clone, ensure they’re present
(
  cd "$WG/dreamgaussian"
  git submodule update --init --recursive || true
)

# Paths to extensions (as in the official repo layout)
DGR_DIR="$WG/dreamgaussian/diff-gaussian-rasterization"
SKNN_DIR="$WG/dreamgaussian/simple-knn"

# Sanity checks and install each if present
if [[ -d "$DGR_DIR" ]]; then
  echo "→ Building diff-gaussian-rasterization …"
  $PIP_BIN install --no-input "$DGR_DIR"
else
  echo "⚠️  diff-gaussian-rasterization directory not found under dreamgaussian."
  echo "    Please check the repo layout or open an issue if this persists."
fi

if [[ -d "$SKNN_DIR" ]]; then
  echo "→ Building simple-knn …"
  $PIP_BIN install --no-input "$SKNN_DIR"
else
  echo "⚠️  simple-knn directory not found under dreamgaussian."
  echo "    Please check the repo layout or open an issue if this persists."
fi

echo "✅ DreamGaussian CUDA extensions installed."
