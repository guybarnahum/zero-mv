#!/usr/bin/env bash
set -euo pipefail

# Non-interactive git (avoid "terminal prompts disabled")
export GIT_TERMINAL_PROMPT=0
export GIT_ASKPASS=

# Use a temp workspace
WORKDIR="$(mktemp -d)"
echo "→ Working in $WORKDIR"
cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT

cd "$WORKDIR"

echo "→ Cloning diff-gaussian-rasterization (with submodules)…"
# Prefer the canonical GraphDECO repos.
git -c credential.helper= \
    -c url."https://github.com/".insteadof=git@github.com: \
    -c url."https://github.com/".insteadof=ssh://git@github.com/ \
    -c url."https://".insteadof=git:// \
    clone --recursive --depth=1 https://github.com/graphdeco-inria/diff-gaussian-rasterization dgr

# If the submodule didn't materialize a simple-knn directory, fetch it explicitly.
if [[ ! -d "dgr/simple-knn" && ! -d "simple-knn" ]]; then
  echo "→ simple-knn submodule not found; cloning explicitly…"
  git -c credential.helper= \
      -c url."https://github.com/".insteadof=git@github.com: \
      -c url."https://github.com/".insteadof=ssh://git@github.com/ \
      -c url."https://".insteadof=git:// \
      clone --depth=1 https://github.com/graphdeco-inria/simple-knn simple-knn
  SIMPLE_KNN_PATH="simple-knn"
else
  # Prefer submodule inside dgr if present
  if [[ -d "dgr/simple-knn" ]]; then
    SIMPLE_KNN_PATH="dgr/simple-knn"
  else
    SIMPLE_KNN_PATH="simple-knn"
  fi
fi

echo "→ Installing simple-knn from: $SIMPLE_KNN_PATH"
python -m pip install --no-input "$SIMPLE_KNN_PATH"

echo "→ Installing diff-gaussian-rasterization…"
python -m pip install --no-input ./dgr

echo "✅ DreamGaussian CUDA extensions installed."
