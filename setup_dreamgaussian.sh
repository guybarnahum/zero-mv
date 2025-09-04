#!/usr/bin/env bash
set -euo pipefail

# Make sure we're in the repo root if called from elsewhere
cd "$(dirname "$0")"

# prevent any interactive git prompts
export GIT_TERMINAL_PROMPT=0
export GIT_ASKPASS=

# --- simple-knn ---
git -c credential.helper= \
    -c url."https://github.com/".insteadof=git@github.com: \
    -c url."https://github.com/".insteadof=ssh://git@github.com/ \
    -c url."https://".insteadof=git:// \
    clone --depth=1 https://github.com/ashawkey/simple-knn /tmp/simple-knn

python -m pip install --no-input /tmp/simple-knn
rm -rf /tmp/simple-knn

# --- diff-gaussian-rasterization ---
git -c credential.helper= \
    -c url."https://github.com/".insteadof=git@github.com: \
    -c url."https://github.com/".insteadof=ssh://git@github.com/ \
    -c url."https://".insteadof=git:// \
    clone --depth=1 https://github.com/ashawkey/diff-gaussian-rasterization /tmp/dgr

python -m pip install --no-input /tmp/dgr
rm -rf /tmp/dgr

echo "DreamGaussian CUDA extensions installed."

