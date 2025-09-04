#!/usr/bin/env bash
set -euo pipefail

# Where to keep the upstream repo(s)
REPO_DIR="external/dreamgaussian"
DGR_DIR="external/diff-gaussian-rasterization"

echo "→ Using python: $(command -v python)"

# --- Clone / update DreamGaussian (contains ./simple-knn) ---
if [ -d "$REPO_DIR/.git" ]; then
  echo "→ DreamGaussian already present; pulling latest"
  git -C "$REPO_DIR" pull --ff-only
else
  echo "→ Cloning DreamGaussian into: $REPO_DIR"
  mkdir -p "$(dirname "$REPO_DIR")"
  git clone --depth=1 https://github.com/dreamgaussian/dreamgaussian "$REPO_DIR"
fi

# --- Clone / update diff-gaussian-rasterization (separate repo!) ---
if [ -d "$DGR_DIR/.git" ]; then
  echo "→ diff-gaussian-rasterization already present; pulling latest"
  git -C "$DGR_DIR" pull --ff-only
else
  echo "→ Cloning diff-gaussian-rasterization into: $DGR_DIR"
  mkdir -p "$(dirname "$DGR_DIR")"
  git clone --depth=1 https://github.com/ashawkey/diff-gaussian-rasterization "$DGR_DIR"
fi

# Ensure build helpers are present
python - <<'PY'
import importlib, sys
try:
    importlib.import_module("torch")
except Exception as e:
    sys.exit(f"❌ PyTorch must be installed before building DG extensions: {e}")
print("→ PyTorch is available")
PY

python -m pip install --no-input ninja cmake

# Build & install CUDA extensions
echo "→ Installing simple-knn …"
python -m pip install --no-input -v "$REPO_DIR/simple-knn"

echo "→ Installing diff-gaussian-rasterization …"
python -m pip install --no-input -v "$DGR_DIR"

# Validate imports
python - <<'PY'
ok = True
try:
    import simple_knn  # noqa
    print("✓ simple_knn import OK")
except Exception as e:
    ok = False; print("✗ simple_knn import failed:", e)

try:
    import diff_gaussian_rasterization  # noqa
    print("✓ diff_gaussian_rasterization import OK")
except Exception as e:
    ok = False; print("✗ diff_gaussian_rasterization import failed:", e)

import sys
sys.exit(0 if ok else 1)
PY

echo "✅ DreamGaussian CUDA extensions installed."
