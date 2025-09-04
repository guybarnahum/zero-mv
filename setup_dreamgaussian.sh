#!/usr/bin/env bash
set -euo pipefail

# Where to clone DreamGaussian (can override with DG_DIR env var)
DG_DIR="${DG_DIR:-external/dreamgaussian}"

# Ensure we're using the current venv's python/pip
PY="${PYTHON:-$(command -v python)}"

echo "→ Using python: $($PY -c 'import sys; print(sys.executable)')"

# 1) Clone the official DreamGaussian repo if not present
if [[ ! -d "$DG_DIR/.git" ]]; then
  echo "→ Cloning DreamGaussian into: $DG_DIR"
  mkdir -p "$(dirname "$DG_DIR")"
  git clone --recursive https://github.com/dreamgaussian/dreamgaussian "$DG_DIR"
else
  echo "→ DreamGaussian already present at: $DG_DIR"
fi

# 2) Install the two local subpackages inside the venv
#    These build the CUDA/C++ extensions so that
#    `import diff_gaussian_rasterization` works.
echo "→ Installing simple-knn …"
"$PY" -m pip install --no-input -e "$DG_DIR/simple-knn"

echo "→ Installing diff-gaussian-rasterization …"
"$PY" -m pip install --no-input -e "$DG_DIR/diff-gaussian-rasterization"

# 3) Quick import sanity check (same interpreter)
echo "→ Verifying imports …"
"$PY" - <<'PY'
import sys
print("sys.executable =", sys.executable)
import torch
print("CUDA available:", torch.cuda.is_available())
try:
    import diff_gaussian_rasterization as dgr
    print("OK: diff_gaussian_rasterization loaded:", dgr.__file__)
except Exception as e:
    print("FAIL: could not import diff_gaussian_rasterization:", e)
    raise SystemExit(1)
try:
    import simple_knn
    print("OK: simple_knn loaded:", getattr(simple_knn, "__file__", "<built-in>"))
except Exception as e:
    print("FAIL: could not import simple_knn:", e)
    raise SystemExit(1)
PY

echo "✅ DreamGaussian CUDA extensions installed."
