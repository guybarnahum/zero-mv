#!/usr/bin/env bash
set -euo pipefail

# Use the same Python as your venv
PY="${PYTHON:-python}"

echo "→ Using python: $($PY -c 'import sys;print(sys.executable)')"

# Where to put the repo (relative to project root)
DG_DIR="external/dreamgaussian"
mkdir -p external

if [[ ! -d "$DG_DIR/.git" ]]; then
  echo "→ Cloning DreamGaussian into: $DG_DIR"
  git clone --recursive https://github.com/dreamgaussian/dreamgaussian "$DG_DIR"
else
  echo "→ DreamGaussian already present; pulling latest"
  (cd "$DG_DIR" && git pull --rebase && git submodule update --init --recursive)
fi

# Core Python deps (many are already in your env; harmless if reinstall)
$PY -m pip install --no-input -r "$DG_DIR/requirements.txt"

# Ensure toolchain bits
$PY -m pip install --no-input ninja cmake

# For Ada Lovelace (L4), set arch if not set; adjust if you have a different GPU
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

# Avoid PEP 517 build isolation so torch is visible during build
export PIP_NO_BUILD_ISOLATION=1

echo "→ Installing simple-knn …"
$PY -m pip install --no-input "$DG_DIR/simple-knn"

echo "→ Installing diff-gaussian-rasterization …"
$PY -m pip install --no-input "$DG_DIR/diff-gaussian-rasterization"

# Quick import check
cat > /tmp/_dg_import_check.py <<'PY'
import sys
err = 0
for m in ("torch","simple_knn","diff_gaussian_rasterization"):
    try:
        __import__(m)
        print(f"[ok] import {m}")
    except Exception as e:
        print(f"[fail] import {m}: {e}", file=sys.stderr)
        err = 1
sys.exit(err)
PY
$PY /tmp/_dg_import_check.py
