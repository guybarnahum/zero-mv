set -e

# --- quiet/sane build env ---
# turns off pip’s own chatter
export PIP_DISABLE_PIP_VERSION_CHECK=1
# silence most Python warnings printed via the warnings module
export PYTHONWARNINGS="ignore"
# hush some setuptools deprecation noise
export SETUPTOOLS_SILENCE_DEPRECATION_WARNING=1
export SETUPTOOLS_ENABLE_FEATURES="legacy-editable"

REPO_DIR="external/dreamgaussian"
DGR_DIR="external/diff-gaussian-rasterization"

# DreamGaussian: clone or pull
if [[ -d "$REPO_DIR/.git" ]]; then
  echo "→ DreamGaussian already present; pulling latest"
  git -C "$REPO_DIR" pull --ff-only
else
  echo "→ Cloning DreamGaussian into: $REPO_DIR"
  git clone --depth=1 https://github.com/dreamgaussian/dreamgaussian "$REPO_DIR"
fi

# diff-gaussian-rasterization: clone/update with submodules (for third_party/glm)
if [[ -d "$DGR_DIR/.git" ]]; then
  echo "→ diff-gaussian-rasterization already present; pulling + updating submodules"
  git -C "$DGR_DIR" pull --ff-only
  git -C "$DGR_DIR" submodule update --init --recursive
else
  echo "→ Cloning diff-gaussian-rasterization into: $DGR_DIR (with submodules)"
  git clone --recursive --depth=1 https://github.com/ashawkey/diff-gaussian-rasterization "$DGR_DIR"
fi

# Ensure torch is importable before building
python - <<'PY'
import importlib, sys
try:
    importlib.import_module("torch")
    print("→ PyTorch is available")
except Exception as e:
    sys.exit(f"❌ PyTorch not installed before DG extensions: {e!r}")
PY

# Build helpers
python -m pip install --no-input ninja cmake

# Build local extensions exactly like README
echo "→ Installing simple-knn …"
python -m pip install --no-input "$REPO_DIR/simple-knn"

echo "→ Installing diff-gaussian-rasterization …"
python -m pip install --no-input "$DGR_DIR"
