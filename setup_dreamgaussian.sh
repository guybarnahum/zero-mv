# inside setup.sh (or in setup_dreamgaussian.sh)
set -e

REPO_DIR="external/dreamgaussian"

# Clone or update
if [[ -d "$REPO_DIR/.git" ]]; then
  echo "→ DreamGaussian already present; pulling latest"
  git -C "$REPO_DIR" pull --ff-only
else
  echo "→ Cloning DreamGaussian into: $REPO_DIR"
  git clone --depth=1 https://github.com/dreamgaussian/dreamgaussian "$REPO_DIR"
fi

# Make sure torch is present before building extensions
python - <<'PY'
import importlib, sys
try:
    importlib.import_module("torch")
except Exception as e:
    sys.exit("❌ PyTorch not installed before DG extensions: %r" % e)
print("→ Using python:", sys.executable)
PY

# Build helpers (ninja/cmake are already handled in your log, keep them installed)
python -m pip install --no-input ninja cmake

# Now install the two local CUDA extensions exactly like the official README:
#   pip install ./diff-gaussian-rasterization
#   pip install ./simple-knn
(
  cd "$REPO_DIR"
  echo "→ Installing simple-knn …"
  python -m pip install --no-input ./simple-knn

  echo "→ Installing diff-gaussian-rasterization …"
  python -m pip install --no-input ./diff-gaussian-rasterization
)
