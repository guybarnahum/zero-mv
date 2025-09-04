#!/usr/bin/env bash
set -euo pipefail

# Prefer the project venv’s Python
if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  PY="$VIRTUAL_ENV/bin/python"
else
  PY="$(command -v python3 || command -v python)"
fi

"$PY" - <<'PY'
import importlib, sys, inspect, pkgutil

def fail(msg: str, code: int = 1):
    print(f"❌ {msg}")
    sys.exit(code)

# 1) diff_gaussian_rasterization should import cleanly
try:
    dgr = importlib.import_module("diff_gaussian_rasterization")
    print(f"✅ diff_gaussian_rasterization @ {getattr(dgr, '__file__', 'built-in')}")
except Exception as e:
    fail(f"Failed to import diff_gaussian_rasterization: {e}")

# 2) simple_knn Python package and its C/CUDA extension
try:
    sk_pkg = importlib.import_module("simple_knn")
    print(f"✅ simple_knn package @ {getattr(sk_pkg, '__file__', 'built-in')}")
except Exception as e:
    fail(f"Failed to import simple_knn: {e}")

try:
    sk = importlib.import_module("simple_knn._C")
    print(f"✅ simple_knn._C @ {getattr(sk, '__file__', 'built-in')}")
except Exception as e:
    fail(f"Failed to import simple_knn._C: {e}")

# 3) Show EXACT exports from the compiled module
exports = sorted([n for n in dir(sk) if not n.startswith("_")])
print("🔎 simple_knn._C exports (names):")
for n in exports:
    obj = getattr(sk, n)
    kind = "callable" if callable(obj) else type(obj).__name__
    sig = ""
    if callable(obj):
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "(signature unavailable)"
    print(f"  - {n} {sig}  [{kind}]")

# 4) Validation: look for a KNN entrypoint by common names
expected = {"knn", "knn_points", "knn_cuda", "knn_search", "knn_gpu"}
found = [n for n in exports if n in expected]

if not found:
    fail("simple_knn._C does not expose a known KNN entrypoint "
         f"(expected one of: {sorted(expected)}).")

# 5) Final OK
print(f"✅ Found KNN entrypoint in simple_knn._C: {', '.join(found)}")
PY
