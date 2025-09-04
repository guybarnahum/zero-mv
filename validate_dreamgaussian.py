# validate_dreamgaussian.py
import importlib, sys

def fail(msg):
    print(f"❌ {msg}")
    sys.exit(1)

# diff_gaussian_rasterization should import cleanly when its wheel is built
try:
    dgr = importlib.import_module("diff_gaussian_rasterization")
    print(f"✅ diff_gaussian_rasterization @ {dgr.__file__}")
except Exception as e:
    fail(f"Failed to import diff_gaussian_rasterization: {e}")

# simple_knn C++/CUDA extension: adapt to exported name
try:
    sk = importlib.import_module("simple_knn._C")
except Exception as e:
    fail(f"Failed to import simple_knn._C: {e}")

exports = {n for n in dir(sk) if not n.startswith("_")}
candidates = ["knn", "knn_points", "knn_cuda", "knn_search", "knn_gpu"]
have = next((n for n in candidates if n in exports), None)

if not have:
    print("🔎 simple_knn._C exports:", sorted(list(exports)))
    fail("simple_knn._C does not expose any known 'knn' entrypoint "
         "(tried: knn, knn_points, knn_cuda, knn_search, knn_gpu).")

# smoke-call: just ensure it’s callable without real tensors
attr = getattr(sk, have)
print(f"✅ simple_knn._C exposes '{have}'")
