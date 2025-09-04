# validate_dreamgaussian.py
import importlib, sys, traceback

CANDIDATES = ["distCUDA2", "knn", "knn2", "knn_points", "knn_cuda", "knn_search", "knn_gpu"]

def fail(msg):
    print(f"❌ {msg}")
    sys.exit(1)

# 1) diff_gaussian_rasterization should import
try:
    dgr = importlib.import_module("diff_gaussian_rasterization")
    print(f"✅ diff_gaussian_rasterization @ {getattr(dgr,'__file__', '<?>')}")
except Exception as e:
    traceback.print_exc()
    fail(f"Failed to import diff_gaussian_rasterization: {e}")

# 2) simple_knn C++/CUDA extension should import
try:
    sk = importlib.import_module("simple_knn._C")
    print(f"✅ simple_knn._C @ {getattr(sk,'__file__', '<?>')}")
except Exception as e:
    traceback.print_exc()
    fail(f"Failed to import simple_knn._C: {e}")

exports = sorted([n for n in dir(sk) if not n.startswith("_")])
print("🔎 simple_knn._C exports:", exports)

have = next((n for n in CANDIDATES if hasattr(sk, n)), None)
if not have:
    fail("No known KNN/dist entrypoint found in simple_knn._C "
         f"(looked for: {', '.join(CANDIDATES)}).")

# 3) Optional smoke test (tiny tensor) if distCUDA2 is present
try:
    if have == "distCUDA2":
        import torch
        # tiny 3D point set; will run on CUDA if available, else CPU is fine for presence test
        pts = torch.randn(8, 3).cuda() if torch.cuda.is_available() else torch.randn(8, 3)
        out = getattr(sk, have)(pts)
        # Just basic shape check (should be NxN or similar)
        if hasattr(out, "shape") and len(out.shape) >= 2:
            print(f"✅ simple_knn._C.{have} smoke-call ok, output shape={tuple(out.shape)}")
        else:
            print(f"ℹ️ simple_knn._C.{have} returned type {type(out)} (no shape check)")
    else:
        # other symbols: just confirm callable presence
        fn = getattr(sk, have)
        print(f"✅ simple_knn._C exposes '{have}' (callable={callable(fn)})")
except Exception as e:
    traceback.print_exc()
    fail(f"Entrypoint '{have}' exists but smoke test failed: {e}")

print("🎉 DreamGaussian CUDA extensions look healthy.")
