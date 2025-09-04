# validate_dreamgaussian.py
import sys, importlib, torch

def _ok(msg): print("✅", msg)
def _fail(msg, e):
    print(f"❌ {msg}. Error: {e}")
    sys.exit(1)

# 1) diff_gaussian_rasterization import
try:
    dgr = importlib.import_module("diff_gaussian_rasterization")
    _ok(f"diff_gaussian_rasterization @ {dgr.__file__}")
except Exception as e:
    _fail("Failed to import diff_gaussian_rasterization", e)

# 2) simple_knn import (robust path)
try:
    sk = importlib.import_module("simple_knn._C")
    assert hasattr(sk, "knn"), "simple_knn._C has no 'knn' symbol"
    _ok("simple_knn._C.knn is available")
except Exception as e:
    _fail("Failed to import 'knn' from simple_knn._C", e)

# 3) micro-run: call knn on tiny tensors
try:
    import torch
    pts = torch.randn(1024, 3, device="cuda")
    q   = torch.randn(128, 3, device="cuda")
    idx = sk.knn(q, pts, 8)  # returns (Nq, K) indices
    _ok(f"simple_knn knn() call ok: idx.shape={tuple(idx.shape)}")
except Exception as e:
    _fail("simple_knn knn() micro-run failed", e)

print("All DreamGaussian extension checks passed.")
