#!/usr/bin/env python3
import importlib, sys, inspect

def fail(msg):
    print(f"❌ {msg}")
    sys.exit(1)

def check(mod_name):
    try:
        m = importlib.import_module(mod_name)
        print(f"✅ {mod_name} @ {getattr(m, '__file__', '<built-in>')}")
        return m
    except Exception as e:
        fail(f"Failed importing {mod_name}: {e}")

dgr = check("diff_gaussian_rasterization")
skc = check("simple_knn._C")

exports = sorted([n for n in dir(skc) if not n.startswith("_")])
print("🔎 simple_knn._C exports:", exports)

candidates = ["knn", "knn_points", "knn_cuda", "knn_search", "knn_gpu"]
found = [c for c in candidates if c in exports]
if not found:
    fail("No known knn entrypoint in simple_knn._C")
print(f"✅ Found entrypoint: {found[0]}")
