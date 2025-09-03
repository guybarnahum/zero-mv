# src/zero_mv/utils/devices.py
from __future__ import annotations

def pick_torch_device(torch) -> str:
    """
    Decide 'cuda' | 'mps' | 'cpu' using the imported torch module.
    Keeps all device logic in one place.
    """
    try:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

