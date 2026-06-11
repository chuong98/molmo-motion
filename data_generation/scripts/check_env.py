#!/usr/bin/env python3
"""Quick environment sanity check for the data-generation pipeline."""
import importlib
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "third_party" / "sam3"))

ok = True


def check(label, fn):
    global ok
    try:
        fn()
        print(f"  [ OK ] {label}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  [FAIL] {label}: {type(e).__name__}: {e}")


print("Python deps:")
for mod in ["torch", "numpy", "cv2", "transformers", "sklearn", "imageio_ffmpeg", "yaml"]:
    check(mod, lambda m=mod: importlib.import_module(m))

print("CUDA:")
def _cuda():
    import torch
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
check("torch.cuda available", _cuda)

print("Vendored models:")
check("sam3 package import", lambda: importlib.import_module("sam3.model_builder"))
check("grounding helpers (molmo2_pointing)", lambda: importlib.import_module("molmo2_pointing"))
check("vipe package import", lambda: importlib.import_module("vipe"))
check("vipe CLI on PATH", lambda: (_ for _ in ()).throw(RuntimeError("vipe not found"))
      if shutil.which("vipe") is None else None)

print()
print("ENVIRONMENT OK" if ok else "ENVIRONMENT INCOMPLETE — see [FAIL] lines above")
sys.exit(0 if ok else 1)
