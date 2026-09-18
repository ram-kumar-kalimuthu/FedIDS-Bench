import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Fix Windows PyTorch DLL loading (WinError 1114)
if sys.platform == "win32":
    try:
        import importlib.util
        spec = importlib.util.find_spec("torch")
        if spec and spec.origin:
            torch_lib = os.path.join(os.path.dirname(spec.origin), "lib")
            if os.path.exists(torch_lib):
                os.add_dll_directory(torch_lib)
    except Exception:
        pass

__version__ = "0.1.0"
