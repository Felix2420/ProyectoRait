"""
Hace visibles las DLLs de CUDA 12 / cuDNN 9 que vienen con PyTorch para que
onnxruntime-gpu pueda cargar `CUDAExecutionProvider` sin instalar el toolkit
de NVIDIA por separado.

Uso (ANTES de cualquier `import onnxruntime`):
    from src.preprocess._cuda_dlls import enable_torch_cuda_dlls
    enable_torch_cuda_dlls()
    import onnxruntime as ort
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def enable_torch_cuda_dlls() -> Path:
    """Añade <venv>/Lib/site-packages/torch/lib al search path de DLLs.

    Necesario en Windows + onnxruntime-gpu cuando NO se instaló CUDA Toolkit
    a nivel de sistema. PyTorch ya trae cublasLt64_12.dll, cudnn64_9.dll, etc.

    Devuelve la ruta añadida (útil para logging).
    """
    if sys.platform != "win32":
        return Path()  # noop fuera de Windows

    import torch
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    if not torch_lib.is_dir():
        raise RuntimeError(f"No existe {torch_lib} — ¿torch instalado?")

    os.add_dll_directory(str(torch_lib))
    return torch_lib
