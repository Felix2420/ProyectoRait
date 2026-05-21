#!/usr/bin/env python
"""Test rápido del fix CUBLAS: verifica que embed() funciona sin errores FP16."""

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from src.pipeline.embed import GaitEmbedder

def test_embedding():
    """Crea un dummy sequence y lo embebe sin errores."""
    print("Cargando GaitEmbedder...")
    emb = GaitEmbedder(
        ckpt=REPO / "checkpoints" / "finetune" / "gaitbase_ft_multisession_best_iter1200.pt",
        class_num=13
    )

    print("Generando dummy sequence (60, 64, 44) uint8...")
    seq = np.random.randint(0, 256, size=(60, 64, 44), dtype=np.uint8)

    print("Embebiendo...")
    try:
        e = emb.embed(seq)
        print(f"[OK] Exito. Embedding shape={e.shape}, norm={np.linalg.norm(e):.4f}")
        assert e.shape == (4096,), f"shape incorrecto: {e.shape}"
        assert np.isfinite(e).all(), "NaN/Inf en embedding"
        print("[OK] Todos los valores finitos")
        print("[PASS] Fix CUBLAS funcionando correctamente")
        return 0
    except Exception as ex:
        print(f"[FAIL] Error: {type(ex).__name__}: {ex}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(test_embedding())
