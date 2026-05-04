"""
Smoke test para SkeletonGait++ (Fase 4, Paso E).

Verifica que:
  1. El modelo instancia correctamente (dual-stream sil + heatmap).
  2. Un forward pass con datos reales de data/pkl_multimodal/ no lanza errores.
  3. El embedding resultante tiene la forma esperada: (1, 256, 16).
  4. VRAM pico es razonable para la RTX 2060 / 6 GB.

Uso:
    python scripts/smoke_test_skeletongaitpp.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
OPENGAIT = REPO / "third_party" / "OpenGait" / "opengait"
sys.path.insert(0, str(OPENGAIT))
sys.path.insert(0, str(REPO / "third_party" / "OpenGait"))

MULTIMODAL_ROOT = REPO / "data" / "pkl_multimodal"
VIEW = "090"
SEQ = "seq00"

# Config del modelo para ProyectoChino
# B=[1,4,4,1] y C=2 → mismo tamaño que Gait3D config oficial
MODEL_CFG = {
    "Backbone": {
        "in_channels": 3,   # 2 heatmap + 1 sil (documentación)
        "blocks": [1, 4, 4, 1],
        "C": 2,
    },
    "SeparateBNNecks": {
        "class_num": 13,    # sujetos train
    },
    "use_emb2": False,
}

# Parámetros de recorte (heatmap 64×64 → 64×44 para coincidir con silueta)
SIL_H, SIL_W = 64, 44
HM_H, HM_W = 64, 64
CUTTING = (SIL_H - SIL_W) // 2   # = 10


def load_sample() -> tuple[np.ndarray, np.ndarray]:
    """Carga la primera secuencia disponible de data/pkl_multimodal/."""
    hm_pkls = sorted(MULTIMODAL_ROOT.glob(f"*/*/{ VIEW}/0_heatmap.pkl"))
    if not hm_pkls:
        raise RuntimeError(
            f"No hay datos en {MULTIMODAL_ROOT}. "
            "Corre build_multimodal_dataset.py primero."
        )
    hm_path = hm_pkls[0]
    sil_path = hm_path.parent / "1_sil.pkl"

    with hm_path.open("rb") as f:
        heatmap = pickle.load(f)   # (T, 2, 64, 64)
    with sil_path.open("rb") as f:
        sil = pickle.load(f)       # (T, 64, 44)

    return heatmap, sil


def build_input_tensor(heatmap: np.ndarray, sil: np.ndarray,
                       device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Convierte heatmap + sil en tensor de entrada para SkeletonGait++.

    Recorta heatmap a 64×44 y concatena con silueta → (1, T, 3, 64, 44).
    """
    T = heatmap.shape[0]

    hm_crop = heatmap[:, :, :, CUTTING:-CUTTING]          # (T, 2, 64, 44)
    sil_exp = sil[:, np.newaxis, :, :]                    # (T, 1, 64, 44)
    combined = np.concatenate([hm_crop, sil_exp], axis=1) # (T, 3, 64, 44)

    x = torch.from_numpy(combined.astype(np.float32) / 255.0).to(device)
    x = x.unsqueeze(0)                                    # (1, T, 3, 64, 44)
    seqL = torch.tensor([[T]], device=device, dtype=torch.int)
    return x, seqL


def _load_skeletongait_pp_class():
    """Carga SkeletonGaitPP desde skeletongait++.py via importlib (el '+' impide import normal)."""
    import importlib.util
    import importlib
    importlib.import_module("modeling")
    importlib.import_module("modeling.models")
    mod_name = "modeling.models.skeletongaitpp"
    if mod_name not in sys.modules:
        model_file = OPENGAIT / "modeling" / "models" / "skeletongait++.py"
        spec = importlib.util.spec_from_file_location(mod_name, str(model_file))
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "modeling.models"
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[mod_name].SkeletonGaitPP


def build_model():
    SkeletonGaitPP = _load_skeletongait_pp_class()
    model = SkeletonGaitPP.__new__(SkeletonGaitPP)
    nn.Module.__init__(model)
    model.training = True
    model.build_network(MODEL_CFG)
    return model


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        print(f"[smoke] GPU={torch.cuda.get_device_name(0)}  "
              f"VRAM={torch.cuda.get_device_properties(0).total_memory//1024**2} MB")

    # 1. Cargar datos
    print("[smoke] cargando muestra de pkl_multimodal...")
    heatmap, sil = load_sample()
    print(f"[smoke] heatmap shape={heatmap.shape}  dtype={heatmap.dtype}")
    print(f"[smoke] sil    shape={sil.shape}  dtype={sil.dtype}")

    # 2. Construir modelo
    print("[smoke] construyendo modelo SkeletonGait++...")
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] params={n_params:,}")
    model = model.to(device).eval()

    # 3. Construir tensor de entrada
    x, seqL = build_input_tensor(heatmap, sil, device)
    print(f"[smoke] input tensor shape={tuple(x.shape)}")

    # 4. Forward pass
    print("[smoke] forward pass...")
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        retval = model(([x], None, None, None, seqL))

    # 5. Verificar salida
    embed = retval["inference_feat"]["embeddings"]     # (1, C, P)
    print(f"[smoke] embedding shape={tuple(embed.shape)}  (esperado: (1, 256, 16))")

    assert embed.shape[0] == 1, f"batch != 1: {embed.shape}"
    assert embed.shape[1] == 256, f"channels != 256: {embed.shape}"  # 128*C con C=2
    assert embed.shape[2] == 16, f"parts != 16: {embed.shape}"

    embed_flat = embed.squeeze(0).cpu().numpy().flatten()
    print(f"[smoke] embedding flat shape={embed_flat.shape}  norm={np.linalg.norm(embed_flat):.3f}")

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() // 1024**2
        print(f"[smoke] VRAM pico: {peak_mb} MB")
        if peak_mb > 5000:
            print("[WARN] VRAM >5 GB con T=1 seq — P×K grande puede exceder 6 GB")

    print("\n[smoke] PASS — SkeletonGait++ instancia y hace forward sin errores.")
    print("Siguiente: python scripts/finetune_skeletongaitpp.py --dry_run")


if __name__ == "__main__":
    main()
