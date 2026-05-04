"""
Smoke test Fase 0 — GaitBase (OpenGait) en GTX 1650 4GB.
Construye el modelo Baseline manualmente (sin BaseModel/DDP), carga pesos
preentrenados de Gait3D-Parsing y corre un forward dummy midiendo VRAM.

Objetivo: confirmar que (a) las clases de OpenGait importan sin error,
(b) los pesos cargan, (c) el forward cabe en 4 GB con AMP.
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
OPENGAIT  = REPO_ROOT / "third_party" / "OpenGait"
CKPT      = REPO_ROOT / "checkpoints" / "pretrained" / "GaitBase_Gait3D_120000.pt"

# OpenGait usa imports absolutos asumiendo opengait/ como root del proyecto.
sys.path.insert(0, str(OPENGAIT / "opengait"))
sys.path.insert(0, str(OPENGAIT))

from modeling.backbones.resnet import ResNet9
from modeling.modules import (
    SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper,
    SeparateFCs, SeparateBNNecks,
)


class GaitBaseSmoke(nn.Module):
    """Reproduce manualmente la arquitectura `Baseline` de OpenGait
    sin depender de BaseModel/DDP, usando los hiperparámetros del config
    `gaitbase_gait3d_parsing_btz32x2_fixed.yaml`."""

    def __init__(self):
        super().__init__()
        backbone = ResNet9(
            block="BasicBlock",
            channels=[64, 128, 256, 512],
            in_channel=1,
            layers=[1, 1, 1, 1],
            strides=[1, 2, 2, 1],
            maxpool=False,
        )
        self.Backbone = SetBlockWrapper(backbone)
        self.FCs      = SeparateFCs(in_channels=512, out_channels=256, parts_num=16)
        self.BNNecks  = SeparateBNNecks(class_num=3000, in_channels=256, parts_num=16)
        self.TP       = PackSequenceWrapper(torch.max)
        self.HPP      = HorizontalPoolingPyramid(bin_num=[16])

    def forward(self, sils, seqL):
        # sils: (N, S, H, W) → añadir canal: (N, 1, S, H, W)
        if sils.dim() == 4:
            sils = sils.unsqueeze(1)
        outs = self.Backbone(sils)                     # (N, C, S, H, W)
        outs = self.TP(outs, seqL, options={"dim": 2})[0]  # (N, C, H, W)
        feat = self.HPP(outs)                          # (N, C, P)
        embed_1 = self.FCs(feat)                       # (N, C', P)
        embed_2, _logits = self.BNNecks(embed_1)
        return embed_1


def fmt_bytes(n):
    return f"{n / 1024**2:7.1f} MB"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device       : {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU          : {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory
        print(f"[INFO] Total VRAM   : {fmt_bytes(total_vram)}")
        torch.cuda.reset_peak_memory_stats()

    # --- 1. construir modelo --------------------------------------------------
    t0 = time.time()
    model = GaitBaseSmoke().to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Build model  : {time.time()-t0:.2f} s   |  params: {n_params:,}")

    # --- 2. cargar pesos ------------------------------------------------------
    t0 = time.time()
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[INFO] Load ckpt    : {time.time()-t0:.2f} s")
    print(f"[INFO] Missing keys : {len(missing)}   Unexpected: {len(unexpected)}")
    if missing[:3]:
        print(f"       sample missing  : {missing[:3]}")
    if unexpected[:3]:
        print(f"       sample unexpect : {unexpected[:3]}")

    # --- 3. forward dummy con AMP --------------------------------------------
    # OpenGait empaca secuencias: dim 0 siempre es 1, S es la suma de longitudes
    # individuales declaradas en seqL. Para "batch de N identidades" se concatena
    # en S y se indica [S, S, ...] en seqL.
    S, H, W = 30, 64, 44
    sils = torch.rand(1, S, H, W, device=device)
    seqL = [torch.tensor([S], device=device)]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            emb = model(sils, seqL)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    print(f"[INFO] Forward N=1  : {dt*1000:.1f} ms   |  embed shape: {tuple(emb.shape)}")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated()
        used = torch.cuda.memory_allocated()
        print(f"[INFO] VRAM used    : {fmt_bytes(used)}   |  peak: {fmt_bytes(peak)}")

    # --- 4. batch (N identidades concatenadas en S) ---------------------------
    for N_test in (2, 4, 8, 16, 32):
        try:
            sils_b = torch.rand(1, N_test * S, H, W, device=device)
            seqL_b = [torch.tensor([S] * N_test, device=device)]
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    out = model(sils_b, seqL_b)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            print(f"[INFO] N={N_test:2d}  forward  : {(time.time()-t0)*1000:6.1f} ms   peak VRAM: {fmt_bytes(peak)}   embed: {tuple(out.shape)}")
        except torch.cuda.OutOfMemoryError:
            print(f"[WARN] N={N_test:2d}  OOM (este es el techo de tu GPU para inferencia)")
            torch.cuda.empty_cache()
            break

    print("\n[OK] Smoke test completado.")


if __name__ == "__main__":
    main()
