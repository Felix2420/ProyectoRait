"""Wrapper de GaitBase iter1200 para producción.

Misma arquitectura y forward que `scripts/openset_eval.py` y
`scripts/build_gallery.py`. Devuelve embeddings L2-normalizados aplanados
(dim 4096 = 256 × 16) listos para max-sim contra la gallery.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
OPENGAIT = REPO / "third_party" / "OpenGait"
sys.path.insert(0, str(OPENGAIT / "opengait"))
sys.path.insert(0, str(OPENGAIT))

from modeling.backbones.resnet import ResNet9  # noqa: E402
from modeling.modules import (  # noqa: E402
    SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper,
    SeparateFCs, SeparateBNNecks,
)

log = logging.getLogger("pipeline.embed")


class GaitBaseInfer(nn.Module):
    def __init__(self, class_num: int = 13) -> None:
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
        self.FCs = SeparateFCs(in_channels=512, out_channels=256, parts_num=16)
        self.BNNecks = SeparateBNNecks(class_num=class_num, in_channels=256, parts_num=16)
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

    def forward(self, sils: torch.Tensor, seqL: List[torch.Tensor]) -> torch.Tensor:
        if sils.dim() == 4:
            sils = sils.unsqueeze(1)
        outs = self.Backbone(sils)
        outs = self.TP(outs, seqL, options={"dim": 2})[0]
        feat = self.HPP(outs)
        return self.FCs(feat)


class GaitEmbedder:
    def __init__(self, ckpt: Path, class_num: int = 13, device: str | None = None,
                 compile: bool = False) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        log.info("GaitBase ckpt=%s device=%s compile=%s", ckpt.name, self.device, compile)
        self.model = GaitBaseInfer(class_num=class_num).to(self.device).eval()
        state = torch.load(ckpt, map_location=self.device, weights_only=False)
        state = state.get("model", state)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            log.warning("missing=%d unexpected=%d", len(missing), len(unexpected))
        if compile and hasattr(torch, "compile"):
            try:
                log.info("Compilando modelo con torch.compile...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception as e:
                log.warning("torch.compile fallback (error: %s). Usando modelo sin compilar.", str(e)[:80])

    @torch.no_grad()
    def embed(self, seq: np.ndarray) -> np.ndarray:
        """Entrada (T, 64, 44) uint8. Salida (4096,) float32 L2-normalizada."""
        if seq.ndim != 3 or seq.shape[1:] != (64, 44):
            raise ValueError(f"shape inesperada {seq.shape}")
        sils = (torch.from_numpy(seq.astype(np.float32) / 255.0)
                .unsqueeze(0).to(self.device))
        seqL = [torch.tensor([sils.shape[1]], device=self.device)]
        with torch.amp.autocast("cuda", enabled=False):
            emb = self.model(sils, seqL)
        e = emb.float().squeeze(0).cpu().numpy().reshape(-1)
        n = float(np.linalg.norm(e))
        return (e / n).astype(np.float32) if n > 1e-12 else e.astype(np.float32)
